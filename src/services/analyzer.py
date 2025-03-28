"""LMS analyzer service for the LMS Content Analysis Dashboard."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

from ..models.data_models import (
    AnalysisResults,
    QualityMetrics,
    ActivityMetrics,
    SimilarityMetrics,
    Recommendation,
    Alert,
    ValidationResult,
    Severity
)

from ..utils.text_processing import (
    clean_text,
    calculate_similarity,
    extract_keywords,
    analyze_text_quality
)

from ..config.analysis_params import (
    QUALITY_THRESHOLDS,
    SIMILARITY_THRESHOLDS,
    ACTIVITY_THRESHOLDS,
    ANALYSIS_CONFIG
)

from ..config.constants import REQUIRED_FIELDS

# Configure logging
logger = logging.getLogger(__name__)

class LMSAnalyzer:
    """Service for analyzing LMS content data."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the analyzer with processed data."""
        self.df = df
        self.preprocess_data()
        # Calculate metrics
        self.quality_metrics = self._calculate_quality_metrics()
        self.activity_metrics = self._calculate_activity_metrics()
        self.similarity_metrics = self._calculate_similarity_metrics()
    
    def preprocess_data(self) -> None:
        """Preprocess the data for analysis."""
        # Fill missing values
        if 'learner_count' in self.df.columns:
            self.df['learner_count'] = self.df['learner_count'].fillna(0)
        
        if 'total_2024_activity' in self.df.columns:
            self.df['total_2024_activity'] = self.df['total_2024_activity'].fillna(0)
        
        # Clean text data
        text_columns = ['course_title', 'course_description', 'category_name', 'region_entity']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).apply(clean_text)
        
        # Add derived columns
        if 'course_available_from' in self.df.columns:
            self.df['course_age_days'] = (datetime.now() - pd.to_datetime(self.df['course_available_from'])).dt.days
        
        # Add active flag
        self.df['is_active'] = True
        if 'course_discontinued_from' in self.df.columns:
            mask = (
                self.df['course_discontinued_from'].notna() & 
                (pd.to_datetime(self.df['course_discontinued_from']) <= datetime.now())
            )
            self.df.loc[mask, 'is_active'] = False
    
    def get_analysis_results(self) -> AnalysisResults:
        """Generate comprehensive analysis results."""
        total_courses = len(self.df)
        active_courses = self.df['is_active'].sum() if 'is_active' in self.df.columns else total_courses
        
        # Calculate basic metrics
        basic_metrics = {
            'total_courses': total_courses,
            'active_courses': active_courses,
            'total_learners': self.df['learner_count'].sum() if 'learner_count' in self.df.columns else None,
            'regions_covered': self.df['region_entity'].nunique() if 'region_entity' in self.df.columns else None,
        }
        
        # Generate detailed metrics
        quality_metrics = self._calculate_quality_metrics()
        activity_metrics = self._calculate_activity_metrics()
        similarity_metrics = self._calculate_similarity_metrics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Generate alerts
        alerts = self._generate_alerts()
        
        # Create a summary dictionary
        summary = {
            "total_courses": total_courses,
            "active_courses": active_courses,
            "quality_score": quality_metrics.overall_score if quality_metrics else 0.0,
            "recommendations_count": len(recommendations)
        }
        
        # Create analysis results
        return AnalysisResults(
            timestamp=datetime.now(),
            total_courses=total_courses,
            active_courses=active_courses,
            total_learners=basic_metrics.get('total_learners'),
            regions_covered=basic_metrics.get('regions_covered'),
            quality_metrics=quality_metrics,
            activity_metrics=activity_metrics,
            similarity_metrics=similarity_metrics,
            recommendations=recommendations,
            alerts=alerts,
            summary=summary
        )
    
    def _calculate_quality_metrics(self) -> QualityMetrics:
        """Calculate quality metrics for the content."""
        # Calculate completeness score
        completeness_columns = ['course_title', 'course_description', 'category_name', 'region_entity']
        completeness_df = pd.DataFrame()
        
        for col in completeness_columns:
            if col in self.df.columns:
                completeness_df[col] = (~self.df[col].isna()).astype(int)
        
        completeness_score = completeness_df.mean(axis=1).mean() if not completeness_df.empty else 0.5
        
        # Calculate metadata score
        metadata_columns = ['course_available_from', 'course_type', 'course_keywords']
        metadata_df = pd.DataFrame()
        
        for col in metadata_columns:
            if col in self.df.columns:
                metadata_df[col] = (~self.df[col].isna()).astype(int)
        
        metadata_score = metadata_df.mean(axis=1).mean() if not metadata_df.empty else 0.5
        
        # Calculate content quality score
        content_score = 0.7  # Default score
        
        if 'course_description' in self.df.columns:
            # Analyze description quality
            desc_lengths = self.df['course_description'].astype(str).apply(len)
            avg_length = desc_lengths.mean()
            content_score = min(avg_length / 500, 1.0)  # Scale by expected length
        
        # Calculate overall score
        overall_score = (
            0.4 * completeness_score +
            0.3 * metadata_score +
            0.3 * content_score
        )
        
        # Identify missing fields
        missing_fields = []
        for col in completeness_columns + metadata_columns:
            if col in self.df.columns and self.df[col].isna().any():
                missing_fields.append(col)
        
        # Identify improvement areas
        improvement_areas = []
        
        if completeness_score < QUALITY_THRESHOLDS['completeness']:
            improvement_areas.append("Improve data completeness")
        
        if metadata_score < QUALITY_THRESHOLDS['metadata']:
            improvement_areas.append("Enhance metadata quality")
        
        if content_score < QUALITY_THRESHOLDS['content']:
            improvement_areas.append("Improve content quality")
        
        return QualityMetrics(
            completeness_score=completeness_score,
            metadata_score=metadata_score,
            content_score=content_score,
            overall_score=overall_score,
            missing_fields=missing_fields,
            improvement_areas=improvement_areas
        )
    
    def _calculate_activity_metrics(self) -> ActivityMetrics:
        """Calculate activity metrics for the content."""
        active_courses = self.df['is_active'].sum() if 'is_active' in self.df.columns else len(self.df)
        
        # Calculate recent completions
        recent_completions = 0
        if 'total_2024_activity' in self.df.columns:
            recent_completions = self.df['total_2024_activity'].sum()
        
        # Calculate completion rate
        average_completion_rate = 0.0
        if 'completion_rate' in self.df.columns:
            average_completion_rate = self.df['completion_rate'].mean()
        else:
            # Estimate completion rate from available data
            if 'total_2024_activity' in self.df.columns and 'learner_count' in self.df.columns:
                mask = self.df['learner_count'] > 0
                if mask.any():
                    average_completion_rate = (
                        self.df.loc[mask, 'total_2024_activity'] / self.df.loc[mask, 'learner_count']
                    ).mean()
        
        # Get last activity date
        last_activity_date = None
        if 'last_activity_date' in self.df.columns:
            if not self.df['last_activity_date'].isna().all():
                last_activity_date = self.df['last_activity_date'].max()
        
        # Generate activity trend
        activity_trend = {}
        if 'total_2024_activity' in self.df.columns and 'course_available_from' in self.df.columns:
            # Group by month and calculate activity
            self.df['month'] = pd.to_datetime(self.df['course_available_from']).dt.strftime('%Y-%m')
            monthly_activity = self.df.groupby('month')['total_2024_activity'].sum()
            
            # Convert to dictionary for the model
            activity_trend = monthly_activity.to_dict()
        
        return ActivityMetrics(
            active_courses=active_courses,
            recent_completions=recent_completions,
            average_completion_rate=average_completion_rate,
            last_activity_date=last_activity_date,
            activity_trend=activity_trend
        )
    
    def _calculate_similarity_metrics(self) -> SimilarityMetrics:
        """Calculate similarity metrics between courses."""
        # Initialize variables
        total_pairs = 0
        high_similarity_pairs = 0
        cross_department_pairs = 0
        average_similarity = 0.0
        duplicate_candidates = []
        similarity_distribution = {"high": 0, "medium": 0, "low": 0}
        
        # Check if we have sufficient data
        if len(self.df) < 2 or 'course_description' not in self.df.columns:
            return SimilarityMetrics(
                total_pairs=0,
                high_similarity_pairs=0,
                cross_department_pairs=0,
                average_similarity=0.0,
                duplicate_candidates=[],
                similarity_distribution=similarity_distribution
            )
        
        try:
            # Generate TF-IDF vectors from course descriptions
            tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
            
            # Get clean descriptions
            descriptions = self.df['course_description'].astype(str).apply(clean_text)
            
            # Calculate TF-IDF matrix
            tfidf_matrix = tfidf_vectorizer.fit_transform(descriptions)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix)
            
            # Process similarity matrix
            total_pairs = 0
            sum_similarity = 0.0
            
            # Track region entities if available
            has_region_data = 'region_entity' in self.df.columns
            
            for i in range(len(cosine_sim)):
                for j in range(i + 1, len(cosine_sim)):  # Only upper triangle
                    total_pairs += 1
                    sum_similarity += cosine_sim[i, j]
                    
                    # Check for high similarity
                    if cosine_sim[i, j] > SIMILARITY_THRESHOLDS['high']:
                        high_similarity_pairs += 1
                        
                        # Add to duplicate candidates
                        candidate = {
                            'course1': self.df.iloc[i]['course_no'],
                            'course2': self.df.iloc[j]['course_no'],
                            'similarity': cosine_sim[i, j],
                            'cross_department': False
                        }
                        
                        # Check for cross-department similarity
                        if has_region_data:
                            region1 = self.df.iloc[i]['region_entity']
                            region2 = self.df.iloc[j]['region_entity']
                            if region1 != region2:
                                cross_department_pairs += 1
                                candidate['cross_department'] = True
                        
                        duplicate_candidates.append(candidate)
                    
                    # Update similarity distribution
                    if cosine_sim[i, j] > SIMILARITY_THRESHOLDS['high']:
                        similarity_distribution["high"] += 1
                    elif cosine_sim[i, j] > SIMILARITY_THRESHOLDS['medium']:
                        similarity_distribution["medium"] += 1
                    elif cosine_sim[i, j] > SIMILARITY_THRESHOLDS['low']:
                        similarity_distribution["low"] += 1
            
            # Calculate average similarity
            average_similarity = sum_similarity / total_pairs if total_pairs > 0 else 0
            
            # Limit duplicate candidates
            duplicate_candidates.sort(key=lambda x: x['similarity'], reverse=True)
            duplicate_candidates = duplicate_candidates[:10]  # Limit to top 10
            
        except Exception as e:
            print(f"Error calculating similarity metrics: {str(e)}")
            # Return default values on error
            return SimilarityMetrics(
                total_pairs=0,
                high_similarity_pairs=0,
                cross_department_pairs=0,
                average_similarity=0.0,
                duplicate_candidates=[],
                similarity_distribution=similarity_distribution
            )
        
        return SimilarityMetrics(
            total_pairs=total_pairs,
            high_similarity_pairs=high_similarity_pairs,
            cross_department_pairs=cross_department_pairs,
            average_similarity=average_similarity,
            duplicate_candidates=duplicate_candidates,
            similarity_distribution=similarity_distribution
        )
    
    def _generate_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Quality-based recommendations
        if self.quality_metrics.overall_score < QUALITY_THRESHOLDS['overall']:
            recommendations.append(
                Recommendation(
                    title="Improve Overall Content Quality",
                    description="Several courses have quality scores below the required threshold",
                    impact="High",
                    effort="Medium",
                    implementation="Review and enhance course content, metadata, and descriptions",
                    priority=1,
                    category="Quality",
                    action_items=["Review course descriptions", "Add missing metadata"],
                    related_courses=self._get_low_quality_courses()
                )
            )
        
        # Sort by priority and limit to max recommendations
        recommendations.sort(key=lambda x: x.priority)
        return recommendations[:ANALYSIS_CONFIG['max_recommendations']]
    
    def _generate_alerts(self) -> List[Alert]:
        """Generate alerts based on analysis results."""
        alerts = []
        
        # Quality alerts
        if self.quality_metrics.completeness_score < QUALITY_THRESHOLDS['completeness']:
            alerts.append(
                Alert(
                    message="Low data completeness detected",
                    severity=Severity.WARNING,
                    details={
                        "score": self.quality_metrics.completeness_score,
                        "threshold": QUALITY_THRESHOLDS['completeness'],
                        "missing_fields": self.quality_metrics.missing_fields
                    }
                )
            )
        
        # Sort by severity and limit to max alerts
        alerts.sort(key=lambda x: x.severity.value, reverse=True)
        return alerts[:ANALYSIS_CONFIG['max_alerts']]
    
    def _get_low_quality_courses(self) -> List[str]:
        """Get list of courses with low quality scores."""
        low_quality_courses = []
        # Use quality metrics to identify low-quality courses
        if 'course_description' in self.df.columns:
            # Check for courses with short descriptions
            desc_lengths = self.df['course_description'].astype(str).apply(len)
            low_quality_idx = desc_lengths < 100  # Example threshold
            if low_quality_idx.any():
                low_quality_courses = self.df.loc[low_quality_idx, 'course_no'].tolist()
        return low_quality_courses
    
    def _get_low_activity_courses(self) -> List[str]:
        """Get list of courses with low activity."""
        low_activity_courses = []
        if 'total_2024_activity' in self.df.columns:
            # Find courses with low activity
            low_activity_idx = self.df['total_2024_activity'] < ACTIVITY_THRESHOLDS.get('min_activity', 5)
            if low_activity_idx.any():
                low_activity_courses = self.df.loc[low_activity_idx, 'course_no'].tolist()
        return low_activity_courses
    
    def _get_duplicate_courses(self) -> List[str]:
        """Get list of courses that are potential duplicates."""
        duplicate_courses = []
        if hasattr(self, 'similarity_metrics') and self.similarity_metrics is not None:
            if self.similarity_metrics.duplicate_candidates:
                # Extract unique course numbers from all candidates
                for candidate in self.similarity_metrics.duplicate_candidates:
                    if 'course1' in candidate and candidate['course1'] not in duplicate_courses:
                        duplicate_courses.append(candidate['course1'])
                    if 'course2' in candidate and candidate['course2'] not in duplicate_courses:
                        duplicate_courses.append(candidate['course2'])
        return duplicate_courses
    
    def _compile_validation_results(self, validation_results: List[ValidationResult]) -> Dict[str, Any]:
        """Compile validation results into a summary."""
        result = {
            "validation_count": len(validation_results),
            "passed": sum(1 for v in validation_results if v.is_valid),
            "failed": sum(1 for v in validation_results if not v.is_valid),
            "warnings": sum(1 for v in validation_results if v.severity == Severity.WARNING),
            "errors": sum(1 for v in validation_results if v.severity == Severity.ERROR),
            "fields": {}
        }
        
        # Track which required fields are missing
        missing_required = {field: 0 for field in REQUIRED_FIELDS}
        
        # Group results by field
        for validation in validation_results:
            field = validation.field or "general"
            
            if field not in result["fields"]:
                result["fields"][field] = {
                    "warnings": 0,
                    "errors": 0,
                    "messages": []
                }
            
            if validation.severity == Severity.WARNING:
                result["fields"][field]["warnings"] += 1
            elif validation.severity == Severity.ERROR:
                result["fields"][field]["errors"] += 1
            
            if validation.message and validation.message not in result["fields"][field]["messages"]:
                result["fields"][field]["messages"].append(validation.message)
            
            # Track missing required fields
            if not validation.is_valid and field in missing_required:
                missing_required[field] += 1
        
        # Add summary of missing required fields
        result["missing_required"] = {field: count for field, count in missing_required.items() if count > 0}
        
        return result 