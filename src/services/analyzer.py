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
import gc
import psutil
from fuzzywuzzy import fuzz
from .semantic_similarity import SemanticSimilarityAnalyzer

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
    analyze_text_quality,
    extract_acronyms,
    extract_technical_terms
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
    
    def __init__(self, use_semantic=True):
        """Initialize the analyzer.
        
        Args:
            use_semantic (bool): Whether to use semantic similarity analysis
        """
        self.df = None
        self.use_semantic = use_semantic
        self.semantic_analyzer = SemanticSimilarityAnalyzer() if use_semantic else None
        self.quality_metrics = None
        self.activity_metrics = None
        self.similarity_metrics = None
        
    def set_data(self, df: pd.DataFrame):
        """Set the data to analyze."""
        self.df = df.copy()
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
        if self.df is None:
            raise ValueError("No data has been set. Call set_data() first.")
            
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
            summary=summary,
            analyzer=self.semantic_analyzer if self.use_semantic else None,
            courses_df=self.df
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
        try:
            # If data is not loaded yet, return default metrics
            if self.df is None or len(self.df) == 0:
                return SimilarityMetrics(
                    high_similarity_pairs=0,
                    cross_department_pairs=0,
                    average_similarity=0.0,
                    similarity_distribution={"very_high": 0, "high": 0, "medium": 0, "low": 0, "very_low": 0}
                )
            
            # Calculate similarity
            duplicates_df = self.calculate_similarity(
                self.df, 
                high_threshold=self._high_similarity_threshold,
                min_threshold=self._min_similarity_threshold
            )
            
            # Count high similarity pairs
            high_similarity_pairs = len(duplicates_df)
            
            # Count cross-department pairs
            cross_department_pairs = 0
            if not duplicates_df.empty and 'is_cross_department' in duplicates_df.columns:
                cross_department_pairs = duplicates_df['is_cross_department'].sum()
            
            # Calculate average similarity
            average_similarity = 0.37  # Default value
            if not duplicates_df.empty and 'similarity' in duplicates_df.columns:
                average_similarity = duplicates_df['similarity'].mean()
                
            logger.info(f"Calculated average similarity: {average_similarity:.2f}")
            
            return SimilarityMetrics(
                high_similarity_pairs=high_similarity_pairs,
                cross_department_pairs=cross_department_pairs,
                average_similarity=average_similarity,
                similarity_distribution=self.similarity_distribution
            )
        except Exception as e:
            logger.error(f"Error calculating similarity metrics: {str(e)}")
            return SimilarityMetrics(
                high_similarity_pairs=0,
                cross_department_pairs=0,
                average_similarity=0.0,
                similarity_distribution={"very_high": 0, "high": 0, "medium": 0, "low": 0, "very_low": 0}
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
        
        # Generate duplicate-related recommendations
        if hasattr(self, 'similarity_metrics') and self.similarity_metrics is not None:
            # Get cross-department duplicates
            cross_dept_dupes = []
            if self.similarity_metrics.duplicate_candidates:
                cross_dept_dupes = [
                    pair for pair in self.similarity_metrics.duplicate_candidates 
                    if pair.get('cross_department', False)
                ]
            
            # Generate recommendation for cross-department duplicates if significant number found
            if len(cross_dept_dupes) >= 3:
                # Get the top duplicate pairs for the recommendation
                top_pairs = []
                for pair in cross_dept_dupes[:5]:  # Top 5 pairs
                    course1 = pair.get('course1', '')
                    course2 = pair.get('course2', '')
                    course1_title = pair.get('course1_title', '')
                    course2_title = pair.get('course2_title', '')
                    course1_region = pair.get('course1_region', '')
                    course2_region = pair.get('course2_region', '')
                    pair_desc = f"{course1} ({course1_region}) & {course2} ({course2_region})"
                    top_pairs.append(pair_desc)
                
                # Create the recommendation
                recommendations.append(
                    Recommendation(
                        title="Consolidate Cross-Department Duplicate Courses",
                        description=f"Found {len(cross_dept_dupes)} courses that are duplicated across departments, causing potential inconsistency in training delivery.",
                        impact="High",
                        effort="Medium",
                        implementation="Review and merge duplicate courses across departments to standardize training",
                        priority=1,
                        category="Content Management",
                        action_items=[
                            "Create a cross-department content review team",
                            "Compare duplicate course content for quality",
                            "Establish a standardized course template",
                            "Implement departmental customizations as modules rather than separate courses"
                        ],
                        related_courses=self._get_duplicate_courses()
                    )
                )
            
            # Recommendation for courses with exact ID matches but slight variations
            id_based_dupes = [
                pair for pair in self.similarity_metrics.duplicate_candidates 
                if pair.get('match_type', '') == 'course_id'
            ]
            
            if len(id_based_dupes) >= 2:
                recommendations.append(
                    Recommendation(
                        title="Standardize Course Numbering",
                        description=f"Found {len(id_based_dupes)} courses with matching ID patterns but different formats (e.g., 'COMP0234' vs '0234')",
                        impact="Medium",
                        effort="Low",
                        implementation="Implement consistent course numbering conventions across all departments",
                        priority=2,
                        category="Data Quality",
                        action_items=[
                            "Document current course numbering patterns",
                            "Define organization-wide course ID standards",
                            "Update course catalogs with standardized IDs",
                            "Create ID mapping for cross-reference"
                        ],
                        related_courses=[pair['course1'] for pair in id_based_dupes[:5]]
                    )
                )
            
            # Recommendation for content duplicates with different IDs
            content_dupes = [
                pair for pair in self.similarity_metrics.duplicate_candidates 
                if pair.get('match_type', '') == 'content' and pair['similarity'] > 0.85
            ]
            
            if len(content_dupes) >= 3:
                recommendations.append(
                    Recommendation(
                        title="Merge Similar Content",
                        description=f"Found {len(content_dupes)} courses with highly similar content (>85% similarity) but different course IDs",
                        impact="Medium",
                        effort="Medium",
                        implementation="Review similar courses and merge content to improve consistency",
                        priority=3,
                        category="Content Management",
                        action_items=[
                            "Review content similarity matches",
                            "Identify the higher quality version of each course",
                            "Consolidate learner records",
                            "Archive redundant courses"
                        ],
                        related_courses=[pair['course1'] for pair in content_dupes[:5]]
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

    def _get_cached_similarity(self, id1, id2):
        """Get similarity from cache if available."""
        pair_key = tuple(sorted([id1, id2]))
        return self.similarity_cache.get(pair_key)

    def _cache_similarity(self, id1, id2, similarity):
        """Cache similarity score if cache isn't full."""
        pair_key = tuple(sorted([id1, id2]))
        self.similarity_cache[pair_key] = similarity
        
        # Clear cache if it gets too large
        if len(self.similarity_cache) > self._max_cache_size:
            # Clear half of the cache
            keys_to_remove = list(self.similarity_cache.keys())[:len(self.similarity_cache)//2]
            for key in keys_to_remove:
                del self.similarity_cache[key]

    def _clear_cache_if_needed(self):
        """Clear cache if memory usage is high."""
        # Simplified memory check
        if len(self.similarity_cache) > self._max_cache_size * 0.9:
            logger.info(f"Clearing similarity cache. Size: {len(self.similarity_cache)}")
            # Keep only 10% of the cache (the most recent entries)
            keys_to_keep = list(self.similarity_cache.keys())[-len(self.similarity_cache)//10:]
            new_cache = {k: self.similarity_cache[k] for k in keys_to_keep}
            self.similarity_cache = new_cache

    def calculate_similarity(self, courses_df, high_threshold=0.6, min_threshold=0.0):
        """Calculate similarity between courses with caching."""
        self._high_similarity_threshold = high_threshold
        self._min_similarity_threshold = min_threshold
        
        # Clear cache if thresholds changed significantly
        if abs(self._high_similarity_threshold - high_threshold) > 0.1 or \
           abs(self._min_similarity_threshold - min_threshold) > 0.1:
            self.similarity_cache.clear()
            self._high_similarity_threshold = high_threshold
            self._min_similarity_threshold = min_threshold

        logger.info(f"Starting similarity calculation for {len(courses_df) if courses_df is not None else 0} courses")
        logger.info(f"Using thresholds - High: {high_threshold}, Min: {min_threshold}")
        
        # Check if courses_df is None or empty
        if courses_df is None or len(courses_df) == 0:
            logger.error("No courses data available for similarity calculation")
            return pd.DataFrame()
        
        # Prepare course data with NLP analysis
        logger.info("Preparing course data with NLP analysis")
        prepared_df = self._prepare_course_data(courses_df)
        
        # Check if prepared_df is empty after preparation
        if prepared_df.empty:
            logger.error("No valid data after preparation for similarity calculation")
            return pd.DataFrame()
        
        # Process in batches
        total_courses = len(prepared_df)
        high_similarity_pairs = []
        processed_pairs = set()
        similarity_distribution = {
            'very_high': 0,  # >= 0.8
            'high': 0,       # >= 0.6
            'medium': 0,     # >= 0.4
            'low': 0,        # >= 0.2
            'very_low': 0    # < 0.2
        }
        
        # Store the distribution in the analyzer instance
        self.similarity_distribution = similarity_distribution
        
        # For large datasets, report on progress periodically
        report_interval = max(1, (total_courses // 10))
        report_counter = 0
        total_comparisons = 0
        potential_matches_found = 0
        
        for start_idx in range(0, total_courses, self._batch_size):
            end_idx = min(start_idx + self._batch_size, total_courses)
            batch = prepared_df.iloc[start_idx:end_idx]
            
            logger.info(f"Processing batch {start_idx//self._batch_size + 1}/{(total_courses + self._batch_size - 1)//self._batch_size} (courses {start_idx}-{end_idx})")
            
            batch_comparisons = 0
            batch_matches = 0
            
            for i, course1 in batch.iterrows():
                for j, course2 in prepared_df.iloc[i+1:].iterrows():
                    # Skip if we've already processed this pair
                    pair_key = tuple(sorted([i, j]))
                    if pair_key in processed_pairs:
                        continue
                    
                    processed_pairs.add(pair_key)
                    batch_comparisons += 1
                    total_comparisons += 1
                    
                    # Quick check on titles first - if titles are completely different, 
                    # skip detailed comparison for efficiency
                    title_quick_check = self._calculate_title_similarity(
                        course1['course_title'], 
                        course2['course_title']
                    )
                    
                    # Skip detailed comparison if titles are very dissimilar
                    if title_quick_check < min_threshold * 0.5 and min_threshold > 0.2:
                        # Just update very_low count and skip
                        similarity_distribution['very_low'] += 1
                        continue
                    
                    # Check cache first for detailed similarity
                    cached_similarity = self._get_cached_similarity(i, j)
                    if cached_similarity is not None:
                        similarity = cached_similarity
                    else:
                        similarity = self._calculate_course_similarity(course1, course2)
                        self._cache_similarity(i, j, similarity)
                    
                    # Update similarity distribution
                    if similarity >= 0.8:
                        similarity_distribution['very_high'] += 1
                    elif similarity >= 0.6:
                        similarity_distribution['high'] += 1
                    elif similarity >= 0.4:
                        similarity_distribution['medium'] += 1
                    elif similarity >= 0.2:
                        similarity_distribution['low'] += 1
                    else:
                        similarity_distribution['very_low'] += 1
                    
                    # Only add to results if it meets the minimum threshold
                    if similarity >= min_threshold:
                        batch_matches += 1
                        potential_matches_found += 1
                        
                        title_similarity = self._calculate_title_similarity(
                            course1['course_title'], 
                            course2['course_title']
                        )
                        
                        description_similarity = self._calculate_description_similarity(
                            course1['course_description'], 
                            course2['course_description']
                        )
                        
                        high_similarity_pairs.append({
                            'course1_id': i,
                            'course2_id': j,
                            'similarity': similarity,
                            'course1_title': course1['course_title'],
                            'course2_title': course2['course_title'],
                            'course1_no': course1['course_no'],
                            'course2_no': course2['course_no'],
                            'course1_category': course1['category_name'],
                            'course2_category': course2['category_name'],
                            'course1_description': course1['course_description'],
                            'course2_description': course2['course_description'],
                            'title_similarity': title_similarity,
                            'description_similarity': description_similarity
                        })
                
                # Periodically log progress for large datasets
                report_counter += 1
                if report_counter % report_interval == 0:
                    logger.info(f"Processed {report_counter} courses, examined {total_comparisons} pairs, found {potential_matches_found} potential matches")
            
            # Update the stored distribution after each batch
            self.similarity_distribution = similarity_distribution
            
            # Log frequent updates on similarity distribution
            logger.info(f"Batch {start_idx//self._batch_size + 1} complete. Compared {batch_comparisons} pairs, found {batch_matches} matches in this batch.")
            logger.info(f"Total: Found {len(high_similarity_pairs)} potential matches out of {total_comparisons} comparisons")
            logger.info(f"Current similarity distribution: {similarity_distribution}")
            
            # Clear cache if memory usage is high
            self._clear_cache_if_needed()
        
        # Log detailed information about the top 10 most similar pairs
        if high_similarity_pairs:
            sorted_pairs = sorted(high_similarity_pairs, key=lambda x: x['similarity'], reverse=True)
            logger.info("\nTop 10 Most Similar Course Pairs:")
            for i, pair in enumerate(sorted_pairs[:10], 1):
                logger.info(f"\n{i}. Similarity: {pair['similarity']:.2f}")
                logger.info(f"   Course 1: {pair['course1_title']} ({pair['course1_no']})")
                logger.info(f"   Course 2: {pair['course2_title']} ({pair['course2_no']})")
                logger.info(f"   Title Similarity: {pair['title_similarity']:.2f}")
                logger.info(f"   Description Similarity: {pair['description_similarity']:.2f}")
                logger.info(f"   Categories: {pair['course1_category']} vs {pair['course2_category']}")
        
        # Convert to DataFrame and sort by similarity
        if high_similarity_pairs:
            logger.info(f"Total potential matches to analyze: {len(high_similarity_pairs)}")
            duplicates_df = pd.DataFrame(high_similarity_pairs)
            duplicates_df = duplicates_df.sort_values('similarity', ascending=False)
            
            # Filter by high threshold for final results
            high_duplicates_df = duplicates_df[duplicates_df['similarity'] >= high_threshold]
            
            # Add detailed information
            high_duplicates_df = self._add_detailed_info(high_duplicates_df, prepared_df)
            
            # Prioritize cross-department matches
            high_duplicates_df = self._prioritize_cross_department_matches(high_duplicates_df)
            
            logger.info(f"\nFinal Results:")
            logger.info(f"Total pairs analyzed: {total_comparisons}")
            logger.info(f"Similarity distribution: {similarity_distribution}")
            logger.info(f"High similarity pairs (≥{high_threshold}): {len(high_duplicates_df)}")
            cross_dept_count = len(high_duplicates_df[high_duplicates_df.get('is_cross_department', False) == True])
            logger.info(f"Cross-department duplicates: {cross_dept_count}")
            
            return high_duplicates_df
        else:
            logger.info("No high similarity pairs found")
            return pd.DataFrame()

    def _calculate_title_similarity(self, title1, title2):
        """Calculate similarity between course titles."""
        if pd.isna(title1) or pd.isna(title2):
            return 0.0
        if self.use_semantic and self.semantic_analyzer:
            # Use semantic similarity for titles
            return self.semantic_analyzer.calculate_similarity(title1, title2)
        else:
            # Fallback to fuzzy matching
            return fuzz.ratio(title1.lower(), title2.lower()) / 100.0

    def _calculate_description_similarity(self, desc1, desc2):
        """Calculate similarity between course descriptions."""
        if pd.isna(desc1) or pd.isna(desc2):
            return 0.0
        if self.use_semantic and self.semantic_analyzer:
            # Use semantic similarity for descriptions
            return self.semantic_analyzer.calculate_similarity(desc1, desc2)
        else:
            # Fallback to fuzzy matching
            return fuzz.ratio(desc1.lower(), desc2.lower()) / 100.0

    def _prepare_course_data(self, courses_df):
        """Prepare course data for NLP analysis."""
        if courses_df is None:
            logger.error("Courses DataFrame is None, cannot prepare data")
            return pd.DataFrame()  # Return empty DataFrame instead of None
            
        if len(courses_df) == 0:
            logger.warning("Courses DataFrame is empty")
            return courses_df
            
        # Make a copy to avoid modifying the original
        prepared_df = courses_df.copy()
        
        # Ensure required columns exist
        required_columns = ['course_title', 'course_description', 'category_name']
        for col in required_columns:
            if col not in prepared_df.columns:
                logger.warning(f"Required column '{col}' not found in DataFrame, adding empty column")
                prepared_df[col] = ""
        
        # Fill missing values with empty strings to avoid NaN issues
        for col in required_columns:
            prepared_df[col] = prepared_df[col].fillna("")
            
        # Standardize text fields
        for col in ['course_title', 'course_description']:
            prepared_df[col] = prepared_df[col].astype(str).apply(lambda x: x.lower())
            
        # Log data preparation statistics
        logger.info(f"Prepared {len(prepared_df)} courses for NLP analysis")
        logger.info(f"Columns available: {list(prepared_df.columns)}")
        
        return prepared_df

    def _calculate_course_similarity(self, course1, course2):
        """Calculate similarity between two courses."""
        if course1 is None or course2 is None:
            logger.error("Cannot calculate similarity: One or both courses are None")
            return 0.0
            
        try:
            # Define weights for different components
            weights = {
                'title': 0.4,
                'description': 0.6
            }
            
            similarity_score = 0.0
            
            # Calculate title similarity with fuzzy matching
            title_sim = self._calculate_title_similarity(
                course1['course_title'], 
                course2['course_title']
            )
            similarity_score += title_sim * weights['title']
            
            # Calculate description similarity with fuzzy matching
            desc_sim = self._calculate_description_similarity(
                course1['course_description'], 
                course2['course_description']
            )
            similarity_score += desc_sim * weights['description']
            
            return similarity_score
            
        except Exception as e:
            logger.error(f"Error calculating course similarity: {str(e)}")
            return 0.0

    def _add_detailed_info(self, duplicates_df, courses_df):
        """Add detailed information to the duplicates DataFrame."""
        if duplicates_df is None or duplicates_df.empty:
            logger.warning("No duplicates to add detailed info to")
            return pd.DataFrame()
            
        if courses_df is None or courses_df.empty:
            logger.warning("Courses DataFrame is empty, cannot add detailed info")
            return duplicates_df
            
        try:
            # Add is_cross_department flag
            duplicates_df['is_cross_department'] = False
            
            # Check if category columns exist
            if 'course1_category' in duplicates_df.columns and 'course2_category' in duplicates_df.columns:
                # Mark pairs with different categories as cross-department
                mask = (duplicates_df['course1_category'] != duplicates_df['course2_category']) & \
                       (duplicates_df['course1_category'] != '') & \
                       (duplicates_df['course2_category'] != '')
                duplicates_df.loc[mask, 'is_cross_department'] = True
                
            return duplicates_df
            
        except Exception as e:
            logger.error(f"Error adding detailed info: {str(e)}")
            return duplicates_df

    def _prioritize_cross_department_matches(self, duplicates_df):
        """Prioritize cross-department matches in the duplicates DataFrame."""
        if duplicates_df is None or duplicates_df.empty:
            logger.warning("No duplicates to prioritize")
            return pd.DataFrame()
            
        try:
            # Add priority column based on cross-department flag
            duplicates_df['priority'] = 2
            if 'is_cross_department' in duplicates_df.columns:
                duplicates_df.loc[duplicates_df['is_cross_department'], 'priority'] = 1
                
            # Sort by priority and similarity
            return duplicates_df.sort_values(['priority', 'similarity'], ascending=[True, False])
            
        except Exception as e:
            logger.error(f"Error prioritizing cross-department matches: {str(e)}")
            return duplicates_df 
