"""Service for analyzing LMS data."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config.constants import (
    TRAINING_CATEGORIES,
    QUALITY_SCORE_WEIGHTS,
    REQUIRED_FIELDS
)
from models.data_models import (
    AnalysisResults,
    QualityMetrics,
    ValidationResult
)

class LMSAnalyzer:
    """Analyzes LMS data and provides insights."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize the analyzer with a DataFrame."""
        self.df = df
        self.results = AnalysisResults(df=df)
        self._calculate_basic_metrics()
    
    def _calculate_basic_metrics(self):
        """Calculate basic metrics about the dataset."""
        self.results.total_courses = len(self.df)
        
        if 'learner_count' in self.df.columns:
            self.results.total_learners = self.df['learner_count'].sum()
        
        if 'course_discontinued_from' in self.df.columns:
            self.results.active_courses = len(self.df[
                (self.df['course_discontinued_from'].isna()) |
                (self.df['course_discontinued_from'] > pd.Timestamp.now())
            ])
        else:
            self.results.active_courses = self.results.total_courses
        
        if 'region_entity' in self.df.columns:
            self.results.regions_covered = self.df['region_entity'].nunique()
            self._calculate_regional_distribution()
        
        if 'data_source' in self.df.columns:
            self.results.data_sources = self.df['data_source'].nunique()
        
        if 'cross_reference_count' in self.df.columns:
            self.results.cross_referenced_courses = len(
                self.df[self.df['cross_reference_count'] > 1]
            )
        
        self._calculate_category_distribution()
        self._calculate_temporal_patterns()
        self._analyze_content_gaps()
    
    def _calculate_category_distribution(self):
        """Calculate the distribution of courses across categories."""
        category_columns = [
            col for col in self.df.columns
            if col.startswith('is_') and col in TRAINING_CATEGORIES
        ]
        
        for category in category_columns:
            category_name = category.replace('is_', '').replace('_', ' ').title()
            self.results.category_distribution[category_name] = int(
                self.df[category].sum()
            )
    
    def _calculate_regional_distribution(self):
        """Calculate the distribution of courses across regions."""
        if 'region_entity' in self.df.columns:
            region_counts = self.df['region_entity'].value_counts()
            self.results.regional_distribution = region_counts.to_dict()
    
    def _calculate_temporal_patterns(self):
        """Analyze temporal patterns in course creation and updates."""
        if 'course_available_from' not in self.df.columns:
            return
        
        # Convert to datetime if needed
        dates = pd.to_datetime(self.df['course_available_from'])
        
        # Initialize temporal patterns dictionary
        self.results.temporal_patterns['creation_patterns'] = {
            'courses_per_year': {},
            'courses_per_month': {}
        }
        
        # Calculate yearly distribution
        yearly_counts = dates.dt.year.value_counts().sort_index()
        self.results.temporal_patterns['creation_patterns']['courses_per_year'] = \
            yearly_counts.to_dict()
        
        # Calculate monthly distribution
        monthly_counts = dates.dt.month.value_counts().sort_index()
        month_names = {
            i: datetime(2000, i, 1).strftime('%B')
            for i in range(1, 13)
        }
        self.results.temporal_patterns['creation_patterns']['courses_per_month'] = {
            month_names[k]: v
            for k, v in monthly_counts.to_dict().items()
        }
    
    def _analyze_content_gaps(self):
        """Analyze content gaps and opportunities."""
        # Initialize content gaps dictionary
        self.results.content_gaps = {
            'critical_gaps': [],
            'opportunity_areas': []
        }
        
        # Analyze regional gaps
        if 'region_entity' in self.df.columns and self.df['region_entity'].notna().any():
            try:
                # Get columns that start with 'is_'
                is_columns = [col for col in self.df.columns if col.startswith('is_')]
                if is_columns:
                    # Create a pivot table instead of crosstab
                    region_category_matrix = pd.pivot_table(
                        self.df,
                        values=is_columns,
                        index='region_entity',
                        aggfunc='sum'
                    )
                    
                    # Calculate mean coverage for each category
                    category_means = region_category_matrix.mean()
                    
                    # Identify regions with low category coverage
                    for region in region_category_matrix.index:
                        region_data = region_category_matrix.loc[region]
                        low_coverage_mask = region_data < category_means
                        low_coverage_categories = region_data[low_coverage_mask].index
                        
                        if len(low_coverage_categories) > 0:
                            self.results.content_gaps['opportunity_areas'].append({
                                'region': str(region),
                                'categories': [
                                    cat.replace('is_', '').replace('_', ' ').title()
                                    for cat in low_coverage_categories
                                ]
                            })
            except Exception as e:
                print(f"Error in regional gap analysis: {str(e)}")
        
        # Analyze critical gaps
        try:
            category_columns = [
                col for col in self.df.columns
                if col.startswith('is_') and col in [f'is_{cat.lower()}' for cat in TRAINING_CATEGORIES.keys()]
            ]
            
            for category in category_columns:
                category_name = category.replace('is_', '').replace('_', ' ').title()
                category_count = self.df[category].sum()
                
                if category_count < len(self.df) * 0.1:  # Less than 10% coverage
                    self.results.content_gaps['critical_gaps'].append({
                        'category': category_name,
                        'current_coverage': f"{(category_count / len(self.df)) * 100:.1f}%",
                        'recommendation': "Consider expanding course offerings"
                    })
        except Exception as e:
            print(f"Error in critical gap analysis: {str(e)}")
    
    def analyze_quality(self) -> QualityMetrics:
        """Analyze the quality of the data."""
        quality_metrics = QualityMetrics()
        
        # Calculate completeness score
        available_fields = [field for field in REQUIRED_FIELDS if field in self.df.columns]
        if available_fields:
            completeness_scores = self.df[available_fields].notna().mean(axis=1)
            quality_metrics.completeness_score = completeness_scores.mean()
        
        # Calculate cross-reference score
        if 'cross_reference_count' in self.df.columns:
            cross_ref_counts = pd.to_numeric(
                self.df['cross_reference_count'], errors='coerce'
            ).fillna(1)
            quality_metrics.cross_reference_score = (
                (cross_ref_counts - 1).div(3).clip(0, 1).mean()
            )
        
        # Calculate validation score
        validation_columns = [col for col in self.df.columns if col.endswith('_is_valid')]
        if validation_columns:
            validation_scores = self.df[validation_columns].astype(float)
            quality_metrics.validation_score = validation_scores.mean(axis=1).mean()
        
        # Calculate final quality score
        quality_metrics.quality_score = (
            QUALITY_SCORE_WEIGHTS['completeness'] * quality_metrics.completeness_score +
            QUALITY_SCORE_WEIGHTS['cross_references'] * quality_metrics.cross_reference_score +
            QUALITY_SCORE_WEIGHTS['data_validation'] * quality_metrics.validation_score
        )
        
        return quality_metrics
    
    def analyze_training_focus(self) -> Dict[str, Dict[str, float]]:
        """Analyze training focus distribution and overlaps."""
        focus_stats = {}
        
        for category in TRAINING_CATEGORIES.keys():
            category_col = f'is_{category.lower()}'
            if category_col in self.df.columns:
                focus_stats[category] = {
                    'count': self.df[category_col].sum(),
                    'percentage': (self.df[category_col].sum() / len(self.df) * 100)
                }
            else:
                focus_stats[category] = {
                    'count': 0,
                    'percentage': 0
                }
        
        return focus_stats
    
    def analyze_temporal_patterns(self) -> Dict[str, Dict]:
        """Analyze temporal patterns in the data."""
        temporal_analysis = {}
        
        try:
            if 'course_available_from' in self.df.columns:
                valid_dates = self.df[self.df['course_available_from_is_valid']]
                if not valid_dates.empty:
                    temporal_analysis['creation_patterns'] = {
                        'courses_per_year': valid_dates.groupby(
                            valid_dates['course_available_from'].dt.year
                        ).size().to_dict(),
                        'courses_per_month': valid_dates.groupby(
                            valid_dates['course_available_from'].dt.month
                        ).size().to_dict()
                    }
            
            if 'course_discontinued_from' in self.df.columns:
                active_courses = self.df[
                    (self.df['course_discontinued_from'].isna()) |
                    (self.df['course_discontinued_from'] > pd.Timestamp.now())
                ]
                temporal_analysis['lifecycle'] = {
                    'active_courses': len(active_courses),
                    'discontinued_courses': len(self.df) - len(active_courses)
                }
        except Exception as e:
            print(f"Error in temporal pattern analysis: {str(e)}")
        
        return temporal_analysis
    
    def analyze_content_gaps(self) -> Dict[str, List]:
        """Analyze content gaps based on industry standards."""
        gaps = {
            'critical_gaps': [],
            'opportunity_areas': [],
            'recommendations': []
        }
        
        # Industry standard ratios
        standard_ratios = {
            'mandatory_compliance': 0.15,
            'profession_specific': 0.30,
            'leadership_development': 0.20,
            'interpersonal_skills': 0.15,
            'technical_skills': 0.20
        }
        
        # Calculate current ratios
        total_courses = len(self.df)
        current_ratios = {}
        for category in TRAINING_CATEGORIES.keys():
            cat_col = f'is_{category.lower()}'
            if cat_col in self.df.columns:
                current_ratios[category] = self.df[cat_col].sum() / total_courses
        
        # Compare with standards
        for category, standard in standard_ratios.items():
            if category in current_ratios:
                difference = standard - current_ratios[category]
                if difference > 0.05:  # More than 5% gap
                    gaps['critical_gaps'].append({
                        'category': category,
                        'current_ratio': current_ratios[category],
                        'target_ratio': standard,
                        'gap': difference,
                        'courses_needed': int(difference * total_courses)
                    })
        
        # Analyze regional gaps
        if 'region_entity' in self.df.columns and self.df['region_entity'].notna().any():
            try:
                # Get unique regions and categories
                regions = self.df['region_entity'].dropna().unique()
                category_columns = [col for col in self.df.columns if col.startswith('is_')]
                
                for region in regions:
                    region_df = self.df[self.df['region_entity'] == region]
                    for category in TRAINING_CATEGORIES.keys():
                        cat_col = f'is_{category.lower()}'
                        if cat_col in category_columns:
                            coverage = region_df[cat_col].sum() if not region_df.empty else 0
                            if coverage < 5:  # Less than 5 courses in this category for this region
                                gaps['opportunity_areas'].append({
                                    'region': region,
                                    'category': category.replace('_', ' ').title(),
                                    'current_coverage': int(coverage),
                                    'recommended_minimum': 5
                                })
            except Exception as e:
                print(f"Warning in regional gap analysis: {str(e)}")
        
        return gaps
    
    def predict_course_utilization(self) -> Dict[str, List[Dict]]:
        """Predict course utilization based on historical patterns."""
        predictions = {}
        
        if all(col in self.df.columns for col in ['learner_count', 'quality_score', 'cross_reference_count']):
            # Simple prediction based on quality and cross-references
            self.df['predicted_engagement'] = (
                self.df['quality_score'] * 0.6 +
                self.df['cross_reference_count'].clip(0, 3) / 3 * 0.4
            ) * 100
            
            predictions['engagement_scores'] = self.df[[
                'course_no', 'course_title', 'predicted_engagement'
            ]].to_dict('records')
            
            # Identify potential high-impact courses
            high_potential = self.df[
                (self.df['learner_count'] < self.df['learner_count'].median()) &
                (self.df['predicted_engagement'] > 70)
            ]
            predictions['high_potential_courses'] = high_potential[[
                'course_no', 'course_title', 'predicted_engagement'
            ]].to_dict('records')
        
        return predictions
    
    def analyze_trends(self) -> Dict[str, List[Dict]]:
        """Analyze trends in data quality and content metrics over time."""
        trends = {}
        
        # Quality score trends
        if 'quality_score' in self.df.columns and 'course_available_from' in self.df.columns:
            self.df['year'] = pd.to_datetime(
                self.df['course_available_from'], errors='coerce'
            ).dt.year
            quality_trends = self.df.groupby('year')['quality_score'].agg([
                'mean', 'count'
            ]).reset_index()
            trends['quality_over_time'] = quality_trends.to_dict('records')
        
        # Category evolution
        for category in TRAINING_CATEGORIES.keys():
            cat_col = f'is_{category.lower()}'
            if cat_col in self.df.columns and 'course_available_from' in self.df.columns:
                cat_trend = self.df[self.df[cat_col]].groupby('year').size().reset_index(
                    name='count'
                )
                if not cat_trend.empty:
                    trends[f'{category}_evolution'] = cat_trend.to_dict('records')
        
        return trends
    
    def analyze_learning_impact(self) -> Dict[str, Dict]:
        """Analyze learning impact and engagement metrics."""
        impact_metrics = {
            'usage_metrics': {},
            'engagement_metrics': {},
            'completion_metrics': {},
            'impact_insights': {}
        }
        
        # Calculate usage metrics
        if 'learner_count' in self.df.columns:
            impact_metrics['usage_metrics'].update({
                'total_learners': int(self.df['learner_count'].sum()),
                'avg_learners_per_course': float(self.df['learner_count'].mean()),
                'median_learners_per_course': float(self.df['learner_count'].median())
            })
        
        if 'total_2024_activity' in self.df.columns:
            impact_metrics['usage_metrics'].update({
                'total_activity': int(self.df['total_2024_activity'].sum()),
                'avg_activity_per_course': float(self.df['total_2024_activity'].mean())
            })
        
        # Calculate engagement metrics
        if 'avg_hours_per_completion' in self.df.columns:
            impact_metrics['engagement_metrics'].update({
                'avg_hours_per_completion': float(self.df['avg_hours_per_completion'].mean()),
                'median_hours_per_completion': float(self.df['avg_hours_per_completion'].median()),
                'total_learning_hours': float(
                    (self.df['avg_hours_per_completion'] * self.df['total_2024_activity']).sum()
                )
            })
        
        # Calculate completion metrics
        if all(col in self.df.columns for col in ['learner_count', 'total_2024_activity']):
            self.df['completion_rate'] = (
                self.df['total_2024_activity'] / self.df['learner_count']
            ).clip(0, 1)
            
            impact_metrics['completion_metrics'].update({
                'avg_completion_rate': float(self.df['completion_rate'].mean()),
                'median_completion_rate': float(self.df['completion_rate'].median()),
                'high_completion_courses': len(self.df[self.df['completion_rate'] > 0.8]),
                'low_completion_courses': len(self.df[self.df['completion_rate'] < 0.3])
            })
            
            # Top performing courses
            top_courses = self.df.nlargest(5, 'completion_rate')
            impact_metrics['completion_metrics']['top_courses'] = [
                {
                    'course_title': row['course_title'],
                    'completion_rate': float(row['completion_rate']),
                    'learner_count': int(row['learner_count'])
                }
                for _, row in top_courses.iterrows()
            ]
        
        # Calculate impact insights
        if 'category_name' in self.df.columns:
            # Category performance
            category_metrics = self.df.groupby('category_name').agg({
                'learner_count': 'sum',
                'total_2024_activity': 'sum',
                'completion_rate': 'mean'
            }).round(2)
            
            impact_metrics['impact_insights']['category_performance'] = \
                category_metrics.to_dict('index')
        
        # Add recommendations
        impact_metrics['impact_insights']['recommendations'] = \
            self._generate_learning_recommendations()
        
        return impact_metrics
    
    def _generate_learning_recommendations(self) -> List[Dict[str, str]]:
        """Generate recommendations based on learning impact analysis."""
        recommendations = []
        
        if 'completion_rate' in self.df.columns:
            avg_completion = self.df['completion_rate'].mean()
            low_completion = self.df[self.df['completion_rate'] < avg_completion * 0.5]
            
            if not low_completion.empty:
                recommendations.append({
                    'type': 'warning',
                    'title': 'Low Completion Rates',
                    'description': (
                        f"{len(low_completion)} courses have completion rates "
                        f"below {(avg_completion * 0.5):.1%}"
                    ),
                    'action': 'Review and optimize these courses for better engagement'
                })
        
        if 'learner_count' in self.df.columns:
            low_enrollment = self.df[
                self.df['learner_count'] < self.df['learner_count'].median() * 0.3
            ]
            
            if not low_enrollment.empty:
                recommendations.append({
                    'type': 'info',
                    'title': 'Underutilized Courses',
                    'description': (
                        f"{len(low_enrollment)} courses have significantly "
                        "lower than average enrollment"
                    ),
                    'action': 'Consider promoting these courses or reviewing their relevance'
                })
        
        if 'avg_hours_per_completion' in self.df.columns:
            long_duration = self.df[
                self.df['avg_hours_per_completion'] > \
                self.df['avg_hours_per_completion'].median() * 2
            ]
            
            if not long_duration.empty:
                recommendations.append({
                    'type': 'info',
                    'title': 'Time-Intensive Courses',
                    'description': (
                        f"{len(long_duration)} courses take significantly "
                        "longer than average to complete"
                    ),
                    'action': 'Review these courses for potential optimization'
                })
        
        return recommendations
    
    def get_analysis_results(self) -> AnalysisResults:
        """Get the complete analysis results."""
        # Update quality metrics
        self.results.quality_metrics = self.analyze_quality()
        
        # Update category distribution
        focus_stats = self.analyze_training_focus()
        self.results.category_distribution = {
            category: stats['count']
            for category, stats in focus_stats.items()
        }
        
        # Update temporal patterns
        self.results.temporal_patterns = self.analyze_temporal_patterns()
        
        # Update content gaps
        self.results.content_gaps = self.analyze_content_gaps()
        
        # Update learning impact metrics
        self.results.learning_impact = self.analyze_learning_impact()
        
        return self.results 