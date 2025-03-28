"""Recommendation service for the LMS Content Analysis Dashboard."""

from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime, timedelta
import logging
from models.data_models import Recommendation, Course, AnalysisConfig

logger = logging.getLogger(__name__)

class RecommendationService:
    """Service for generating course recommendations."""
    
    def __init__(self, df: pd.DataFrame, config: Optional[AnalysisConfig] = None):
        """Initialize the recommendation service.
        
        Args:
            df: DataFrame containing course data
            config: Optional analysis configuration
        """
        self.df = df
        self.config = config or AnalysisConfig()
        logger.info("Recommendation service initialized")
    
    def generate_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on course analysis.
        
        Returns:
            List of recommendations
        """
        try:
            recommendations = []
            
            # Quality-based recommendations
            quality_recs = self._generate_quality_recommendations()
            recommendations.extend(quality_recs)
            
            # Activity-based recommendations
            activity_recs = self._generate_activity_recommendations()
            recommendations.extend(activity_recs)
            
            # Similarity-based recommendations
            similarity_recs = self._generate_similarity_recommendations()
            recommendations.extend(similarity_recs)
            
            # Sort by impact/effort ratio
            recommendations.sort(key=lambda x: x.impact_score / x.effort_score, reverse=True)
            
            # Limit to max recommendations
            recommendations = recommendations[:self.config.max_recommendations]
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _generate_quality_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on content quality."""
        recommendations = []
        
        try:
            # Check if required fields exist
            required_fields = ['course_no', 'course_title', 'category_name', 'course_description']
            if not all(field in self.df.columns for field in required_fields):
                logger.warning("Missing required fields for quality recommendations")
                return recommendations
            
            # Check description quality
            desc_mask = self.df['course_description'].str.len() < self.config.min_description_length
            for _, row in self.df[desc_mask].iterrows():
                recommendations.append(Recommendation(
                    course_no=row['course_no'],
                    title=row['course_title'],
                    category=row['category_name'],
                    type="quality",
                    description=f"Course description is too short (current: {len(row['course_description'])} chars, minimum: {self.config.min_description_length} chars)",
                    impact_score=0.7,
                    effort_score=0.3,
                    priority="high"
                ))
            
            # Check keyword coverage if keywords field exists
            if 'keywords' in self.df.columns:
                keyword_mask = self.df['keywords'].apply(lambda x: len(x) < self.config.min_keywords if isinstance(x, list) else False)
                for _, row in self.df[keyword_mask].iterrows():
                    recommendations.append(Recommendation(
                        course_no=row['course_no'],
                        title=row['course_title'],
                        category=row['category_name'],
                        type="quality",
                        description=f"Course has insufficient keywords (current: {len(row['keywords'])}, minimum: {self.config.min_keywords})",
                        impact_score=0.6,
                        effort_score=0.4,
                        priority="medium"
                    ))
            
            logger.info(f"Generated {len(recommendations)} quality-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating quality recommendations: {str(e)}")
            return []
    
    def _generate_activity_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on course activity."""
        recommendations = []
        
        try:
            # Check if required fields exist
            required_fields = ['course_no', 'course_title', 'category_name']
            if not all(field in self.df.columns for field in required_fields):
                logger.warning("Missing required fields for activity recommendations")
                return recommendations
            
            # Check learner count if field exists
            if 'learner_count' in self.df.columns:
                learner_mask = self.df['learner_count'] < self.config.min_learner_count
                for _, row in self.df[learner_mask].iterrows():
                    recommendations.append(Recommendation(
                        course_no=row['course_no'],
                        title=row['course_title'],
                        category=row['category_name'],
                        type="activity",
                        description=f"Low learner engagement (current: {row['learner_count']}, minimum: {self.config.min_learner_count})",
                        impact_score=0.8,
                        effort_score=0.5,
                        priority="high"
                    ))
            
            # Check recent activity if field exists
            if 'total_2024_activity' in self.df.columns:
                recent_days = self.config.recent_activity_days
                activity_mask = self.df['total_2024_activity'] < self.config.activity_threshold
                for _, row in self.df[activity_mask].iterrows():
                    recommendations.append(Recommendation(
                        course_no=row['course_no'],
                        title=row['course_title'],
                        category=row['category_name'],
                        type="activity",
                        description=f"Low recent activity (last {recent_days} days)",
                        impact_score=0.7,
                        effort_score=0.4,
                        priority="medium"
                    ))
            
            logger.info(f"Generated {len(recommendations)} activity-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating activity recommendations: {str(e)}")
            return []
    
    def _generate_similarity_recommendations(self) -> List[Recommendation]:
        """Generate recommendations based on content similarity."""
        recommendations = []
        
        try:
            # Check if required fields exist
            required_fields = ['course_no', 'course_title', 'category_name']
            if not all(field in self.df.columns for field in required_fields):
                logger.warning("Missing required fields for similarity recommendations")
                return recommendations
            
            # Check for high similarity courses if field exists
            if 'similarity_score' in self.df.columns:
                similarity_mask = self.df['similarity_score'] > self.config.similarity_threshold
                for _, row in self.df[similarity_mask].iterrows():
                    recommendations.append(Recommendation(
                        course_no=row['course_no'],
                        title=row['course_title'],
                        category=row['category_name'],
                        type="similarity",
                        description=f"High content similarity detected (score: {row['similarity_score']:.2f})",
                        impact_score=0.6,
                        effort_score=0.7,
                        priority="medium"
                    ))
            
            logger.info(f"Generated {len(recommendations)} similarity-based recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating similarity recommendations: {str(e)}")
            return [] 