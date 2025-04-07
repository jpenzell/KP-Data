"""Data models for the LMS Content Analysis Dashboard."""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union, Set
from datetime import datetime
from enum import Enum
import re
from dataclasses import dataclass, field
import pandas as pd

class AnalysisConfig(BaseModel):
    """Configuration for content analysis."""
    # Quality thresholds
    quality_threshold: float = 0.7
    min_description_length: int = 50
    min_keywords: int = 3
    max_keywords: int = 10
    
    # Activity thresholds
    activity_threshold: int = 100
    recent_activity_days: int = 30
    min_learner_count: int = 5
    
    # Similarity thresholds
    similarity_threshold: float = 0.8
    min_similarity_score: float = 0.6
    max_similarity_score: float = 0.95
    
    # Analysis parameters
    analysis_period_days: int = 365
    min_course_age_days: int = 30
    max_course_age_days: int = 365 * 5  # 5 years
    
    # Alert thresholds
    alert_thresholds: Dict[str, float] = {
        "quality": 0.6,
        "activity": 0.5,
        "similarity": 0.8
    }
    
    # Recommendation settings
    max_recommendations: int = 10
    min_impact_score: float = 0.5
    max_effort_score: float = 0.8
    
    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True

class Severity(str, Enum):
    """Severity levels for alerts and validation results."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"

class Recommendation(BaseModel):
    """Model for course recommendations."""
    title: str
    description: str
    impact: str = Field(default="Medium")  # High, Medium, Low
    effort: str = Field(default="Medium")  # High, Medium, Low
    implementation: str = ""
    priority: int = Field(default=1)
    category: str = Field(default="General")
    action_items: List[str] = []
    related_courses: List[str] = []
    
    @validator('impact', 'effort')
    def validate_impact_effort(cls, v):
        valid_values = {"High", "Medium", "Low"}
        if v not in valid_values:
            raise ValueError(f'Value must be one of: {", ".join(valid_values)}')
        return v
    
    @validator('priority')
    def validate_priority(cls, v):
        if v < 1 or v > 10:
            raise ValueError(f'Priority must be between 1 and 10')
        return v

class Alert(BaseModel):
    """Model for system alerts and notifications."""
    message: str
    severity: Severity
    timestamp: datetime = Field(default_factory=datetime.now)
    source: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class ValidationResult(BaseModel):
    """Validation result for data validation."""
    severity: Severity = Severity.INFO
    message: str = ""
    is_valid: bool = True
    field: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class QualityMetrics(BaseModel):
    """Model for course quality metrics."""
    completeness_score: float = Field(ge=0.0, le=1.0)
    metadata_score: float = Field(ge=0.0, le=1.0)
    content_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    missing_fields: List[str] = []
    improvement_areas: List[str] = []

class ActivityMetrics(BaseModel):
    """Model for course activity metrics."""
    active_courses: int = Field(ge=0)
    recent_completions: int = Field(ge=0)
    average_completion_rate: Optional[float] = Field(default=None)
    last_activity_date: Optional[datetime] = None
    activity_trend: Dict[str, int] = Field(default_factory=dict)
    activity_distribution: Dict[str, int] = Field(default_factory=dict)
    recent_activities: List[Dict[str, Any]] = Field(default_factory=list)
    activity_recommendations: List[str] = Field(default_factory=list)
    
    @validator('average_completion_rate')
    def check_completion_rate(cls, v):
        if v is not None:
            try:
                rate = float(v)
                return max(0.0, min(1.0, rate))  # Clamp between 0 and 1
            except (ValueError, TypeError):
                return None
        return None

class SimilarityMetrics(BaseModel):
    """Model for course similarity metrics."""
    total_pairs: int = Field(ge=0)
    high_similarity_pairs: int = Field(ge=0)
    cross_department_pairs: int = Field(ge=0)
    average_similarity: float = Field(ge=0.0, le=1.0, default=0.0)
    similarity_distribution: Dict[str, int] = Field(default_factory=lambda: {"high": 0, "medium": 0, "low": 0})
    duplicate_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    
    @validator('average_similarity')
    def check_similarity(cls, v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0
            
    @validator('duplicate_candidates')
    def check_candidates(cls, v):
        # Ensure each candidate has consistent field names
        for candidate in v:
            if 'course1' not in candidate:
                candidate['course1'] = candidate.get('course_no', '')
            if 'course2' not in candidate:
                candidate['course2'] = candidate.get('similar_course_no', '')
            if 'similarity' not in candidate:
                candidate['similarity'] = candidate.get('similarity_score', 0.0)
            if 'cross_department' not in candidate:
                candidate['cross_department'] = False
        return v

class AnalysisResults(BaseModel):
    """Model for analysis results."""
    timestamp: datetime = Field(default_factory=datetime.now)
    total_courses: int = Field(ge=0)
    active_courses: int = Field(ge=0)
    total_learners: Optional[int] = None
    regions_covered: Optional[int] = None
    quality_metrics: Optional[QualityMetrics] = None
    activity_metrics: Optional[ActivityMetrics] = None
    similarity_metrics: Optional[SimilarityMetrics] = None
    recommendations: List[Recommendation] = []
    alerts: List[Alert] = []
    summary: Dict[str, Any] = {}
    analyzer: Optional[Any] = None
    courses_df: Optional[Any] = None
    
    class Config:
        """Pydantic model configuration."""
        arbitrary_types_allowed = True

class Course(BaseModel):
    """Course model."""
    course_no: str
    course_title: str
    category_name: str
    course_description: str
    course_type: str
    region_entity: str
    course_available_from: datetime
    course_discontinued_from: Optional[datetime] = None
    learner_count: Optional[int] = None
    total_2024_activity: Optional[int] = None
    data_source: Optional[str] = None
    cross_reference_count: Optional[int] = 0
    last_updated: Optional[datetime] = None
    keywords: Optional[List[str]] = []
    tags: Optional[Set[str]] = set()
    metadata: Optional[Dict[str, Any]] = None
    course_version: Optional[str] = None
    course_created_by: Optional[str] = None
    course_duration_hours: Optional[float] = None
    is_active: Optional[bool] = True
    quality_score: Optional[float] = None
    activity_level: Optional[str] = None
    course_age_days: Optional[int] = None

    @validator('course_no')
    def validate_course_no(cls, v):
        if not v or not v.strip():
            raise ValueError('Course number cannot be empty')
        if not re.match(r'^[A-Z0-9-_]+$', v.strip()):
            raise ValueError('Course number can only contain uppercase letters, numbers, hyphens, and underscores')
        return v.strip()

    @validator('course_title')
    def validate_course_title(cls, v):
        if not v or not v.strip():
            raise ValueError('Course title cannot be empty')
        if len(v.strip()) < 3:
            raise ValueError('Course title must be at least 3 characters long')
        if len(v.strip()) > 200:
            raise ValueError('Course title cannot exceed 200 characters')
        return v.strip()

    @validator('course_description')
    def validate_course_description(cls, v):
        if not v or not v.strip():
            raise ValueError('Course description cannot be empty')
        if len(v.strip()) < 50:
            raise ValueError('Course description must be at least 50 characters long')
        if len(v.strip()) > 5000:
            raise ValueError('Course description cannot exceed 5000 characters')
        return v.strip()

    @validator('course_type')
    def validate_course_type(cls, v):
        valid_types = {"e-learning", "instructor-led", "blended", "assessment"}
        if not v or not v.strip():
            raise ValueError('Course type cannot be empty')
        if v.lower().strip() not in valid_types:
            raise ValueError(f'Course type must be one of: {", ".join(valid_types)}')
        return v.lower().strip()

    @validator('region_entity')
    def validate_region_entity(cls, v):
        if not v or not v.strip():
            raise ValueError('Region/Department cannot be empty')
        if len(v.strip()) > 100:
            raise ValueError('Region/Department name cannot exceed 100 characters')
        return v.strip()

    @validator('learner_count')
    def validate_learner_count(cls, v):
        if v is not None:
            if v < 0:
                raise ValueError('Learner count cannot be negative')
            if v > 10000:
                raise ValueError('Learner count exceeds reasonable limit')
        return v

    @validator('total_2024_activity')
    def validate_activity_count(cls, v):
        if v is not None:
            if v < 0:
                raise ValueError('Activity count cannot be negative')
            if v > 100000:
                raise ValueError('Activity count exceeds reasonable limit')
        return v

    @validator('course_version')
    def validate_course_version(cls, v):
        if v is not None:
            if not re.match(r'^\d+\.\d+\.\d+$', v):
                raise ValueError('Version must be in format X.Y.Z')
        return v

    @validator('keywords')
    def validate_keywords(cls, v):
        if v is not None:
            if len(v) > 10:
                raise ValueError('Maximum 10 keywords allowed')
            if any(len(k) > 50 for k in v):
                raise ValueError('Each keyword cannot exceed 50 characters')
        return v

    @validator('course_duration_hours')
    def validate_duration(cls, v):
        if v is not None:
            if v < 0:
                raise ValueError('Duration cannot be negative')
            if v > 1000:  # Reasonable upper limit
                raise ValueError('Duration exceeds reasonable limit')
        return v

    @validator('quality_score')
    def validate_quality_score(cls, v):
        if v is not None:
            if not 0 <= v <= 1:
                raise ValueError('Quality score must be between 0 and 1')
        return v

    @validator('activity_level')
    def validate_activity_level(cls, v):
        if v is not None:
            valid_levels = {'none', 'low', 'medium', 'high'}
            if v.lower() not in valid_levels:
                raise ValueError(f'Activity level must be one of: {", ".join(valid_levels)}')
        return v

    @validator('course_age_days')
    def validate_course_age(cls, v):
        if v is not None:
            if v < 0:
                raise ValueError('Course age cannot be negative')
            if v > 365*5:  # 5 years
                raise ValueError('Course age exceeds reasonable limit')
        return v

    class Config:
        """Pydantic model configuration."""
        validate_assignment = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            set: list
        }

class VisualizationConfig(BaseModel):
    """Configuration for visualizations."""
    title: Optional[str] = None
    subtitle: Optional[str] = None
    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    width: Optional[int] = 800
    height: Optional[int] = 400
    theme: Optional[str] = "streamlit"
    color_scheme: Optional[str] = "tableau10"
    show_legend: bool = True
    interactive: bool = True 
    tooltips: Optional[List[str]] = None
    annotations: Optional[List[Dict[str, Any]]] = None 