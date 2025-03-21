"""Data models for the LMS Analyzer application."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd

@dataclass
class Course:
    """Represents a course in the LMS system."""
    course_no: str
    course_title: str
    duration_mins: Optional[int] = None
    course_version: Optional[str] = None
    course_available_from: Optional[datetime] = None
    category_name: Optional[str] = None
    data_source: Optional[str] = None
    total_2024_activity: Optional[int] = 0
    avg_hours_per_completion: Optional[float] = None
    learner_count: Optional[int] = 0
    course_description: Optional[str] = None
    course_abstract: Optional[str] = None
    course_keywords: Optional[str] = None
    course_type: Optional[str] = None
    region_entity: Optional[str] = None
    course_created_by: Optional[str] = None
    course_discontinued_from: Optional[datetime] = None
    cross_reference_count: int = 1
    quality_score: float = 0.0
    
    # Category flags
    is_leadership_development: bool = False
    is_managerial_supervisory: bool = False
    is_mandatory_compliance: bool = False
    is_profession_specific: bool = False
    is_interpersonal_skills: bool = False
    is_it_systems: bool = False
    is_clinical: bool = False
    is_nursing: bool = False
    is_pharmacy: bool = False
    is_safety: bool = False
    has_direct_reports: bool = False

@dataclass
class QualityMetrics:
    """Represents quality metrics for a course."""
    completeness_score: float = 0.0
    cross_reference_score: float = 0.0
    validation_score: float = 0.0
    quality_score: float = 0.0
    validation_issues: Dict[str, List[str]] = field(default_factory=dict)

@dataclass
class AnalysisResults:
    """Container for various analysis results."""
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_courses: int = 0
    active_courses: int = 0
    total_learners: int = 0
    regions_covered: int = 0
    data_sources: int = 0
    cross_referenced_courses: int = 0
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    category_distribution: Dict[str, int] = field(default_factory=dict)
    regional_distribution: Dict[str, int] = field(default_factory=dict)
    temporal_patterns: Dict[str, Dict] = field(default_factory=dict)
    content_gaps: Dict[str, List] = field(default_factory=dict)
    learning_impact: Dict[str, Dict] = field(default_factory=dict)

@dataclass
class ValidationResult:
    """Represents the result of a data validation operation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

@dataclass
class MissingDataConfig:
    """Configuration for handling missing data."""
    field: str
    default_value: Union[int, float, str]
    description: str
    handling_approach: str = "skip"  # One of: skip, use_default, prompt_user

@dataclass
class VisualizationConfig:
    """Configuration class for visualization settings."""
    title: str
    template: str = "plotly_white"
    height: int = 600
    width: int = 800
    color: Optional[str] = None
    size: Optional[str] = None
    hover_data: Optional[List[str]] = None
    show_legend: bool = True
    legend_title: Optional[str] = None
    margin: Dict[str, int] = None
    background_color: str = "white"
    font_family: str = "Arial"
    font_size: int = 12
    title_font_size: int = 16
    axis_font_size: int = 12
    x_label: Optional[str] = None
    y_label: Optional[str] = None

    def __post_init__(self):
        if self.margin is None:
            self.margin = {"l": 50, "r": 50, "t": 50, "b": 50}
        if self.x_label is None:
            self.x_label = ""
        if self.y_label is None:
            self.y_label = "" 