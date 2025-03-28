"""LMS Content Analysis Dashboard package."""

__version__ = "1.0.0"

from .models.data_models import (
    AnalysisResults,
    ValidationResult,
    QualityMetrics,
    ActivityMetrics,
    SimilarityMetrics,
    Recommendation,
    Alert,
    Course,
    Severity,
    VisualizationConfig
)

__all__ = [
    'AnalysisResults',
    'ValidationResult',
    'QualityMetrics',
    'ActivityMetrics',
    'SimilarityMetrics',
    'Recommendation',
    'Alert',
    'Course',
    'Severity',
    'VisualizationConfig'
] 