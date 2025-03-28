"""UI components for the LMS analyzer."""

from .overview import display_metrics
from .quality import display_quality_metrics
from .similarity import display_similarity_metrics
from .activity import display_activity_metrics
from .recommendations import display_recommendations

__all__ = [
    'display_metrics',
    'display_quality_metrics',
    'display_similarity_metrics',
    'display_activity_metrics',
    'display_recommendations'
] 