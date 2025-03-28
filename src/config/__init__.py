"""Configuration for the LMS analyzer."""

from .column_mappings import get_column_mapping as COLUMN_MAPPINGS, REQUIRED_COLUMNS, COLUMN_TYPES
from .validation_rules import VALIDATION_RULES
from .analysis_params import (
    QUALITY_THRESHOLDS,
    SIMILARITY_THRESHOLDS,
    ACTIVITY_THRESHOLDS,
    VIZ_PARAMS
)

__all__ = [
    'COLUMN_MAPPINGS',
    'REQUIRED_COLUMNS',
    'COLUMN_TYPES',
    'VALIDATION_RULES',
    'QUALITY_THRESHOLDS',
    'SIMILARITY_THRESHOLDS',
    'ACTIVITY_THRESHOLDS',
    'VIZ_PARAMS'
]