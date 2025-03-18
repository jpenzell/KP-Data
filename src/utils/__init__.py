"""
Utility functions for the LMS Analyzer application.

This module contains various utility functions for data processing, visualization,
and optimization that are used throughout the application.
"""

# Import visualization utilities
from .visualization import (
    plot_quality_distribution,
    plot_timeline,
    create_wordcloud,
    plot_data_quality_metrics
)

# Import optimization utilities
from .optimizations import (
    remove_unnamed_columns,
    clean_column_names,
    enhanced_date_repair,
    process_in_chunks,
    optimize_memory_usage_enhanced,
    measure_execution_time,
    fix_row_count_discrepancy
)

__all__ = [
    'plot_quality_distribution', 
    'plot_timeline', 
    'create_wordcloud',
    'plot_data_quality_metrics',
    'remove_unnamed_columns',
    'clean_column_names',
    'enhanced_date_repair',
    'process_in_chunks',
    'optimize_memory_usage_enhanced',
    'measure_execution_time',
    'fix_row_count_discrepancy'
] 