"""
Utility functions for the LMS Analyzer application.

This module contains various utility functions for data processing, visualization,
and optimization that are used throughout the application.
"""

# Import visualization utilities
from .visualization import (
    create_bar_chart,
    create_line_chart,
    create_scatter_plot,
    create_heatmap,
    create_wordcloud
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

# Import similarity analysis utilities
from .similarity import compute_content_similarity

__all__ = [
    # Visualization
    'create_bar_chart',
    'create_line_chart',
    'create_scatter_plot',
    'create_heatmap',
    'create_wordcloud',
    
    # Optimization
    'remove_unnamed_columns',
    'clean_column_names',
    'enhanced_date_repair',
    'process_in_chunks',
    'optimize_memory_usage_enhanced',
    'measure_execution_time',
    'fix_row_count_discrepancy',
    
    # Similarity Analysis
    'compute_content_similarity'
] 