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

# Import similarity analysis utilities
from .similarity import (
    compute_content_similarity,
    find_potential_duplicates,
    plot_similarity_network,
    plot_department_similarity_heatmap,
    generate_consolidation_recommendations,
    analyze_content_similarity
)

__all__ = [
    # Visualization
    'plot_quality_distribution', 
    'plot_timeline', 
    'create_wordcloud',
    'plot_data_quality_metrics',
    
    # Optimization
    'remove_unnamed_columns',
    'clean_column_names',
    'enhanced_date_repair',
    'process_in_chunks',
    'optimize_memory_usage_enhanced',
    'measure_execution_time',
    'fix_row_count_discrepancy',
    
    # Similarity Analysis
    'compute_content_similarity',
    'find_potential_duplicates',
    'plot_similarity_network',
    'plot_department_similarity_heatmap',
    'generate_consolidation_recommendations',
    'analyze_content_similarity'
] 