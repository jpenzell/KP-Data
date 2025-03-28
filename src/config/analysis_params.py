"""Analysis parameters configuration for the LMS analyzer."""

from typing import Dict, Any
from datetime import datetime

# Quality Analysis Parameters
QUALITY_THRESHOLDS = {
    "completeness": 0.8,  # Minimum required completeness score
    "metadata": 0.7,      # Minimum required metadata score
    "content": 0.7,       # Minimum required content quality score
    "overall": 0.75       # Minimum required overall quality score
}

# Similarity Analysis Parameters
SIMILARITY_THRESHOLDS = {
    "high": 0.9,          # Threshold for high similarity
    "medium": 0.7,        # Threshold for medium similarity
    "low": 0.5           # Threshold for low similarity
}

# Activity Analysis Parameters
ACTIVITY_THRESHOLDS = {
    "recent_days": 90,    # Number of days to consider for recent activity
    "min_learners": 5,    # Minimum number of learners for active course
    "min_completions": 3, # Minimum number of completions for active course
    "min_activity": 10,   # Minimum activity count for active course
    "high_activity": 50   # Threshold for high activity level
}

# Content Analysis Parameters
CONTENT_ANALYSIS = {
    "min_description_length": 50,    # Minimum length for course description
    "min_title_length": 10,          # Minimum length for course title
    "max_duplicate_threshold": 0.95,  # Maximum allowed similarity for duplicates
    "min_keywords": 3,               # Minimum number of keywords required
    "max_keywords": 10,              # Maximum number of keywords allowed
    "min_tags": 1,                   # Minimum number of tags required
    "max_tags": 5                    # Maximum number of tags allowed
}

# Visualization Parameters
VIZ_PARAMS = {
    "max_categories_display": 10,    # Maximum number of categories to display
    "max_departments_display": 15,   # Maximum number of departments to display
    "chart_height": 400,             # Default chart height
    "chart_width": 800,              # Default chart width
    "color_scheme": "tableau10",     # Default color scheme
    "font_size": 12,                 # Default font size
    "legend_position": "right",      # Default legend position
    "tooltip_format": "html"         # Default tooltip format
}

# Analysis Configuration
ANALYSIS_CONFIG = {
    "enable_duplicate_detection": True,    # Whether to detect duplicate content
    "enable_cross_reference": True,        # Whether to analyze cross-references
    "enable_quality_scoring": True,        # Whether to calculate quality scores
    "enable_activity_analysis": True,      # Whether to analyze activity patterns
    "enable_recommendations": True,        # Whether to generate recommendations
    "enable_alerts": True,                 # Whether to generate alerts
    "max_recommendations": 10,             # Maximum number of recommendations to generate
    "max_alerts": 20,                      # Maximum number of alerts to generate
    "min_confidence": 0.7                  # Minimum confidence for recommendations
}

# Export Configuration
EXPORT_CONFIG = {
    "excel_format": {
        "sheet_name": "LMS Analysis",
        "max_rows_per_sheet": 10000,
        "include_charts": True,
        "include_summary": True
    },
    "csv_format": {
        "include_header": True,
        "encoding": "utf-8",
        "delimiter": ","
    },
    "json_format": {
        "indent": 2,
        "include_metadata": True
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": f"lms_analyzer_{datetime.now().strftime('%Y%m%d')}.log",
    "max_size": 10485760,  # 10MB
    "backup_count": 5
}

def get_analysis_params() -> Dict[str, Any]:
    """Get all analysis parameters."""
    return {
        "quality_thresholds": QUALITY_THRESHOLDS,
        "similarity_thresholds": SIMILARITY_THRESHOLDS,
        "activity_thresholds": ACTIVITY_THRESHOLDS,
        "content_analysis": CONTENT_ANALYSIS,
        "viz_params": VIZ_PARAMS,
        "analysis_config": ANALYSIS_CONFIG,
        "export_config": EXPORT_CONFIG,
        "logging_config": LOGGING_CONFIG
    } 