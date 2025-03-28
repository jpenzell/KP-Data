"""Column mappings configuration for the LMS analyzer."""

from typing import Dict, List, Any

# Standard column names used throughout the application
STANDARD_COLUMNS = {
    "course_no": "Course Number",
    "course_title": "Course Title",
    "category_name": "Category",
    "course_description": "Description",
    "course_type": "Type",
    "region_entity": "Region/Department",
    "course_available_from": "Available From",
    "course_discontinued_from": "Discontinued From",
    "learner_count": "Learner Count",
    "total_2024_activity": "2024 Activity",
    "data_source": "Data Source",
    "cross_reference_count": "Cross References",
    "last_updated": "Last Updated",
    "keywords": "Keywords",
    "tags": "Tags"
}

# Alternative column names that might appear in input files
ALTERNATIVE_COLUMNS = {
    "course_no": ["course_id", "course_number", "course_code", "id"],
    "course_title": ["title", "name", "course_name"],
    "category_name": ["category", "department", "division", "group"],
    "course_description": ["description", "desc", "details", "summary"],
    "course_type": ["type", "format", "delivery_method"],
    "region_entity": ["region", "department", "division", "location"],
    "course_available_from": ["available_from", "start_date", "published_date"],
    "course_discontinued_from": ["discontinued_from", "end_date", "expiry_date"],
    "learner_count": ["learners", "enrolled_count", "participants"],
    "total_2024_activity": ["activity_2024", "completions_2024", "usage_2024"],
    "data_source": ["source", "origin", "file_source"],
    "cross_reference_count": ["references", "related_courses"],
    "last_updated": ["updated_at", "modified_date", "last_modified"],
    "keywords": ["tags", "topics", "subjects"],
    "tags": ["labels", "categories", "topics"]
}

# Required columns for data validation
REQUIRED_COLUMNS = [
    "course_no",
    "course_title",
    "category_name",
    "course_description",
    "course_type",
    "region_entity",
    "course_available_from"
]

# Optional columns that may be present
OPTIONAL_COLUMNS = [
    "course_discontinued_from",
    "learner_count",
    "total_2024_activity",
    "data_source",
    "cross_reference_count",
    "last_updated",
    "keywords",
    "tags"
]

# Data types for each column
COLUMN_TYPES = {
    "course_no": str,
    "course_title": str,
    "category_name": str,
    "course_description": str,
    "course_type": str,
    "region_entity": str,
    "course_available_from": "datetime64[ns]",
    "course_discontinued_from": "datetime64[ns]",
    "learner_count": "Int64",
    "total_2024_activity": "Int64",
    "data_source": str,
    "cross_reference_count": "Int64",
    "last_updated": "datetime64[ns]",
    "keywords": list,
    "tags": set
}

# Column descriptions for documentation and tooltips
COLUMN_DESCRIPTIONS = {
    "course_no": "Unique identifier for the course",
    "course_title": "Full title of the course",
    "category_name": "Category or department the course belongs to",
    "course_description": "Detailed description of the course content",
    "course_type": "Type of course delivery (e-learning, instructor-led, etc.)",
    "region_entity": "Region or department responsible for the course",
    "course_available_from": "Date when the course becomes available",
    "course_discontinued_from": "Date when the course is discontinued",
    "learner_count": "Number of learners enrolled in the course",
    "total_2024_activity": "Total activity/completions in 2024",
    "data_source": "Source of the course data",
    "cross_reference_count": "Number of related or cross-referenced courses",
    "last_updated": "Last date the course was updated",
    "keywords": "List of keywords describing the course",
    "tags": "Set of tags categorizing the course"
}

def get_column_mapping() -> Dict[str, str]:
    """Get the complete column mapping including alternatives."""
    mapping = {}
    for standard, alternatives in ALTERNATIVE_COLUMNS.items():
        mapping[standard] = standard
        for alt in alternatives:
            mapping[alt] = standard
    return mapping

def get_column_type(column_name: str) -> Any:
    """Get the data type for a column."""
    return COLUMN_TYPES.get(column_name, str)

def get_column_description(column_name: str) -> str:
    """Get the description for a column."""
    return COLUMN_DESCRIPTIONS.get(column_name, "")

def is_required_column(column_name: str) -> bool:
    """Check if a column is required."""
    return column_name in REQUIRED_COLUMNS

def is_optional_column(column_name: str) -> bool:
    """Check if a column is optional."""
    return column_name in OPTIONAL_COLUMNS 