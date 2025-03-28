"""Validation rules configuration for the LMS analyzer."""

from typing import Dict, Any
from datetime import datetime, timedelta
import re

def validate_course_no(value: str) -> Dict[str, Any]:
    """Validate course number format."""
    if not value:
        return {"valid": False, "message": "Course number cannot be empty"}
    if not isinstance(value, str):
        return {"valid": False, "message": f"Course number must be a string, got {type(value)}"}
    if not value.strip():
        return {"valid": False, "message": "Course number cannot contain only whitespace"}
    if not re.match(r'^[A-Z0-9-_]+$', value):
        invalid_chars = set(re.findall(r'[^A-Z0-9-_]', value))
        return {"valid": False, "message": f"Course number contains invalid characters: {', '.join(invalid_chars)}"}
    if len(value) > 50:
        return {"valid": False, "message": "Course number cannot exceed 50 characters"}
    return {"valid": True, "message": ""}

def validate_course_title(value: str) -> Dict[str, Any]:
    """Validate course title."""
    if not value:
        return {"valid": False, "message": "Course title cannot be empty"}
    if not isinstance(value, str):
        return {"valid": False, "message": f"Course title must be a string, got {type(value)}"}
    if not value.strip():
        return {"valid": False, "message": "Course title cannot contain only whitespace"}
    if len(value.strip()) < 3:
        return {"valid": False, "message": "Course title must be at least 3 characters long"}
    if len(value.strip()) > 200:
        return {"valid": False, "message": "Course title cannot exceed 200 characters"}
    if re.match(r'^\d+$', value.strip()):
        return {"valid": False, "message": "Course title cannot be a number only"}
    return {"valid": True, "message": ""}

def validate_category_name(value: str) -> Dict[str, Any]:
    """Validate category name."""
    if not value:
        return {"valid": False, "message": "Category name cannot be empty"}
    if not isinstance(value, str):
        return {"valid": False, "message": f"Category name must be a string, got {type(value)}"}
    if not value.strip():
        return {"valid": False, "message": "Category name cannot contain only whitespace"}
    if len(value.strip()) > 100:
        return {"valid": False, "message": "Category name cannot exceed 100 characters"}
    if re.match(r'^\d+$', value.strip()):
        return {"valid": False, "message": "Category name cannot be a number only"}
    return {"valid": True, "message": ""}

def validate_course_description(value: str) -> Dict[str, Any]:
    """Validate course description."""
    if not value:
        return {"valid": False, "message": "Course description cannot be empty"}
    if not isinstance(value, str):
        return {"valid": False, "message": f"Course description must be a string, got {type(value)}"}
    if not value.strip():
        return {"valid": False, "message": "Course description cannot contain only whitespace"}
    if len(value.strip()) < 50:
        return {"valid": False, "message": "Course description must be at least 50 characters long"}
    if len(value.strip()) > 5000:
        return {"valid": False, "message": "Course description cannot exceed 5000 characters"}
    # Check for common issues
    if value.strip().lower() == "no description available":
        return {"valid": False, "message": "Course description cannot be a placeholder"}
    if len(set(value.strip().split())) < 10:  # Check for minimum unique words
        return {"valid": False, "message": "Course description must contain meaningful content"}
    return {"valid": True, "message": ""}

def validate_course_type(value: str) -> Dict[str, Any]:
    """Validate course type."""
    valid_types = ["e-learning", "instructor-led", "blended", "assessment"]
    if not value:
        return {"valid": False, "message": "Course type cannot be empty"}
    if not isinstance(value, str):
        return {"valid": False, "message": f"Course type must be a string, got {type(value)}"}
    if not value.strip():
        return {"valid": False, "message": "Course type cannot contain only whitespace"}
    if value.lower().strip() not in valid_types:
        return {"valid": False, "message": f"Course type must be one of: {', '.join(valid_types)}"}
    return {"valid": True, "message": ""}

def validate_region_entity(value: str) -> Dict[str, Any]:
    """Validate region or department."""
    if not value:
        return {"valid": False, "message": "Region/Department cannot be empty"}
    if not isinstance(value, str):
        return {"valid": False, "message": f"Region/Department must be a string, got {type(value)}"}
    if not value.strip():
        return {"valid": False, "message": "Region/Department cannot contain only whitespace"}
    if len(value.strip()) > 100:
        return {"valid": False, "message": "Region/Department name cannot exceed 100 characters"}
    if re.match(r'^\d+$', value.strip()):
        return {"valid": False, "message": "Region/Department name cannot be a number only"}
    return {"valid": True, "message": ""}

def validate_dates(available_from: Any, discontinued_from: Any = None) -> Dict[str, Any]:
    """Validate course dates."""
    try:
        # Convert to datetime if string
        if isinstance(available_from, str):
            try:
                available_from = datetime.strptime(available_from, "%Y-%m-%d")
            except ValueError:
                return {"valid": False, "message": "Invalid available_from date format. Use YYYY-MM-DD"}
        
        if discontinued_from and isinstance(discontinued_from, str):
            try:
                discontinued_from = datetime.strptime(discontinued_from, "%Y-%m-%d")
            except ValueError:
                return {"valid": False, "message": "Invalid discontinued_from date format. Use YYYY-MM-DD"}
        
        if not available_from:
            return {"valid": False, "message": "Course available from date is required"}
        
        # Validate available_from is not in the future
        if available_from > datetime.now():
            return {"valid": False, "message": "Course available from date cannot be in the future"}
        
        # Validate available_from is not too far in the past
        if available_from < datetime.now() - timedelta(days=365*5):  # 5 years
            return {"valid": False, "message": "Course available from date cannot be more than 5 years in the past"}
        
        if discontinued_from:
            if discontinued_from < available_from:
                return {"valid": False, "message": "Discontinued date cannot be before available date"}
            if discontinued_from > datetime.now():
                return {"valid": False, "message": "Discontinued date cannot be in the future"}
            if discontinued_from - available_from < timedelta(days=30):
                return {"valid": False, "message": "Course duration must be at least 30 days"}
            if discontinued_from - available_from > timedelta(days=365*5):  # 5 years
                return {"valid": False, "message": "Course duration cannot exceed 5 years"}
        
        return {"valid": True, "message": ""}
    except ValueError as e:
        return {"valid": False, "message": f"Invalid date format: {str(e)}"}
    except Exception as e:
        return {"valid": False, "message": f"Error validating dates: {str(e)}"}

def validate_learner_count(value: Any) -> Dict[str, Any]:
    """Validate learner count."""
    if value is None:
        return {"valid": True, "message": ""}
    try:
        count = int(value)
        if count < 0:
            return {"valid": False, "message": "Learner count cannot be negative"}
        if count > 10000:  # Reasonable upper limit
            return {"valid": False, "message": "Learner count exceeds reasonable limit (10,000)"}
        return {"valid": True, "message": ""}
    except (ValueError, TypeError):
        return {"valid": False, "message": f"Learner count must be a valid integer, got {type(value)}"}

def validate_activity_count(value: Any) -> Dict[str, Any]:
    """Validate activity count."""
    if value is None:
        return {"valid": True, "message": ""}
    try:
        count = int(value)
        if count < 0:
            return {"valid": False, "message": "Activity count cannot be negative"}
        if count > 100000:  # Reasonable upper limit
            return {"valid": False, "message": "Activity count exceeds reasonable limit (100,000)"}
        return {"valid": True, "message": ""}
    except (ValueError, TypeError):
        return {"valid": False, "message": f"Activity count must be a valid integer, got {type(value)}"}

def validate_course_keywords(value: Any) -> Dict[str, Any]:
    """Validate course keywords."""
    if value is None:
        return {"valid": True, "message": ""}
    try:
        if isinstance(value, str):
            keywords = [k.strip() for k in value.split(',')]
        elif isinstance(value, list):
            keywords = [k.strip() for k in value]
        else:
            return {"valid": False, "message": f"Keywords must be a string or list, got {type(value)}"}
        
        if not keywords:
            return {"valid": False, "message": "Keywords cannot be empty"}
        if len(keywords) > 10:
            return {"valid": False, "message": "Maximum 10 keywords allowed"}
        if any(len(k) > 50 for k in keywords):
            return {"valid": False, "message": "Each keyword cannot exceed 50 characters"}
        if any(not k for k in keywords):
            return {"valid": False, "message": "Keywords cannot be empty strings"}
        if any(re.match(r'^\d+$', k) for k in keywords):
            return {"valid": False, "message": "Keywords cannot be numbers only"}
        
        return {"valid": True, "message": ""}
    except Exception as e:
        return {"valid": False, "message": f"Error validating keywords: {str(e)}"}

def validate_course_version(value: Any) -> Dict[str, Any]:
    """Validate course version."""
    if value is None:
        return {"valid": True, "message": ""}
    try:
        if not isinstance(value, str):
            return {"valid": False, "message": f"Version must be a string, got {type(value)}"}
        if not value.strip():
            return {"valid": False, "message": "Version cannot be empty"}
        if not re.match(r'^\d+\.\d+\.\d+$', value):
            return {"valid": False, "message": "Version must be in format X.Y.Z (e.g., 1.0.0)"}
        # Validate version numbers are reasonable
        major, minor, patch = map(int, value.split('.'))
        if major > 99 or minor > 99 or patch > 99:
            return {"valid": False, "message": "Version numbers cannot exceed 99"}
        return {"valid": True, "message": ""}
    except Exception as e:
        return {"valid": False, "message": f"Error validating version: {str(e)}"}

VALIDATION_RULES = {
    # Required fields
    "course_no": validate_course_no,
    "course_title": validate_course_title,
    "category_name": validate_category_name,
    "course_description": validate_course_description,
    "course_type": validate_course_type,
    "region_entity": validate_region_entity,
    "course_available_from": lambda x: validate_dates(x),
    
    # Optional fields
    "course_discontinued_from": lambda x: validate_dates(None, x),
    "learner_count": validate_learner_count,
    "total_2024_activity": validate_activity_count,
    "course_keywords": validate_course_keywords,
    "course_version": validate_course_version
} 