"""Data validation service for the LMS Content Analysis Dashboard."""

import pandas as pd
import re
import logging
from typing import List, Dict, Any
from pathlib import Path

from ..config.column_mappings import REQUIRED_COLUMNS, OPTIONAL_COLUMNS, COLUMN_TYPES
from ..config.validation_rules import VALIDATION_RULES
from ..models.data_models import ValidationResult, Severity
from ..config.constants import REQUIRED_FIELDS

# Get logger
logger = logging.getLogger(__name__)

class DataValidator:
    """Service for validating LMS data."""
    
    def __init__(self):
        self.validation_results: List[ValidationResult] = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validate the entire dataframe."""
        self.validation_results = []
        
        # Check required columns - be more lenient with column name casing
        lowercase_columns = [col.lower() for col in df.columns]
        self._validate_required_columns(df, lowercase_columns)
        
        # Validate each row - use warnings not errors
        for idx, row in df.iterrows():
            self._validate_row(row, idx)
        
        return self.validation_results
    
    def _validate_required_columns(self, df: pd.DataFrame, lowercase_columns: List[str]) -> None:
        """Validate that all required columns are present."""
        missing_columns = [col for col in REQUIRED_COLUMNS 
                           if col not in df.columns and col.lower() not in lowercase_columns]
        
        if missing_columns:
            missing_cols_message = f"Missing columns: {', '.join(missing_columns)}"
            logger.warning(missing_cols_message)
            self.validation_results.append(
                ValidationResult(
                    severity=Severity.WARNING,
                    message=missing_cols_message,
                    is_valid=True  # Still valid but with warnings
                )
            )
    
    def _validate_row(self, row: pd.Series, row_index: int) -> None:
        """Validate a single row of data."""
        # Validate required fields - but only warn, don't fail
        for col in REQUIRED_COLUMNS:
            if col in row.index and pd.isna(row[col]):
                self.validation_results.append(
                    ValidationResult(
                        severity=Severity.WARNING,
                        message=f"Missing required value for {col} in row {row_index + 1}",
                        is_valid=True
                    )
                )
        
        # Safely validate course number format
        if 'course_no' in row.index and pd.notna(row['course_no']):
            course_no = str(row['course_no'])
            # Check for invalid characters but don't fail on it
            invalid_chars = [':', ',', ' ']
            # Handle zero-width space separately to avoid f-string escape sequence issues
            if any(c in course_no for c in invalid_chars) or '\u200b' in course_no:
                # Create list of found characters
                found_chars = [c for c in invalid_chars if c in course_no]
                if '\u200b' in course_no:
                    found_chars.append('zero-width space')
                
                self.validation_results.append(
                    ValidationResult(
                        severity=Severity.INFO,  # Downgrade to info, since we'll fix it
                        message=f"Row {row_index + 1}: Course number contains invalid characters: {', '.join(found_chars)}",
                        is_valid=True
                    )
                )
        
        # Safely validate version format
        if 'course_version' in row.index and pd.notna(row['course_version']):
            try:
                version = str(row['course_version']).strip()
                # Just log format issues but don't fail
                if not re.match(r'^\d+(\.\d+)*$', version):
                    self.validation_results.append(
                        ValidationResult(
                            severity=Severity.INFO,
                            message=f"Row {row_index + 1}: Version should ideally be in format X.Y.Z (e.g., 1.0.0)",
                            is_valid=True
                        )
                    )
            except Exception as e:
                self.validation_results.append(
                    ValidationResult(
                        severity=Severity.WARNING,
                        message=f"Row {row_index + 1}: Version format error: {str(e)}",
                        is_valid=True
                    )
                )
        
        # Safely validate date formats
        for date_field in ['course_available_from', 'course_discontinued_from']:
            if date_field in row.index and pd.notna(row[date_field]):
                if not isinstance(row[date_field], pd.Timestamp):
                    try:
                        # Try to parse as date
                        pd.to_datetime(row[date_field])
                    except:
                        # Just log the issue but don't fail
                        self.validation_results.append(
                            ValidationResult(
                                severity=Severity.INFO,
                                message=f"Row {row_index + 1}: Invalid {date_field.replace('course_', '')} date format. Will attempt automatic conversion",
                                is_valid=True
                            )
                        )
        
        # Check if course available from is missing - but don't fail
        if 'course_available_from' in row.index and pd.isna(row['course_available_from']):
            self.validation_results.append(
                ValidationResult(
                    severity=Severity.INFO,
                    message=f"Row {row_index + 1}: Course available from date is recommended",
                    is_valid=True
                )
            )
        
        # Validate data types of other fields
        # Category name
        if 'category_name' in row.index and pd.notna(row['category_name']):
            if not isinstance(row['category_name'], str):
                # Just convert and log
                self.validation_results.append(
                    ValidationResult(
                        severity=Severity.INFO,
                        message=f"Row {row_index + 1}: Category name converted from {type(row['category_name']).__name__} to string",
                        is_valid=True
                    )
                )
        
        # Keywords
        if 'course_keywords' in row.index and pd.notna(row['course_keywords']):
            if not isinstance(row['course_keywords'], (str, list)):
                # Just convert and log
                self.validation_results.append(
                    ValidationResult(
                        severity=Severity.INFO,
                        message=f"Row {row_index + 1}: Keywords converted from {type(row['course_keywords']).__name__} to string",
                        is_valid=True
                    )
                )
    
    def get_validation_results(self) -> List[ValidationResult]:
        """Get all validation results."""
        return self.validation_results 