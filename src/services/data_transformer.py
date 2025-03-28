"""Data transformation service for the LMS analyzer."""

import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from datetime import datetime

from src.config.column_mappings import COLUMN_TYPES

class DataTransformer:
    """Service for transforming LMS data."""
    
    def transform_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the dataframe to the required format."""
        # Convert column types
        df = self._convert_column_types(df)
        
        # Clean and standardize data
        df = self._clean_data(df)
        
        # Add derived columns
        df = self._add_derived_columns(df)
        
        return df
    
    def _convert_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to their required types."""
        for col, dtype in COLUMN_TYPES.items():
            if col in df.columns:
                try:
                    if dtype == "datetime64[ns]":
                        df[col] = pd.to_datetime(df[col])
                    elif dtype == "Int64":
                        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                    else:
                        df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {str(e)}")
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        # Remove leading/trailing whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        # Standardize course types
        if 'course_type' in df.columns:
            df['course_type'] = df['course_type'].str.lower()
        
        # Standardize region/department names
        if 'region_entity' in df.columns:
            df['region_entity'] = df['region_entity'].str.title()
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for analysis."""
        # Add course age
        if 'course_available_from' in df.columns:
            df['course_age_days'] = (datetime.now() - df['course_available_from']).dt.days
        
        # Add course status
        df['course_status'] = 'active'
        if 'course_discontinued_from' in df.columns:
            mask = (df['course_discontinued_from'].notna()) & (df['course_discontinued_from'] <= datetime.now())
            df.loc[mask, 'course_status'] = 'discontinued'
        
        # Add activity level
        if 'total_2024_activity' in df.columns:
            df['activity_level'] = pd.cut(
                df['total_2024_activity'],
                bins=[-float('inf'), 0, 10, 50, float('inf')],
                labels=['none', 'low', 'medium', 'high']
            )
        
        return df 