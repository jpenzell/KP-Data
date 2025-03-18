"""
Optimization utilities for the LMS Analyzer application.
These functions improve performance and data quality.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import time
import re

@st.cache_data(ttl=3600)
def remove_unnamed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary unnamed columns that are likely index columns from the original files.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with unnamed columns removed
    """
    unnamed_cols = [col for col in df.columns if str(col).lower().startswith('unnamed:')]
    if unnamed_cols:
        print(f"Removing {len(unnamed_cols)} unnecessary unnamed columns")
        return df.drop(columns=unnamed_cols)
    return df

@st.cache_data(ttl=3600)
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced column name cleaning to ensure consistency.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with cleaned column names
    """
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # First pass: Convert to lowercase and strip whitespace
    df_cleaned.columns = df_cleaned.columns.str.lower().str.strip()
    
    # Common column name patterns that need standardization
    common_patterns = {
        # Exact matches (in lowercase)
        'course title': 'course_title',
        'course no': 'course_no',
        'course description': 'course_description',
        'course abstract': 'course_abstract',
        'course version': 'course_version',
        'course created by': 'course_created_by',
        'course available from': 'course_available_from',
        'course discontinued from': 'course_discontinued_from',
        'course keywords': 'course_keywords',
        'course categories': 'course_categories',
        'category name': 'category_name',
        'sponsoring dept.': 'sponsoring_dept',
        'course intended audience type names': 'course_intended_audience_types',
        'similar to': 'similar_to',
        'similarity score': 'similarity_score',
        'kp learn learning insights report': 'kp_learn_insights_report',
        'region entity': 'region_entity',
        'offering template no': 'offering_template_no',
        'person type': 'person_type',
        'avg hrs spent per completion': 'avg_hours_per_completion',
        'no of learners': 'learner_count',
        'has direct reports': 'has_direct_reports',
        'total 2024 activity': 'total_2024_activity'
    }
    
    # Apply common patterns mapping
    for original, standardized in common_patterns.items():
        if original in df_cleaned.columns:
            df_cleaned = df_cleaned.rename(columns={original: standardized})
    
    # Handle unnamed columns - either drop or rename them meaningfully
    unnamed_cols = [col for col in df_cleaned.columns if 'unnamed:' in str(col).lower()]
    if unnamed_cols:
        print(f"Found {len(unnamed_cols)} unnamed columns - standardizing or removing")
        
        # Check if any unnamed columns have meaningful data
        for col in unnamed_cols:
            non_null_values = df_cleaned[col].count()
            if non_null_values < len(df_cleaned) * 0.05:  # Less than 5% of rows have data
                # Safe to drop columns with very little data
                df_cleaned = df_cleaned.drop(columns=[col])
                print(f"  Dropped nearly empty column: {col}")
            else:
                # Rename to a more useful format
                new_name = f"unknown_col_{unnamed_cols.index(col)}"
                df_cleaned = df_cleaned.rename(columns={col: new_name})
                print(f"  Renamed column with data: {col} -> {new_name}")
    
    # Second pass: Replace remaining spaces and special characters with underscores
    new_columns = []
    for col in df_cleaned.columns:
        # Skip columns that were already standardized
        if col in common_patterns.values():
            new_columns.append(col)
        else:
            # Apply regex for remaining columns
            new_col = re.sub(r'[ \.\/\\\(\)\[\]\{\}\-\+\*\&\^\%\$\#\@\!\?\>\<\,\:\;]', '_', col)
            # Collapse multiple underscores
            new_col = re.sub(r'_+', '_', new_col)
            # Remove underscores at the start/end
            new_col = re.sub(r'^_|_$', '', new_col)
            new_columns.append(new_col)
    
    df_cleaned.columns = new_columns
    
    # Print any column names that might still need attention
    remaining_problematic = [col for col in df_cleaned.columns if ' ' in col or any(c in col for c in r'./\()[]{}-+*&^%$#@!?><,:;')]
    if remaining_problematic:
        print(f"Warning: {len(remaining_problematic)} column names still need attention: {remaining_problematic}")
    
    return df_cleaned

@st.cache_data(ttl=3600)
def enhanced_date_repair(df: pd.DataFrame) -> pd.DataFrame:
    """
    More aggressive date field repair using multiple sources of information
    and advanced validation.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with repaired date fields
    """
    print("Applying enhanced date repair...")
    df_fixed = df.copy()
    
    # List of date columns to check and repair
    date_columns = [
        'course_version', 
        'course_available_from', 
        'course_discontinued_from'
    ]
    
    # Track metrics for reporting
    metrics = {col: {'before': 0, 'after': 0} for col in date_columns}
    
    # Define valid date range (discard clearly invalid dates)
    min_valid_date = pd.Timestamp('1990-01-01')  # Earliest reasonable date
    max_valid_date = pd.Timestamp.now() + pd.DateOffset(years=10)  # Allow future dates for discontinued_from
    
    # First pass: Convert all date fields to proper datetime and validate ranges
    for col in date_columns:
        if col in df_fixed.columns:
            valid_flag = f"{col}_is_valid"
            
            # First convert to datetime with lenient parsing
            df_fixed[col] = pd.to_datetime(df_fixed[col], errors='coerce')
            
            # Add validation flag 
            if valid_flag not in df_fixed.columns:
                # Initial validation: not null and within reasonable date range
                df_fixed[valid_flag] = (
                    pd.notna(df_fixed[col]) & 
                    (df_fixed[col] >= min_valid_date) & 
                    (df_fixed[col] <= max_valid_date)
                )
            
            # Validate date consistency for available vs discontinued
            if col == 'course_discontinued_from' and 'course_available_from' in df_fixed.columns:
                availability_valid = 'course_available_from_is_valid' in df_fixed.columns
                if availability_valid:
                    # Mark discontinued dates as invalid if they're before available dates
                    invalid_sequence = (
                        df_fixed['course_available_from_is_valid'] & 
                        df_fixed[valid_flag] &
                        (df_fixed['course_discontinued_from'] < df_fixed['course_available_from'])
                    )
                    if invalid_sequence.any():
                        print(f"Found {invalid_sequence.sum()} courses with discontinued date before available date")
                        df_fixed.loc[invalid_sequence, valid_flag] = False
            
            # Track initial state
            metrics[col]['before'] = df_fixed[valid_flag].sum()
    
    # Prioritize using valid dates across fields
    for col in date_columns:
        if col in df_fixed.columns:
            valid_flag = f"{col}_is_valid"
            
            # 1. Try to infer dates from other date columns
            for other_col in date_columns:
                if other_col != col and other_col in df_fixed.columns:
                    other_valid_flag = f"{other_col}_is_valid"
                    
                    if other_valid_flag in df_fixed.columns:
                        # Use other valid date to fill missing dates
                        mask = (~df_fixed[valid_flag]) & df_fixed[other_valid_flag]
                        df_fixed.loc[mask, col] = df_fixed.loc[mask, other_col]
                        df_fixed.loc[mask, valid_flag] = True
    
    # 2. For courses with similar IDs, try to use dates from related courses
    if 'course_no' in df_fixed.columns:
        # Extract course number prefix (first 3-4 characters) - do this BEFORE creating filtered dataframes
        if 'course_no' in df_fixed.columns:
            df_fixed['course_prefix'] = df_fixed['course_no'].str[:4]
        
        for col in date_columns:
            if col in df_fixed.columns:
                valid_flag = f"{col}_is_valid"
                
                # Get courses with valid dates to use as reference - do this AFTER creating course_prefix
                valid_courses = df_fixed[df_fixed[valid_flag]]
                invalid_courses = df_fixed[~df_fixed[valid_flag]]
                
                if not valid_courses.empty and not invalid_courses.empty:
                    print(f"Trying to infer {col} from similar courses...")
                    
                    try:
                        # Group by prefix and get median date
                        prefix_dates = valid_courses.groupby('course_prefix')[col].median()
                        
                        # Fill missing dates based on prefix
                        for prefix, median_date in prefix_dates.items():
                            if pd.notna(median_date):
                                mask = (~df_fixed[valid_flag]) & (df_fixed['course_prefix'] == prefix)
                                df_fixed.loc[mask, col] = median_date
                                df_fixed.loc[mask, valid_flag] = True
                    except Exception as e:
                        print(f"Error inferring dates by course prefix: {str(e)}")
        
        # Clean up temporary column
        if 'course_prefix' in df_fixed.columns:
            df_fixed = df_fixed.drop(columns=['course_prefix'])
    
    # 3. Special handling for specific date columns
    # For course_available_from, most courses should have a date
    if 'course_available_from' in df_fixed.columns:
        # For remaining missing dates, use a reasonable default
        missing_mask = ~df_fixed['course_available_from_is_valid']
        if missing_mask.any():
            # Generate more reasonable default dates based on medians or source-specific patterns
            if 'data_source' in df_fixed.columns:
                # Try to use source-specific median dates first
                try:
                    source_medians = df_fixed[df_fixed['course_available_from_is_valid']].groupby('data_source')['course_available_from'].median()
                    for source, median_date in source_medians.items():
                        if pd.notna(median_date):
                            source_mask = missing_mask & (df_fixed['data_source'] == source)
                            if source_mask.any():
                                df_fixed.loc[source_mask, 'course_available_from'] = median_date
                                df_fixed.loc[source_mask, 'course_available_from_is_valid'] = False  # Mark as artificial
                except Exception as e:
                    print(f"Error generating source-specific dates: {str(e)}")
                
            # Use start of previous year as fallback
            still_missing = ~df_fixed['course_available_from_is_valid']
            if still_missing.any():
                current_year = datetime.now().year
                default_date = pd.Timestamp(f"{current_year-1}-01-01")
                df_fixed.loc[still_missing, 'course_available_from'] = default_date
                df_fixed.loc[still_missing, 'course_available_from_is_valid'] = False  # Mark as artificial
    
    # For course_discontinued_from, use far future date for active courses
    if 'course_discontinued_from' in df_fixed.columns:
        # Courses without discontinued date are likely still active
        missing_mask = ~df_fixed['course_discontinued_from_is_valid']
        if missing_mask.any():
            future_date = pd.Timestamp.now() + pd.DateOffset(years=10)
            df_fixed.loc[missing_mask, 'course_discontinued_from'] = future_date
            df_fixed.loc[missing_mask, 'course_discontinued_from_is_valid'] = True
    
    # 4. Final validation pass - check for impossible date combinations
    date_validation_metrics = {'impossible_combinations': 0, 'fixed_combinations': 0}
    
    if all(col in df_fixed.columns for col in ['course_available_from', 'course_discontinued_from']):
        # Courses discontinued before available
        impossible = (
            df_fixed['course_available_from_is_valid'] & 
            df_fixed['course_discontinued_from_is_valid'] &
            (df_fixed['course_discontinued_from'] < df_fixed['course_available_from'])
        )
        
        if impossible.any():
            date_validation_metrics['impossible_combinations'] += impossible.sum()
            print(f"Fixing {impossible.sum()} impossible date combinations (discontinued before available)")
            
            # Fix by making discontinued date 1 year after available
            df_fixed.loc[impossible, 'course_discontinued_from'] = (
                df_fixed.loc[impossible, 'course_available_from'] + pd.DateOffset(years=1)
            )
            date_validation_metrics['fixed_combinations'] += impossible.sum()
    
    # Track final state
    for col in date_columns:
        if col in df_fixed.columns:
            valid_flag = f"{col}_is_valid"
            if valid_flag in df_fixed.columns:
                metrics[col]['after'] = df_fixed[valid_flag].sum()
                fixed_count = metrics[col]['after'] - metrics[col]['before']
                total_count = len(df_fixed)
                print(f"Fixed {fixed_count} {col} dates ({fixed_count/total_count*100:.1f}% of total)")
    
    if date_validation_metrics['impossible_combinations'] > 0:
        print(f"Resolved {date_validation_metrics['fixed_combinations']}/{date_validation_metrics['impossible_combinations']} impossible date combinations")
    
    return df_fixed

@st.cache_data(ttl=3600)
def process_in_chunks(df: pd.DataFrame, 
                     func, 
                     chunk_size: int = 10000, 
                     **kwargs) -> pd.DataFrame:
    """
    Process a large DataFrame in chunks to reduce memory usage.
    
    Args:
        df: DataFrame to process
        func: Function to apply to each chunk
        chunk_size: Number of rows per chunk
        kwargs: Additional arguments to pass to func
        
    Returns:
        Processed DataFrame
    """
    if len(df) <= chunk_size:
        return func(df, **kwargs)
    
    print(f"Processing {len(df)} rows in chunks of {chunk_size}...")
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        print(f"Processing chunk {i+1}/{len(chunks)}...")
        processed_chunk = func(chunk, **kwargs)
        processed_chunks.append(processed_chunk)
    
    print(f"Combining {len(processed_chunks)} processed chunks...")
    result = pd.concat(processed_chunks, ignore_index=True)
    return result

@st.cache_data(ttl=3600)
def optimize_memory_usage_enhanced(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced version of optimize_memory_usage that also removes unnecessary columns
    and applies more aggressive optimization.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Memory-optimized DataFrame
    """
    print("Applying enhanced memory optimization...")
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage before optimization: {start_mem:.2f} MB")
    
    # Remove unnamed columns
    df_optimized = remove_unnamed_columns(df)
    
    # Apply standard memory optimizations
    for col in df_optimized.columns:
        # Reduce precision of float columns
        if df_optimized[col].dtype == 'float64':
            df_optimized[col] = df_optimized[col].astype('float32')
        
        # Reduce precision of integer columns
        elif df_optimized[col].dtype == 'int64':
            df_optimized[col] = df_optimized[col].astype('int32')
        
        # Convert string columns with low cardinality to category
        elif df_optimized[col].dtype == 'object':
            num_unique = df_optimized[col].nunique()
            if num_unique < len(df_optimized) * 0.5:  # If fewer than 50% unique values
                df_optimized[col] = df_optimized[col].astype('category')
    
    # Optimize boolean columns 
    bool_columns = [col for col in df_optimized.columns if col.startswith('is_') 
                   or col.endswith('_is_valid') or df_optimized[col].dtype == 'bool']
    
    for col in bool_columns:
        if col in df_optimized.columns:
            df_optimized[col] = df_optimized[col].astype('bool')
    
    end_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2
    reduction = (start_mem - end_mem) / start_mem * 100
    print(f"Memory usage after optimization: {end_mem:.2f} MB ({reduction:.1f}% reduction)")
    
    return df_optimized

def measure_execution_time(func):
    """
    Decorator to measure execution time of functions.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time for {func.__name__}: {execution_time:.2f} seconds")
        return result
    return wrapper

@st.cache_data(ttl=3600)
def fix_row_count_discrepancy(df_merged: pd.DataFrame, df_analyzer: pd.DataFrame) -> pd.DataFrame:
    """
    Investigate and fix discrepancy between merged data (47,360 rows) and 
    analyzer data (11,928 rows).
    
    Args:
        df_merged: DataFrame from merge_and_standardize_data
        df_analyzer: DataFrame from LMSContentAnalyzer
        
    Returns:
        DataFrame with consistent row count
    """
    print(f"Investigating row count discrepancy: {len(df_merged)} vs {len(df_analyzer)}")
    
    # If analyzer dataframe is a subset, determine what filter is being applied
    if len(df_analyzer) < len(df_merged):
        # Check for null values in key columns that might be causing filtering
        key_columns = ['course_title', 'course_no', 'course_id']
        for col in key_columns:
            if col in df_merged.columns:
                null_count = df_merged[col].isna().sum()
                if null_count > 0:
                    print(f"Found {null_count} null values in {col}")
        
        # Return the merged dataframe to ensure we use all data
        return df_merged
    
    return df_analyzer 