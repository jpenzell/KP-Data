import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="LMS Content Analysis", layout="wide")

import pandas as pd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from pathlib import Path
import numpy as np
import sys
import os

# Add src directory to path to allow importing from src.utils
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from lms_content_analyzer import LMSContentAnalyzer
# Import the new optimization functions
try:
    from src.utils.optimizations import (
        remove_unnamed_columns,
        clean_column_names,
        enhanced_date_repair,
        process_in_chunks,
        optimize_memory_usage_enhanced,
        measure_execution_time,
        fix_row_count_discrepancy
    )
    print("Successfully imported optimization utilities")
except ImportError as e:
    print(f"Warning: Could not import optimization utilities: {e}")
    print("Falling back to basic optimization functions")
    # Define basic fallback functions if imports fail
    @st.cache_data(ttl=3600)
    def optimize_memory_usage_enhanced(df):
        return optimize_memory_usage(df)
        
    @st.cache_data(ttl=3600)
    def enhanced_date_repair(df):
        return repair_date_fields(df)
        
    def measure_execution_time(func):
        return func

# Performance optimization functions
@st.cache_data(ttl=3600)
def optimize_memory_usage(df):
    """
    Optimize memory usage by converting data types to more efficient ones.
    """
    df_optimized = df.copy()
    for col in df_optimized.columns:
        if df_optimized[col].dtype == 'float64':
            df_optimized[col] = df_optimized[col].astype('float32')
        elif df_optimized[col].dtype == 'int64':
            df_optimized[col] = df_optimized[col].astype('int32')
        elif df_optimized[col].dtype == 'object':
            if df_optimized[col].nunique() / len(df_optimized[col]) < 0.5:  # Low cardinality
                df_optimized[col] = df_optimized[col].astype('category')
    return df_optimized

@st.cache_data(ttl=3600)
def repair_date_fields(df):
    """
    Repair and validate date fields where possible.
    """
    df_fixed = df.copy()
    
    # For course_version, use the earliest available date if missing
    if 'course_available_from' in df_fixed.columns and 'course_version' in df_fixed.columns:
        mask = df_fixed['course_version'].isna() & df_fixed['course_available_from'].notna()
        df_fixed.loc[mask, 'course_version'] = df_fixed.loc[mask, 'course_available_from']
    
    # Handle missing discontinued dates by setting a future date for active courses
    if 'course_discontinued_from' in df_fixed.columns:
        mask = df_fixed['course_discontinued_from'].isna()
        # Set to a far future date (e.g., 10 years from now) for still-active courses
        df_fixed.loc[mask, 'course_discontinued_from'] = pd.Timestamp.now() + pd.DateOffset(years=10)
        
    return df_fixed

# Add caching to data loading and processing functions
@st.cache_data(ttl=3600)
@measure_execution_time  # Add execution time measurement
def merge_and_standardize_data(dataframes):
    if not dataframes:
        return pd.DataFrame()
    
    print(f"\nProcessing {len(dataframes)} data sources...")
    
    standardized_dfs = []
    total_rows = 0
    
    for idx, df in enumerate(dataframes):
        print(f"\nProcessing source {idx + 1}:")
        print(f"Initial rows: {len(df)}")
        
        df_copy = df.copy()
        
        # Convert all column names to lowercase and strip whitespace
        df_copy.columns = df_copy.columns.str.lower().str.strip()
        
        # Handle duplicate columns by adding suffixes
        duplicate_cols = df_copy.columns[df_copy.columns.duplicated()].tolist()
        if duplicate_cols:
            print(f"Found duplicate columns in source {idx + 1}: {duplicate_cols}")
            # Create a mapping of duplicated columns to their new names with suffixes
            col_mapping = {}
            for col in df_copy.columns:
                if col in duplicate_cols:
                    suffix = 1
                    new_col = f"{col}_{suffix}"
                    while new_col in col_mapping.values():
                        suffix += 1
                        new_col = f"{col}_{suffix}"
                    col_mapping[col] = new_col
                else:
                    col_mapping[col] = col
            # Rename columns using the mapping
            df_copy.columns = [col_mapping.get(col, col) for col in df_copy.columns]
        
        # Add source tracking
        source_name = f"Source_{idx + 1}"
        df_copy['data_source'] = source_name
        df_copy['cross_reference_count'] = 1
        
        # Generate a unique identifier if course_no doesn't exist
        if 'course_no' not in df_copy.columns:
            print(f"Generating course_no for source {idx + 1}")
            if 'offering_template_no' in df_copy.columns:
                df_copy['course_no'] = df_copy['offering_template_no'].astype(str)
            elif 'course_title' in df_copy.columns:
                df_copy['course_no'] = df_copy['course_title'].str.lower().str.strip()
            else:
                df_copy['course_no'] = f"source_{idx + 1}_" + df_copy.index.astype(str)
        
        # Ensure course_no is string and standardized
        df_copy['course_no'] = df_copy['course_no'].astype(str).str.lower().str.strip()
        
        # Track the original number of rows
        initial_rows = len(df_copy)
        total_rows += initial_rows
        
        standardized_dfs.append(df_copy)
        print(f"Processed rows from {source_name}: {initial_rows}")
    
    if standardized_dfs:
        print(f"\nMerging {len(standardized_dfs)} dataframes...")
        print(f"Total input rows: {total_rows}")
        
        # Concatenate all dataframes vertically
        all_df = pd.concat(standardized_dfs, ignore_index=True)
        print(f"Combined rows before deduplication: {len(all_df)}")
        
        # For large datasets, use chunked processing
        if len(all_df) > 10000:
            print("Large dataset detected, using chunked processing for deduplication key creation...")
            chunk_size = 10000
            chunks = [all_df.iloc[i:i+chunk_size] for i in range(0, len(all_df), chunk_size)]
            for i, chunk in enumerate(chunks):
                print(f"Processing chunk {i+1}/{len(chunks)}...")
                # Create a robust deduplication key using enhanced approaches
                chunks[i] = create_deduplication_key(chunk)
            all_df = pd.concat(chunks, ignore_index=True)
        else:
            # Create a robust deduplication key using enhanced approaches
            all_df = create_deduplication_key(all_df)
        
        # Apply fuzzy matching to further identify duplicates
        print("Applying fuzzy matching to identify additional duplicates...")
        try:
            # Only apply if dataset is a reasonable size (less than 20k rows)
            if len(all_df) <= 20000:
                all_df = fuzzy_match_courses(all_df)
            else:
                print("Skipping fuzzy matching due to large dataset size")
        except Exception as e:
            print(f"Warning: Fuzzy matching failed: {str(e)}")
        
        # Remove duplicate column names before groupby
        print("Removing duplicate columns...")
        # Get list of duplicate columns
        duplicates = all_df.columns[all_df.columns.duplicated()].tolist()
        if duplicates:
            print(f"Found duplicate columns: {duplicates}")
            # Keep only the first occurrence of each duplicate column
            all_df = all_df.loc[:, ~all_df.columns.duplicated()]
        
        # Group by dedup_key and aggregate
        print("Performing enhanced deduplication and merging...")
        agg_dict = {
            'cross_reference_count': 'sum',
            'data_source': lambda x: ' | '.join(sorted(set(x)))
        }
        
        # Add aggregation rules for other columns with improved strategy
        for col in all_df.columns:
            if col not in ['dedup_key', 'cross_reference_count', 'data_source']:
                if col == 'course_title':
                    # For titles, prefer the longest non-null value (usually most informative)
                    agg_dict[col] = lambda x: max(x.dropna().astype(str), key=len) if not x.dropna().empty else None
                elif col == 'course_description' or col == 'course_abstract':
                    # For descriptions, also prefer the longest
                    agg_dict[col] = lambda x: max(x.dropna().astype(str), key=len) if not x.dropna().empty else None
                elif str(col).startswith('is_'):
                    agg_dict[col] = 'max'  # Use OR operation for boolean flags
                elif pd.api.types.is_numeric_dtype(all_df[col]):
                    # For numeric columns, use sum for counts, mean for rates/measurements
                    if any(count_word in col for count_word in ['count', 'activity', 'completions', 'learner']):
                        agg_dict[col] = 'sum'
                    else:
                        agg_dict[col] = 'mean'
                elif 'date' in col or col in ['course_version', 'course_available_from', 'course_discontinued_from']:
                    # For dates, prefer the latest non-null value
                    agg_dict[col] = lambda x: pd.to_datetime(x, errors='coerce').max()
                else:
                    # Default strategy: use first non-null value
                    agg_dict[col] = lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
        
        # Perform the groupby operation with improved aggregation
        merged_df = all_df.groupby('dedup_key', as_index=False).agg(agg_dict)
        initial_row_count = len(all_df)
        final_row_count = len(merged_df)
        dedup_reduction = initial_row_count - final_row_count
        dedup_percent = (dedup_reduction / initial_row_count) * 100 if initial_row_count > 0 else 0
        
        print(f"Enhanced deduplication reduced dataset from {initial_row_count:,} to {final_row_count:,} rows")
        print(f"Removed {dedup_reduction:,} duplicate courses ({dedup_percent:.1f}% reduction)")
        
        # Clean up and standardize the final dataset
        print("\nStandardizing columns...")
        merged_df = standardize_columns(merged_df)
        
        # Add quality metrics
        print("Calculating quality scores...")
        merged_df = calculate_quality_score(merged_df)
        
        # Apply enhanced date field repairs
        print("Applying enhanced date repair...")
        merged_df = enhanced_date_repair(merged_df)
        
        # Apply enhanced memory optimization
        print("Applying enhanced memory optimization...")
        merged_df = optimize_memory_usage_enhanced(merged_df)
        
        # Ensure all column names are properly standardized
        print("Applying enhanced column name standardization...")
        merged_df = clean_column_names(merged_df)
        
        print(f"\nSuccessfully loaded {len(merged_df)} rows of data")
        
        # Print final column list for verification
        print("\nFinal columns after processing:")
        for col in merged_df.columns:
            print(f"- {col}")
        
        return merged_df
    else:
        return pd.DataFrame()

def create_deduplication_key(df):
    """
    Create a sophisticated deduplication key using multiple fields and fuzzy matching.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with enhanced deduplication key added
    """
    print("Creating enhanced deduplication key...")
    
    # Create a copy to avoid modification warnings
    df = df.copy()
    
    # Step 1: Clean and standardize fields that will be used for deduplication
    # Standardize course_no - remove source prefixes to enable cross-source matching
    if 'course_no' in df.columns:
        # Remove source prefixes like "source_1_" to enable matching across sources
        df['clean_course_no'] = df['course_no'].astype(str).str.lower().str.strip()
        df['clean_course_no'] = df['clean_course_no'].str.replace(r'^source_\d+_', '', regex=True)
        
        # Further clean by removing common prefixes and non-alphanumeric characters
        df['clean_course_no'] = df['clean_course_no'].str.replace(r'[^a-z0-9]', '_', regex=True)
    else:
        df['clean_course_no'] = ""
    
    # Standardize course title
    if 'course_title' in df.columns:
        df['clean_title'] = df['course_title'].astype(str).str.lower().str.strip()
        # Remove punctuation and extra spaces
        df['clean_title'] = df['clean_title'].str.replace(r'[^\w\s]', ' ', regex=True)
        df['clean_title'] = df['clean_title'].str.replace(r'\s+', ' ', regex=True)
    else:
        df['clean_title'] = ""
    
    # Step 2: Create a primary deduplication key
    # Priority order: clean_course_no > offering_template_no > normalized title
    
    # Initialize with clean course number if it exists and isn't generic
    df['dedup_key'] = df['clean_course_no']
    
    # For rows where course_no is missing, generic, or just a source prefix
    generic_values = ['', 'nan', 'none', 'null', 'unknown', 'test', 'temp']
    mask = (df['dedup_key'].isna() | 
            df['dedup_key'].isin(generic_values) | 
            df['dedup_key'] == "")
    
    # Try offering_template_no next
    if 'offering_template_no' in df.columns:
        template_values = df.loc[mask, 'offering_template_no'].astype(str).str.lower().str.strip()
        df.loc[mask, 'dedup_key'] = template_values
    
    # For rows still missing a key, use normalized title
    mask = (df['dedup_key'].isna() | 
            df['dedup_key'].isin(generic_values) | 
            df['dedup_key'] == "")
    
    if 'clean_title' in df.columns:
        df.loc[mask, 'dedup_key'] = df.loc[mask, 'clean_title']
    
    # Step 3: For stronger deduplication, create a composite key for similar titles 
    # when direct ID matching fails
    
    # If we have description data, incorporate it into the key for better matching
    if 'course_description' in df.columns and 'clean_title' in df.columns:
        # Create a short description fingerprint (first 50 chars)
        df['desc_fingerprint'] = df['course_description'].astype(str).str.lower().str.strip()
        df['desc_fingerprint'] = df['desc_fingerprint'].str.replace(r'[^\w\s]', ' ', regex=True)
        df['desc_fingerprint'] = df['desc_fingerprint'].str.replace(r'\s+', ' ', regex=True)
        df['desc_fingerprint'] = df['desc_fingerprint'].str[:50]  # First 50 chars as fingerprint
        
        # For rows with weak primary keys (e.g., just using title), strengthen with description
        mask = ~df['clean_course_no'].astype(bool)  # Rows without a real course number
        df.loc[mask, 'dedup_key'] = (df.loc[mask, 'clean_title'] + 
                                    "_" + 
                                    df.loc[mask, 'desc_fingerprint'])
    
    # Step 4: Add type/category information for even better matching
    category_fields = ['category_name', 'course_type', 'sponsoring_dept']
    for field in category_fields:
        if field in df.columns:
            # Only add to rows with weak keys (using just title or description)
            mask = ~df['clean_course_no'].astype(bool)
            field_values = df.loc[mask, field].astype(str).str.lower().str.strip()
            # Add first 10 chars of category to dedup key
            df.loc[mask, 'dedup_key'] = df.loc[mask, 'dedup_key'] + "_" + field_values.str[:10]
    
    # Step 5: Add duration as a final discriminator for generic courses
    if 'duration_mins' in df.columns:
        # For rows using just title-based matching and having duration
        mask = (~df['clean_course_no'].astype(bool)) & (df['duration_mins'].notna())
        # Group durations into bins (nearest 30 mins) to allow for minor differences
        duration_values = df.loc[mask, 'duration_mins'].astype(float) // 30 * 30
        df.loc[mask, 'dedup_key'] = df.loc[mask, 'dedup_key'] + "_" + duration_values.astype(str)
    
    # Clean up any messy keys
    df['dedup_key'] = df['dedup_key'].str.replace(r'_+', '_', regex=True)  # Replace multiple underscores
    df['dedup_key'] = df['dedup_key'].str.replace(r'^_|_$', '', regex=True)  # Remove leading/trailing underscores
    
    # Ensure every row has a dedup_key
    mask = df['dedup_key'].isna() | df['dedup_key'] == ""
    if mask.any():
        print(f"Creating random keys for {mask.sum()} rows with no usable identification data")
        df.loc[mask, 'dedup_key'] = "unknown_" + df.loc[mask].index.astype(str)
    
    # Clean up temporary columns
    for col in ['clean_course_no', 'clean_title', 'desc_fingerprint']:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    return df

def generate_course_id(row):
    """Generate a unique identifier for a course based on available fields"""
    identifiers = []
    
    # Try different combinations of identifiers
    if 'course_no' in row.index and pd.notna(row['course_no']):
        identifiers.append(str(row['course_no']).lower().strip())
    elif 'offering_template_no' in row.index and pd.notna(row['offering_template_no']):
        identifiers.append(str(row['offering_template_no']).lower().strip())
    
    if 'course_title' in row.index and pd.notna(row['course_title']):
        identifiers.append(str(row['course_title']).lower().strip())
    elif 'title' in row.index and pd.notna(row['title']):
        identifiers.append(str(row['title']).lower().strip())
    
    if 'course_version' in row.index and pd.notna(row['course_version']):
        identifiers.append(str(row['course_version']).lower().strip())
    elif 'version' in row.index and pd.notna(row['version']):
        identifiers.append(str(row['version']).lower().strip())
    
    return '_'.join(identifiers) if identifiers else f"unknown_{pd.util.hash_pandas_object(row).sum()}"

def create_derived_columns(df):
    """Create and transform columns dynamically based on available data"""
    
    # Handle missing but required columns first
    required_columns = {
        'cross_reference_count': 1,
        'data_source': [],
        'quality_score': 0.0,
        'duration_mins': None,
        'learner_count': 0
    }
    
    for col, default_value in required_columns.items():
        if col not in df.columns:
            print(f"Adding missing required column: {col}")
            df[col] = default_value
    
    # Create full_course_id
    if all(col in df.columns for col in ['course_no', 'course_version']):
        print("Creating derived column: full_course_id")
        try:
            df['full_course_id'] = df.apply(
                lambda row: f"{row['course_no']}_{row['course_version']}" if pd.notna(row['course_version']) 
                else str(row['course_no']),
                axis=1
            )
        except Exception as e:
            print(f"Warning: Could not create full_course_id: {str(e)}")
            df['full_course_id'] = df['course_no'].astype(str)
    
    # Create course_duration_hours
    if 'duration_mins' in df.columns:
        print("Creating derived column: course_duration_hours")
        try:
            # Convert to numeric first to handle any string values
            duration_mins = pd.to_numeric(df['duration_mins'], errors='coerce')
            df['course_duration_hours'] = duration_mins.div(60)
        except Exception as e:
            print(f"Warning: Could not create course_duration_hours: {str(e)}")
            df['course_duration_hours'] = None
    
    # Create is_active
    if 'course_discontinued_from' in df.columns:
        print("Creating derived column: is_active")
        try:
            now = pd.Timestamp.now()
            df['is_active'] = df['course_discontinued_from'].apply(
                lambda x: True if pd.isna(x) else pd.to_datetime(x, errors='coerce') > now
            )
        except Exception as e:
            print(f"Warning: Could not create is_active: {str(e)}")
            df['is_active'] = True
    
    return df

def standardize_columns(df):
    """Standardize column names and data formats"""
    # First, convert all column names to lowercase and strip whitespace
    df.columns = df.columns.str.lower().str.strip()
    
    # Remove duplicate columns before any processing
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Standardize column names
    column_mapping = {
        # Original exact matches (lowercase)
        'course_id': 'course_id',
        'course_title': 'course_title',
        'total_2024_activity': 'total_2024_activity',
        'duration_mins': 'duration_mins',
        'avg hrs spent per completion': 'avg_hours_per_completion',
        'data_source': 'data_source',
        'cross_reference_count': 'cross_reference_count',
        'course_duration_hours': 'course_duration_hours',
        'quality_score': 'quality_score',
        'learner_count': 'learner_count',
        'no_of_learners': 'learner_count',
        
        # Alternative names
        'course title': 'course_title',
        'course no': 'course_no',
        'course description': 'course_description',
        'course abstract': 'course_abstract',
        'course version': 'course_version',
        'course created by': 'course_created_by',
        'course available from': 'course_available_from',
        'avail_from': 'course_available_from',
        'course discontinued from': 'course_discontinued_from',
        'disc_from': 'course_discontinued_from',
        'course keywords': 'course_keywords',
        'category name': 'category_name',
        'dur mins': 'duration_mins',
        'duration minutes': 'duration_mins',
        'duration': 'duration_mins',
        'data source': 'data_source',
        'source': 'data_source',
        'students': 'learner_count',
        'region': 'region_entity',
        'region entity': 'region_entity',
        'cross references': 'cross_reference_count',
        'title': 'course_title',
        'description': 'course_description',
        'abstract': 'course_abstract',
        'version': 'course_version',
        'type': 'course_type',
        'average hours per completion': 'avg_hours_per_completion',
        'avg hours per completion': 'avg_hours_per_completion',
        'hours per completion': 'avg_hours_per_completion'
    }
    
    # Apply column mapping
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Remove any remaining duplicate columns after mapping
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    
    # Ensure course_no is always treated as string
    if 'course_no' in df.columns:
        df['course_no'] = df['course_no'].astype(str)
    
    # Convert date columns with better error handling
    date_columns = ['course_version', 'course_available_from', 'course_discontinued_from']
    for col in date_columns:
        if col in df.columns:
            try:
                # Create a new Series for dates and validation flags
                temp_dates = pd.Series(index=df.index, dtype='datetime64[ns]')
                validation_flags = pd.Series(index=df.index, dtype=bool)
                
                # Process each date value
                for idx in df.index:
                    try:
                        val = df.at[idx, col]
                        if pd.isna(val):
                            temp_dates.at[idx] = pd.NaT
                            validation_flags.at[idx] = False
                        else:
                            parsed_date = pd.to_datetime(val, errors='coerce')
                            temp_dates.at[idx] = parsed_date
                            validation_flags.at[idx] = pd.notna(parsed_date)
                    except:
                        temp_dates.at[idx] = pd.NaT
                        validation_flags.at[idx] = False
                
                # Assign processed dates and validation flags
                df[col] = temp_dates
                df[f'{col}_is_valid'] = validation_flags
                
                # Log warning for invalid dates
                invalid_count = (~validation_flags).sum()
                if invalid_count > 0:
                    print(f"Warning: {col} has {invalid_count} invalid or missing dates")
            
            except Exception as e:
                print(f"Warning: Could not process dates for {col}: {str(e)}")
                df[col] = pd.NaT
                df[f'{col}_is_valid'] = False
    
    # Convert numeric columns with better error handling
    numeric_columns = [
        'duration_mins', 
        'learner_count', 
        'cross_reference_count',
        'total_2024_activity',
        'avg_hours_per_completion',
        'course_duration_hours'
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            try:
                # Create a temporary series for numeric conversion
                temp_numeric = pd.Series(index=df.index, dtype='float64')
                
                for idx in df.index:
                    try:
                        val = df.at[idx, col]
                        # Handle complex objects
                        if isinstance(val, (list, set)) and len(val) > 0:
                            val = val[0]
                        elif isinstance(val, dict) and len(val) > 0:
                            val = list(val.values())[0]
                        
                        # Convert to numeric
                        temp_numeric.at[idx] = pd.to_numeric(val, errors='coerce')
                    except:
                        temp_numeric.at[idx] = np.nan
                
                df[col] = temp_numeric
                
                # Fill NaN with 0 for count-based columns
                if col in ['learner_count', 'cross_reference_count', 'total_2024_activity']:
                    df[col] = df[col].fillna(0)
            except Exception as e:
                print(f"Warning: Could not convert {col} to numeric: {str(e)}")
                continue
    
    # Convert boolean columns
    boolean_columns = [
        'is_leadership_development',
        'is_managerial_supervisory',
        'is_mandatory_compliance',
        'is_profession_specific',
        'is_interpersonal_skills',
        'is_it_systems',
        'is_clinical',
        'is_nursing',
        'is_pharmacy',
        'is_safety',
        'has_direct_reports'
    ]
    
    for col in boolean_columns:
        if col in df.columns:
            try:
                df[col] = df[col].fillna(False).astype(bool)
            except Exception as e:
                print(f"Warning: Could not convert {col} to boolean: {str(e)}")
                df[col] = False
    
    # Print column info for debugging
    print("\nFinal column names:")
    for col in df.columns:
        print(f"- {col}")
    
    return df

def calculate_quality_score(df):
    """Calculate quality score based on data completeness and cross-references"""
    try:
        # Initialize quality score column if it doesn't exist
        if 'quality_score' not in df.columns:
            df['quality_score'] = 0.0
        
        # Weight factors for quality score
        weights = {
            'completeness': 0.4,
            'cross_references': 0.3,
            'data_validation': 0.3
        }
        
        # Calculate completeness score
        required_fields = ['course_title', 'course_description', 'course_no', 'category_name']
        available_fields = [field for field in required_fields if field in df.columns]
        
        if available_fields:
            # Calculate completeness for each field
            completeness_scores = pd.DataFrame()
            for field in available_fields:
                completeness_scores[field] = df[field].notna().astype(float)
            
            # Calculate mean completeness score
            df['completeness_score'] = completeness_scores.mean(axis=1)
        else:
            df['completeness_score'] = 0.0
        
        # Calculate cross-reference score
        if 'cross_reference_count' in df.columns:
            cross_ref_counts = pd.to_numeric(df['cross_reference_count'], errors='coerce').fillna(1)
            df['cross_reference_score'] = (cross_ref_counts - 1).div(3).clip(0, 1)
        else:
            df['cross_reference_score'] = 0.0
        
        # Calculate data validation score
        validation_columns = [col for col in df.columns if col.endswith('_is_valid')]
        if validation_columns:
            validation_scores = df[validation_columns].astype(float)
            df['validation_score'] = validation_scores.mean(axis=1)
        else:
            df['validation_score'] = 0.0
        
        # Calculate final quality score
        df['quality_score'] = (
            weights['completeness'] * df['completeness_score'] +
            weights['cross_references'] * df['cross_reference_score'] +
            weights['data_validation'] * df['validation_score']
        )
        
        # Clean up intermediate columns
        columns_to_drop = ['completeness_score', 'cross_reference_score', 'validation_score']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        return df
    
    except Exception as e:
        print(f"Warning: Error calculating quality score: {str(e)}")
        df['quality_score'] = 0.0
        return df

# Constants for mapping and categorization
TRAINING_CATEGORIES = {
    'leadership_development': [
        'leadership', 'management', 'executive', 'strategic', 'decision making',
        'organizational development', 'change management', 'business acumen'
    ],
    'managerial_supervisory': [
        'supervisor', 'manager', 'team lead', 'performance management',
        'delegation', 'coaching', 'mentoring', 'employee development'
    ],
    'mandatory_compliance': [
        'compliance', 'required', 'mandatory', 'regulatory', 'policy',
        'hipaa', 'privacy', 'security awareness', 'ethics', 'code of conduct'
    ],
    'profession_specific': [
        'clinical', 'technical', 'specialized', 'certification',
        'professional development', 'continuing education', 'skill-specific'
    ],
    'interpersonal_skills': [
        'soft skills', 'communication', 'teamwork', 'collaboration',
        'emotional intelligence', 'conflict resolution', 'presentation',
        'customer service', 'cultural competency'
    ],
    'it_systems': [
        'technology', 'software', 'systems', 'digital', 'computer',
        'application', 'platform', 'tool', 'database', 'security'
    ],
    'clinical': [
        'medical', 'healthcare', 'patient care', 'diagnosis', 'treatment',
        'clinical practice', 'patient safety', 'medical procedures',
        'health assessment', 'clinical skills'
    ],
    'nursing': [
        'nurse', 'nursing', 'clinical care', 'patient assessment',
        'medication administration', 'care planning', 'nursing procedures',
        'nursing documentation', 'nursing practice'
    ],
    'pharmacy': [
        'pharmacy', 'medication', 'prescription', 'drug', 'pharmaceutical',
        'dispensing', 'pharmacology', 'pharmacy practice', 'medication safety'
    ],
    'safety': [
        'safety', 'security', 'protection', 'emergency', 'risk management',
        'workplace safety', 'infection control', 'hazard', 'prevention'
    ]
}

DELIVERY_METHODS = {
    'instructor_led_in_person': 'Traditional classroom',
    'instructor_led_virtual': 'Online live sessions',
    'self_paced_elearning': 'Asynchronous learning',
    'microlearning': 'Bite-sized content'
}

def create_wordcloud(text):
    """Create and return a wordcloud visualization"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_timeline(df, date_col):
    """Create a timeline visualization for the specified date column"""
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        fig = px.histogram(df, x=date_col, title=f'Distribution of {date_col}')
        fig.update_layout(bargap=0.2)
        return fig
    return None

def plot_quality_distribution(df):
    """Plot the distribution of quality scores"""
    if 'quality_score' not in df.columns:
        return None
        
    fig = px.histogram(df, x='quality_score', nbins=20,
                      title='Distribution of Content Quality Scores',
                      labels={'quality_score': 'Quality Score', 'count': 'Number of Courses'})
    fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                  annotation_text="Attention Threshold")
    return fig

def categorize_content(df):
    """Categorize content based on text fields and keywords"""
    # Initialize category columns
    for category in TRAINING_CATEGORIES.keys():
        category_col = f'is_{category.lower()}'
        df[category_col] = False
    
    # Text columns to analyze with their weights
    text_columns = {
        'course_title': 1.0,
        'course_description': 0.8,
        'course_abstract': 0.6,
        'course_keywords': 0.9,
        'description': 0.8,  # Alternative name
        'abstract': 0.6,     # Alternative name
        'title': 1.0        # Alternative name
    }
    
    # Process each row
    for idx, row in df.iterrows():
        combined_text = ""
        
        # Combine text fields with weights
        for col, weight in text_columns.items():
            if col in df.columns and isinstance(row.get(col), (str, float, int)) and pd.notna(row.get(col)):
                text = str(row.get(col, '')).lower()
                combined_text += ' ' + text  # Add space to separate fields
        
        # Check for category matches
        for category, keywords in TRAINING_CATEGORIES.items():
            category_col = f'is_{category.lower()}'
            
            # Check category name
            if category.lower() in combined_text:
                df.at[idx, category_col] = True
                continue
            
            # Check keywords
            if keywords:
                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        df.at[idx, category_col] = True
                        break
    
    return df

def analyze_training_focus(df):
    """Analyze training focus distribution and overlaps"""
    focus_stats = {}
    for category in TRAINING_CATEGORIES.keys():
        category_col = f'is_{category.lower()}'
        if category_col in df.columns:
            focus_stats[category] = {
                'count': df[category_col].sum(),
                'percentage': (df[category_col].sum() / len(df) * 100)
            }
        else:
            focus_stats[category] = {
                'count': 0,
                'percentage': 0
            }
    return focus_stats

def analyze_delivery_methods(df):
    """Analyze training delivery methods"""
    if 'delivery_method' not in df.columns:
        return None
    
    delivery_stats = df['delivery_method'].value_counts().to_dict()
    return {
        'distribution': delivery_stats,
        'total_courses': len(df),
        'avg_duration': df['duration_mins'].mean() if 'duration_mins' in df.columns else None
    }

def analyze_metadata_completeness(df):
    """Analyze metadata completeness and quality"""
    metadata_fields = {
        'basic': ['course_title', 'course_no', 'course_description'],
        'temporal': ['course_available_from', 'course_discontinued_from', 'course_version'],
        'classification': ['course_type', 'category_name', 'person_type'],
        'usage': ['learner_count', 'activity_count', 'duration_mins'],
        'organizational': ['region_entity', 'course_created_by'],
        'enrichment': ['course_keywords', 'course_abstract']
    }
    
    completeness_stats = {}
    for category, fields in metadata_fields.items():
        available_fields = [f for f in fields if f in df.columns]
        if available_fields:
            completeness = df[available_fields].notna().mean(axis=1)
            completeness_stats[category] = {
                'avg_completeness': completeness.mean(),
                'fields_present': len(available_fields),
                'total_fields': len(fields),
                'missing_fields': set(fields) - set(available_fields)
            }
    
    return completeness_stats

def analyze_cross_references(df):
    """Analyze cross-references between data sources"""
    stats = {
        'total_courses': len(df),
        'multi_source_courses': 0,
        'consistency_metrics': {}
    }
    
    # Count multi-source courses if the data is available
    if 'cross_reference_count' in df.columns:
        stats['multi_source_courses'] = len(df[df['cross_reference_count'] > 1])
    
    # Check field consistency across sources if we have the necessary columns
    if 'course_no' in df.columns and 'data_source' in df.columns:
        fields_to_check = ['course_title', 'course_description', 'course_type', 'category_name']
        available_fields = [f for f in fields_to_check if f in df.columns]
        
        for field in available_fields:
            consistency = df.groupby('course_no')[field].nunique()
            stats['consistency_metrics'][field] = {
                'consistent': len(consistency[consistency == 1]),
                'inconsistent': len(consistency[consistency > 1])
            }
    
    return stats

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data"""
    temporal_analysis = {}
    
    if 'course_available_from' in df.columns:
        valid_dates = df[df['course_available_from_is_valid']]
        if not valid_dates.empty:
            temporal_analysis['creation_patterns'] = {
                'courses_per_year': valid_dates.groupby(
                    valid_dates['course_available_from'].dt.year
                ).size().to_dict(),
                'courses_per_month': valid_dates.groupby(
                    valid_dates['course_available_from'].dt.month
                ).size().to_dict()
            }
    
    if 'course_discontinued_from' in df.columns:
        active_courses = df[
            (df['course_discontinued_from'].isna()) |
            (df['course_discontinued_from'] > pd.Timestamp.now())
        ]
        temporal_analysis['lifecycle'] = {
            'active_courses': len(active_courses),
            'discontinued_courses': len(df) - len(active_courses)
        }
    
    return temporal_analysis

def analyze_content_relationships(df):
    """Analyze relationships between different content attributes"""
    relationships = {}
    
    # Analyze category-audience relationships
    if all(col in df.columns for col in ['person_type', 'category_name']):
        relationships['category_audience'] = pd.crosstab(
            df['category_name'],
            df['person_type']
        ).to_dict()
    
    # Analyze duration patterns
    if 'duration_mins' in df.columns:
        duration_stats = df.groupby('course_type')['duration_mins'].agg([
            'mean', 'median', 'std'
        ]).to_dict()
        relationships['duration_patterns'] = duration_stats
    
    # Analyze regional patterns
    if 'region_entity' in df.columns:
        for category in TRAINING_CATEGORIES.keys():
            if f'is_{category}' in df.columns:
                regional_dist = df[df[f'is_{category}']]['region_entity'].value_counts()
                relationships[f'regional_{category}'] = regional_dist.to_dict()
    
    return relationships

def analyze_trends(df):
    """Analyze trends in data quality and content metrics over time"""
    trends = {}
    
    # Quality score trends
    if 'quality_score' in df.columns and 'course_available_from' in df.columns:
        # Convert course_available_from to datetime if it isn't already
        df['course_available_from'] = pd.to_datetime(df['course_available_from'], errors='coerce')
        df['year'] = df['course_available_from'].dt.year
        quality_trends = df.groupby('year')['quality_score'].agg(['mean', 'count']).reset_index()
        trends['quality_over_time'] = quality_trends.to_dict('records')
    
    # Category evolution
    for category in TRAINING_CATEGORIES.keys():
        cat_col = f'is_{category.lower()}'
        if cat_col in df.columns and 'course_available_from' in df.columns:
            # Ensure we have a datetime column and create year
            df['course_available_from'] = pd.to_datetime(df['course_available_from'], errors='coerce')
            df['year'] = df['course_available_from'].dt.year
            
            # Group by year and count occurrences where the category is True
            cat_trend = df[df[cat_col]].groupby('year').size().reset_index(name='count')
            
            if not cat_trend.empty:
                trends[f'{category}_evolution'] = cat_trend.to_dict('records')
    
    return trends

def predict_course_utilization(df):
    """Predict course utilization based on historical patterns"""
    predictions = {}
    
    if all(col in df.columns for col in ['learner_count', 'quality_score', 'cross_reference_count']):
        # Simple prediction based on quality and cross-references
        df['predicted_engagement'] = (
            df['quality_score'] * 0.6 +
            df['cross_reference_count'].clip(0, 3) / 3 * 0.4
        ) * 100
        
        predictions['engagement_scores'] = df[['course_id', 'course_title', 'predicted_engagement']].to_dict('records')
        
        # Identify potential high-impact courses
        high_potential = df[
            (df['learner_count'] < df['learner_count'].median()) &
            (df['predicted_engagement'] > 70)
        ]
        predictions['high_potential_courses'] = high_potential[
            ['course_id', 'course_title', 'predicted_engagement']
        ].to_dict('records')
    
    return predictions

def analyze_content_gaps(df):
    """Analyze content gaps based on industry standards and current coverage"""
    gaps = {
        'critical_gaps': [],
        'opportunity_areas': [],
        'recommendations': []
    }
    
    # Industry standard ratios (example values)
    standard_ratios = {
        'mandatory_compliance': 0.15,
        'profession_specific': 0.30,
        'leadership_development': 0.20,
        'interpersonal_skills': 0.15,
        'technical_skills': 0.20
    }
    
    # Calculate current ratios
    total_courses = len(df)
    current_ratios = {}
    for category in TRAINING_CATEGORIES.keys():
        cat_col = f'is_{category.lower()}'
        if cat_col in df.columns:
            current_ratios[category] = df[cat_col].sum() / total_courses
    
    # Compare with standards
    for category, standard in standard_ratios.items():
        if category in current_ratios:
            difference = standard - current_ratios[category]
            if difference > 0.05:  # More than 5% gap
                gaps['critical_gaps'].append({
                    'category': category,
                    'current_ratio': current_ratios[category],
                    'target_ratio': standard,
                    'gap': difference,
                    'courses_needed': int(difference * total_courses)
                })
    
    # Analyze regional gaps
    if 'region_entity' in df.columns:
        for region in df['region_entity'].unique():
            region_df = df[df['region_entity'] == region]
            for category in TRAINING_CATEGORIES.keys():
                cat_col = f'is_{category.lower()}'
                if cat_col in df.columns:
                    coverage = region_df[cat_col].sum()
                    if coverage < 5:
                        gaps['opportunity_areas'].append({
                            'region': region,
                            'category': category,
                            'current_coverage': int(coverage),
                            'recommended_minimum': 5
                        })
    
    return gaps

def handle_missing_data(df, section):
    """Handle missing data with user input options"""
    missing_data_options = {
        'financial': {
            'cost': {'default': 500, 'description': 'Average cost per course'},
            'cost_per_learner': {'default': 50, 'description': 'Average cost per learner'},
            'reimbursement_amount': {'default': 1000, 'description': 'Average tuition reimbursement'}
        },
        'administrative': {
            'admin_count': {'default': 5, 'description': 'Average number of administrators'},
            'support_tickets': {'default': 10, 'description': 'Average monthly support tickets'},
            'resolution_time': {'default': 24, 'description': 'Average resolution time (hours)'}
        },
        'learning': {
            'duration_mins': {'default': 60, 'description': 'Average course duration (minutes)'},
            'learner_count': {'default': 100, 'description': 'Average learners per course'}
        }
    }

    if section not in missing_data_options:
        return df

    st.markdown(f"### Missing Data Handling - {section.title()}")
    st.markdown("Choose how to handle missing data:")
    
    handling_choice = st.radio(
        "Select missing data approach:",
        ["Skip analysis for missing data", 
         "Use industry averages",
         "Provide custom values"],
        key=f"missing_data_{section}"
    )

    if handling_choice == "Skip analysis for missing data":
        return df
    
    df_modified = df.copy()
    
    if handling_choice == "Use industry averages":
        for field, info in missing_data_options[section].items():
            if field not in df.columns or df[field].isna().any():
                df_modified[field] = df_modified[field].fillna(info['default'])
                st.info(f"Using industry average for {field}: {info['default']}")
    
    elif handling_choice == "Provide custom values":
        st.markdown("Enter custom values for missing data:")
        for field, info in missing_data_options[section].items():
            if field not in df.columns or df[field].isna().any():
                custom_value = st.number_input(
                    f"{info['description']} ({field})",
                    value=float(info['default']),
                    key=f"custom_{section}_{field}"
                )
                df_modified[field] = df_modified[field].fillna(custom_value)
    
    return df_modified

def fuzzy_match_courses(df, similarity_threshold=0.85):
    """
    Performs a second-pass fuzzy matching on courses to identify additional duplicates
    after initial deduplication. This helps catch courses with slightly different names
    or descriptions that are actually the same course.
    
    Args:
        df: DataFrame with initial deduplication already performed
        similarity_threshold: Minimum string similarity to consider courses as duplicates
        
    Returns:
        DataFrame with updated deduplication keys that group similar courses
    """
    try:
        import jellyfish  # For string similarity
        from tqdm.auto import tqdm  # For progress tracking
    except ImportError:
        print("Warning: jellyfish or tqdm package not available. Installing them for fuzzy matching...")
        import subprocess
        subprocess.check_call(["pip", "install", "jellyfish", "tqdm"])
        import jellyfish
        from tqdm.auto import tqdm
    
    print("Performing fuzzy course matching...")
    
    # Skip if too few courses or too many (performance reasons)
    if len(df) < 10 or len(df) > 20000:
        print(f"Skipping fuzzy matching: {'too few' if len(df) < 10 else 'too many'} courses")
        return df
    
    # Create a copy to avoid modifying the original
    df_fuzzy = df.copy()
    
    # Create a map to track which courses should be grouped together
    group_map = {}  # original_id -> group_id
    
    # First, extract courses that have proper course numbers (higher confidence)
    high_confidence = df_fuzzy[df_fuzzy['course_no'].str.contains(r'^source_\d+_', regex=True) == False]
    high_confidence = high_confidence[~high_confidence['course_no'].isin(['', 'nan', 'none', 'null', 'unknown'])]
    
    # If we have a reasonable number of high confidence courses, use them as reference
    if len(high_confidence) > 0 and len(high_confidence) < 5000:
        reference_courses = high_confidence
    else:
        # Otherwise just sample from all courses
        sample_size = min(5000, len(df_fuzzy))
        reference_courses = df_fuzzy.sample(sample_size)
    
    # Prepare fields for comparison
    if 'course_title' in df_fuzzy.columns:
        reference_courses['title_for_match'] = reference_courses['course_title'].astype(str).str.lower()
    else:
        # Can't do title matching without titles
        print("Skipping fuzzy matching: no course_title field available")
        return df
    
    # Create a list of courses to check against reference courses
    # Since we're comparing everything to reference courses, we can check all remaining courses
    all_other_courses = df_fuzzy[~df_fuzzy.index.isin(reference_courses.index)]
    
    # Prepare comparison field for these courses too
    all_other_courses['title_for_match'] = all_other_courses['course_title'].astype(str).str.lower()
    
    # Store the current deduplication keys
    reference_courses['original_dedup_key'] = reference_courses['dedup_key']
    all_other_courses['original_dedup_key'] = all_other_courses['dedup_key']
    
    # Counter for matches found
    matches_found = 0
    
    print(f"Comparing {len(all_other_courses)} courses against {len(reference_courses)} reference courses")
    
    # Reasonable batch size for memory management
    batch_size = 1000
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, len(all_other_courses), batch_size):
        end_idx = min(start_idx + batch_size, len(all_other_courses))
        batch = all_other_courses.iloc[start_idx:end_idx]
        
        print(f"Processing batch {start_idx//batch_size + 1}/{(len(all_other_courses) + batch_size - 1)//batch_size}")
        
        # For each course in the batch
        for i, (idx, course) in enumerate(batch.iterrows()):
            if i % 100 == 0:
                print(f"  Processing course {i}/{len(batch)}")
                
            # Skip if already matched to a group
            if idx in group_map:
                continue
                
            course_title = course['title_for_match']
            
            # Find potential matches (skip empty titles)
            if not course_title or course_title in ['nan', 'none', 'null', 'unknown']:
                continue
                
            # Check against reference courses
            for ref_idx, ref_course in reference_courses.iterrows():
                ref_title = ref_course['title_for_match']
                
                # Skip empty reference titles
                if not ref_title or ref_title in ['nan', 'none', 'null', 'unknown']:
                    continue
                
                # Calculate similarity
                similarity = jellyfish.jaro_winkler_similarity(course_title, ref_title)
                
                # If similar enough, use the reference course's dedup key
                if similarity >= similarity_threshold:
                    group_map[idx] = ref_course['original_dedup_key']
                    matches_found += 1
                    # Once we find a match, no need to check others
                    break
    
    print(f"Fuzzy matching found {matches_found} additional duplicate courses")
    
    # Update deduplication keys based on the matches
    for idx, group_key in group_map.items():
        df_fuzzy.loc[idx, 'dedup_key'] = group_key
    
    # Clean up temporary columns
    df_fuzzy = df_fuzzy.drop(columns=['title_for_match', 'original_dedup_key'])
    
    return df_fuzzy

def main():
    # Application Header
    st.title("LMS Content Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes your Learning Management System (LMS) content to provide insights into:
    - Content quality and completeness
    - Training distribution and focus
    - Usage patterns and engagement
    - Resource allocation and coverage
    """)
    
    # File Upload Section
    st.header("📤 Data Upload")
    with st.expander("Upload Instructions", expanded=True):
        st.markdown("""
        1. Prepare your Excel files containing LMS data
        2. Files should include course information, usage data, and metadata
        3. Multiple files can be uploaded for cross-reference analysis
        4. Supported format: .xlsx
        """)
    
    uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("👆 Upload your LMS data files to begin the analysis")
        return
        
    # Process uploaded files
    dataframes = []
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_excel(uploaded_file)
                dataframes.append(df)
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                continue
    
    if not dataframes:
        st.error("No valid data files were uploaded. Please check your files and try again.")
        return
        
    # Data Processing
    try:
        with st.spinner("Analyzing data..."):
            # Merge and standardize all data
            combined_data = merge_and_standardize_data(dataframes)
            combined_data = calculate_quality_score(combined_data)
            
            # Categorize content before analysis
            combined_data = categorize_content(combined_data)
            
            # Create analyzer instance
            combined_buffer = io.BytesIO()
            with pd.ExcelWriter(combined_buffer, engine='openpyxl') as writer:
                combined_data.to_excel(writer, index=False)
            combined_buffer.seek(0)
            analyzer = LMSContentAnalyzer(combined_buffer)
            
            # Fix row count discrepancy if it exists
            if len(analyzer.df) < len(combined_data):
                print(f"Row count discrepancy detected: {len(combined_data)} merged rows vs {len(analyzer.df)} analyzer rows")
                print("Using the complete merged dataset to ensure all data is included")
                analyzer.df = combined_data
            
            # High-level Overview
            st.header("📊 Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Courses", f"{len(combined_data):,}")
                if 'learner_count' in combined_data.columns:
                    st.metric("Total Learners", f"{combined_data['learner_count'].sum():,.0f}")
            
            with col2:
                active_courses = len(combined_data)
                if 'course_discontinued_from' in combined_data.columns:
                    active_courses = len(combined_data[
                        (combined_data['course_discontinued_from'].isna()) | 
                        (combined_data['course_discontinued_from'] > pd.Timestamp.now())
                    ])
                st.metric("Active Courses", f"{active_courses:,}")
                if 'region_entity' in combined_data.columns:
                    st.metric("Regions Covered", combined_data['region_entity'].nunique())
            
            with col3:
                if 'data_source' in combined_data.columns:
                    st.metric("Data Sources", combined_data['data_source'].nunique())
                if 'cross_reference_count' in combined_data.columns:
                    multi_source = len(combined_data[combined_data['cross_reference_count'] > 1])
                    st.metric("Cross-Referenced Courses", f"{multi_source:,}")
            
            # Data Quality Dashboard
            st.header("🔍 Data Quality Dashboard")
            st.markdown("""
            This dashboard provides detailed metrics about the quality of your learning content data. 
            Use these insights to identify areas that need attention for data cleansing and enrichment.
            """)
            
            # Get data quality metrics and visualizations
            from src.utils.visualization import plot_data_quality_metrics
            quality_results = plot_data_quality_metrics(combined_data)
            
            # Create tabs for different quality aspects
            quality_dashboard_tabs = st.tabs([
                "Overall Metrics", 
                "Data Completeness", 
                "Date Quality",
                "Cross-References",
                "Issues Summary"
            ])
            
            # Overall Metrics Tab
            with quality_dashboard_tabs[0]:
                if 'overall_quality' in quality_results['metrics']:
                    metrics = quality_results['metrics']['overall_quality']
                    
                    # Display key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Mean Quality Score", f"{metrics['mean_score']}")
                    with col2:
                        st.metric("High Quality Records", f"{metrics['high_quality']:,} ({metrics['percent_high_quality']}%)")
                    with col3:
                        st.metric("Low Quality Records", f"{metrics['low_quality']:,}")
                
                # Display quality categories chart
                if 'quality_categories' in quality_results['figures']:
                    st.plotly_chart(quality_results['figures']['quality_categories'], use_container_width=True)
                    
                    st.markdown("""
                    **Quality Score Categories:**
                    - **Excellent (0.8-1.0)**: Complete data with multiple cross-references
                    - **Good (0.6-0.8)**: Mostly complete with some cross-references
                    - **Fair (0.4-0.6)**: Partially complete with limited verification
                    - **Poor (0.2-0.4)**: Incomplete data with significant gaps
                    - **Very Poor (0.0-0.2)**: Minimal usable data
                    """)
            
            # Data Completeness Tab
            with quality_dashboard_tabs[1]:
                if 'completeness' in quality_results['figures']:
                    st.plotly_chart(quality_results['figures']['completeness'], use_container_width=True)
                    
                    st.markdown("""
                    **Understanding Data Completeness:**
                    - Columns below the 80% threshold need attention for data enrichment
                    - Missing critical fields limit analysis capabilities and data usability
                    - Consider prioritizing enrichment for columns with the lowest completeness rates
                    """)
            
            # Date Quality Tab
            with quality_dashboard_tabs[2]:
                if 'date_quality' in quality_results['figures']:
                    st.plotly_chart(quality_results['figures']['date_quality'], use_container_width=True)
                    
                    if 'date_quality' in quality_results['tables']:
                        date_df = quality_results['tables']['date_quality']
                        st.dataframe(date_df.sort_values('percent_valid'), use_container_width=True)
                        
                        st.markdown("""
                        **Date Field Importance:**
                        - Course version and availability dates are critical for timeline analysis
                        - Invalid dates can lead to incorrect filtering and reporting
                        - Discontinuation dates determine course lifecycle metrics
                        """)
            
            # Cross-References Tab
            with quality_dashboard_tabs[3]:
                if 'sources' in quality_results['figures']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(quality_results['figures']['sources'])
                    with col2:
                        if 'cross_references' in quality_results['figures']:
                            st.plotly_chart(quality_results['figures']['cross_references'])
                
                if 'verified_data' in quality_results['metrics']:
                    metrics = quality_results['metrics']['verified_data']
                    st.markdown(f"""
                    **Cross-Reference Analysis:**
                    - {metrics['multi_source']:,} records ({metrics['percent_verified']}%) are verified across multiple sources
                    - {metrics['single_source']:,} records appear in only one data source
                    - Cross-referenced data is generally more reliable and complete
                    """)
            
            # Issues Summary Tab
            with quality_dashboard_tabs[4]:
                if 'issues' in quality_results['tables']:
                    issues_df = quality_results['tables']['issues']
                    
                    st.markdown("### Critical Data Issues")
                    st.dataframe(issues_df, use_container_width=True)
                    
                    # Generate recommendations
                    st.markdown("### Recommendations")
                    recommendations = []
                    
                    # Check for missing critical data
                    missing_critical = issues_df[issues_df['issue_type'] == 'Missing Data']
                    if not missing_critical.empty:
                        recommendations.append("**Priority 1:** Fill in missing values for critical fields, especially course titles and IDs.")
                    
                    # Check for date issues
                    date_issues = issues_df[issues_df['issue_type'] == 'Invalid Date']
                    if not date_issues.empty:
                        recommendations.append("**Priority 2:** Fix invalid dates to enable proper timeline and lifecycle analysis.")
                    
                    # Check cross-references
                    ref_issues = issues_df[issues_df['issue_type'] == 'No Cross-References']
                    if not ref_issues.empty:
                        recommendations.append("**Priority 3:** Improve data verification by finding matching records across sources.")
                    
                    for rec in recommendations:
                        st.markdown(rec)
                    
                    # Export option
                    csv = issues_df.to_csv(index=False)
                    st.download_button(
                        "Download Issues Report",
                        csv,
                        "data_quality_issues.csv",
                        "text/csv"
                    )
            
            # Navigation
            st.header("🔍 Analysis Sections")
            
            # Content Similarity Analysis
            st.subheader("📊 Content Similarity Analysis")
            
            st.markdown("""
            This analysis identifies duplicate or highly similar courses across departments, 
            helping you identify opportunities for content consolidation and collaboration.
            """)
            
            # Run the content similarity analysis
            from src.utils.similarity import analyze_content_similarity
            
            # Add loading state indicator
            with st.spinner("Analyzing content similarity... This may take a minute for large datasets."):
                # Execute the similarity analysis
                similarity_results = analyze_content_similarity(combined_data)
            
            # Create tabs for different analysis views
            similarity_tabs = st.tabs([
                "Overview", 
                "Similarity Network", 
                "Department Overlap",
                "Potential Duplicates",
                "Consolidation Recommendations"
            ])
            
            # Overview tab
            with similarity_tabs[0]:
                if 'metrics' in similarity_results and 'similarity_counts' in similarity_results['metrics']:
                    metrics = similarity_results['metrics']['similarity_counts']
                    
                    # Display key metrics in columns
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Similar Pairs", f"{metrics['total_pairs']:,}")
                    with col2:
                        st.metric("High Similarity (≥80%)", f"{metrics['high_similarity']:,}")
                    with col3:
                        st.metric("Potential Duplicates", f"{metrics['potential_duplicates']:,}")
                    
                    # Display cross-department metrics if available
                    if 'cross_department' in similarity_results['metrics']:
                        cross_dept = similarity_results['metrics']['cross_department']
                        st.metric(
                            "Cross-Department Similarity", 
                            f"{cross_dept['total_cross_dept_pairs']:,} pairs ({cross_dept['percent_cross_dept']:.1f}%)"
                        )
                    
                    st.markdown("""
                    ### Understanding the Results
                    
                    - **High Similarity (≥80%)**: These courses have significant content overlap and are candidates for consolidation
                    - **Medium Similarity (60-80%)**: These courses have partial overlap and might benefit from coordination
                    - **Cross-Department Similarity**: Identifies similar content across different departments
                    
                    Use the other tabs to explore detailed visualizations and specific recommendations.
                    """)
                else:
                    st.info("No similarity data available. This could be due to insufficient text data in the courses.")
            
            # Similarity Network tab
            with similarity_tabs[1]:
                if 'figures' in similarity_results and 'similarity_network' in similarity_results['figures']:
                    st.plotly_chart(similarity_results['figures']['similarity_network'], use_container_width=True)
                    
                    st.markdown("""
                    **About the Network Visualization:**
                    
                    - Each node represents a course
                    - Connected nodes have content similarity above the threshold
                    - Larger nodes have more connections (similar to more courses)
                    - Node color indicates number of connections (darker = more connections)
                    - Hover over nodes and edges to see details
                    
                    Clusters in the network indicate groups of related courses that could be reviewed together.
                    """)
                else:
                    st.info("No network visualization available. Try adjusting the similarity threshold.")
            
            # Department Overlap tab
            with similarity_tabs[2]:
                if 'figures' in similarity_results and 'department_similarity' in similarity_results['figures']:
                    st.plotly_chart(similarity_results['figures']['department_similarity'], use_container_width=True)
                    
                    st.markdown("""
                    **About the Department Heatmap:**
                    
                    - Colors represent the average similarity between departments
                    - Darker colors indicate higher similarity between department content
                    - The diagonal shows self-similarity (always 100%)
                    
                    High similarity between different departments indicates potential for collaboration
                    or standardization of training content.
                    """)
                else:
                    st.info("Department similarity data not available. This may be due to missing department information in the data.")
            
            # Potential Duplicates tab
            with similarity_tabs[3]:
                if 'tables' in similarity_results and 'potential_duplicates' in similarity_results['tables']:
                    duplicates_df = similarity_results['tables']['potential_duplicates']
                    
                    # Show filtered version of the dataframe
                    display_cols = ['title_1', 'title_2', 'similarity_score', 'category_1', 'category_2']
                    
                    # Add department columns if available
                    if 'dept_1' in duplicates_df.columns:
                        display_cols.extend(['dept_1', 'dept_2', 'is_cross_dept'])
                    
                    st.dataframe(
                        duplicates_df[display_cols].sort_values('similarity_score', ascending=False),
                        use_container_width=True
                    )
                    
                    # Add download option
                    csv = duplicates_df.to_csv(index=False)
                    st.download_button(
                        "Download Potential Duplicates Data",
                        csv,
                        "potential_duplicates.csv",
                        "text/csv"
                    )
                    
                    st.markdown("""
                    **Interpreting Potential Duplicates:**
                    
                    The table above shows course pairs with high content similarity (≥80%). 
                    These courses are candidates for review and potential consolidation.
                    
                    Prioritize cross-department duplicates, as these may indicate lack of coordination
                    between different parts of the organization.
                    """)
                else:
                    st.info("No potential duplicates identified in the dataset.")
            
            # Consolidation Recommendations tab
            with similarity_tabs[4]:
                if 'tables' in similarity_results and 'consolidation_recommendations' in similarity_results['tables']:
                    recommendations_df = similarity_results['tables']['consolidation_recommendations']
                    
                    # Select columns for display
                    display_cols = ['title_1', 'title_2', 'similarity_score', 'recommendation']
                    
                    # Add columns if available
                    if 'learners_1' in recommendations_df.columns:
                        display_cols.extend(['learners_1', 'learners_2', 'total_enrollment'])
                    
                    if 'potential_savings' in recommendations_df.columns:
                        display_cols.append('potential_savings')
                        display_cols.append('savings_impact')
                    
                    # Display recommendations
                    st.dataframe(
                        recommendations_df[display_cols].sort_values(
                            'total_enrollment' if 'total_enrollment' in recommendations_df.columns else 'similarity_score', 
                            ascending=False
                        ),
                        use_container_width=True
                    )
                    
                    # Add download option
                    csv = recommendations_df.to_csv(index=False)
                    st.download_button(
                        "Download Detailed Recommendations",
                        csv,
                        "consolidation_recommendations.csv",
                        "text/csv"
                    )
                    
                    # Summary of recommendations
                    if 'potential_savings' in recommendations_df.columns:
                        total_savings = recommendations_df['potential_savings'].sum()
                        high_impact = (recommendations_df['savings_impact'] == 'High').sum()
                        
                        st.markdown(f"""
                        ### Key Findings
                        
                        - **Total potential learner savings:** {total_savings:,} enrollments
                        - **High-impact opportunities:** {high_impact} course pairs
                        
                        Consolidating these courses could reduce content management overhead,
                        improve content quality, and simplify the learner experience.
                        """)
                    
                    st.markdown("""
                    **Implementation Strategy:**
                    
                    1. Start with high-impact, high-similarity courses
                    2. For each pair, review the recommendation (typically keep the newer version)
                    3. Create a migration plan for learners
                    4. Archive the discontinued course after migration
                    """)
            
            # Time Analysis
            st.subheader("⏱️ Time Analysis")
            
            # Navigation
            st.header("🔍 Analysis Sections")
            analysis_choice = st.radio(
                "Select an analysis area to explore:",
                ["Data Quality & Completeness",
                 "Content Distribution",
                 "Learning Impact",
                 "Resource Allocation",
                 "Financial Analysis",
                 "Administrative Metrics",
                 "Trends & Predictions",
                 "Recommendations"],
                horizontal=True
            )
            
            if analysis_choice == "Data Quality & Completeness":
                st.subheader("📋 Data Quality Analysis")
                
                # Quality Score Overview
                quality_col1, quality_col2 = st.columns([2, 1])
                with quality_col1:
                    fig = plot_quality_distribution(combined_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with quality_col2:
                    st.markdown("### Quality Metrics")
                    completeness_stats = analyze_metadata_completeness(combined_data)
                    for category, stats in completeness_stats.items():
                        with st.expander(f"{category.title()} ({stats['avg_completeness']*100:.1f}% complete)"):
                            st.metric("Fields Present", f"{stats['fields_present']}/{stats['total_fields']}")
                            if stats['missing_fields']:
                                st.info("Missing: " + ", ".join(stats['missing_fields']))
                
                # NEW: Detailed Quality Issues
                st.markdown("### 🔍 Detailed Quality Issues")
                
                # Create tabs for different quality aspects
                quality_tabs = st.tabs(["Missing Data", "Cross-Reference Issues", "Data Inconsistencies"])
                
                with quality_tabs[0]:
                    st.markdown("#### Courses with Missing Critical Data")
                    missing_data = pd.DataFrame()
                    
                    # Define required columns and their display names
                    required_fields = ['course_title', 'course_description', 'course_no', 'category_name']
                    display_columns = ['course_id', 'course_title', 'course_no']
                    
                    # Ensure all display columns exist, create if missing
                    for col in display_columns:
                        if col not in combined_data.columns:
                            combined_data[col] = None
                    
                    for field in required_fields:
                        if field in combined_data.columns:
                            missing = combined_data[combined_data[field].isna()]
                            if not missing.empty:
                                field_missing = missing[display_columns].copy()
                                field_missing['missing_field'] = field
                                missing_data = pd.concat([missing_data, field_missing])
                    
                    if not missing_data.empty:
                        st.warning(f"Found {len(missing_data)} instances of missing critical data")
                        st.dataframe(missing_data.sort_values('course_title'), use_container_width=True)
                        csv = missing_data.to_csv(index=False)
                        st.download_button(
                            "Download Missing Data Report",
                            csv,
                            "missing_data_report.csv",
                            "text/csv"
                        )
                    else:
                        st.success("No critical missing data found!")

                with quality_tabs[1]:
                    st.markdown("#### Cross-Reference Analysis")
                    cross_ref_issues = combined_data[combined_data['cross_reference_count'] < 2]
                    if not cross_ref_issues.empty:
                        st.warning(f"{len(cross_ref_issues)} courses have insufficient cross-references")
                        st.dataframe(
                            cross_ref_issues[['course_id', 'course_title', 'cross_reference_count', 'data_source']],
                            use_container_width=True
                        )
                    else:
                        st.success("All courses have sufficient cross-references!")

                with quality_tabs[2]:
                    st.markdown("#### Data Inconsistencies")
                    inconsistencies = pd.DataFrame()
                    
                    # Check date inconsistencies
                    date_cols = ['course_version', 'course_available_from', 'course_discontinued_from']
                    for col in date_cols:
                        if col in combined_data.columns:
                            invalid_dates = combined_data[~pd.to_datetime(combined_data[col], errors='coerce').notna()]
                            if not invalid_dates.empty:
                                col_issues = invalid_dates[['course_id', 'course_title', 'course_no']].copy()
                                col_issues['issue_type'] = f'Invalid {col}'
                                inconsistencies = pd.concat([inconsistencies, col_issues])
                    
                    if not inconsistencies.empty:
                        st.dataframe(inconsistencies.sort_values('course_id'), use_container_width=True)
                        csv = inconsistencies.to_csv(index=False)
                        st.download_button(
                            "Download Inconsistencies Report",
                            csv,
                            "inconsistencies_report.csv",
                            "text/csv"
                        )
                    else:
                        st.success("No data inconsistencies found!")

            elif analysis_choice == "Content Distribution":
                st.subheader("📚 Content Distribution Analysis")
                
                # Category Distribution
                st.markdown("### Training Categories")
                focus_stats = analyze_training_focus(combined_data)
                focus_df = pd.DataFrame(focus_stats).T
                
                cat_col1, cat_col2 = st.columns([2, 1])
                with cat_col1:
                    fig = px.bar(focus_df, x=focus_df.index, y='percentage',
                               title='Distribution of Training Categories')
                    st.plotly_chart(fig, use_container_width=True)
                
                with cat_col2:
                    st.markdown("### Key Insights")
                    top_categories = focus_df.sort_values('percentage', ascending=False).head(3)
                    st.markdown("**Top Categories:**")
                    for idx, row in top_categories.iterrows():
                        st.markdown(f"- {idx}: {row['percentage']:.1f}%")
                
            elif analysis_choice == "Learning Impact":
                st.subheader("📈 Learning Impact Analysis")
                
                # Handle missing learning data
                learning_data = handle_missing_data(combined_data, 'learning')
                
                impact_col1, impact_col2 = st.columns(2)
                
                with impact_col1:
                    if 'learner_count' in learning_data.columns:
                        st.markdown("### Usage Patterns")
                        fig = px.histogram(learning_data, x='learner_count',
                                         title='Course Usage Distribution',
                                         nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                
                with impact_col2:
                    if 'course_type' in learning_data.columns and 'learner_count' in learning_data.columns:
                        st.markdown("### Impact by Course Type")
                        volume_by_type = learning_data.groupby('course_type')['learner_count'].sum()
                        fig = px.pie(values=volume_by_type.values, names=volume_by_type.index,
                                   title='Learning Volume Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_choice == "Resource Allocation":
                st.subheader("📊 Resource Allocation Analysis")
                
                resource_col1, resource_col2 = st.columns(2)
                
                with resource_col1:
                    if 'region_entity' in combined_data.columns:
                        st.markdown("### Regional Distribution")
                        region_dist = combined_data['region_entity'].value_counts()
                        fig = px.pie(values=region_dist.values, names=region_dist.index,
                                   title='Course Distribution by Region')
                        st.plotly_chart(fig, use_container_width=True)
                
                with resource_col2:
                    if 'data_source' in combined_data.columns:
                        st.markdown("### Content Sources")
                        source_dist = combined_data['data_source'].value_counts()
                        fig = px.pie(values=source_dist.values, names=source_dist.index,
                                   title='Content Source Distribution')
                        st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_choice == "Financial Analysis":
                st.subheader("💰 Financial Analysis")
                
                # Handle missing financial data
                financial_data = handle_missing_data(combined_data, 'financial')
                
                # Get financial metrics and validation results
                financials = analyzer.analyze_financial_metrics()
                
                if 'validation_issues' in financials:
                    with st.expander("Data Requirements & Options", expanded=True):
                        st.warning(financials['message'])
                        st.markdown("### Required Data Fields")
                        for missing in financials['validation_issues']['missing_columns']:
                            st.markdown(f"- **{missing['column']}**: {missing['description']}")
                            st.markdown(f"  - *Suggested default*: {missing.get('suggested_default', 'N/A')}")
                            st.markdown(f"  - *Impact if missing*: {missing.get('impact', 'Unknown')}")
                
                # Training Spend Analysis
                st.markdown("### Training Spend by System/Vendor")
                financials = analyzer.analyze_financial_metrics()
                
                if financials['training_spend']:
                    spend_df = pd.DataFrame(list(financials['training_spend'].items()),
                                          columns=['Vendor', 'Spend'])
                    fig = px.pie(spend_df, values='Spend', names='Vendor',
                               title='Training Spend Distribution')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Add training cost data to see spend analysis")
                
                # Learner Costs
                st.markdown("### 2024 Learner Training Costs")
                if financials['learner_costs']:
                    costs_df = pd.DataFrame(list(financials['learner_costs'].items()),
                                          columns=['Learner Type', 'Average Cost'])
                    st.dataframe(costs_df)
                
                # Tuition Reimbursement
                if financials['tuition_reimbursement']:
                    st.markdown("### Higher Education Support")
                    st.metric("Total 2024 Tuition Reimbursements",
                            f"${financials['tuition_reimbursement']['total']:,.2f}")
                    
                    program_df = pd.DataFrame(list(financials['tuition_reimbursement']['by_program'].items()),
                                            columns=['Program', 'Amount'])
                    fig = px.bar(program_df, x='Program', y='Amount',
                                title='Tuition Reimbursement by Program')
                    st.plotly_chart(fig, use_container_width=True)
            
            elif analysis_choice == "Administrative Metrics":
                st.subheader("👥 Administrative Analysis")
                
                # Handle missing administrative data
                admin_data = handle_missing_data(combined_data, 'administrative')
                
                # Get admin metrics and validation results
                admin_metrics = analyzer.analyze_admin_metrics()
                
                if 'validation_issues' in admin_metrics:
                    with st.expander("Data Requirements & Options", expanded=True):
                        st.warning(admin_metrics['message'])
                        st.markdown("### Required Data Fields")
                        for missing in admin_metrics['validation_issues']['missing_columns']:
                            st.markdown(f"- **{missing['column']}**: {missing['description']}")
                            st.markdown(f"  - *Suggested default*: {missing.get('suggested_default', 'N/A')}")
                            st.markdown(f"  - *Impact if missing*: {missing.get('impact', 'Unknown')}")
                
                # Market Split
                st.markdown("### 2024 Admin Split by Market and Function")
                if admin_metrics['market_split']:
                    market_df = pd.DataFrame(list(admin_metrics['market_split'].items()),
                                           columns=['Market', 'Count'])
                    fig = px.pie(market_df, values='Count', names='Market',
                               title='Admin Distribution by Market')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Efficiency Metrics
                st.markdown("### Admin Expertise and Efficiency")
                if admin_metrics['efficiency_ratios']:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Courses per Admin",
                                f"{admin_metrics['efficiency_ratios']['courses_per_admin']:.1f}")
                    with col2:
                        st.metric("Classes per Admin",
                                f"{admin_metrics['efficiency_ratios']['classes_per_admin']:.1f}")
                
                # Support Volume Analysis
                if 'support_volume' in admin_metrics:
                    st.markdown("### Support Volume Metrics")
                    volume_df = pd.DataFrame(admin_metrics['support_volume'])
                    st.dataframe(volume_df)
            
            elif analysis_choice == "Trends & Predictions":
                st.subheader("📈 Trends & Predictive Analysis")
                
                # Analyze trends
                trends = analyze_trends(combined_data)
                
                # Quality Score Evolution
                if 'quality_over_time' in trends:
                    st.markdown("### Quality Score Evolution")
                    quality_df = pd.DataFrame(trends['quality_over_time'])
                    fig = px.line(quality_df, x='year', y='mean',
                                title='Average Quality Score Over Time',
                                labels={'mean': 'Average Quality Score'})
                    st.plotly_chart(fig, use_container_width=True)
                
                # Category Evolution
                st.markdown("### Content Category Evolution")
                cat_tabs = st.tabs(list(TRAINING_CATEGORIES.keys()))
                for idx, (category, tab) in enumerate(zip(TRAINING_CATEGORIES.keys(), cat_tabs)):
                    with tab:
                        if f'{category}_evolution' in trends:
                            cat_df = pd.DataFrame(trends[f'{category}_evolution'])
                            if not cat_df.empty:
                                try:
                                    fig = px.line(cat_df, x='year', y='count',
                                                title=f'{category.replace("_", " ").title()} Courses Over Time')
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.warning(f"Could not plot trend for {category}: {str(e)}")
                            else:
                                st.info(f"No trend data available for {category}")
                
                # Predictive Analysis
                st.markdown("### 🔮 Predictive Insights")
                predictions = predict_course_utilization(combined_data)
                
                if predictions.get('high_potential_courses'):
                    st.markdown("#### High-Potential Underutilized Courses")
                    potential_df = pd.DataFrame(predictions['high_potential_courses'])
                    st.dataframe(potential_df.sort_values('predicted_engagement', ascending=False),
                               use_container_width=True)
                
                # Content Gap Analysis
                st.markdown("### 🎯 Content Gap Analysis")
                gaps = analyze_content_gaps(combined_data)
                
                if gaps['critical_gaps']:
                    st.warning("Critical Content Gaps Identified")
                    gaps_df = pd.DataFrame(gaps['critical_gaps'])
                    st.dataframe(gaps_df, use_container_width=True)
                    
                    # Visualization of gaps
                    fig = px.bar(gaps_df,
                               x='category',
                               y='gap',
                               title='Content Coverage Gaps',
                               labels={'gap': 'Coverage Gap (percentage points)'},
                               color='courses_needed')
                    st.plotly_chart(fig, use_container_width=True)
                
                if gaps['opportunity_areas']:
                    st.markdown("#### Regional Opportunity Areas")
                    opp_df = pd.DataFrame(gaps['opportunity_areas'])
                    st.dataframe(opp_df, use_container_width=True)

            else:  # Recommendations
                st.subheader("💡 Advanced Recommendations")
                
                # Create tabs for different recommendation categories
                rec_tabs = st.tabs([
                    "Critical Issues",
                    "Quality Improvements",
                    "Content Strategy",
                    "Resource Optimization",
                    "Action Plan"
                ])
                
                with rec_tabs[0]:
                    st.markdown("### 🚨 Critical Issues")
                    critical_issues = analyze_critical_issues(combined_data)
                    
                    if critical_issues['immediate_attention']:
                        st.error("Issues Requiring Immediate Action")
                        for issue in critical_issues['immediate_attention']:
                            with st.expander(f"{issue['category']} ({issue['count']} items)"):
                                st.dataframe(pd.DataFrame(issue['items']))
                                
                                # Export option
                                if issue['items']:
                                    csv = pd.DataFrame(issue['items']).to_csv(index=False)
                                    st.download_button(
                                        f"Download {issue['category']} Report",
                                        csv,
                                        f"{issue['category'].lower()}_issues.csv",
                                        "text/csv"
                                    )
                
                with rec_tabs[1]:
                    st.markdown("### 🔄 Quality Enhancement Strategy")
                    quality_strategy = analyze_quality_improvements(combined_data)
                    
                    # Display prioritized improvements
                    if quality_strategy['priorities']:
                        for priority in ['High', 'Medium', 'Low']:
                            items = [item for item in quality_strategy['priorities'] 
                                   if item['priority'] == priority]
                            if items:
                                with st.expander(f"{priority} Priority Items ({len(items)})"):
                                    st.dataframe(pd.DataFrame(items))
                
                with rec_tabs[2]:
                    st.markdown("### 📚 Content Strategy Recommendations")
                    content_strategy = analyze_content_strategy(combined_data)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if content_strategy.get('coverage_gaps'):
                            st.markdown("#### Coverage Gaps")
                            gaps_df = pd.DataFrame(content_strategy['coverage_gaps'])
                            fig = px.bar(gaps_df, x='category', y='gap',
                                       title='Content Coverage Gaps')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if content_strategy.get('optimization_opportunities'):
                            st.markdown("#### Optimization Opportunities")
                            opt_df = pd.DataFrame(content_strategy['optimization_opportunities'])
                            st.dataframe(opt_df, use_container_width=True)
                
                with rec_tabs[3]:
                    st.markdown("### 📊 Resource Optimization")
                    resource_insights = analyze_resource_optimization(combined_data)
                    
                    if resource_insights:
                        # Display efficiency metrics
                        metrics_col1, metrics_col2 = st.columns(2)
                        with metrics_col1:
                            if 'efficiency_score' in resource_insights:
                                st.metric("Resource Efficiency Score",
                                        f"{resource_insights['efficiency_score']:.2f}")
                        with metrics_col2:
                            if 'potential_savings' in resource_insights:
                                st.metric("Potential Resource Optimization",
                                        f"{resource_insights['potential_savings']:.1f}%")
                        
                        # Show specific recommendations
                        if resource_insights.get('recommendations'):
                            st.markdown("#### Optimization Recommendations")
                            for rec in resource_insights['recommendations']:
                                st.markdown(f"- {rec}")
                
                with rec_tabs[4]:
                    st.markdown("### 📝 Action Plan")
                    action_plan = generate_action_plan(combined_data)
                    
                    if action_plan:
                        # Timeline view
                        st.markdown("#### Implementation Timeline")
                        timeline_df = pd.DataFrame(action_plan['timeline'])
                        fig = px.timeline(timeline_df,
                                        x_start='start_date',
                                        x_end='end_date',
                                        y='action',
                                        color='priority')
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Detailed steps
                        st.markdown("#### Detailed Steps")
                        for phase in action_plan['phases']:
                            with st.expander(f"Phase {phase['number']}: {phase['name']}"):
                                for step in phase['steps']:
                                    st.markdown(f"- **{step['title']}**: {step['description']}")
                                    if step.get('resources'):
                                        st.markdown("  *Required resources:* " + 
                                                  ", ".join(step['resources']))
            
            # Export Options
            st.header("📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Analysis Report"):
                    # Generate report logic here
                    st.info("Report generation feature coming soon!")
            
            with col2:
                if st.button("Download Processed Data"):
                    # Download data logic here
                    st.info("Data export feature coming soon!")
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 