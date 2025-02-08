"""Service for processing and standardizing LMS data."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union
from typing import Any  # Separate import for Any
from pathlib import Path
import re
from scipy import stats

from config.constants import (
    COLUMN_MAPPING,
    DATE_FIELDS,
    NUMERIC_FIELDS,
    BOOLEAN_FIELDS,
    REQUIRED_FIELDS
)
from models.data_models import Course, ValidationResult

class DataProcessor:
    """Handles data processing and standardization for LMS data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.validation_results: List[ValidationResult] = []
    
    def process_excel_files(self, file_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """Process multiple Excel files and combine them into a single DataFrame."""
        dataframes = []
        
        for file_path in file_paths:
            try:
                print(f"Processing file: {file_path}")  # Debug print
                df = pd.read_excel(file_path)
                print(f"Shape after reading: {df.shape}")  # Debug print
                df['data_source'] = Path(file_path).stem
                df['cross_reference_count'] = 1
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")  # Debug print
                self.validation_results.append(
                    ValidationResult(
                        is_valid=False,
                        errors=[f"Error processing {file_path}: {str(e)}"]
                    )
                )
        
        if not dataframes:
            raise ValueError("No valid data files were processed")
        
        try:
            result = self.merge_and_standardize_data(dataframes)
            print(f"Final shape: {result.shape}")  # Debug print
            return result
        except Exception as e:
            print(f"Error in merge_and_standardize_data: {str(e)}")  # Debug print
            raise
    
    def merge_and_standardize_data(self, dataframes: List[pd.DataFrame]) -> pd.DataFrame:
        """Merge and standardize multiple dataframes with dynamic reconciliation."""
        if not dataframes:
            return pd.DataFrame()
        
        standardized_dfs = []
        column_sources = {}
        file_analyses = []
        
        # First pass: Analyze all files
        print("\nAnalyzing file structures...")
        for idx, df in enumerate(dataframes):
            try:
                df_copy = df.copy()
                source_name = df_copy['data_source'].iloc[0] if not df_copy['data_source'].empty else f"source_{idx + 1}"
                print(f"\nAnalyzing source: {source_name}")
                
                # Standardize column names first
                df_copy = self._standardize_column_names(df_copy)
                
                # Analyze file structure
                analysis = self._analyze_file_structure(df_copy)
                merge_strategy = self._determine_merge_strategy(df_copy, analysis)
                
                file_analyses.append({
                    'df': df_copy,
                    'source_name': source_name,
                    'analysis': analysis,
                    'strategy': merge_strategy
                })
                
                print(f"File type: {analysis['file_type']} (confidence: {analysis['confidence']:.2f})")
                print(f"Merge strategy: {merge_strategy['merge_type']} (confidence: {merge_strategy['confidence']:.2f})")
                
            except Exception as e:
                print(f"Error analyzing dataframe {idx + 1}: {str(e)}")
                raise
        
        # Sort analyses by confidence for processing order
        file_analyses.sort(key=lambda x: (x['analysis']['confidence'] + x['strategy']['confidence']), reverse=True)
        
        # Second pass: Process files in order of confidence
        print("\nProcessing files in order of confidence...")
        for analysis in file_analyses:
            try:
                print(f"\nProcessing source: {analysis['source_name']}")
                df_copy = analysis['df']
                
                # Process the dataframe
                processed_df = self._process_single_dataframe(
                    df_copy,
                    analysis['strategy']['merge_keys'],
                    analysis['strategy']['aggregation_rules'],
                    column_sources
                )
                
                standardized_dfs.append(processed_df)
                
            except Exception as e:
                print(f"Error processing source {analysis['source_name']}: {str(e)}")
                continue
        
        try:
            print("\nCombining processed data...")
            combined_df = pd.concat(standardized_dfs, ignore_index=True)
            print(f"Shape after concat: {combined_df.shape}")
            
            # Process the combined dataframe
            result_df = self._process_combined_data(combined_df)
            
            # Add metadata about sources
            result_df['data_sources_count'] = result_df['data_source'].str.count('\|') + 1
            result_df['source_analysis'] = result_df.apply(
                lambda row: self._analyze_row_sources(row, column_sources),
                axis=1
            )
            
            # Print analysis summaries
            self._print_data_completeness_summary(result_df)
            
            return result_df
            
        except Exception as e:
            print(f"Error in final processing: {str(e)}")
            raise
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using the mapping."""
        try:
            # Convert all column names to lowercase and strip whitespace
            df.columns = df.columns.str.lower().str.strip()
            
            # Print original columns for debugging
            print(f"Original columns: {df.columns.tolist()}")
            
            # Create reverse mapping for standardized names
            reverse_mapping = {}
            for original, standard in COLUMN_MAPPING.items():
                if standard not in reverse_mapping:
                    reverse_mapping[standard] = []
                reverse_mapping[standard].append(original.lower())
            
            # Create new mapping based on actual columns
            new_mapping = {}
            seen_targets = set()  # Track already mapped target columns
            
            for col in df.columns:
                col_lower = col.lower()
                target_col = None
                
                # Check if column is already a standard name
                if col_lower in COLUMN_MAPPING.values():
                    target_col = col_lower
                # Check if column is in original mapping
                elif col_lower in COLUMN_MAPPING:
                    target_col = COLUMN_MAPPING[col_lower]
                # Check if column matches any standard name's variations
                else:
                    for standard, variations in reverse_mapping.items():
                        if any(variation in col_lower for variation in variations):
                            target_col = standard
                            break
                
                # If no match found, use some common mappings
                if not target_col:
                    if 'title' in col_lower and 'course' not in col_lower:
                        target_col = 'course_title'
                    elif 'desc' in col_lower:
                        target_col = 'course_description'
                    elif 'region' in col_lower:
                        target_col = 'region_entity'
                    elif 'duration' in col_lower or 'mins' in col_lower:
                        target_col = 'duration_mins'
                    elif 'learner' in col_lower and 'count' in col_lower:
                        target_col = 'learner_count'
                    else:
                        target_col = col  # Keep original name
                
                # Handle duplicate target columns
                if target_col in seen_targets:
                    suffix = 1
                    while f"{target_col}_{suffix}" in seen_targets:
                        suffix += 1
                    target_col = f"{target_col}_{suffix}"
                
                new_mapping[col] = target_col
                seen_targets.add(target_col)
            
            # Apply the mapping
            df = df.rename(columns=new_mapping)
            
            # Print final columns for debugging
            print(f"Standardized columns: {df.columns.tolist()}")
            print(f"Applied mappings: {new_mapping}")
            
            return df
        except Exception as e:
            print(f"Error standardizing column names: {str(e)}")
            return df
    
    def _handle_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle duplicate columns by adding suffixes."""
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            # Create a mapping of duplicated columns to their new names
            col_mapping = {}
            for col in df.columns:
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
            df.columns = [col_mapping.get(col, col) for col in df.columns]
        
        return df
    
    def _ensure_course_no(self, df: pd.DataFrame, source_idx: int) -> pd.DataFrame:
        """Ensure each row has a course_no."""
        # First check for any existing course number columns
        course_id_columns = ['course_no', 'course_id', 'offering_template_no']
        existing_col = next((col for col in course_id_columns if col in df.columns), None)
        
        if existing_col:
            # If we found a course ID column, standardize it
            df['course_no'] = df[existing_col].astype(str).str.lower().str.strip()
            # Remove the original column if it's not already course_no
            if existing_col != 'course_no' and existing_col in df.columns:
                df = df.drop(columns=[existing_col])
        else:
            # Special handling for files without course IDs
            if 'region_entity' in df.columns and 'duration_mins' in df.columns:
                # For duration/region files, create a composite key
                df['course_no'] = df.apply(
                    lambda row: f"region_{row['region_entity']}_{row['duration_mins']}",
                    axis=1
                )
            elif 'course_type' in df.columns and 'region_entity' in df.columns:
                # For type/region files, create a composite key
                df['course_no'] = df.apply(
                    lambda row: f"type_{row['course_type']}_{row['region_entity']}",
                    axis=1
                )
            elif 'course_title' in df.columns:
                # Create a more robust course number from title
                df['course_no'] = df['course_title'].str.lower().str.strip().str.replace(r'[^a-z0-9]+', '_', regex=True)
            else:
                # Use source and index as last resort
                df['course_no'] = f"source_{source_idx + 1}_" + df.index.astype(str)
        
        # Ensure course_no is string and standardized
        df['course_no'] = df['course_no'].astype(str).str.lower().str.strip()
        
        # Add source index prefix if not already present to ensure uniqueness across sources
        if not df['course_no'].str.startswith(f"source_{source_idx + 1}_"):
            df['course_no'] = f"source_{source_idx + 1}_" + df['course_no']
        
        # Add metadata about how the course_no was generated
        df['course_no_source'] = 'direct' if existing_col else 'generated'
        df['course_no_confidence'] = 1.0 if existing_col else 0.5
        
        return df
    
    def _process_single_dataframe(
        self,
        df: pd.DataFrame,
        merge_keys: List[str],
        aggregation_rules: Dict[str, Dict],
        column_sources: Dict[str, set]
    ) -> pd.DataFrame:
        """Process a single dataframe with dynamic merge strategy."""
        source_name = df['data_source'].iloc[0] if not df['data_source'].empty else 'unknown'
        
        # Track columns from this source
        for col in df.columns:
            if col not in column_sources:
                column_sources[col] = set()
            column_sources[col].add(source_name)
        
        # Handle duplicate columns
        df = self._handle_duplicate_columns(df)
        
        # Apply data transformations
        df = self._apply_data_transformations(df, merge_keys)
        
        # Create or standardize merge keys
        if not merge_keys:
            # Generate synthetic key if no merge keys available
            df['synthetic_key'] = f"source_{source_name}_" + df.index.astype(str)
            merge_keys = ['synthetic_key']
        
        # Validate merge keys
        self._validate_merge_keys(df, merge_keys)
        
        # Apply aggregation rules
        for col, rules in aggregation_rules.items():
            if col in df.columns and col not in merge_keys:
                df = self._apply_aggregation_rule(df, col, rules)
        
        # Add source tracking
        for col in df.columns:
            if col not in merge_keys + ['data_source', 'cross_reference_count']:
                df[f"{col}_source"] = source_name
                df[f"{col}_confidence"] = self._calculate_column_confidence(df, col)
        
        return df
    
    def _apply_data_transformations(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Apply necessary data transformations to prepare for merging."""
        for col in columns:
            if col not in df.columns:
                continue
            
            # Handle text columns
            if df[col].dtype == 'object':
                # Standardize text
                df[col] = df[col].astype(str).str.lower().str.strip()
                # Remove special characters
                df[col] = df[col].str.replace(r'[^a-z0-9\s]', '', regex=True)
            
            # Handle numeric columns
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Convert to float for consistency
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Handle date columns
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                # Standardize to UTC
                df[col] = pd.to_datetime(df[col], utc=True, errors='coerce')
        
        return df
    
    def _validate_merge_keys(self, df: pd.DataFrame, merge_keys: List[str]) -> None:
        """Validate merge keys for data quality."""
        for key in merge_keys:
            if key not in df.columns:
                raise ValueError(f"Merge key {key} not found in dataframe")
            
            # Check for null values
            null_count = df[key].isna().sum()
            if null_count > 0:
                self.validation_results.append(
                    ValidationResult(
                        is_valid=False,
                        errors=[f"Merge key {key} contains {null_count} null values"]
                    )
                )
            
            # Check for duplicates if single key
            if len(merge_keys) == 1:
                duplicate_count = len(df) - df[key].nunique()
                if duplicate_count > 0:
                    self.validation_results.append(
                        ValidationResult(
                            is_valid=True,
                            warnings=[f"Merge key {key} contains {duplicate_count} duplicate values"]
                        )
                    )
    
    def _apply_aggregation_rule(
        self,
        df: pd.DataFrame,
        column: str,
        rules: Dict[str, any]
    ) -> pd.DataFrame:
        """Apply aggregation rules to a column."""
        method = rules.get('method', 'first')
        
        try:
            if method == 'sum':
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
            elif method == 'mean':
                df[column] = pd.to_numeric(df[column], errors='coerce')
            elif method == 'concatenate':
                separator = rules.get('params', {}).get('separator', ' | ')
                unique = rules.get('params', {}).get('unique', True)
                if unique:
                    df[column] = df[column].astype(str).drop_duplicates()
                df[column] = df.groupby(level=0)[column].agg(lambda x: separator.join(
                    sorted(set(str(v) for v in x if pd.notna(v)))
                ))
            elif method == 'most_frequent':
                df[column] = df[column].mode().iloc[0] if not df[column].empty else None
            elif method == 'most_recent':
                if pd.api.types.is_datetime64_any_dtype(df[column]):
                    df[column] = df[column].max()
                else:
                    df[column] = df[column].iloc[-1]
        except Exception as e:
            print(f"Error applying aggregation rule for {column}: {str(e)}")
            # Fallback to first value
            df[column] = df[column].iloc[0] if not df[column].empty else None
        
        return df
    
    def _calculate_column_confidence(self, df: pd.DataFrame, column: str) -> float:
        """Calculate confidence score for a column's data quality."""
        try:
            # Start with base confidence
            confidence = 1.0
            
            # Reduce confidence based on null values
            null_percentage = (df[column].isna().sum() / len(df)) * 100
            confidence *= (1 - (null_percentage / 100))
            
            # Adjust based on data type and content
            if pd.api.types.is_numeric_dtype(df[column]):
                # Check for outliers
                z_scores = np.abs(stats.zscore(df[column].dropna()))
                outlier_percentage = (z_scores > 3).mean()
                confidence *= (1 - outlier_percentage)
            elif pd.api.types.is_string_dtype(df[column]):
                # Check for standardization
                unique_percentage = df[column].nunique() / len(df)
                if unique_percentage > 0.8:  # High cardinality
                    confidence *= 0.8
            
            return round(confidence, 2)
        except Exception:
            return 0.5  # Default confidence if calculation fails
    
    def _process_combined_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the combined dataframe with all standardization steps."""
        try:
            # Convert date columns
            for col in DATE_FIELDS:
                if col in df.columns:
                    df = self._process_date_column(df, col)
            
            # Convert numeric columns
            print("Processing numeric columns...")  # Debug print
            for col in NUMERIC_FIELDS:
                if col in df.columns:
                    print(f"Processing {col}...")  # Debug print
                    df = self._process_numeric_column(df, col)
            
            # Convert boolean columns
            for col in BOOLEAN_FIELDS:
                if col in df.columns:
                    df[col] = df[col].fillna(False).astype(bool)
            
            # Create derived columns
            df = self._create_derived_columns(df)
            
            # Group by course_no and aggregate
            print("Aggregating by course...")  # Debug print
            df = self._aggregate_by_course(df)
            print("Aggregation complete")  # Debug print
            
            return df
            
        except Exception as e:
            print(f"Error in _process_combined_data: {str(e)}")  # Debug print
            raise
    
    def _process_date_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Process a date column with validation."""
        try:
            # First try common date formats
            date_formats = [
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%m/%d/%Y',
                '%Y/%m/%d',
                '%d-%m-%Y',
                '%m-%d-%Y'
            ]
            
            temp_dates = None
            for date_format in date_formats:
                try:
                    temp_dates = pd.to_datetime(df[col], format=date_format, errors='coerce')
                    if not temp_dates.isna().all():
                        break
                except ValueError:
                    continue
            
            # If no standard format works, fall back to flexible parsing
            if temp_dates is None or temp_dates.isna().all():
                temp_dates = pd.to_datetime(df[col], errors='coerce')
            
            df[f'{col}_is_valid'] = pd.notna(temp_dates)
            df[col] = temp_dates
            
            invalid_count = (~df[f'{col}_is_valid']).sum()
            if invalid_count > 0:
                self.validation_results.append(
                    ValidationResult(
                        is_valid=True,
                        warnings=[f"{col} has {invalid_count} invalid or missing dates"]
                    )
                )
        except Exception as e:
            self.validation_results.append(
                ValidationResult(
                    is_valid=False,
                    errors=[f"Error processing dates for {col}: {str(e)}"]
                )
            )
            df[col] = pd.NaT
            df[f'{col}_is_valid'] = False
        
        return df
    
    def _process_numeric_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Process a numeric column with validation."""
        try:
            print(f"Processing numeric column: {col}")  # Debug print
            if col not in df.columns:
                print(f"Column {col} not found in dataframe. Available columns: {df.columns.tolist()}")  # Debug print
                return df
            
            # Print sample of the column data for debugging
            print(f"Sample of {col} data: {df[col].head()}")
            print(f"Data type of {col}: {df[col].dtype}")
            
            # Check if the column is already numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                print(f"Column {col} is already numeric")  # Debug print
                return df
            
            # Convert to numeric, handling errors
            temp_numeric = pd.to_numeric(df[col], errors='coerce')
            
            # Fill NaN with 0 for count-based columns
            if col in ['learner_count', 'cross_reference_count', 'total_2024_activity']:
                temp_numeric = temp_numeric.fillna(0)
            
            df[col] = temp_numeric
            print(f"Successfully processed column {col}")  # Debug print
            
        except Exception as e:
            print(f"Error processing numeric column {col}: {str(e)}")  # Debug print
            print(f"Column data type: {df[col].dtype}")  # Debug print
            print(f"Sample data causing error: {df[col].head()}")  # Debug print
            self.validation_results.append(
                ValidationResult(
                    is_valid=False,
                    errors=[f"Error converting {col} to numeric: {str(e)}"]
                )
            )
        
        return df
    
    def _create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived columns based on available data."""
        # Create full_course_id
        if all(col in df.columns for col in ['course_no', 'course_version']):
            df['full_course_id'] = df.apply(
                lambda row: f"{row['course_no']}_{row['course_version']}"
                if pd.notna(row['course_version']) else str(row['course_no']),
                axis=1
            )
        
        # Create course_duration_hours
        if 'duration_mins' in df.columns:
            df['course_duration_hours'] = pd.to_numeric(
                df['duration_mins'], errors='coerce'
            ).div(60)
        
        # Create is_active
        if 'course_discontinued_from' in df.columns:
            now = pd.Timestamp.now()
            df['is_active'] = df['course_discontinued_from'].apply(
                lambda x: True if pd.isna(x) else pd.to_datetime(x, errors='coerce') > now
            )
        
        return df
    
    def _process_column(self, df: pd.DataFrame, col: str, result: pd.DataFrame) -> pd.DataFrame:
        """Process a single column during aggregation."""
        try:
            if pd.api.types.is_numeric_dtype(df[col]):
                # For numeric columns
                if any(term in col.lower() for term in ['activity', 'count', 'learner']):
                    # Sum for activity and count metrics
                    temp_df = df.groupby('course_no', as_index=False)[col].sum()
                else:
                    # Mean for other numeric columns
                    temp_df = df.groupby('course_no', as_index=False)[col].mean()
            else:
                # For string columns, take the first non-null value
                temp_df = df.groupby('course_no', as_index=False).agg({
                    col: lambda x: next((v for v in x if pd.notna(v)), None)
                })
            
            # Merge back to result
            return result.merge(temp_df, on='course_no', how='left')
        except Exception as e:
            print(f"Error processing column {col}: {str(e)}")
            return result

    def _aggregate_by_course(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by course number with source tracking."""
        result = pd.DataFrame()  # Initialize result outside try block
        
        try:
            print("Starting aggregation...")
            print(f"Initial shape: {df.shape}")
            
            # Ensure course_no is properly formatted
            df['course_no'] = df['course_no'].astype(str).str.strip()
            
            # Create base DataFrame with course_no
            result = pd.DataFrame({'course_no': df['course_no'].unique()})
            print(f"Base shape after deduplication: {result.shape}")
            
            # Define column groups for different aggregation strategies
            sum_columns = ['total_2024_activity', 'learner_count', 'cross_reference_count']
            mean_columns = ['duration_mins', 'avg_hours_per_completion']
            first_non_null_columns = [
                'course_title', 'course_description', 'course_abstract', 
                'course_version', 'course_created_by', 'category_name', 
                'course_keywords', 'course_type', 'course_type_1'
            ]
            max_columns = [col for col in df.columns if str(col).startswith('is_')]
            latest_columns = ['course_available_from', 'course_discontinued_from']
            
            # Special handling for regional data
            regional_columns = ['region_entity', 'has_direct_reports']
            
            # Process each column group
            for col in df.columns:
                if col == 'course_no' or col == 'data_source':
                    continue
                
                try:
                    if col in sum_columns:
                        temp_df = df.groupby('course_no', as_index=False)[col].sum()
                    elif col in mean_columns:
                        temp_df = df.groupby('course_no', as_index=False)[col].mean()
                    elif col in first_non_null_columns:
                        temp_df = df.groupby('course_no', as_index=False).agg({
                            col: lambda x: next((v for v in x if pd.notna(v)), None)
                        })
                    elif col in max_columns:
                        temp_df = df.groupby('course_no', as_index=False)[col].max()
                    elif col in latest_columns:
                        temp_df = df.groupby('course_no', as_index=False)[col].max()
                    elif col in regional_columns:
                        # For regional data, concatenate unique values
                        temp_df = df.groupby('course_no', as_index=False).agg({
                            col: lambda x: ' | '.join(sorted(set(str(v) for v in x if pd.notna(v) and str(v).strip())))
                        })
                        # Clean up empty strings
                        temp_df[col] = temp_df[col].replace('', None)
                    else:
                        # Default to first non-null for unspecified columns
                        temp_df = df.groupby('course_no', as_index=False).agg({
                            col: lambda x: next((v for v in x if pd.notna(v)), None)
                        })
                    
                    result = result.merge(temp_df, on='course_no', how='left')
                except Exception as e:
                    print(f"Error processing column {col}: {str(e)}")
                    continue
            
            # Handle data source column specially
            if 'data_source' in df.columns:
                print("Processing data source column...")
                try:
                    source_df = df.groupby('course_no', as_index=False)['data_source'].agg(
                        lambda x: ' | '.join(sorted(set(str(v) for v in x if pd.notna(v))))
                    )
                    result = result.merge(source_df, on='course_no', how='left')
                except Exception as e:
                    print(f"Error processing data_source: {str(e)}")
            
            print(f"Final aggregation shape: {result.shape}")
            print(f"Final columns: {result.columns.tolist()}")
            print("Aggregation complete")
            
        except Exception as e:
            print(f"Error during aggregation: {str(e)}")
            return pd.DataFrame()
            
        return result
    
    def _analyze_row_sources(self, row: pd.Series, column_sources: Dict) -> Dict:
        """Analyze the sources for each field in a row."""
        try:
            sources = {}
            for col in row.index:
                if col in column_sources and pd.notna(row[col]):
                    sources[col] = column_sources[col]
            return sources
        except Exception as e:
            print(f"Error analyzing row sources: {str(e)}")
            return {}
    
    def _print_data_completeness_summary(self, df: pd.DataFrame) -> None:
        """Print summary of data completeness."""
        print("\nData Completeness Summary:")
        total_rows = len(df)
        
        # Analyze completeness by column
        completeness = {}
        for col in df.columns:
            if col not in ['data_source', 'cross_reference_count', 'data_sources_count', 'source_analysis']:
                non_null = df[col].notna().sum()
                completeness[col] = {
                    'present': non_null,
                    'missing': total_rows - non_null,
                    'percentage': (non_null / total_rows) * 100
                }
        
        # Print completeness report
        print("\nColumn Completeness:")
        for col, stats in completeness.items():
            print(f"{col}:")
            print(f"  Present: {stats['present']} ({stats['percentage']:.1f}%)")
            print(f"  Missing: {stats['missing']}")
        
        # Analyze cross-referencing
        print("\nCross-referencing Analysis:")
        source_counts = df['data_sources_count'].value_counts().sort_index()
        for count, freq in source_counts.items():
            print(f"Courses found in {count} sources: {freq} ({(freq/total_rows)*100:.1f}%)")
    
    def get_validation_results(self) -> List[ValidationResult]:
        """Get the validation results from processing."""
        return self.validation_results

    def _analyze_file_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the structure of a DataFrame."""
        analysis = {
            'row_count': 0,
            'column_count': 0,
            'columns': [],
            'columns_analysis': {},
            'file_type': 'unknown',
            'confidence': 0.0,
            'possible_id_columns': [],
            'categorical_columns': [],
            'numeric_columns': [],
            'date_columns': [],
            'value_distributions': {},
            'data_quality': {
                'null_percentages': {},
                'unique_percentages': {},
                'data_types': {}
            }
        }
        
        try:
            # Basic information
            analysis['row_count'] = len(df)
            analysis['column_count'] = len(df.columns)
            analysis['columns'] = df.columns.tolist()
            
            # Detect file type and confidence
            file_type_scores = {
                'course_data': 0,
                'activity_data': 0,
                'regional_data': 0,
                'reference_data': 0
            }
            
            # Analyze each column
            column_analysis = {}
            for col in df.columns:
                try:
                    # Ensure we're working with a Series, not a DataFrame
                    series = df[col].squeeze() if isinstance(df[col], pd.DataFrame) else df[col]
                    
                    column_info = {
                        'dtype': str(series.dtype),
                        'null_count': series.isna().sum(),
                        'null_percentage': (series.isna().sum() / len(df)) * 100,
                        'unique_count': series.nunique(),
                        'unique_percentage': (series.nunique() / len(df)) * 100
                    }
                    
                    # Track column type
                    if pd.api.types.is_string_dtype(series):
                        analysis['categorical_columns'].append(col)
                        non_null_series = series.dropna()
                        if len(non_null_series) > 0:
                            column_info.update({
                                'avg_length': non_null_series.str.len().mean(),
                                'max_length': non_null_series.str.len().max(),
                                'sample_values': non_null_series.head().tolist()
                            })
                            
                            # Update file type scores based on column content
                            if 'course' in col.lower():
                                file_type_scores['course_data'] += 1
                            if 'activity' in col.lower() or 'completion' in col.lower():
                                file_type_scores['activity_data'] += 1
                            if 'region' in col.lower() or 'entity' in col.lower():
                                file_type_scores['regional_data'] += 1
                    
                    elif pd.api.types.is_numeric_dtype(series):
                        analysis['numeric_columns'].append(col)
                        non_null_series = series.dropna()
                        if len(non_null_series) > 0:
                            column_info.update({
                                'min': float(non_null_series.min()),
                                'max': float(non_null_series.max()),
                                'mean': float(non_null_series.mean()),
                                'median': float(non_null_series.median()),
                                'std': float(non_null_series.std()),
                                'sample_values': non_null_series.head().tolist()
                            })
                    
                    elif pd.api.types.is_datetime64_any_dtype(series):
                        analysis['date_columns'].append(col)
                        non_null_series = series.dropna()
                        if len(non_null_series) > 0:
                            column_info.update({
                                'min_date': non_null_series.min().isoformat(),
                                'max_date': non_null_series.max().isoformat(),
                                'sample_values': [d.isoformat() for d in non_null_series.head()]
                            })
                    
                    # Check for potential ID columns
                    if column_info['unique_percentage'] > 80 and column_info['null_percentage'] < 20:
                        analysis['possible_id_columns'].append({
                            'column': col,
                            'confidence': column_info['unique_percentage'] / 100,
                            'unique_percentage': column_info['unique_percentage']
                        })
                    
                    # Store value distributions for categorical columns
                    if column_info['unique_count'] < 100 and column_info['null_percentage'] < 50:
                        value_counts = series.value_counts(normalize=True).head(10).to_dict()
                        analysis['value_distributions'][col] = value_counts
                    
                    # Store data quality metrics
                    analysis['data_quality']['null_percentages'][col] = column_info['null_percentage']
                    analysis['data_quality']['unique_percentages'][col] = column_info['unique_percentage']
                    analysis['data_quality']['data_types'][col] = column_info['dtype']
                    
                    column_analysis[col] = column_info
                except Exception as e:
                    print(f"Error analyzing column {col}: {str(e)}")
                    column_analysis[col] = {
                        'error': str(e),
                        'dtype': 'unknown',
                        'null_count': df[col].isna().sum()
                    }
            
            analysis['columns_analysis'] = column_analysis
            
            # Determine file type and confidence
            max_score = max(file_type_scores.values())
            if max_score > 0:
                analysis['file_type'] = max(
                    file_type_scores.items(),
                    key=lambda x: x[1]
                )[0]
                analysis['confidence'] = min(max_score / 3, 1.0)  # Normalize confidence
            else:
                analysis['file_type'] = 'unknown'
                analysis['confidence'] = 0.5
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing file structure: {str(e)}")
            return analysis

    def _detect_text_patterns(self, series: pd.Series) -> Dict[str, float]:
        """Detect common patterns in text data."""
        patterns = {
            'all_caps': 0,
            'title_case': 0,
            'contains_numbers': 0,
            'contains_special_chars': 0,
            'formatted_code': 0
        }
        
        sample = series.dropna().head(100)
        total = len(sample)
        
        if total == 0:
            return patterns
        
        for value in sample:
            if isinstance(value, str):
                if value.isupper():
                    patterns['all_caps'] += 1
                if value.istitle():
                    patterns['title_case'] += 1
                if any(c.isdigit() for c in value):
                    patterns['contains_numbers'] += 1
                if any(not c.isalnum() for c in value):
                    patterns['contains_special_chars'] += 1
                if re.match(r'^[A-Z0-9-_]+$', value):
                    patterns['formatted_code'] += 1
        
        # Convert to percentages
        return {k: (v / total) * 100 for k, v in patterns.items()}

    def _detect_column_relationships(self, df: pd.DataFrame) -> List[Dict[str, any]]:
        """Detect potential relationships between columns."""
        try:
            relationships = []
            
            # Get numeric columns for correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                try:
                    correlations = df[numeric_cols].corr()
                    for col1 in numeric_cols:
                        for col2 in numeric_cols:
                            if col1 < col2:  # Avoid duplicate pairs
                                correlation = correlations.loc[col1, col2]
                                if abs(correlation) > 0.7:  # Strong correlation threshold
                                    relationships.append({
                                        'type': 'correlation',
                                        'columns': [col1, col2],
                                        'strength': correlation,
                                        'confidence': abs(correlation)
                                    })
                except Exception as e:
                    print(f"Error calculating correlations: {str(e)}")
            
            # Detect potential foreign key relationships
            for col1 in df.columns:
                for col2 in df.columns:
                    if col1 != col2:
                        try:
                            # Check if one column's values are a subset of another
                            if set(df[col1].dropna().unique()).issubset(set(df[col2].dropna().unique())):
                                relationships.append({
                                    'type': 'potential_foreign_key',
                                    'columns': [col1, col2],
                                    'confidence': 0.8
                                })
                        except Exception as e:
                            print(f"Error checking relationship between {col1} and {col2}: {str(e)}")
                            continue
            
            return relationships
        except Exception as e:
            print(f"Error in column relationship detection: {str(e)}")
            return []

    def _identify_join_opportunities(self, df: pd.DataFrame, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Identify potential join opportunities with other data sources."""
        try:
            opportunities = []
            
            # Check ID columns
            for id_col in analysis.get('possible_id_columns', []):
                try:
                    opportunities.append({
                        'type': 'primary_key_join',
                        'column': id_col['column'],
                        'confidence': id_col.get('confidence', 0.5),
                        'unique_percentage': id_col.get('unique_percentage', 0)
                    })
                except Exception as e:
                    print(f"Error processing ID column {id_col}: {str(e)}")
                    continue
            
            # Check categorical columns that might be used for joining
            for col in analysis.get('categorical_columns', []):
                try:
                    if col in analysis.get('value_distributions', {}):
                        opportunities.append({
                            'type': 'categorical_join',
                            'column': col,
                            'confidence': 0.6,
                            'distinct_values': len(analysis['value_distributions'][col])
                        })
                except Exception as e:
                    print(f"Error processing categorical column {col}: {str(e)}")
                    continue
            
            # Check composite join possibilities
            if analysis.get('file_type') == 'regional_data':
                try:
                    region_cols = [col for col in df.columns if 'region' in col.lower()]
                    time_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'period'])]
                    
                    if region_cols and time_cols:
                        opportunities.append({
                            'type': 'composite_join',
                            'columns': region_cols + time_cols,
                            'confidence': 0.7,
                            'join_type': 'region_temporal'
                        })
                except Exception as e:
                    print(f"Error processing composite join possibilities: {str(e)}")
            
            return opportunities
        except Exception as e:
            print(f"Error identifying join opportunities: {str(e)}")
            return []

    def _determine_merge_strategy(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine how to merge this dataframe based on its structure."""
        strategy = {
            'merge_keys': [],
            'merge_type': 'direct',  # direct, composite, or derived
            'confidence': 0.0,
            'aggregation_rules': {},
            'error': None
        }
        
        try:
            # First try to use ID columns if available
            if analysis.get('possible_id_columns'):
                # Sort by confidence and uniqueness
                id_columns = sorted(
                    analysis['possible_id_columns'],
                    key=lambda x: (x['confidence'], x['unique_percentage']),
                    reverse=True
                )
                
                # Use the best ID column
                best_id = id_columns[0]
                strategy.update({
                    'merge_keys': [best_id['column']],
                    'merge_type': 'direct',
                    'confidence': best_id['confidence']
                })
            
            # If no good ID columns, try composite keys based on file type
            elif analysis.get('file_type') == 'regional_data':
                region_cols = [
                    col for col in analysis['categorical_columns']
                    if 'region' in col.lower() or 'entity' in col.lower()
                ]
                
                if region_cols:
                    strategy.update({
                        'merge_keys': region_cols[:1],  # Use first region column
                        'merge_type': 'composite',
                        'confidence': 0.7
                    })
                    
                    # Add duration if available for more precise matching
                    duration_cols = [
                        col for col in analysis['numeric_columns']
                        if 'duration' in col.lower() or 'mins' in col.lower()
                    ]
                    if duration_cols:
                        strategy['merge_keys'].append(duration_cols[0])
                        strategy['confidence'] = 0.8
            
            # Fallback to title-based matching
            elif any('title' in col.lower() for col in analysis['categorical_columns']):
                title_col = next(
                    col for col in analysis['categorical_columns']
                    if 'title' in col.lower()
                )
                strategy.update({
                    'merge_keys': [title_col],
                    'merge_type': 'derived',
                    'confidence': 0.5
                })
            
            # Determine aggregation rules for each column
            for col in df.columns:
                if col in strategy['merge_keys']:
                    continue
                
                if col in ['data_source', 'cross_reference_count']:
                    continue
                
                # Numeric columns
                if col in analysis.get('numeric_columns', []):
                    if any(term in col.lower() for term in ['count', 'total', 'sum', 'activity']):
                        strategy['aggregation_rules'][col] = {
                            'method': 'sum',
                            'confidence': 0.9
                        }
                    else:
                        strategy['aggregation_rules'][col] = {
                            'method': 'mean',
                            'confidence': 0.8
                        }
                
                # Date columns
                elif col in analysis.get('date_columns', []):
                    strategy['aggregation_rules'][col] = {
                        'method': 'most_recent',
                        'confidence': 0.7
                    }
                
                # Categorical columns
                elif col in analysis.get('categorical_columns', []):
                    if analysis['data_quality']['unique_percentages'].get(col, 100) > 80:
                        # High cardinality - keep first non-null
                        strategy['aggregation_rules'][col] = {
                            'method': 'first',
                            'confidence': 0.6
                        }
                    else:
                        # Low cardinality - concatenate unique values
                        strategy['aggregation_rules'][col] = {
                            'method': 'concatenate',
                            'params': {'separator': ' | ', 'unique': True},
                            'confidence': 0.7
                        }
            
            return strategy
            
        except Exception as e:
            print(f"Error determining merge strategy: {str(e)}")
            strategy['error'] = str(e)
            return strategy

    def calculate_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality scores for each course."""
        try:
            # Initialize quality score components
            df['completeness_score'] = 0.0
            df['cross_reference_score'] = 0.0
            df['validation_score'] = 0.0
            df['quality_score'] = 0.0
            
            # Calculate completeness score (40%)
            required_fields = ['course_title', 'course_description', 'course_no', 'category_name']
            if any(field in df.columns for field in required_fields):
                completeness_scores = pd.DataFrame()
                for field in required_fields:
                    if field in df.columns:
                        completeness_scores[field] = df[field].notna().astype(float)
                df['completeness_score'] = completeness_scores.mean(axis=1)
            
            # Calculate cross-reference score (30%)
            if 'cross_reference_count' in df.columns:
                df['cross_reference_score'] = (
                    pd.to_numeric(df['cross_reference_count'], errors='coerce')
                    .fillna(1)
                    .sub(1)
                    .div(3)
                    .clip(0, 1)
                )
            
            # Calculate validation score (30%)
            validation_columns = [col for col in df.columns if col.endswith('_is_valid')]
            if validation_columns:
                validation_scores = df[validation_columns].astype(float)
                df['validation_score'] = validation_scores.mean(axis=1)
            
            # Calculate final quality score
            df['quality_score'] = (
                0.4 * df['completeness_score'] +
                0.3 * df['cross_reference_score'] +
                0.3 * df['validation_score']
            )
            
            # Clean up intermediate score columns
            df = df.drop(columns=['completeness_score', 'cross_reference_score', 'validation_score'])
            
            return df
        except Exception as e:
            print(f"Error calculating quality scores: {str(e)}")
            return df 