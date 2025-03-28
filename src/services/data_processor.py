"""Data processing service for the LMS Content Analysis Dashboard."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Union
from typing import Any  # Separate import for Any
from pathlib import Path
import re
from scipy import stats
import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'lms_analyzer_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

from .data_validator import DataValidator
from .data_transformer import DataTransformer
from ..config.column_mappings import REQUIRED_COLUMNS, COLUMN_TYPES
from ..config.validation_rules import VALIDATION_RULES
from ..models.data_models import ValidationResult, AnalysisConfig

from ..config.constants import (
    COLUMN_MAPPING,
    DATE_FIELDS,
    NUMERIC_FIELDS,
    BOOLEAN_FIELDS,
    REQUIRED_FIELDS
)
from ..models.data_models import Course, ValidationResult, Severity

class DataProcessor:
    """Service for processing and validating LMS data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.required_columns = [
            'course_no', 'course_title', 'category_name',
            'course_description', 'course_type', 'region_entity'
        ]
        self.validation_results: List[ValidationResult] = []
        self.validator = DataValidator()
        self.transformer = DataTransformer()
    
    def process_excel_files(self, uploaded_files: List) -> pd.DataFrame:
        """Process multiple Excel files and combine them into a single DataFrame."""
        dfs = []
        file_errors = []
        
        for idx, uploaded_file in enumerate(uploaded_files):
            try:
                # Get the file name for logging
                file_name = uploaded_file.name
                
                # Read Excel file with error handling - use more flexible options
                try:
                    df = pd.read_excel(
                        uploaded_file,
                        engine='openpyxl',
                        na_values=['', 'NA', 'N/A', 'null', 'NULL', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'],
                        keep_default_na=True,
                    )
                except Exception as e:
                    logger.error(f"Error reading Excel file {file_name}: {str(e)}")
                    # Try again with different options
                    try:
                        df = pd.read_excel(
                            uploaded_file,
                            engine='openpyxl',
                            dtype=str,  # Read everything as strings
                        )
                    except Exception as e2:
                        raise ValueError(f"Could not read Excel file {file_name}: {str(e2)}")
                
                # Add source tracking
                df['data_source'] = file_name
                
                # Log file info
                logger.info(f"Processing file: {file_name}")
                logger.info(f"Shape: {df.shape}")
                logger.info(f"Columns: {df.columns.tolist()}")
                
                # Clean up unnamed columns - this is critical
                df = self._cleanup_unnamed_columns(df)
                
                # Standardize column names - very important to do this FIRST
                # Convert column names to lowercase and replace spaces with underscores
                df.columns = df.columns.str.lower().str.replace(' ', '_', regex=False).str.strip()
                
                # Apply specific column mappings using the COLUMN_MAPPING dictionary
                df = self._apply_column_mappings(df)
                
                # Clean up course_no column - CRITICAL
                if 'course_no' in df.columns:
                    # Clean up course numbers - remove invalid characters and standardize
                    df['course_no'] = df['course_no'].astype(str).str.strip()
                    df['course_no'] = df['course_no'].str.replace(':', '_', regex=False)
                    df['course_no'] = df['course_no'].str.replace(',', '_', regex=False)
                    df['course_no'] = df['course_no'].str.replace(' ', '_', regex=False)
                    df['course_no'] = df['course_no'].str.replace('\u200b', '', regex=False)  # Remove zero-width space
                    df['course_no'] = df['course_no'].str.replace(r'[^\w\d_-]', '', regex=True)  # Remove other invalid chars
                
                # Ensure course_no exists (this is critical for later steps)
                if 'course_no' not in df.columns:
                    logger.warning(f"No course_no column found in {file_name}, creating synthetic course numbers")
                    df['course_no'] = f"file_{idx}_" + df.index.astype(str)
                
                # Pre-process version field to avoid validation errors
                if 'course_version' in df.columns:
                    df['course_version'] = df['course_version'].astype(str).str.strip()
                
                # Pre-process category_name to avoid validation errors
                if 'category_name' in df.columns:
                    df['category_name'] = df['category_name'].astype(str).str.strip()
                    # Replace NaN or empty strings with a default category
                    mask = (df['category_name'].isna()) | (df['category_name'] == '') | (df['category_name'] == 'nan')
                    df.loc[mask, 'category_name'] = "Uncategorized"
                
                # Pre-process keywords field to avoid validation errors
                if 'course_keywords' in df.columns:
                    df['course_keywords'] = df['course_keywords'].astype(str).str.strip()
                
                # Clean and standardize data
                df = self._clean_data(df)
                
                # Validate the dataframe
                validation_results = self.validator.validate_dataframe(df)
                self.validation_results.extend(validation_results)
                
                # Log validation results
                for result in validation_results:
                    if result.severity == Severity.ERROR:
                        logger.error(f"Validation error in {file_name}: {result.message}")
                    elif result.severity == Severity.WARNING:
                        logger.warning(f"Validation warning in {file_name}: {result.message}")
                
                # Log data quality metrics
                null_counts = df.isna().sum()
                logger.info(f"Null value counts:\n{null_counts[null_counts > 0]}")
                
                dfs.append(df)
                logger.info(f"Successfully processed file: {file_name}")
                
            except Exception as e:
                error_msg = f"Error processing file {uploaded_file.name}: {str(e)}"
                logger.error(error_msg)
                file_errors.append(error_msg)
                self.validation_results.append(
                    ValidationResult(
                        message=error_msg,
                        severity=Severity.ERROR
                    )
                )
        
        if not dfs:
            error_msg = "No valid data found in the uploaded files"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Combine all dataframes
        try:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Successfully combined {len(dfs)} dataframes")
            logger.info(f"Combined shape: {combined_df.shape}")
            
            # Remove duplicates based on course number
            if 'course_no' in combined_df.columns:
                initial_count = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=['course_no'])
                if len(combined_df) < initial_count:
                    logger.warning(f"Removed {initial_count - len(combined_df)} duplicate courses")
            else:
                logger.warning("No course_no column found in combined dataset")
            
            # Clean and standardize data using the transformer
            combined_df = self.transformer.transform_dataframe(combined_df)
            logger.info("Successfully transformed combined dataframe")
            
            # Log final data quality metrics
            logger.info(f"Final shape: {combined_df.shape}")
            logger.info(f"Final columns: {combined_df.columns.tolist()}")
            logger.info(f"Data types:\n{combined_df.dtypes}")
            
            return combined_df
            
        except Exception as e:
            error_msg = f"Error combining dataframes: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _cleanup_unnamed_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up unnamed columns."""
        # First, identify unnamed columns
        unnamed_cols = [col for col in df.columns if 'unnamed' in str(col).lower() or not str(col).strip()]
        
        # Check if any of these columns have data we care about
        for col in unnamed_cols:
            # If the column is empty (all null or empty strings), drop it
            if df[col].isna().all() or (df[col].astype(str).str.strip() == '').all():
                df = df.drop(columns=[col])
            else:
                # Try to infer a better name for the column
                if df[col].dtype == 'object':
                    # Look at non-null values and see if they follow a pattern
                    sample_values = df[col].dropna().head(10).astype(str).tolist()
                    # Check for common keywords
                    keywords = ['course', 'title', 'id', 'date', 'type', 'category', 'region']
                    detected_keywords = []
                    
                    for keyword in keywords:
                        if any(keyword in value.lower() for value in sample_values):
                            detected_keywords.append(keyword)
                    
                    if detected_keywords:
                        new_name = f"inferred_{'_'.join(detected_keywords)}"
                        df = df.rename(columns={col: new_name})
                    else:
                        # Use a generic name
                        new_name = f"data_column_{df.columns.get_loc(col)}"
                        df = df.rename(columns={col: new_name})
        
        return df
    
    def _validate_dataframe(self, df: pd.DataFrame, file_name: str) -> None:
        """Validate the structure and content of a DataFrame."""
        # Check required columns
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            self.validation_results.append(
                ValidationResult(
                    message=f"Missing required columns in {file_name}: {', '.join(missing_columns)}",
                    severity=Severity.ERROR
                )
            )
        
        # Check for empty values in required columns
        for col in self.required_columns:
            if col in df.columns:
                empty_count = df[col].isna().sum()
                if empty_count > 0:
                    self.validation_results.append(
                        ValidationResult(
                            message=f"{empty_count} empty values found in {col} column in {file_name}",
                            severity=Severity.WARNING,
                            field=col
                        )
                    )
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize the data."""
        try:
            # Clean string columns
            for col in df.select_dtypes(include=['object']).columns:
                if col in df.columns:
                    try:
                        # Only apply string operations if the column has string values
                        if df[col].notna().any():
                            # Handle possible mixed types by forcing to string
                            df[col] = df[col].astype(str)
                            # Then clean the strings
                            df[col] = df[col].str.strip()
                            # Replace 'nan' strings from the astype conversion
                            df.loc[df[col] == 'nan', col] = None
                    except Exception as e:
                        logger.warning(f"Error cleaning string column {col}: {str(e)}")
            
            # Convert date columns
            date_columns = ['course_available_from', 'course_discontinued_from']
            for col in date_columns:
                if col in df.columns:
                    try:
                        # Handle mixed date formats more robustly
                        df[col] = self._convert_dates_robustly(df[col])
                    except Exception as e:
                        logger.warning(f"Error converting {col} to datetime: {str(e)}")
            
            # Convert numeric columns
            numeric_columns = ['duration_mins', 'learner_count', 'total_2024_activity']
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        # Clean the column first to handle mixed data
                        df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        # Then convert to numeric
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        # Replace NaN with 0 for count columns
                        if col in ['learner_count', 'total_2024_activity']:
                            df[col] = df[col].fillna(0)
                    except Exception as e:
                        logger.warning(f"Error converting {col} to numeric: {str(e)}")
            
            return df
        
        except Exception as e:
            logger.error(f"Error in _clean_data: {str(e)}")
            # Return the original dataframe if something went wrong
            return df
    
    def _convert_dates_robustly(self, series: pd.Series) -> pd.Series:
        """Convert a series to datetime using multiple formats."""
        # First clean the input
        series = series.astype(str).str.strip()
        series = series.replace('nan', pd.NA)
        series = series.replace('None', pd.NA)
        series = series.replace('', pd.NA)
        
        # Try common date formats
        formats = [
            '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
            '%d-%m-%Y', '%m-%d-%Y', '%Y.%m.%d',
            '%d.%m.%Y', '%m.%d.%Y', '%d %b %Y', '%b %d %Y',
            '%d %B %Y', '%B %d %Y', '%Y-%m-%d %H:%M:%S'
        ]
        
        result = pd.Series(index=series.index, dtype='object')
        
        # Try each format for each value
        for idx, value in series.items():
            if pd.isna(value):
                result[idx] = pd.NaT
                continue
                
            for fmt in formats:
                try:
                    date_value = pd.to_datetime(value, format=fmt)
                    result[idx] = date_value
                    break
                except:
                    pass
            
            # If no format worked, use the flexible parser
            if not isinstance(result[idx], pd.Timestamp):
                try:
                    result[idx] = pd.to_datetime(value, errors='coerce')
                except:
                    result[idx] = pd.NaT
        
        return result
    
    def get_validation_results(self) -> List[ValidationResult]:
        """Get the list of validation results."""
        return self.validation_results
    
    def calculate_basic_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic metrics from the processed data."""
        return {
            'total_courses': len(df),
            'active_courses': len(df[df['course_discontinued_from'].isna()]),
            'total_learners': df['learner_count'].sum(),
            'regions_covered': df['region_entity'].nunique(),
            'data_sources': df['data_source'].nunique(),
            'cross_referenced_courses': len(df[df['cross_reference_count'] > 1])
        }
    
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
                # Check if column matches any original names
                else:
                    for standard, originals in reverse_mapping.items():
                        if col_lower in originals:
                            target_col = standard
                            break
                
                if target_col and target_col not in seen_targets:
                    new_mapping[col] = target_col
                    seen_targets.add(target_col)
            
            # Apply the new mapping
            df = df.rename(columns=new_mapping)
            
            # Add missing required columns with null values
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            
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

    def _analyze_file_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the structure of a dataframe to determine its type and quality."""
        analysis = {
            'file_type': 'unknown',
            'confidence': 0.0,
            'column_coverage': 0.0,
            'data_quality': 0.0
        }
        
        # Calculate column coverage
        required_cols = set(REQUIRED_COLUMNS)
        present_cols = set(df.columns)
        coverage = len(required_cols.intersection(present_cols)) / len(required_cols)
        analysis['column_coverage'] = coverage
        
        # Determine file type based on column patterns
        if all(col in df.columns for col in ['course_no', 'course_title', 'category_name']):
            analysis['file_type'] = 'course_catalog'
            analysis['confidence'] = 0.9
        elif all(col in df.columns for col in ['learner_count', 'total_2024_activity']):
            analysis['file_type'] = 'activity_data'
            analysis['confidence'] = 0.8
        elif all(col in df.columns for col in ['course_available_from', 'course_discontinued_from']):
            analysis['file_type'] = 'course_schedule'
            analysis['confidence'] = 0.7
        
        # Calculate data quality score
        quality_score = 0.0
        if not df.empty:
            # Check for missing values
            missing_ratio = df.isna().mean().mean()
            quality_score = 1 - missing_ratio
            
            # Check for data consistency
            if 'course_no' in df.columns:
                unique_ratio = df['course_no'].nunique() / len(df)
                quality_score = (quality_score + unique_ratio) / 2
        
        analysis['data_quality'] = quality_score
        return analysis

    def _determine_merge_strategy(self, df: pd.DataFrame, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the best merge strategy based on file analysis."""
        strategy = {
            'merge_type': 'unknown',
            'confidence': 0.0,
            'merge_keys': [],
            'aggregation_rules': {}
        }
        
        # Determine merge keys based on file type
        if analysis['file_type'] == 'course_catalog':
            strategy['merge_type'] = 'primary'
            strategy['merge_keys'] = ['course_no']
            strategy['confidence'] = 0.9
        elif analysis['file_type'] == 'activity_data':
            strategy['merge_type'] = 'activity'
            strategy['merge_keys'] = ['course_no']
            strategy['aggregation_rules'] = {
                'learner_count': 'sum',
                'total_2024_activity': 'sum'
            }
            strategy['confidence'] = 0.8
        elif analysis['file_type'] == 'course_schedule':
            strategy['merge_type'] = 'schedule'
            strategy['merge_keys'] = ['course_no']
            strategy['confidence'] = 0.7
        
        return strategy

    def _process_single_dataframe_v2(
        self,
        df: pd.DataFrame,
        merge_keys: List[str],
        aggregation_rules: Dict[str, str],
        column_sources: Dict[str, List[str]]
    ) -> pd.DataFrame:
        """Process a single dataframe according to the merge strategy."""
        processed_df = df.copy()
        
        # Standardize column names
        processed_df = self._standardize_column_names_v2(processed_df)
        
        # Apply aggregation rules if any
        if aggregation_rules:
            processed_df = processed_df.groupby(merge_keys).agg(aggregation_rules).reset_index()
        
        # Track column sources
        for col in processed_df.columns:
            if col not in column_sources:
                column_sources[col] = []
            column_sources[col].append(processed_df['data_source'].iloc[0] if 'data_source' in processed_df.columns else 'unknown')
        
        return processed_df

    def _standardize_column_names_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using the mapping."""
        try:
            # Convert all column names to lowercase and strip whitespace
            df.columns = df.columns.str.lower().str.strip()
            
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
                # Check if column matches any original names
                else:
                    for standard, originals in reverse_mapping.items():
                        if col_lower in originals:
                            target_col = standard
                            break
                
                if target_col and target_col not in seen_targets:
                    new_mapping[col] = target_col
                    seen_targets.add(target_col)
            
            # Apply the new mapping
            df = df.rename(columns=new_mapping)
            
            # Add missing required columns with null values
            for col in REQUIRED_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            
            return df
            
        except Exception as e:
            print(f"Error standardizing column names: {str(e)}")
            return df

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

    def _apply_column_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply specific column mappings to standardize column names."""
        # Common column name mappings
        mapping_dict = {
            'course_number': 'course_no',
            'course_id': 'course_no',
            'id': 'course_no',
            'course_code': 'course_no',
            'template_no': 'course_no',
            'offering_template_no': 'course_no',
            'offering_no': 'course_no',
            
            'title': 'course_title',
            'name': 'course_title',
            'course_name': 'course_title',
            
            'description': 'course_description',
            'summary': 'course_description',
            'abstract': 'course_abstract',
            
            'version': 'course_version',
            'ver': 'course_version',
            
            'category': 'category_name',
            'cat': 'category_name',
            'subject': 'category_name',
            
            'keywords': 'course_keywords',
            'tags': 'course_keywords',
            'key_words': 'course_keywords',
            
            'course_from': 'course_available_from',
            'available_from': 'course_available_from',
            'start_date': 'course_available_from',
            'from_date': 'course_available_from',
            
            'course_to': 'course_discontinued_from',
            'discontinued_from': 'course_discontinued_from',
            'end_date': 'course_discontinued_from',
            'to_date': 'course_discontinued_from',
            
            'region': 'region_entity',
            'entity': 'region_entity',
            'business_unit': 'region_entity',
            
            'type': 'course_type',
            'format': 'course_type',
            'delivery_mode': 'course_type',
            
            'duration': 'duration_mins',
            'length': 'duration_mins',
            'minutes': 'duration_mins',
            
            'learners': 'learner_count',
            'students': 'learner_count',
            'participants': 'learner_count',
            'enrollment': 'learner_count',
            
            'activity': 'total_2024_activity',
            'completions': 'total_2024_activity',
            'total_activity': 'total_2024_activity'
        }
        
        # Apply the mappings
        for old_col, new_col in mapping_dict.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        return df 