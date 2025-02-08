"""Tests for data processing functionality."""

import pandas as pd
import numpy as np
from datetime import datetime
import pytest
from pathlib import Path

from services.data_processor import DataProcessor
from services.analyzer import LMSAnalyzer

def create_test_dataframes():
    """Create test dataframes with variations in column names."""
    # Create first test dataframe
    df1 = pd.DataFrame({
        'Course No': ['C001', 'C002', 'C003'],
        'Course Title': ['Python Basics', 'Data Analysis', 'Machine Learning'],
        'Dur Mins': [60, 120, 180],
        'Course Version': ['1.0', '2.0', None],
        'Course Available From': ['2023-01-01', '2023-02-01', '2023-03-01'],
        'Category Name': ['Programming', 'Data Science', 'AI'],
        'Data Source': ['System A', 'System A', 'System A'],
        'total_2024_activity': [100, 150, 200],
        'avg hrs spent per completion': [2, 3, 4],
        'learner_count': [50, 75, 100],
        'is_leadership_development': [False, False, True],
        'is_managerial_supervisory': [False, True, False],
        'is_mandatory_compliance': [True, False, False],
        'is_profession_specific': [True, True, True],
        'is_interpersonal_skills': [False, True, True],
        'is_it_systems': [True, False, False],
        'is_clinical': [False, False, False],
        'is_nursing': [False, False, False],
        'is_pharmacy': [False, False, False],
        'is_safety': [False, False, False],
        'Course Description': ['Learn Python fundamentals', 'Master data analysis', 'AI/ML fundamentals']
    })

    # Create second dataframe with different column names
    df2 = pd.DataFrame({
        'course_no': ['C002', 'C003', 'C004'],
        'title': ['Data Analysis Advanced', 'Machine Learning', 'Deep Learning'],
        'duration': [150, 180, 240],
        'version': ['2.1', '1.0', '1.0'],
        'available_from': ['2023-02-15', '2023-03-01', '2023-04-01'],
        'category_name': ['Data Science', 'AI', 'AI'],
        'source': ['System B', 'System B', 'System B'],
        'total_2024_activity': [175, 225, 300],
        'avg hrs spent per completion': [3.5, 4, 5],
        'learner_count': [80, 110, 150],
        'is_leadership_development': [False, True, True],
        'is_managerial_supervisory': [True, False, False],
        'is_mandatory_compliance': [False, False, False],
        'is_profession_specific': [True, True, True],
        'is_interpersonal_skills': [True, True, True],
        'is_it_systems': [False, False, False],
        'is_clinical': [False, False, False],
        'is_nursing': [False, False, False],
        'is_pharmacy': [False, False, False],
        'is_safety': [False, False, False],
        'description': ['Advanced data analysis', 'Machine learning in practice', 'Deep learning basics']
    })

    # Create third dataframe with missing data
    df3 = pd.DataFrame({
        'Course No': ['C001', 'C004', 'C005'],
        'Title': ['Python Basics', 'Deep Learning Fundamentals', None],
        'Duration Minutes': [60, None, 90],
        'Course Description': ['Learn Python', 'Advanced DL', 'TBD'],
        'data_sources': ['System C', 'System C', 'System C'],
        'total_2024_activity': [120, 280, 50],
        'avg hrs spent per completion': [2.2, 4.5, 1.5],
        'learner_count': [55, 140, 25],
        'is_leadership_development': [False, True, False],
        'is_managerial_supervisory': [False, False, False],
        'is_mandatory_compliance': [True, False, True],
        'is_profession_specific': [True, True, False],
        'is_interpersonal_skills': [False, True, False],
        'is_it_systems': [True, False, False],
        'is_clinical': [False, False, True],
        'is_nursing': [False, False, True],
        'is_pharmacy': [False, False, False],
        'is_safety': [False, False, True]
    })

    return [df1, df2, df3]

def test_data_processing():
    """Test data processing functionality."""
    print("Starting data processing test...")
    
    # Create test dataframes
    dataframes = create_test_dataframes()
    
    # Initialize processor
    processor = DataProcessor()
    
    try:
        # Process the dataframes
        result_df = processor.merge_and_standardize_data(dataframes)
        
        # Verify basic properties
        assert len(result_df) > 0, "Result DataFrame is empty"
        assert 'course_no' in result_df.columns, "Missing course_no column"
        assert 'quality_score' in result_df.columns, "Missing quality_score column"
        
        # Verify data standardization
        assert result_df['course_no'].notna().all(), "Found null course_no values"
        assert result_df['cross_reference_count'].max() > 1, "Cross-reference counting failed"
        
        # Initialize analyzer
        analyzer = LMSAnalyzer(result_df)
        
        # Get analysis results
        results = analyzer.get_analysis_results()
        
        # Verify analysis results
        assert results.total_courses == len(result_df), "Total courses mismatch"
        assert results.quality_metrics is not None, "Missing quality metrics"
        assert results.quality_metrics.quality_score > 0, "Invalid quality score"
        
        print("\nTest Results:")
        print(f"Total Courses: {results.total_courses}")
        print(f"Active Courses: {results.active_courses}")
        print(f"Quality Score: {results.quality_metrics.quality_score:.2f}")
        print(f"Completeness Score: {results.quality_metrics.completeness_score:.2f}")
        print(f"Cross-Reference Score: {results.quality_metrics.cross_reference_score:.2f}")
        print(f"Validation Score: {results.quality_metrics.validation_score:.2f}")
        
        return True, result_df, results
        
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        return False, None, None

if __name__ == "__main__":
    success, df, results = test_data_processing()
    print(f"\nTest {'passed' if success else 'failed'}") 