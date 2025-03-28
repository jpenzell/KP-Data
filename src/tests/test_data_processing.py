import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from lms_analyzer_app import merge_and_standardize_data, standardize_columns, create_derived_columns, calculate_quality_score

def create_test_dataframes():
    # Create first test dataframe with some variations in column names
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

    # Create second dataframe with different column names and some overlapping data
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
        'description': ['Advanced data analysis techniques', 'Machine learning in practice', 'Deep learning basics']
    })

    # Create third dataframe with missing data and different formats
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

def save_test_data_to_excel():
    """Save each test dataframe to an Excel file"""
    dataframes = create_test_dataframes()
    
    for i, df in enumerate(dataframes, 1):
        filename = f'test_data_{i}.xlsx'
        df.to_excel(filename, index=False)
        print(f"Saved {filename}")

def test_data_processing():
    print("Starting data processing test...")
    
    # Create test dataframes
    dataframes = create_test_dataframes()
    
    # Print original dataframes
    print("\nOriginal Dataframes:")
    for i, df in enumerate(dataframes, 1):
        print(f"\nDataframe {i} columns:")
        print(df.columns.tolist())
        print(f"\nDataframe {i} head:")
        print(df.head())
    
    # Process the dataframes
    print("\nProcessing dataframes...")
    try:
        result_df = merge_and_standardize_data(dataframes)
        
        # Print results
        print("\nProcessed DataFrame:")
        print("\nColumns:")
        print(result_df.columns.tolist())
        print("\nShape:", result_df.shape)
        print("\nSample of processed data:")
        print(result_df.head())
        
        # Print derived columns
        derived_cols = ['full_course_id', 'course_duration_hours', 'is_active', 'quality_score']
        print("\nDerived Columns:")
        for col in derived_cols:
            if col in result_df.columns:
                print(f"\n{col}:")
                print(result_df[col].head())
        
        # Print quality metrics
        print("\nQuality Metrics:")
        if 'quality_score' in result_df.columns:
            print(f"Average Quality Score: {result_df['quality_score'].mean():.2f}")
            print(f"Quality Score Range: {result_df['quality_score'].min():.2f} - {result_df['quality_score'].max():.2f}")
        
        # Print cross-reference statistics
        print("\nCross-reference Statistics:")
        cross_ref_counts = result_df['cross_reference_count'].value_counts()
        print(cross_ref_counts)
        
        return True, result_df
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        return False, None

if __name__ == "__main__":
    # First save test data to Excel files
    save_test_data_to_excel()
    
    # Then run the test
    success, result = test_data_processing()
    print(f"\nTest {'passed' if success else 'failed'}") 