"""Test script to verify the setup of the LMS Content Analysis Dashboard."""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported."""
    try:
        import pandas
        import numpy
        import scipy
        import pydantic
        import openpyxl
        import streamlit
        import sklearn
        import plotly
        import dateutil
        print("✅ All required packages imported successfully")
    except ImportError as e:
        print(f"❌ Error importing packages: {str(e)}")
        return False
    return True

def test_directory_structure():
    """Test if all required directories and files exist."""
    required_dirs = [
        'src',
        'src/services',
        'src/models',
        'src/config',
        'src/utils',
        'src/ui',
        'src/tests'
    ]
    
    required_files = [
        'src/main.py',
        'src/services/data_processor.py',
        'src/services/analyzer.py',
        'src/models/data_models.py',
        'src/config/validation_rules.py',
        'src/config/column_mappings.py',
        'src/config/constants.py'
    ]
    
    base_path = Path(__file__).parent.parent
    
    # Check directories
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if not full_path.exists():
            print(f"❌ Missing directory: {dir_path}")
            return False
        print(f"✅ Found directory: {dir_path}")
    
    # Check files
    for file_path in required_files:
        full_path = base_path / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {file_path}")
            return False
        print(f"✅ Found file: {file_path}")
    
    return True

def test_python_version():
    """Test if Python version is compatible."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is not compatible. Please use Python 3.8 or higher.")
        return False

def main():
    """Run all tests."""
    print("Starting setup verification...\n")
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Directory Structure", test_directory_structure)
    ]
    
    all_passed = True
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        if not test_func():
            all_passed = False
            print(f"❌ {test_name} test failed")
        else:
            print(f"✅ {test_name} test passed")
    
    if all_passed:
        print("\n🎉 All tests passed! The environment is ready to run the LMS Content Analysis Dashboard.")
    else:
        print("\n❌ Some tests failed. Please fix the issues before running the dashboard.")

if __name__ == "__main__":
    main() 