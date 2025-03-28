"""Test script to verify the LMS analyzer app setup."""

import sys
import os
from pathlib import Path

# Add the parent directory to the path to ensure modules can be imported
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("Testing LMS Analyzer App Setup...")

# Try to import core modules
try:
    # Import services
    from lms_analyzer.services.data_processor import DataProcessor
    print("✅ Imported DataProcessor")
    
    from lms_analyzer.services.data_validator import DataValidator
    print("✅ Imported DataValidator")
    
    from lms_analyzer.services.data_transformer import DataTransformer
    print("✅ Imported DataTransformer")
    
    from lms_analyzer.services.analyzer import LMSAnalyzer
    print("✅ Imported LMSAnalyzer")
    
    from lms_analyzer.services.export_service import ExportService
    print("✅ Imported ExportService")
    
    from lms_analyzer.services.visualization_service import VisualizationService
    print("✅ Imported VisualizationService")
    
    # Import models
    from lms_analyzer.models.data_models import (
        AnalysisResults, 
        ValidationResult, 
        QualityMetrics,
        ActivityMetrics,
        SimilarityMetrics,
        Recommendation,
        Alert,
        Course,
        Severity,
        VisualizationConfig
    )
    print("✅ Imported data models")
    
    # Import UI components
    from lms_analyzer.ui.pages.home import render_home_page
    from lms_analyzer.ui.pages.analysis import render_analysis_page
    print("✅ Imported UI pages")
    
    from lms_analyzer.ui.components.overview import display_metrics
    from lms_analyzer.ui.components.quality import display_quality_metrics
    from lms_analyzer.ui.components.similarity import display_similarity_metrics
    from lms_analyzer.ui.components.activity import display_activity_metrics
    from lms_analyzer.ui.components.recommendations import display_recommendations
    print("✅ Imported UI components")
    
    # Import config
    from lms_analyzer.config.column_mappings import COLUMN_MAPPINGS
    from lms_analyzer.config.analysis_params import QUALITY_THRESHOLDS
    from lms_analyzer.config.validation_rules import VALIDATION_RULES
    print("✅ Imported configuration")
    
    # Import utils
    from lms_analyzer.utils.text_processing import calculate_similarity
    from lms_analyzer.utils.date_utils import format_date
    from lms_analyzer.utils.file_utils import read_excel_file
    print("✅ Imported utilities")
    
    print("\n✅ All core modules imported successfully")
    
except ImportError as e:
    print(f"\n❌ Import error: {str(e)}")
    print(f"Current sys.path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Files in current directory: {os.listdir('.')}")
    print(f"Files in src directory: {os.listdir('./src')}")
    
except Exception as e:
    print(f"\n❌ Error: {str(e)}")

# Try to create instances of key classes
print("\nAttempting to create service instances...")
try:
    data_processor = DataProcessor()
    print("✅ Created DataProcessor")
    
    data_validator = DataValidator()
    print("✅ Created DataValidator")
    
    data_transformer = DataTransformer()
    print("✅ Created DataTransformer")
    
    export_service = ExportService()
    print("✅ Created ExportService")
    
    visualization_service = VisualizationService()
    print("✅ Created VisualizationService")
    
    print("\n✅ All service classes instantiated successfully")
    
except Exception as e:
    print(f"\n❌ Error creating service instances: {str(e)}")

print("\nSetup test complete.")

# Test the main application
try:
    from lms_analyzer.src.main import main
    print("\nTesting main application...")
    main()
    print("✅ Main application test successful!")
except Exception as e:
    print(f"\n❌ Main application error: {str(e)}")
    sys.exit(1) 