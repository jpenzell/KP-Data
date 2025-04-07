"""Main application entry point for the LMS Content Analysis Dashboard."""

import streamlit as st
from pathlib import Path
from typing import List, Optional
import pandas as pd
import logging
import sys
from datetime import datetime
import argparse
import os

from .services.data_processor import DataProcessor
from .services.analyzer import LMSAnalyzer
from .ui.pages.home import render_home_page
from .ui.pages.analysis import render_analysis_page
from .models.data_models import AnalysisResults, ValidationResult, Severity
from .config.analysis_params import LOGGING_CONFIG
from .ui.app import run_app

# Set up logging configuration
log_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(os.path.dirname(log_dir), f"lms_analyzer_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def initialize_app():
    """Initialize the Streamlit application."""
    try:
        st.set_page_config(
            page_title="LMS Content Analysis",
            layout="wide"
        )
        
        st.title("LMS Content Analysis Dashboard")
        st.markdown("""
        This dashboard analyzes your Learning Management System (LMS) content to provide insights into:
        - Content quality and completeness
        - Training distribution and focus
        - Usage patterns and engagement
        - Resource allocation and coverage
        """)

        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing application: {str(e)}")
        st.error("Failed to initialize the application. Please try again later.")


def handle_file_upload() -> List:
    """Handle file upload section."""
    try:
        st.header("📤 Data Upload")
        with st.expander("Upload Instructions", expanded=True):
            st.markdown("""
            1. Prepare your Excel files containing LMS data
            2. Files should include course information, usage data, and metadata
            3. Multiple files can be uploaded for cross-reference analysis
            4. Supported format: .xlsx
            """)
        
        uploaded_files = st.file_uploader(
            "Upload Excel files",
            type=["xlsx"],
            accept_multiple_files=True
        )
        
        if not uploaded_files:
            st.info("👆 Upload your LMS data files to begin the analysis")
            return []
        
        file_names = [f.name for f in uploaded_files]
        logger.info(f"Files uploaded successfully: {file_names}")
        return uploaded_files
        
    except Exception as e:
        logger.error(f"Error handling file upload: {str(e)}")
        st.error("Failed to process file upload. Please try again.")
        return []


def process_data(uploaded_files: List) -> tuple[Optional[AnalysisResults], List[ValidationResult]]:
    """Process the uploaded data files."""
    try:
        with st.spinner("Processing uploaded files..."):
            st.info("Processing files - this may take a few moments for large datasets")
            progress_bar = st.progress(0)
            
            # Initialize data processor
            processor = DataProcessor()
            
            # Update progress
            progress_bar.progress(0.2)
            st.write("Parsing Excel files...")
            
            # Process files
            df = processor.process_excel_files(uploaded_files)
            progress_bar.progress(0.5)
            st.write(f"Successfully processed {len(uploaded_files)} files")
            
            # Show a preview of the data
            with st.expander("Preview of processed data", expanded=False):
                st.dataframe(df.head(5))
            
            # Initialize analyzer with semantic similarity
            st.write("Running analysis...")
            analyzer = LMSAnalyzer(use_semantic=True)
            progress_bar.progress(0.8)
            
            # Get analysis results
            results = analyzer.get_analysis_results()
            progress_bar.progress(1.0)
            
            # Log processing results
            logger.info(f"Successfully processed {len(uploaded_files)} files")
            logger.info(f"Analysis completed with {len(results.recommendations)} recommendations")
            
            # Get validation results
            validation_results = processor.get_validation_results()
            
            return results, validation_results
            
    except Exception as e:
        logger.error(f"Error in process_data: {str(e)}")
        st.error(f"An error occurred during data processing: {str(e)}")
        return None, []


def display_validation_results(validation_results: List[ValidationResult]):
    """Display validation results with appropriate styling."""
    try:
        for validation in validation_results:
            if validation.severity == Severity.WARNING:
                st.warning(validation.message)
            elif validation.severity == Severity.ERROR:
                st.error(validation.message)
            elif validation.severity == Severity.INFO:
                st.info(validation.message)
            elif validation.severity == Severity.SUCCESS:
                st.success(validation.message)
            
            # Log validation results
            logger.info(f"Validation {validation.severity}: {validation.message}")
            
    except Exception as e:
        logger.error(f"Error displaying validation results: {str(e)}")
        st.error("Failed to display validation results")


def main():
    """Main entry point of the application."""
    try:
        logger.info("Application initialized successfully")
        
        # Initialize components
        processor = DataProcessor()
        
        # Initialize analyzer with semantic similarity
        analyzer = LMSAnalyzer(use_semantic=True)
        
        # Run app
        run_app(processor, analyzer)
            
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main() 