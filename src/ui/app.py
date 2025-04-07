"""Main Streamlit application module."""
import logging
import streamlit as st
import pandas as pd
from typing import List, Optional, Dict, Any
from ..services.data_processor import DataProcessor
from ..services.analyzer import LMSAnalyzer
from ..models.data_models import AnalysisResults, ValidationResult, Severity
from .pages.home import render_home_page
from .pages.analysis import render_analysis_page
from .components.sidebar import render_sidebar

# Set up logging
logger = logging.getLogger(__name__)

def run_app(processor: DataProcessor, analyzer: LMSAnalyzer):
    """
    Run the Streamlit application.
    
    Args:
        processor: The data processor.
        analyzer: The analyzer.
    """
    # Set page configuration
    st.set_page_config(
        page_title="LMS Course Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize application state if needed
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = False
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'validation_results' not in st.session_state:
        st.session_state.validation_results = []
    if 'files_uploaded' not in st.session_state:
        st.session_state.files_uploaded = False
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
        
    # Render sidebar
    render_sidebar()
    
    # Main application title
    st.title("LMS Course Analysis Dashboard")
    
    # Handle file upload
    uploaded_files = st.file_uploader(
        "Upload LMS course data files", 
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )
    
    # Store uploaded files in session state
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.session_state.files_uploaded = True
        logger.info(f"Files uploaded successfully: {[f.name for f in uploaded_files]}")
    
    # Process button
    if st.session_state.files_uploaded and st.button("Process Data"):
        results, validation_results = process_data(processor, analyzer, st.session_state.uploaded_files)
        
        if results:
            st.session_state.analysis_results = results
            st.session_state.validation_results = validation_results
            st.session_state.data_processed = True
            
            # Display success message
            st.success(f"Successfully processed {len(st.session_state.uploaded_files)} files")
            
            # Display validation warnings
            display_validation_results(validation_results)
        else:
            st.error("Error processing data. Please check the logs for more information.")
    
    # Display analysis if data has been processed
    if st.session_state.data_processed and st.session_state.analysis_results:
        # Create tabs for different pages
        tab1, tab2 = st.tabs(["Overview", "Similarity Analysis"])
        
        with tab1:
            render_home_page(st.session_state.analysis_results)
            
        with tab2:
            render_analysis_page(st.session_state.analysis_results)
    
def process_data(processor: DataProcessor, analyzer: LMSAnalyzer, uploaded_files: List) -> tuple:
    """
    Process uploaded data files.
    
    Args:
        processor: The data processor.
        analyzer: The analyzer.
        uploaded_files: The list of uploaded files.
        
    Returns:
        tuple: (analysis_results, validation_results) or (None, None) on error.
    """
    if not uploaded_files:
        return None, None
    
    try:
        # Display progress bar
        progress_bar = st.progress(0)
        st.write("Processing uploaded files...")
        
        # Process all files - use process_excel_files instead of process_files
        df = processor.process_excel_files(uploaded_files)
        validation_results = processor.get_validation_results()
        progress_bar.progress(0.5)
        
        # Set data in analyzer
        analyzer.set_data(df)
        progress_bar.progress(0.7)
        
        # Generate analysis results
        st.write("Running analysis...")
        analysis_results = analyzer.get_analysis_results()
        progress_bar.progress(1.0)
        
        logger.info(f"Successfully processed {len(uploaded_files)} files")
        logger.info(f"Analysis completed with {len(validation_results)} recommendations")
        
        return analysis_results, validation_results
    except Exception as e:
        logger.error(f"Error processing data: {str(e)}")
        st.error(f"Error processing data: {str(e)}")
        return None, None

def display_validation_results(validation_results: List[ValidationResult]):
    """
    Display validation warnings.
    
    Args:
        validation_results: The list of validation results.
    """
    for result in validation_results:
        if result.severity == Severity.WARNING:
            logger.info(f"Validation warning: {result.message}")
        elif result.severity == Severity.ERROR:
            logger.error(f"Validation error: {result.message}")
            st.error(f"Validation error: {result.message}") 