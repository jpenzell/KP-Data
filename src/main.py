"""Main application entry point for the LMS Content Analysis Dashboard."""

import streamlit as st
from pathlib import Path
from typing import List, Optional
import pandas as pd
import plotly.express as px
import numpy as np

from services.data_processor import DataProcessor
from services.analyzer import LMSAnalyzer
from ui.components.quality_analysis import display_quality_analysis
from models.data_models import AnalysisResults, ValidationResult

def initialize_app():
    """Initialize the Streamlit application."""
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

def handle_file_upload() -> List[Path]:
    """Handle file upload and return list of file paths."""
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
    
    # Save uploaded files temporarily
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved_paths.append(file_path)
    
    return saved_paths

def process_data(file_paths: List[Path]) -> tuple[Optional[AnalysisResults], List[ValidationResult]]:
    """Process the uploaded data files."""
    with st.spinner("Processing uploaded files..."):
        try:
            # Initialize data processor
            processor = DataProcessor()
            
            # Process files
            df = processor.process_excel_files(file_paths)
            
            # Initialize analyzer
            analyzer = LMSAnalyzer(df)
            
            # Get analysis results
            results = analyzer.get_analysis_results()
            
            return results, processor.get_validation_results()
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None, []

def display_overview(results: AnalysisResults):
    """Display high-level overview metrics."""
    st.header("📊 Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Courses", f"{results.total_courses:,}")
        if results.total_learners:
            st.metric("Total Learners", f"{results.total_learners:,.0f}")
    
    with col2:
        st.metric("Active Courses", f"{results.active_courses:,}")
        if results.regions_covered:
            st.metric("Regions Covered", results.regions_covered)
    
    with col3:
        if results.data_sources:
            st.metric("Data Sources", results.data_sources)
        if results.cross_referenced_courses:
            st.metric(
                "Cross-Referenced Courses",
                f"{results.cross_referenced_courses:,}"
            )

def display_navigation() -> str:
    """Display navigation and return selected analysis section."""
    st.header("🔍 Analysis Sections")
    
    return st.radio(
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

def display_content_distribution(results: AnalysisResults):
    """Display content distribution analysis."""
    st.subheader("📚 Content Distribution Analysis")
    
    # Category Distribution
    st.markdown("### Training Categories")
    
    # Create columns for visualization and insights
    cat_col1, cat_col2 = st.columns([2, 1])
    
    with cat_col1:
        if results.category_distribution:
            # Convert to percentages
            total_courses = results.total_courses
            category_percentages = {
                k: (v / total_courses) * 100
                for k, v in results.category_distribution.items()
            }
            
            # Create bar chart
            fig = px.bar(
                x=list(category_percentages.keys()),
                y=list(category_percentages.values()),
                title='Distribution of Training Categories',
                labels={'x': 'Category', 'y': 'Percentage of Courses'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with cat_col2:
        st.markdown("### Key Insights")
        if results.category_distribution:
            # Sort categories by count
            sorted_categories = sorted(
                results.category_distribution.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Display top categories
            st.markdown("**Top Categories:**")
            for category, count in sorted_categories[:3]:
                percentage = (count / total_courses) * 100
                st.markdown(f"- {category}: {percentage:.1f}%")
    
    # Regional Distribution
    st.markdown("### 🌍 Regional Distribution")
    if results.regional_distribution:
        reg_col1, reg_col2 = st.columns(2)
        
        with reg_col1:
            # Create pie chart for regional distribution
            fig = px.pie(
                values=list(results.regional_distribution.values()),
                names=list(results.regional_distribution.keys()),
                title='Course Distribution by Region'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with reg_col2:
            # Display regional statistics
            st.markdown("#### Regional Coverage")
            for region, count in results.regional_distribution.items():
                percentage = (count / total_courses) * 100
                st.metric(
                    region,
                    f"{count:,} courses",
                    f"{percentage:.1f}% of total"
                )
    
    # Temporal Distribution
    st.markdown("### 📅 Temporal Distribution")
    if results.temporal_patterns.get('creation_patterns'):
        temp_col1, temp_col2 = st.columns(2)
        
        with temp_col1:
            # Yearly distribution
            yearly_data = results.temporal_patterns['creation_patterns']['courses_per_year']
            if yearly_data:
                fig = px.line(
                    x=list(yearly_data.keys()),
                    y=list(yearly_data.values()),
                    title='Course Creation by Year',
                    labels={'x': 'Year', 'y': 'Number of Courses'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with temp_col2:
            # Monthly distribution
            monthly_data = results.temporal_patterns['creation_patterns']['courses_per_month']
            if monthly_data:
                fig = px.bar(
                    x=list(monthly_data.keys()),
                    y=list(monthly_data.values()),
                    title='Course Creation by Month',
                    labels={'x': 'Month', 'y': 'Number of Courses'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Content Gaps Analysis
    st.markdown("### 🎯 Content Gaps Analysis")
    if results.content_gaps:
        gaps_col1, gaps_col2 = st.columns(2)
        
        with gaps_col1:
            if results.content_gaps.get('critical_gaps'):
                st.warning("Critical Content Gaps Identified")
                gaps_df = pd.DataFrame(results.content_gaps['critical_gaps'])
                st.dataframe(gaps_df, use_container_width=True)
        
        with gaps_col2:
            if results.content_gaps.get('opportunity_areas'):
                st.info("Regional Opportunity Areas")
                opp_df = pd.DataFrame(results.content_gaps['opportunity_areas'])
                st.dataframe(opp_df, use_container_width=True)

def display_learning_impact(results: AnalysisResults):
    """Display learning impact analysis."""
    st.subheader("📈 Learning Impact Analysis")
    
    # Usage Patterns
    st.markdown("### 📊 Usage Patterns")
    usage_col1, usage_col2 = st.columns(2)
    
    with usage_col1:
        if 'learner_count' in results.df.columns:
            fig = px.histogram(
                results.df,
                x='learner_count',
                nbins=30,
                title='Distribution of Learner Count per Course',
                labels={'learner_count': 'Number of Learners', 'count': 'Number of Courses'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with usage_col2:
        if 'total_2024_activity' in results.df.columns:
            fig = px.histogram(
                results.df,
                x='total_2024_activity',
                nbins=30,
                title='Distribution of Course Activity',
                labels={'total_2024_activity': 'Activity Count', 'count': 'Number of Courses'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Engagement Metrics
    st.markdown("### 💡 Engagement Metrics")
    if 'avg_hours_per_completion' in results.df.columns:
        eng_col1, eng_col2 = st.columns(2)
        
        with eng_col1:
            avg_duration = results.df['avg_hours_per_completion'].mean()
            median_duration = results.df['avg_hours_per_completion'].median()
            st.metric(
                "Average Hours per Completion",
                f"{avg_duration:.1f}",
                f"{avg_duration - median_duration:+.1f} vs median"
            )
            
            fig = px.box(
                results.df,
                y='avg_hours_per_completion',
                title='Distribution of Hours per Completion'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with eng_col2:
            if 'category_name' in results.df.columns:
                # Calculate average hours by category
                avg_by_category = results.df.groupby('category_name')['avg_hours_per_completion'].mean().reset_index()
                fig = px.bar(
                    avg_by_category,
                    x='category_name',
                    y='avg_hours_per_completion',
                    title='Average Hours per Completion by Category',
                    labels={'category_name': 'Category', 'avg_hours_per_completion': 'Average Hours'}
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Completion Analysis
    st.markdown("### ✅ Completion Analysis")
    if all(col in results.df.columns for col in ['learner_count', 'total_2024_activity']):
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            # Calculate completion rate
            results.df['completion_rate'] = (
                results.df['total_2024_activity'] / results.df['learner_count']
            ).clip(0, 1)
            
            fig = px.histogram(
                results.df,
                x='completion_rate',
                nbins=20,
                title='Distribution of Course Completion Rates',
                labels={'completion_rate': 'Completion Rate', 'count': 'Number of Courses'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with comp_col2:
            # Top performing courses
            st.markdown("#### Top Performing Courses")
            top_courses = results.df.nlargest(5, 'completion_rate')[[
                'course_title', 'completion_rate', 'learner_count'
            ]]
            top_courses['completion_rate'] = top_courses['completion_rate'].map(
                lambda x: f"{x:.1%}"
            )
            st.dataframe(top_courses, use_container_width=True)
    
    # Impact Insights
    st.markdown("### 🎯 Impact Insights")
    impact_col1, impact_col2 = st.columns(2)
    
    with impact_col1:
        # Calculate and display key metrics
        if all(col in results.df.columns for col in ['avg_hours_per_completion', 'total_2024_activity']):
            total_learning_hours = (
                results.df['avg_hours_per_completion'] * results.df['total_2024_activity']
            ).sum()
            st.metric(
                "Total Learning Hours",
                f"{total_learning_hours:,.0f}",
                help="Total hours spent by all learners"
            )
        
        if 'completion_rate' in results.df.columns:
            avg_completion_rate = results.df['completion_rate'].mean()
            st.metric(
                "Average Completion Rate",
                f"{avg_completion_rate:.1%}",
                help="Average course completion rate"
            )
    
    with impact_col2:
        # Display recommendations based on analysis
        st.markdown("#### 💡 Recommendations")
        
        if 'completion_rate' in results.df.columns:
            # Identify courses needing attention
            avg_completion_rate = results.df['completion_rate'].mean()
            low_completion = results.df[
                results.df['completion_rate'] < avg_completion_rate * 0.5
            ]
            if not low_completion.empty:
                st.warning(
                    f"{len(low_completion)} courses have completion rates "
                    f"below {(avg_completion_rate * 0.5):.1%}"
                )
                
                with st.expander("View Courses Needing Attention"):
                    st.dataframe(
                        low_completion[[
                            'course_title',
                            'completion_rate',
                            'learner_count'
                        ]],
                        use_container_width=True
                    )

def main():
    """Main application entry point."""
    # Initialize the application
    initialize_app()
    
    # Handle file upload
    file_paths = handle_file_upload()
    
    if not file_paths:
        return
    
    # Process data
    results, validation_results = process_data(file_paths)
    
    if not results:
        return
    
    # Display overview
    display_overview(results)
    
    # Navigation
    selected_section = display_navigation()
    
    # Display selected analysis section
    if selected_section == "Data Quality & Completeness":
        display_quality_analysis(
            results.df,
            results.quality_metrics,
            validation_results
        )
    elif selected_section == "Content Distribution":
        display_content_distribution(results)
    elif selected_section == "Learning Impact":
        display_learning_impact(results)
    elif selected_section == "Resource Allocation":
        st.info("Resource Allocation analysis section is under development")
    elif selected_section == "Financial Analysis":
        st.info("Financial Analysis section is under development")
    elif selected_section == "Administrative Metrics":
        st.info("Administrative Metrics section is under development")
    elif selected_section == "Trends & Predictions":
        st.info("Trends & Predictions section is under development")
    else:  # Recommendations
        st.info("Recommendations section is under development")
    
    # Cleanup temporary files
    for file_path in file_paths:
        try:
            file_path.unlink()
        except Exception:
            pass
    
    try:
        Path("temp").rmdir()
    except Exception:
        pass

if __name__ == "__main__":
    main() 