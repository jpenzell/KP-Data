"""UI component for displaying quality analysis results."""

import streamlit as st
import pandas as pd
from typing import Dict, List

from models.data_models import QualityMetrics, ValidationResult
from utils.visualization import plot_quality_distribution, plot_box
from config.constants import REQUIRED_FIELDS

def display_quality_metrics(quality_metrics: QualityMetrics):
    """Display quality metrics with explanations."""
    st.markdown("### 📊 Quality Score Breakdown")
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Completeness Score",
            f"{quality_metrics.completeness_score:.2%}",
            help="Measures the presence of required fields"
        )
    
    with col2:
        st.metric(
            "Cross-Reference Score",
            f"{quality_metrics.cross_reference_score:.2%}",
            help="Measures data consistency across sources"
        )
    
    with col3:
        st.metric(
            "Validation Score",
            f"{quality_metrics.validation_score:.2%}",
            help="Measures data format and type validity"
        )
    
    # Overall quality score with color coding
    score_color = (
        "red" if quality_metrics.quality_score < 0.6
        else "orange" if quality_metrics.quality_score < 0.8
        else "green"
    )
    
    st.markdown(
        f"""
        <div style='text-align: center; padding: 1rem;'>
            <h2 style='color: {score_color}'>
                Overall Quality Score: {quality_metrics.quality_score:.1%}
            </h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def display_validation_issues(
    validation_results: List[ValidationResult],
    show_warnings: bool = True
):
    """Display validation issues and warnings."""
    st.markdown("### 🔍 Data Validation Results")
    
    # Count errors and warnings
    error_count = sum(
        len(result.errors) for result in validation_results
        if not result.is_valid
    )
    warning_count = sum(
        len(result.warnings) for result in validation_results
        if result.warnings
    )
    
    # Display summary
    if error_count > 0:
        st.error(f"Found {error_count} validation errors")
    elif warning_count > 0:
        st.warning(f"Found {warning_count} validation warnings")
    else:
        st.success("All validation checks passed!")
    
    # Show detailed issues
    if error_count > 0:
        with st.expander("View Validation Errors", expanded=True):
            for result in validation_results:
                if not result.is_valid:
                    for error in result.errors:
                        st.error(error)
    
    if show_warnings and warning_count > 0:
        with st.expander("View Validation Warnings"):
            for result in validation_results:
                if result.warnings:
                    for warning in result.warnings:
                        st.warning(warning)

def display_missing_data_analysis(df: pd.DataFrame):
    """Display analysis of missing data."""
    st.markdown("### 📉 Missing Data Analysis")
    
    # Calculate missing data statistics
    missing_stats = []
    
    for field in REQUIRED_FIELDS:
        if field in df.columns:
            missing_count = df[field].isna().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_stats.append({
                'Field': field,
                'Missing Count': missing_count,
                'Missing Percentage': f"{missing_percentage:.1f}%"
            })
    
    if missing_stats:
        st.dataframe(
            pd.DataFrame(missing_stats),
            use_container_width=True
        )
        
        # Highlight critical issues
        critical_fields = [
            stat for stat in missing_stats
            if float(stat['Missing Percentage'].rstrip('%')) > 20
        ]
        
        if critical_fields:
            st.error("Critical Missing Data Issues:")
            for field in critical_fields:
                st.markdown(
                    f"- **{field['Field']}**: {field['Missing Percentage']} missing"
                )

def display_quality_distribution(df: pd.DataFrame):
    """Display quality score distribution analysis."""
    st.markdown("### 📊 Quality Score Distribution")
    
    if 'quality_score' in df.columns:
        # Plot quality distribution
        fig = plot_quality_distribution(df)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Quality statistics
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.metric(
                "Average Quality Score",
                f"{df['quality_score'].mean():.2f}",
                help="Mean quality score across all courses"
            )
        
        with stats_col2:
            below_threshold = (df['quality_score'] < 0.6).sum()
            st.metric(
                "Courses Below Threshold",
                below_threshold,
                help="Number of courses with quality score below 0.6"
            )
        
        # Show courses needing attention
        if below_threshold > 0:
            with st.expander("View Courses Needing Attention"):
                attention_needed = df[df['quality_score'] < 0.6].sort_values(
                    'quality_score'
                )
                st.dataframe(
                    attention_needed[[
                        'course_no',
                        'course_title',
                        'quality_score'
                    ]],
                    use_container_width=True
                )

def display_quality_analysis(
    df: pd.DataFrame,
    quality_metrics: QualityMetrics,
    validation_results: List[ValidationResult]
):
    """Main function to display all quality analysis components."""
    st.markdown("## 🎯 Data Quality Analysis")
    
    # Quality metrics overview
    display_quality_metrics(quality_metrics)
    
    # Add tabs for different analyses
    quality_tabs = st.tabs([
        "Quality Distribution",
        "Validation Issues",
        "Missing Data",
        "Recommendations"
    ])
    
    with quality_tabs[0]:
        display_quality_distribution(df)
    
    with quality_tabs[1]:
        display_validation_issues(validation_results)
    
    with quality_tabs[2]:
        display_missing_data_analysis(df)
    
    with quality_tabs[3]:
        st.markdown("### 💡 Quality Improvement Recommendations")
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if quality_metrics.completeness_score < 0.8:
            recommendations.append({
                "priority": "High",
                "area": "Data Completeness",
                "recommendation": "Focus on completing required fields",
                "impact": "Improved data quality and analysis accuracy"
            })
        
        if quality_metrics.cross_reference_score < 0.6:
            recommendations.append({
                "priority": "Medium",
                "area": "Data Consistency",
                "recommendation": "Increase cross-referencing between sources",
                "impact": "Better data validation and reliability"
            })
        
        if quality_metrics.validation_score < 0.7:
            recommendations.append({
                "priority": "High",
                "area": "Data Validation",
                "recommendation": "Address data format and type issues",
                "impact": "Reduced errors and improved data integrity"
            })
        
        if recommendations:
            st.dataframe(
                pd.DataFrame(recommendations),
                use_container_width=True
            )
        else:
            st.success("No critical quality improvements needed at this time!") 