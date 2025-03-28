"""Home page component for the LMS Analysis Dashboard."""

import streamlit as st
import plotly.express as px
from src.models.data_models import AnalysisResults
from src.ui.components.overview import display_metrics

def render_home_page(results: AnalysisResults):
    """Render the main dashboard home page."""
    # Overview Section
    st.header("📊 Overview")
    
    # Create three columns for key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Courses",
            f"{results.total_courses:,}",
            f"{results.active_courses:,} active"
        )
    
    with col2:
        if results.total_learners:
            st.metric(
                "Total Learners",
                f"{results.total_learners:,}"
            )
    
    with col3:
        if results.regions_covered:
            st.metric(
                "Regions Covered",
                f"{results.regions_covered:,}"
            )
    
    # Quick Insights
    st.header("💡 Quick Insights")
    
    # Create three columns for different types of insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Quality Overview")
        if results.quality_metrics:
            st.metric(
                "Overall Quality Score",
                f"{results.quality_metrics.overall_score:.2f}",
                f"{len(results.quality_metrics.improvement_areas)} areas to improve"
            )
        else:
            st.info("Quality metrics not available")
    
    with col2:
        st.subheader("Content Similarity")
        if results.similarity_metrics:
            st.metric(
                "High Similarity Pairs",
                f"{results.similarity_metrics.high_similarity_pairs:,}",
                f"{results.similarity_metrics.total_pairs:,} total pairs"
            )
        else:
            st.info("Similarity metrics not available")
    
    with col3:
        st.subheader("Recent Activity")
        if results.activity_metrics:
            st.metric(
                "Active Courses",
                f"{results.activity_metrics.active_courses:,}",
                f"{results.activity_metrics.recent_completions:,} recent completions"
            )
        else:
            st.info("Activity metrics not available")
    
    # Key Recommendations
    st.header("🎯 Key Recommendations")
    if results.recommendations:
        for i, rec in enumerate(results.recommendations[:3], 1):
            with st.expander(f"Priority {i}: {rec.title}"):
                st.markdown(f"""
                **Impact:** {rec.impact}  
                **Effort:** {rec.effort}
                
                {rec.description}
                
                **Category:** {rec.category}
                """)
    else:
        st.info("Recommendations will be available after running the full analysis") 