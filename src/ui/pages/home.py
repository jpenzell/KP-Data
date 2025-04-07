"""Home page component for the LMS Analysis Dashboard."""

import streamlit as st
import plotly.express as px
import pandas as pd
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
    
    # Duplicate Courses Dashboard
    st.header("🔄 Duplicate Courses Dashboard")
    
    if results.similarity_metrics and results.similarity_metrics.duplicate_candidates:
        metrics = results.similarity_metrics
        
        # Create metrics for duplicates
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total Potential Duplicates",
                f"{len(metrics.duplicate_candidates):,}"
            )
        
        with col2:
            st.metric(
                "Cross-Department Duplicates",
                f"{metrics.cross_department_pairs:,}"
            )
        
        with col3:
            # Calculate percentage of cross-department duplicates
            if metrics.high_similarity_pairs > 0:
                cross_dept_pct = (metrics.cross_department_pairs / metrics.high_similarity_pairs) * 100
                st.metric(
                    "Cross-Department %",
                    f"{cross_dept_pct:.1f}%"
                )
            else:
                st.metric("Cross-Department %", "0%")
        
        # Show top cross-department duplicates
        cross_dept_dupes = [
            pair for pair in metrics.duplicate_candidates 
            if pair.get('cross_department', False)
        ]
        
        if cross_dept_dupes:
            st.subheader("Top Cross-Department Duplicate Courses")
            
            # Create a table of top duplicates
            top_dupes_data = []
            for pair in cross_dept_dupes[:5]:  # Take top 5
                course1_title = pair.get('course1_title', 'Unknown')
                course2_title = pair.get('course2_title', 'Unknown')
                course1_region = pair.get('course1_region', 'Unknown')
                course2_region = pair.get('course2_region', 'Unknown')
                match_type = pair.get('match_type', 'content')
                
                top_dupes_data.append({
                    'Department 1': course1_region,
                    'Course 1': f"{pair['course1']}: {course1_title}",
                    'Department 2': course2_region,
                    'Course 2': f"{pair['course2']}: {course2_title}",
                    'Similarity': f"{pair['similarity']:.2f}",
                    'Match Type': match_type.capitalize()
                })
            
            if top_dupes_data:
                st.table(pd.DataFrame(top_dupes_data))
            
            # Add a call to action
            st.info(
                "⚠️ **Duplicate Alert:** The analysis has identified significant course duplication "
                "across departments, which may be causing training redundancy and inconsistency. "
                "View the Similarity Analysis tab for detailed insights."
            )
        else:
            st.success("No cross-department duplicates detected.")
    else:
        st.info("Duplicate analysis not available. Run analysis with multiple files to detect duplicates.")
    
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