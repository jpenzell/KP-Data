"""Overview metrics component for displaying high-level statistics."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.models.data_models import AnalysisResults

def display_metrics(results: AnalysisResults):
    """Display high-level overview metrics in a three-column layout."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Courses",
            f"{results.total_courses:,}"
        )
        if hasattr(results, 'total_learners'):
            st.metric(
                "Total Learners",
                f"{results.total_learners:,.0f}"
            )
    
    with col2:
        st.metric(
            "Active Courses",
            f"{results.active_courses:,}"
        )
        if hasattr(results, 'regions_covered'):
            st.metric(
                "Regions Covered",
                results.regions_covered
            )
    
    with col3:
        if hasattr(results, 'data_sources'):
            st.metric(
                "Data Sources",
                results.data_sources
            )
        if hasattr(results, 'cross_referenced_courses'):
            st.metric(
                "Cross-Referenced Courses",
                f"{results.cross_referenced_courses:,}"
            )
    
    # Display any alerts or warnings
    if hasattr(results, 'alerts'):
        for alert in results.alerts:
            if alert.severity == "warning":
                st.warning(alert.message)
            elif alert.severity == "info":
                st.info(alert.message)
            elif alert.severity == "error":
                st.error(alert.message)
            elif alert.severity == "success":
                st.success(alert.message) 