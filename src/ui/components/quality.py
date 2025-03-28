"""Quality metrics component for the home page."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from src.models.data_models import AnalysisResults
import plotly.graph_objects as go

def display_quality_metrics(results: AnalysisResults):
    """Display quality metrics in the analysis page."""
    st.header("Content Quality Analysis")
    display_quality_summary(results)

def display_quality_summary(results: AnalysisResults):
    """Display a summary of quality metrics in the home page."""
    if not results.quality_metrics:
        st.info("Quality metrics not available")
        return
    
    metrics = results.quality_metrics
    
    # Create a gauge chart for overall quality
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=metrics.overall_score * 100 if metrics.overall_score is not None else 0,
        title={'text': "Overall Quality"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 60], 'color': "lightgray"},
                {'range': [60, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightblue"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': metrics.overall_score * 100 if metrics.overall_score is not None else 0
            }
        }
    ))
    
    fig.update_layout(height=200, margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        completeness = metrics.completeness_score if metrics.completeness_score is not None else 0
        st.metric(
            "Completeness",
            f"{completeness:.1%}"
        )
    
    with col2:
        metadata = metrics.metadata_score if metrics.metadata_score is not None else 0
        st.metric(
            "Metadata Quality",
            f"{metadata:.1%}"
        )
    
    # Display improvement areas if any
    if metrics.improvement_areas:
        st.markdown("**Areas for Improvement:**")
        for area in metrics.improvement_areas:
            st.markdown(f"- {area}")
            
    # Display missing fields if any
    if metrics.missing_fields:
        st.markdown("**Missing Fields:**")
        for field in metrics.missing_fields:
            st.markdown(f"- {field}") 