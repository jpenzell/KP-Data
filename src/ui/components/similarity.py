"""Similarity metrics component for the home page."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from src.models.data_models import AnalysisResults

def display_similarity_metrics(results: AnalysisResults):
    """Display similarity metrics in the analysis page."""
    st.header("Content Similarity Analysis")
    display_similarity_summary(results)

def display_similarity_summary(results: AnalysisResults):
    """Display a summary of similarity metrics in the home page."""
    if not results.similarity_metrics:
        st.info("Similarity metrics not available")
        return
    
    # Create a bar chart for similarity distribution
    similarity_data = {
        'Category': ['Total Pairs', 'High Similarity', 'Cross-Department'],
        'Count': [
            results.similarity_metrics.total_pairs,
            results.similarity_metrics.high_similarity_pairs,
            results.similarity_metrics.cross_department_pairs
        ]
    }
    
    fig = px.bar(
        similarity_data,
        x='Category',
        y='Count',
        title="Similarity Analysis Overview"
    )
    
    fig.update_layout(height=200, margin={'l': 0, 'r': 0, 't': 30, 'b': 0})
    st.plotly_chart(fig, use_container_width=True)
    
    # Display key metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "High Similarity Pairs",
            f"{results.similarity_metrics.high_similarity_pairs:,}"
        )
    
    with col2:
        st.metric(
            "Average Similarity",
            f"{results.similarity_metrics.average_similarity:.1%}"
        )
    
    # Display warning if many high similarity pairs
    if results.similarity_metrics.high_similarity_pairs > 5:
        st.warning(
            f"Found {results.similarity_metrics.high_similarity_pairs} highly similar "
            "course pairs. Consider reviewing for potential consolidation."
        ) 