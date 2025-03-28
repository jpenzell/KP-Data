"""Activity metrics component for the LMS analyzer."""

import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, Any
import plotly.express as px
import plotly.graph_objects as go

from src.models.data_models import AnalysisResults, ActivityMetrics
from src.config.analysis_params import ACTIVITY_THRESHOLDS, VIZ_PARAMS

def display_activity_metrics(results: AnalysisResults) -> None:
    """Display activity metrics and visualizations."""
    if not results.activity_metrics:
        st.warning("No activity metrics available.")
        return
    
    metrics = results.activity_metrics
    st.header("📊 Activity Analysis")
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Courses", metrics.active_courses)
    with col2:
        st.metric("Recent Completions", metrics.recent_completions)
    with col3:
        completion_rate = metrics.average_completion_rate
        st.metric(
            "Average Completion Rate", 
            f"{completion_rate:.1%}" if completion_rate is not None else "N/A"
        )
    
    # Show last activity date if available
    if hasattr(metrics, 'last_activity_date') and metrics.last_activity_date:
        st.info(f"Last activity: {metrics.last_activity_date.strftime('%Y-%m-%d')}")
    
    # Activity trend visualization
    if hasattr(metrics, 'activity_trend') and metrics.activity_trend:
        st.subheader("Activity Trend")
        df = pd.DataFrame({
            'Date': list(metrics.activity_trend.keys()),
            'Activities': list(metrics.activity_trend.values())
        })
        if not df.empty:
            fig = px.line(
                df,
                x='Date',
                y='Activities',
                title="Course Activity Over Time",
                markers=True
            )
            fig.update_layout(
                height=VIZ_PARAMS.get("chart_height", 400),
                width=VIZ_PARAMS.get("chart_width", 800)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Activity distribution if available 
    if hasattr(metrics, 'activity_distribution') and metrics.activity_distribution:
        st.subheader("Activity Distribution")
        dist_df = pd.DataFrame({
            'Category': list(metrics.activity_distribution.keys()),
            'Count': list(metrics.activity_distribution.values())
        })
        if not dist_df.empty:
            st.bar_chart(dist_df.set_index('Category'))
    
    # Recent activities if available
    if hasattr(metrics, 'recent_activities') and metrics.recent_activities:
        st.subheader("Recent Activities")
        act_df = pd.DataFrame(metrics.recent_activities)
        if not act_df.empty:
            st.dataframe(act_df)
    
    # Activity recommendations if available
    if hasattr(metrics, 'activity_recommendations') and metrics.activity_recommendations:
        st.subheader("Activity Recommendations")
        for rec in metrics.activity_recommendations:
            st.markdown(f"- {rec}")
    
    # Activity insights
    st.subheader("Activity Insights")
    if metrics.last_activity_date:
        days_since_last = (pd.Timestamp.now() - metrics.last_activity_date).days
        if days_since_last > ACTIVITY_THRESHOLDS["recent_days"]:
            st.warning(f"⚠️ No recent activity detected in the last {days_since_last} days")
        else:
            st.success(f"✅ Active with recent updates ({days_since_last} days ago)") 