"""Visualization service for the LMS analyzer."""

import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import pandas as pd

from ..config.analysis_params import VIZ_PARAMS

class VisualizationService:
    """Service for generating visualizations."""
    
    def create_quality_chart(self, metrics: Any) -> go.Figure:
        """Create quality metrics chart."""
        fig = go.Figure()
        
        # Add radar chart
        fig.add_trace(go.Scatterpolar(
            r=[
                metrics.completeness_score,
                metrics.metadata_score,
                metrics.content_score,
                metrics.overall_score
            ],
            theta=[
                'Completeness',
                'Metadata',
                'Content',
                'Overall'
            ],
            fill='toself',
            name='Quality Scores'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            title="Quality Metrics Overview",
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig
    
    def create_activity_chart(self, metrics: Any) -> go.Figure:
        """Create activity metrics chart."""
        if not metrics.activity_trend:
            return None
        
        fig = px.line(
            x=list(metrics.activity_trend.keys()),
            y=list(metrics.activity_trend.values()),
            title="Course Activity Over Time",
            labels={"x": "Date", "y": "Number of Activities"}
        )
        
        fig.update_layout(
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig
    
    def create_similarity_chart(self, metrics: Any) -> go.Figure:
        """Create similarity metrics chart."""
        if not metrics.similarity_distribution:
            return None
        
        fig = px.bar(
            x=list(metrics.similarity_distribution.keys()),
            y=list(metrics.similarity_distribution.values()),
            title="Similarity Distribution",
            labels={"x": "Similarity Range", "y": "Number of Pairs"}
        )
        
        fig.update_layout(
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig
    
    def create_category_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create category distribution chart."""
        category_counts = df['category_name'].value_counts()
        
        fig = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Course Distribution by Category"
        )
        
        fig.update_layout(
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig
    
    def create_department_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create department distribution chart."""
        dept_counts = df['region_entity'].value_counts()
        
        fig = px.bar(
            x=dept_counts.index,
            y=dept_counts.values,
            title="Course Distribution by Department",
            labels={"x": "Department", "y": "Number of Courses"}
        )
        
        fig.update_layout(
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig
    
    def create_activity_level_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create activity level distribution chart."""
        if 'activity_level' not in df.columns:
            return None
        
        activity_counts = df['activity_level'].value_counts()
        
        fig = px.pie(
            values=activity_counts.values,
            names=activity_counts.index,
            title="Course Distribution by Activity Level"
        )
        
        fig.update_layout(
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig
    
    def create_quality_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create quality score distribution chart."""
        if 'quality_score' not in df.columns:
            return None
        
        fig = px.histogram(
            df,
            x='quality_score',
            title="Distribution of Quality Scores",
            labels={"x": "Quality Score", "y": "Number of Courses"}
        )
        
        fig.update_layout(
            height=VIZ_PARAMS["chart_height"],
            width=VIZ_PARAMS["chart_width"]
        )
        
        return fig 