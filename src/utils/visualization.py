"""Utilities for creating visualizations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Union

from models.data_models import VisualizationConfig

def create_wordcloud(text: str, width: int = 800, height: int = 400) -> plt.Figure:
    """Create a wordcloud visualization from text."""
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(width/100, height/100))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_timeline(
    df: pd.DataFrame,
    date_col: str,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Create a timeline visualization for the specified date column."""
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        return None
    
    if config is None:
        config = VisualizationConfig(
            title=f'Distribution of {date_col}',
            x_label=date_col,
            y_label='Count',
            chart_type='histogram'
        )
    
    fig = px.histogram(
        df,
        x=date_col,
        title=config.title,
        labels={
            date_col: config.x_label,
            'count': config.y_label
        }
    )
    
    fig.update_layout(
        bargap=0.2,
        width=config.width,
        height=config.height
    )
    
    return fig

def plot_quality_distribution(
    df: pd.DataFrame,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Plot the distribution of quality scores."""
    if 'quality_score' not in df.columns:
        return None
    
    if config is None:
        config = VisualizationConfig(
            title='Distribution of Content Quality Scores',
            x_label='Quality Score',
            y_label='Number of Courses',
            chart_type='histogram'
        )
    
    fig = px.histogram(
        df,
        x='quality_score',
        nbins=20,
        title=config.title,
        labels={
            'quality_score': config.x_label,
            'count': config.y_label
        }
    )
    
    fig.add_vline(
        x=0.6,
        line_dash="dash",
        line_color="red",
        annotation_text="Attention Threshold"
    )
    
    fig.update_layout(
        width=config.width,
        height=config.height
    )
    
    return fig

def plot_category_distribution(
    data: Dict[str, float],
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Create a bar or pie chart of category distribution."""
    if config is None:
        config = VisualizationConfig(
            title='Distribution by Category',
            x_label='Category',
            y_label='Percentage',
            chart_type='bar'
        )
    
    df = pd.DataFrame(list(data.items()), columns=['Category', 'Value'])
    
    if config.chart_type == 'pie':
        fig = px.pie(
            df,
            values='Value',
            names='Category',
            title=config.title
        )
    else:
        fig = px.bar(
            df,
            x='Category',
            y='Value',
            title=config.title,
            labels={
                'Category': config.x_label,
                'Value': config.y_label
            }
        )
    
    fig.update_layout(
        width=config.width,
        height=config.height
    )
    
    return fig

def plot_trend_line(
    data: List[Dict],
    x_col: str,
    y_col: str,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Create a line plot for trend analysis."""
    if not data:
        return None
    
    if config is None:
        config = VisualizationConfig(
            title=f'Trend of {y_col} over {x_col}',
            x_label=x_col,
            y_label=y_col,
            chart_type='line'
        )
    
    df = pd.DataFrame(data)
    
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=config.title,
        labels={
            x_col: config.x_label,
            y_col: config.y_label
        }
    )
    
    if config.color_scheme:
        fig.update_traces(line_color=config.color_scheme)
    
    fig.update_layout(
        width=config.width,
        height=config.height
    )
    
    return fig

def plot_heatmap(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    value_col: str,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Create a heatmap visualization."""
    if config is None:
        config = VisualizationConfig(
            title=f'Heatmap of {value_col}',
            x_label=x_col,
            y_label=y_col,
            chart_type='heatmap'
        )
    
    pivot_table = df.pivot_table(
        values=value_col,
        index=y_col,
        columns=x_col,
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdBu'
    ))
    
    fig.update_layout(
        title=config.title,
        xaxis_title=config.x_label,
        yaxis_title=config.y_label,
        width=config.width,
        height=config.height
    )
    
    return fig

def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Create a scatter plot with optional color and size dimensions."""
    if config is None:
        config = VisualizationConfig(
            title=f'{y_col} vs {x_col}',
            x_label=x_col,
            y_label=y_col,
            chart_type='scatter'
        )
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        title=config.title,
        labels={
            x_col: config.x_label,
            y_col: config.y_label
        }
    )
    
    fig.update_layout(
        width=config.width,
        height=config.height
    )
    
    return fig

def plot_box(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    config: Optional[VisualizationConfig] = None
) -> go.Figure:
    """Create a box plot for distribution analysis."""
    if config is None:
        config = VisualizationConfig(
            title=f'Distribution of {value_col} by {group_col}',
            x_label=group_col,
            y_label=value_col,
            chart_type='box'
        )
    
    fig = px.box(
        df,
        x=group_col,
        y=value_col,
        title=config.title,
        labels={
            group_col: config.x_label,
            value_col: config.y_label
        }
    )
    
    fig.update_layout(
        width=config.width,
        height=config.height
    )
    
    return fig 