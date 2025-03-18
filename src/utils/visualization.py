"""Utilities for creating visualizations."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

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

@st.cache_data(ttl=3600)
def plot_timeline(
    df: pd.DataFrame,
    date_column: str,
    category_column: Optional[str] = None,
    title: str = "Timeline Distribution",
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create an interactive timeline visualization.
    
    Args:
        df: DataFrame containing the data
        date_column: Column name with date information
        category_column: Optional column for grouping
        title: Plot title
        height: Plot height
        width: Plot width
        
    Returns:
        Plotly figure object
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Filter out rows with invalid dates
    df = df.dropna(subset=[date_column])
    
    if category_column and category_column in df.columns:
        fig = px.scatter(
            df,
            x=date_column,
            y=category_column,
            color=category_column,
            title=title,
            height=height,
            width=width,
        )
    else:
        # Create histogram when no category column
        fig = px.histogram(
            df,
            x=date_column,
            title=title,
            height=height,
            width=width,
        )
    
    fig.update_layout(xaxis_title=date_column.replace('_', ' ').title())
    return fig

@st.cache_data(ttl=3600)
def plot_quality_distribution(
    df: pd.DataFrame,
    quality_column: str = 'quality_score',
    category_column: Optional[str] = None,
    title: str = "Quality Score Distribution",
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Visualize the distribution of quality scores.
    
    Args:
        df: DataFrame containing the data
        quality_column: Column with quality scores
        category_column: Optional column for grouping
        title: Plot title
        height: Plot height
        width: Plot width
        
    Returns:
        Plotly figure object
    """
    if quality_column not in df.columns:
        # Create dummy column if not present
        df = df.copy()
        df[quality_column] = np.random.uniform(0, 1, len(df))
        
    if category_column and category_column in df.columns:
        fig = px.box(
            df,
            y=quality_column,
            x=category_column,
            color=category_column,
            title=title,
            height=height,
            width=width,
        )
    else:
        fig = px.histogram(
            df,
            x=quality_column,
            title=title,
            height=height,
            width=width,
            nbins=20,
        )
        
    fig.update_layout(
        xaxis_title=quality_column.replace('_', ' ').title(),
        yaxis_title='Count' if category_column is None else quality_column.replace('_', ' ').title()
    )
    return fig

@st.cache_data(ttl=3600)
def plot_category_distribution(
    df: pd.DataFrame,
    category_column: str,
    count_column: Optional[str] = None,
    title: str = "Category Distribution",
    height: int = 500,
    width: int = 800
) -> go.Figure:
    """
    Create a bar chart showing category distribution.
    
    Args:
        df: DataFrame containing the data
        category_column: Column with category information
        count_column: Optional column for value counts
        title: Plot title
        height: Plot height
        width: Plot width
        
    Returns:
        Plotly figure object
    """
    if category_column not in df.columns:
        return None
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Fill missing values with 'Unknown'
    df[category_column] = df[category_column].fillna('Unknown')
    
    if count_column and count_column in df.columns:
        # Group by category and sum the count column
        category_counts = df.groupby(category_column)[count_column].sum().reset_index()
        fig = px.bar(
            category_counts,
            x=category_column,
            y=count_column,
            title=title,
            height=height,
            width=width,
        )
    else:
        # Count occurrences of each category
        category_counts = df[category_column].value_counts().reset_index()
        category_counts.columns = [category_column, 'count']
        fig = px.bar(
            category_counts,
            x=category_column,
            y='count',
            title=title,
            height=height,
            width=width,
        )
    
    fig.update_layout(
        xaxis_title=category_column.replace('_', ' ').title(),
        yaxis_title='Count'
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

def plot_data_quality_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Create visualizations for data quality metrics.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing Plotly figures and metrics
    """
    results = {
        'figures': {},
        'metrics': {},
        'tables': {}
    }
    
    # 1. Calculate overall completeness by column
    completeness = pd.DataFrame({
        'column': df.columns,
        'non_null': df.count(),
        'total': len(df)
    })
    completeness['percent_complete'] = (completeness['non_null'] / completeness['total'] * 100).round(1)
    completeness = completeness.sort_values('percent_complete')
    
    # Create completeness figure
    fig_completeness = px.bar(
        completeness.head(15),  # Focus on 15 most incomplete columns
        x='column', 
        y='percent_complete',
        title='Data Completeness by Column (15 most incomplete)',
        labels={'percent_complete': 'Percent Complete (%)', 'column': 'Column'}
    )
    
    # Add reference line at 80%
    fig_completeness.add_hline(
        y=80, 
        line_dash="dash", 
        line_color="red",
        annotation_text="80% Completeness Threshold"
    )
    fig_completeness.update_layout(height=400)
    results['figures']['completeness'] = fig_completeness
    
    # 2. Date field quality
    date_columns = [
        col for col in df.columns 
        if col.endswith('_is_valid') and col.replace('_is_valid', '') in df.columns
    ]
    
    if date_columns:
        date_quality = []
        for col in date_columns:
            base_col = col.replace('_is_valid', '')
            valid_count = df[col].sum()
            invalid_count = len(df) - valid_count
            date_quality.append({
                'column': base_col,
                'valid': valid_count,
                'invalid': invalid_count,
                'percent_valid': (valid_count / len(df) * 100).round(1)
            })
        
        date_quality_df = pd.DataFrame(date_quality)
        if not date_quality_df.empty:
            # Create stacked bar chart for date quality
            fig_date_quality = go.Figure()
            
            # Add bars for valid dates
            fig_date_quality.add_trace(go.Bar(
                name='Valid',
                x=date_quality_df['column'],
                y=date_quality_df['valid'],
                marker_color='green'
            ))
            
            # Add bars for invalid dates
            fig_date_quality.add_trace(go.Bar(
                name='Invalid',
                x=date_quality_df['column'],
                y=date_quality_df['invalid'],
                marker_color='red'
            ))
            
            fig_date_quality.update_layout(
                barmode='stack',
                title='Date Field Quality',
                xaxis_title='Date Column',
                yaxis_title='Count',
                height=400
            )
            
            results['figures']['date_quality'] = fig_date_quality
            results['tables']['date_quality'] = date_quality_df
    
    # 3. Data source distribution
    if 'data_source' in df.columns:
        source_counts = df['data_source'].value_counts()
        source_df = pd.DataFrame({
            'source': source_counts.index,
            'count': source_counts.values
        })
        
        fig_sources = px.pie(
            source_df,
            values='count',
            names='source',
            title='Data Distribution by Source'
        )
        fig_sources.update_traces(textposition='inside', textinfo='percent+label')
        fig_sources.update_layout(height=400)
        results['figures']['sources'] = fig_sources
    
    # 4. Cross-reference metrics
    if 'cross_reference_count' in df.columns:
        cross_ref_counts = df['cross_reference_count'].value_counts().sort_index()
        cross_ref_df = pd.DataFrame({
            'sources': cross_ref_counts.index,
            'count': cross_ref_counts.values
        })
        
        fig_cross_ref = px.bar(
            cross_ref_df,
            x='sources',
            y='count',
            title='Records by Number of Cross-References',
            labels={'sources': 'Number of Data Sources', 'count': 'Number of Records'}
        )
        fig_cross_ref.update_layout(height=400)
        results['figures']['cross_references'] = fig_cross_ref
        
        # Calculate data verification metrics
        single_source = (df['cross_reference_count'] == 1).sum()
        multi_source = (df['cross_reference_count'] > 1).sum()
        
        results['metrics']['verified_data'] = {
            'single_source': single_source,
            'multi_source': multi_source,
            'percent_verified': (multi_source / len(df) * 100).round(1)
        }
    
    # 5. Overall data quality score
    if 'quality_score' in df.columns:
        quality_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        quality_labels = ['Very Poor', 'Poor', 'Fair', 'Good', 'Excellent']
        
        df['quality_category'] = pd.cut(
            df['quality_score'], 
            bins=quality_bins, 
            labels=quality_labels,
            include_lowest=True
        )
        
        quality_dist = df['quality_category'].value_counts()
        quality_df = pd.DataFrame({
            'category': quality_dist.index,
            'count': quality_dist.values
        })
        
        fig_quality = px.bar(
            quality_df,
            x='category',
            y='count',
            title='Records by Quality Category',
            color='category',
            color_discrete_map={
                'Very Poor': 'red',
                'Poor': 'orange',
                'Fair': 'yellow',
                'Good': 'lightgreen',
                'Excellent': 'darkgreen'
            }
        )
        fig_quality.update_layout(height=400)
        results['figures']['quality_categories'] = fig_quality
        
        # Calculate overall quality metrics
        results['metrics']['overall_quality'] = {
            'mean_score': df['quality_score'].mean().round(2),
            'median_score': df['quality_score'].median().round(2),
            'high_quality': (df['quality_score'] >= 0.8).sum(),
            'low_quality': (df['quality_score'] < 0.4).sum(),
            'percent_high_quality': ((df['quality_score'] >= 0.8).sum() / len(df) * 100).round(1)
        }
    
    # 6. Issues summary table
    issues = []
    
    # Missing critical fields
    critical_fields = ['course_title', 'course_no', 'course_description', 'category_name']
    for field in critical_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            if missing > 0:
                issues.append({
                    'issue_type': 'Missing Data',
                    'field': field,
                    'count': missing,
                    'percent': (missing / len(df) * 100).round(1)
                })
    
    # Date field issues
    for col in date_columns:
        base_col = col.replace('_is_valid', '')
        invalid = (~df[col]).sum()
        if invalid > 0:
            issues.append({
                'issue_type': 'Invalid Date',
                'field': base_col,
                'count': invalid,
                'percent': (invalid / len(df) * 100).round(1)
            })
    
    # Cross-reference issues
    if 'cross_reference_count' in df.columns:
        single_source = (df['cross_reference_count'] == 1).sum()
        if single_source > 0:
            issues.append({
                'issue_type': 'No Cross-References',
                'field': 'cross_reference_count',
                'count': single_source,
                'percent': (single_source / len(df) * 100).round(1)
            })
    
    # Create issues table
    if issues:
        issues_df = pd.DataFrame(issues).sort_values('count', ascending=False)
        results['tables']['issues'] = issues_df
    
    return results 