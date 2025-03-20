"""
Content similarity analysis utilities for the LMS Analyzer application.

This module provides advanced NLP-based techniques for analyzing content similarity,
identifying potential duplicates, and visualizing content overlap between courses.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import re
import streamlit as st
import math

@st.cache_data(ttl=3600)
def preprocess_text(text: str) -> str:
    """
    Preprocess text for similarity analysis by removing special characters,
    converting to lowercase, and removing extra whitespace.
    
    Args:
        text: The text to preprocess
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@st.cache_data(ttl=3600)
def compute_content_similarity(df: pd.DataFrame, 
                             text_columns: List[str] = ['course_title', 'course_description', 'course_abstract', 'course_keywords'], 
                             min_similarity: float = 0.7,  # Increased from 0.5 to 0.7 for better performance
                             max_results: int = 1000,
                             use_progress_bar: bool = True) -> pd.DataFrame:
    """
    Compute similarity between courses based on text content.
    
    Args:
        df: DataFrame containing course data
        text_columns: List of columns to use for similarity calculation
        min_similarity: Minimum similarity threshold
        max_results: Maximum number of pairs to return
        use_progress_bar: Whether to display a progress bar
        
    Returns:
        DataFrame with similar course pairs
    """
    print(f"Computing content similarity using columns: {text_columns}")
    print(f"Using minimum similarity threshold: {min_similarity}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Ensure course_id is string type to prevent comparison issues
    if 'course_id' in df.columns:
        df['course_id'] = df['course_id'].astype(str)
    
    # Prepare a combined text column for similarity analysis
    df['combined_text'] = ''
    
    # Weight different text columns differently
    weights = {
        'course_title': 3.0,  # Title is most important
        'course_description': 2.0,  # Description is next
        'course_abstract': 1.5,
        'course_keywords': 1.0
    }
    
    # Create weighted combined text
    for col in text_columns:
        if col in df.columns:
            # Check if column is categorical and convert to string if needed
            if pd.api.types.is_categorical_dtype(df[col]):
                print(f"Converting categorical column {col} to string")
                df[col] = df[col].astype(str)
            
            # Apply the weight by repeating the text
            weight = weights.get(col, 1.0)
            
            # For title, repeat it based on weight
            if col == 'course_title':
                # Handle non-string columns safely
                df['combined_text'] += df[col].fillna('').astype(str).apply(
                    lambda x: (preprocess_text(x) + ' ') * int(weight)
                )
            else:
                # For other columns, just add the text
                df['combined_text'] += df[col].fillna('').astype(str).apply(
                    lambda x: preprocess_text(x) + ' '
                )
    
    # Remove rows with empty combined text
    df_valid = df[df['combined_text'].str.strip() != ''].copy()
    
    if len(df_valid) < 2:
        print("Insufficient data for similarity analysis")
        return pd.DataFrame()
    
    # Use TF-IDF to convert text to numerical vectors
    print(f"Converting text to TF-IDF vectors for {len(df_valid)} courses")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    try:
        tfidf_matrix = tfidf_vectorizer.fit_transform(df_valid['combined_text'])
    except Exception as e:
        print(f"Error creating TF-IDF matrix: {str(e)}")
        return pd.DataFrame()
    
    # Compute cosine similarity
    print("Computing cosine similarity with optimized chunking")
    
    # More efficient chunking - smaller chunks and avoid redundant comparisons
    chunk_size = 500  # Reduced from 1000 to 500 for better memory management
    similar_pairs = []
    
    try:
        # Set up progress bar if requested
        total_chunks = (len(df_valid) + chunk_size - 1) // chunk_size
        total_comparisons = total_chunks * (total_chunks + 1) // 2  # Sum of 1 to n
        
        if use_progress_bar:
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            progress_count = 0
        
        for i in range(0, len(df_valid), chunk_size):
            i_end = min(i + chunk_size, len(df_valid))
            chunk_i = tfidf_matrix[i:i_end]
            
            # Only compute similarities with chunks j where j >= i to avoid redundancy
            for j in range(i, len(df_valid), chunk_size):
                j_end = min(j + chunk_size, len(df_valid))
                chunk_j = tfidf_matrix[j:j_end]
                
                # Update progress
                if use_progress_bar:
                    progress_count += 1
                    progress_bar.progress(progress_count / total_comparisons)
                    status_text.text(f"Comparing chunk {progress_count}/{total_comparisons}...")
                
                # Compute similarity between chunks
                chunk_sim = cosine_similarity(chunk_i, chunk_j)
                
                # Extract similar pairs
                for row_idx in range(chunk_sim.shape[0]):
                    course_i_idx = i + row_idx
                    course_i_id = df_valid.iloc[course_i_idx]['course_id']
                    
                    # Get column indices of similar courses
                    # Only look at upper triangular part when i==j to avoid duplicates
                    start_col = row_idx + 1 if i == j else 0
                    
                    for col_idx in range(start_col, chunk_sim.shape[1]):
                        similarity = chunk_sim[row_idx, col_idx]
                        
                        # Skip if below threshold
                        if similarity < min_similarity:
                            continue
                            
                        course_j_idx = j + col_idx
                        
                        # Skip self-similarity
                        if course_i_idx == course_j_idx:
                            continue
                            
                        course_j_id = df_valid.iloc[course_j_idx]['course_id']
                        
                        # Add to similar pairs
                        similar_pairs.append((
                            str(course_i_id), 
                            str(course_j_id), 
                            float(similarity)
                        ))
        
        # Clear progress
        if use_progress_bar:
            status_text.empty()
            progress_bar.empty()
            
    except Exception as e:
        print(f"Error computing content similarity: {str(e)}")
        import traceback
        traceback.print_exc()
        if use_progress_bar:
            st.error(f"Error during similarity computation: {str(e)}")
        return pd.DataFrame()
    
    # Create DataFrame from pairs
    if similar_pairs:
        try:
            print(f"Found {len(similar_pairs)} similar pairs. Creating similarity DataFrame...")
            similar_df = pd.DataFrame(similar_pairs, columns=['course_id_1', 'course_id_2', 'similarity_score'])
            
            # Sort by similarity score (highest first)
            similar_df = similar_df.sort_values('similarity_score', ascending=False)
            
            # Limit number of results
            similar_df = similar_df.head(max_results)
            
            # Add titles and other metadata for easier interpretation
            print("Adding metadata to similarity pairs...")
            id_to_title = df.set_index('course_id')['course_title'].astype(str).to_dict()
            
            # Handle category_name which might be categorical
            if 'category_name' in df.columns:
                if pd.api.types.is_categorical_dtype(df['category_name']):
                    id_to_category = df.set_index('course_id')['category_name'].astype(str).to_dict()
                else:
                    id_to_category = df.set_index('course_id')['category_name'].astype(str).to_dict()
            else:
                id_to_category = {}
            
            # Handle sponsoring_dept which might be categorical
            if 'sponsoring_dept' in df.columns:
                if pd.api.types.is_categorical_dtype(df['sponsoring_dept']):
                    id_to_dept = df.set_index('course_id')['sponsoring_dept'].astype(str).to_dict()
                else:
                    id_to_dept = df.set_index('course_id')['sponsoring_dept'].astype(str).to_dict()
            else:
                id_to_dept = {}
            
            # Map values safely
            similar_df['title_1'] = similar_df['course_id_1'].map(lambda x: id_to_title.get(x, "Unknown"))
            similar_df['title_2'] = similar_df['course_id_2'].map(lambda x: id_to_title.get(x, "Unknown"))
            
            if id_to_category:
                similar_df['category_1'] = similar_df['course_id_1'].map(lambda x: id_to_category.get(x, "Unknown"))
                similar_df['category_2'] = similar_df['course_id_2'].map(lambda x: id_to_category.get(x, "Unknown"))
            
            if id_to_dept:
                similar_df['dept_1'] = similar_df['course_id_1'].map(lambda x: id_to_dept.get(x, "Unknown"))
                similar_df['dept_2'] = similar_df['course_id_2'].map(lambda x: id_to_dept.get(x, "Unknown"))
                
                # Add flag for cross-department similarity - ensure string comparison
                similar_df['is_cross_dept'] = (similar_df['dept_1'] != similar_df['dept_2']) & \
                                             (~similar_df['dept_1'].isna()) & \
                                             (~similar_df['dept_2'].isna())
            
            print(f"Found {len(similar_df)} similar course pairs with similarity ≥ {min_similarity}")
            return similar_df
        except Exception as e:
            print(f"Error creating similarity DataFrame: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    else:
        print("No similar course pairs found")
        return pd.DataFrame()

def find_potential_duplicates(similarity_df: pd.DataFrame, 
                            high_threshold: float = 0.8) -> pd.DataFrame:
    """
    Identify potential duplicate courses based on high similarity.
    
    Args:
        similarity_df: DataFrame with similarity scores
        high_threshold: Threshold for potential duplicates
        
    Returns:
        DataFrame with potential duplicate courses
    """
    if similarity_df.empty:
        return pd.DataFrame()
    
    # Ensure similarity_score is float type
    similarity_df = similarity_df.copy()
    similarity_df['similarity_score'] = similarity_df['similarity_score'].astype(float)
    
    # Filter for high similarity scores
    duplicates = similarity_df[similarity_df['similarity_score'] >= high_threshold].copy()
    
    if duplicates.empty:
        return pd.DataFrame()
    
    # Add columns to help with interpretation
    duplicates['recommendation'] = 'Review for potential consolidation'
    
    # If cross-department flag exists, add specific recommendations
    if 'is_cross_dept' in duplicates.columns:
        mask = duplicates['is_cross_dept']
        duplicates.loc[mask, 'recommendation'] = 'Cross-department duplicate - consider consolidation or collaboration'
    
    return duplicates

def plot_similarity_network(similarity_df: pd.DataFrame, 
                          min_nodes: int = 5, 
                          max_nodes: int = 50,
                          min_edge_weight: float = 0.6) -> go.Figure:
    """
    Create a network visualization of course similarities.
    
    Args:
        similarity_df: DataFrame with similarity data
        min_nodes: Minimum number of nodes to include
        max_nodes: Maximum number of nodes to show
        min_edge_weight: Minimum edge weight (similarity) to include
        
    Returns:
        Plotly figure with network visualization
    """
    if similarity_df.empty:
        # Return empty figure if no data
        return go.Figure()
    
    try:
        # Create a copy to avoid modifying the original
        similarity_df = similarity_df.copy()
        
        # Ensure similarity_score is float
        similarity_df['similarity_score'] = similarity_df['similarity_score'].astype(float)
        
        # Filter by minimum edge weight
        filtered_df = similarity_df[similarity_df['similarity_score'] >= min_edge_weight].copy()
        
        if filtered_df.empty:
            # If no edges meet criteria, lower the threshold
            min_edge_weight = similarity_df['similarity_score'].min()
            filtered_df = similarity_df.copy()
        
        # Get unique nodes (courses)
        course_ids = set()
        for col in ['course_id_1', 'course_id_2']:
            course_ids.update(filtered_df[col].astype(str).unique())
        
        # Count connections for each node
        node_connections = {}
        for node in course_ids:
            node_connections[node] = (
                (filtered_df['course_id_1'] == node).sum() + 
                (filtered_df['course_id_2'] == node).sum()
            )
        
        # Sort nodes by connection count
        sorted_nodes = sorted(node_connections.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to max_nodes, but ensure at least min_nodes
        if len(sorted_nodes) > max_nodes:
            top_nodes = [node for node, _ in sorted_nodes[:max_nodes]]
        else:
            top_nodes = [node for node, _ in sorted_nodes]
        
        # Filter edges to only include top nodes
        filtered_edges = filtered_df[
            (filtered_df['course_id_1'].isin(top_nodes)) & 
            (filtered_df['course_id_2'].isin(top_nodes))
        ].copy()
        
        # If we have title information, use it for node labels
        node_labels = {}
        if 'title_1' in filtered_edges.columns:
            # Create id to title mapping
            for i, row in filtered_edges.iterrows():
                node_labels[str(row['course_id_1'])] = str(row['title_1'])
                node_labels[str(row['course_id_2'])] = str(row['title_2'])
        else:
            # Use IDs as labels
            node_labels = {node: node for node in top_nodes}
        
        # Extract node and edge data
        G = nx.Graph()
        
        # Add nodes
        for node in top_nodes:
            G.add_node(str(node), name=node_labels.get(str(node), str(node)))
        
        # Add edges
        for _, row in filtered_edges.iterrows():
            G.add_edge(
                str(row['course_id_1']), 
                str(row['course_id_2']), 
                weight=float(row['similarity_score'])
            )
        
        # Calculate node positions
        pos = nx.spring_layout(G, k=1/math.sqrt(len(G.nodes())), iterations=50)
        
        # Create edge traces
        edge_trace = []
        
        # Create edges with varying widths based on weight
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 0.5)
            
            # Scale line width based on similarity
            width = 2 + 8 * (weight - min_edge_weight) / (1 - min_edge_weight)
            
            # Color scale based on weight
            color = f'rgba({int(255 * (1 - weight))}, {int(255 * weight)}, 120, 0.7)'
            
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    line=dict(width=width, color=color),
                    hoverinfo='none',
                    mode='lines'
                )
            )
        
        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(G.nodes[node]['name'])
            
            # Size node based on degree
            degree = G.degree(node)
            node_size.append(10 + 5 * degree)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode='markers',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='YlGnBu',
                size=node_size,
                color=[G.degree(node) for node in G.nodes()],
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)
            )
        )
        
        # Create figure
        fig = go.Figure(
            data=edge_trace + [node_trace],
            layout=go.Layout(
                title=f'Course Similarity Network (Similarity ≥ {min_edge_weight:.2f})',
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=600
            )
        )
        
        return fig
    
    except Exception as e:
        print(f"Error creating similarity network visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        return go.Figure()

def plot_department_similarity_heatmap(similarity_df: pd.DataFrame) -> go.Figure:
    """
    Create a heatmap showing content similarity between departments.
    
    Args:
        similarity_df: DataFrame with similarity data
        
    Returns:
        Plotly figure with heatmap
    """
    if similarity_df.empty or 'dept_1' not in similarity_df.columns or 'dept_2' not in similarity_df.columns:
        # Return empty figure if no proper data
        return go.Figure()
    
    # Group by departments and calculate average similarity
    dept_sim = similarity_df.groupby(['dept_1', 'dept_2'])['similarity_score'].agg(['mean', 'count']).reset_index()
    
    # Convert to pivot table for heatmap
    pivot_df = dept_sim.pivot_table(
        index='dept_1', 
        columns='dept_2', 
        values='mean',
        fill_value=0
    )
    
    # Ensure the matrix is symmetric by adding both directions
    all_depts = sorted(set(pivot_df.index) | set(pivot_df.columns))
    full_matrix = pd.DataFrame(index=all_depts, columns=all_depts, data=0.0)
    
    for i in all_depts:
        for j in all_depts:
            if i in pivot_df.index and j in pivot_df.columns and not pd.isna(pivot_df.loc[i, j]):
                full_matrix.loc[i, j] = pivot_df.loc[i, j]
            elif j in pivot_df.index and i in pivot_df.columns and not pd.isna(pivot_df.loc[j, i]):
                full_matrix.loc[i, j] = pivot_df.loc[j, i]
            elif i == j:
                full_matrix.loc[i, j] = 1.0  # Self-similarity
    
    # Create heatmap
    fig = px.imshow(
        full_matrix,
        labels=dict(x="Department", y="Department", color="Avg. Similarity"),
        x=full_matrix.columns,
        y=full_matrix.index,
        color_continuous_scale="Viridis",
        title="Content Similarity Between Departments"
    )
    
    fig.update_layout(
        height=600,
        width=800,
        xaxis=dict(tickangle=45),
    )
    
    return fig

def generate_consolidation_recommendations(similarity_df: pd.DataFrame,
                                        df: pd.DataFrame,
                                        high_threshold: float = 0.8,
                                        max_recommendations: int = 20) -> pd.DataFrame:
    """
    Generate recommendations for course consolidation based on similarity.
    
    Args:
        similarity_df: DataFrame with similarity data
        df: Original course data
        high_threshold: Threshold for consolidation recommendations
        max_recommendations: Maximum number of recommendations to return
        
    Returns:
        DataFrame with consolidation recommendations
    """
    if similarity_df.empty:
        return pd.DataFrame()
    
    # Create a copy of the dataframes to avoid modifying the originals
    similarity_df = similarity_df.copy()
    df = df.copy()
    
    # Ensure course_id is string type
    if 'course_id' in df.columns:
        df['course_id'] = df['course_id'].astype(str)
    
    # Ensure similarity_score is float
    similarity_df['similarity_score'] = similarity_df['similarity_score'].astype(float)
    
    # Convert all categorical columns to string
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)
    
    # Identify potential duplicates
    try:
        duplicates = find_potential_duplicates(similarity_df, high_threshold)
        
        if duplicates.empty:
            return pd.DataFrame()
            
        # Prepare recommendations dataframe
        recommendations = duplicates.copy()
        
        # Add more metadata to help with decision making
        if 'total_2024_activity' in df.columns:
            # Create dictionaries mapping course IDs to activity
            id_to_activity = df.set_index('course_id')['total_2024_activity'].fillna(0).to_dict()
            
            # Add activity data for both courses in each pair
            recommendations['activity_1'] = recommendations['course_id_1'].map(
                lambda x: id_to_activity.get(x, 0)
            )
            recommendations['activity_2'] = recommendations['course_id_2'].map(
                lambda x: id_to_activity.get(x, 0)
            )
            
            # Calculate activity difference and recommend keeping the more active course
            recommendations['activity_diff'] = recommendations['activity_1'] - recommendations['activity_2']
            recommendations['keep_course'] = recommendations.apply(
                lambda row: str(row['course_id_1']) if row['activity_1'] >= row['activity_2'] else str(row['course_id_2']),
                axis=1
            )
            recommendations['consolidate_course'] = recommendations.apply(
                lambda row: str(row['course_id_2']) if row['activity_1'] >= row['activity_2'] else str(row['course_id_1']),
                axis=1
            )
        else:
            # If no activity data, just make a placeholder recommendation
            recommendations['keep_course'] = recommendations['course_id_1']
            recommendations['consolidate_course'] = recommendations['course_id_2']
        
        # Add specific recommendation reasoning
        if 'is_cross_dept' in recommendations.columns:
            # Different recommendations for cross-department duplicates
            cross_dept_mask = recommendations['is_cross_dept']
            recommendations.loc[cross_dept_mask, 'recommendation_detail'] = 'Cross-department content overlap - consider collaboration'
            recommendations.loc[~cross_dept_mask, 'recommendation_detail'] = 'Content duplicate - consider consolidation'
        else:
            recommendations['recommendation_detail'] = 'Similar content - consider consolidation'
        
        # Sort by similarity score (highest first)
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        # Limit number of recommendations
        recommendations = recommendations.head(max_recommendations)
        
        return recommendations
        
    except Exception as e:
        print(f"Error generating consolidation recommendations: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def analyze_content_similarity(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Complete content similarity analysis pipeline.
    
    Args:
        df: DataFrame with course data
        
    Returns:
        Dictionary with similarity analysis results
    """
    results = {
        'figures': {},
        'metrics': {},
        'tables': {}
    }
    
    # Create a safe copy of the dataframe
    df = df.copy()
    
    # Ensure required columns exist
    required_columns = ['course_id', 'course_title']
    if not all(col in df.columns for col in required_columns):
        print("Missing required columns for content similarity analysis")
        return results
    
    # Ensure course_id is string type to prevent comparison issues
    if 'course_id' in df.columns:
        df['course_id'] = df['course_id'].astype(str)
    
    # Convert any categorical columns to string
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            print(f"Converting categorical column {col} to string in analysis function")
            df[col] = df[col].astype(str)
    
    # Sample large datasets to improve performance
    dataset_size = len(df)
    max_sample_size = 10000  # Maximum courses to analyze for performance
    
    if dataset_size > max_sample_size:
        st.warning(f"""
        Large dataset detected ({dataset_size:,} courses). 
        Sampling {max_sample_size:,} courses for similarity analysis to improve performance.
        To analyze all courses, please reduce the dataset size with filters first.
        """)
        
        # Try to take a stratified sample by department or category if available
        try:
            if 'sponsoring_dept' in df.columns and df['sponsoring_dept'].nunique() > 1:
                print(f"Taking stratified sample by department from {dataset_size} courses")
                # Get departments with largest number of courses
                top_depts = df['sponsoring_dept'].value_counts().nlargest(20).index
                dept_sample = df[df['sponsoring_dept'].isin(top_depts)]
                
                if len(dept_sample) > max_sample_size // 2:
                    # Sample from each department
                    sampled_df = dept_sample.groupby('sponsoring_dept', group_keys=False).apply(
                        lambda x: x.sample(min(max(50, max_sample_size // len(top_depts)), len(x)))
                    )
                    # Add some random courses to ensure diversity
                    remaining = df[~df.index.isin(sampled_df.index)]
                    if len(remaining) > 0 and len(sampled_df) < max_sample_size:
                        random_sample = remaining.sample(min(max_sample_size - len(sampled_df), len(remaining)))
                        sampled_df = pd.concat([sampled_df, random_sample])
                else:
                    # If the total is small, just use the top departments
                    sampled_df = dept_sample
            elif 'category_name' in df.columns and df['category_name'].nunique() > 1:
                print(f"Taking stratified sample by category from {dataset_size} courses")
                # Sample by category
                sampled_df = df.groupby('category_name', group_keys=False).apply(
                    lambda x: x.sample(min(max(50, max_sample_size // df['category_name'].nunique()), len(x)))
                )
                # Ensure we have at most max_sample_size
                if len(sampled_df) > max_sample_size:
                    sampled_df = sampled_df.sample(max_sample_size)
            else:
                # Random sample if no stratification column
                print(f"Taking random sample from {dataset_size} courses")
                sampled_df = df.sample(max_sample_size)
            
            print(f"Analyzing a sample of {len(sampled_df)} courses for similarity")
        except Exception as e:
            print(f"Error during sampling: {str(e)}. Falling back to random sample.")
            sampled_df = df.sample(max_sample_size)
    else:
        # Use the full dataset if it's small enough
        sampled_df = df
        
    # Compute text similarity
    try:
        with st.spinner("Computing content similarity... This may take a few minutes."):
            similarity_df = compute_content_similarity(sampled_df)
    except Exception as e:
        print(f"Error computing content similarity: {str(e)}")
        import traceback
        traceback.print_exc()
        st.error(f"Error computing similarity: {str(e)}")
        return results
    
    if not similarity_df.empty:
        # Store similarity data
        results['tables']['similarity_pairs'] = similarity_df
        
        # Find potential duplicates
        try:
            duplicates = find_potential_duplicates(similarity_df)
            if not duplicates.empty:
                results['tables']['potential_duplicates'] = duplicates
                
                # Generate consolidation recommendations
                try:
                    recommendations = generate_consolidation_recommendations(similarity_df, sampled_df)
                    if not recommendations.empty:
                        results['tables']['consolidation_recommendations'] = recommendations
                except Exception as e:
                    print(f"Error generating consolidation recommendations: {str(e)}")
        except Exception as e:
            print(f"Error finding potential duplicates: {str(e)}")
        
        # Create network visualization
        try:
            with st.spinner("Creating similarity network visualization..."):
                network_fig = plot_similarity_network(similarity_df)
                results['figures']['similarity_network'] = network_fig
        except Exception as e:
            print(f"Error creating network visualization: {str(e)}")
        
        # If department data exists, create department similarity heatmap
        if 'dept_1' in similarity_df.columns and 'dept_2' in similarity_df.columns:
            try:
                with st.spinner("Creating department similarity heatmap..."):
                    dept_fig = plot_department_similarity_heatmap(similarity_df)
                    results['figures']['department_similarity'] = dept_fig
            except Exception as e:
                print(f"Error creating department heatmap: {str(e)}")
        
        # Calculate metrics
        try:
            # Ensure similarity_score is float
            similarity_df['similarity_score'] = similarity_df['similarity_score'].astype(float)
            
            high_similarity_pairs = len(similarity_df[similarity_df['similarity_score'] >= 0.8])
            medium_similarity_pairs = len(similarity_df[(similarity_df['similarity_score'] >= 0.6) & 
                                                     (similarity_df['similarity_score'] < 0.8)])
            
            results['metrics']['similarity_counts'] = {
                'total_pairs': len(similarity_df),
                'high_similarity': high_similarity_pairs,
                'medium_similarity': medium_similarity_pairs,
                'potential_duplicates': len(duplicates) if 'potential_duplicates' in results['tables'] else 0
            }
            
            # Calculate cross-department similarity if possible
            if 'is_cross_dept' in similarity_df.columns:
                cross_dept_pairs = similarity_df['is_cross_dept'].sum()
                results['metrics']['cross_department'] = {
                    'total_cross_dept_pairs': cross_dept_pairs,
                    'percent_cross_dept': (cross_dept_pairs / len(similarity_df) * 100) if len(similarity_df) > 0 else 0
                }
                
            # Add sampling metadata to results
            if dataset_size > max_sample_size:
                results['metadata'] = {
                    'sampled': True,
                    'original_size': dataset_size,
                    'sample_size': len(sampled_df),
                    'sample_method': 'stratified' if 'sponsoring_dept' in df.columns or 'category_name' in df.columns else 'random'
                }
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
    
    return results 