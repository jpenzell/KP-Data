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
                             min_similarity: float = 0.5,
                             max_results: int = 1000) -> pd.DataFrame:
    """
    Compute similarity between courses based on text content.
    
    Args:
        df: DataFrame containing course data
        text_columns: List of columns to use for similarity calculation
        min_similarity: Minimum similarity threshold
        max_results: Maximum number of pairs to return
        
    Returns:
        DataFrame with similar course pairs
    """
    print(f"Computing content similarity using columns: {text_columns}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
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
    print("Computing cosine similarity")
    
    # Process in chunks to avoid memory issues with large datasets
    chunk_size = 1000
    similar_pairs = []
    
    for i in range(0, len(df_valid), chunk_size):
        end = min(i + chunk_size, len(df_valid))
        chunk = tfidf_matrix[i:end]
        
        # Compute similarity between this chunk and all other courses
        chunk_sim = cosine_similarity(chunk, tfidf_matrix)
        
        # Extract similar pairs
        for j in range(chunk_sim.shape[0]):
            course_idx = i + j
            course_id = df_valid.iloc[course_idx]['course_id']
            
            # Get indices of similar courses (excluding self)
            similar_indices = np.where(chunk_sim[j] >= min_similarity)[0]
            
            for sim_idx in similar_indices:
                # Skip self-similarity
                if sim_idx == course_idx:
                    continue
                
                sim_id = df_valid.iloc[sim_idx]['course_id']
                similarity = chunk_sim[j, sim_idx]
                
                # Only include pairs where first ID < second ID to avoid duplicates
                if course_id < sim_id:
                    similar_pairs.append((course_id, sim_id, similarity))
    
    # Create DataFrame from pairs
    if similar_pairs:
        similar_df = pd.DataFrame(similar_pairs, columns=['course_id_1', 'course_id_2', 'similarity_score'])
        
        # Sort by similarity score (highest first)
        similar_df = similar_df.sort_values('similarity_score', ascending=False)
        
        # Limit number of results
        similar_df = similar_df.head(max_results)
        
        # Add titles and other metadata for easier interpretation
        id_to_title = df.set_index('course_id')['course_title'].astype(str).to_dict()
        
        # Handle category_name which might be categorical
        if 'category_name' in df.columns:
            if pd.api.types.is_categorical_dtype(df['category_name']):
                id_to_category = df.set_index('course_id')['category_name'].astype(str).to_dict()
            else:
                id_to_category = df.set_index('course_id')['category_name'].to_dict()
        else:
            id_to_category = {}
        
        # Handle sponsoring_dept which might be categorical
        if 'sponsoring_dept' in df.columns:
            if pd.api.types.is_categorical_dtype(df['sponsoring_dept']):
                id_to_dept = df.set_index('course_id')['sponsoring_dept'].astype(str).to_dict()
            else:
                id_to_dept = df.set_index('course_id')['sponsoring_dept'].to_dict()
        else:
            id_to_dept = {}
        
        similar_df['title_1'] = similar_df['course_id_1'].map(id_to_title)
        similar_df['title_2'] = similar_df['course_id_2'].map(id_to_title)
        
        if id_to_category:
            similar_df['category_1'] = similar_df['course_id_1'].map(id_to_category)
            similar_df['category_2'] = similar_df['course_id_2'].map(id_to_category)
        
        if id_to_dept:
            similar_df['dept_1'] = similar_df['course_id_1'].map(id_to_dept)
            similar_df['dept_2'] = similar_df['course_id_2'].map(id_to_dept)
            
            # Add flag for cross-department similarity
            similar_df['is_cross_dept'] = (similar_df['dept_1'] != similar_df['dept_2']) & \
                                         (~similar_df['dept_1'].isna()) & \
                                         (~similar_df['dept_2'].isna())
        
        print(f"Found {len(similar_df)} similar course pairs with similarity ≥ {min_similarity}")
        return similar_df
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
                          min_threshold: float = 0.7,
                          max_nodes: int = 100) -> go.Figure:
    """
    Create a network visualization of course similarity.
    
    Args:
        similarity_df: DataFrame with similarity data
        min_threshold: Minimum similarity threshold for visualization
        max_nodes: Maximum number of nodes to include
        
    Returns:
        Plotly figure with network graph
    """
    if similarity_df.empty:
        # Return empty figure if no data
        return go.Figure()
    
    # Filter by threshold
    filtered_df = similarity_df[similarity_df['similarity_score'] >= min_threshold].copy()
    
    if filtered_df.empty:
        return go.Figure()
    
    # Create a graph
    G = nx.Graph()
    
    # Add edges with similarity as weight
    for _, row in filtered_df.iterrows():
        G.add_edge(
            row['course_id_1'], 
            row['course_id_2'], 
            weight=row['similarity_score'],
            title1=row['title_1'],
            title2=row['title_2']
        )
    
    # Limit to largest connected component if too many nodes
    if len(G.nodes) > max_nodes:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
    
    # Still limit to max_nodes if needed
    if len(G.nodes) > max_nodes:
        # Calculate node importance (degree centrality)
        centrality = nx.degree_centrality(G)
        # Sort nodes by centrality
        important_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        # Create subgraph with important nodes
        G = G.subgraph([node for node, _ in important_nodes]).copy()
    
    # Calculate layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    edge_text = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Add edge attributes to hover text
        similarity = G.edges[edge]['weight']
        title1 = G.edges[edge]['title1']
        title2 = G.edges[edge]['title2']
        edge_text.append(f"Similarity: {similarity:.2f}<br>{title1} - {title2}")
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text
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
        
        # Get node's titles from connected edges
        titles = set()
        for edge in G.edges(node):
            if edge[0] == node:
                titles.add(G.edges[edge]['title1'])
            else:
                titles.add(G.edges[edge]['title2'])
        
        # Use first title found
        title = list(titles)[0] if titles else "Unknown"
        node_text.append(f"ID: {node}<br>Title: {title}")
        
        # Node size based on degree
        node_size.append(5 + 3 * G.degree[node])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2
        ),
        text=node_text
    )
    
    # Color node points by the number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title=f'Course Similarity Network (Similarity ≥ {min_threshold:.1f})',
                      titlefont_size=16,
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                  ))
    
    return fig

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
    
    # Create a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Convert any categorical columns to string to avoid issues
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]):
            df[col] = df[col].astype(str)
    
    # Identify potential duplicates
    duplicates = find_potential_duplicates(similarity_df, high_threshold)
    
    if duplicates.empty:
        return pd.DataFrame()
    
    # Get more context for better recommendations
    if 'course_version' in df.columns and 'course_available_from' in df.columns:
        # Map creation dates
        id_to_date = df.set_index('course_id')['course_available_from'].to_dict()
        id_to_version = df.set_index('course_id')['course_version'].to_dict()
        
        duplicates['date_1'] = duplicates['course_id_1'].map(id_to_date)
        duplicates['date_2'] = duplicates['course_id_2'].map(id_to_date)
        duplicates['version_1'] = duplicates['course_id_1'].map(id_to_version)
        duplicates['version_2'] = duplicates['course_id_2'].map(id_to_version)
        
        # Determine which course is newer
        duplicates['newer_course'] = duplicates.apply(
            lambda x: x['course_id_1'] if x['date_1'] > x['date_2'] else x['course_id_2']
            if not pd.isna(x['date_1']) and not pd.isna(x['date_2']) else 'Unknown',
            axis=1
        )
        
        # Refine recommendations based on dates
        duplicates['recommendation'] = duplicates.apply(
            lambda x: f"Keep {x['title_1'] if x['newer_course'] == x['course_id_1'] else x['title_2']} (newer)"
            if x['newer_course'] != 'Unknown' else 'Review both for consolidation',
            axis=1
        )
    
    # Get learner count if available
    if 'learner_count' in df.columns:
        id_to_learners = df.set_index('course_id')['learner_count'].to_dict()
        duplicates['learners_1'] = duplicates['course_id_1'].map(id_to_learners)
        duplicates['learners_2'] = duplicates['course_id_2'].map(id_to_learners)
        
        # Consider high-enrollment courses in recommendations
        mask = (~duplicates['learners_1'].isna()) & (~duplicates['learners_2'].isna())
        if mask.any():
            duplicates.loc[mask, 'total_enrollment'] = duplicates.loc[mask, 'learners_1'] + duplicates.loc[mask, 'learners_2']
            
            # Prioritize high-enrollment duplicates
            duplicates = duplicates.sort_values('total_enrollment', ascending=False)
    else:
        # If no enrollment data, sort by similarity
        duplicates = duplicates.sort_values('similarity_score', ascending=False)
    
    # Limit to max recommendations
    duplicates = duplicates.head(max_recommendations)
    
    # Calculate potential savings
    if 'learners_1' in duplicates.columns and 'learners_2' in duplicates.columns:
        duplicates['potential_savings'] = duplicates.apply(
            lambda x: min(x['learners_1'], x['learners_2']) if not pd.isna(x['learners_1']) and not pd.isna(x['learners_2']) else 0,
            axis=1
        )
        duplicates['savings_impact'] = duplicates['potential_savings'].map(
            lambda x: 'High' if x > 100 else ('Medium' if x > 50 else 'Low')
        )
    
    return duplicates

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
    
    # Compute text similarity
    try:
        similarity_df = compute_content_similarity(df)
    except Exception as e:
        print(f"Error computing content similarity: {str(e)}")
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
                recommendations = generate_consolidation_recommendations(similarity_df, df)
                if not recommendations.empty:
                    results['tables']['consolidation_recommendations'] = recommendations
        except Exception as e:
            print(f"Error finding potential duplicates: {str(e)}")
        
        # Create network visualization
        try:
            network_fig = plot_similarity_network(similarity_df)
            results['figures']['similarity_network'] = network_fig
        except Exception as e:
            print(f"Error creating network visualization: {str(e)}")
        
        # If department data exists, create department similarity heatmap
        if 'dept_1' in similarity_df.columns and 'dept_2' in similarity_df.columns:
            try:
                dept_fig = plot_department_similarity_heatmap(similarity_df)
                results['figures']['department_similarity'] = dept_fig
            except Exception as e:
                print(f"Error creating department heatmap: {str(e)}")
        
        # Calculate metrics
        try:
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
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
    
    return results 