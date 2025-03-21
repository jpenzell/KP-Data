"""
Enhanced content analysis utilities for the LMS Analyzer application.

This module provides advanced content analysis capabilities including:
- Semantic understanding of course content
- Pattern recognition in course structures
- Metadata quality scoring
- Cross-department content overlap analysis
- Local LLM integration for natural language insights
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import json
import os
from pathlib import Path

class ContentAnalyzer:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the content analyzer.
        
        Args:
            model_path: Optional path to a local LLM model
        """
        self.model_path = model_path
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
    def analyze_metadata_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the quality and completeness of course metadata.
        
        Args:
            df: DataFrame containing course data
            
        Returns:
            Dictionary with metadata quality analysis results
        """
        results = {
            'completeness': {},
            'patterns': {},
            'recommendations': []
        }
        
        # Define critical and optional fields
        critical_fields = ['course_title', 'course_description', 'sponsoring_dept', 'category_name']
        optional_fields = ['course_abstract', 'course_keywords', 'region_entity', 'quality_score']
        
        # Analyze completeness for each field
        for field in critical_fields + optional_fields:
            if field in df.columns:
                # Calculate completeness
                total = len(df)
                non_empty = df[field].notna().sum()
                non_empty_str = df[field].astype(str).str.strip().ne('').sum()
                
                # Calculate unique values
                unique_values = df[field].nunique()
                
                results['completeness'][field] = {
                    'total': total,
                    'non_empty': non_empty,
                    'non_empty_str': non_empty_str,
                    'completeness_rate': (non_empty_str / total) * 100,
                    'unique_values': unique_values
                }
                
                # Look for patterns in the data
                if field in critical_fields:
                    # Analyze value distribution
                    value_counts = df[field].value_counts().head(10)
                    results['patterns'][field] = {
                        'top_values': value_counts.to_dict(),
                        'has_duplicates': len(df[df.duplicated(subset=[field], keep=False)]) > 0
                    }
        
        # Generate recommendations based on analysis
        for field, stats in results['completeness'].items():
            if stats['completeness_rate'] < 90:
                results['recommendations'].append(
                    f"Improve {field} completeness (currently {stats['completeness_rate']:.1f}%)"
                )
            
            if field in critical_fields and stats['unique_values'] < 5:
                results['recommendations'].append(
                    f"Low variety in {field} (only {stats['unique_values']} unique values)"
                )
        
        return results
    
    def analyze_content_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze patterns in course content and structure.
        
        Args:
            df: DataFrame containing course data
            
        Returns:
            Dictionary with content pattern analysis results
        """
        results = {
            'content_clusters': {
                'total_clusters': 0,
                'cluster_sizes': {},
                'unclustered': 0
            },
            'keyword_analysis': {
                'top_keywords': {},
                'total_unique_terms': 0
            },
            'structure_patterns': {},
            'insights': []
        }
        
        # Prepare text content for analysis
        text_columns = ['course_title', 'course_description', 'course_abstract', 'course_keywords']
        
        # Check if all required columns exist
        missing_columns = [col for col in text_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns for content analysis: {missing_columns}")
            # Use only available columns
            text_columns = [col for col in text_columns if col in df.columns]
            if not text_columns:
                print("No text columns available for analysis")
                results['insights'].append("Content analysis skipped: No text columns available")
                return results
        
        # Create combined text from available columns
        df['combined_text'] = df[text_columns].fillna('').astype(str).apply(
            lambda x: ' '.join(x), axis=1
        )
        
        # Perform TF-IDF analysis
        try:
            tfidf_matrix = self.tfidf.fit_transform(df['combined_text'])
            
            # Cluster courses based on content similarity
            clustering = DBSCAN(eps=0.3, min_samples=5, metric='cosine')
            clusters = clustering.fit_predict(tfidf_matrix.toarray())
            
            # Analyze clusters
            cluster_sizes = pd.Series(clusters).value_counts()
            results['content_clusters'] = {
                'total_clusters': len(cluster_sizes),
                'cluster_sizes': cluster_sizes.to_dict(),
                'unclustered': int(cluster_sizes.get(-1, 0))
            }
            
            # Analyze keywords
            feature_names = self.tfidf.get_feature_names_out()
            tfidf_sum = tfidf_matrix.sum(axis=0).A1
            top_keywords = pd.Series(tfidf_sum, index=feature_names).nlargest(20)
            
            results['keyword_analysis'] = {
                'top_keywords': top_keywords.to_dict(),
                'total_unique_terms': len(feature_names)
            }
            
            # Look for structural patterns
            if 'course_no' in df.columns:
                course_patterns = df['course_no'].str.extract(r'([A-Za-z]+)(\d+)')
                results['structure_patterns']['course_numbering'] = {
                    'prefixes': course_patterns[0].value_counts().to_dict(),
                    'number_ranges': course_patterns[1].astype(float).describe().to_dict()
                }
            
            # Generate insights
            if results['content_clusters']['total_clusters'] > 0:
                results['insights'].append(
                    f"Found {results['content_clusters']['total_clusters']} distinct content clusters"
                )
            
            if results['keyword_analysis']['total_unique_terms'] > 1000:
                results['insights'].append(
                    f"High content diversity with {results['keyword_analysis']['total_unique_terms']} unique terms"
                )
            
        except Exception as e:
            print(f"Error in content pattern analysis: {str(e)}")
            results['insights'].append(f"Content analysis encountered an error: {str(e)}")
        
        return results
    
    def analyze_cross_department_overlap(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze content overlap between departments.
        
        Args:
            df: DataFrame containing course data
            
        Returns:
            Dictionary with cross-department analysis results
        """
        results = {
            'department_overlap': {},
            'content_bridges': [],
            'opportunities': []
        }
        
        if 'sponsoring_dept' not in df.columns:
            return results
        
        # Group courses by department
        dept_courses = df.groupby('sponsoring_dept')['combined_text'].apply(list)
        
        # Analyze overlap between departments
        for dept1 in dept_courses.index:
            for dept2 in dept_courses.index:
                if dept1 >= dept2:  # Avoid duplicate analysis
                    continue
                
                # Calculate similarity between department content
                dept1_text = ' '.join(dept_courses[dept1])
                dept2_text = ' '.join(dept_courses[dept2])
                
                # Use TF-IDF to compare departments
                tfidf_matrix = self.tfidf.fit_transform([dept1_text, dept2_text])
                similarity = (tfidf_matrix * tfidf_matrix.T).A[0, 1]
                
                results['department_overlap'][f"{dept1}-{dept2}"] = {
                    'similarity': float(similarity),
                    'dept1_courses': len(dept_courses[dept1]),
                    'dept2_courses': len(dept_courses[dept2])
                }
        
        # Identify content bridges (high overlap between departments)
        high_overlap = {
            k: v for k, v in results['department_overlap'].items()
            if v['similarity'] > 0.3
        }
        
        for dept_pair, stats in high_overlap.items():
            dept1, dept2 = dept_pair.split('-')
            results['content_bridges'].append({
                'departments': [dept1, dept2],
                'similarity': stats['similarity'],
                'course_counts': [stats['dept1_courses'], stats['dept2_courses']]
            })
        
        # Generate opportunities
        if results['content_bridges']:
            results['opportunities'].append(
                "Consider cross-department course consolidation or collaboration"
            )
        
        return results
    
    def generate_insights(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive insights about the course content.
        
        Args:
            df: DataFrame containing course data
            
        Returns:
            Dictionary with generated insights
        """
        # Run all analyses
        metadata_quality = self.analyze_metadata_quality(df)
        content_patterns = self.analyze_content_patterns(df)
        cross_dept = self.analyze_cross_department_overlap(df)
        
        # Combine results
        insights = {
            'metadata_quality': metadata_quality,
            'content_patterns': content_patterns,
            'cross_department': cross_dept,
            'summary': {
                'key_findings': [],
                'recommendations': [],
                'opportunities': []
            }
        }
        
        # Generate summary insights
        if metadata_quality['recommendations']:
            insights['summary']['recommendations'].extend(metadata_quality['recommendations'])
        
        if content_patterns['insights']:
            insights['summary']['key_findings'].extend(content_patterns['insights'])
        
        if cross_dept['opportunities']:
            insights['summary']['opportunities'].extend(cross_dept['opportunities'])
        
        return insights
    
    def plot_insights(self, insights: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create visualizations for the insights.
        
        Args:
            insights: Dictionary containing analysis insights
            
        Returns:
            Dictionary of Plotly figures
        """
        figures = {}
        
        # Metadata quality heatmap
        if 'metadata_quality' in insights and 'completeness' in insights['metadata_quality']:
            completeness_data = []
            for field, stats in insights['metadata_quality']['completeness'].items():
                completeness_data.append({
                    'field': field,
                    'completeness': stats['completeness_rate']
                })
            
            if completeness_data:  # Only create the plot if we have data
                completeness_df = pd.DataFrame(completeness_data)
                fig = px.bar(
                    completeness_df,
                    x='field',
                    y='completeness',
                    title='Metadata Completeness by Field',
                    labels={'completeness': 'Completeness Rate (%)'}
                )
                figures['metadata_completeness'] = fig
        
        # Content clusters visualization
        if ('content_patterns' in insights and 
            'content_clusters' in insights['content_patterns'] and 
            'cluster_sizes' in insights['content_patterns']['content_clusters']):
            
            cluster_data = []
            for k, v in insights['content_patterns']['content_clusters']['cluster_sizes'].items():
                if k != -1:  # Exclude unclustered
                    cluster_data.append({
                        'cluster': str(k),
                        'size': v
                    })
            
            if cluster_data:  # Only create the plot if we have data
                cluster_df = pd.DataFrame(cluster_data)
                fig = px.pie(
                    cluster_df,
                    values='size',
                    names='cluster',
                    title='Content Clusters Distribution'
                )
                figures['content_clusters'] = fig
        
        # Department overlap heatmap
        if ('cross_department' in insights and 
            'department_overlap' in insights['cross_department']):
            
            overlap_data = []
            for dept_pair, stats in insights['cross_department']['department_overlap'].items():
                dept1, dept2 = dept_pair.split('-')
                overlap_data.append({
                    'dept1': dept1,
                    'dept2': dept2,
                    'similarity': stats['similarity']
                })
            
            if overlap_data:  # Only create the plot if we have data
                overlap_df = pd.DataFrame(overlap_data)
                pivot_df = overlap_df.pivot(
                    index='dept1',
                    columns='dept2',
                    values='similarity'
                )
                
                fig = px.imshow(
                    pivot_df,
                    title='Department Content Overlap',
                    labels={'color': 'Similarity'}
                )
                figures['department_overlap'] = fig
        
        return figures 