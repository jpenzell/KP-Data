"""Analysis page component for the LMS Analysis Dashboard."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.models.data_models import AnalysisResults
import pandas as pd

def render_analysis_page(results: AnalysisResults):
    """Render the detailed analysis sections."""
    # Create tabs for different analysis sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Content Quality",
        "Activity Analysis",
        "Similarity Analysis",
        "Recommendations"
    ])
    
    with tab1:
        render_quality_analysis(results)
    
    with tab2:
        render_activity_analysis(results)
    
    with tab3:
        render_similarity_analysis(results)
    
    with tab4:
        render_recommendations(results)

def render_quality_analysis(results: AnalysisResults):
    """Render the content quality analysis section."""
    st.header("📊 Content Quality Analysis")
    
    if not results.quality_metrics:
        st.info("Quality metrics not available")
        return
    
    metrics = results.quality_metrics
    
    # Display quality scores
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall = metrics.overall_score if metrics.overall_score is not None else 0
        st.metric(
            "Overall Quality",
            f"{overall:.1%}"
        )
    
    with col2:
        completeness = metrics.completeness_score if metrics.completeness_score is not None else 0
        st.metric(
            "Completeness",
            f"{completeness:.1%}"
        )
    
    with col3:
        metadata = metrics.metadata_score if metrics.metadata_score is not None else 0
        st.metric(
            "Metadata Quality",
            f"{metadata:.1%}"
        )
    
    with col4:
        content = metrics.content_score if metrics.content_score is not None else 0
        st.metric(
            "Content Quality",
            f"{content:.1%}"
        )
    
    # Display missing fields
    if hasattr(metrics, 'missing_fields') and metrics.missing_fields:
        st.subheader("Missing Required Fields")
        for field in metrics.missing_fields:
            st.warning(f"Missing {field}")
    
    # Display improvement areas
    if hasattr(metrics, 'improvement_areas') and metrics.improvement_areas:
        st.subheader("Areas for Improvement")
        for area in metrics.improvement_areas:
            st.info(f"Improve {area}")

def render_activity_analysis(results: AnalysisResults):
    """Render the activity analysis section."""
    st.header("📈 Activity Analysis")
    
    if not results.activity_metrics:
        st.info("Activity metrics not available")
        return
    
    metrics = results.activity_metrics
    
    # Display activity metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        active = metrics.active_courses if metrics.active_courses is not None else 0
        st.metric(
            "Active Courses",
            f"{active:,}"
        )
    
    with col2:
        completions = metrics.recent_completions if metrics.recent_completions is not None else 0
        st.metric(
            "Recent Completions",
            f"{completions:,}"
        )
    
    with col3:
        if metrics.average_completion_rate is not None:
            completion_rate = metrics.average_completion_rate * 100
            st.metric(
                "Average Completion Rate",
                f"{completion_rate:.1f}%"
            )
        else:
            st.metric(
                "Average Completion Rate",
                "N/A"
            )
    
    # Display activity trend
    st.subheader("Activity Trend")
    if metrics.activity_trend and len(metrics.activity_trend) > 0:
        activity_data = pd.DataFrame({
            "Date": list(metrics.activity_trend.keys()),
            "Completions": list(metrics.activity_trend.values())
        })
        st.line_chart(activity_data.set_index("Date"))
    else:
        st.info("No activity trend data available")
    
    # Display activity distribution
    st.subheader("Activity Distribution by Category")
    if metrics.activity_distribution and len(metrics.activity_distribution) > 0:
        distribution_data = pd.DataFrame({
            "Category": list(metrics.activity_distribution.keys()),
            "Completions": list(metrics.activity_distribution.values())
        })
        distribution_data = distribution_data.sort_values("Completions", ascending=False)
        
        st.bar_chart(distribution_data.set_index("Category"))
    else:
        st.info("No activity distribution data available")
    
    # Display recent activities
    st.subheader("Recent Activities")
    if metrics.recent_activities and len(metrics.recent_activities) > 0:
        activities_df = pd.DataFrame(metrics.recent_activities)
        activities_df = activities_df.rename(columns={
            "course_no": "Course ID",
            "course_title": "Course Title",
            "completion_date": "Date",
            "learner_count": "Learners"
        })
        st.dataframe(activities_df)
    else:
        st.info("No recent activity data available")
    
    # Display activity recommendations
    st.subheader("Activity Recommendations")
    if metrics.activity_recommendations and len(metrics.activity_recommendations) > 0:
        for rec in metrics.activity_recommendations:
            st.markdown(f"- {rec}")
    else:
        st.info("No activity recommendations available")

def render_similarity_analysis(results: AnalysisResults):
    """Render similarity analysis section."""
    st.header("🔍 Content Similarity Analysis")
    
    if not results.similarity_metrics:
        st.info("No similarity metrics available.")
        return
    
    # Display key metrics
    metrics = results.similarity_metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("High Similarity Pairs", metrics.high_similarity_pairs)
    with col2:
        avg_similarity = metrics.average_similarity if metrics.average_similarity is not None else 0
        st.metric("Average Similarity", f"{avg_similarity:.1%}")
    
    # Display similarity distribution if available
    if hasattr(metrics, 'similarity_distribution') and metrics.similarity_distribution:
        st.subheader("Similarity Distribution")
        dist_data = pd.DataFrame({
            'Level': list(metrics.similarity_distribution.keys()),
            'Count': list(metrics.similarity_distribution.values())
        })
        fig = px.bar(
            dist_data,
            x='Level',
            y='Count',
            title="Similarity Distribution",
            color='Level',
            color_discrete_map={
                'high': 'red',
                'medium': 'orange',
                'low': 'blue'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Display duplicate candidates
    if metrics.duplicate_candidates:
        st.subheader("Potential Duplicates")
        
        # Create a DataFrame for display
        dupes_data = []
        for pair in metrics.duplicate_candidates:
            if not all(k in pair for k in ['course1', 'course2', 'similarity']):
                continue  # Skip entries with missing data
                
            dupes_data.append({
                'Course 1': pair['course1'],
                'Course 2': pair['course2'],
                'Similarity': f"{pair['similarity']:.2f}",
                'Cross-Department': "Yes" if pair.get('cross_department', False) else "No"
            })
        
        if dupes_data:
            st.dataframe(pd.DataFrame(dupes_data))

def render_recommendations(results: AnalysisResults):
    """Render the recommendations section."""
    st.header("🎯 Recommendations")
    
    if not results.recommendations:
        st.info("No recommendations available")
        return
    
    # Group recommendations by category
    categories = {}
    for rec in results.recommendations:
        if rec.category not in categories:
            categories[rec.category] = []
        categories[rec.category].append(rec)
    
    # Display recommendations by category
    for category, recs in categories.items():
        st.subheader(f"{category} Recommendations")
        
        # Sort by priority
        recs.sort(key=lambda x: x.priority)
        
        for rec in recs:
            with st.expander(f"{rec.title} ({rec.priority} priority)"):
                st.markdown(f"""
                **Impact:** {rec.impact}  
                **Effort:** {rec.effort}
                
                {rec.description}
                
                **Implementation:** {rec.implementation if hasattr(rec, 'implementation') else ''}
                """) 