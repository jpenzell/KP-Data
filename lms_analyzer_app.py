import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from pathlib import Path

from lms_content_analyzer import LMSContentAnalyzer

# Must be the first Streamlit command
st.set_page_config(page_title="LMS Content Analysis", layout="wide")

def create_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_timeline(df, date_col):
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        fig = px.histogram(df, x=date_col, title=f'Distribution of {date_col}')
        fig.update_layout(bargap=0.2)
        return fig
    return None

def plot_quality_distribution(df):
    fig = px.histogram(df, x='quality_score', nbins=20,
                      title='Distribution of Content Quality Scores',
                      labels={'quality_score': 'Quality Score', 'count': 'Number of Courses'})
    fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                  annotation_text="Attention Threshold")
    return fig

def display_missing_data_info(missing_data):
    """Display information about missing data and its impact"""
    st.warning("⚠️ Some analyses are limited due to missing data")
    for item in missing_data:
        with st.expander(f"Impact: {item['impact']}"):
            st.write("Required fields:")
            for field in item['required_fields']:
                st.write(f"- {field}")

def main():
    st.title("LMS Content Analysis Dashboard")
    
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            analyzer = LMSContentAnalyzer(uploaded_file)
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Data Quality Overview",
                "Learning Effectiveness",
                "Engagement Analysis",
                "Learning Paths",
                "Business Impact"
            ])
            
            with tab1:
                st.header("Data Quality Overview")
                quality_metrics = analyzer.get_data_quality_metrics()
                
                # Display quality scores
                if quality_metrics.get('completeness'):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Completeness Score", f"{quality_metrics['completeness']:.1f}%")
                    with col2:
                        st.metric("Consistency Score", f"{quality_metrics['consistency']:.1f}%")
                    with col3:
                        st.metric("Validity Score", f"{quality_metrics['validity']:.1f}%")
                
                # Display data quality issues
                issues = analyzer.get_data_quality_issues()
                if issues:
                    st.subheader("Data Quality Issues")
                    for category, items in issues.items():
                        with st.expander(category):
                            for item in items:
                                st.write(f"- {item}")
            
            with tab2:
                st.header("Learning Effectiveness")
                effectiveness = analyzer.analyze_learning_effectiveness()
                
                if effectiveness['missing_data']:
                    display_missing_data_info(effectiveness['missing_data'])
                
                if effectiveness.get('available_metrics'):
                    metrics = effectiveness['available_metrics']
                    
                    # Display completion metrics
                    if 'completion' in metrics:
                        st.subheader("Completion Analysis")
                        completion = metrics['completion']
                        
                        if 'trends' in completion:
                            st.write("Completion Trends")
                            st.write(completion['trends'])
                        
                        if 'dropout_analysis' in completion:
                            with st.expander("View Dropout Analysis"):
                                st.write(completion['dropout_analysis'])
                    
                    # Display performance metrics
                    if 'performance' in metrics:
                        st.subheader("Performance Analysis")
                        performance = metrics['performance']
                        
                        if 'skill_mastery' in performance:
                            st.write("Skill Mastery Levels")
                            st.write(performance['skill_mastery'])
                    
                    # Display skill progression
                    if 'skill_progression' in metrics:
                        st.subheader("Skill Development")
                        progression = metrics['skill_progression']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if 'skill_gaps_closed' in progression:
                                st.write("Skill Gaps Closed")
                                st.write(progression['skill_gaps_closed'])
                        
                        with col2:
                            if 'retention_rates' in progression:
                                st.write("Knowledge Retention")
                                st.write(progression['retention_rates'])
                
                if effectiveness.get('recommendations'):
                    st.subheader("Recommendations")
                    for rec in effectiveness['recommendations']:
                        st.write(f"- {rec}")
            
            with tab3:
                st.header("Engagement Analysis")
                engagement = analyzer.analyze_engagement_patterns()
                
                if engagement['missing_data']:
                    display_missing_data_info(engagement['missing_data'])
                
                if engagement.get('available_metrics'):
                    metrics = engagement['available_metrics']
                    
                    # Display engagement levels
                    if 'engagement_levels' in metrics:
                        st.subheader("Engagement Distribution")
                        st.write(metrics['engagement_levels'])
                    
                    # Display interaction analysis
                    if 'interaction_analysis' in metrics:
                        st.subheader("Interaction Patterns")
                        with st.expander("View Interaction Analysis"):
                            st.write(metrics['interaction_analysis'])
                    
                    # Display engagement quality
                    if 'engagement_quality' in metrics:
                        st.subheader("Engagement Quality")
                        st.write(metrics['engagement_quality'])
                
                if engagement.get('recommendations'):
                    st.subheader("Recommendations")
                    for rec in engagement['recommendations']:
                        st.write(f"- {rec}")
            
            with tab4:
                st.header("Learning Paths")
                paths = analyzer.analyze_learning_paths()
                
                if paths['missing_data']:
                    display_missing_data_info(paths['missing_data'])
                
                if paths.get('available_metrics'):
                    metrics = paths['available_metrics']
                    
                    # Display learning sequences
                    if 'learning_sequences' in metrics:
                        st.subheader("Common Learning Paths")
                        st.write("Course Sequences")
                        st.write(metrics['learning_sequences']['common_sequences'])
                        
                        with st.expander("View Sequence Effectiveness"):
                            st.write(metrics['learning_sequences']['effectiveness'])
                    
                    # Display content relationships
                    if 'content_relationships' in metrics:
                        st.subheader("Related Content")
                        selected_course = st.selectbox(
                            "Select a course to see related content:",
                            options=list(metrics['content_relationships'].keys())
                        )
                        if selected_course:
                            st.write("Related Courses:")
                            course_data = metrics['content_relationships'][selected_course]
                            for course, score in zip(
                                course_data['related_courses'],
                                course_data['similarity_scores']
                            ):
                                st.write(f"- {course} (Similarity: {score:.2f})")
                    
                    # Display skill coverage
                    if 'skill_coverage' in metrics:
                        st.subheader("Skill-Based Paths")
                        selected_skill = st.selectbox(
                            "Select a skill to see recommended courses:",
                            options=list(metrics['skill_coverage'].keys())
                        )
                        if selected_skill:
                            st.write("Recommended Courses:")
                            for course in metrics['skill_coverage'][selected_skill]:
                                st.write(f"- {course}")
                
                if paths.get('recommendations'):
                    st.subheader("Recommendations")
                    for rec in paths['recommendations']:
                        st.write(f"- {rec}")
            
            with tab5:
                st.header("Business Impact")
                impact = analyzer.analyze_business_impact()
                
                if impact['missing_data']:
                    display_missing_data_info(impact['missing_data'])
                
                if impact.get('available_metrics'):
                    metrics = impact['available_metrics']
                    
                    # Display ROI metrics
                    if 'roi' in metrics:
                        st.subheader("ROI Analysis")
                        roi = metrics['roi']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Investment", f"${roi['total_investment']:,.2f}")
                        with col2:
                            st.metric("Cost per Completion", f"${roi['cost_per_completion']:,.2f}")
                        with col3:
                            st.metric("Completion Rate", f"{roi['completion_rate']:.1f}%")
                    
                    # Display skill development impact
                    if 'skill_development' in metrics:
                        st.subheader("Skill Development Impact")
                        skill_dev = metrics['skill_development']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Average Improvement", f"{skill_dev['average_improvement']:.1f}%")
                        with col2:
                            st.metric("Significant Improvements", f"{skill_dev['significant_improvements']:.1f}%")
                        
                        with st.expander("View Impact by Area"):
                            st.write(skill_dev['areas_of_impact'])
                    
                    # Display productivity impact
                    if 'productivity' in metrics:
                        st.subheader("Productivity Impact")
                        prod = metrics['productivity']
                        st.metric("Average Productivity Improvement", f"{prod['average_improvement']:.1f}%")
                        
                        with st.expander("View High-Impact Courses"):
                            for course in prod['high_impact_courses']:
                                st.write(f"- {course}")
                
                if impact.get('recommendations'):
                    st.subheader("Recommendations")
                    for rec in impact['recommendations']:
                        st.write(f"- {rec}")
        
        except Exception as e:
            st.error("Error analyzing file: " + str(e))
            st.write("Please ensure your file contains the required data and try again.")
    else:
        st.info("Please upload an Excel file to begin analysis.")

if __name__ == "__main__":
    main() 