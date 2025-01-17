import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from pathlib import Path
from lms_content_analyzer import LMSContentAnalyzer
from datetime import datetime
import tempfile
import numpy as np

st.set_page_config(page_title="LMS Content Library Analyzer", layout="wide")

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

def main():
    st.title("LMS Content Library Analyzer")
    st.write("Upload your LMS content spreadsheet for detailed analysis and insights.")

    uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        try:
            # Initialize analyzer with the temporary file
            analyzer = LMSContentAnalyzer(temp_path)
            insights = analyzer.get_actionable_insights()
            
            # Create tabs for different analyses
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Executive Summary", 
                "Content Quality", 
                "Content Gaps & Redundancies",
                "Temporal Analysis",
                "Detailed Analysis"
            ])
            
            with tab1:
                st.header("Executive Summary")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Courses", len(analyzer.df))
                with col2:
                    st.metric("Average Quality", f"{analyzer.df['quality_score'].mean():.2f}")
                with col3:
                    st.metric("Needs Attention", len(analyzer.df[analyzer.df['needs_attention']]))
                with col4:
                    st.metric("Categories", len(analyzer.df['category_name'].unique()))
                
                # Urgent attention section
                st.subheader("🚨 Urgent Attention Required")
                if insights['urgent_attention_needed']:
                    for course in insights['urgent_attention_needed'][:5]:  # Show top 5
                        with st.expander(f"{course['course_title']} (Score: {course['quality_score']:.2f})"):
                            st.write("Issues:")
                            for issue in course['issues']:
                                st.write(f"- Missing {issue.replace('_complete', '')}")
                else:
                    st.success("No courses require urgent attention!")
                
                # Quick insights
                st.subheader("📊 Quick Insights")
                col1, col2 = st.columns(2)
                with col1:
                    if insights['temporal_insights']:
                        st.metric("Monthly New Content", 
                                f"{insights['temporal_insights']['average_monthly_new_content']:.1f}")
                        st.metric("Content Age Range", 
                                f"{insights['temporal_insights']['oldest_content_age']:.1f} years")
                
                with col2:
                    if insights['content_redundancies']:
                        st.metric("Potential Redundancies", 
                                len(insights['content_redundancies']))
                    if insights['content_gaps']:
                        st.metric("Categories with Gaps", 
                                len(insights['content_gaps']))

            with tab2:
                st.header("Content Quality Analysis")
                
                # Quality score distribution
                st.subheader("Quality Score Distribution")
                fig = plot_quality_distribution(analyzer.df)
                st.plotly_chart(fig)
                
                # Quality improvement opportunities
                st.subheader("🔄 Quality Improvement Opportunities")
                if insights['quality_improvements']:
                    for course in insights['quality_improvements']:
                        with st.expander(f"{course['course_title']} (Score: {course['current_score']:.2f})"):
                            for area, score in course['improvement_areas'].items():
                                if score < 0.6:
                                    st.write(f"- {area.title()}: {score:.2f}")
                                    if area == 'description':
                                        st.write("  → Add more detailed description")
                                    elif area == 'metadata':
                                        st.write("  → Complete missing metadata fields")
                                    elif area == 'freshness':
                                        st.write("  → Review and update content")
                                    elif area == 'keywords':
                                        st.write("  → Add more relevant keywords")

            with tab3:
                st.header("Content Gaps & Redundancies")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📉 Content Gaps")
                    if insights['content_gaps']:
                        for gap in insights['content_gaps']:
                            with st.expander(f"{gap['category']}"):
                                st.write(f"Current Count: {gap['current_count']}")
                                st.write(f"Gap Percentage: {gap['gap_percentage']:.1f}%")
                                st.write(f"Average Category Count: {gap['avg_category_count']:.1f}")
                    else:
                        st.info("No significant content gaps detected")
                
                with col2:
                    st.subheader("🔄 Content Redundancies")
                    if insights['content_redundancies']:
                        for redundancy in insights['content_redundancies']:
                            with st.expander(f"{redundancy['course']}"):
                                for similar in redundancy['similar_courses']:
                                    st.write(f"- {similar['title']}")
                                    st.write(f"  Similarity: {similar['similarity_score']:.2f}")
                    else:
                        st.info("No significant content redundancies detected")

            with tab4:
                st.header("Temporal Analysis")
                
                # Content velocity
                if insights['temporal_insights']:
                    st.subheader("📈 Content Velocity")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("New Content (6 months)", 
                                insights['temporal_insights']['content_velocity'])
                        st.metric("Monthly Average", 
                                f"{insights['temporal_insights']['average_monthly_new_content']:.1f}")
                    
                    # Category trends
                    st.subheader("📊 Category Trends")
                    if insights['category_distribution']:
                        trend_data = pd.DataFrame(insights['category_distribution'])
                        fig = px.bar(trend_data, x='category', y='percentage',
                                   color='trend',
                                   title='Category Distribution and Trends')
                        st.plotly_chart(fig)

            with tab5:
                st.header("Detailed Analysis")
                
                # Text analysis
                st.subheader("📝 Text Analysis")
                text_analysis = analyzer.analyze_text_fields()
                selected_field = st.selectbox(
                    "Select field to analyze:",
                    analyzer.text_columns
                )
                
                if selected_field in text_analysis:
                    col1, col2 = st.columns(2)
                    with col1:
                        all_text = ' '.join(analyzer.df[selected_field].astype(str).fillna(''))
                        fig = create_wordcloud(all_text)
                        st.pyplot(fig)
                        plt.close()
                    
                    with col2:
                        st.metric("Unique Values", text_analysis[selected_field]['unique_values'])
                        st.metric("Average Length", f"{text_analysis[selected_field]['avg_length']:.1f}")
                        
                        # Top keywords
                        st.write("Top Keywords:")
                        keywords_df = pd.DataFrame(
                            list(text_analysis[selected_field]['top_keywords'].items()),
                            columns=['Keyword', 'Count']
                        )
                        st.dataframe(keywords_df)
                
                # Export options
                st.subheader("📤 Export Analysis")
                if st.button("Generate Detailed Report"):
                    report = analyzer.generate_enhanced_report()
                    report_time = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="Download Analysis Report",
                        data=report,
                        file_name=f"lms_analysis_report_{report_time}.txt",
                        mime="text/plain"
                    )

        except Exception as e:
            st.error(f"Error analyzing file: {str(e)}")
        finally:
            # Clean up temporary file
            Path(temp_path).unlink()

if __name__ == "__main__":
    main() 