import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from pathlib import Path

from lms_content_analyzer import LMSContentAnalyzer

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

def main():
    st.title("LMS Content Analysis Dashboard")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            analyzer = LMSContentAnalyzer(uploaded_file)
            
            # Create tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Data Quality",
                "Training Categories",
                "Delivery & Usage",
                "Content Production",
                "Financial Analysis"
            ])
            
            with tab1:
                st.header("Data Quality Overview")
                
                # Get metrics
                quality_metrics = analyzer.get_data_quality_metrics()
                missing_data = analyzer.get_missing_data_summary()
                value_distributions = analyzer.get_value_distributions()
                date_ranges = analyzer.get_date_ranges()
                text_stats = analyzer.get_text_field_stats()
                recommendations = analyzer.get_recommendations()
                
                # Display metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Completeness")
                    completeness_df = pd.DataFrame.from_dict(quality_metrics['completeness'], orient='index', columns=['Score'])
                    st.dataframe(completeness_df.style.format("{:.1f}%"))
                
                with col2:
                    st.subheader("Consistency")
                    consistency_df = pd.DataFrame.from_dict(quality_metrics['consistency'], orient='index', columns=['Score'])
                    st.dataframe(consistency_df.style.format("{:.1f}%"))
                
                with col3:
                    st.subheader("Validity")
                    validity_df = pd.DataFrame.from_dict(quality_metrics['validity'], orient='index', columns=['Score'])
                    st.dataframe(validity_df.style.format("{:.1f}%"))
                
                # Display recommendations
                st.subheader("Recommendations")
                for rec in recommendations:
                    st.warning(rec)
                
                # Display detailed stats
                with st.expander("View Detailed Statistics"):
                    st.subheader("Missing Data Summary")
                    missing_df = pd.DataFrame.from_dict(missing_data, orient='index')
                    st.dataframe(missing_df.style.format({
                        'count': '{:,.0f}',
                        'percentage': '{:.1f}%'
                    }))
                    
                    st.subheader("Value Distributions")
                    for col, dist in value_distributions.items():
                        st.write(f"\n**{col}**")
                        dist_df = pd.DataFrame.from_dict(dist, orient='index', columns=['Count'])
                        st.dataframe(dist_df)
                    
                    st.subheader("Date Ranges")
                    for col, ranges in date_ranges.items():
                        st.write(f"\n**{col}**")
                        st.write(f"Min: {ranges['min']}")
                        st.write(f"Max: {ranges['max']}")
                    
                    st.subheader("Text Field Statistics")
                    for col, stats in text_stats.items():
                        st.write(f"\n**{col}**")
                        st.write(f"Min length: {stats['min_length']:.0f}")
                        st.write(f"Max length: {stats['max_length']:.0f}")
                        st.write(f"Average length: {stats['avg_length']:.1f}")
            
            with tab2:
                st.header("Training Categories Analysis")
                
                # Training Split by Categories
                st.subheader("Training Split by Categories")
                category_data = analyzer.get_category_distribution()
                fig = px.treemap(category_data, 
                               path=['category_type', 'subcategory'],
                               values='count',
                               title='Training Distribution by Category')
                st.plotly_chart(fig)
                
                # Training Focus by Persona
                st.subheader("Training Focus by Persona")
                col1, col2 = st.columns(2)
                with col1:
                    persona_data = analyzer.get_persona_distribution()
                    fig = px.pie(persona_data, 
                               values='percentage',
                               names='persona',
                               title='Training Split by Persona')
                    st.plotly_chart(fig)
                
                # Top 10 Learner Interests
                with col2:
                    interests = analyzer.get_top_learner_interests()
                    fig = px.bar(interests,
                               x='interest',
                               y='count',
                               title='Most Popular Learning Areas')
                    st.plotly_chart(fig)
            
            with tab3:
                st.header("Delivery & Usage Analysis")
                
                # Training Delivery Split
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Training Delivery Methods")
                    delivery_data = analyzer.get_delivery_split()
                    fig = px.pie(delivery_data,
                               values='percentage',
                               names='method',
                               title='Training Delivery Split')
                    st.plotly_chart(fig)
                
                # Content Usage Length
                with col2:
                    st.subheader("Content Usage Length")
                    usage_data = analyzer.get_content_usage_length()
                    fig = px.bar(usage_data,
                               x='duration',
                               y='percentage',
                               title='Content Usage Duration')
                    st.plotly_chart(fig)
            
            with tab4:
                st.header("Content Production Analysis")
                
                # Content Source Distribution
                st.subheader("Content by Source")
                source_data = analyzer.get_content_source_distribution()
                fig = px.pie(source_data,
                           values='percentage',
                           names='source',
                           title='Content Source Distribution')
                st.plotly_chart(fig)
                
                # Training Volume by Organization
                st.subheader("Training Volume by Organization")
                org_data = analyzer.get_training_volume_by_org()
                fig = px.bar(org_data,
                           x='organization',
                           y='volume',
                           title='Training Volume by Organization')
                st.plotly_chart(fig)
            
            with tab5:
                st.header("Financial Analysis")
                
                # Training Costs
                st.subheader("Training Costs Analysis")
                cost_metrics = analyzer.get_training_cost_metrics()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Cost per Learner", f"${cost_metrics['avg_cost_per_learner']:,.2f}")
                with col2:
                    st.metric("Total Training Spend", f"${cost_metrics['total_training_spend']:,.2f}")
                with col3:
                    st.metric("Tuition Reimbursement", f"${cost_metrics['tuition_reimbursement']:,.2f}")
                
                # Cost Breakdown
                st.subheader("Cost Breakdown by Role")
                cost_by_role = analyzer.get_cost_by_role()
                fig = px.bar(cost_by_role,
                           x='role',
                           y='yearly_average',
                           title='Average Yearly Training Cost by Role')
                st.plotly_chart(fig)
                
                # Export Report Option
                st.subheader("📊 Export Analysis Report")
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
            st.exception(e)
    else:
        st.info("Please upload an Excel file to begin analysis.")

if __name__ == "__main__":
    main() 