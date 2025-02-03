import streamlit as st
# Must be the first Streamlit command
st.set_page_config(page_title="LMS Content Analysis", layout="wide")

import pandas as pd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import io
from pathlib import Path

from lms_content_analyzer import LMSContentAnalyzer

# Constants for mapping and categorization
TRAINING_CATEGORIES = {
    'leadership_development': [
        'leadership', 'management', 'executive', 'strategic', 'decision making',
        'organizational development', 'change management', 'business acumen'
    ],
    'managerial_supervisory': [
        'supervisor', 'manager', 'team lead', 'performance management',
        'delegation', 'coaching', 'mentoring', 'employee development'
    ],
    'mandatory_compliance': [
        'compliance', 'required', 'mandatory', 'regulatory', 'policy',
        'hipaa', 'privacy', 'security awareness', 'ethics', 'code of conduct'
    ],
    'profession_specific': [
        'clinical', 'technical', 'specialized', 'certification',
        'professional development', 'continuing education', 'skill-specific'
    ],
    'interpersonal_skills': [
        'soft skills', 'communication', 'teamwork', 'collaboration',
        'emotional intelligence', 'conflict resolution', 'presentation',
        'customer service', 'cultural competency'
    ],
    'it_systems': [
        'technology', 'software', 'systems', 'digital', 'computer',
        'application', 'platform', 'tool', 'database', 'security'
    ],
    'clinical': [
        'medical', 'healthcare', 'patient care', 'diagnosis', 'treatment',
        'clinical practice', 'patient safety', 'medical procedures',
        'health assessment', 'clinical skills'
    ],
    'nursing': [
        'nurse', 'nursing', 'clinical care', 'patient assessment',
        'medication administration', 'care planning', 'nursing procedures',
        'nursing documentation', 'nursing practice'
    ],
    'pharmacy': [
        'pharmacy', 'medication', 'prescription', 'drug', 'pharmaceutical',
        'dispensing', 'pharmacology', 'pharmacy practice', 'medication safety'
    ],
    'safety': [
        'safety', 'security', 'protection', 'emergency', 'risk management',
        'workplace safety', 'infection control', 'hazard', 'prevention'
    ]
}

DELIVERY_METHODS = {
    'instructor_led_in_person': 'Traditional classroom',
    'instructor_led_virtual': 'Online live sessions',
    'self_paced_elearning': 'Asynchronous learning',
    'microlearning': 'Bite-sized content'
}

def create_wordcloud(text):
    """Create and return a wordcloud visualization"""
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def plot_timeline(df, date_col):
    """Create a timeline visualization for the specified date column"""
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        fig = px.histogram(df, x=date_col, title=f'Distribution of {date_col}')
        fig.update_layout(bargap=0.2)
        return fig
    return None

def plot_quality_distribution(df):
    """Plot the distribution of quality scores"""
    if 'quality_score' not in df.columns:
        return None
        
    fig = px.histogram(df, x='quality_score', nbins=20,
                      title='Distribution of Content Quality Scores',
                      labels={'quality_score': 'Quality Score', 'count': 'Number of Courses'})
    fig.add_vline(x=0.6, line_dash="dash", line_color="red",
                  annotation_text="Attention Threshold")
    return fig

def calculate_quality_score(df):
    """Calculate quality scores based on available data"""
    df['quality_score'] = 0.0
    
    # Completeness score (based on required fields)
    required_fields = ['course_title', 'description', 'course_abstract']
    available_required_fields = [field for field in required_fields if field in df.columns]
    if available_required_fields:
        completeness = df[available_required_fields].notna().mean(axis=1)
        df['quality_score'] += completeness * 0.4
    
    # Date validity score
    date_fields = ['course_version', 'course_available_from', 'course_discontinued_from']
    date_validity_cols = [col + '_is_valid' for col in date_fields if col + '_is_valid' in df.columns]
    if date_validity_cols:
        date_validity = df[date_validity_cols].mean(axis=1)
        df['quality_score'] += date_validity * 0.3
    
    # Additional metadata score
    metadata_fields = ['course_no', 'duration_mins', 'region_entity', 'person_type', 'course_type']
    available_metadata_fields = [field for field in metadata_fields if field in df.columns]
    if available_metadata_fields:
        metadata_completeness = df[available_metadata_fields].notna().mean(axis=1)
        df['quality_score'] += metadata_completeness * 0.3
    
    return df

def categorize_content(df):
    """Categorize content based on text fields and keywords"""
    # Initialize category columns
    for category in TRAINING_CATEGORIES.keys():
        df[f'is_{category.lower().replace(" ", "_")}'] = False
    
    # Text columns to analyze with their weights
    text_columns = {
        'course_title': 1.0,
        'description': 0.8,
        'course_abstract': 0.6,
        'course_keywords': 0.9
    }
    
    # Process each row
    for idx, row in df.iterrows():
        combined_text = ""
        
        # Combine text fields with weights
        for col, weight in text_columns.items():
            if col in df.columns:
                text = str(row.get(col, '')).lower()
                combined_text += ' ' + text  # Add space to separate fields
        
        # Check for category matches
        for category, keywords in TRAINING_CATEGORIES.items():
            category_col = f'is_{category.lower().replace(" ", "_")}'
            
            # Check category name
            if category.lower() in combined_text:
                df.at[idx, category_col] = True
                continue
            
            # Check keywords
            if keywords:
                for keyword in keywords:
                    if keyword.lower() in combined_text:
                        df.at[idx, category_col] = True
                        break
    
    return df

def analyze_training_focus(df):
    """Analyze training focus distribution and overlaps"""
    focus_stats = {}
    for category in TRAINING_CATEGORIES.keys():
        focus_stats[category] = {
            'count': df[f'is_{category}'].sum(),
            'percentage': (df[f'is_{category}'].sum() / len(df) * 100)
        }
    return focus_stats

def analyze_delivery_methods(df):
    """Analyze training delivery methods"""
    if 'delivery_method' not in df.columns:
        return None
    
    delivery_stats = df['delivery_method'].value_counts().to_dict()
    return {
        'distribution': delivery_stats,
        'total_courses': len(df),
        'avg_duration': df['duration_mins'].mean() if 'duration_mins' in df.columns else None
    }

def merge_and_standardize_data(dataframes):
    """Merge and standardize data from multiple sources with intelligent cross-referencing"""
    if not dataframes:
        return pd.DataFrame()
    
    # Debug information
    for idx, df in enumerate(dataframes):
        st.write(f"File {idx + 1} columns:", df.columns.tolist())
        
    # Initialize empty master dataframe
    master_df = pd.DataFrame()
    
    # First pass: Identify all unique identifiers and create mapping
    id_mapping = {}
    for idx, df in enumerate(dataframes):
        # Standardize column names - make lowercase and strip whitespace
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Try to find any course identifiers
        potential_ids = ['course_id', 'course_no', 'offering_template_no']
        for id_col in potential_ids:
            if id_col in df.columns:
                if id_col not in id_mapping:
                    id_mapping[id_col] = set()
                id_mapping[id_col].update(df[id_col].astype(str).dropna().unique())
    
    # Second pass: Standardize and merge data
    for idx, df in enumerate(dataframes):
        df_copy = df.copy()
        
        # Map various column names to standard names
        column_mapping = {
            'course_title': 'course_title',
            'title': 'course_title',
            'course_no': 'course_no',
            'course_number': 'course_no',
            'description': 'course_description',
            'course_description': 'course_description',
            'abstract': 'course_abstract',
            'course_abstract': 'course_abstract',
            'version': 'course_version',
            'course_version': 'course_version',
            'created_by': 'course_created_by',
            'course_created_by': 'course_created_by',
            'avail_from': 'course_available_from',
            'available_from': 'course_available_from',
            'course_available_from': 'course_available_from',
            'disc_from': 'course_discontinued_from',
            'discontinued_from': 'course_discontinued_from',
            'course_discontinued_from': 'course_discontinued_from',
            'duration': 'duration_mins',
            'duration_mins': 'duration_mins',
            'region': 'region_entity',
            'region_entity': 'region_entity',
            'person_type': 'person_type',
            'type': 'course_type',
            'course_type': 'course_type',
            'no_of_learners': 'learner_count',
            'learner_count': 'learner_count',
            'course_keywords': 'course_keywords',
            'course_keywords': 'course_keywords',
            'total_2024_activity': 'activity_count'
        }
        
        # Debug information
        st.write(f"File {idx + 1} original columns:", df_copy.columns.tolist())
        
        # Rename columns based on mapping
        df_copy.rename(columns={k: v for k, v in column_mapping.items() if k in df_copy.columns}, inplace=True)
        
        # Debug information
        st.write(f"File {idx + 1} mapped columns:", df_copy.columns.tolist())
        
        # Add source tracking
        df_copy['data_source'] = f"source_{idx}"
        
        if master_df.empty:
            master_df = df_copy
        else:
            # Try different merge strategies
            merged = False
            
            # Try course_id first
            if 'course_id' in df_copy.columns and 'course_id' in master_df.columns:
                master_df = pd.merge(master_df, df_copy, on='course_id', how='outer', suffixes=('', '_new'))
                merged = True
            
            # Try course_no next
            elif 'course_no' in df_copy.columns and 'course_no' in master_df.columns:
                master_df = pd.merge(master_df, df_copy, on='course_no', how='outer', suffixes=('', '_new'))
                merged = True
            
            # Try offering_template_no as another option
            elif 'offering_template_no' in df_copy.columns and 'offering_template_no' in master_df.columns:
                master_df = pd.merge(master_df, df_copy, on='offering_template_no', how='outer', suffixes=('', '_new'))
                merged = True
            
            # Try course_title as last resort
            elif 'course_title' in df_copy.columns and 'course_title' in master_df.columns:
                master_df = pd.merge(master_df, df_copy, on='course_title', how='outer', suffixes=('', '_new'))
                merged = True
            
            # If no merge possible, append with tracking
            if not merged:
                master_df = pd.concat([master_df, df_copy], ignore_index=True)
            
            # Clean up duplicate columns
            for col in master_df.columns:
                if col.endswith('_new'):
                    base_col = col[:-4]
                    if base_col in master_df.columns:
                        master_df[base_col] = master_df[base_col].fillna(master_df[col])
                    master_df.drop(col, axis=1, inplace=True)
    
    # Debug information
    st.write("Final columns after merge:", master_df.columns.tolist())
    
    # Convert and clean data types
    date_columns = ['course_available_from', 'course_discontinued_from', 'course_version']
    for col in date_columns:
        if col in master_df.columns:
            # Try multiple date formats
            master_df[col] = pd.to_datetime(master_df[col], format='mixed', errors='coerce')
            master_df[f'{col}_is_valid'] = ~master_df[col].isna()
            if master_df[f'{col}_is_valid'].sum() == 0:
                st.warning(f"No valid dates found in {col}")
    
    numeric_columns = ['duration_mins', 'learner_count', 'activity_count']
    for col in numeric_columns:
        if col in master_df.columns:
            master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
    
    # Add metadata columns
    master_df['data_sources'] = master_df['data_source'].astype(str)
    master_df['cross_reference_count'] = master_df.groupby(['course_title'])['data_source'].transform('nunique')
    
    # Categorize content
    master_df = categorize_content(master_df)
    
    return master_df

def analyze_metadata_completeness(df):
    """Analyze metadata completeness and quality"""
    metadata_fields = {
        'basic': ['course_title', 'course_no', 'course_description'],
        'temporal': ['course_available_from', 'course_discontinued_from', 'course_version'],
        'classification': ['course_type', 'category_name', 'person_type'],
        'usage': ['learner_count', 'activity_count', 'duration_mins'],
        'organizational': ['region_entity', 'course_created_by'],
        'enrichment': ['course_keywords', 'course_abstract']
    }
    
    completeness_stats = {}
    for category, fields in metadata_fields.items():
        available_fields = [f for f in fields if f in df.columns]
        if available_fields:
            completeness = df[available_fields].notna().mean(axis=1)
            completeness_stats[category] = {
                'avg_completeness': completeness.mean(),
                'fields_present': len(available_fields),
                'total_fields': len(fields),
                'missing_fields': set(fields) - set(available_fields)
            }
    
    return completeness_stats

def analyze_cross_references(df):
    """Analyze cross-references between data sources"""
    stats = {
        'total_courses': len(df),
        'multi_source_courses': 0,
        'consistency_metrics': {}
    }
    
    # Count multi-source courses if the data is available
    if 'cross_reference_count' in df.columns:
        stats['multi_source_courses'] = len(df[df['cross_reference_count'] > 1])
    
    # Check field consistency across sources if we have the necessary columns
    if 'course_no' in df.columns and 'data_source' in df.columns:
        fields_to_check = ['course_title', 'course_description', 'course_type', 'category_name']
        available_fields = [f for f in fields_to_check if f in df.columns]
        
        for field in available_fields:
            consistency = df.groupby('course_no')[field].nunique()
            stats['consistency_metrics'][field] = {
                'consistent': len(consistency[consistency == 1]),
                'inconsistent': len(consistency[consistency > 1])
            }
    
    return stats

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in the data"""
    temporal_analysis = {}
    
    if 'course_available_from' in df.columns:
        valid_dates = df[df['course_available_from_is_valid']]
        if not valid_dates.empty:
            temporal_analysis['creation_patterns'] = {
                'courses_per_year': valid_dates.groupby(
                    valid_dates['course_available_from'].dt.year
                ).size().to_dict(),
                'courses_per_month': valid_dates.groupby(
                    valid_dates['course_available_from'].dt.month
                ).size().to_dict()
            }
    
    if 'course_discontinued_from' in df.columns:
        active_courses = df[
            (df['course_discontinued_from'].isna()) |
            (df['course_discontinued_from'] > pd.Timestamp.now())
        ]
        temporal_analysis['lifecycle'] = {
            'active_courses': len(active_courses),
            'discontinued_courses': len(df) - len(active_courses)
        }
    
    return temporal_analysis

def analyze_content_relationships(df):
    """Analyze relationships between different content attributes"""
    relationships = {}
    
    # Analyze category-audience relationships
    if all(col in df.columns for col in ['person_type', 'category_name']):
        relationships['category_audience'] = pd.crosstab(
            df['category_name'],
            df['person_type']
        ).to_dict()
    
    # Analyze duration patterns
    if 'duration_mins' in df.columns:
        duration_stats = df.groupby('course_type')['duration_mins'].agg([
            'mean', 'median', 'std'
        ]).to_dict()
        relationships['duration_patterns'] = duration_stats
    
    # Analyze regional patterns
    if 'region_entity' in df.columns:
        for category in TRAINING_CATEGORIES.keys():
            if f'is_{category}' in df.columns:
                regional_dist = df[df[f'is_{category}']]['region_entity'].value_counts()
                relationships[f'regional_{category}'] = regional_dist.to_dict()
    
    return relationships

def main():
    # Application Header
    st.title("LMS Content Analysis Dashboard")
    st.markdown("""
    This dashboard analyzes your Learning Management System (LMS) content to provide insights into:
    - Content quality and completeness
    - Training distribution and focus
    - Usage patterns and engagement
    - Resource allocation and coverage
    """)
    
    # File Upload Section
    st.header("📤 Data Upload")
    with st.expander("Upload Instructions", expanded=True):
        st.markdown("""
        1. Prepare your Excel files containing LMS data
        2. Files should include course information, usage data, and metadata
        3. Multiple files can be uploaded for cross-reference analysis
        4. Supported format: .xlsx
        """)
    
    uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)
    
    if not uploaded_files:
        st.info("👆 Upload your LMS data files to begin the analysis")
        return
        
    # Process uploaded files
    dataframes = []
    with st.spinner("Processing uploaded files..."):
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_excel(uploaded_file)
                dataframes.append(df)
                st.success(f"✅ Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                continue
    
    if not dataframes:
        st.error("No valid data files were uploaded. Please check your files and try again.")
        return
        
    # Data Processing
    try:
        with st.spinner("Analyzing data..."):
            # Merge and standardize all data
            combined_data = merge_and_standardize_data(dataframes)
            combined_data = calculate_quality_score(combined_data)
            
            # Create analyzer instance
            combined_buffer = io.BytesIO()
            with pd.ExcelWriter(combined_buffer, engine='openpyxl') as writer:
                combined_data.to_excel(writer, index=False)
            combined_buffer.seek(0)
            analyzer = LMSContentAnalyzer(combined_buffer)
            analyzer.df = combined_data
            
            # High-level Overview
            st.header("📊 Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Courses", f"{len(combined_data):,}")
                if 'learner_count' in combined_data.columns:
                    st.metric("Total Learners", f"{combined_data['learner_count'].sum():,.0f}")
            
            with col2:
                active_courses = len(combined_data)
                if 'course_discontinued_from' in combined_data.columns:
                    active_courses = len(combined_data[
                        (combined_data['course_discontinued_from'].isna()) | 
                        (combined_data['course_discontinued_from'] > pd.Timestamp.now())
                    ])
                st.metric("Active Courses", f"{active_courses:,}")
                if 'region_entity' in combined_data.columns:
                    st.metric("Regions Covered", combined_data['region_entity'].nunique())
            
            with col3:
                if 'data_source' in combined_data.columns:
                    st.metric("Data Sources", combined_data['data_source'].nunique())
                if 'cross_reference_count' in combined_data.columns:
                    multi_source = len(combined_data[combined_data['cross_reference_count'] > 1])
                    st.metric("Cross-Referenced Courses", f"{multi_source:,}")
            
            # Navigation
            st.header("🔍 Analysis Sections")
            analysis_choice = st.radio(
                "Select an analysis area to explore:",
                ["Data Quality & Completeness",
                 "Content Distribution",
                 "Learning Impact",
                 "Resource Allocation",
                 "Recommendations"],
                horizontal=True
            )
            
            if analysis_choice == "Data Quality & Completeness":
                st.subheader("📋 Data Quality Analysis")
                
                # Quality Score Overview
                quality_col1, quality_col2 = st.columns([2, 1])
                with quality_col1:
                    fig = plot_quality_distribution(combined_data)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                
                with quality_col2:
                    st.markdown("### Quality Metrics")
                    completeness_stats = analyze_metadata_completeness(combined_data)
                    for category, stats in completeness_stats.items():
                        with st.expander(f"{category.title()} ({stats['avg_completeness']*100:.1f}% complete)"):
                            st.metric("Fields Present", f"{stats['fields_present']}/{stats['total_fields']}")
                            if stats['missing_fields']:
                                st.info("Missing: " + ", ".join(stats['missing_fields']))
                
            elif analysis_choice == "Content Distribution":
                st.subheader("📚 Content Distribution Analysis")
                
                # Category Distribution
                st.markdown("### Training Categories")
                focus_stats = analyze_training_focus(combined_data)
                focus_df = pd.DataFrame(focus_stats).T
                
                cat_col1, cat_col2 = st.columns([2, 1])
                with cat_col1:
                    fig = px.bar(focus_df, x=focus_df.index, y='percentage',
                               title='Distribution of Training Categories')
                    st.plotly_chart(fig, use_container_width=True)
                
                with cat_col2:
                    st.markdown("### Key Insights")
                    top_categories = focus_df.sort_values('percentage', ascending=False).head(3)
                    st.markdown("**Top Categories:**")
                    for idx, row in top_categories.iterrows():
                        st.markdown(f"- {idx}: {row['percentage']:.1f}%")
                
            elif analysis_choice == "Learning Impact":
                st.subheader("📈 Learning Impact Analysis")
                
                impact_col1, impact_col2 = st.columns(2)
                
                with impact_col1:
                    if 'learner_count' in combined_data.columns:
                        st.markdown("### Usage Patterns")
                        fig = px.histogram(combined_data, x='learner_count',
                                         title='Course Usage Distribution',
                                         nbins=30)
                        st.plotly_chart(fig, use_container_width=True)
                
                with impact_col2:
                    if 'course_type' in combined_data.columns and 'learner_count' in combined_data.columns:
                        st.markdown("### Impact by Course Type")
                        volume_by_type = combined_data.groupby('course_type')['learner_count'].sum()
                        fig = px.pie(values=volume_by_type.values, names=volume_by_type.index,
                                   title='Learning Volume Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                
            elif analysis_choice == "Resource Allocation":
                st.subheader("📊 Resource Allocation Analysis")
                
                resource_col1, resource_col2 = st.columns(2)
                
                with resource_col1:
                    if 'region_entity' in combined_data.columns:
                        st.markdown("### Regional Distribution")
                        region_dist = combined_data['region_entity'].value_counts()
                        fig = px.pie(values=region_dist.values, names=region_dist.index,
                                   title='Course Distribution by Region')
                        st.plotly_chart(fig, use_container_width=True)
                
                with resource_col2:
                    if 'data_source' in combined_data.columns:
                        st.markdown("### Content Sources")
                        source_dist = combined_data['data_source'].value_counts()
                        fig = px.pie(values=source_dist.values, names=source_dist.index,
                                   title='Content Source Distribution')
                        st.plotly_chart(fig, use_container_width=True)
            
            else:  # Recommendations
                st.subheader("💡 Recommendations")
                
                # Quality Improvements
                st.markdown("### Quality Improvement Opportunities")
                low_quality = combined_data[combined_data['quality_score'] < 0.6]
                if not low_quality.empty:
                    st.warning(f"Found {len(low_quality)} courses needing quality improvements")
                    with st.expander("View Details"):
                        st.dataframe(low_quality[['course_title', 'quality_score']].head(10))
                
                # Content Gaps
                st.markdown("### Content Coverage Gaps")
                if 'region_entity' in combined_data.columns and 'course_type' in combined_data.columns:
                    coverage = pd.crosstab(
                        combined_data['region_entity'],
                        combined_data['course_type']
                    )
                    gaps = coverage[coverage < 5]
                    if not gaps.empty:
                        st.info("Areas with limited content coverage:")
                        st.dataframe(gaps)
                
                # Action Items
                st.markdown("### Recommended Actions")
                actions = []
                
                if 'quality_score' in combined_data.columns:
                    avg_quality = combined_data['quality_score'].mean()
                    if avg_quality < 0.7:
                        actions.append("Improve metadata completeness for better searchability")
                
                if 'learner_count' in combined_data.columns:
                    unused_courses = len(combined_data[combined_data['learner_count'] == 0])
                    if unused_courses > 0:
                        actions.append(f"Review {unused_courses} unused courses for retirement or promotion")
                
                for action in actions:
                    st.markdown(f"- {action}")
            
            # Export Options
            st.header("📥 Export Options")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Download Analysis Report"):
                    # Generate report logic here
                    st.info("Report generation feature coming soon!")
            
            with col2:
                if st.button("Download Processed Data"):
                    # Download data logic here
                    st.info("Data export feature coming soon!")
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main() 