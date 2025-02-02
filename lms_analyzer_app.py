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
    st.title("LMS Content Analysis Dashboard")
    
    st.write("Upload your LMS data files. You can upload multiple files for combined analysis.")
    uploaded_files = st.file_uploader("Upload Excel files", type=["xlsx"], accept_multiple_files=True)
    
    if uploaded_files:
        dataframes = []
        
        for uploaded_file in uploaded_files:
            try:
                df = pd.read_excel(uploaded_file)
                dataframes.append(df)
                st.success(f"Successfully loaded: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        if dataframes:
            # Merge and standardize all data
            combined_data = merge_and_standardize_data(dataframes)
            
            # Calculate quality scores
            combined_data = calculate_quality_score(combined_data)
            
            # Create analyzer with combined data
            combined_buffer = io.BytesIO()
            with pd.ExcelWriter(combined_buffer, engine='openpyxl') as writer:
                combined_data.to_excel(writer, index=False)
            combined_buffer.seek(0)
            analyzer = LMSContentAnalyzer(combined_buffer)
            analyzer.df = combined_data  # Use the data with quality scores
            
            # Display data overview with cross-reference information
            st.header("Data Overview")
            st.write(f"Total unique courses: {len(combined_data)}")
            st.write(f"Courses with multiple data sources: {len(combined_data[combined_data['cross_reference_count'] > 1])}")
            st.write(f"Data sources: {combined_data['data_source'].nunique()}")
            
            # Show key metrics that are available
            metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
            
            with metrics_col1:
                if 'region_entity' in combined_data.columns:
                    st.metric("Regions", combined_data['region_entity'].nunique())
                if 'course_type' in combined_data.columns:
                    st.metric("Course Types", combined_data['course_type'].nunique())
            
            with metrics_col2:
                if 'learner_count' in combined_data.columns:
                    st.metric("Total Learners", f"{combined_data['learner_count'].sum():,.0f}")
                if 'person_type' in combined_data.columns:
                    st.metric("Audience Types", combined_data['person_type'].nunique())
            
            with metrics_col3:
                if 'course_keywords' in combined_data.columns:
                    st.metric("Unique Keywords", combined_data['course_keywords'].nunique())
                if 'category_name' in combined_data.columns:
                    st.metric("Categories", combined_data['category_name'].nunique())
            
            with metrics_col4:
                if 'course_created_by' in combined_data.columns:
                    st.metric("Content Authors", combined_data['course_created_by'].nunique())
                # Calculate active courses based on available data
                active_courses = len(combined_data)
                if 'course_discontinued_from' in combined_data.columns:
                    active_courses = len(combined_data[
                        (combined_data['course_discontinued_from'].isna()) | 
                        (combined_data['course_discontinued_from'] > pd.Timestamp.now())
                    ])
                st.metric("Active Courses", active_courses)
            
            # Create tabs for different analyses
            tabs = st.tabs([
                "Data Quality",
                "Training Categories",
                "Training Focus",
                "Training Breadth",
                "Delivery Methods",
                "Content Usage",
                "Training Volume",
                "Content Production",
                "Training Types",
                "Learner Interests"
            ])
            
            with tabs[0]:
                st.header("Data Quality Overview")
                
                # Quality score distribution
                st.subheader("Quality Score Distribution")
                fig = plot_quality_distribution(combined_data)
                if fig:
                    st.plotly_chart(fig)
                
                # Metadata completeness analysis
                st.subheader("Metadata Completeness")
                completeness_stats = analyze_metadata_completeness(combined_data)
                for category, stats in completeness_stats.items():
                    st.write(f"**{category.title()} Metadata**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            "Completeness",
                            f"{stats['avg_completeness']*100:.1f}%"
                        )
                    with col2:
                        st.metric(
                            "Fields Present",
                            f"{stats['fields_present']}/{stats['total_fields']}"
                        )
                    if stats['missing_fields']:
                        st.info(f"Missing fields: {', '.join(stats['missing_fields'])}")
                
                # Cross-reference analysis
                st.subheader("Cross-Reference Analysis")
                cross_ref_stats = analyze_cross_references(combined_data)
                st.write(f"**Multi-source Courses:** {cross_ref_stats['multi_source_courses']} "
                        f"({cross_ref_stats['multi_source_courses']/cross_ref_stats['total_courses']*100:.1f}%)")
                
                if cross_ref_stats['consistency_metrics']:
                    st.write("**Field Consistency Across Sources:**")
                    for field, metrics in cross_ref_stats['consistency_metrics'].items():
                        consistent_pct = metrics['consistent']/(metrics['consistent'] + metrics['inconsistent'])*100
                        st.metric(
                            field,
                            f"{consistent_pct:.1f}% consistent"
                        )
                
                # Temporal analysis
                st.subheader("Temporal Analysis")
                temporal_stats = analyze_temporal_patterns(combined_data)
                if 'lifecycle' in temporal_stats:
                    active_pct = temporal_stats['lifecycle']['active_courses']/len(combined_data)*100
                    st.metric(
                        "Active Courses",
                        f"{temporal_stats['lifecycle']['active_courses']} ({active_pct:.1f}%)"
                    )
                
                if 'creation_patterns' in temporal_stats:
                    st.write("**Course Creation Trends**")
                    year_data = pd.Series(temporal_stats['creation_patterns']['courses_per_year'])
                    fig = px.line(x=year_data.index, y=year_data.values,
                                title='Courses Created per Year')
                    st.plotly_chart(fig)
            
            with tabs[1]:
                st.header("Training Categories Analysis")
                
                # Display training focus distribution
                focus_stats = analyze_training_focus(combined_data)
                focus_df = pd.DataFrame(focus_stats).T
                
                st.subheader("Training Category Distribution")
                fig = px.bar(focus_df, x=focus_df.index, y='percentage',
                           title='Distribution of Training Categories')
                st.plotly_chart(fig)
                
                # Show category overlaps
                st.subheader("Category Overlaps")
                overlap_matrix = pd.DataFrame(index=TRAINING_CATEGORIES.keys(),
                                           columns=TRAINING_CATEGORIES.keys())
                
                for cat1 in TRAINING_CATEGORIES.keys():
                    for cat2 in TRAINING_CATEGORIES.keys():
                        overlap = (combined_data[f'is_{cat1}'] & 
                                 combined_data[f'is_{cat2}']).sum() / len(combined_data) * 100
                        overlap_matrix.loc[cat1, cat2] = overlap
                
                fig = px.imshow(overlap_matrix,
                              title='Category Overlap Heatmap (%)',
                              labels=dict(color="Overlap %"))
                st.plotly_chart(fig)
            
            with tabs[2]:
                st.header("Training Focus Analysis")
                
                if 'person_type' in combined_data.columns:
                    st.subheader("Training by Audience")
                    audience_dist = combined_data['person_type'].value_counts()
                    fig = px.pie(values=audience_dist.values, names=audience_dist.index,
                               title='Distribution by Audience Type')
                    st.plotly_chart(fig)
                else:
                    st.info("Add person_type data to see audience distribution")
                
                if 'course_type' in combined_data.columns:
                    st.subheader("Training by Course Type")
                    type_dist = combined_data['course_type'].value_counts()
                    fig = px.bar(x=type_dist.index, y=type_dist.values,
                               title='Distribution by Course Type')
                    st.plotly_chart(fig)
            
            with tabs[3]:
                st.header("Training Breadth Analysis")
                
                if 'region_entity' in combined_data.columns:
                    st.subheader("Regional Distribution")
                    region_dist = combined_data['region_entity'].value_counts()
                    fig = px.pie(values=region_dist.values, names=region_dist.index,
                               title='Course Distribution by Region')
                    st.plotly_chart(fig)
                
                if all(col in combined_data.columns for col in ['region_entity', 'course_type']):
                    st.subheader("Course Types by Region")
                    region_type_matrix = pd.crosstab(
                        combined_data['region_entity'],
                        combined_data['course_type']
                    )
                    fig = px.imshow(region_type_matrix,
                                  title='Course Type Distribution by Region')
                    st.plotly_chart(fig)
            
            with tabs[4]:
                st.header("Delivery Methods Analysis")
                
                delivery_stats = analyze_delivery_methods(combined_data)
                if delivery_stats:
                    st.subheader("Delivery Method Distribution")
                    delivery_df = pd.DataFrame(list(delivery_stats['distribution'].items()),
                                            columns=['Method', 'Count'])
                    fig = px.pie(delivery_df, values='Count', names='Method',
                               title='Distribution of Delivery Methods')
                    st.plotly_chart(fig)
                    
                    if delivery_stats['avg_duration'] is not None:
                        st.metric("Average Course Duration (minutes)",
                                f"{delivery_stats['avg_duration']:.0f}")
                else:
                    st.info("Add delivery_method data to see delivery analysis")
            
            with tabs[5]:
                st.header("Content Usage Analysis")
                
                if 'course_available_from' in combined_data.columns:
                    st.subheader("Content Age Distribution")
                    timeline_fig = plot_timeline(combined_data, 'course_available_from')
                    if timeline_fig:
                        st.plotly_chart(timeline_fig)
                
                if 'learner_count' in combined_data.columns:
                    st.subheader("Usage Intensity")
                    fig = px.histogram(combined_data, x='learner_count',
                                     title='Distribution of Learner Count',
                                     nbins=30)
                    st.plotly_chart(fig)
            
            with tabs[6]:
                st.header("Training Volume Analysis")
                
                if 'course_created_by' in combined_data.columns:
                    st.subheader("Content Creation Volume")
                    creator_stats = combined_data['course_created_by'].value_counts()
                    fig = px.bar(x=creator_stats.index[:20], y=creator_stats.values[:20],
                               title='Top 20 Content Creators')
                    st.plotly_chart(fig)
                
                if 'learner_count' in combined_data.columns and 'course_type' in combined_data.columns:
                    st.subheader("Learning Volume by Course Type")
                    volume_by_type = combined_data.groupby('course_type')['learner_count'].sum()
                    fig = px.pie(values=volume_by_type.values, names=volume_by_type.index,
                               title='Learning Volume Distribution')
                    st.plotly_chart(fig)
            
            with tabs[7]:
                st.header("Content Production Analysis")
                
                if 'data_source' in combined_data.columns:
                    st.subheader("Content Sources")
                    source_dist = combined_data['data_source'].value_counts()
                    fig = px.pie(values=source_dist.values, names=source_dist.index,
                               title='Content Source Distribution')
                    st.plotly_chart(fig)
                
                st.subheader("Cross-Reference Analysis")
                cross_ref_dist = combined_data['cross_reference_count'].value_counts().sort_index()
                fig = px.bar(x=cross_ref_dist.index, y=cross_ref_dist.values,
                           title='Distribution of Cross-References',
                           labels={'x': 'Number of Sources', 'y': 'Number of Courses'})
                st.plotly_chart(fig)
            
            with tabs[8]:
                st.header("Training Types Analysis")
                
                if 'course_type' in combined_data.columns:
                    st.subheader("Course Type Distribution")
                    type_dist = combined_data['course_type'].value_counts()
                    fig = px.pie(values=type_dist.values, names=type_dist.index,
                               title='Distribution by Course Type')
                    st.plotly_chart(fig)
                    
                    if 'learner_count' in combined_data.columns:
                        st.subheader("Enrollment by Course Type")
                        enrollment_by_type = combined_data.groupby('course_type')['learner_count'].sum()
                        fig = px.bar(x=enrollment_by_type.index, y=enrollment_by_type.values,
                                   title='Total Enrollments by Course Type')
                        st.plotly_chart(fig)
            
            with tabs[9]:
                st.header("Learner Interests Analysis")
                
                if 'description' in combined_data.columns:
                    st.subheader("Popular Topics")
                    descriptions = combined_data['description'].dropna()
                    if not descriptions.empty:
                        wordcloud_fig = create_wordcloud(" ".join(descriptions.astype(str)))
                        st.pyplot(wordcloud_fig)
                
                if all(col in combined_data.columns for col in ['learner_count', 'course_type']):
                    st.subheader("Popular Course Types")
                    popular_types = combined_data.groupby('course_type')['learner_count'].sum().sort_values(ascending=False)
                    fig = px.bar(x=popular_types.index, y=popular_types.values,
                               title='Course Types by Popularity')
                    st.plotly_chart(fig)

if __name__ == "__main__":
    main() 