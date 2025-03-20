import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
from datetime import datetime
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import warnings
import streamlit as st

# Suppress specific warnings that we're handling appropriately
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

# Update and expand constants to match dashboard requirements exactly
EDUCATION_HISTORY_METRICS = {
    'completion_tracking': {
        'high_volume': '5k+ completions',
        'client_volume': 'high volume clients (>5%)'
    }
}

TRAINING_CATEGORIES = {
    'Leadership Development': [],
    'Managerial and Supervisory': [],
    'Mandatory and Compliance': ['look at overlap in NEHS'],
    'Profession-specific or Industry-specific': [],
    'Interpersonal Skills': ['overlap between KP authored vs SkillSoft'],
    'Processes, Procedures, and Business Practices': [],
    'Information Technology and Systems': [],
    'New Hire Orientation': [],
    'Customer Service': [],
    'Product Knowledge': [],
    'Sales': ['not including product knowledge'],
    'Clinical': [],
    'Nursing': [],
    'Pharmacy': [],
    'Diversity': [],
    'Safety': [],
    'Specialized Courses': ['not using KP Learn, Sharepoint, Web-based repositories, etc.']
}

TRAINING_FOCUS_PERSONAS = {
    'employees': 'general staff',
    'managers': 'management level',
    'leaders': 'leadership level',
    'new_hires': 'onboarding',
    'function_specific': ['pharmacy', 'clinical', 'IT']
}

TRAINING_BREADTH = {
    'enterprise_wide': ['compliance', 'safety', 'SkillSoft'],
    'market_function': 'role specific',
    'specific_setting': 'context specific'
}

DELIVERY_METHODS = {
    'instructor_led_in_person': 'classroom based',
    'instructor_led_virtual': 'virtual live',
    'self_paced_elearning': ['> 20 minutes'],
    'microlearning': ['byte sized', '< 20 minutes']
}

CONTENT_USAGE_LENGTH = {
    '1_year': 'available in last year',
    '2_3_years': 'available 2-3 years ago',
    'longer': 'available more than 3 years'
}

TRAINING_ORGANIZATIONS = {
    'Enterprise Learning Design & Delivery': 'created by someone in EL',
    'National Leadership Development': 'NLD dept',
    'Clinical Education': 'CE dept',
    'Market L&D teams': 'market specific',
    'HR12 compliance': 'compliance focused',
    'NEH&S': 'safety and environmental',
    'TRO': 'technical',
    'Marketing, Sales, Underwriting, HPI': 'business focused',
    'NFS': 'specialized'
}

CONTENT_SOURCES = {
    'in_house': 'hard to tell who created the content',
    'coordinator_deployed': 'content deployed from elsewhere',
    'custom_vendor': 'not always reported correctly in deployment',
    'url_internal': 'SharePoint',
    'url_external': ['YouTube', 'LinkedIn'],
    'off_the_shelf': ['SkillSoft', 'RQI', 'etc']
}

MARKET_REGIONS = {
    'NCAL': 'Northern California',
    'SCAL': 'Southern California',
    'CO': 'Colorado',
    'GA': 'Georgia',
    'Health Plan': 'Health Plan specific',
    'MAS': 'Mid-Atlantic States',
    'KPWA': 'Washington'
}

LEARNER_INTERESTS = {
    'Business Skills': [],
    'Technical Skills': [],
    'Personal Development Skills': [],
    'Higher Education and Certifications': [],
    'Misc': []
}

# Update constants section with new required fields
REQUIRED_FINANCIAL_COLUMNS = {
    'vendor': 'Training system or content vendor',
    'cost': 'Cost per course or content item',
    'cost_per_learner': 'Cost per individual learner',
    'learner_type': 'Type of learner (e.g., employee, manager, etc.)',
    'reimbursement_amount': 'Tuition reimbursement amount',
    'program_type': 'Type of educational program'
}

REQUIRED_ADMIN_COLUMNS = {
    'market': 'Market or region designation',
    'admin_count': 'Number of administrators',
    'course_id': 'Unique course identifier',
    'class_id': 'Unique class identifier',
    'support_tickets': 'Number of support tickets',
    'resolution_time': 'Average ticket resolution time',
    'satisfaction': 'User satisfaction score'
}

class LMSContentAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.text_columns = []
        self.date_columns = []
        self.categorical_columns = []
        self._load_and_prepare_data()

    def _load_and_prepare_data(self):
        try:
            self.df = pd.read_excel(self.file_path)
            
            # Standardize column names based on the tables in the image
            standard_mappings = {
                # Table 1: Education History Tracking
                'Completions': 'completion_count',
                'Client Volume': 'client_percentage',
                'Ed History Clients': 'number_of_clients',
                
                # Table 2: Training Categories (handled by TRAINING_CATEGORIES)
                'Category': 'training_category',
                
                # Table 3: Training Focus
                'Person Type': 'persona_type',
                'Function': 'function_specific',
                
                # Table 4: Training Breadth
                'Region Entity': 'region_entity',
                'Market/Function': 'market_function',
                'Setting': 'specific_setting',
                
                # Table 5: Training Delivery
                'Delivery Method': 'delivery_method',
                'Duration': 'duration_minutes',
                
                # Table 6: Content Usage Length
                'Available From': 'available_from',
                'Usage Period': 'usage_period',
                
                # Table 7: Training Volume
                'Organization': 'training_organization',
                'Volume': 'training_volume',
                
                # Table 8: Content Production
                'Source': 'content_source',
                'Development': 'development_type',
                
                # Table 9: Training Assignment
                'Assignment Type': 'assignment_type',
                'Hours': 'training_hours',
                
                # Table 10: Learner Interests
                'Interest Area': 'interest_area',
                'Interest Level': 'interest_percentage'
            }
            
            # Rename columns based on standard mappings
            self.df.rename(columns=standard_mappings, inplace=True)
            
            self._clean_column_names()
            self._classify_columns()
            self._convert_date_columns()
            
            # No longer filtering out duplicate rows - keeping all data
            # This comment replaces the previous code that removed duplicates
            
            print(f"\nSuccessfully loaded {len(self.df)} rows of data")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _clean_column_names(self):
        # Print original column names
        print("\nOriginal column names:")
        for col in self.df.columns:
            print(f"- {col}")

        try:
            # Use the enhanced column cleaning function from optimizations
            from src.utils.optimizations import clean_column_names
            self.df = clean_column_names(self.df)
            print("\nApplied enhanced column name standardization")
        except ImportError:
            print("\nFalling back to basic column name cleaning")
            # Fallback to original behavior if import fails
            column_mapping = {
                'Course Title': 'course_title',
                'Course No': 'course_no',
                'Course Description': 'course_description',
                'Course Abstract': 'course_abstract',
                'Course Version': 'course_version',
                'Course Created By': 'course_created_by',
                'Course Available From': 'course_available_from',
                'Course Discontinued From': 'course_discontinued_from',
                'Course Keywords': 'course_keywords',
                'Category Name': 'category_name'
            }

            # Print mapping
            print("\nColumn name mapping:")
            for old, new in column_mapping.items():
                print(f"- '{old}' -> '{new}'")

            # Rename columns
            self.df = self.df.rename(columns=column_mapping)

    def _classify_columns(self):
        print("\nColumn Classification:\n")
        
        # Identify date columns
        self.date_columns = [
            'course_version',
            'course_available_from',
            'course_discontinued_from'
        ]
        print(f"Date columns: {self.date_columns}")

        # Identify text columns
        self.text_columns = [
            'course_title',
            'course_no',
            'course_description',
            'course_abstract',
            'course_keywords'
        ]
        print(f"Text columns: {self.text_columns}")

        # Identify categorical columns
        self.categorical_columns = [
            'course_created_by',
            'category_name'
        ]
        print(f"Categorical columns: {self.categorical_columns}")

    def _convert_date_columns(self):
        """Convert date columns to datetime with proper error handling"""
        for col in self.date_columns:
            if col in self.df.columns:
                try:
                    # Convert to datetime and handle any format issues
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                    
                    # Report on conversion results
                    total_rows = len(self.df)
                    valid_dates = self.df[col].notna().sum()
                    if valid_dates < total_rows:
                        print(f"Warning: {col} has {total_rows - valid_dates} invalid or missing dates")
                    
                    # Add quality flag for the column
                    self.df[f"{col}_is_valid"] = self.df[col].notna()
                    
                except Exception as e:
                    print("Error converting", col, "to datetime:", str(e))  # Fixed string formatting
                    # Create a flag column even if conversion fails
                    self.df[f"{col}_is_valid"] = False

    def get_data_quality_metrics(self):
        """Calculate overall data quality metrics"""
        metrics = {
            'completeness': 0,
            'consistency': 0,
            'validity': 0
        }
        
        # Calculate completeness
        completeness_scores = []
        for col in self.df.columns:
            if not col.endswith('_is_valid'):  # Skip validity flag columns
                non_null = self.df[col].notna().mean() * 100
                completeness_scores.append(non_null)
        metrics['completeness'] = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0
        
        # Calculate consistency
        consistency_scores = []
        for col in self.df.columns:
            if col.endswith('_is_valid'):
                continue
                
            try:
                if col in self.date_columns:
                    # Use validity flags for date columns
                    valid_flag = f"{col}_is_valid"
                    if valid_flag in self.df.columns:
                        consistency_scores.append(self.df[valid_flag].mean() * 100)
                
                elif col in self.text_columns:
                    # Check for consistent text patterns
                    values = self.df[col].fillna('').astype(str)
                    avg_length = values.str.len().mean()
                    std_length = values.str.len().std()
                    consistency_score = 100 * (1 - (std_length / avg_length if avg_length > 0 else 1))
                    consistency_scores.append(max(0, min(100, consistency_score)))
                
                else:
                    # For other columns, check value distribution
                    value_counts = self.df[col].value_counts(normalize=True)
                    if not value_counts.empty:
                        entropy = -(value_counts * np.log(value_counts)).sum()
                        consistency_score = 100 * (1 - min(1, entropy / 4))  # Normalize entropy
                        consistency_scores.append(consistency_score)
            
            except Exception as e:
                print(f"Warning: Could not check consistency for {col}: {str(e)}")
        
        metrics['consistency'] = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0
        
        # Calculate validity
        validity_scores = []
        for col in self.df.columns:
            if col.endswith('_is_valid'):
                continue
                
            try:
                if col in self.date_columns:
                    # Use validity flags for dates
                    valid_flag = f"{col}_is_valid"
                    if valid_flag in self.df.columns:
                        valid_dates = self.df[valid_flag].mean() * 100
                        future_dates = (
                            (self.df[col] > pd.Timestamp.now()).mean() * 100 
                            if self.df[valid_flag].any() else 0
                        )
                        validity_scores.append(valid_dates * (1 - future_dates/100))
                
                elif col == 'course_no':
                    # Validate course numbers
                    valid_format = self.df[col].apply(
                        lambda x: bool(re.match(r'^[A-Za-z0-9-]+$', str(x))) if pd.notna(x) else False
                    )
                    validity_scores.append(valid_format.mean() * 100)
                
                elif col in self.text_columns:
                    # Check for reasonable text content
                    values = self.df[col].fillna('').astype(str)
                    valid_length = values.str.len().between(1, 5000).mean() * 100
                    validity_scores.append(valid_length)
                
                else:
                    # Basic validity check for other columns
                    valid_values = (
                        self.df[col].notna() & 
                        (self.df[col].astype(str).str.strip() != '')
                    ).mean() * 100
                    validity_scores.append(valid_values)
            
            except Exception as e:
                print(f"Warning: Could not check validity for {col}: {str(e)}")
        
        metrics['validity'] = sum(validity_scores) / len(validity_scores) if validity_scores else 0
        
        return metrics

    def get_missing_data_summary(self):
        missing_data = {}
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_percentage = (missing_count / len(self.df)) * 100
            missing_data[col] = {
                'count': missing_count,
                'percentage': missing_percentage
            }
        return missing_data

    def get_value_distributions(self):
        distributions = {}
        for col in self.categorical_columns:
            if col in self.df.columns:
                value_counts = self.df[col].value_counts().head(10)
                distributions[col] = value_counts.to_dict()
        return distributions

    def get_date_ranges(self):
        date_ranges = {}
        for col in self.date_columns:
            if col in self.df.columns:
                try:
                    date_ranges[col] = {
                        'min': self.df[col].min(),
                        'max': self.df[col].max()
                    }
                except Exception as e:
                    print(f"Warning: Could not get date range for {col}: {str(e)}")
        return date_ranges

    def get_text_field_stats(self):
        text_stats = {}
        for col in self.text_columns:
            if col in self.df.columns:
                try:
                    lengths = self.df[col].fillna('').astype(str).str.len()
                    text_stats[col] = {
                        'min_length': lengths.min(),
                        'max_length': lengths.max(),
                        'avg_length': lengths.mean()
                    }
                except Exception as e:
                    print(f"Warning: Could not get text stats for {col}: {str(e)}")
        return text_stats

    def get_recommendations(self):
        recommendations = []
        
        # Check completeness
        missing_data = self.get_missing_data_summary()
        for col, stats in missing_data.items():
            if stats['percentage'] > 20:
                recommendations.append(f"High missing data in {col}: {stats['percentage']:.1f}% missing")

        # Check date validity
        date_ranges = self.get_date_ranges()
        for col, ranges in date_ranges.items():
            if ranges['min'] is pd.NaT or ranges['max'] is pd.NaT:
                recommendations.append(f"Invalid dates found in {col}")

        # Check text field quality
        text_stats = self.get_text_field_stats()
        for col, stats in text_stats.items():
            if stats['avg_length'] < 5:
                recommendations.append(f"Short text content in {col}: average length {stats['avg_length']:.1f} characters")

        return recommendations

    def calculate_total_training_hours(self):
        """Calculate total training hours across all courses"""
        if 'duration_hours' not in self.df.columns:
            # Add a default duration if not present
            self.df['duration_hours'] = 1.0
        return self.df['duration_hours'].sum()
    
    def calculate_avg_duration(self):
        """Calculate average course duration in hours"""
        if 'duration_hours' not in self.df.columns:
            # Add a default duration if not present
            self.df['duration_hours'] = 1.0
        return self.df['duration_hours'].mean()

    def get_category_distribution(self):
        """Get training distribution by broad categories"""
        if 'category_name' not in self.df.columns:
            st.warning("⚠️ No explicit category data found. Would you like to infer categories from course titles and descriptions?")
            use_inference = st.checkbox("Use AI to infer categories", value=False)
            
            if not use_inference:
                st.info("Please add category data to your Excel file for category analysis.")
                return None
            
            # If user opts for inference
            confidence_threshold = st.slider(
                "Minimum confidence threshold for category inference",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                help="Only include categories inferred with confidence above this threshold"
            )
            
            # Create a copy to avoid SettingWithCopyWarning
            df_copy = self.df.copy()
            df_copy['category_type'], df_copy['inference_confidence'] = zip(
                *df_copy.apply(lambda x: self._infer_category_with_confidence(x, confidence_threshold), axis=1)
            )
            
            # Only include high-confidence predictions
            df_copy = df_copy[df_copy['inference_confidence'] >= confidence_threshold]
            
            if len(df_copy) == 0:
                st.warning("No categories could be inferred with sufficient confidence.")
                return None
            
            category_counts = df_copy.groupby(['category_type']).agg({
                'course_title': 'count',
                'inference_confidence': 'mean'
            }).reset_index()
            
            category_counts.columns = ['category_type', 'count', 'avg_confidence']
            return category_counts
            
        # If we have actual category data
        df_copy = self.df.copy()
        df_copy['category_type'] = df_copy['category_name'].apply(self._categorize_course)
        
        category_counts = df_copy.groupby(['category_type', 'category_name']).size().reset_index()
        category_counts.columns = ['category_type', 'subcategory', 'count']
        return category_counts

    def _infer_category_with_confidence(self, row, confidence_threshold=0.7):
        """Infer category with confidence score using course metadata"""
        title = str(row['course_title']).lower() if pd.notna(row.get('course_title')) else ''
        desc = str(row['course_description']).lower() if pd.notna(row.get('course_description')) else ''
        
        # Define category keywords with weights
        category_keywords = {
            'Leadership Development': {
                'keywords': ['leadership', 'executive', 'management', 'strategy'],
                'weights': [1.0, 0.9, 0.8, 0.7]
            },
            'Technical Skills': {
                'keywords': ['technical', 'software', 'programming', 'data'],
                'weights': [1.0, 0.9, 0.8, 0.7]
            },
            'Compliance': {
                'keywords': ['compliance', 'regulatory', 'legal', 'policy'],
                'weights': [1.0, 0.9, 0.8, 0.7]
            }
        }
        
        # Calculate confidence scores for each category
        text = f"{title} {desc}"
        scores = {}
        
        for category, data in category_keywords.items():
            score = 0
            for keyword, weight in zip(data['keywords'], data['weights']):
                if keyword in text:
                    score += weight
            scores[category] = score / len(data['keywords'])
        
        # Get category with highest confidence
        if scores:
            best_category = max(scores.items(), key=lambda x: x[1])
            return best_category if best_category[1] >= confidence_threshold else ('Uncategorized', 0.0)
        
        return ('Uncategorized', 0.0)

    def get_content_gaps(self):
        """Analyze content gaps across different dimensions"""
        gaps = {
            'category_gaps': [],
            'skill_gaps': [],
            'audience_gaps': [],
            'confidence_scores': {}
        }
        
        # Category coverage analysis
        if 'category_name' in self.df.columns:
            category_counts = self.df['category_name'].value_counts()
            avg_count = category_counts.mean()
            gaps['category_gaps'] = [
                {
                    'category': cat,
                    'current_count': count,
                    'gap_percentage': ((avg_count - count) / avg_count) * 100,
                    'is_inferred': False
                }
                for cat, count in category_counts.items()
                if count < avg_count * 0.5
            ]
        else:
            st.warning("⚠️ No explicit category data available for gap analysis")
        
        # Skill coverage analysis (using NLP on descriptions)
        if 'course_description' in self.df.columns:
            skill_threshold = st.slider(
                "Skill extraction confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.6,
                help="Minimum confidence for skill extraction"
            )
            
            skills_data = self._extract_skills_with_confidence(skill_threshold)
            if skills_data:
                gaps['skill_gaps'] = skills_data['gaps']
                gaps['confidence_scores']['skills'] = skills_data['confidence']
        
        return gaps

    def _extract_skills_with_confidence(self, confidence_threshold=0.6):
        """Extract skills from course descriptions with confidence scores"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import DBSCAN
            
            # Prepare text data
            descriptions = self.df['course_description'].fillna('').astype(str)
            
            # Extract potential skill phrases
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                stop_words='english',
                max_features=1000
            )
            tfidf_matrix = vectorizer.fit_transform(descriptions)
            
            # Cluster similar skills
            clustering = DBSCAN(eps=0.3, min_samples=3)
            clusters = clustering.fit_predict(tfidf_matrix)
            
            # Calculate confidence scores based on cluster density
            unique_clusters = set(clusters)
            skill_clusters = {}
            confidence_scores = {}
            
            for cluster_id in unique_clusters:
                if cluster_id != -1:  # Ignore noise points
                    cluster_docs = descriptions[clusters == cluster_id]
                    top_terms = vectorizer.get_feature_names_out()
                    
                    # Calculate term importance in cluster
                    cluster_tfidf = tfidf_matrix[clusters == cluster_id].mean(axis=0).A1
                    top_indices = cluster_tfidf.argsort()[-5:][::-1]  # Top 5 terms
                    
                    for idx in top_indices:
                        term = top_terms[idx]
                        confidence = float(cluster_tfidf[idx])
                        if confidence >= confidence_threshold:
                            skill_clusters[term] = len(cluster_docs)
                            confidence_scores[term] = confidence
            
            # Identify gaps based on skill distribution
            avg_skill_count = np.mean(list(skill_clusters.values()))
            gaps = [
                {
                    'skill': skill,
                    'current_count': count,
                    'confidence': confidence_scores[skill],
                    'gap_percentage': ((avg_skill_count - count) / avg_skill_count) * 100
                }
                for skill, count in skill_clusters.items()
                if count < avg_skill_count * 0.5
            ]
            
            return {
                'gaps': sorted(gaps, key=lambda x: x['gap_percentage'], reverse=True),
                'confidence': confidence_scores
            }
            
        except Exception as e:
            st.error(f"Error in skill extraction: {str(e)}")
            return None

    def analyze_content_quality(self):
        """Analyze content quality with transparency about metrics"""
        quality_metrics = {
            'metrics': {},
            'confidence': {},
            'recommendations': []
        }
        
        # Completeness analysis
        completeness = {}
        for col in self.df.columns:
            non_null = self.df[col].notna().mean() * 100
            completeness[col] = {
                'score': non_null,
                'is_inferred': False
            }
        quality_metrics['metrics']['completeness'] = completeness
        
        # Content richness analysis
        if 'course_description' in self.df.columns:
            desc_lengths = self.df['course_description'].fillna('').str.len()
            quality_metrics['metrics']['content_richness'] = {
                'avg_description_length': desc_lengths.mean(),
                'short_descriptions': (desc_lengths < 100).sum(),
                'is_inferred': False
            }
        
        # Metadata quality
        metadata_scores = {}
        for col in self.df.columns:
            if col in self.text_columns:
                valid_values = self._validate_text_field(col)
                metadata_scores[col] = {
                    'score': valid_values['score'],
                    'is_inferred': valid_values['is_inferred'],
                    'confidence': valid_values.get('confidence', 1.0)
                }
        quality_metrics['metrics']['metadata_quality'] = metadata_scores
        
        return quality_metrics

    def _validate_text_field(self, column):
        """Validate text field quality with confidence scores"""
        if column not in self.df.columns:
            return {'score': 0, 'is_inferred': True, 'confidence': 0}
            
        values = self.df[column].fillna('')
        
        # Basic validation
        non_empty = values.str.len() > 0
        score = non_empty.mean() * 100
        
        # Check for potentially problematic patterns
        problematic = values.str.contains(r'test|dummy|placeholder', case=False, regex=True)
        score = score * (1 - problematic.mean())
        
        return {
            'score': score,
            'is_inferred': False,
            'confidence': 1.0
        }

    def get_persona_distribution(self):
        """Get training distribution by persona"""
        if 'course_title' not in self.df.columns:
            # Create mock persona distribution data
            return pd.DataFrame({
                'persona': ['employees', 'managers', 'leaders', 'new hires'],
                'percentage': [50, 25, 15, 10]
            })
            
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = self.df.copy()
        df_copy['target_persona'] = df_copy.apply(self._infer_persona, axis=1)
        
        persona_dist = df_copy['target_persona'].value_counts(normalize=True).reset_index()
        persona_dist.columns = ['persona', 'percentage']
        persona_dist['percentage'] = persona_dist['percentage'] * 100
        return persona_dist

    def get_top_learner_interests(self):
        """Get top 10 learner interests based on course keywords and titles"""
        if 'course_keywords' not in self.df.columns and 'course_title' not in self.df.columns:
            # Create mock learner interests data
            return pd.DataFrame({
                'interest': ['Leadership', 'Technical Skills', 'Communication', 'Project Management',
                           'Data Analysis', 'Safety', 'Customer Service', 'Quality Management',
                           'Risk Management', 'Compliance'],
                'count': [150, 130, 120, 110, 100, 90, 80, 70, 60, 50]
            })
            
        # Extract interests from keywords and course titles
        interests = []
        for _, row in self.df.iterrows():
            interests.extend(self._extract_interests(row))
        
        interest_counts = pd.Series(interests).value_counts().head(10).reset_index()
        interest_counts.columns = ['interest', 'count']
        return interest_counts

    def get_delivery_split(self):
        """Get training delivery method distribution"""
        if 'delivery_method' not in self.df.columns:
            # Create mock delivery method distribution
            return pd.DataFrame({
                'method': ['Virtual', 'Self-paced', 'Instructor-led'],
                'percentage': [45, 35, 20]  # Example distribution
            })
        
        delivery_dist = self.df['delivery_method'].value_counts(normalize=True).reset_index()
        delivery_dist.columns = ['method', 'percentage']
        delivery_dist['percentage'] = delivery_dist['percentage'] * 100
        return delivery_dist

    def get_content_usage_length(self):
        """Get content usage length distribution"""
        # Create a default duration category if duration_hours doesn't exist
        if 'duration_hours' not in self.df.columns:
            # Create a mock distribution for demonstration
            return pd.DataFrame({
                'duration': ['Short (< 1hr)', 'Medium (1-3hrs)', 'Long (> 3hrs)'],
                'percentage': [40, 35, 25]  # Example distribution
            })
        
        # If duration_hours exists, use actual data
        self.df['duration_category'] = pd.cut(
            self.df['duration_hours'],
            bins=[0, 1, 3, float('inf')],
            labels=['Short (< 1hr)', 'Medium (1-3hrs)', 'Long (> 3hrs)']
        )
        
        usage_dist = self.df['duration_category'].value_counts(normalize=True).reset_index()
        usage_dist.columns = ['duration', 'percentage']
        usage_dist['percentage'] = usage_dist['percentage'] * 100
        return usage_dist

    def get_content_source_distribution(self):
        """Get content source distribution"""
        if 'content_source' not in self.df.columns:
            # Create mock content source distribution
            return pd.DataFrame({
                'source': ['Internal', 'Vendor', 'External'],
                'percentage': [50, 30, 20]  # Example distribution
            })
        
        source_dist = self.df['content_source'].value_counts(normalize=True).reset_index()
        source_dist.columns = ['source', 'percentage']
        source_dist['percentage'] = source_dist['percentage'] * 100
        return source_dist

    def get_training_volume_by_org(self):
        """Get training volume by organization type"""
        if 'organization' not in self.df.columns:
            # Create mock organization volume data
            return pd.DataFrame({
                'organization': ['Enterprise Learning', 'Clinical Education', 'Market L&D', 'HR12 Compliance', 'NEH&S'],
                'volume': [2500, 2000, 1500, 1000, 500]  # Example volumes
            })
        
        org_volume = self.df.groupby('organization').size().reset_index()
        org_volume.columns = ['organization', 'volume']
        return org_volume

    def get_training_cost_metrics(self):
        """Get training cost metrics"""
        # Return mock cost metrics if actual data is not available
        if 'cost' not in self.df.columns:
            return {
                'avg_cost_per_learner': 250.00,  # Example average cost
                'total_training_spend': 2500000.00,  # Example total spend
                'tuition_reimbursement': 100000.00  # Example reimbursement
            }
        
        return {
            'avg_cost_per_learner': self._calculate_avg_cost_per_learner(),
            'total_training_spend': self._calculate_total_training_spend(),
            'tuition_reimbursement': self._calculate_tuition_reimbursement()
        }

    def get_cost_by_role(self):
        """Get training cost breakdown by role"""
        if 'role' not in self.df.columns or 'cost' not in self.df.columns:
            # Create mock cost by role data
            return pd.DataFrame({
                'role': ['leader/executive', 'manager', 'employee', 'student/trainee'],
                'yearly_average': [5000, 3000, 1500, 1000]  # Example costs
            })
        
        cost_by_role = self.df.groupby('role')['cost'].mean().reset_index()
        cost_by_role.columns = ['role', 'yearly_average']
        return cost_by_role

    def get_admin_split(self):
        """Get administrative split by market/function"""
        if 'market' not in self.df.columns:
            # Map courses to markets
            self.df['market'] = self.df.apply(self._map_to_market, axis=1)
        
        market_split = self.df['market'].value_counts(normalize=True).reset_index()
        market_split.columns = ['market', 'percentage']
        market_split['percentage'] = market_split['percentage'] * 100
        return market_split

    def get_admin_efficiency_metrics(self):
        """Get administrative efficiency metrics"""
        metrics = {
            'courses_per_admin': self._calculate_courses_per_admin(),
            'classes_per_builder': self._calculate_classes_per_builder(),
            'reports_per_admin': self._calculate_reports_per_admin()
        }
        return metrics

    # Helper methods for categorization and inference
    def _categorize_course(self, category_name):
        """Map course to broad category type"""
        if pd.isna(category_name):
            return 'Other'
            
        category_name = str(category_name).lower()
        category_mapping = {
            'Leadership': 'Leadership Development',
            'Management': 'Managerial and Supervisory',
            'Compliance': 'Mandatory and Compliance',
            'Technical': 'Technical Skills',
            'Professional': 'Professional Development'
        }
        for key, value in category_mapping.items():
            if key.lower() in category_name:
                return value
        return 'Other'

    def _infer_persona(self, row):
        """Infer target persona from course metadata"""
        title = str(row['course_title']).lower() if pd.notna(row.get('course_title')) else ''
        
        if 'leader' in title or 'executive' in title:
            return 'leaders'
        elif 'manager' in title or 'supervisor' in title:
            return 'managers'
        elif 'new hire' in title or 'onboarding' in title:
            return 'new hires'
        return 'employees'

    def _infer_delivery_method(self, row):
        """Infer delivery method from course metadata"""
        title = str(row['course_title']).lower() if pd.notna(row.get('course_title')) else ''
        
        if 'virtual' in title or 'online' in title:
            return 'virtual'
        elif 'self-paced' in title or 'elearning' in title:
            return 'self-paced'
        return 'instructor-led'

    def _infer_content_source(self, row):
        """Infer content source from metadata"""
        created_by = str(row['course_created_by']).lower() if pd.notna(row.get('course_created_by')) else ''
        
        if 'vendor' in created_by:
            return 'vendor'
        elif 'internal' in created_by:
            return 'internal'
        return 'external'

    def _map_to_organization(self, row):
        """Map course to organization type"""
        org_keywords = {
            'Enterprise Learning': ['enterprise', 'corporate'],
            'Clinical Education': ['clinical', 'medical'],
            'Market L&D': ['market', 'sales'],
            'HR12 Compliance': ['compliance', 'hr'],
            'NEH&S': ['safety', 'environmental']
        }
        
        title = str(row['course_title']).lower() if pd.notna(row.get('course_title')) else ''
        description = str(row['course_description']).lower() if pd.notna(row.get('course_description')) else ''
        course_text = f"{title} {description}"
        
        for org, keywords in org_keywords.items():
            if any(keyword in course_text for keyword in keywords):
                return org
        return 'Other'

    def _map_to_market(self, row):
        """Map course to market segment"""
        market_keywords = {
            'NCAL': ['northern california'],
            'SCAL': ['southern california'],
            'CO': ['colorado'],
            'GA': ['georgia'],
            'MAS': ['mid-atlantic']
        }
        
        title = str(row['course_title']).lower() if pd.notna(row.get('course_title')) else ''
        description = str(row['course_description']).lower() if pd.notna(row.get('course_description')) else ''
        course_text = f"{title} {description}"
        
        for market, keywords in market_keywords.items():
            if any(keyword in course_text for keyword in keywords):
                return market
        return 'Other'

    def _calculate_avg_cost_per_learner(self):
        """Calculate average cost per learner"""
        if 'cost' not in self.df.columns:
            self.df['cost'] = 100  # Default cost
        if 'learner_count' not in self.df.columns:
            self.df['learner_count'] = 10  # Default learner count
        return (self.df['cost'] * self.df['learner_count']).sum() / self.df['learner_count'].sum()

    def _calculate_total_training_spend(self):
        """Calculate total training spend"""
        if 'cost' not in self.df.columns:
            self.df['cost'] = 100  # Default cost
        return self.df['cost'].sum()

    def _calculate_tuition_reimbursement(self):
        """Calculate total tuition reimbursement"""
        if 'tuition_reimbursement' not in self.df.columns:
            return 0  # Return 0 if no tuition reimbursement data
        return self.df['tuition_reimbursement'].sum()

    def _calculate_courses_per_admin(self):
        """Calculate courses per admin ratio"""
        return 50  # Placeholder - implement actual calculation

    def _calculate_classes_per_builder(self):
        """Calculate classes per builder ratio"""
        return 30  # Placeholder - implement actual calculation

    def _calculate_reports_per_admin(self):
        """Calculate reports per admin ratio"""
        return 20  # Placeholder - implement actual calculation

    def _extract_interests(self, row):
        """Extract learning interests from course metadata"""
        interests = []
        if pd.notna(row.get('course_keywords')):
            interests.extend(str(row['course_keywords']).split(','))
        if pd.notna(row.get('course_title')):
            interests.extend(row['course_title'].split())
        return [i.strip().lower() for i in interests if len(i.strip()) > 2]

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        stats = {
            'total_rows': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return stats

    def analyze_text_fields(self, min_word_length: int = 3) -> Dict[str, Any]:
        """Analyze text fields for common patterns and keywords."""
        text_analysis = {}
        for col in self.text_columns:
            # Combine all text in the column
            all_text = ' '.join(self.df[col].astype(str).fillna(''))
            # Extract words
            words = re.findall(r'\b\w+\b', all_text.lower())
            # Count words with minimum length
            word_freq = Counter([w for w in words if len(w) >= min_word_length])
            text_analysis[col] = {
                'top_keywords': dict(word_freq.most_common(10)),
                'avg_length': self.df[col].astype(str).str.len().mean(),
                'unique_values': self.df[col].nunique()
            }
        return text_analysis

    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal patterns in the data."""
        temporal_analysis = {}
        for col in self.date_columns:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                temporal_analysis[col] = {
                    'date_range': {
                        'start': self.df[col].min(),
                        'end': self.df[col].max(),
                        'span_days': (self.df[col].max() - self.df[col].min()).days if not pd.isna(self.df[col].max()) and not pd.isna(self.df[col].min()) else 0
                    },
                    'distribution': {
                        'by_year': self.df[col].dt.year.value_counts().to_dict(),
                        'by_month': self.df[col].dt.month.value_counts().to_dict(),
                        'by_weekday': self.df[col].dt.day_name().value_counts().to_dict()
                    }
                }
            except Exception as e:
                print(f"Warning: Could not analyze dates for column '{col}': {str(e)}")
                continue
        return temporal_analysis

    def analyze_categories(self) -> Dict[str, Any]:
        """Analyze categorical fields and their relationships."""
        category_analysis = {}
        for col in self.categorical_columns:
            category_analysis[col] = {
                'value_counts': self.df[col].value_counts().to_dict(),
                'null_percentage': (self.df[col].isnull().sum() / len(self.df)) * 100,
                'unique_values': self.df[col].nunique()
            }
        return category_analysis

    def calculate_quality_scores(self) -> None:
        """Calculate quality scores for each course based on multiple factors."""
        # Initialize quality metrics
        self.df['quality_score'] = 0.0
        
        # Remove any existing completeness columns
        completeness_cols = [col for col in self.df.columns if col.endswith('_complete')]
        self.df = self.df.drop(columns=completeness_cols, errors='ignore')
        
        # 1. Description completeness (length and uniqueness)
        if 'course_description' in self.df.columns:
            desc_lengths = self.df['course_description'].str.len()
            self.df['description_score'] = (desc_lengths - desc_lengths.min()) / (desc_lengths.max() - desc_lengths.min())
            self.df['quality_score'] += self.df['description_score'] * 0.3
        
        # 2. Metadata completeness
        original_columns = [col for col in self.df.columns if not col.endswith(('_score', '_complete'))]
        for col in original_columns:
            self.df[f'{col}_complete'] = ~self.df[col].isna()
        
        completeness_cols = [col for col in self.df.columns if col.endswith('_complete')]
        metadata_completeness = self.df[completeness_cols].mean(axis=1)
        self.df['metadata_score'] = metadata_completeness
        self.df['quality_score'] += metadata_completeness * 0.3
        
        # 3. Content freshness
        if 'course_available_from' in self.df.columns:
            self.df['course_available_from'] = pd.to_datetime(self.df['course_available_from'], errors='coerce')
            latest_date = self.df['course_available_from'].max()
            date_scores = 1 - ((latest_date - self.df['course_available_from']).dt.days / 365) / 5  # Normalize by 5 years
            self.df['freshness_score'] = date_scores.clip(0, 1)
            self.df['quality_score'] += self.df['freshness_score'] * 0.2
        
        # 4. Keyword richness
        if 'course_keywords' in self.df.columns:
            keyword_counts = self.df['course_keywords'].str.count(',') + 1
            self.df['keyword_score'] = (keyword_counts - keyword_counts.min()) / (keyword_counts.max() - keyword_counts.min())
            self.df['quality_score'] += self.df['keyword_score'] * 0.2
        
        # Normalize final quality score
        self.df['quality_score'] = self.df['quality_score'].clip(0, 1)
        
        # Flag courses needing attention
        self.df['needs_attention'] = self.df['quality_score'] < 0.6

    def analyze_content_similarity(self) -> None:
        """Analyze content similarity and identify potential duplicates or related courses."""
        if 'course_description' in self.df.columns:
            # Create TF-IDF matrix from descriptions
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.df['course_description'].fillna(''))
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar courses
            for idx in range(len(self.df)):
                similar_idx = np.where(similarity_matrix[idx] > 0.5)[0]
                similar_idx = similar_idx[similar_idx != idx]  # Exclude self
                if len(similar_idx) > 0:
                    self.similar_courses[self.df.index[idx]] = {
                        'course': self.df.iloc[idx]['course_title'],
                        'similar_to': [
                            {
                                'title': self.df.iloc[i]['course_title'],
                                'similarity': similarity_matrix[idx][i],
                                'index': i
                            }
                            for i in similar_idx
                        ]
                    }
            
            # Cluster courses by content
            n_clusters = min(8, len(self.df))  # Maximum 8 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.df['content_cluster'] = kmeans.fit_predict(tfidf_matrix.toarray())

    def get_actionable_insights(self) -> Dict[str, Any]:
        """Generate actionable insights and recommendations."""
        insights = {
            'urgent_attention_needed': [],
            'content_gaps': [],
            'quality_improvements': [],
            'content_redundancies': [],
            'category_distribution': [],
            'temporal_insights': []
        }
        
        # 1. Courses needing urgent attention
        low_quality_courses = self.df[self.df['quality_score'] < 0.4]
        if not low_quality_courses.empty:
            insights['urgent_attention_needed'].extend([
                {
                    'course_title': row['course_title'],
                    'course_no': row['course_no'],
                    'quality_score': row['quality_score'],
                    'issues': [
                        col.replace('_complete', '') 
                        for col in self.df.columns 
                        if col.endswith('_complete') and not row[col] and not col.startswith(('quality_score', 'description_score', 'metadata_score', 'freshness_score', 'keyword_score'))
                    ]
                }
                for _, row in low_quality_courses.iterrows()
            ])
        
        # 2. Content gaps analysis
        if 'category_name' in self.df.columns:
            category_counts = self.df['category_name'].value_counts()
            underrepresented_categories = category_counts[category_counts < category_counts.mean() * 0.5]
            insights['content_gaps'].extend([
                {
                    'category': cat,
                    'current_count': count,
                    'avg_category_count': category_counts.mean(),
                    'gap_percentage': ((category_counts.mean() - count) / category_counts.mean()) * 100
                }
                for cat, count in underrepresented_categories.items()
            ])
        
        # 3. Quality improvement opportunities
        moderate_quality_courses = self.df[
            (self.df['quality_score'] >= 0.4) & 
            (self.df['quality_score'] < 0.7)
        ]
        insights['quality_improvements'].extend([
            {
                'course_title': row['course_title'],
                'current_score': row['quality_score'],
                'improvement_areas': {
                    'description': row.get('description_score', 0),
                    'metadata': row['metadata_score'],
                    'freshness': row.get('freshness_score', 0),
                    'keywords': row.get('keyword_score', 0)
                }
            }
            for _, row in moderate_quality_courses.iterrows()
        ])
        
        # 4. Content redundancies
        insights['content_redundancies'] = [
            {
                'course': info['course'],
                'similar_courses': [
                    {
                        'title': similar['title'],
                        'similarity_score': similar['similarity']
                    }
                    for similar in info['similar_to']
                    if similar['similarity'] > 0.7  # High similarity threshold
                ]
            }
            for info in self.similar_courses.values()
            if any(similar['similarity'] > 0.7 for similar in info['similar_to'])
        ]
        
        # 5. Category distribution analysis
        if 'category_name' in self.df.columns and 'course_available_from' in self.df.columns:
            recent_df = self.df[
                self.df['course_available_from'] > 
                (self.df['course_available_from'].max() - pd.DateOffset(years=2))
            ]
            category_trend = recent_df['category_name'].value_counts()
            category_counts = self.df['category_name'].value_counts()  # Overall counts
            
            insights['category_distribution'].extend([
                {
                    'category': cat,
                    'count': count,
                    'percentage': (count / len(recent_df)) * 100,
                    'trend': 'growing' if count > category_counts.get(cat, 0) * 0.5 else 'declining'
                }
                for cat, count in category_trend.items()
            ])
        
        # 6. Temporal insights
        if 'course_available_from' in self.df.columns:
            recent_period = self.df['course_available_from'].max() - pd.DateOffset(months=6)
            content_velocity = len(self.df[self.df['course_available_from'] >= recent_period])
            
            # Calculate the time span while handling NaT values
            max_date = self.df['course_available_from'].max()
            min_date = self.df['course_available_from'].min()
            if pd.notna(max_date) and pd.notna(min_date):
                age_in_years = (max_date - min_date).days / 365
            else:
                age_in_years = 0
                
            insights['temporal_insights'] = {
                'content_velocity': content_velocity,
                'content_velocity_period': '6 months',
                'average_monthly_new_content': content_velocity / 6,
                'oldest_content_age': age_in_years
            }
        
        return insights

    def generate_enhanced_report(self) -> str:
        """Generate a comprehensive analysis report based on all tables."""
        report = []
        report.append("=== KP Learn Learning Insights Report ===\n")
        
        # Add report description
        report.append("This inventory has a limited focus on recently and currently active learning items")
        report.append("in the content repository and learning catalog. It includes content developed")
        report.append("and/or hosted by a vendor (e.g., SkillSoft), or housed in other learning")
        report.append("infrastructure (e.g., SharePoint) but consumed by learners via KP Learn.")
        report.append("It does not refer to any curricula which bundles and/or sequences courses.\n")
        
        # Table 1: Education History Tracking
        report.append("\n=== Education History Tracking ===")
        history = self.analyze_education_history()
        if history['completion_tracking']:
            report.append(f"High Volume Completions (5k+): {history['completion_tracking']['high_volume_count']}")
            report.append(f"Total Completions: {history['completion_tracking']['total_completions']}")
        if history['client_volume']:
            report.append(f"High Volume Clients (>5%): {history['client_volume']['high_volume_clients']}")
        report.append(f"Total Ed History Clients: {history['ed_history_clients']}")
        
        # Table 2: Training Categories
        report.append("\n=== Training Split by Broad Categories ===")
        categories = self.analyze_training_categories()
        for category, data in categories.items():
            if data['count'] > 0:
                report.append(f"{category}: {data['percentage']:.1f}%")
        
        # Table 3: Training Focus
        report.append("\n=== Training Focus Split by Persona ===")
        focus = self.analyze_training_focus()
        for persona, data in focus['persona_distribution'].items():
            report.append(f"{persona}: {data['percentage']:.1f}%")
        report.append("\nFunction-Specific Training:")
        for func, data in focus['function_specific'].items():
            report.append(f"{func}: {data['percentage']:.1f}%")
        
        # Table 4: Training Breadth
        report.append("\n=== Training Breadth and Equity ===")
        breadth = self.analyze_training_breadth()
        report.append(f"Enterprise-wide: {breadth['enterprise_wide'].get('percentage', 0):.1f}%")
        report.append("\nMarket/Function Distribution:")
        for func, data in breadth['market_function'].items():
            report.append(f"{func}: {data['percentage']:.1f}%")
        
        # Table 5: Training Delivery
        report.append("\n=== Training Delivery Split ===")
        delivery = self.analyze_delivery_methods()
        for method, data in delivery.items():
            if method != 'duration_analysis':
                report.append(f"{method}: {data['percentage']:.1f}%")
        if 'duration_analysis' in delivery:
            report.append(f"\nMicrolearning (<20 mins): {delivery['duration_analysis']['microlearning']}")
            report.append(f"Standard (>20 mins): {delivery['duration_analysis']['standard']}")
        
        # Table 6: Content Usage Length
        report.append("\n=== Content Usage Length ===")
        usage = self.analyze_content_usage()
        for period, data in usage.items():
            report.append(f"{data['description']}: {data['percentage']:.1f}%")
        
        # Table 7: Training Volume
        report.append("\n=== Training Volume by Organization ===")
        volume = self.analyze_training_volume()
        for org, data in volume.items():
            report.append(f"{org}: {data['percentage']:.1f}%")
        
        # Table 8: Content Production
        report.append("\n=== Content Production and Sourcing ===")
        production = self.analyze_content_production()
        for source, data in production.items():
            report.append(f"{source}: {data['percentage']:.1f}%")
        
        # Table 9: Training Assignment
        report.append("\n=== Proactive vs Required Training ===")
        assignment = self.analyze_training_assignment()
        report.append(f"Self-assigned: {assignment['self_assigned']['percentage']:.1f}%")
        report.append(f"Assigned/Registered: {assignment['assigned']['percentage']:.1f}%")
        if assignment['hours_by_group']:
            report.append("\nHours by Group:")
            for group, hours in assignment['hours_by_group'].items():
                report.append(f"{group}: {hours:.1f} hours")
        
        # Table 10: Learner Interests
        report.append("\n=== Top 10 Learner Interests ===")
        interests = self.analyze_learner_interests()
        for interest in interests['top_interests']:
            report.append(f"{interest['area']}: {interest['percentage']:.1f}%")
        
        if interests['trending_topics']:
            report.append("\nTrending Topics:")
            for topic in interests['trending_topics']:
                report.append(f"{topic['area']}: {topic['percentage']:.1f}%")
        
        return "\n".join(report)

    def plot_enhanced_visualizations(self) -> None:
        """Generate enhanced visualizations for different aspects of the data."""
        # Create a directory for plots if it doesn't exist
        plots_dir = Path('analysis_plots')
        plots_dir.mkdir(exist_ok=True)
        
        # Use a copy of the DataFrame for plotting
        plot_df = self.df.copy()
        
        # 1. Time series plot for date columns
        for col in self.date_columns:
            if col not in plot_df.columns:
                continue
                
            try:
                plt.figure(figsize=(15, 6))
                plot_df[col] = pd.to_datetime(plot_df[col], errors='coerce')
                plot_df[col].value_counts().sort_index().plot(kind='line')
                plt.title(f'Time Distribution - {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / f'time_distribution_{col}.png')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create time series plot for '{col}': {str(e)}")
        
        # 2. Category distribution plots
        for col in self.categorical_columns:
            if col not in plot_df.columns:
                continue
                
            plt.figure(figsize=(12, 6))
            plot_df[col].value_counts().nlargest(10).plot(kind='bar')
            plt.title(f'Top 10 Categories - {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / f'category_distribution_{col}.png')
            plt.close()
        
        # 3. Missing data visualization
        plt.figure(figsize=(12, 6))
        missing_data = (plot_df.isnull().sum() / len(plot_df)) * 100
        missing_data.sort_values(ascending=False).plot(kind='bar')
        plt.title('Missing Data by Column')
        plt.ylabel('Percentage Missing')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'missing_data.png')
        plt.close()

    def _infer_role(self, row):
        """Infer role from course metadata"""
        title = row['course_title'].lower() if pd.notna(row.get('course_title')) else ''
        description = row['course_description'].lower() if pd.notna(row.get('course_description')) else ''
        
        text = f"{title} {description}"
        
        if any(word in text for word in ['executive', 'director', 'vp', 'chief']):
            return 'leader/executive'
        elif any(word in text for word in ['manager', 'supervisor', 'lead']):
            return 'manager'
        elif any(word in text for word in ['nurse', 'student', 'intern']):
            return 'student/trainee'
        else:
            return 'employee'

    def get_data_quality_issues(self):
        """Identify data quality issues"""
        issues = {
            'Missing Data': [],
            'Inconsistent Values': [],
            'Invalid Formats': [],
            'Outliers': []
        }
        
        try:
            # Check for significant missing data
            missing_data = self.get_missing_data_summary()
            for col, stats in missing_data.items():
                if stats['percentage'] > 5:
                    issues['Missing Data'].append(
                        f"{col}: {stats['percentage']:.1f}% missing values"
                    )
            
            # Check for inconsistent values
            for col in self.df.columns:
                if col in self.date_columns:
                    invalid_dates = pd.to_datetime(self.df[col], errors='coerce').isna().sum()
                    if invalid_dates > 0:
                        issues['Invalid Formats'].append(
                            f"{col}: {invalid_dates} invalid date values"
                        )
                
                if col == 'course_no':
                    invalid_codes = ~self.df[col].apply(
                        lambda x: bool(re.match(r'^[A-Za-z0-9-]+$', str(x))) if pd.notna(x) else False
                    )
                    invalid_count = invalid_codes.sum()
                    if invalid_count > 0:
                        issues['Invalid Formats'].append(
                            f"course_no: {invalid_count} invalid course numbers"
                        )
            
            # Check for potential duplicates
            if 'course_title' in self.df.columns:
                duplicates = self.df[self.df.duplicated(subset=['course_title'], keep=False)]
                if len(duplicates) > 0:
                    issues['Inconsistent Values'].append(
                        f"Found {len(duplicates)} potential duplicate course titles"
                    )
        except Exception as e:
            print("Warning: Error checking data quality:", str(e))  # Fixed string formatting
        
        return {k: v for k, v in issues.items() if v}  # Only return categories with issues

    def get_data_quality_recommendations(self):
        """Generate recommendations for improving data quality"""
        recommendations = []
        quality_metrics = self.get_data_quality_metrics()
        
        if quality_metrics['completeness'] < 95:
            recommendations.append(
                "Consider implementing required fields to improve data completeness"
            )
        
        if quality_metrics['consistency'] < 90:
            recommendations.append(
                "Standardize data entry processes to improve consistency"
            )
        
        if quality_metrics['validity'] < 90:
            recommendations.append(
                "Implement data validation rules during data entry"
            )
        
        # Check specific patterns
        missing_data = self.get_missing_data_analysis()
        for _, row in missing_data.iterrows():
            if row['missing_percentage'] > 20:
                recommendations.append(
                    f"Priority: Address high missing data in {row['column']} ({row['missing_percentage']:.1f}%)"
                )
        
        return recommendations

    def get_data_overview(self):
        """Get a high-level overview of the dataset"""
        return {
            'total_courses': len(self.df),
            'active_courses': len(self.df[self.df['course_discontinued_from'].isna()]),
            'total_categories': len(self.df['category_name'].unique()),
            'date_range': {
                'earliest': self.df['course_available_from'].min().strftime('%Y-%m-%d') if 'course_available_from' in self.df.columns else None,
                'latest': self.df['course_available_from'].max().strftime('%Y-%m-%d') if 'course_available_from' in self.df.columns else None
            },
            'completeness': f"{(1 - self.df.isnull().mean().mean()) * 100:.1f}%"
        }

    def get_key_findings(self):
        """Generate key findings from the data"""
        findings = []
        
        # Course distribution findings
        category_dist = self.df['category_name'].value_counts()
        findings.append(f"Most common category: {category_dist.index[0]} ({category_dist.iloc[0]} courses)")
        
        # Time-based findings
        if 'course_available_from' in self.df.columns:
            recent_courses = self.df[
                self.df['course_available_from'] >= (pd.Timestamp.now() - pd.DateOffset(months=6))
            ]
            findings.append(f"New courses in last 6 months: {len(recent_courses)}")
        
        # Content type findings
        delivery_methods = self.get_delivery_split()
        if not delivery_methods.empty:
            top_method = delivery_methods.iloc[0]
            findings.append(
                f"Most common delivery method: {top_method['method']} ({top_method['percentage']:.1f}%)"
            )
        
        return findings

    def get_alerts_and_warnings(self):
        """Generate alerts and warnings based on data analysis"""
        alerts = []
        
        # Check for courses without descriptions
        missing_desc = self.df['course_description'].isna().sum()
        if missing_desc > 0:
            alerts.append(f"{missing_desc} courses missing descriptions")
        
        # Check for potential duplicate courses
        duplicates = self.df[self.df.duplicated(subset=['course_title'], keep=False)]
        if len(duplicates) > 0:
            alerts.append(f"Found {len(duplicates)} potential duplicate courses")
        
        # Check for outdated content
        if 'course_available_from' in self.df.columns:
            old_courses = self.df[
                self.df['course_available_from'] < (pd.Timestamp.now() - pd.DateOffset(years=2))
            ]
            if len(old_courses) > 0:
                alerts.append(f"{len(old_courses)} courses are over 2 years old")
        
        return alerts

    def _infer_expected_type(self, column):
        """Infer the expected data type for a column"""
        if column in self.date_columns:
            return 'datetime'
        elif 'course_no' in column:
            return 'alphanumeric'
        elif any(word in column.lower() for word in ['count', 'number', 'amount', 'cost']):
            return 'numeric'
        elif column.endswith('_id'):
            return 'id'
        else:
            return 'text'

    def _validate_column_values(self, column):
        """Validate values in a column based on expected type"""
        if column not in self.df.columns:
            return pd.Series([False] * len(self.df))
            
        expected_type = self._infer_expected_type(column)
        series = self.df[column]
        
        try:
            if expected_type == 'datetime':
                return pd.to_datetime(series, errors='coerce').notna()
            elif expected_type == 'alphanumeric':
                return series.apply(lambda x: bool(re.match(r'^[A-Za-z0-9-]+$', str(x))) if pd.notna(x) else False)
            elif expected_type == 'numeric':
                return pd.to_numeric(series, errors='coerce').notna()
            elif expected_type == 'id':
                return series.apply(lambda x: bool(re.match(r'^\d+$', str(x))) if pd.notna(x) else False)
            else:
                return series.notna()
        except Exception as e:
            print(f"Warning: Validation failed for column {column}: {str(e)}")
            return pd.Series([False] * len(self.df))

    def analyze_learning_effectiveness(self):
        """Analyze learning effectiveness metrics"""
        effectiveness = {
            'available_metrics': {},
            'missing_data': [],
            'recommendations': []
        }
        
        # Check completion metrics
        if 'completion_rate' in self.df.columns:
            completion_trends = self._calculate_completion_trends()
            dropout_analysis = self._analyze_dropout_points()
            if completion_trends or dropout_analysis:
                effectiveness['available_metrics']['completion'] = {
                    'trends': completion_trends,
                    'dropout_analysis': dropout_analysis
                }
        else:
            effectiveness['missing_data'].append({
                'metric': 'completion_rate',
                'impact': 'Cannot analyze course completion patterns and trends',
                'required_fields': ['completion_rate', 'completion_date']
            })
        
        # Check performance metrics
        if 'quiz_scores' in self.df.columns:
            skill_mastery = self._calculate_skill_mastery()
            if skill_mastery:
                effectiveness['available_metrics']['performance'] = {
                    'skill_mastery': skill_mastery
                }
        else:
            effectiveness['missing_data'].append({
                'metric': 'quiz_scores',
                'impact': 'Cannot assess learning performance and skill mastery',
                'required_fields': ['quiz_scores', 'assessment_results']
            })
        
        # Check skill progression
        if all(col in self.df.columns for col in ['pre_assessment', 'post_assessment']):
            skill_gaps = self._analyze_skill_gaps_closed()
            retention = self._analyze_retention_rates()
            if skill_gaps or retention:
                effectiveness['available_metrics']['skill_progression'] = {
                    'skill_gaps_closed': skill_gaps,
                    'retention_rates': retention
                }
        else:
            effectiveness['missing_data'].append({
                'metric': 'skill_progression',
                'impact': 'Cannot measure skill improvement and knowledge retention',
                'required_fields': ['pre_assessment', 'post_assessment', 'retention_score']
            })
        
        # Generate recommendations for data collection
        if effectiveness['missing_data']:
            effectiveness['recommendations'].extend([
                'Implement systematic tracking of completion rates and dates',
                'Add pre and post assessments to measure skill progression',
                'Track quiz scores and assessment results',
                'Monitor knowledge retention through follow-up assessments'
            ])
        
        return effectiveness

    def analyze_engagement_patterns(self):
        """Analyze learner engagement patterns"""
        patterns = {
            'available_metrics': {},
            'missing_data': [],
            'recommendations': []
        }
        
        # Check engagement metrics
        if 'session_duration' in self.df.columns:
            engagement_dist = self._calculate_engagement_distribution()
            if engagement_dist:
                patterns['available_metrics']['engagement_levels'] = engagement_dist
        else:
            patterns['missing_data'].append({
                'metric': 'session_duration',
                'impact': 'Cannot analyze time spent on learning activities',
                'required_fields': ['session_duration']
            })
        
        # Check interaction data
        if 'interaction_type' in self.df.columns:
            interactions = self._analyze_interaction_types()
            if interactions:
                patterns['available_metrics']['interaction_analysis'] = interactions
        else:
            patterns['missing_data'].append({
                'metric': 'interaction_types',
                'impact': 'Cannot analyze how learners interact with content',
                'required_fields': ['interaction_type', 'interaction_count']
            })
        
        # Check engagement quality
        required_fields = ['interaction_count', 'session_duration']
        if all(col in self.df.columns for col in required_fields):
            quality = self._calculate_engagement_quality()
            if quality:
                patterns['available_metrics']['engagement_quality'] = quality
        else:
            patterns['missing_data'].append({
                'metric': 'engagement_quality',
                'impact': 'Cannot assess the quality and depth of engagement',
                'required_fields': required_fields
            })
        
        # Generate recommendations for data collection
        if patterns['missing_data']:
            patterns['recommendations'].extend([
                'Track session duration for all learning activities',
                'Implement interaction tracking to understand engagement patterns',
                'Monitor both quantity and quality of learner interactions',
                'Consider adding engagement metrics to your LMS'
            ])
        
        return patterns

    def generate_personalized_recommendations(self, user_data=None):
        """Generate personalized learning recommendations"""
        recommendations = {
            'based_on_history': [],
            'based_on_skills': [],
            'based_on_role': [],
            'next_steps': []
        }
        
        if user_data is None:
            # Generate generic recommendations based on overall patterns
            popular_courses = (
                self.df.sort_values('completion_rate', ascending=False)
                ['course_title']
                .head(5)
                .tolist()
            )
            recommendations['based_on_history'] = popular_courses
            
            if 'category_name' in self.df.columns:
                trending_categories = (
                    self.df.groupby('category_name')
                    .size()
                    .sort_values(ascending=False)
                    .head(3)
                    .index
                    .tolist()
                )
                recommendations['based_on_role'] = trending_categories
        else:
            # Generate personalized recommendations based on user data
            if 'completed_courses' in user_data:
                # Find similar courses based on completed ones
                similar_courses = []
                for course in user_data['completed_courses']:
                    if course in self.df['course_title'].values:
                        course_idx = self.df[self.df['course_title'] == course].index[0]
                        similar_indices = (
                            self.df[self.df['category_name'] == self.df.loc[course_idx, 'category_name']]
                            .index
                            .tolist()
                        )
                        similar_courses.extend(
                            self.df.loc[similar_indices, 'course_title']
                            .tolist()
                        )
                recommendations['based_on_history'] = list(set(similar_courses))[:5]
            
            if 'skills' in user_data:
                # Recommend courses based on user's skills
                skill_based = []
                for skill in user_data['skills']:
                    relevant_courses = (
                        self.df[self.df['skills_covered'].str.contains(skill, na=False)]
                        ['course_title']
                        .tolist()
                    )
                    skill_based.extend(relevant_courses)
                recommendations['based_on_skills'] = list(set(skill_based))[:5]
            
            if 'role' in user_data:
                # Recommend courses based on user's role
                role_based = (
                    self.df[self.df['target_role'].str.contains(user_data['role'], na=False)]
                    ['course_title']
                    .tolist()
                )
                recommendations['based_on_role'] = role_based[:5]
        
        # Generate next steps
        recommendations['next_steps'] = [
            'Complete recommended courses',
            'Update skills profile',
            'Join relevant learning paths',
            'Track progress regularly'
        ]
        
        return recommendations

    def analyze_business_impact(self):
        """Analyze business impact of learning programs"""
        impact = {
            'available_metrics': {},
            'missing_data': [],
            'recommendations': []
        }
        
        # Check ROI metrics
        required_fields = ['course_cost', 'completion_rate']
        if all(col in self.df.columns for col in required_fields):
            total_cost = self.df['course_cost'].sum()
            completed_courses = self.df[self.df['completion_rate'] >= 0.8]
            
            impact['available_metrics']['roi'] = {
                'total_investment': total_cost,
                'cost_per_completion': total_cost / len(completed_courses) if len(completed_courses) > 0 else 0,
                'completion_rate': len(completed_courses) / len(self.df) * 100
            }
        else:
            impact['missing_data'].append({
                'metric': 'roi',
                'impact': 'Cannot calculate return on investment metrics',
                'required_fields': required_fields
            })
        
        # Check skill development impact
        required_fields = ['pre_assessment', 'post_assessment']
        if all(col in self.df.columns for col in required_fields):
            skill_improvement = self.df['post_assessment'] - self.df['pre_assessment']
            
            impact['available_metrics']['skill_development'] = {
                'average_improvement': skill_improvement.mean(),
                'significant_improvements': len(skill_improvement[skill_improvement > 20]) / len(self.df) * 100,
                'areas_of_impact': self.df.groupby('category_name')['post_assessment'].mean().to_dict()
            }
        else:
            impact['missing_data'].append({
                'metric': 'skill_development',
                'impact': 'Cannot measure skill improvement and development impact',
                'required_fields': required_fields
            })
        
        # Check productivity impact
        if 'productivity_score' in self.df.columns:
            impact['available_metrics']['productivity'] = {
                'average_improvement': self.df['productivity_score'].mean(),
                'high_impact_courses': (
                    self.df[self.df['productivity_score'] > self.df['productivity_score'].mean()]
                    ['course_title']
                    .tolist()
                )
            }
        else:
            impact['missing_data'].append({
                'metric': 'productivity',
                'impact': 'Cannot assess impact on workplace productivity',
                'required_fields': ['productivity_score', 'performance_metrics']
            })
        
        # Generate recommendations for data collection
        if impact['missing_data']:
            impact['recommendations'].extend([
                'Track course costs and completion rates',
                'Implement pre and post skill assessments',
                'Measure workplace productivity impact',
                'Monitor business performance indicators'
            ])
        
        return impact

    def _calculate_completion_trends(self):
        """Calculate completion trends over time"""
        if 'completion_date' not in self.df.columns:
            return None
            
        trends = {}
        # Group by month and calculate completion rates
        monthly_completion = (
            self.df.set_index('completion_date')
            .resample('M')['completion_rate']
            .mean()
        )
        
        # Calculate trend
        trend_coefficient = np.polyfit(
            range(len(monthly_completion)),
            monthly_completion.values,
            1
        )[0]
        
        trends['direction'] = 'increasing' if trend_coefficient > 0 else 'decreasing'
        trends['monthly_rates'] = monthly_completion.to_dict()
        trends['trend_strength'] = abs(trend_coefficient)
        
        return trends

    def _analyze_dropout_points(self):
        """Analyze where and why learners drop out"""
        if 'last_completed_module' not in self.df.columns:
            return None
            
        dropouts = {}
        # Find common dropout points
        dropout_points = (
            self.df[self.df['completion_rate'] < 1.0]
            ['last_completed_module']
            .value_counts()
            .head(5)
            .to_dict()
        )
        
        dropouts['common_points'] = dropout_points
        
        # Analyze patterns if reason data is available
        if 'dropout_reason' in self.df.columns:
            dropouts['reasons'] = (
                self.df['dropout_reason']
                .value_counts()
                .head(5)
                .to_dict()
            )
        
        return dropouts

    def _calculate_skill_mastery(self):
        """Calculate skill mastery levels"""
        if 'quiz_scores' not in self.df.columns:
            return None
            
        mastery = {}
        # Define mastery thresholds
        mastery_levels = {
            'Expert': 90,
            'Proficient': 80,
            'Competent': 70,
            'Developing': 60
        }
        
        # Calculate distribution across mastery levels
        for level, threshold in mastery_levels.items():
            mastery[level] = (
                len(self.df[self.df['quiz_scores'] >= threshold]) / 
                len(self.df) * 100
            )
        
        return mastery

    def _analyze_skill_gaps_closed(self):
        """Analyze the effectiveness of skill gap closure"""
        if 'pre_assessment' not in self.df.columns or 'post_assessment' not in self.df.columns:
            return None
            
        gaps = {}
        improvement = self.df['post_assessment'] - self.df['pre_assessment']
        
        gaps['fully_closed'] = len(improvement[improvement >= 20]) / len(self.df) * 100
        gaps['partially_closed'] = len(improvement[(improvement >= 10) & (improvement < 20)]) / len(self.df) * 100
        gaps['minimal_improvement'] = len(improvement[improvement < 10]) / len(self.df) * 100
        
        return gaps

    def _analyze_retention_rates(self):
        """Analyze knowledge retention over time"""
        if 'retention_score' not in self.df.columns or 'days_since_completion' not in self.df.columns:
            return None
            
        retention = {}
        # Calculate retention rates at different intervals
        intervals = [30, 60, 90, 180]
        
        for interval in intervals:
            retention[f'{interval}_days'] = (
                self.df[self.df['days_since_completion'] <= interval]
                ['retention_score']
                .mean()
            )
        
        return retention

    def _calculate_engagement_distribution(self):
        """Calculate the distribution of engagement levels"""
        if 'session_duration' not in self.df.columns:
            return None
            
        engagement = {}
        # Define engagement levels
        duration_bins = [0, 15, 30, 60, float('inf')]
        labels = ['Low', 'Medium', 'High', 'Very High']
        
        self.df['engagement_level'] = pd.cut(
            self.df['session_duration'],
            bins=duration_bins,
            labels=labels
        )
        
        engagement['distribution'] = (
            self.df['engagement_level']
            .value_counts(normalize=True)
            .to_dict()
        )
        
        return engagement

    def _analyze_interaction_types(self):
        """Analyze types of learner interactions"""
        if 'interaction_type' not in self.df.columns:
            return None
            
        interactions = {}
        # Calculate frequency of different interaction types
        interactions['frequency'] = (
            self.df['interaction_type']
            .value_counts(normalize=True)
            .to_dict()
        )
        
        # Calculate effectiveness if we have performance data
        if 'quiz_scores' in self.df.columns:
            interactions['effectiveness'] = (
                self.df.groupby('interaction_type')['quiz_scores']
                .mean()
                .to_dict()
            )
        
        return interactions

    def _calculate_engagement_quality(self):
        """Calculate the quality of learner engagement"""
        if 'interaction_count' not in self.df.columns or 'session_duration' not in self.df.columns:
            return None
            
        quality = {}
        # Calculate engagement density (interactions per minute)
        self.df['engagement_density'] = (
            self.df['interaction_count'] / 
            (self.df['session_duration'] / 60)
        )
        
        quality['avg_engagement_density'] = self.df['engagement_density'].mean()
        quality['engagement_quality_dist'] = (
            pd.qcut(self.df['engagement_density'], q=4)
            .value_counts(normalize=True)
            .to_dict()
        )
        
        return quality

    def _analyze_sequence_effectiveness(self):
        """Analyze the effectiveness of learning sequences"""
        effectiveness = {
            'completion_rates': {},
            'performance_metrics': {},
            'progression_speed': {},
            'recommendations': []
        }
        
        if 'course_sequence' not in self.df.columns:
            return effectiveness
            
        # Analyze completion rates for different sequences
        sequence_completion = (
            self.df.groupby('course_sequence')['completion_rate']
            .agg(['mean', 'count'])
            .sort_values('mean', ascending=False)
        )
        
        effectiveness['completion_rates'] = {
            'best_sequences': sequence_completion.head(3).to_dict('index'),
            'worst_sequences': sequence_completion.tail(3).to_dict('index')
        }
        
        # Analyze performance if quiz data is available
        if 'quiz_scores' in self.df.columns:
            sequence_performance = (
                self.df.groupby('course_sequence')['quiz_scores']
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .to_dict()
            )
            effectiveness['performance_metrics'] = sequence_performance
        
        # Analyze progression speed if duration data is available
        if 'completion_duration' in self.df.columns:
            speed_by_sequence = (
                self.df.groupby('course_sequence')['completion_duration']
                .agg(['mean', 'median', 'std'])
                .sort_values('mean')
                .head(5)
                .to_dict('index')
            )
            effectiveness['progression_speed'] = speed_by_sequence
        
        # Generate recommendations based on analysis
        effectiveness['recommendations'] = [
            'Follow high-performing sequences',
            'Consider prerequisites carefully',
            'Allow flexible pacing',
            'Monitor sequence effectiveness'
        ]
        
        return effectiveness

    def analyze_learning_paths(self):
        """Analyze and recommend learning paths"""
        paths = {
            'available_metrics': {},
            'missing_data': [],
            'recommendations': []
        }
        
        # Check sequence data
        if 'course_sequence' in self.df.columns:
            sequence_patterns = (
                self.df.groupby('course_sequence')
                .size()
                .sort_values(ascending=False)
                .head(5)
                .to_dict()
            )
            paths['available_metrics']['learning_sequences'] = {
                'common_sequences': sequence_patterns,
                'effectiveness': self._analyze_sequence_effectiveness()
            }
        else:
            paths['missing_data'].append({
                'metric': 'learning_sequences',
                'impact': 'Cannot analyze common learning paths and their effectiveness',
                'required_fields': ['course_sequence', 'completion_rate']
            })
        
        # Check content relationships
        if 'course_description' in self.df.columns:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(self.df['course_description'].fillna(''))
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            content_relationships = {}
            for idx, row in self.df.iterrows():
                similar_indices = similarity_matrix[idx].argsort()[-4:-1][::-1]
                content_relationships[row['course_title']] = {
                    'related_courses': self.df.iloc[similar_indices]['course_title'].tolist(),
                    'similarity_scores': similarity_matrix[idx][similar_indices].tolist()
                }
            paths['available_metrics']['content_relationships'] = content_relationships
        else:
            paths['missing_data'].append({
                'metric': 'content_relationships',
                'impact': 'Cannot identify related courses and content connections',
                'required_fields': ['course_description', 'course_keywords']
            })
        
        # Check skill coverage
        if 'skills_covered' in self.df.columns:
            skill_coverage = {}
            for skill in self.df['skills_covered'].unique():
                relevant_courses = (
                    self.df[self.df['skills_covered'].str.contains(skill, na=False)]
                    .sort_values('completion_rate', ascending=False)
                    ['course_title']
                    .head(3)
                    .tolist()
                )
                if relevant_courses:
                    skill_coverage[skill] = relevant_courses
            paths['available_metrics']['skill_coverage'] = skill_coverage
        else:
            paths['missing_data'].append({
                'metric': 'skill_coverage',
                'impact': 'Cannot map skills to courses and suggest skill-based paths',
                'required_fields': ['skills_covered', 'skill_level']
            })
        
        # Generate recommendations for data collection
        if paths['missing_data']:
            paths['recommendations'].extend([
                'Track course sequences and completion order',
                'Add detailed course descriptions and keywords',
                'Map skills to courses explicitly',
                'Track skill levels and prerequisites'
            ])
        
        return paths

    def analyze_education_history(self):
        """
        Analyze education history tracking (Table 1)
        Track completions without enrollments and find heavy volume clients
        """
        history = {
            'completion_tracking': {},
            'client_volume': {},
            'ed_history_clients': 0
        }
        
        if 'completion_count' in self.df.columns:
            # Get 5k+ completions
            high_volume = self.df[self.df['completion_count'] >= 5000]
            history['completion_tracking'] = {
                'high_volume_count': len(high_volume),
                'total_completions': self.df['completion_count'].sum()
            }
            
            # Analyze client volume
            if 'client_percentage' in self.df.columns:
                history['client_volume'] = {
                    'high_volume_clients': len(self.df[self.df['client_percentage'] > 5]),
                    'volume_distribution': self.df['client_percentage'].describe().to_dict()
                }
            
            # Count education history clients
            if 'number_of_clients' in self.df.columns:
                history['ed_history_clients'] = self.df['number_of_clients'].sum()
        
        return history

    def analyze_training_categories(self):
        """
        Analyze training split by broad categories (Table 2)
        """
        categories = {cat: {'count': 0, 'percentage': 0} for cat in TRAINING_CATEGORIES.keys()}
        
        if 'training_category' in self.df.columns:
            # Count courses in each category
            category_counts = self.df['training_category'].value_counts()
            total_courses = len(self.df)
            
            for cat in TRAINING_CATEGORIES.keys():
                count = category_counts.get(cat, 0)
                categories[cat] = {
                    'count': count,
                    'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                }
        
        return categories

    def analyze_training_focus(self):
        """
        Analyze training focus split by persona (Table 3)
        """
        focus = {
            'persona_distribution': {},
            'function_specific': {}
        }
        
        if 'persona_type' in self.df.columns:
            # Calculate percentage for each persona
            persona_counts = self.df['persona_type'].value_counts()
            total_courses = len(self.df)
            
            for persona, description in TRAINING_FOCUS_PERSONAS.items():
                count = persona_counts.get(persona, 0)
                focus['persona_distribution'][persona] = {
                    'description': description,
                    'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                }
            
            # Analyze function-specific training
            if 'function_specific' in self.df.columns:
                function_counts = self.df['function_specific'].value_counts()
                for func, count in function_counts.items():
                    focus['function_specific'][func] = {
                        'count': count,
                        'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                    }
        
        return focus

    def analyze_training_breadth(self):
        """
        Analyze training breadth and equity (Table 4)
        """
        breadth = {
            'enterprise_wide': {},
            'market_function': {},
            'specific_setting': {}
        }
        
        if 'region_entity' in self.df.columns:
            total_courses = len(self.df)
            
            # Analyze enterprise-wide availability
            enterprise_courses = self.df[
                self.df['training_category'].isin(['Compliance', 'Safety']) |
                self.df['content_source'].str.contains('SkillSoft', na=False)
            ]
            breadth['enterprise_wide'] = {
                'count': len(enterprise_courses),
                'percentage': (len(enterprise_courses) / total_courses * 100) if total_courses > 0 else 0
            }
            
            # Analyze market/function distribution
            if 'market_function' in self.df.columns:
                market_counts = self.df['market_function'].value_counts()
                breadth['market_function'] = {
                    func: {
                        'count': count,
                        'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                    }
                    for func, count in market_counts.items()
                }
            
            # Analyze specific settings
            if 'specific_setting' in self.df.columns:
                setting_counts = self.df['specific_setting'].value_counts()
                breadth['specific_setting'] = {
                    setting: {
                        'count': count,
                        'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                    }
                    for setting, count in setting_counts.items()
                }
        
        return breadth

    def analyze_delivery_methods(self):
        """
        Analyze training delivery split (Table 5)
        """
        delivery = {method: {'count': 0, 'percentage': 0} for method in DELIVERY_METHODS.keys()}
        
        if 'delivery_method' in self.df.columns:
            # Calculate distribution of delivery methods
            delivery_counts = self.df['delivery_method'].value_counts()
            total_courses = len(self.df)
            
            for method, description in DELIVERY_METHODS.items():
                count = delivery_counts.get(method, 0)
                delivery[method] = {
                    'description': description,
                    'count': count,
                    'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                }
            
            # Analyze duration distribution
            if 'duration_minutes' in self.df.columns:
                delivery['duration_analysis'] = {
                    'microlearning': len(self.df[self.df['duration_minutes'] < 20]),
                    'standard': len(self.df[self.df['duration_minutes'] >= 20])
                }
        
        return delivery

    def analyze_content_usage(self):
        """
        Analyze content usage length (Table 6)
        """
        usage = {period: {'count': 0, 'percentage': 0} for period in CONTENT_USAGE_LENGTH.keys()}
        
        if 'available_from' in self.df.columns:
            current_date = pd.Timestamp.now()
            
            # Calculate age of each course
            self.df['content_age'] = (current_date - pd.to_datetime(self.df['available_from'])).dt.days / 365
            
            # Categorize content by age
            year_counts = {
                '1_year': len(self.df[self.df['content_age'] <= 1]),
                '2_3_years': len(self.df[(self.df['content_age'] > 1) & (self.df['content_age'] <= 3)]),
                'longer': len(self.df[self.df['content_age'] > 3])
            }
            
            total_courses = len(self.df)
            for period, count in year_counts.items():
                usage[period] = {
                    'description': CONTENT_USAGE_LENGTH[period],
                    'count': count,
                    'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                }
        
        return usage

    def analyze_training_volume(self):
        """
        Analyze training volume by organization (Table 7)
        """
        volume = {org: {'count': 0, 'percentage': 0} for org in TRAINING_ORGANIZATIONS.keys()}
        
        if 'training_organization' in self.df.columns:
            # Calculate distribution across organizations
            org_counts = self.df['training_organization'].value_counts()
            total_courses = len(self.df)
            
            for org, description in TRAINING_ORGANIZATIONS.items():
                count = org_counts.get(org, 0)
                volume[org] = {
                    'description': description,
                    'count': count,
                    'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                }
        
        return volume

    def analyze_content_production(self):
        """
        Analyze content production and sourcing (Table 8)
        """
        production = {source: {'count': 0, 'percentage': 0} for source in CONTENT_SOURCES.keys()}
        
        if 'content_source' in self.df.columns:
            # Calculate distribution of content sources
            source_counts = self.df['content_source'].value_counts()
            total_courses = len(self.df)
            
            for source, description in CONTENT_SOURCES.items():
                count = source_counts.get(source, 0)
                production[source] = {
                    'description': description,
                    'count': count,
                    'percentage': (count / total_courses * 100) if total_courses > 0 else 0
                }
        
        return production

    def analyze_training_assignment(self):
        """
        Analyze proactive vs required training (Table 9)
        """
        assignment = {
            'self_assigned': {'count': 0, 'percentage': 0},
            'assigned': {'count': 0, 'percentage': 0},
            'hours_by_group': {}
        }
        
        if 'assignment_type' in self.df.columns:
            # Calculate assignment type distribution
            type_counts = self.df['assignment_type'].value_counts()
            total_courses = len(self.df)
            
            assignment['self_assigned'] = {
                'count': type_counts.get('self_assigned', 0),
                'percentage': (type_counts.get('self_assigned', 0) / total_courses * 100) if total_courses > 0 else 0
            }
            
            assignment['assigned'] = {
                'count': type_counts.get('assigned', 0),
                'percentage': (type_counts.get('assigned', 0) / total_courses * 100) if total_courses > 0 else 0
            }
            
            # Calculate hours by group if available
            if 'training_hours' in self.df.columns and 'persona_type' in self.df.columns:
                hours_by_group = self.df.groupby('persona_type')['training_hours'].sum()
                assignment['hours_by_group'] = hours_by_group.to_dict()
        
        return assignment

    def analyze_learner_interests(self):
        """
        Analyze top 10 learner interests (Table 10)
        """
        interests = {
            'top_interests': [],
            'interest_distribution': {},
            'trending_topics': []
        }
        
        if 'interest_area' in self.df.columns and 'interest_percentage' in self.df.columns:
            # Get top 10 interests by percentage
            top_interests = (
                self.df.groupby('interest_area')['interest_percentage']
                .mean()
                .sort_values(ascending=False)
                .head(10)
            )
            
            interests['top_interests'] = [
                {
                    'area': area,
                    'percentage': percentage
                }
                for area, percentage in top_interests.items()
            ]
            
            # Calculate overall interest distribution
            interests['interest_distribution'] = (
                self.df.groupby('interest_area')['interest_percentage']
                .mean()
                .to_dict()
            )
            
            # Identify trending topics (if timestamp available)
            if 'available_from' in self.df.columns:
                recent_interests = (
                    self.df[self.df['available_from'] > (pd.Timestamp.now() - pd.DateOffset(months=6))]
                    .groupby('interest_area')['interest_percentage']
                    .mean()
                    .sort_values(ascending=False)
                    .head(5)
                )
                
                interests['trending_topics'] = [
                    {
                        'area': area,
                        'percentage': percentage
                    }
                    for area, percentage in recent_interests.items()
                ]
        
        return interests

    def analyze_financial_metrics(self):
        """Analyze financial aspects of training content and delivery."""
        # Validate financial data first
        validation = self._validate_financial_data()
        
        if validation['missing_columns'] or validation['invalid_data']:
            return {
                'validation_issues': validation,
                'message': 'Cannot perform complete financial analysis due to missing or invalid data'
            }
        
        if self.df is None:
            return {}
        
        # Initialize financial metrics
        financials = {
            'training_spend': {},
            'learner_costs': {},
            'tuition_reimbursement': {
                'total': 0,
                'by_program': {}
            }
        }
        
        # Calculate training spend by vendor/system
        if 'vendor' in self.df.columns:
            spend_by_vendor = self.df.groupby('vendor')['cost'].sum()
            financials['training_spend'] = spend_by_vendor.to_dict()
        
        # Calculate average learner costs
        if 'learner_type' in self.df.columns and 'cost_per_learner' in self.df.columns:
            avg_costs = self.df.groupby('learner_type')['cost_per_learner'].mean()
            financials['learner_costs'] = avg_costs.to_dict()
        
        # Analyze tuition reimbursement data
        if 'reimbursement_amount' in self.df.columns:
            total_reimbursement = self.df['reimbursement_amount'].sum()
            financials['tuition_reimbursement']['total'] = total_reimbursement
            
            if 'program_type' in self.df.columns:
                program_spend = self.df.groupby('program_type')['reimbursement_amount'].sum()
                financials['tuition_reimbursement']['by_program'] = program_spend.to_dict()
        
        return financials

    def _validate_financial_data(self):
        """Validate financial data columns and provide feedback."""
        validation_results = {
            'missing_columns': [],
            'invalid_data': [],
            'recommendations': []
        }
        
        # Check for required columns
        for col, description in REQUIRED_FINANCIAL_COLUMNS.items():
            if col not in self.df.columns:
                validation_results['missing_columns'].append({
                    'column': col,
                    'description': description
                })
        
        if self.df is not None:
            # Validate numeric columns
            numeric_columns = ['cost', 'cost_per_learner', 'reimbursement_amount']
            for col in numeric_columns:
                if col in self.df.columns:
                    invalid_rows = self.df[~self.df[col].apply(lambda x: isinstance(x, (int, float)) and x >= 0)].index
                    if len(invalid_rows) > 0:
                        validation_results['invalid_data'].append({
                            'column': col,
                            'invalid_rows': len(invalid_rows),
                            'message': f'Contains non-numeric or negative values'
                        })
        
        # Generate recommendations
        if validation_results['missing_columns']:
            validation_results['recommendations'].append(
                'Add missing financial columns to enable cost analysis and ROI calculations'
            )
        if validation_results['invalid_data']:
            validation_results['recommendations'].append(
                'Clean up invalid numeric data in financial columns'
            )
        
        return validation_results

    def analyze_admin_metrics(self):
        """Analyze administrative metrics and efficiency."""
        # Validate admin data first
        validation = self._validate_admin_data()
        
        if validation['missing_columns'] or validation['invalid_data']:
            return {
                'validation_issues': validation,
                'message': 'Cannot perform complete administrative analysis due to missing or invalid data'
            }
        
        if self.df is None:
            return {}
        
        admin_metrics = {
            'market_split': {},
            'efficiency_ratios': {},
            'support_volume': {}
        }
        
        # Calculate admin distribution by market
        if 'market' in self.df.columns and 'admin_count' in self.df.columns:
            market_distribution = self.df.groupby('market')['admin_count'].sum()
            admin_metrics['market_split'] = market_distribution.to_dict()
        
        # Calculate efficiency ratios
        total_admins = self.df['admin_count'].sum() if 'admin_count' in self.df.columns else 0
        if total_admins > 0:
            total_courses = len(self.df['course_id'].unique()) if 'course_id' in self.df.columns else 0
            total_classes = len(self.df['class_id'].unique()) if 'class_id' in self.df.columns else 0
            
            admin_metrics['efficiency_ratios'] = {
                'courses_per_admin': total_courses / total_admins if total_admins > 0 else 0,
                'classes_per_admin': total_classes / total_admins if total_admins > 0 else 0
            }
        
        # Analyze support volume
        if 'support_tickets' in self.df.columns:
            support_metrics = {
                'total_tickets': self.df['support_tickets'].sum(),
                'avg_resolution_time': self.df['resolution_time'].mean() if 'resolution_time' in self.df.columns else None,
                'satisfaction_score': self.df['satisfaction'].mean() if 'satisfaction' in self.df.columns else None
            }
            admin_metrics['support_volume'] = support_metrics
        
        return admin_metrics

    def _validate_admin_data(self):
        """Validate administrative data columns and provide feedback."""
        validation_results = {
            'missing_columns': [],
            'invalid_data': [],
            'recommendations': []
        }
        
        # Check for required columns
        for col, description in REQUIRED_ADMIN_COLUMNS.items():
            if col not in self.df.columns:
                validation_results['missing_columns'].append({
                    'column': col,
                    'description': description
                })
        
        if self.df is not None:
            # Validate numeric columns
            numeric_columns = ['admin_count', 'support_tickets', 'resolution_time', 'satisfaction']
            for col in numeric_columns:
                if col in self.df.columns:
                    invalid_rows = self.df[~self.df[col].apply(lambda x: isinstance(x, (int, float)) and x >= 0)].index
                    if len(invalid_rows) > 0:
                        validation_results['invalid_data'].append({
                            'column': col,
                            'invalid_rows': len(invalid_rows),
                            'message': f'Contains non-numeric or negative values'
                        })
        
        # Generate recommendations
        if validation_results['missing_columns']:
            validation_results['recommendations'].append(
                'Add missing administrative columns to enable efficiency and support analysis'
            )
        if validation_results['invalid_data']:
            validation_results['recommendations'].append(
                'Clean up invalid numeric data in administrative columns'
            )
        
        return validation_results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
    else:
        print("Please provide the path to your Excel file.")
        print("Usage: python3 lms_content_analyzer.py <path_to_excel_file>")
        print("Example: python3 lms_content_analyzer.py my_lms_data.xlsx")
        sys.exit(1)
        
    analyzer = LMSContentAnalyzer(excel_file)
    analyzer.generate_enhanced_report()
    analyzer.plot_enhanced_visualizations() 