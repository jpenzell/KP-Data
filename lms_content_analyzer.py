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
            self._clean_column_names()
            self._classify_columns()
            self._convert_date_columns()
            print(f"\nSuccessfully loaded {len(self.df)} rows of data")
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _clean_column_names(self):
        # Print original column names
        print("\nOriginal column names:")
        for col in self.df.columns:
            print(f"- {col}")

        # Create mapping
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
        for col in self.date_columns:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except Exception as e:
                    print(f"Warning: Could not convert {col} to datetime: {str(e)}")

    def get_data_quality_metrics(self):
        """Calculate overall data quality metrics"""
        metrics = {
            'completeness': {},
            'consistency': {},
            'validity': {}
        }
        
        # Calculate completeness for each column
        for col in self.df.columns:
            metrics['completeness'][col] = (1 - self.df[col].isna().mean()) * 100

        # Calculate consistency for each column
        for col in self.df.columns:
            try:
                if col in self.date_columns:
                    # Check if dates are in a consistent format
                    dates = pd.to_datetime(self.df[col], errors='coerce')
                    metrics['consistency'][col] = (dates.notna().mean()) * 100
                elif col in self.text_columns:
                    # Check if text values follow expected patterns
                    non_empty = self.df[col].fillna('').astype(str).str.strip().str.len() > 0
                    metrics['consistency'][col] = (non_empty.mean()) * 100
                else:
                    # For other columns, check for non-null values
                    metrics['consistency'][col] = (self.df[col].notna().mean()) * 100
            except Exception as e:
                print(f"Warning: Could not check consistency for {col}: {str(e)}")
                metrics['consistency'][col] = 0

        # Calculate validity for each column
        for col in self.df.columns:
            try:
                if col in self.date_columns:
                    # Check if dates are valid and not in the future
                    dates = pd.to_datetime(self.df[col], errors='coerce')
                    valid_dates = dates.notna() & (dates <= pd.Timestamp.now())
                    metrics['validity'][col] = (valid_dates.mean()) * 100
                elif col == 'course_no':
                    # Check if course numbers follow the expected format
                    valid_format = self.df[col].apply(
                        lambda x: bool(re.match(r'^[A-Za-z0-9-]+$', str(x))) if pd.notna(x) else False
                    )
                    metrics['validity'][col] = (valid_format.mean()) * 100
                elif col in self.text_columns:
                    # Check if text values are within reasonable length
                    valid_length = self.df[col].fillna('').astype(str).str.len().between(1, 5000)
                    metrics['validity'][col] = (valid_length.mean()) * 100
                else:
                    # For other columns, check for non-null and non-empty values
                    valid_values = self.df[col].notna() & (self.df[col].astype(str).str.strip() != '')
                    metrics['validity'][col] = (valid_values.mean()) * 100
            except Exception as e:
                print(f"Warning: Could not check validity for {col}: {str(e)}")
                metrics['validity'][col] = 0

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
            # Create mock category distribution data
            return pd.DataFrame({
                'category_type': ['Leadership Development', 'Leadership Development', 'Managerial and Supervisory', 
                                'Mandatory and Compliance', 'Technical Skills'] * 2,
                'subcategory': ['Executive Leadership', 'Team Leadership', 'New Manager Training',
                               'Annual Compliance', 'Technical Skills', 'Professional Skills',
                               'Leadership Skills', 'Management Skills', 'Supervisory Skills',
                               'Safety Training'],
                'count': [100, 80, 120, 200, 150, 90, 70, 110, 95, 180]
            })
            
        # Create a copy to avoid SettingWithCopyWarning
        df_copy = self.df.copy()
        df_copy['category_type'] = df_copy['category_name'].apply(self._categorize_course)
        
        category_counts = df_copy.groupby(['category_type', 'category_name']).size().reset_index()
        category_counts.columns = ['category_type', 'subcategory', 'count']
        return category_counts

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
        """Generate a comprehensive analysis report with actionable insights."""
        insights = self.get_actionable_insights()
        
        report = []
        report.append("=== LMS Content Library Analysis Report ===\n")
        
        # 1. Executive Summary
        report.append("\n=== Executive Summary ===")
        report.append(f"Total Courses: {len(self.df)}")
        report.append(f"Average Quality Score: {self.df['quality_score'].mean():.2f}")
        report.append(f"Courses Needing Attention: {len(self.df[self.df['needs_attention']])}")
        
        # 2. Urgent Attention Required
        report.append("\n=== Urgent Attention Required ===")
        for course in insights['urgent_attention_needed']:
            report.append(f"\nCourse: {course['course_title']}")
            report.append(f"Quality Score: {course['quality_score']:.2f}")
            report.append("Issues:")
            for issue in course['issues']:
                report.append(f"- Missing {issue.replace('_complete', '')}")
        
        # 3. Content Gaps
        report.append("\n=== Content Gaps ===")
        for gap in insights['content_gaps']:
            report.append(f"\nCategory: {gap['category']}")
            report.append(f"Current Count: {gap['current_count']}")
            report.append(f"Gap Percentage: {gap['gap_percentage']:.1f}%")
        
        # 4. Quality Improvement Opportunities
        report.append("\n=== Quality Improvement Opportunities ===")
        for course in insights['quality_improvements']:
            report.append(f"\nCourse: {course['course_title']}")
            report.append("Improvement Areas:")
            for area, score in course['improvement_areas'].items():
                if score < 0.6:
                    report.append(f"- {area.title()}: {score:.2f}")
        
        # 5. Content Redundancies
        report.append("\n=== Potential Content Redundancies ===")
        for redundancy in insights['content_redundancies']:
            report.append(f"\nCourse: {redundancy['course']}")
            report.append("Similar to:")
            for similar in redundancy['similar_courses']:
                report.append(f"- {similar['title']} (Similarity: {similar['similarity_score']:.2f})")
        
        # 6. Category Distribution
        report.append("\n=== Category Distribution Analysis ===")
        for category in insights['category_distribution']:
            report.append(f"\nCategory: {category['category']}")
            report.append(f"Count: {category['count']}")
            report.append(f"Percentage: {category['percentage']:.1f}%")
            report.append(f"Trend: {category['trend']}")
        
        # 7. Temporal Insights
        report.append("\n=== Content Velocity and Aging ===")
        temporal = insights['temporal_insights']
        report.append(f"New content in last 6 months: {temporal['content_velocity']}")
        report.append(f"Average monthly new content: {temporal['average_monthly_new_content']:.1f}")
        report.append(f"Oldest content age: {temporal['oldest_content_age']:.1f} years")
        
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
            missing_data = self.get_missing_data_analysis()
            for _, row in missing_data.iterrows():
                if row['missing_percentage'] > 5:
                    issues['Missing Data'].append(
                        f"{row['column']}: {row['missing_percentage']:.1f}% missing values"
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
            print(f"Warning: Error checking data quality: {str(e)}")
        
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