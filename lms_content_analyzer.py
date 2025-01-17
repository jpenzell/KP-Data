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

class LMSContentAnalyzer:
    def __init__(self, excel_path: str):
        """Initialize the analyzer with the path to the Excel file."""
        self.excel_path = Path(excel_path)
        self.df = None
        self.date_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self.text_columns = []
        self.quality_scores = {}
        self.content_clusters = {}
        self.similar_courses = {}
        self.load_data()
        self.clean_column_names()
        self.classify_columns()
        self.calculate_quality_scores()
        self.analyze_content_similarity()
    
    def load_data(self) -> None:
        """Load the Excel file into a pandas DataFrame."""
        if not self.excel_path.exists():
            print(f"Error: File '{self.excel_path}' does not exist.")
            print("Please make sure your Excel file is in the correct location and try again.")
            sys.exit(1)
            
        try:
            self.df = pd.read_excel(self.excel_path)
            print(f"Successfully loaded {len(self.df)} rows of data")
            print("\nOriginal column names:")
            for col in self.df.columns:
                print(f"- {col}")
        except Exception as e:
            print(f"Error loading file: {e}")
            print("Please make sure the file is a valid Excel file and try again.")
            sys.exit(1)

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the dataset."""
        stats = {
            'total_rows': len(self.df),
            'columns': list(self.df.columns),
            'missing_values': self.df.isnull().sum().to_dict(),
            'data_types': self.df.dtypes.to_dict()
        }
        return stats

    def clean_column_names(self) -> None:
        """Normalize column names by removing spaces, special characters, and converting to lowercase."""
        original_columns = list(self.df.columns)
        self.df.columns = self.df.columns.str.lower().str.replace(' ', '_')
        self.df.columns = self.df.columns.str.replace('[^a-z0-9_]', '', regex=True)
        
        # Print column name mapping
        print("\nColumn name mapping:")
        for orig, new in zip(original_columns, self.df.columns):
            print(f"- '{orig}' -> '{new}'")

    def classify_columns(self) -> None:
        """Automatically classify columns into different types for analysis."""
        date_patterns = ['available_from', 'discontinued_from', 'date', 'created', 'modified', 'version']
        
        for col in self.df.columns:
            # Check for date columns based on patterns and content
            if any(pattern in col.lower() for pattern in date_patterns):
                self.date_columns.append(col)
            # Check other types
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_columns.append(col)
            elif self.df[col].nunique() < len(self.df) * 0.1:  # If less than 10% unique values
                self.categorical_columns.append(col)
            else:
                self.text_columns.append(col)
        
        print("\nColumn Classification:")
        print("\nDate columns:", self.date_columns)
        print("Numeric columns:", self.numeric_columns)
        print("Categorical columns:", self.categorical_columns)
        print("Text columns:", self.text_columns)

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
        
        # 1. Time series plot for date columns
        for col in self.date_columns:
            try:
                plt.figure(figsize=(15, 6))
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                self.df[col].value_counts().sort_index().plot(kind='line')
                plt.title(f'Time Distribution - {col}')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(plots_dir / f'time_distribution_{col}.png')
                plt.close()
            except Exception as e:
                print(f"Warning: Could not create time series plot for '{col}': {str(e)}")
        
        # 2. Category distribution plots
        for col in self.categorical_columns:
            plt.figure(figsize=(12, 6))
            self.df[col].value_counts().nlargest(10).plot(kind='bar')
            plt.title(f'Top 10 Categories - {col}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(plots_dir / f'category_distribution_{col}.png')
            plt.close()
        
        # 3. Missing data visualization
        plt.figure(figsize=(12, 6))
        missing_data = (self.df.isnull().sum() / len(self.df)) * 100
        missing_data.sort_values(ascending=False).plot(kind='bar')
        plt.title('Missing Data by Column')
        plt.ylabel('Percentage Missing')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(plots_dir / 'missing_data.png')
        plt.close()

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