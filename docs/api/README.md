# API Documentation

## Data Models

### AnalysisResults
Main model for analysis results containing all metrics and insights.

```python
class AnalysisResults(BaseModel):
    # Basic metrics
    total_courses: int
    active_courses: int
    total_learners: Optional[int]
    regions_covered: Optional[int]
    data_sources: Optional[int]
    cross_referenced_courses: Optional[int]
    
    # Detailed metrics
    quality_metrics: Optional[QualityMetrics]
    activity_metrics: Optional[ActivityMetrics]
    similarity_metrics: Optional[SimilarityMetrics]
    
    # Analysis results
    recommendations: List[Recommendation]
    alerts: List[Alert]
    validation_results: List[ValidationResult]
```

### QualityMetrics
Metrics related to content quality.

```python
class QualityMetrics(BaseModel):
    completeness_score: float
    metadata_score: float
    content_score: float
    overall_score: float
    missing_fields: List[str]
    improvement_areas: List[str]
```

### ActivityMetrics
Metrics related to course activity and engagement.

```python
class ActivityMetrics(BaseModel):
    active_courses: int
    recent_completions: int
    average_completion_rate: float
    last_activity_date: Optional[datetime]
    activity_trend: Optional[Dict[str, float]]
```

### SimilarityMetrics
Metrics related to content similarity analysis.

```python
class SimilarityMetrics(BaseModel):
    total_pairs: int
    high_similarity_pairs: int
    cross_department_pairs: int
    average_similarity: float
    duplicate_candidates: List[Dict[str, Any]]
```

## Services

### DataProcessor
Service for processing and validating LMS data.

```python
class DataProcessor:
    def process_excel_files(self, file_paths: List[Path]) -> pd.DataFrame:
        """Process multiple Excel files and combine them into a single DataFrame."""
        
    def calculate_quality_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality scores for each course."""
```

### LMSAnalyzer
Service for analyzing LMS content and generating insights.

```python
class LMSAnalyzer:
    def get_analysis_results(self) -> AnalysisResults:
        """Generate comprehensive analysis results."""
        
    def _calculate_quality_metrics(self) -> QualityMetrics:
        """Calculate quality metrics for the content."""
        
    def _calculate_activity_metrics(self) -> ActivityMetrics:
        """Calculate activity metrics for the content."""
        
    def _calculate_similarity_metrics(self) -> SimilarityMetrics:
        """Calculate similarity metrics between courses."""
```

## UI Components

### Pages
- `home.py`: Main dashboard page
- `analysis.py`: Detailed analysis pages

### Components
- `overview.py`: Overview metrics component
- `quality.py`: Quality metrics component
- `similarity.py`: Similarity metrics component

## Configuration

The application uses configuration files in the `config/` directory for:
- Column mappings
- Field definitions
- Validation rules
- Analysis parameters

## Error Handling

The application uses Pydantic models for data validation and provides detailed error messages through the `ValidationResult` model. 