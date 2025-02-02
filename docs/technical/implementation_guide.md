# Implementation Guide

## Getting Started

### Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Project Structure
```
lms_analyzer/
├── docs/
│   └── technical/
│       ├── data_processing.md
│       ├── dashboard_analysis.md
│       ├── system_architecture.md
│       ├── data_model.md
│       └── implementation_guide.md
├── src/
│   ├── __init__.py
│   ├── lms_analyzer_app.py
│   ├── lms_content_analyzer.py
│   └── utils/
│       ├── __init__.py
│       ├── data_processing.py
│       ├── visualization.py
│       └── validation.py
├── tests/
│   └── test_data/
├── requirements.txt
└── README.md
```

## Key Implementation Areas

### 1. Data Processing Implementation

#### Column Standardization
```python
def standardize_columns(df):
    """
    Standardize column names and handle variations
    """
    # Convert to lowercase and replace spaces
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    # Apply standard mappings
    column_mapping = {
        'title': 'course_title',
        'course_number': 'course_no',
        # ... other mappings
    }
    return df.rename(columns=column_mapping)
```

#### Date Handling
```python
def process_dates(df):
    """
    Handle multiple date formats and validate
    """
    date_columns = ['course_version', 'course_available_from', 'course_discontinued_from']
    
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='mixed', errors='coerce')
            df[f'{col}_is_valid'] = ~df[col].isna()
            
    return df
```

### 2. Content Analysis Implementation

#### Category Detection
```python
def detect_categories(text, categories, weights):
    """
    Detect categories with confidence scoring
    """
    scores = {}
    for category, keywords in categories.items():
        confidence = 0.0
        for keyword in keywords:
            # Exact match
            if f" {keyword} " in f" {text} ":
                confidence += 1.0
            # Partial match
            elif keyword in text:
                confidence += 0.5
        scores[category] = min(confidence / len(keywords), 1.0)
    return scores
```

### 3. Dashboard Implementation

#### Tab Creation
```python
def create_analysis_tabs():
    """
    Create and configure dashboard tabs
    """
    tabs = st.tabs([
        "Data Quality",
        "Training Categories",
        "Training Focus",
        # ... other tabs
    ])
    return tabs
```

#### Visualization
```python
def create_category_heatmap(df, categories):
    """
    Create category overlap heatmap
    """
    overlap_matrix = pd.DataFrame(
        index=categories,
        columns=categories
    )
    
    for cat1 in categories:
        for cat2 in categories:
            overlap = (df[f'is_{cat1}'] & 
                      df[f'is_{cat2}']).sum() / len(df) * 100
            overlap_matrix.loc[cat1, cat2] = overlap
            
    return px.imshow(
        overlap_matrix,
        title='Category Overlap Heatmap (%)',
        labels=dict(color="Overlap %")
    )
```

## Best Practices

### 1. Data Processing
- Always validate input data before processing
- Use consistent column naming conventions
- Handle missing values explicitly
- Log processing steps and errors

### 2. Performance Optimization
```python
# Example: Efficient pandas operations
def optimize_dataframe(df):
    # Reduce memory usage
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df
```

### 3. Error Handling
```python
def safe_process_file(file):
    try:
        df = pd.read_excel(file)
        df = standardize_columns(df)
        df = process_dates(df)
        return df, None
    except Exception as e:
        return None, str(e)
```

## Testing Guidelines

### 1. Data Validation Tests
```python
def test_data_validation():
    # Test required fields
    assert validate_required_fields(df), "Missing required fields"
    
    # Test date validity
    assert validate_dates(df), "Invalid date formats"
    
    # Test numeric ranges
    assert validate_numeric_ranges(df), "Invalid numeric values"
```

### 2. Category Detection Tests
```python
def test_category_detection():
    # Test exact matches
    text = "leadership development course"
    categories = detect_categories(text, TRAINING_CATEGORIES)
    assert categories['leadership_development'] > 0.8
    
    # Test partial matches
    text = "basic leader training"
    categories = detect_categories(text, TRAINING_CATEGORIES)
    assert categories['leadership_development'] > 0.3
```

## Deployment Steps

1. **Development**
   ```bash
   streamlit run src/lms_analyzer_app.py
   ```

2. **Testing**
   ```bash
   pytest tests/
   ```

3. **Production**
   ```bash
   # Set environment variables
   export STREAMLIT_SERVER_PORT=8501
   export STREAMLIT_SERVER_ADDRESS=0.0.0.0
   
   # Run with production settings
   streamlit run src/lms_analyzer_app.py --server.port $STREAMLIT_SERVER_PORT
   ```

## Troubleshooting Guide

### Common Issues

1. **Date Processing Errors**
   - Check input date formats
   - Verify timezone consistency
   - Validate date ranges

2. **Memory Issues**
   - Implement chunked processing
   - Use data type optimization
   - Clear unused dataframes

3. **Performance Issues**
   - Profile slow operations
   - Optimize pandas operations
   - Cache intermediate results

### Debugging Tips
```python
# Add debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

def process_data(df):
    logging.debug(f"Processing dataframe with shape: {df.shape}")
    logging.debug(f"Columns: {df.columns.tolist()}")
    logging.debug(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
``` 