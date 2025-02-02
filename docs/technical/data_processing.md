# Data Processing Pipeline

## Overview
The LMS Content Analysis platform processes learning management system data through a sophisticated pipeline that handles data ingestion, standardization, analysis, and visualization.

## Data Ingestion
- **Supported Formats**: Excel files (.xlsx)
- **Multiple File Support**: System can process and merge multiple data sources
- **Column Standardization**: Automatic mapping of variant column names to standard format

### Standard Column Mappings
```python
{
    'course_title': ['Course Title', 'title'],
    'course_no': ['Course No', 'course_number'],
    'course_description': ['description', 'Course Description'],
    'course_abstract': ['abstract', 'Course Abstract'],
    'course_version': ['version', 'Course Version'],
    'course_created_by': ['created_by', 'Course Created By'],
    'course_available_from': ['avail_from', 'available_from'],
    'course_discontinued_from': ['disc_from', 'discontinued_from'],
    'duration_mins': ['duration'],
    'region_entity': ['region'],
    'person_type': ['person_type'],
    'course_type': ['type'],
    'learner_count': ['no_of_learners'],
    'course_keywords': ['course_keywords'],
    'activity_count': ['total_2024_activity']
}
```

## Data Processing Steps

1. **Initial Load**
   - File validation
   - Column name standardization
   - Data type inference

2. **Data Merging**
   - Hierarchical merge strategy using identifiers:
     1. course_id
     2. course_no
     3. offering_template_no
     4. course_title (fallback)
   - Cross-reference tracking
   - Duplicate handling

3. **Data Cleaning**
   - Date format standardization
   - Numeric value conversion
   - Missing value handling
   - Text field normalization

4. **Metadata Enhancement**
   - Quality score calculation
   - Cross-reference counting
   - Source tracking
   - Validation flags

## Data Validation

### Date Validation
- Tracks validity of key date fields:
  - course_version
  - course_available_from
  - course_discontinued_from
- Flags invalid dates
- Reports validation statistics

### Quality Scoring
Quality scores (0-1) are calculated based on:
- Completeness of required fields (40%)
- Date validity (30%)
- Additional metadata presence (30%)

## Current Data Statistics
Based on latest processing:
- Total Records: 12,259
- Invalid Dates:
  - course_version: 8,991
  - course_available_from: 2,399
  - course_discontinued_from: 9,177

## Content Categorization

### Category Detection
Uses weighted text analysis across multiple fields:
```python
weights = {
    'course_description': 1.0,
    'description': 1.0,
    'course_abstract': 0.8,
    'course_title': 1.2,
    'course_keywords': 1.5,
    'category_name': 1.3
}
```

### Training Categories
- Leadership Development
- Managerial/Supervisory
- Mandatory Compliance
- Profession Specific
- Interpersonal Skills
- IT Systems
- Clinical
- Nursing
- Pharmacy
- Safety

Each category uses exact and partial matching with confidence scoring.

## Performance Considerations
- Efficient pandas operations for large datasets
- Optimized merge strategies
- Memory-efficient data processing
- Scalable visualization generation 