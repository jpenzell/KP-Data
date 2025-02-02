# Data Model Documentation

## Core Data Structure

### Course Data
```python
{
    # Primary Identifiers
    'course_id': str,          # Unique identifier
    'course_no': str,          # Course number
    'offering_template_no': str,# Template identifier
    
    # Basic Information
    'course_title': str,       # Course title
    'course_description': str, # Detailed description
    'course_abstract': str,    # Brief overview
    'course_keywords': str,    # Keywords/tags
    
    # Classification
    'course_type': str,        # Type of course
    'category_name': str,      # Primary category
    'person_type': str,        # Target audience
    
    # Temporal Data
    'course_version': datetime,# Version date
    'course_available_from': datetime,  # Start date
    'course_discontinued_from': datetime,# End date
    
    # Usage Metrics
    'learner_count': int,      # Total learners
    'activity_count': int,     # Activity metric
    'duration_mins': int,      # Course duration
    
    # Administrative
    'course_created_by': str,  # Author
    'region_entity': str,      # Regional designation
    'has_direct_reports': bool,# Management flag
    
    # System Fields
    'data_source': str,        # Source identifier
    'data_sources': str,       # Combined sources
    'cross_reference_count': int,# Reference count
}
```

## Derived Data

### Category Flags
```python
{
    'is_leadership_development': bool,
    'is_managerial_supervisory': bool,
    'is_mandatory_compliance': bool,
    'is_profession_specific': bool,
    'is_interpersonal_skills': bool,
    'is_it_systems': bool,
    'is_clinical': bool,
    'is_nursing': bool,
    'is_pharmacy': bool,
    'is_safety': bool
}
```

### Confidence Scores
```python
{
    'leadership_development_confidence': float,
    'managerial_supervisory_confidence': float,
    'mandatory_compliance_confidence': float,
    'profession_specific_confidence': float,
    'interpersonal_skills_confidence': float,
    'it_systems_confidence': float,
    'clinical_confidence': float,
    'nursing_confidence': float,
    'pharmacy_confidence': float,
    'safety_confidence': float
}
```

### Validation Flags
```python
{
    'course_version_is_valid': bool,
    'course_available_from_is_valid': bool,
    'course_discontinued_from_is_valid': bool
}
```

## Data Relationships

### Primary Keys
- course_id (preferred)
- course_no (alternate)
- offering_template_no (alternate)
- course_title (fallback)

### Foreign Keys
- region_entity → regions
- person_type → audience_types
- course_type → course_types
- category_name → categories

## Data Validation Rules

### Required Fields
- course_title
- course_no OR course_id
- course_description OR course_abstract

### Date Validation
- course_version: Valid date
- course_available_from: Valid date
- course_discontinued_from: Valid date or null
- course_available_from < course_discontinued_from

### Numeric Validation
- duration_mins: ≥ 0
- learner_count: ≥ 0
- activity_count: ≥ 0
- cross_reference_count: ≥ 1

## Data Quality Metrics

### Quality Score Components
1. **Completeness (40%)**
   - Required fields presence
   - Optional fields presence
   
2. **Date Validity (30%)**
   - Valid date formats
   - Logical date relationships
   
3. **Metadata Quality (30%)**
   - Field population
   - Cross-reference presence

### Confidence Score Calculation
- Exact match: 1.0 weight
- Partial match: 0.5 weight
- Normalized to 0-1 range
- Threshold: 0.15

## Data Transformations

### Text Standardization
- Lowercase conversion
- Whitespace normalization
- Special character handling

### Date Standardization
- Multiple format parsing
- Timezone handling
- Invalid date flagging

### Numeric Standardization
- Type conversion
- Missing value handling
- Range validation

## Extension Points

### Custom Fields
- Flexible schema support
- Additional metadata fields
- Custom categorization

### Derived Metrics
- Custom calculations
- Aggregated statistics
- Cross-reference metrics

## Current Data Statistics

### Volume Metrics
- Total Records: 12,259
- Unique Courses: [Based on course_id]
- Data Sources: [Count of unique sources]

### Quality Metrics
- Average Quality Score: [0-1]
- Cross-Reference Rate: [%]
- Field Completeness: [%]

### Invalid Data
- Invalid Dates: 
  - course_version: 8,991
  - course_available_from: 2,399
  - course_discontinued_from: 9,177 