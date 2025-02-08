"""Constants and configuration values for the LMS Analyzer application."""

from typing import Dict, List

# Training Categories and Keywords
TRAINING_CATEGORIES: Dict[str, List[str]] = {
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

# Delivery Methods
DELIVERY_METHODS: Dict[str, str] = {
    'instructor_led_in_person': 'Traditional classroom',
    'instructor_led_virtual': 'Online live sessions',
    'self_paced_elearning': 'Asynchronous learning',
    'microlearning': 'Bite-sized content'
}

# Column Mappings for Standardization
COLUMN_MAPPING: Dict[str, str] = {
    # Course identifiers
    'course_id': 'course_no',
    'course_no': 'course_no',
    'offering_template_no': 'course_no',
    'course_number': 'course_no',
    'courseno': 'course_no',
    
    # Course title
    'course_title': 'course_title',
    'title': 'course_title',
    'name': 'course_title',
    'course name': 'course_title',
    
    # Course description
    'course_description': 'course_description',
    'description': 'course_description',
    'desc': 'course_description',
    'course desc': 'course_description',
    
    # Course abstract/summary
    'course_abstract': 'course_description',
    'abstract': 'course_description',
    'summary': 'course_description',
    'course_summary': 'course_description',
    'overview': 'course_description',
    
    # Activity metrics
    'total_2024_activity': 'total_2024_activity',
    'activity_count': 'total_2024_activity',
    'completions': 'total_2024_activity',
    
    # Duration
    'duration_mins': 'duration_mins',
    'dur mins': 'duration_mins',
    'duration minutes': 'duration_mins',
    'duration': 'duration_mins',
    'length_mins': 'duration_mins',
    'time_mins': 'duration_mins',
    
    # Hours per completion
    'avg hrs spent per completion': 'avg_hours_per_completion',
    'average hours per completion': 'avg_hours_per_completion',
    'avg hours per completion': 'avg_hours_per_completion',
    'hours per completion': 'avg_hours_per_completion',
    'avg_completion_hours': 'avg_hours_per_completion',
    
    # Source tracking
    'data_source': 'data_source',
    'source': 'data_source',
    'data source': 'data_source',
    
    # Cross references
    'cross_reference_count': 'cross_reference_count',
    'cross references': 'cross_reference_count',
    'xref_count': 'cross_reference_count',
    
    # Learner counts
    'learner_count': 'learner_count',
    'no_of_learners': 'learner_count',
    'students': 'learner_count',
    'participants': 'learner_count',
    'enrollment': 'learner_count',
    
    # Region/Entity
    'region': 'region_entity',
    'region entity': 'region_entity',
    'entity': 'region_entity',
    'business_unit': 'region_entity',
    
    # Course metadata
    'course_version': 'course_version',
    'version': 'course_version',
    'course created by': 'course_created_by',
    'creator': 'course_created_by',
    'author': 'course_created_by',
    'course_available_from': 'course_available_from',
    'avail_from': 'course_available_from',
    'start_date': 'course_available_from',
    'course_discontinued_from': 'course_discontinued_from',
    'disc_from': 'course_discontinued_from',
    'end_date': 'course_discontinued_from',
    'course_keywords': 'course_keywords',
    'keywords': 'course_keywords',
    'tags': 'course_keywords',
    'category name': 'category_name',
    'category': 'category_name',
    'course_category': 'category_name',
    
    # Course type
    'type': 'course_type',
    'course type': 'course_type',
    'person_type': 'person_type',
    'learner_type': 'person_type'
}

# Quality Score Weights
QUALITY_SCORE_WEIGHTS: Dict[str, float] = {
    'completeness': 0.4,
    'cross_references': 0.3,
    'data_validation': 0.3
}

# Required Fields for Data Quality
REQUIRED_FIELDS: List[str] = [
    'course_title',
    'course_description',
    'course_no',
    'category_name'
]

# Date Fields for Standardization
DATE_FIELDS: List[str] = [
    'course_version',
    'course_available_from',
    'course_discontinued_from'
]

# Numeric Fields for Standardization
NUMERIC_FIELDS: List[str] = [
    'duration_mins',
    'learner_count',
    'cross_reference_count',
    'total_2024_activity',
    'avg_hours_per_completion',
    'course_duration_hours'
]

# Boolean Fields for Standardization
BOOLEAN_FIELDS: List[str] = [
    'is_leadership_development',
    'is_managerial_supervisory',
    'is_mandatory_compliance',
    'is_profession_specific',
    'is_interpersonal_skills',
    'is_it_systems',
    'is_clinical',
    'is_nursing',
    'is_pharmacy',
    'is_safety',
    'has_direct_reports'
]

# Missing Data Options
MISSING_DATA_OPTIONS: Dict[str, Dict[str, Dict[str, str | int | float]]] = {
    'financial': {
        'cost': {'default': 500, 'description': 'Average cost per course'},
        'cost_per_learner': {'default': 50, 'description': 'Average cost per learner'},
        'reimbursement_amount': {'default': 1000, 'description': 'Average tuition reimbursement'}
    },
    'administrative': {
        'admin_count': {'default': 5, 'description': 'Average number of administrators'},
        'support_tickets': {'default': 10, 'description': 'Average monthly support tickets'},
        'resolution_time': {'default': 24, 'description': 'Average resolution time (hours)'}
    },
    'learning': {
        'duration_mins': {'default': 60, 'description': 'Average course duration (minutes)'},
        'learner_count': {'default': 100, 'description': 'Average learners per course'}
    }
} 