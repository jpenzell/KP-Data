"""Text processing utilities for the LMS analyzer."""

import re
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def clean_text(text: str) -> str:
    """Clean and standardize text."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using TF-IDF and cosine similarity."""
    if not text1 or not text2:
        return 0.0
    
    # Clean texts
    text1 = clean_text(text1)
    text2 = clean_text(text2)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    try:
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return float(similarity)
    except:
        return 0.0

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using TF-IDF."""
    if not text:
        return []
    
    # Clean text
    text = clean_text(text)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_keywords)
    
    try:
        # Transform text to TF-IDF vector
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names (keywords)
        feature_names = vectorizer.get_feature_names_out()
        
        return list(feature_names)
    except:
        return []

def analyze_text_quality(text: str) -> Dict[str, Any]:
    """Analyze text quality metrics."""
    if not text:
        return {
            "length": 0,
            "word_count": 0,
            "sentence_count": 0,
            "avg_word_length": 0,
            "has_numbers": False,
            "has_special_chars": False
        }
    
    # Clean text
    text = clean_text(text)
    
    # Calculate metrics
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    metrics = {
        "length": len(text),
        "word_count": len(words),
        "sentence_count": len([s for s in sentences if s.strip()]),
        "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
        "has_numbers": bool(re.search(r'\d', text)),
        "has_special_chars": bool(re.search(r'[^\w\s]', text))
    }
    
    return metrics

def standardize_course_title(title: str) -> str:
    """Standardize course title format."""
    if not title:
        return ""
    
    # Clean text
    title = clean_text(title)
    
    # Capitalize first letter of each word
    title = title.title()
    
    # Remove common prefixes/suffixes
    title = re.sub(r'^(Course|Training|Module|Lesson)\s+', '', title)
    title = re.sub(r'\s+(Course|Training|Module|Lesson)$', '', title)
    
    return title.strip()

def standardize_course_number(course_no: str) -> tuple[str, str]:
    """
    Standardize course number format to handle variations like 'COMP0234' vs '0234'.
    
    This function:
    1. Removes all non-alphanumeric characters
    2. Extracts just the numeric portion if it exists
    3. Preserves common prefixes (like 'COMP', 'IT', etc.)
    
    Returns both the standardized full ID and the numeric portion for comparison.
    """
    if not course_no:
        return "", ""
    
    # Convert to string and clean up
    course_no = str(course_no).strip().upper()
    
    # Remove special characters except alphanumeric
    clean_id = re.sub(r'[^A-Z0-9]', '', course_no)
    
    # Extract the numeric portion (if any)
    numeric_match = re.search(r'(\d+)', clean_id)
    numeric_part = numeric_match.group(0) if numeric_match else ""
    
    # Common department prefixes to preserve
    common_prefixes = ['COMP', 'IT', 'HR', 'FIN', 'MKT', 'MGMT', 'BUS', 'TECH', 'ENG', 'SCI']
    
    # Check if the course ID starts with a known prefix
    prefix = ""
    for p in common_prefixes:
        if clean_id.startswith(p):
            prefix = p
            break
    
    # Create standardized version with prefix if it exists
    if prefix and numeric_part:
        standardized = f"{prefix}{numeric_part}"
    else:
        standardized = clean_id
    
    return standardized, numeric_part

def extract_acronyms(text: str) -> set:
    """Extract acronyms from text using pattern matching and context analysis."""
    acronyms = set()
    
    # Common patterns for acronyms
    patterns = [
        r'\b[A-Z]{2,}\b',  # Standard acronyms (2+ uppercase letters)
        r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\b',  # CamelCase or PascalCase
        r'\b[A-Z]\.(?:[A-Z]\.)+',  # A.B.C. style acronyms
        r'\b[A-Z]{2,}(?:\s+[A-Z]{2,})*\b',  # Multiple acronyms
        r'\b[A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)*\s*\([A-Z]+\)'  # Acronyms with definitions
    ]
    
    # Extract potential acronyms
    for pattern in patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            acronym = match.group(0)
            # Clean up the acronym
            acronym = re.sub(r'[^\w\s]', '', acronym)  # Remove special characters
            if len(acronym) >= 2:  # Only keep if at least 2 characters
                acronyms.add(acronym)
    
    # Context-based filtering
    filtered_acronyms = set()
    for acronym in acronyms:
        # Check if it's followed by a definition
        definition_pattern = f"{acronym}\\s*\\([^)]+\\)"
        if re.search(definition_pattern, text):
            filtered_acronyms.add(acronym)
            continue
        
        # Check if it's used multiple times
        if text.count(acronym) > 1:
            filtered_acronyms.add(acronym)
            continue
        
        # Check if it's in a list or table context
        if re.search(f"\\b{acronym}\\b\\s*[:;]", text):
            filtered_acronyms.add(acronym)
            continue
    
    return filtered_acronyms

def extract_technical_terms(text: str) -> set:
    """Extract technical terms from text using pattern matching and domain knowledge."""
    technical_terms = set()
    
    # Common technical term patterns
    patterns = [
        # Programming and development
        r'\b(?:function|method|class|interface|module|package|library|framework|API|SDK|IDE|CLI|GUI)\b',
        r'\b(?:variable|constant|parameter|argument|return|void|null|undefined|NaN|Infinity)\b',
        r'\b(?:loop|iteration|recursion|algorithm|data structure|array|list|set|map|dictionary)\b',
        r'\b(?:syntax|semantic|compiler|interpreter|runtime|debugger|profiler|optimizer)\b',
        
        # Database and data
        r'\b(?:database|table|schema|query|index|constraint|trigger|view|stored procedure)\b',
        r'\b(?:SQL|NoSQL|MongoDB|PostgreSQL|MySQL|Redis|Elasticsearch|Cassandra)\b',
        r'\b(?:ETL|data warehouse|data lake|data pipeline|data model|data type)\b',
        
        # Cloud and infrastructure
        r'\b(?:cloud|server|client|host|container|virtualization|microservice|API gateway)\b',
        r'\b(?:AWS|Azure|GCP|Kubernetes|Docker|Terraform|Ansible|Jenkins)\b',
        r'\b(?:load balancer|firewall|proxy|cache|CDN|DNS|SSL|TLS)\b',
        
        # Security and compliance
        r'\b(?:authentication|authorization|encryption|decryption|hash|salt|token|certificate)\b',
        r'\b(?:firewall|antivirus|malware|phishing|spam|DDoS|XSS|CSRF)\b',
        r'\b(?:GDPR|HIPAA|PCI DSS|SOC 2|ISO 27001|NIST)\b',
        
        # Project management and methodology
        r'\b(?:agile|scrum|kanban|waterfall|sprint|backlog|epic|story|task)\b',
        r'\b(?:CI/CD|DevOps|SRE|QA|UAT|staging|production|deployment)\b',
        r'\b(?:version control|git|branch|merge|commit|pull request|code review)\b'
    ]
    
    # Extract technical terms
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            term = match.group(0)
            technical_terms.add(term)
    
    # Extract compound technical terms
    compound_patterns = [
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # PascalCase terms
        r'\b[a-z]+(?:\s+[a-z]+)*\b(?:\s+[A-Z][a-z]+)+',  # Terms with technical suffixes
        r'\b[A-Z]+(?:\s+[A-Z]+)*\b(?:\s+[A-Z][a-z]+)*'  # Acronyms with technical terms
    ]
    
    for pattern in compound_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            term = match.group(0)
            if len(term.split()) > 1:  # Only add compound terms
                technical_terms.add(term)
    
    return technical_terms
