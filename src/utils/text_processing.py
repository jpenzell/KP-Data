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