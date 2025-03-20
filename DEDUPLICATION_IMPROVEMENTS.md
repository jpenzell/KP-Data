# Enhanced Course Deduplication System

This document explains the improvements made to the course deduplication system in the LMS Analyzer application to achieve a more accurate count of unique courses.

## Problem Statement

The original deduplication approach had several limitations:

1. It used a simplistic deduplication key based primarily on `course_no`
2. It added source-specific prefixes (like "source_1_") that prevented cross-source matching
3. It didn't account for variations in course names or slight differences in identifiers
4. It didn't use multiple fields together to identify duplicates

This led to an artificially high course count, where the same course appearing in multiple data sources would be counted multiple times.

## Solution: Multi-layered Deduplication

The enhanced deduplication system implements a multi-layered approach:

### 1. Improved Deduplication Key Creation

The system now creates more robust deduplication keys that:

- Removes source prefixes from course identifiers, enabling cross-source matching
- Uses a hierarchy of identifiers (course_no → offering_template_no → title)
- Handles generic or missing values more intelligently
- Standardizes and cleans input fields before using them for matching

### 2. Composite Keys for Stronger Matching

For courses without strong identifiers (like a course number), the system creates composite keys using:

- Normalized course titles
- Description fingerprints (first 50 characters)
- Category/department information
- Duration (binned to nearest 30 minutes)

This creates a more unique signature for each course, even when direct identifiers are missing.

### 3. Fuzzy String Matching

For courses with slight variations in titles or identifiers, a second-pass fuzzy matching system:

- Uses Jaro-Winkler string similarity to find courses with similar titles
- Applies a similarity threshold (default 85%)
- Processes in batches to maintain performance
- Focuses on high-confidence courses as reference points

### 4. Improved Aggregation Strategy

When combining duplicate courses, the system now:

- Uses intelligent field-specific aggregation rules
- Selects the most informative values (e.g., longest title/description)
- Properly handles dates, numeric fields, and boolean flags
- Combines cross-references and source information

## Results

The enhanced deduplication provides:

- A more accurate count of unique courses
- Better quality data by combining information from multiple sources
- Proper tracking of cross-references between sources
- Detailed feedback on the deduplication process

The system is also designed to scale, with:
- Chunked processing for large datasets
- Sampling techniques for performance optimization
- Thorough logging and metrics reporting

## Technical Implementation

The key implementation components include:

1. `create_deduplication_key()` - Creates sophisticated deduplication keys
2. `fuzzy_match_courses()` - Identifies similar courses using string similarity
3. Enhanced aggregation in `merge_and_standardize_data()` - Uses field-specific strategies

These components work together to provide a more accurate and robust deduplication system. 