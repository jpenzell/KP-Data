# LMS Content Analysis Platform

## Overview
A comprehensive learning management system (LMS) content analysis platform that processes, analyzes, and visualizes training data across multiple dimensions. The system provides insights through interactive dashboards covering training focus, administrative metrics, content usage, and more.

## Current Status
- **Total Records**: 12,259 courses analyzed
- **Data Sources**: Multiple Excel files integrated
- **Key Metrics**: Training categories, usage patterns, and quality scores implemented
- **Documentation**: Complete technical documentation available

## Quick Start

### Installation
```bash
# Clone the repository
git clone [repository-url]
cd lms-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/lms_analyzer_app.py
```

## Documentation Index

### Technical Documentation
1. [Data Processing Pipeline](docs/technical/data_processing.md)
   - Data ingestion and standardization
   - Processing steps
   - Content categorization

2. [Dashboard Analysis](docs/technical/dashboard_analysis.md)
   - Analysis components
   - Visualization details
   - Metrics and KPIs

3. [System Architecture](docs/technical/system_architecture.md)
   - Component architecture
   - Data flow
   - Technical stack

4. [Data Model](docs/technical/data_model.md)
   - Core data structures
   - Validation rules
   - Quality metrics

5. [Implementation Guide](docs/technical/implementation_guide.md)
   - Setup instructions
   - Code examples
   - Best practices

6. [Data Quality Report](docs/technical/data_quality_report.md)
   - Current statistics
   - Quality issues
   - Recommendations

## Known Issues and Next Steps

### Critical Issues to Address
1. **Date Validation**
   - High number of invalid course versions (8,991)
   - Missing discontinued dates (9,177)
   - Priority: High

2. **Data Quality**
   - Temporal data gaps
   - Cross-reference inconsistencies
   - Priority: Medium

3. **Performance**
   - Large dataset optimization needed
   - Caching implementation required
   - Priority: Medium

### Immediate Next Steps
1. Implement date validation improvements
2. Add data quality monitoring
3. Optimize large dataset handling
4. Complete category confidence scoring

### Future Enhancements
1. Advanced filtering capabilities
2. Custom report generation
3. API development
4. Machine learning integration

## Project Structure
```
lms_analyzer/
├── docs/               # Complete documentation
├── src/               # Source code
├── tests/             # Test suite
└── requirements.txt   # Dependencies
```

## Key Files
- `src/lms_analyzer_app.py`: Main application
- `src/lms_content_analyzer.py`: Core analysis engine
- `src/utils/`: Utility functions

## Development Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Document all functions
- Add unit tests for new features

### Git Workflow
- Create feature branches
- Write descriptive commits
- Submit PRs for review
- Keep documentation updated

## Support and Contact
- Original Developer: [Your Contact Info]
- Project Manager: [PM Contact]
- Documentation: [Links to additional resources]

## License
[License Information]
