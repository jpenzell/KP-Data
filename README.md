# LMS Content Analysis Dashboard

A comprehensive tool for analyzing and visualizing LMS (Learning Management System) content data.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd KP-Data
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Setup Verification

Run the setup verification script to ensure everything is properly configured:
```bash
python src/test_setup.py
```

## Running the Application

1. Start the Streamlit application:
```bash
streamlit run src/main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Data Format

The application expects Excel files with the following columns:
- course_no
- course_title
- category_name
- course_description
- course_type
- region_entity
- course_available_from
- course_discontinued_from (optional)
- learner_count (optional)
- total_2024_activity (optional)

## Features

- Data validation and cleaning
- Quality analysis
- Activity tracking
- Content similarity detection
- Cross-referencing analysis
- Interactive visualizations
- Export capabilities

## Logging

Logs are stored in files named `lms_analyzer_YYYYMMDD.log` in the current directory.

## Troubleshooting

If you encounter any issues:

1. Check the log files for error messages
2. Ensure all required packages are installed
3. Verify your data format matches the expected structure
4. Check the Python version (3.8 or higher required)

## Support

For support or questions, please contact the development team.

## Project Structure

```
KP-Data/
├── src/                    # Source code
│   ├── main.py            # Main application entry point
│   ├── models/            # Data models and schemas
│   │   └── data_models.py # Pydantic models for data validation
│   ├── services/          # Core services
│   │   ├── data_processor.py  # Data processing service
│   │   └── analyzer.py    # Analysis service
│   ├── ui/                # User interface components
│   │   ├── pages/        # Page components
│   │   │   ├── home.py   # Home page
│   │   │   └── analysis.py # Analysis pages
│   │   └── components/   # Reusable UI components
│   │       ├── overview.py
│   │       ├── quality.py
│   │       └── similarity.py
│   ├── utils/            # Utility functions
│   ├── config/           # Configuration files
│   └── tests/            # Test files
├── docs/                 # Documentation
│   ├── development/      # Development documentation
│   ├── user-guide/       # User documentation
│   └── api/             # API documentation
├── requirements.txt      # Project dependencies
└── README.md            # This file
```

## Documentation

- [User Guide](docs/user-guide/GETTING_STARTED.md) - Getting started with the application
- [Development Guide](docs/development/) - Development documentation and guidelines
- [API Documentation](docs/api/) - API reference and integration guides

## Development

### Running Tests
```bash
python -m pytest src/tests/
```

### Code Style
The project follows PEP 8 guidelines. Use the following command to check code style:
```bash
flake8 src/
```

### Adding New Features
1. Create a new branch for your feature
2. Add tests in `src/tests/`
3. Implement the feature
4. Update documentation
5. Submit a pull request

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
