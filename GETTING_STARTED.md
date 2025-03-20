# Getting Started with KP-Data

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/jpenzell/KP-Data.git
   cd KP-Data
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run lms_analyzer_app.py
   ```

## Project Structure Overview

- `lms_analyzer_app.py`: Main Streamlit application and user interface
- `lms_content_analyzer.py`: Core analysis engine and data processing logic
- `src/`: Additional source code modules
- `docs/`: Project documentation
  - `docs/technical/`: Technical documentation and implementation guides
  - `docs/strategic_objectives.md`: Strategic objectives for the project

## Common Tasks

### 1. Data Processing

The application processes multiple data sources and merges them together. The data processing logic is primarily in `lms_analyzer_app.py` and `lms_content_analyzer.py`.

### 2. Adding New Features

To add new features:
1. Review the technical documentation in the `docs/technical/` directory
2. Update the relevant files based on the feature type:
   - UI changes: Modify `lms_analyzer_app.py`
   - Analysis logic: Modify `lms_content_analyzer.py`
   - Data model: Update according to `docs/technical/data_model.md`

### 3. Fixing Known Issues

Refer to the "Known Issues and Next Steps" section in the README.md file for priority tasks:
- Date validation improvements
- Data quality monitoring
- Performance optimizations
- Category confidence scoring

## Troubleshooting

If you encounter issues:

1. **Import errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Data loading errors**: Check file paths and formats

3. **Performance issues**: Refer to the implementation guide for optimization tips
   ```
   docs/technical/implementation_guide.md
   ```

## Development Guidelines

- Follow the PEP 8 style guide for Python code
- Use type hints for function parameters and return values
- Document all functions and classes
- Add unit tests for new features

## Contact

For questions or assistance, refer to the contact information in the README.md file. 