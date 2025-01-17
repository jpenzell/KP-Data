# LMS Content Library Analyzer

This program helps analyze and normalize Excel spreadsheet data from Learning Management Systems (LMS), providing insights into your content library.

## Features

- Loads and processes Excel spreadsheet data
- Normalizes column names and data formats
- Generates basic statistics and insights
- Identifies duplicate entries
- Creates visualizations of data completeness
- Provides detailed summary reports

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place your Excel file in the project directory

3. Run the analyzer:
```bash
python lms_content_analyzer.py
```

## Usage

1. Modify the Excel file path in `lms_content_analyzer.py`:
```python
analyzer = LMSContentAnalyzer("your_excel_file.xlsx")
```

2. The program will:
   - Load and clean the data
   - Generate a summary report
   - Create a visualization of data completeness
   - Save the visualization as 'data_completeness.png'

## Output

The program generates:
- A console report with basic statistics
- Information about missing data
- A bar chart visualization showing data completeness
- Detailed analysis of potential duplicate entries 