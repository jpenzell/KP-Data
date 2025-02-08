"""Setup file for LMS Content Analysis package."""

from setuptools import setup, find_packages

setup(
    name="lms_analyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "streamlit>=1.30.0",
        "plotly>=5.18.0",
        "matplotlib>=3.8.0",
        "wordcloud>=1.9.0",
        "openpyxl>=3.1.0",
        "python-dateutil>=2.8.0",
        "typing-extensions>=4.8.0"
    ],
    python_requires=">=3.8",
) 