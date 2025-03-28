#!/usr/bin/env python3
"""Entry point script for the LMS Content Analysis Dashboard."""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import the main function
from src.main import main

if __name__ == "__main__":
    main() 