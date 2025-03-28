"""Main entry point for the LMS analyzer package."""

import os
import sys

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

# Import main module
from src.main import main

if __name__ == "__main__":
    main() 