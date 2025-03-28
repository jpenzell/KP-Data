"""File utilities for the LMS analyzer."""

import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime

def ensure_directory(path: Path) -> None:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)

def get_file_extension(file_path: Path) -> str:
    """Get file extension."""
    return file_path.suffix.lower()

def is_valid_file(file_path: Path, allowed_extensions: List[str] = ['.xlsx', '.xls']) -> bool:
    """Check if file is valid."""
    return file_path.exists() and get_file_extension(file_path) in allowed_extensions

def read_excel_file(file_path: Path) -> pd.DataFrame:
    """Read Excel file."""
    try:
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return pd.DataFrame()

def save_excel_file(df: pd.DataFrame, file_path: Path) -> bool:
    """Save DataFrame to Excel file."""
    try:
        df.to_excel(file_path, index=False)
        return True
    except Exception as e:
        print(f"Error saving file {file_path}: {str(e)}")
        return False

def get_file_info(file_path: Path) -> Dict[str, Any]:
    """Get file information."""
    if not file_path.exists():
        return {}
    
    stats = file_path.stat()
    
    return {
        "name": file_path.name,
        "extension": get_file_extension(file_path),
        "size": stats.st_size,
        "created": datetime.fromtimestamp(stats.st_ctime),
        "modified": datetime.fromtimestamp(stats.st_mtime),
        "accessed": datetime.fromtimestamp(stats.st_atime)
    }

def list_files(directory: Path, pattern: str = "*") -> List[Path]:
    """List files in directory."""
    if not directory.exists():
        return []
    
    return list(directory.glob(pattern))

def create_backup(file_path: Path) -> Path:
    """Create backup of file."""
    if not file_path.exists():
        return file_path
    
    backup_path = file_path.with_suffix(f"{file_path.suffix}.bak")
    try:
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        print(f"Error creating backup: {str(e)}")
        return file_path

def cleanup_old_files(directory: Path, pattern: str = "*.bak", days: int = 30) -> List[Path]:
    """Clean up old files."""
    if not directory.exists():
        return []
    
    removed_files = []
    cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
    
    for file_path in directory.glob(pattern):
        if file_path.stat().st_mtime < cutoff_date:
            try:
                file_path.unlink()
                removed_files.append(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {str(e)}")
    
    return removed_files

def get_unique_filename(directory: Path, base_name: str, extension: str) -> Path:
    """Get unique filename in directory."""
    counter = 1
    while True:
        if counter == 1:
            filename = f"{base_name}{extension}"
        else:
            filename = f"{base_name}_{counter}{extension}"
        
        file_path = directory / filename
        if not file_path.exists():
            return file_path
        
        counter += 1 