"""Date utilities for the LMS analyzer."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import pandas as pd

def parse_date(date_str: str) -> Optional[datetime]:
    """Parse date string to datetime object."""
    if not date_str or pd.isna(date_str):
        return None
    
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def get_date_range(start_date: datetime, end_date: Optional[datetime] = None) -> Dict[str, datetime]:
    """Get date range for analysis."""
    if not end_date:
        end_date = datetime.now()
    
    return {
        "start": start_date,
        "end": end_date
    }

def get_recent_date_range(days: int = 90) -> Dict[str, datetime]:
    """Get date range for recent analysis."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    return {
        "start": start_date,
        "end": end_date
    }

def format_date(date: datetime) -> str:
    """Format date for display."""
    if not date:
        return "N/A"
    return date.strftime("%Y-%m-%d")

def get_date_difference(date1: datetime, date2: datetime) -> Dict[str, Any]:
    """Calculate difference between two dates."""
    if not date1 or not date2:
        return {
            "days": 0,
            "months": 0,
            "years": 0
        }
    
    diff = date2 - date1
    
    return {
        "days": diff.days,
        "months": diff.days // 30,
        "years": diff.days // 365
    }

def is_date_in_range(date: datetime, start_date: datetime, end_date: datetime) -> bool:
    """Check if date is within range."""
    if not date:
        return False
    return start_date <= date <= end_date

def get_date_bins(start_date: datetime, end_date: datetime, bin_size: str = "M") -> pd.DatetimeIndex:
    """Get date bins for analysis."""
    return pd.date_range(start=start_date, end=end_date, freq=bin_size)

def aggregate_by_date(df: pd.DataFrame, date_col: str, value_col: str, bin_size: str = "M") -> pd.DataFrame:
    """Aggregate data by date bins."""
    if date_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create date bins
    df['date_bin'] = df[date_col].dt.to_period(bin_size)
    
    # Aggregate data
    agg_df = df.groupby('date_bin')[value_col].sum().reset_index()
    
    # Convert period to datetime
    agg_df['date_bin'] = agg_df['date_bin'].dt.to_timestamp()
    
    return agg_df 