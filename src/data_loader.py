"""
Data Loader Module
Handles loading and validation of property and sentiment data
"""
import pandas as pd
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import sys

# Add parent to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROPERTY_DATA_FILE, SENTIMENT_CORPUS_FILE, PUNE_LOCALITIES


def load_property_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the Pune property dataset from CSV
    
    Args:
        file_path: Optional path to CSV file. Uses default if not provided.
        
    Returns:
        DataFrame with property data
    """
    path = file_path or PROPERTY_DATA_FILE
    
    if not path.exists():
        raise FileNotFoundError(
            f"Property data file not found at {path}. "
            f"Please ensure scraped data has been processed and placed in data/ directory."
        )
    
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['area', 'square_feet', 'num_bedrooms', 'num_bathrooms', 
                     'year_built', 'has_garage', 'price']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Clean and standardize area names
    df['area'] = df['area'].str.strip().str.title()
    
    # Calculate derived features
    df['property_age'] = 2024 - df['year_built']
    df['price_per_sqft'] = df['price'] / df['square_feet']
    
    print(f"✓ Loaded {len(df):,} properties from {path.name}")
    print(f"  Localities: {df['area'].nunique()} unique areas")
    
    return df


def load_sentiment_corpus(file_path: Optional[Path] = None) -> List[Dict]:
    """
    Load the sentiment text corpus from JSON
    
    Args:
        file_path: Optional path to JSON file. Uses default if not provided.
        
    Returns:
        List of sentiment text records
    """
    path = file_path or SENTIMENT_CORPUS_FILE
    
    if not path.exists():
        print(f"⚠ Sentiment corpus not found at {path}. Will generate synthetic data.")
        return []
    
    with open(path, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print(f"✓ Loaded {len(corpus)} sentiment records from {path.name}")
    
    return corpus


def get_unique_localities(df: pd.DataFrame) -> List[str]:
    """
    Extract unique locality names from property data
    
    Args:
        df: Property DataFrame
        
    Returns:
        List of unique locality names
    """
    return sorted(df['area'].unique().tolist())


def get_locality_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate statistics per locality
    
    Args:
        df: Property DataFrame
        
    Returns:
        DataFrame with locality-level statistics
    """
    stats = df.groupby('area').agg({
        'price': ['mean', 'median', 'std', 'count'],
        'square_feet': 'mean',
        'price_per_sqft': 'mean',
        'num_bedrooms': 'mean',
        'property_age': 'mean'
    }).round(2)
    
    # Flatten column names
    stats.columns = ['_'.join(col).strip() for col in stats.columns]
    stats = stats.reset_index()
    
    return stats


def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate property data for integrity
    
    Args:
        df: Property DataFrame
        
    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []
    
    # Check for nulls
    null_counts = df.isnull().sum()
    null_cols = null_counts[null_counts > 0]
    if len(null_cols) > 0:
        issues.append(f"Null values found: {null_cols.to_dict()}")
    
    # Check for negative values
    if (df['price'] <= 0).any():
        issues.append(f"Found {(df['price'] <= 0).sum()} properties with invalid prices")
    
    if (df['square_feet'] <= 0).any():
        issues.append(f"Found {(df['square_feet'] <= 0).sum()} properties with invalid square feet")
    
    # Check for reasonable ranges
    if df['num_bedrooms'].max() > 10:
        issues.append(f"Unusual bedroom count: max = {df['num_bedrooms'].max()}")
    
    is_valid = len(issues) == 0
    
    if is_valid:
        print("✓ Data validation passed")
    else:
        print(f"⚠ Data validation found {len(issues)} issues")
        for issue in issues:
            print(f"  - {issue}")
    
    return is_valid, issues


if __name__ == "__main__":
    # Test the data loader
    try:
        df = load_property_data()
        print("\nSample data:")
        print(df.head())
        
        print("\nLocality statistics:")
        stats = get_locality_stats(df)
        print(stats)
        
        validate_data(df)
    except FileNotFoundError as e:
        print(f"Error: {e}")
