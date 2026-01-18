"""
Geospatial Data Module
Provides latitude/longitude coordinates for Pune localities
"""
import pandas as pd
from typing import Dict, List, Tuple

# Approximate coordinates for Pune localities
# Format: "Locality Name": (Latitude, Longitude)
LOCALITY_COORDINATES = {
    "Koregaon Park": (18.5362, 73.8939),
    "Hinjewadi": (18.5913, 73.7389),
    "Pimpri-Chinchwad": (18.6298, 73.7997),
    "Viman Nagar": (18.5679, 73.9143),
    "Kalyani Nagar": (18.5463, 73.9033),
    "Baner": (18.5590, 73.7868),
    "Wakad": (18.5983, 73.7756),
    "Kharadi": (18.5515, 73.9348),
    "Hadapsar": (18.5089, 73.9260),
    "Magarpatta": (18.5158, 73.9272),
    "Aundh": (18.5580, 73.8075),
    "Shivaji Nagar": (18.5314, 73.8446),
    "Deccan": (18.5158, 73.8418),
    "Kothrud": (18.5074, 73.8077),
    "Warje": (18.4800, 73.8000),
    "Undri": (18.4578, 73.9197),
    "Kondhwa": (18.4695, 73.8887),
    "NIBM Road": (18.4750, 73.9000),
    "Bavdhan": (18.5132, 73.7766),
    "Pashan": (18.5350, 73.7900)
}

def get_locality_coordinates_df(rankings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rankings dataframe with coordinate data
    
    Args:
        rankings_df: DataFrame containing 'area' column
        
    Returns:
        DataFrame with 'lat' and 'lon' columns added
    """
    df = rankings_df.copy()
    
    # Add coordinates
    df['lat'] = df['area'].map(lambda x: LOCALITY_COORDINATES.get(x, (None, None))[0])
    df['lon'] = df['area'].map(lambda x: LOCALITY_COORDINATES.get(x, (None, None))[1])
    
    # Filter out unmapped localities (if any)
    mapped_df = df.dropna(subset=['lat', 'lon'])
    
    if len(mapped_df) < len(df):
        missing = set(df['area']) - set(mapped_df['area'])
        print(f"Warning: Could not map coordinates for: {missing}")
        
    return mapped_df
