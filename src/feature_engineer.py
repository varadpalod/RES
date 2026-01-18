"""
Feature Engineering Module
Combines property attributes with sentiment features for ML
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROCESSED_DATA_FILE


class FeatureEngineer:
    """
    Feature engineering for real estate demand prediction
    Combines property attributes with locality sentiment
    """
    
    def __init__(self):
        self.sentiment_features = [
            'price_perception',
            'infrastructure_satisfaction',
            'investment_confidence',
            'buying_urgency',
            'overall_sentiment',
            'sentiment_volatility'
        ]
        
        self.property_features = [
            'square_feet',
            'num_bedrooms',
            'num_bathrooms',
            'property_age',
            'has_garage'
        ]
        
        # New enhanced features
        self.enhanced_features = [
            'rera_registered',
            'rera_expiry_years',
            'avg_rental_yield',
            'estimated_monthly_rent',
            'metro_distance_km',
            'it_park_distance_km',
            'avg_commute_time_min',
            'locality_highlights_count',
            'locality_issues_count',
            'locality_net_score'
        ]
        
        self.affordability_features = [
            'down_payment',
            'loan_amount',
            'monthly_emi',
            'min_monthly_income',
            'min_annual_income'
        ]
        
        self.locality_encodings = {}
    
    def merge_property_sentiment(self, 
                                 property_df: pd.DataFrame,
                                 sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge property data with locality sentiment
        
        Args:
            property_df: Property data with 'area' column
            sentiment_df: Locality sentiment profiles
            
        Returns:
            Merged DataFrame
        """
        # Standardize column names
        property_df = property_df.copy()
        sentiment_df = sentiment_df.copy()
        
        # Ensure area column is title case
        property_df['area'] = property_df['area'].str.strip().str.title()
        sentiment_df['locality'] = sentiment_df['locality'].str.strip().str.title()
        
        # Merge on locality
        merged = property_df.merge(
            sentiment_df,
            left_on='area',
            right_on='locality',
            how='left'
        )
        
        # Fill missing sentiment with neutral values
        for col in self.sentiment_features:
            if col in merged.columns:
                if col == 'buying_urgency':
                    merged[col] = merged[col].fillna(0.5)
                else:
                    merged[col] = merged[col].fillna(0)
        
        print(f"✓ Merged {len(merged):,} properties with sentiment data")
        print(f"  Properties with sentiment: {merged['overall_sentiment'].notna().sum():,}")
        
        return merged
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features for ML model
        
        Args:
            df: Merged property + sentiment DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Property-based features
        df['price_per_sqft'] = df['price'] / df['square_feet']
        df['rooms_total'] = df['num_bedrooms'] + df['num_bathrooms']
        df['sqft_per_bedroom'] = df['square_feet'] / df['num_bedrooms'].replace(0, 1)
        df['is_new_construction'] = (df['property_age'] <= 3).astype(int)
        df['is_old_property'] = (df['property_age'] > 20).astype(int)
        
        # Configuration category
        df['config_category'] = pd.cut(
            df['num_bedrooms'],
            bins=[0, 1, 2, 3, 10],
            labels=['1BHK', '2BHK', '3BHK', '4BHK+']
        )
        
        # RERA-based features
        if 'rera_registered' in df.columns:
            df['rera_score'] = df['rera_registered'] + (df['rera_expiry_years'] / 5).clip(0, 1)
        
        # Rental yield features
        if 'avg_rental_yield' in df.columns:
            df['yield_premium'] = df['avg_rental_yield'] - df['avg_rental_yield'].mean()
            df['rent_to_price_ratio'] = df['estimated_monthly_rent'] / df['price'] * 100
        
        # Locality connectivity score
        if 'metro_distance_km' in df.columns:
            df['connectivity_score'] = (
                (10 - df['metro_distance_km'].clip(0, 10)) / 10 * 0.4 +
                (15 - df['it_park_distance_km'].clip(0, 15)) / 15 * 0.4 +
                (60 - df['avg_commute_time_min'].clip(0, 60)) / 60 * 0.2
            )
        
        # Locality quality score from highlights
        if 'locality_net_score' in df.columns:
            df['locality_quality'] = (df['locality_net_score'] + 3) / 6  # Normalize to 0-1
        
        # Sentiment-adjusted features
        if 'price_perception' in df.columns:
            # Value score: combines actual price with price perception
            df['sentiment_value_score'] = (
                df['price_perception'] * 0.4 + 
                df['infrastructure_satisfaction'] * 0.3 +
                df['investment_confidence'] * 0.3
            )
            
            # Demand indicator based on urgency and sentiment
            df['demand_indicator'] = (
                df['buying_urgency'] * 0.3 + 
                (df['overall_sentiment'] + 1) / 2 * 0.4 +  # Normalize to 0-1
                (df['investment_confidence'] + 1) / 2 * 0.3
            )
        
        # Locality encoding (target encoding using price)
        locality_price_mean = df.groupby('area')['price'].transform('mean')
        overall_mean = df['price'].mean()
        df['locality_price_index'] = locality_price_mean / overall_mean
        
        print(f"✓ Created {len([c for c in df.columns if c not in self.property_features])} derived features")
        
        return df
    
    def calculate_demand_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variable for demand prediction
        
        The demand score is based on:
        - Price premium/discount relative to locality average
        - Configuration popularity
        - Property attributes quality
        
        Args:
            df: Feature DataFrame
            
        Returns:
            DataFrame with demand_score target
        """
        df = df.copy()
        
        # Price premium relative to locality
        locality_avg = df.groupby('area')['price'].transform('mean')
        df['price_premium'] = (df['price'] - locality_avg) / locality_avg
        
        # Normalize to 0-1 demand score
        # Lower price premium = higher demand (more affordable)
        # Better sentiment = higher demand
        
        price_component = 1 - (df['price_premium'].clip(-0.5, 0.5) + 0.5)  # Invert so lower price = higher demand
        
        if 'demand_indicator' in df.columns:
            sentiment_component = df['demand_indicator']
        else:
            sentiment_component = 0.5
        
        # Quality component based on property features
        quality_component = (
            (df['has_garage'] * 0.2) +
            (df['sqft_per_bedroom'].clip(0, 500) / 500 * 0.3) +
            ((1 - df['property_age'].clip(0, 30) / 30) * 0.5)
        )
        
        df['demand_score'] = (
            price_component * 0.35 +
            sentiment_component * 0.35 +
            quality_component * 0.30
        )
        
        # Normalize to 0-100 scale
        df['demand_score'] = (df['demand_score'] * 100).clip(0, 100).round(1)
        
        print(f"✓ Created demand_score target (mean: {df['demand_score'].mean():.1f})")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns for ML model"""
        
        exclude_cols = [
            'id', 'area', 'locality', 'price', 'demand_score',
            'year_built', 'config_category', 'latest_date', 'oldest_date',
            'sample_count', 'price_premium', 'income_bracket', 'rera_number'
        ]
        
        # Also exclude std columns and string columns
        exclude_cols.extend([c for c in df.columns if c.endswith('_std')])
        
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        return feature_cols
    
    def prepare_ml_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare data for ML model training
        
        Args:
            df: Feature DataFrame with demand_score
            
        Returns:
            Tuple of (X features, y target, feature_names)
        """
        feature_cols = self.get_feature_columns(df)
        
        X = df[feature_cols].copy()
        y = df['demand_score'].copy()
        
        # Fill any remaining NaN
        X = X.fillna(0)
        
        print(f"✓ Prepared ML data: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        return X, y, feature_cols
    
    def save_processed_data(self, df: pd.DataFrame, 
                           file_path: Optional[Path] = None) -> None:
        """Save processed data to CSV"""
        path = file_path or PROCESSED_DATA_FILE
        df.to_csv(path, index=False)
        print(f"✓ Saved processed data to {path}")


if __name__ == "__main__":
    # Test with sample data
    from data_loader import load_property_data
    
    try:
        # This will fail if data not present, just for testing
        prop_df = pd.DataFrame({
            'area': ['Hinjewadi', 'Hinjewadi', 'Baner', 'Baner'],
            'square_feet': [1000, 1200, 1500, 1100],
            'num_bedrooms': [2, 2, 3, 2],
            'num_bathrooms': [2, 2, 2, 2],
            'year_built': [2020, 2018, 2022, 2015],
            'has_garage': [1, 0, 1, 1],
            'price': [5000000, 5500000, 8000000, 6000000]
        })
        
        sentiment_df = pd.DataFrame({
            'locality': ['Hinjewadi', 'Baner'],
            'price_perception': [-0.2, 0.3],
            'infrastructure_satisfaction': [0.4, 0.6],
            'investment_confidence': [0.7, 0.5],
            'buying_urgency': [0.6, 0.4],
            'overall_sentiment': [0.3, 0.5],
            'sentiment_volatility': [0.2, 0.1]
        })
        
        engineer = FeatureEngineer()
        
        merged = engineer.merge_property_sentiment(prop_df, sentiment_df)
        featured = engineer.create_features(merged)
        featured['property_age'] = 2024 - featured['year_built']
        final = engineer.calculate_demand_target(featured)
        
        X, y, cols = engineer.prepare_ml_data(final)
        
        print("\nFeature columns:", cols)
        print("\nSample X:")
        print(X.head())
        print("\nTarget stats:")
        print(y.describe())
        
    except Exception as e:
        print(f"Test failed: {e}")
