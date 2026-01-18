"""
Locality Aggregator Module
Consolidates sentiment scores at the locality level
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PUNE_LOCALITIES, LOCALITY_SENTIMENT_FILE


class LocalityAggregator:
    """
    Aggregates sentiment scores by locality to create market profiles
    """
    
    def __init__(self, localities: List[str] = None):
        self.localities = localities or PUNE_LOCALITIES
        self.sentiment_columns = [
            'price_perception',
            'infrastructure_satisfaction', 
            'investment_confidence',
            'buying_urgency',
            'overall_sentiment'
        ]
    
    def aggregate_from_corpus(self, corpus: List[Dict]) -> pd.DataFrame:
        """
        Aggregate sentiment scores from analyzed corpus
        
        Args:
            corpus: Analyzed corpus with sentiment_scores field
            
        Returns:
            DataFrame with locality-level sentiment profiles
        """
        records = []
        
        for item in corpus:
            locality = item.get('locality', '')
            scores = item.get('sentiment_scores', {})
            date = item.get('date', datetime.now().strftime('%Y-%m-%d'))
            
            if locality and scores:
                record = {
                    'locality': locality,
                    'date': date,
                    **scores
                }
                records.append(record)
        
        if not records:
            print("⚠ No valid records to aggregate")
            return self._create_empty_profile()
        
        df = pd.DataFrame(records)
        return self._aggregate_dataframe(df)
    
    def _aggregate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform aggregation on sentiment dataframe
        
        Args:
            df: DataFrame with locality, date, and sentiment columns
            
        Returns:
            Aggregated locality profiles
        """
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate recency weight (more recent = higher weight)
        max_date = df['date'].max()
        df['days_ago'] = (max_date - df['date']).dt.days
        df['recency_weight'] = np.exp(-df['days_ago'] / 30)  # 30-day decay
        
        # Aggregate by locality
        aggregated = []
        
        for locality in df['locality'].unique():
            loc_df = df[df['locality'] == locality]
            
            profile = {'locality': locality}
            
            # Weighted averages for sentiment scores
            for col in self.sentiment_columns:
                if col in loc_df.columns:
                    weights = loc_df['recency_weight']
                    values = loc_df[col]
                    
                    # Weighted mean
                    weighted_mean = np.average(values, weights=weights)
                    profile[col] = round(weighted_mean, 3)
                    
                    # Standard deviation (volatility)
                    profile[f'{col}_std'] = round(values.std(), 3)
            
            # Calculate overall volatility
            std_cols = [c for c in profile.keys() if c.endswith('_std')]
            if std_cols:
                profile['sentiment_volatility'] = round(
                    np.mean([profile[c] for c in std_cols]), 3
                )
            
            # Count and date stats
            profile['sample_count'] = len(loc_df)
            profile['latest_date'] = loc_df['date'].max().strftime('%Y-%m-%d')
            profile['oldest_date'] = loc_df['date'].min().strftime('%Y-%m-%d')
            
            aggregated.append(profile)
        
        result = pd.DataFrame(aggregated)
        
        # Fill missing localities with neutral scores
        result = self._fill_missing_localities(result)
        
        # Sort by overall sentiment
        result = result.sort_values('overall_sentiment', ascending=False)
        
        print(f"✓ Aggregated sentiment for {len(result)} localities")
        
        return result
    
    def _fill_missing_localities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill in localities that have no sentiment data"""
        
        existing = set(df['locality'].unique())
        missing = [loc for loc in self.localities if loc not in existing]
        
        if missing:
            neutral_records = []
            for loc in missing:
                record = {
                    'locality': loc,
                    'price_perception': 0,
                    'infrastructure_satisfaction': 0,
                    'investment_confidence': 0,
                    'buying_urgency': 0.5,
                    'overall_sentiment': 0,
                    'sentiment_volatility': 0,
                    'sample_count': 0,
                    'latest_date': None,
                    'oldest_date': None
                }
                neutral_records.append(record)
            
            missing_df = pd.DataFrame(neutral_records)
            df = pd.concat([df, missing_df], ignore_index=True)
            print(f"  Added {len(missing)} localities with neutral sentiment")
        
        return df
    
    def _create_empty_profile(self) -> pd.DataFrame:
        """Create empty profile DataFrame"""
        return pd.DataFrame(columns=[
            'locality'] + self.sentiment_columns + [
            'sentiment_volatility', 'sample_count'
        ])
    
    def calculate_sentiment_trends(self, corpus: List[Dict], 
                                   window_days: int = 7) -> pd.DataFrame:
        """
        Calculate sentiment trends over time windows
        
        Args:
            corpus: Analyzed corpus
            window_days: Size of rolling window
            
        Returns:
            DataFrame with trend data
        """
        records = []
        for item in corpus:
            locality = item.get('locality', '')
            scores = item.get('sentiment_scores', {})
            date = item.get('date')
            
            if locality and scores and date:
                records.append({
                    'locality': locality,
                    'date': pd.to_datetime(date),
                    **scores
                })
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        
        # Group by locality and date, then calculate rolling averages
        trends = []
        for locality in df['locality'].unique():
            loc_df = df[df['locality'] == locality].sort_values('date')
            
            for col in self.sentiment_columns:
                if col in loc_df.columns:
                    # Rolling average
                    rolling = loc_df[col].rolling(
                        window=min(window_days, len(loc_df)), 
                        min_periods=1
                    ).mean()
                    
                    # Trend direction
                    if len(rolling) >= 2:
                        trend = 'improving' if rolling.iloc[-1] > rolling.iloc[0] else 'declining'
                    else:
                        trend = 'stable'
                    
                    trends.append({
                        'locality': locality,
                        'metric': col,
                        'current_value': rolling.iloc[-1],
                        'trend': trend,
                        'change': rolling.iloc[-1] - rolling.iloc[0] if len(rolling) >= 2 else 0
                    })
        
        return pd.DataFrame(trends)
    
    def get_locality_ranking(self, profiles: pd.DataFrame, 
                            metric: str = 'investment_confidence') -> pd.DataFrame:
        """
        Rank localities by a specific metric
        
        Args:
            profiles: Aggregated locality profiles
            metric: Column to rank by
            
        Returns:
            Ranked DataFrame
        """
        if metric not in profiles.columns:
            raise ValueError(f"Metric '{metric}' not found in profiles")
        
        ranked = profiles.sort_values(metric, ascending=False).reset_index(drop=True)
        ranked['rank'] = range(1, len(ranked) + 1)
        
        return ranked[['rank', 'locality', metric, 'sample_count']]
    
    def save_profiles(self, profiles: pd.DataFrame, 
                     file_path: Optional[Path] = None) -> None:
        """Save locality profiles to CSV"""
        path = file_path or LOCALITY_SENTIMENT_FILE
        profiles.to_csv(path, index=False)
        print(f"✓ Saved locality profiles to {path}")


if __name__ == "__main__":
    # Test with sample data
    sample_corpus = [
        {
            'locality': 'Hinjewadi',
            'date': '2024-01-15',
            'sentiment_scores': {
                'price_perception': -0.3,
                'infrastructure_satisfaction': 0.4,
                'investment_confidence': 0.6,
                'buying_urgency': 0.7,
                'overall_sentiment': 0.3,
                'confidence': 0.8
            }
        },
        {
            'locality': 'Hinjewadi',
            'date': '2024-01-10',
            'sentiment_scores': {
                'price_perception': -0.2,
                'infrastructure_satisfaction': 0.5,
                'investment_confidence': 0.7,
                'buying_urgency': 0.6,
                'overall_sentiment': 0.4,
                'confidence': 0.7
            }
        },
        {
            'locality': 'Koregaon Park',
            'date': '2024-01-14',
            'sentiment_scores': {
                'price_perception': 0.5,
                'infrastructure_satisfaction': 0.8,
                'investment_confidence': 0.6,
                'buying_urgency': 0.4,
                'overall_sentiment': 0.6,
                'confidence': 0.9
            }
        }
    ]
    
    aggregator = LocalityAggregator()
    profiles = aggregator.aggregate_from_corpus(sample_corpus)
    
    print("\nLocality Sentiment Profiles:")
    print(profiles[['locality', 'overall_sentiment', 'investment_confidence', 'sample_count']])
    
    print("\nRanking by Investment Confidence:")
    ranking = aggregator.get_locality_ranking(profiles, 'investment_confidence')
    print(ranking.head(10))
