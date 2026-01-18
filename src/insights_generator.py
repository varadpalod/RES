"""
Insights Generator Module
Creates actionable outputs from demand predictions and sentiment data
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
import json
# sys.path hacking removed
# from config import OUTPUT_DIR


class InsightsGenerator:
    """
    Generates actionable insights and reports from the prediction model
    """
    
    def __init__(self):
        # Import config locally
        try:
            from config import OUTPUT_DIR
            self.output_dir = OUTPUT_DIR
        except ImportError:
            self.output_dir = Path("outputs")
            
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_demand_scores(self, df: pd.DataFrame, 
                               predictions: np.ndarray) -> pd.DataFrame:
        """
        Attach demand predictions to property data with additional metrics
        
        Args:
            df: Property DataFrame
            predictions: Model predictions
            
        Returns:
            DataFrame with predicted demand scores and additional metrics
        """
        result = df.copy()
        result['predicted_demand'] = predictions.round(1)
        
        # Categorize demand
        result['demand_category'] = pd.cut(
            result['predicted_demand'],
            bins=[0, 40, 60, 80, 100],
            labels=['Low', 'Moderate', 'High', 'Very High']
        )
        
        # NEW: Liquidity Score (0-100)
        # How easily the property can be sold based on demand, location, and market factors
        result['liquidity_score'] = self._calculate_liquidity_score(result)
        
        # NEW: Infrastructure Risk Score (0-100, higher = more risk)
        # Based on negative sentiment about traffic, water, metro delays
        result['infrastructure_risk_score'] = self._calculate_infrastructure_risk(result)
        
        # NEW: Sale Probability (0-100%)
        # Chance the property will sell within 90 days
        result['sale_probability'] = self._calculate_sale_probability(result)
        
        # NEW: Expected Time to Sell (days)
        # Estimated days to close the deal
        result['expected_time_to_sell'] = self._calculate_time_to_sell(result)
        
        return result
    
    def _calculate_liquidity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate liquidity score (0-100)
        Higher score = easier to sell
        """
        liquidity = pd.Series(index=df.index, dtype=float)
        
        # Base liquidity from demand
        liquidity = df['predicted_demand'] * 0.4
        
        # Boost from rental yield (higher yield = more investor interest)
        if 'avg_rental_yield' in df.columns:
            yield_factor = (df['avg_rental_yield'] - 3) / 3  # Normalized around 3%
            liquidity += yield_factor.clip(-1, 1) * 15
        
        # Boost from connectivity
        if 'connectivity_score' in df.columns:
            liquidity += df['connectivity_score'] * 20
        
        # Boost from RERA (registered = easier to sell)
        if 'rera_registered' in df.columns:
            liquidity += df['rera_registered'] * 10
        
        # Penalty for old properties
        if 'property_age' in df.columns:
            age_penalty = (df['property_age'] / 30).clip(0, 1) * 15
            liquidity -= age_penalty
        
        # Positive sentiment boost
        if 'overall_sentiment' in df.columns:
            liquidity += (df['overall_sentiment'] + 1) / 2 * 15
        
        return liquidity.clip(0, 100).round(1)
    
    def _calculate_infrastructure_risk(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate infrastructure risk score (0-100)
        Higher score = more risk (traffic, water, connectivity issues)
        """
        risk = pd.Series(50, index=df.index, dtype=float)  # Base neutral risk
        
        # Risk from negative infrastructure sentiment
        if 'infrastructure_satisfaction' in df.columns:
            # Negative sentiment = higher risk
            risk -= df['infrastructure_satisfaction'] * 30
        
        # Risk from long commute times
        if 'avg_commute_time_min' in df.columns:
            commute_risk = (df['avg_commute_time_min'] / 60).clip(0, 1) * 20
            risk += commute_risk
        
        # Risk from distance to metro
        if 'metro_distance_km' in df.columns:
            metro_risk = (df['metro_distance_km'] / 10).clip(0, 1) * 15
            risk += metro_risk
        
        # Risk from locality issues
        if 'locality_issues_count' in df.columns:
            risk += df['locality_issues_count'] * 5
        
        return risk.clip(0, 100).round(1)
    
    def _calculate_sale_probability(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate probability of sale within 90 days (0-100%)
        """
        # Base probability from demand score
        prob = df['predicted_demand'] * 0.6
        
        # Boost from liquidity
        prob += df['liquidity_score'] * 0.3
        
        # Penalty from infrastructure risk
        prob -= df['infrastructure_risk_score'] * 0.1
        
        # Price factor - overpriced properties sell slower
        if 'price_premium' in df.columns:
            price_penalty = df['price_premium'].clip(-0.3, 0.3) * 30
            prob -= price_penalty
        
        return prob.clip(5, 95).round(1)  # Cap between 5-95%
    
    def _calculate_time_to_sell(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate expected days to sell
        """
        # Base time inversely related to sale probability
        base_days = 180 - (df['sale_probability'] * 1.5)
        
        # Adjust for demand category
        demand_adjustment = df['predicted_demand'].apply(
            lambda x: -30 if x >= 75 else (-15 if x >= 60 else (15 if x < 40 else 0))
        )
        
        days = base_days + demand_adjustment
        
        # Add some variance based on price tier
        if 'price' in df.columns:
            price_median = df['price'].median()
            price_factor = (df['price'] / price_median - 1).clip(-0.5, 0.5) * 20
            days += price_factor
        
        return days.clip(15, 365).round(0).astype(int)  # 15 days to 1 year
    
    def generate_locality_rankings(self, df: pd.DataFrame,
                                   sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank localities by investment potential
        
        Args:
            df: Property DataFrame with predictions
            sentiment_df: Locality sentiment profiles
            
        Returns:
            Ranked locality DataFrame
        """
        # Build aggregation dict dynamically based on available columns
        agg_dict = {
            'predicted_demand': 'mean',
            'price': 'median',
            'id': 'count'
        }
        
        if 'price_per_sqft' in df.columns:
            agg_dict['price_per_sqft'] = 'mean'
        if 'avg_rental_yield' in df.columns:
            agg_dict['avg_rental_yield'] = 'mean'
        if 'min_annual_income' in df.columns:
            agg_dict['min_annual_income'] = 'median'
        if 'rera_registered' in df.columns:
            agg_dict['rera_registered'] = 'mean'
            
        # Add builder-focused metrics
        if 'property_age' in df.columns:
            agg_dict['property_age'] = 'mean'
        if 'square_feet' in df.columns:
            agg_dict['square_feet'] = 'mean'
        if 'num_bedrooms' in df.columns:
            agg_dict['num_bedrooms'] = lambda x: x.mode()[0] if not x.mode().empty else 2
        
        locality_stats = df.groupby('area').agg(agg_dict).rename(columns={'id': 'property_count'})
        
        # Rename for clarity
        if 'rera_registered' in locality_stats.columns:
            locality_stats['rera_compliance_pct'] = (locality_stats['rera_registered'] * 100).round(1)
            locality_stats = locality_stats.drop('rera_registered', axis=1)
            
        # Rename new metrics
        rename_map = {}
        if 'property_age' in locality_stats.columns:
            rename_map['property_age'] = 'avg_property_age'
        if 'square_feet' in locality_stats.columns:
            rename_map['square_feet'] = 'avg_square_feet'
        if 'num_bedrooms' in locality_stats.columns:
            rename_map['num_bedrooms'] = 'most_common_config'
            
        if rename_map:
            locality_stats = locality_stats.rename(columns=rename_map)
        
        # Merge with sentiment
        locality_stats = locality_stats.reset_index()
        locality_stats = locality_stats.merge(
            sentiment_df[['locality', 'investment_confidence', 'overall_sentiment', 
                         'buying_urgency', 'sentiment_volatility']],
            left_on='area',
            right_on='locality',
            how='left'
        )
        
        # Calculate investment potential score (now including rental yield)
        base_score = (
            locality_stats['predicted_demand'] * 0.25 +
            (locality_stats['investment_confidence'].fillna(0) + 1) * 25 * 0.25 +
            (locality_stats['overall_sentiment'].fillna(0) + 1) * 25 * 0.15 +
            (1 - locality_stats['sentiment_volatility'].fillna(0.5)) * 50 * 0.15
        )
        
        # Add rental yield bonus
        if 'avg_rental_yield' in locality_stats.columns:
            yield_bonus = (locality_stats['avg_rental_yield'] - 3.5) * 5  # Bonus for above avg yield
            base_score += yield_bonus.clip(-10, 10) * 0.2
        
        locality_stats['investment_score'] = base_score.round(1)
        
        # Rank
        locality_stats = locality_stats.sort_values('investment_score', ascending=False)
        locality_stats['rank'] = range(1, len(locality_stats) + 1)
        
        # Investment recommendation
        locality_stats['recommendation'] = locality_stats['investment_score'].apply(
            lambda x: 'Strong Buy' if x >= 70 else ('Buy' if x >= 55 else ('Hold' if x >= 40 else 'Avoid'))
        )
        
        columns = ['rank', 'area', 'investment_score', 'recommendation', 
                   'predicted_demand', 'price', 'avg_rental_yield', 'min_annual_income',
                   'rera_compliance_pct', 'investment_confidence', 
                   'overall_sentiment', 'property_count']
        
        return locality_stats[[c for c in columns if c in locality_stats.columns]]
    
    def generate_affordability_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate affordability analysis by locality
        
        Args:
            df: Property DataFrame with affordability metrics
            
        Returns:
            DataFrame with affordability stats per locality
        """
        if 'min_annual_income' not in df.columns:
            print("⚠ Affordability data not available. Run enhance_property_data.py first.")
            return pd.DataFrame()
        
        # Aggregate affordability by locality
        affordability_stats = df.groupby('area').agg({
            'price': 'median',
            'min_annual_income': 'median',
            'monthly_emi': 'median',
            'down_payment': 'median',
            'avg_rental_yield': 'mean'
        }).round(0)
        
        affordability_stats = affordability_stats.reset_index()
        
        # Calculate rent vs EMI ratio
        if 'estimated_monthly_rent' in df.columns:
            rent_stats = df.groupby('area')['estimated_monthly_rent'].median()
            affordability_stats = affordability_stats.merge(
                rent_stats.reset_index(), on='area', how='left'
            )
            affordability_stats['rent_to_emi_ratio'] = (
                affordability_stats['estimated_monthly_rent'] / 
                affordability_stats['monthly_emi'] * 100
            ).round(1)
        
        # Income bracket distribution
        income_brackets = df.groupby(['area', 'income_bracket']).size().unstack(fill_value=0)
        income_brackets = income_brackets.div(income_brackets.sum(axis=1), axis=0) * 100
        
        # Determine primary target audience
        affordability_stats['target_audience'] = affordability_stats['min_annual_income'].apply(
            lambda x: 'Entry Level (<5L)' if x < 500000 else 
                     ('Mid Level (5-10L)' if x < 1000000 else
                      ('Senior Level (10-20L)' if x < 2000000 else
                       ('Executive (20-50L)' if x < 5000000 else 'Premium (50L+)')))
        )
        
        affordability_stats = affordability_stats.sort_values('min_annual_income')
        
        # Save to output
        affordability_stats.to_csv(self.output_dir / 'affordability_analysis.csv', index=False)
        print(f"✓ Saved affordability analysis to {self.output_dir / 'affordability_analysis.csv'}")
        
        return affordability_stats
    
    def generate_sentiment_alerts(self, sentiment_df: pd.DataFrame) -> List[Dict]:
        """
        Generate alerts for significant sentiment changes
        
        Args:
            sentiment_df: Locality sentiment profiles
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        
        for _, row in sentiment_df.iterrows():
            locality = row['locality']
            
            # Check for negative price perception
            if row.get('price_perception', 0) < -0.3:
                alerts.append({
                    'type': 'warning',
                    'locality': locality,
                    'category': 'Price Perception',
                    'message': f'{locality}: Negative price perception detected. Buyers consider properties overpriced.',
                    'score': row['price_perception']
                })
            
            # Check for infrastructure concerns
            if row.get('infrastructure_satisfaction', 0) < -0.2:
                alerts.append({
                    'type': 'warning',
                    'locality': locality,
                    'category': 'Infrastructure',
                    'message': f'{locality}: Infrastructure satisfaction is low. Traffic or connectivity issues mentioned.',
                    'score': row['infrastructure_satisfaction']
                })
            
            # Check for high investment confidence
            if row.get('investment_confidence', 0) > 0.5:
                alerts.append({
                    'type': 'opportunity',
                    'locality': locality,
                    'category': 'Investment',
                    'message': f'{locality}: High investment confidence. Positive sentiment about growth potential.',
                    'score': row['investment_confidence']
                })
            
            # Check for high volatility
            if row.get('sentiment_volatility', 0) > 0.4:
                alerts.append({
                    'type': 'caution',
                    'locality': locality,
                    'category': 'Volatility',
                    'message': f'{locality}: High sentiment volatility. Market opinions are divided.',
                    'score': row['sentiment_volatility']
                })
        
        return sorted(alerts, key=lambda x: abs(x['score']), reverse=True)
    
    def generate_builder_recommendations(self, rankings: pd.DataFrame) -> List[Dict]:
        """
        Generate detailed recommendations for builders planning new projects
        
        Args:
            rankings: Locality rankings DataFrame
            
        Returns:
            List of detailed recommendations with text insights
        """
        recommendations = []
        
        for _, row in rankings.head(10).iterrows():
            locality = row['area']
            score = row['investment_score']
            rec = row['recommendation']
            
            # Generate detailed text insights
            insights = self._generate_locality_insights(row)
            
            # Generate action items (Rule Based Default)
            actions = self._generate_action_items(row, rec)
            
            # Generate target segment advice (Rule Based Default)
            target_advice = self._generate_target_advice(row)
            
            # Generate pricing strategy (Rule Based Default)
            pricing_strategy = self._generate_pricing_strategy(row)
            
            # 🤖 HYBRID AI: Override with AI Strategy if available
            try:
                # Lazy import to avoid circular dependencies
                from src.llm_insights import LLMInsights
                llm = LLMInsights()
                
                if llm and llm.available:
                    # Prepare data for AI
                    loc_data = row.to_dict()
                    loc_data['overall_sentiment'] = row.get('overall_sentiment', 0)
                    
                    print(f"  [AI] Asking AI for {locality} strategy...")
                    ai_strategy = llm.generate_detailed_strategy(loc_data)
                    
                    if ai_strategy:
                        # Override defaults with AI insights
                        if ai_strategy.get('action_items'):
                            actions = ai_strategy.get('action_items')
                        if ai_strategy.get('target_segment'):
                            target_advice = ai_strategy.get('target_segment')
                        if ai_strategy.get('pricing_strategy'):
                            pricing_strategy = ai_strategy.get('pricing_strategy')
                        print(f"  [AI] Strategy generated for {locality}")
            except Exception as e:
                print(f"  [Warn] AI generation failed for {locality}: {e}")
            
            recommendations.append({
                'locality': locality,
                'investment_score': score,
                'recommendation': rec,
                'summary': insights['summary'],
                'strengths': insights['strengths'],
                'weaknesses': insights['weaknesses'],
                'action_items': actions,
                'target_segment': target_advice,
                'pricing_strategy': pricing_strategy,
                'avg_price': row.get('price', 'N/A'),
                'demand_score': row.get('predicted_demand', 'N/A'),
                'rental_yield': row.get('avg_rental_yield', 'N/A')
            })
        
        # Save detailed recommendations
        import json
        with open(self.output_dir / 'detailed_builder_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2, default=str)
        
        return recommendations
    
    def _generate_locality_insights(self, row: pd.Series) -> Dict:
        """Generate insights summary for a locality"""
        locality = row['area']
        score = row['investment_score']
        demand = row.get('predicted_demand', 50)
        sentiment = row.get('overall_sentiment', 0)
        rental_yield = row.get('avg_rental_yield', 4)
        rera_pct = row.get('rera_compliance_pct', 85)
        
        # Summary based on overall score
        if score >= 70:
            summary = f"{locality} is a prime location for new development with strong fundamentals."
        elif score >= 55:
            summary = f"{locality} shows good potential with some areas for improvement."
        elif score >= 40:
            summary = f"{locality} requires careful analysis before investment. Mixed signals present."
        else:
            summary = f"{locality} currently faces challenges. Consider alternative locations or timing."
        
        # Identify strengths
        strengths = []
        if sentiment > 0.3:
            strengths.append("Strong positive market sentiment")
        if rental_yield > 4.5:
            strengths.append(f"High rental yield ({rental_yield:.1f}%) attracts investors")
        if rera_pct > 90:
            strengths.append(f"Excellent RERA compliance ({rera_pct:.0f}%) builds trust")
        if demand > 60:
            strengths.append("Above-average buyer demand")
        if not strengths:
            strengths.append("Established locality with market presence")
        
        # Identify weaknesses
        weaknesses = []
        if sentiment < -0.1:
            weaknesses.append("Negative sentiment affecting buyer confidence")
        if rental_yield < 3.5:
            weaknesses.append(f"Below-average rental yield ({rental_yield:.1f}%)")
        if demand < 45:
            weaknesses.append("Lower than expected buyer interest")
        if not weaknesses:
            weaknesses.append("No major concerns identified")
        
        return {
            'summary': summary,
            'strengths': strengths,
            'weaknesses': weaknesses
        }
    
    def _generate_action_items(self, row: pd.Series, rec: str) -> List[str]:
        """Generate actionable items for builders"""
        actions = []
        
        if rec == 'Strong Buy':
            actions.append("✅ Proceed with land acquisition and project planning")
            actions.append("✅ Target premium pricing given strong demand")
            actions.append("✅ Fast-track approvals to capitalize on positive sentiment")
        elif rec == 'Buy':
            actions.append("✅ Begin feasibility studies and market research")
            actions.append("⚠️ Consider competitive pricing to drive initial sales")
            actions.append("✅ Focus on differentiators (amenities, design)")
        elif rec == 'Hold':
            actions.append("⏸️ Delay major investment decisions")
            actions.append("📊 Monitor sentiment trends over next 2-3 months")
            actions.append("🔍 Conduct deeper market research before committing")
        else:  # Avoid
            actions.append("❌ Not recommended for new project launches")
            actions.append("🔄 Consider alternative localities with better metrics")
            actions.append("📈 Wait for infrastructure improvements or sentiment shift")
        
        return actions
    
    def _generate_target_advice(self, row: pd.Series) -> str:
        """Generate target segment advice"""
        min_income = row.get('min_annual_income', 1500000)
        rental_yield = row.get('avg_rental_yield', 4)
        
        if min_income < 800000:
            segment = "First-time homebuyers and young professionals"
            advice = "Focus on compact 1-2 BHK units with modern amenities. Emphasize affordability and EMI schemes."
        elif min_income < 1500000:
            segment = "Mid-level professionals and growing families"
            advice = "Offer 2-3 BHK configurations with family-friendly amenities. Highlight schools and connectivity."
        elif min_income < 3000000:
            segment = "Senior professionals and executives"  
            advice = "Premium 3-4 BHK units with luxury amenities. Focus on brand, quality, and exclusivity."
        else:
            segment = "HNIs and luxury buyers"
            advice = "Ultra-premium configurations. Emphasize investment value, privacy, and bespoke features."
        
        if rental_yield > 4.5:
            advice += f" High rental yield ({rental_yield:.1f}%) also attracts investors - consider investor-friendly payment plans."
        
        return f"**Target Segment**: {segment}\n\n{advice}"
    
    def _generate_pricing_strategy(self, row: pd.Series) -> str:
        """Generate pricing strategy recommendation"""
        demand = row.get('predicted_demand', 50)
        sentiment = row.get('overall_sentiment', 0)
        avg_price = row.get('price', 8000000)
        
        if demand >= 70 and sentiment > 0.3:
            strategy = "Premium Pricing"
            details = f"Strong demand and positive sentiment support pricing 5-10% above market average (₹{avg_price/100000:.0f}L)."
        elif demand >= 55:
            strategy = "Market Pricing"
            details = f"Price at market levels (around ₹{avg_price/100000:.0f}L) with attractive launch offers."
        elif demand >= 40:
            strategy = "Competitive Pricing"
            details = f"Price 5-8% below market (₹{avg_price*0.93/100000:.0f}L) to drive initial momentum."
        else:
            strategy = "Value Pricing"
            details = f"Aggressive pricing needed. Consider ₹{avg_price*0.88/100000:.0f}L with flexible payment schemes."
        
        return f"**Strategy**: {strategy}\n\n{details}"
    
    def generate_broker_priorities(self, df: pd.DataFrame, 
                                   top_n: int = 50) -> pd.DataFrame:
        """
        Generate priority listings for brokers based on demand
        
        Args:
            df: Property DataFrame with predictions
            top_n: Number of priority properties
            
        Returns:
            DataFrame with priority listings
        """
        # High demand, reasonable price properties
        priority = df.nlargest(top_n, 'predicted_demand')
        
        priority['priority_reason'] = priority.apply(
            lambda x: self._get_priority_reason(x), axis=1
        )
        
        columns = ['id', 'area', 'predicted_demand', 'demand_category',
                   'price', 'square_feet', 'num_bedrooms', 'priority_reason']
        
        return priority[[c for c in columns if c in priority.columns]]
    
    def _get_priority_reason(self, row: pd.Series) -> str:
        """Generate reason for priority listing"""
        reasons = []
        
        if row.get('predicted_demand', 0) > 75:
            reasons.append('Very high predicted demand')
        elif row.get('predicted_demand', 0) > 60:
            reasons.append('High predicted demand')
        
        if row.get('investment_confidence', 0) > 0.5:
            reasons.append('Strong investment sentiment')
        
        if row.get('buying_urgency', 0) > 0.7:
            reasons.append('High buying urgency in area')
        
        return '; '.join(reasons) if reasons else 'Moderate demand'
    
    def save_insights(self, rankings: pd.DataFrame, 
                      alerts: List[Dict],
                      recommendations: List[Dict]) -> None:
        """Save all insights to files"""
        
        # Save rankings
        rankings.to_csv(self.output_dir / 'locality_rankings.csv', index=False)
        
        # Save alerts
        with open(self.output_dir / 'sentiment_alerts.json', 'w') as f:
            json.dump(alerts, f, indent=2)
        
        # Save recommendations
        with open(self.output_dir / 'builder_recommendations.json', 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        # Generate summary report
        report = {
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_localities': len(rankings),
                'strong_buy_count': len(rankings[rankings['recommendation'] == 'Strong Buy']),
                'buy_count': len(rankings[rankings['recommendation'] == 'Buy']),
                'total_alerts': len(alerts),
                'opportunity_alerts': len([a for a in alerts if a['type'] == 'opportunity']),
                'warning_alerts': len([a for a in alerts if a['type'] == 'warning'])
            },
            'top_localities': rankings.head(5)[['area', 'investment_score', 'recommendation']].to_dict('records')
        }
        
        with open(self.output_dir / 'insights_summary.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Saved insights to {self.output_dir}")


if __name__ == "__main__":
    # Test with sample data
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'id': range(1, 101),
        'area': np.random.choice(['Hinjewadi', 'Baner', 'Koregaon Park', 'Wakad'], 100),
        'price': np.random.randint(4000000, 15000000, 100),
        'square_feet': np.random.randint(800, 2000, 100),
        'num_bedrooms': np.random.choice([2, 3, 4], 100),
        'predicted_demand': np.random.uniform(30, 90, 100),
        'investment_confidence': np.random.uniform(-0.5, 0.8, 100),
        'buying_urgency': np.random.uniform(0.3, 0.9, 100)
    })
    sample_df['price_per_sqft'] = sample_df['price'] / sample_df['square_feet']
    
    sentiment_df = pd.DataFrame({
        'locality': ['Hinjewadi', 'Baner', 'Koregaon Park', 'Wakad'],
        'price_perception': [-0.2, 0.3, 0.5, 0.1],
        'infrastructure_satisfaction': [0.4, 0.6, 0.7, -0.3],
        'investment_confidence': [0.7, 0.5, 0.4, 0.3],
        'buying_urgency': [0.6, 0.5, 0.4, 0.7],
        'overall_sentiment': [0.4, 0.5, 0.6, 0.2],
        'sentiment_volatility': [0.3, 0.2, 0.1, 0.5]
    })
    
    generator = InsightsGenerator()
    
    rankings = generator.generate_locality_rankings(sample_df, sentiment_df)
    print("\nLocality Rankings:")
    print(rankings)
    
    alerts = generator.generate_sentiment_alerts(sentiment_df)
    print(f"\nGenerated {len(alerts)} alerts")
    
    recommendations = generator.generate_builder_recommendations(rankings)
    print("\nBuilder Recommendations:")
    for rec in recommendations[:3]:
        print(f"  {rec['locality']}: {rec['recommendation']} - {rec['action']}")
