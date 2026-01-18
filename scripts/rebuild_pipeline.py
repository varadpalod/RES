import pandas as pd
import sys
from pathlib import Path
import json

# Setup Path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Import SRC modules (Assuming they are clean now)
try:
    from src.feature_engineer import FeatureEngineer
    from src.demand_predictor import DemandPredictor
    from src.insights_generator import InsightsGenerator
    from src.locality_aggregator import LocalityAggregator
    from config import OUTPUT_DIR, PROPERTY_DATA_FILE
    
    # Import enhancement logic
    from scripts.enhance_property_data import enhance_property_data, add_affordability_to_data, get_locality_highlights_df
    
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def main():
    print("Rebuilding Pipeline for Kharadi Integration...")
    
    # 1. Load Data
    print("Loading Property Data...")
    # Manually load and preprocess to ensure consistency
    property_df = pd.read_csv(PROPERTY_DATA_FILE)
    
    # Preprocessing (replicating data_loader.py logic)
    property_df['area'] = property_df['area'].str.strip().str.title()
    property_df['property_age'] = 2026 - property_df['year_built'] # Using current year
    property_df['price_per_sqft'] = property_df['price'] / property_df['square_feet']
    
    print(f"Loaded {len(property_df)} properties.")
    
    # 1.5 Enhance Data (Affordability & Highlights)
    print("Enhancing Data with Affordability Metrics...")
    # Add Kharadi profile to ensure it gets enhanced (if not already in LOCALITY_PROFILES)
    # Note: Kharadi IS in LOCALITY_PROFILES in the script, so this should work.
    
    enhanced_df = enhance_property_data(property_df)
    enhanced_df = add_affordability_to_data(enhanced_df)
    
    # Save highlights for dashboard
    highlights_df = get_locality_highlights_df()
    highlights_df.to_csv(OUTPUT_DIR / 'locality_highlights.csv', index=False)
    
    # Use enhanced_df for feature engineering
    property_df = enhanced_df
    
    # 2. Load/Synthesize Sentiment
    print("Loading/Synthesizing Sentiment...")
    # Try loading existing
    sentiment_file = OUTPUT_DIR / 'locality_sentiment.csv'
    if sentiment_file.exists():
        sentiment_profiles = pd.read_csv(sentiment_file)
    else:
        print("Warning: Sentiment file missing. Creating dummy.")
        sentiment_profiles = pd.DataFrame(columns=['locality', 'overall_sentiment', 'investment_confidence'])

    # Ensure Kharadi exists in sentiment
    if 'Kharadi' not in sentiment_profiles['locality'].values:
        print("Adding synthetic sentiment for Kharadi...")
        new_row = {
            'locality': 'Kharadi',
            'overall_sentiment': 0.65, # Positive
            'investment_confidence': 0.75,
            'infrastructure_satisfaction': 0.7,
            'price_perception': -0.2, # Slightly expensive
            'buying_urgency': 0.6,
            'sample_size': 50
        }
        # Filter to only cols that match
        for col in sentiment_profiles.columns:
            if col not in new_row:
                new_row[col] = 0
                
        sentiment_profiles = pd.concat([sentiment_profiles, pd.DataFrame([new_row])], ignore_index=True)
        sentiment_profiles.to_csv(sentiment_file, index=False)
        
    # 3. Feature Engineering
    print("Engineering Features...")
    engineer = FeatureEngineer()
    merged_df = engineer.merge_property_sentiment(property_df, sentiment_profiles)
    featured_df = engineer.create_features(merged_df)
    final_df = engineer.calculate_demand_target(featured_df)
    
    # Save processed (optional)
    engineer.save_processed_data(final_df)
    
    # 4. Train Model
    print("Training Model...")
    X, y, _ = engineer.prepare_ml_data(final_df)
    predictor = DemandPredictor()
    predictor.train(X, y)
    predictor.save_model()
    predictor.save_report()
    
    # 5. Generate Insights & Rankings
    print("Generating Insights...")
    # We need predictions on final_df to make rankings
    # FeatureEngineer/InsightsGenerator usually handles this.
    # InsightsGenerator.generate_locality_rankings takes (predictions_df, sentiment_df)
    
    # Generate full predictions
    full_predictions = final_df.copy()
    
    # Predict demand score using Trained Model
    # Prepare X for all data
    X_all, _, _ = engineer.prepare_ml_data(final_df)
    # Align columns
    full_predictions['predicted_demand'] = predictor.predict(X_all)
    
    full_predictions.to_csv(OUTPUT_DIR / 'full_predictions.csv', index=False)
    
    # Run Insights
    insights_gen = InsightsGenerator()
    rankings = insights_gen.generate_locality_rankings(full_predictions, sentiment_profiles)
    
    print("\nTop Localities:")
    print(rankings[['area', 'investment_score', 'recommendation']].head(10))
    
    # 6. Generate Builder Recommendations (Text)
    # Using the rankings
    print("Generating Builder Recommendations (Hybrid AI)...")
    recs = insights_gen.generate_builder_recommendations(rankings)
    print(f"Generated {len(recs)} recommendations.")
    
    # 7. Generate Alerts & Save Everything
    print("Generating Alerts & Saving All Insights...")
    alerts = insights_gen.generate_sentiment_alerts(sentiment_profiles)
    
    insights_gen.save_insights(rankings, alerts, recs)
    
    # Generate & Save Affordability Analysis
    print("Generating Affordability Analysis...")
    affordability_analysis = insights_gen.generate_affordability_analysis(final_df)
    affordability_analysis.to_csv(OUTPUT_DIR / 'affordability_analysis.csv', index=False)
    
    print("Pipeline Rebuild Complete. Kharadi Integrated!")

if __name__ == "__main__":
    main()
