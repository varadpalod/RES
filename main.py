"""
Main Pipeline Orchestrator
End-to-end execution of the sentiment-aware real estate intelligence system
"""
import argparse
import sys
import io

# Fix encoding for Windows consoles
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, OUTPUT_DIR, MODEL_DIR,
    PROPERTY_DATA_FILE, SENTIMENT_CORPUS_FILE
)


def run_pipeline(generate_data: bool = False, 
                 train_model: bool = True,
                 test_mode: bool = False):
    """
    Run the complete pipeline
    
    Args:
        generate_data: Whether to generate synthetic sentiment data
        train_model: Whether to train the demand prediction model
        test_mode: Use small subset for testing
    """
    print("=" * 60)
    print("PUNE REAL ESTATE SENTIMENT INTELLIGENCE SYSTEM")
    print("=" * 60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Ensure directories exist
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    # Step 1: Load/Generate Sentiment Data
    print("\n[Step 1/7] Loading Sentiment Data...")
    print("-" * 40)
    
    from src.data_loader import load_sentiment_corpus
    
    corpus = load_sentiment_corpus()
    
    if not corpus or generate_data:
        print("Generating synthetic sentiment data...")
        from scripts.generate_sentiment_data import generate_corpus, save_corpus
        corpus = generate_corpus(500 if test_mode else 600)
        save_corpus(corpus)
    
    # Step 2: Load Property Data (with enhancements if available)
    print("\n[Step 2/7] Loading Property Data...")
    print("-" * 40)
    
    from src.data_loader import load_property_data, validate_data
    import pandas as pd
    
    # Check for enhanced data file first
    enhanced_file = DATA_DIR / 'pune_house_prices_enhanced.csv'
    
    try:
        if enhanced_file.exists():
            print("Loading enhanced property data with RERA, rental yield, and affordability...")
            property_df = pd.read_csv(enhanced_file)
            print(f"✓ Loaded {len(property_df):,} enhanced properties")
            print(f"  RERA registered: {property_df['rera_registered'].sum():,}")
            print(f"  Avg rental yield: {property_df['avg_rental_yield'].mean():.2f}%")
        else:
            property_df = load_property_data()
            print("  Note: Run 'python scripts/enhance_property_data.py' for RERA & rental yield features")
        
        if test_mode:
            property_df = property_df.sample(n=min(5000, len(property_df)), random_state=42)
            print(f"  Test mode: Using {len(property_df):,} samples")
        
        # Calculate property_age if not present
        if 'property_age' not in property_df.columns and 'year_built' in property_df.columns:
            property_df['property_age'] = 2024 - property_df['year_built']
        
        validate_data(property_df)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print(f"\nPlease run 'python scripts/consolidate_scraped_data.py' to prepare the dataset.")
        print(f"Scraped data should be in: {ROOT_DIR / 'scrape'}")
        return False
    
    # Step 3: Process and Analyze Sentiment
    print("\n[Step 3/7] Analyzing Sentiment...")
    print("-" * 40)
    
    from src.sentiment_analyzer import analyze_corpus
    from src.text_processor import TextProcessor
    
    processor = TextProcessor()
    
    # Clean and analyze corpus
    analyzed_corpus = analyze_corpus(corpus)
    
    # Step 4: Aggregate Sentiment by Locality
    print("\n[Step 4/7] Aggregating Locality Sentiment...")
    print("-" * 40)
    
    from src.locality_aggregator import LocalityAggregator
    
    aggregator = LocalityAggregator()
    sentiment_profiles = aggregator.aggregate_from_corpus(analyzed_corpus)
    aggregator.save_profiles(sentiment_profiles)
    
    print("\nTop 5 Localities by Sentiment:")
    print(sentiment_profiles[['locality', 'overall_sentiment', 'investment_confidence']].head())
    
    # Step 5: Feature Engineering
    print("\n[Step 5/7] Engineering Features...")
    print("-" * 40)
    
    from src.feature_engineer import FeatureEngineer
    
    engineer = FeatureEngineer()
    
    merged_df = engineer.merge_property_sentiment(property_df, sentiment_profiles)
    featured_df = engineer.create_features(merged_df)
    final_df = engineer.calculate_demand_target(featured_df)
    engineer.save_processed_data(final_df)
    
    # Step 6: Train Demand Model
    if train_model:
        print("\n[Step 6/7] Training Demand Model...")
        print("-" * 40)
        
        from src.demand_predictor import DemandPredictor
        
        X, y, feature_cols = engineer.prepare_ml_data(final_df)
        
        predictor = DemandPredictor()
        metrics = predictor.train(X, y)
        predictor.save_model()
        predictor.save_report()
        
        print("\nFeature Importance (Top 10):")
        print(predictor.get_feature_importance(10))
        
        # Make predictions on all data
        predictions = predictor.predict(X)
    else:
        print("\n[Step 6/7] Skipping Model Training...")
        print("-" * 40)
        predictions = final_df['demand_score'].values
    
    # Step 7: Generate Insights
    print("\n[Step 7/7] Generating Insights...")
    print("-" * 40)
    
    from src.insights_generator import InsightsGenerator
    
    generator = InsightsGenerator()
    
    # Add predictions to data
    result_df = generator.generate_demand_scores(final_df, predictions)
    
    # Generate outputs
    rankings = generator.generate_locality_rankings(result_df, sentiment_profiles)
    alerts = generator.generate_sentiment_alerts(sentiment_profiles)
    recommendations = generator.generate_builder_recommendations(rankings)
    priorities = generator.generate_broker_priorities(result_df)
    
    # Generate affordability analysis (new feature)
    affordability = generator.generate_affordability_analysis(result_df)
    
    # Save all outputs
    generator.save_insights(rankings, alerts, recommendations)
    priorities.to_csv(OUTPUT_DIR / 'broker_priorities.csv', index=False)
    result_df.to_csv(OUTPUT_DIR / 'full_predictions.csv', index=False)
    
    # Print Summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob("*"):
        print(f"  - {f.name}")
    
    print("\n📊 LOCALITY INVESTMENT RANKINGS (Top 5):")
    print(rankings[['rank', 'area', 'investment_score', 'recommendation']].head().to_string(index=False))
    
    print(f"\n⚠️  ALERTS: {len(alerts)} sentiment alerts generated")
    
    print("\n🏗️  BUILDER RECOMMENDATIONS:")
    for rec in recommendations[:3]:
        summary = rec.get('summary', rec.get('action_items', ['N/A'])[0] if rec.get('action_items') else 'N/A')
        print(f"  {rec['locality']}: {rec['recommendation']} - {summary[:60]}...")
    
    print("\n✅ Run 'streamlit run dashboard.py' to launch the interactive dashboard")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Pune Real Estate Sentiment Intelligence Pipeline'
    )
    parser.add_argument(
        '--generate-data', '-g',
        action='store_true',
        help='Generate fresh synthetic sentiment data'
    )
    parser.add_argument(
        '--no-train', '-n',
        action='store_true',
        help='Skip model training (use existing demand scores)'
    )
    parser.add_argument(
        '--test-mode', '-t',
        action='store_true',
        help='Run in test mode with reduced data'
    )
    parser.add_argument(
        '--stage',
        choices=['data-only', 'sentiment-only', 'full'],
        default='full',
        help='Pipeline stage to run'
    )
    
    args = parser.parse_args()
    
    success = run_pipeline(
        generate_data=args.generate_data,
        train_model=not args.no_train,
        test_mode=args.test_mode
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
