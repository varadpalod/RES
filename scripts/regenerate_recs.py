import sys
from pathlib import Path
import pandas as pd
import io

# Fix encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Generating Hybrid AI Recommendations...")

try:
    # Add root to path
    ROOT_DIR = Path(__file__).parent.parent
    sys.path.insert(0, str(ROOT_DIR))
    
    from src.insights_generator import InsightsGenerator
    
    # Load rankings
    rankings_file = ROOT_DIR / "outputs/locality_rankings.csv"
    if not rankings_file.exists():
        print(f"Error: {rankings_file} not found")
        sys.exit(1)
        
    rankings_df = pd.read_csv(rankings_file)
    print(f"Loaded rankings for {len(rankings_df)} localities")
    
    # Generate recommendations
    gen = InsightsGenerator()
    recs = gen.generate_builder_recommendations(rankings_df)
    
    print(f"Successfully generated {len(recs)} recommendations")
    print("Saved to outputs/detailed_builder_recommendations.json")
    
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
