"""
Create Sample Dataset
Generates a sample Pune housing dataset for testing if Kaggle download is unavailable
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PUNE_LOCALITIES


def generate_sample_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """
    Generate a synthetic sample dataset matching Kaggle schema
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame matching Kaggle dataset structure
    """
    np.random.seed(42)
    
    # Locality-specific parameters (realistic for Pune)
    locality_params = {
        "Koregaon Park": {"price_base": 15000, "price_range": 8000, "sqft_base": 1200},
        "Hinjewadi": {"price_base": 7500, "price_range": 4000, "sqft_base": 1100},
        "Pimpri-Chinchwad": {"price_base": 6500, "price_range": 3500, "sqft_base": 1000},
        "Viman Nagar": {"price_base": 10000, "price_range": 5000, "sqft_base": 1150},
        "Kalyani Nagar": {"price_base": 12000, "price_range": 6000, "sqft_base": 1200},
        "Baner": {"price_base": 9500, "price_range": 5000, "sqft_base": 1100},
        "Wakad": {"price_base": 7000, "price_range": 3500, "sqft_base": 1050},
        "Kharadi": {"price_base": 8500, "price_range": 4500, "sqft_base": 1100},
        "Hadapsar": {"price_base": 6000, "price_range": 3000, "sqft_base": 950},
        "Magarpatta": {"price_base": 9000, "price_range": 4000, "sqft_base": 1100},
        "Aundh": {"price_base": 10500, "price_range": 5500, "sqft_base": 1150},
        "Shivaji Nagar": {"price_base": 11000, "price_range": 6000, "sqft_base": 1100},
        "Deccan": {"price_base": 11500, "price_range": 6500, "sqft_base": 1050},
        "Kothrud": {"price_base": 9000, "price_range": 4500, "sqft_base": 1050},
        "Warje": {"price_base": 7500, "price_range": 4000, "sqft_base": 1000},
        "Undri": {"price_base": 5500, "price_range": 2500, "sqft_base": 950},
        "Kondhwa": {"price_base": 6500, "price_range": 3500, "sqft_base": 1000},
        "NIBM Road": {"price_base": 7000, "price_range": 3500, "sqft_base": 1050},
        "Bavdhan": {"price_base": 8000, "price_range": 4000, "sqft_base": 1100},
        "Pashan": {"price_base": 8500, "price_range": 4500, "sqft_base": 1100}
    }
    
    # Default params for any missing localities
    default_params = {"price_base": 7500, "price_range": 4000, "sqft_base": 1050}
    
    # Generate data
    data = []
    
    for i in range(n_samples):
        # Select locality with weighted distribution
        area = np.random.choice(PUNE_LOCALITIES)
        params = locality_params.get(area, default_params)
        
        # Generate property attributes
        num_bedrooms = np.random.choice([1, 2, 3, 4], p=[0.1, 0.45, 0.35, 0.1])
        num_bathrooms = min(num_bedrooms, np.random.choice([1, 2, 3], p=[0.2, 0.6, 0.2]))
        
        # Square feet based on bedrooms
        sqft_multiplier = {1: 0.6, 2: 0.85, 3: 1.2, 4: 1.6}
        base_sqft = params['sqft_base'] * sqft_multiplier[num_bedrooms]
        square_feet = int(np.random.normal(base_sqft, base_sqft * 0.15))
        square_feet = max(400, min(5000, square_feet))
        
        # Year built
        year_built = np.random.randint(1990, 2024)
        
        # Garage (more likely in newer, larger properties)
        garage_prob = 0.3 + (0.02 * (year_built - 1990)) + (0.05 * num_bedrooms)
        has_garage = 1 if np.random.random() < garage_prob else 0
        
        # Price calculation
        price_per_sqft = params['price_base'] + np.random.uniform(-params['price_range']/2, params['price_range']/2)
        
        # Adjustments
        if year_built >= 2020:
            price_per_sqft *= 1.15
        elif year_built < 2000:
            price_per_sqft *= 0.85
        
        if has_garage:
            price_per_sqft *= 1.08
        
        price = int(square_feet * price_per_sqft)
        
        data.append({
            'id': i + 1,
            'area': area,
            'square_feet': square_feet,
            'num_bedrooms': num_bedrooms,
            'num_bathrooms': num_bathrooms,
            'year_built': year_built,
            'has_garage': has_garage,
            'price': price
        })
    
    df = pd.DataFrame(data)
    
    print(f"✓ Generated {len(df):,} sample properties")
    print(f"  Localities: {df['area'].nunique()}")
    print(f"  Price range: ₹{df['price'].min():,} - ₹{df['price'].max():,}")
    print(f"  Avg price: ₹{df['price'].mean():,.0f}")
    
    return df


def save_sample_dataset(df: pd.DataFrame) -> None:
    """Save dataset to data directory"""
    DATA_DIR.mkdir(exist_ok=True)
    file_path = DATA_DIR / 'pune_house_prices.csv'
    df.to_csv(file_path, index=False)
    print(f"✓ Saved to {file_path}")


if __name__ == "__main__":
    print("Generating sample Pune housing dataset...")
    print("(This is for testing - replace with Kaggle dataset for production)")
    print()
    
    df = generate_sample_dataset(10000)
    save_sample_dataset(df)
    
    print("\nSample records:")
    print(df.head())
