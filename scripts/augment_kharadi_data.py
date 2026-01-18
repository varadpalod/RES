import pandas as pd
import numpy as np
from pathlib import Path
import random

ROOT_DIR = Path(__file__).parent.parent
SEED_FILE = ROOT_DIR / "scrape/magicbricks_kharadi_structured.csv"
MAIN_DATA_FILE = ROOT_DIR / "data/pune_house_prices.csv"

def parse_price(price_str):
    if not isinstance(price_str, str): return None
    clean = price_str.replace('₹', '').strip()
    try:
        if "Cr" in clean:
            val = float(clean.replace("Cr", "").strip())
            return int(val * 10000000)
        elif "Lac" in clean:
            val = float(clean.replace("Lac", "").strip())
            return int(val * 100000)
    except:
        return None
    return None

def main():
    print("Starting Synthetic Augmentation for Kharadi...")
    
    # 1. Load Seed Data
    if not SEED_FILE.exists():
        print(f"Error: Seed file {SEED_FILE} not found.")
        return
        
    seed_df = pd.read_csv(SEED_FILE)
    print(f"Loaded {len(seed_df)} seed records.")
    
    # 2. Extract Base Stats
    parsed_records = []
    
    for _, row in seed_df.iterrows():
        price = parse_price(str(row['Price']))
        if not price: continue
        
        # Parse BHK
        title = str(row['Title']).lower()
        bhk = 2 # Default
        if "1 bhk" in title: bhk = 1
        elif "2 bhk" in title: bhk = 2
        elif "3 bhk" in title: bhk = 3
        elif "4 bhk" in title: bhk = 4
            
        # Parse Area (approximate if missing)
        area = 0
        if bhk == 1: area = 650
        elif bhk == 2: area = 1000
        elif bhk == 3: area = 1400
        elif bhk == 4: area = 2200
            
        parsed_records.append({
            'price': price,
            'bhk': bhk,
            'area': area
        })
    
    if not parsed_records:
        print("No valid price data found in seed.")
        return
        
    df_seed = pd.DataFrame(parsed_records)
    
    # 3. Generate Clones
    TARGET_COUNT = 1000
    clones = []
    
    start_id = 200000 # High ID to avoid conflict
    if MAIN_DATA_FILE.exists():
        main_df = pd.read_csv(MAIN_DATA_FILE)
        start_id = main_df['id'].max() + 1
    
    print(f"Generating {TARGET_COUNT} clones based on {len(df_seed)} templates...")
    
    for i in range(TARGET_COUNT):
        # Pick a random template
        template = df_seed.sample(1).iloc[0]
        
        # Add noise (+/- 10%)
        price_noise = random.uniform(0.9, 1.1)
        area_noise = random.uniform(0.95, 1.05)
        
        new_price = int(template['price'] * price_noise)
        new_area = int(template['area'] * area_noise)
        
        # Derived fields
        bathrooms = template['bhk']
        if random.random() > 0.8: bathrooms += 1
        
        year_built = random.randint(2018, 2026)
        garage = 1 if random.random() > 0.4 else 0
        
        clones.append({
            'id': start_id,
            'area': 'Kharadi',
            'square_feet': new_area,
            'num_bedrooms': template['bhk'],
            'num_bathrooms': bathrooms,
            'year_built': year_built,
            'has_garage': garage,
            'price': new_price
        })
        start_id += 1
        
    # 4. Save
    new_df = pd.DataFrame(clones)
    
    if MAIN_DATA_FILE.exists():
        # Clean existing Kharadi synthetic data first? 
        # For now, just append. The user wanted volume.
        combined = pd.concat([main_df, new_df], ignore_index=True)
    else:
        combined = new_df
        
    combined.to_csv(MAIN_DATA_FILE, index=False)
    print(f"✅ Successfully added {len(new_df)} Kharadi records.")
    print(f"Total Dataset Size: {len(combined)}")

if __name__ == "__main__":
    main()
