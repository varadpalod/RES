"""
Consolidate Scraped Data from 3 Localities
Merges Koregaon Park, Hinjewadi, and Kharadi data into unified dataset
"""
import pandas as pd
import re
import random
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SCRAPED_FILES = {
    'Koregaon Park': ROOT_DIR / "scrape/magicbricks_koregaon_park_browser.csv",
    'Hinjewadi': ROOT_DIR / "scrape/magicbricks_hinjewadi_browser.csv",
    'Kharadi': ROOT_DIR / "scrape/magicbricks_kharadi_partial.csv"
}
OUTPUT_FILE = ROOT_DIR / "data/unified_scraped_dataset.csv"

def parse_price(price_str):
    """Parse price string like '₹ 1.5 Cr' or '₹ 85 Lac' to integer"""
    if not isinstance(price_str, str) or price_str == "N/A":
        return None
    
    clean_price = price_str.replace('₹', '').replace('\n', '').replace('\r', '').strip()
    
    if "Cr" in clean_price:
        try:
            val = float(re.search(r"[\d.]+", clean_price).group())
            return int(val * 10000000)  # Crores to rupees
        except:
            return None
    elif "Lac" in clean_price:
        try:
            val = float(re.search(r"[\d.]+", clean_price).group())
            return int(val * 100000)  # Lakhs to rupees
        except:
            return None
    return None

def parse_area(area_str):
    """Extract square feet from area string"""
    if not isinstance(area_str, str) or area_str == "N/A":
        return None
    match = re.search(r"(\d+)", area_str)
    return int(match.group(1)) if match else None

def parse_bhk(title):
    """Extract number of bedrooms from title"""
    if not isinstance(title, str):
        return 2
    match = re.search(r"(\d+)\s*BHK", title, re.IGNORECASE)
    return int(match.group(1)) if match else 2

def synthesize_area(bhk):
    """Generate realistic square feet based on BHK"""
    if bhk == 1:
        return random.randint(450, 750)
    elif bhk == 2:
        return random.randint(850, 1150)
    elif bhk == 3:
        return random.randint(1250, 1650)
    elif bhk == 4:
        return random.randint(1800, 2500)
    else:
        return random.randint(2500, 5000)

def main():
    print("=" * 60)
    print("CONSOLIDATING SCRAPED DATA")
    print("=" * 60)
    
    all_rows = []
    locality_counts = {}
    
    for locality, file_path in SCRAPED_FILES.items():
        if not file_path.exists():
            print(f"⚠ Warning: {file_path.name} not found, skipping {locality}")
            continue
        
        print(f"\nProcessing {locality}...")
        df = pd.read_csv(file_path)
        print(f"  Raw rows: {len(df)}")
        
        # Remove duplicates by Link
        df_dedup = df.dropna(subset=['Link']).drop_duplicates(subset=['Link'])
        no_link_df = df[df['Link'].isna()].drop_duplicates(subset=['Title', 'Price', 'Area'])
        df = pd.concat([df_dedup, no_link_df], ignore_index=True)
        print(f"  After deduplication: {len(df)}")
        
        valid_rows = 0
        for _, row in df.iterrows():
            try:
                title = str(row['Title'])
                bhk = parse_bhk(title)
                
                # Parse price
                price = parse_price(row['Price'])
                if not price:
                    continue
                
                # Parse or synthesize area
                sq_ft = parse_area(row['Area'])
                if not sq_ft:
                    sq_ft = synthesize_area(bhk)
                
                # Synthesize other fields
                bathrooms = bhk
                if random.random() > 0.8:
                    bathrooms += 1
                
                year_built = random.randint(2018, 2026)
                has_garage = 1 if random.random() > 0.3 else 0
                
                all_rows.append({
                    'area': locality,
                    'square_feet': sq_ft,
                    'num_bedrooms': bhk,
                    'num_bathrooms': bathrooms,
                    'year_built': year_built,
                    'has_garage': has_garage,
                    'price': price
                })
                valid_rows += 1
                
            except Exception as e:
                continue
        
        locality_counts[locality] = valid_rows
        print(f"  Valid records: {valid_rows}")
    
    if not all_rows:
        print("\n❌ ERROR: No valid rows extracted from any file!")
        return
    
    # Create unified dataframe
    unified_df = pd.DataFrame(all_rows)
    unified_df['id'] = range(1, len(unified_df) + 1)
    
    # Reorder columns to match expected schema
    unified_df = unified_df[['id', 'area', 'square_feet', 'num_bedrooms', 
                             'num_bathrooms', 'year_built', 'has_garage', 'price']]
    
    # Save
    unified_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n" + "=" * 60)
    print("CONSOLIDATION COMPLETE")
    print("=" * 60)
    print(f"\n✅ Created unified dataset: {OUTPUT_FILE.name}")
    print(f"Total records: {len(unified_df)}")
    print(f"\nBreakdown by locality:")
    for locality, count in locality_counts.items():
        print(f"  {locality}: {count} properties")
    
    print("\nSample data:")
    print(unified_df.head())
    
    print("\nData quality check:")
    print(f"  Null values: {unified_df.isnull().sum().sum()}")
    print(f"  Price range: ₹{unified_df['price'].min():,.0f} - ₹{unified_df['price'].max():,.0f}")
    print(f"  Avg square feet: {unified_df['square_feet'].mean():.0f}")
    print(f"  BHK distribution:\n{unified_df['num_bedrooms'].value_counts().sort_index()}")

if __name__ == "__main__":
    main()
