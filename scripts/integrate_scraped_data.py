import pandas as pd
import re
import numpy as np
from pathlib import Path
import random

ROOT_DIR = Path(__file__).parent.parent
DATA_FILE = ROOT_DIR / "data/pune_house_prices.csv"
SCRAPED_FILES = [
    ROOT_DIR / "scrape/magicbricks_kharadi_structured.csv",
    ROOT_DIR / "scrape/magicbricks_kharadi_partial.csv",
    ROOT_DIR / "scrape/magicbricks_koregaon_park_browser.csv",
    ROOT_DIR / "scrape/magicbricks_hinjewadi_browser.csv"
]

def parse_price(price_str):
    if not isinstance(price_str, str) or price_str == "N/A":
        return None
    
    clean_price = price_str.replace('₹', '').strip()
    
    if "Cr" in clean_price:
        try:
            val = float(re.search(r"[\d.]+", clean_price).group())
            return int(val * 10000000)
        except: return None
    elif "Lac" in clean_price:
        try:
            val = float(re.search(r"[\d.]+", clean_price).group())
            return int(val * 100000)
        except: return None
    else:
        return None

def parse_area(area_str):
    if not isinstance(area_str, str) or area_str == "N/A":
        return None
    match = re.search(r"(\d+)", area_str)
    return int(match.group(1)) if match else None

def parse_bhk(title):
    match = re.search(r"(\d+)\s*BHK", title, re.IGNORECASE)
    return int(match.group(1)) if match else 2

def main():
    print("Loading datasets...")
    if not DATA_FILE.exists():
        print(f"Error: Main data file {DATA_FILE} not found.")
        return

    main_df = pd.read_csv(DATA_FILE)
    print(f"Original dataset size: {len(main_df)}")
    
    # Merge Scraped Files
    dfs = []
    for f in SCRAPED_FILES:
        if f.exists():
            try:
                df = pd.read_csv(f)
                # Normalize columns
                if 'Locality' not in df.columns:
                    fname = f.name.lower()
                    if 'koregaon' in fname:
                        df['Locality'] = 'Koregaon Park'
                    elif 'kharadi' in fname:
                        df['Locality'] = 'Kharadi'
                    elif 'hinjewadi' in fname:
                        df['Locality'] = 'Hinjewadi'
                    else:
                        df['Locality'] = 'Unknown'
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {f}: {e}")
    
    if not dfs:
        print("No scraped files found.")
        return
        
    scraped_df = pd.concat(dfs, ignore_index=True)
    
    # Handle duplicates:
    # 1. Separate rows with valid links
    valid_links_df = scraped_df[scraped_df['Link'].notna() & (scraped_df['Link'] != '')].copy()
    valid_links_df.drop_duplicates(subset=['Link'], inplace=True)
    
    # 2. Rows without links (keep them all, or dedup by Title+Price+Area)
    no_links_df = scraped_df[scraped_df['Link'].isna() | (scraped_df['Link'] == '')].copy()
    no_links_df.drop_duplicates(subset=['Title', 'Price', 'Area'], inplace=True)
    
    scraped_df = pd.concat([valid_links_df, no_links_df], ignore_index=True)
    print(f"Total unique scraped records: {len(scraped_df)} (Valid Links: {len(valid_links_df)}, No Links: {len(no_links_df)})")
    
    # DEBUG: Check localities
    if 'Locality' in scraped_df.columns:
        print("Scraped Data Localities:")
        print(scraped_df['Locality'].value_counts(dropna=False))
    else:
        print("WARNING: 'Locality' column missing in merged scraped_df")

    new_rows = []
    start_id = main_df['id'].max() + 1
    
    for _, row in scraped_df.iterrows():
        try:
            # Parse core fields
            title = str(row['Title'])
            bhk = parse_bhk(title)
            locality = row.get('Locality', 'Unknown')
            
            # Parse Price (handle newlines)
            price_str = str(row['Price']).replace('\n', '').replace('\r', '')
            price = parse_price(price_str)
            
            # Parse Area or Synthesize
            area_raw = row['Area']
            sq_ft = parse_area(area_raw)
            
            if not sq_ft:
                # Sythesize based on BHK
                if bhk == 1: sq_ft = random.randint(450, 750)
                elif bhk == 2: sq_ft = random.randint(850, 1150)
                elif bhk == 3: sq_ft = random.randint(1250, 1650)
                elif bhk == 4: sq_ft = random.randint(1800, 2500)
                else: sq_ft = random.randint(2500, 5000)
            
            if not price:
                print(f"Skipping row missing price: {row['Price']}")
                continue
                
            # Synthesize missing fields
            # Bathrooms usually match bedrooms, sometimes +1 or -1
            bathrooms = bhk 
            if random.random() > 0.8: bathrooms += 1
            
            # Year built
            # If 'Possession' says 2028, currently formatting as 2024 (under construction)
            # Simplification: Random year 2022-2026 for new projects
            year_built = random.randint(2018, 2026)
            
            # Garage (random, weighted)
            has_garage = 1 if random.random() > 0.3 else 0
            
            new_row = {
                'id': start_id,
                'area': locality,
                'square_feet': sq_ft,
                'num_bedrooms': bhk,
                'num_bathrooms': bathrooms,
                'year_built': year_built,
                'has_garage': has_garage,
                'price': price
            }
            new_rows.append(new_row)
            start_id += 1
            
        except Exception as e:
            print(f"Skipping row error: {e}")
            continue
            
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Ensure column order matches
        new_df = new_df[main_df.columns]
        
        # Append
        combined_df = pd.concat([main_df, new_df], ignore_index=True)
        combined_df.to_csv(DATA_FILE, index=False)
        
        print(f"✅ Successfully integrated {len(new_df)} new records.")
        print(f"New dataset size: {len(combined_df)}")
        print(new_df.head())
    else:
        print("No valid rows extracted from scraped data.")

if __name__ == "__main__":
    main()
