import pandas as pd
import shutil
from pathlib import Path

DATA_FILE = Path("data/pune_house_prices.csv")
BACKUP_FILE = Path("data/pune_house_prices_backup.csv")

def main():
    if not DATA_FILE.exists():
        print(f"Error: {DATA_FILE} not found.")
        return

    # Backup if not exists
    if not BACKUP_FILE.exists():
        print(f"Creating backup at {BACKUP_FILE}")
        shutil.copy(DATA_FILE, BACKUP_FILE)
    else:
        print(f"Backup already exists at {BACKUP_FILE}")

    # Load original (or backup to be safe we are stripping from original source?) 
    # Actually, let's load from the current file, remove KP, so we don't accidentally keep re-adding if we run multiple times.
    # But wait, integrate script appends to this file. So this file might ALREADY contain scraped data if I just ran integrate.
    # The user wants to remove "Kaggle dataset" entries. The best way is to strict reset: 
    # 1. Load Backup (original Kaggle). 
    # 2. Filter out KP. 
    # 3. Save as current.
    # 4. Then let integrate script add the scraped ones.
    
    if BACKUP_FILE.exists():
        print("Loading from backup to ensure clean slate...")
        df = pd.read_csv(BACKUP_FILE)
    else:
        print("Loading from current file...")
        df = pd.read_csv(DATA_FILE)

    print(f"Original size: {len(df)}")
    
    # Filter
    # Check column name. Usually 'area' or 'Locality' -> view_file showed 'area' in previous turns, but let's be robust.
    col = 'area' if 'area' in df.columns else 'Locality'
    
    # Filter out Koregaon Park AND Hinjewadi (synthetic data)
    print("Filtering out synthetic 'Koregaon Park' and 'Hinjewadi' data...")
    koregaon_clean_df = df[~df['area'].astype(str).str.contains('Koregaon', case=False)]
    
    # Filter Hinjewadi from the previously filtered dataframe
    clean_df = koregaon_clean_df[~koregaon_clean_df['area'].astype(str).str.contains('Hinjewadi', case=False)]
    
    print(f"Filtered dataset size: {len(clean_df)}")
    print(f"Removed {len(df) - len(clean_df)} synthetic records.")
    
    clean_df.to_csv(DATA_FILE, index=False)
    print(f"✅ Saved cleaned dataset to {DATA_FILE}")

if __name__ == "__main__":
    main()
