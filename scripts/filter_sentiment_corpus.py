"""
Regenerate Sentiment Corpus for 3 Scraped Localities Only
Filters existing sentiment corpus to keep only Koregaon Park, Hinjewadi, Kharadi
"""
import json
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SENTIMENT_FILE = ROOT_DIR / "data/sentiment_corpus.json"

KEEP_LOCALITIES = ["Koregaon Park", "Hinjewadi", "Kharadi"]

def main():
    print("Filtering sentiment corpus to 3 localities...")
    
    with open(SENTIMENT_FILE, 'r', encoding='utf-8') as f:
        corpus = json.load(f)
    
    print(f"Original corpus size: {len(corpus)}")
    
    # Filter to only keep our 3 localities
    filtered_corpus = [
        record for record in corpus
        if record.get('locality') in KEEP_LOCALITIES
    ]
    
    print(f"Filtered corpus size: {len(filtered_corpus)}")
    print(f"\nBreakdown by locality:")
    for loc in KEEP_LOCALITIES:
        count = sum(1 for r in filtered_corpus if r.get('locality') == loc)
        print(f"  {loc}: {count} records")
    
    # Save filtered corpus
    with open(SENTIMENT_FILE, 'w', encoding='utf-8') as f:
        json.dump(filtered_corpus, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Sentiment corpus filtered and saved to {SENTIMENT_FILE.name}")

if __name__ == "__main__":
    main()
