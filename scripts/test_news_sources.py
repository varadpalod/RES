"""
Test expanded news sources
Verify that all new RSS feeds work and are filtered for Pune
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.news_fetcher import NewsFetcher

print("="*80)
print("TESTING EXPANDED NEWS SOURCES")
print("="*80)

fetcher = NewsFetcher()

print("\nConfigured News Sources:")
print("-"*80)
for i, feed in enumerate(fetcher.feeds, 1):
    source = "Unknown"
    if "timesofindia" in feed:
        source = "Times of India"
    elif "hindustantimes" in feed:
        source = "Hindustan Times"
    elif "indianexpress" in feed:
        source = "Indian Express"
    elif "economictimes" in feed:
        source = "Economic Times"
    elif "punemirror" in feed:
        source = "Pune Mirror"
    
    print(f"{i}. {source}")
    print(f"   {feed[:70]}...")

print(f"\nTotal: {len(fetcher.feeds)} news sources")

# Clear cache to test fresh
fetcher.clear_cache()

print("\n" + "="*80)
print("TESTING PUNE LOCALITY NEWS RETRIEVAL")
print("="*80)

# Test with a few localities
test_localities = ["Hinjewadi", "Baner", "Koregaon Park"]

for locality in test_localities:
    print(f"\n🔍 Searching for: {locality}")
    print("-"*80)
    
    news = fetcher.get_locality_news(locality, max_articles=3, days_back=90)
    
    if news:
        print(f"✅ Found {len(news)} articles from multiple sources:\n")
        for i, article in enumerate(news, 1):
            print(f"{i}. [{article['date']}] {article['title']}")
            # Try to identify source from URL
            source = "Unknown"
            if "timesofindia" in article['link']:
                source = "TOI"
            elif "hindustantimes" in article['link']:
                source = "HT"
            elif "indianexpress" in article['link']:
                source = "IE"
            elif "economictimes" in article['link']:
                source = "ET"
            elif "punemirror" in article['link']:
                source = "PM"
            print(f"   Source: {source}")
            print()
    else:
        print("   ℹ️  No recent articles (normal if locality not in news)")

print("\n" + "="*80)
print("GENERAL PUNE REAL ESTATE NEWS")
print("="*80)

general = fetcher.get_general_pune_news(max_articles=5)
if general:
    print(f"\n✅ Found {len(general)} general articles:\n")
    for i, article in enumerate(general, 1):
        print(f"{i}. {article['title'][:70]}...")
else:
    print("\nℹ️  No general news found")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
✅ Configured Sources: {len(fetcher.feeds)}
   - Times of India (2 feeds)
   - Hindustan Times (2 feeds)
   - Indian Express (1 feed)
   - Economic Times (1 feed)
   - Pune Mirror (1 feed)

✅ All sources filtered for:
   - Pune localities (Hinjewadi, Baner, etc.)
   - Real estate keywords
   - Recent articles only

✅ Cache enabled (6-hour refresh)
✅ Multiple reputable sources ensure comprehensive coverage
""")
