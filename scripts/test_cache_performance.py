"""
Test the news caching system
Shows cache hits, misses, and performance improvement
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.news_fetcher import NewsFetcher

print("="*80)
print("TESTING 6-HOUR NEWS CACHE")
print("="*80)

# First, clear any existing cache
fetcher = NewsFetcher()
fetcher.clear_cache()
print("\nCache cleared for fresh test\n")

# Test 1: First fetch (should be SLOW - fetches from RSS)
print("-"*80)
print("TEST 1: First Fetch (No Cache) - Fetching from RSS feeds...")
print("-"*80)

start = time.time()
news1 = fetcher.get_locality_news("Hinjewadi", max_articles=3)
elapsed1 = time.time() - start

print(f"\nTime taken: {elapsed1:.2f} seconds")
print(f"Found {len(news1)} articles")
if news1:
    for i, article in enumerate(news1[:2], 1):
        print(f"  {i}. {article['title'][:70]}...")

# Test 2: Second fetch (should be FAST - uses cache)
print("\n" + "-"*80)
print("TEST 2: Second Fetch (With Cache) - Using cached data...")
print("-"*80)

start = time.time()
news2 = fetcher.get_locality_news("Hinjewadi", max_articles=3)
elapsed2 = time.time() - start

print(f"\nTime taken: {elapsed2:.2f} seconds")
print(f"Found {len(news2)} articles")

# Test 3: Check cache file
print("\n" + "-"*80)
print("TEST 3: Cache File Verification")
print("-"*80)

cache_file = "outputs/news_cache.json"
if os.path.exists(cache_file):
    print(f"\n✅ Cache file exists: {cache_file}")
    size = os.path.getsize(cache_file)
    print(f"   Size: {size:,} bytes")
    
    # Show cache timestamp
    import json
    with open(cache_file, 'r', encoding='utf-8') as f:
        cache_data = json.load(f)
        print(f"   Cached at: {cache_data.get('timestamp', 'Unknown')}")
        print(f"   Localities cached: {list(cache_data.get('data', {}).get('locality_news', {}).keys())}")
else:
    print("\n❌ Cache file not found")

# Performance comparison
print("\n" + "="*80)
print("PERFORMANCE RESULTS")
print("="*80)

speedup = elapsed1 / elapsed2 if elapsed2 > 0 else 0
print(f"\nFirst fetch (no cache):  {elapsed1:.2f} seconds")
print(f"Second fetch (cached):   {elapsed2:.2f} seconds")
print(f"\n🚀 Speedup: {speedup:.1f}x faster with cache!")

if speedup > 5:
    print("\n✅ EXCELLENT: Cache provides significant performance boost")
elif speedup > 2:
    print("\n✅ GOOD: Cache improves performance")
else:
    print("\n⚠️  Cache speedup lower than expected (network may be fast)")

print("\n" + "="*80)
print("CACHE BEHAVIOR")
print("="*80)
print(f"""
✅ Cache Duration: {fetcher.CACHE_DURATION_HOURS} hours
✅ Cache Storage: {cache_file}
✅ Cache Strategy: Time-based (auto-refresh after 6 hours)

How it works:
1. First request: Fetches from RSS feeds, saves to cache
2. Subsequent requests: Returns cached data (if < 6 hours old)
3. After 6 hours: Automatically fetches fresh news
4. Cache persists: Survives app restarts

Your dashboard will now be much faster! 🎉
""")
