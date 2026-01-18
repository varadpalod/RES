"""
Test script to demonstrate news-enhanced LLM insights
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_insights import LLMInsights

# Sample locality data (similar to what your dashboard uses)
sample_data = {
    'area': 'Hinjewadi',
    'price': 6500000,
    'avg_rental_yield': 4.5,
    'predicted_demand': 72.5,
    'investment_score': 68.0,
    'overall_sentiment': 0.45,
    'rera_compliance_pct': 92,
    'min_annual_income': 1200000,
    'it_park_distance_km': 0.5,
    'metro_distance_km': 2.0,
    'avg_property_age': 5.2,
    'avg_square_feet': 1200,
    'most_common_config': 2
}

print("="*80)
print("🧪 TESTING NEWS-ENHANCED LLM INSIGHTS")
print("="*80)
print(f"\nLocality: {sample_data['area']}")
print(f"Price: ₹{sample_data['price']:,}")
print(f"Investment Score: {sample_data['investment_score']}")
print("\n" + "-"*80)

# Initialize LLM with news context enabled
llm = LLMInsights(enable_news_context=True)

if llm.available:
    print("\n✅ LLM Available")
    
    if llm.news_fetcher:
        print("✅ News context enabled - fetching recent articles...")
        
        # Get news manually to show what's being fetched
        try:
            news = llm.news_fetcher.get_locality_news(sample_data['area'], max_articles=3)
            if news:
                print(f"\n📰 Found {len(news)} recent articles:\n")
                for i, article in enumerate(news, 1):
                    print(f"{i}. [{article['date']}] {article['title']}")
            else:
                print("\n📰 No recent news found (will use general analysis)")
        except Exception as e:
            print(f"\n⚠️  Could not fetch news: {e}")
    else:
        print("⚠️  News context disabled")
    
    print("\n" + "-"*80)
    print("Generating AI insights with news context...")
    print("-"*80 + "\n")
    
    # Generate insights (news will be automatically included)
    analysis = llm.analyze_locality(sample_data, use_news=True)
    print(analysis)
    
    print("\n" + "="*80)
    print("✅ Demo Complete!")
    print("="*80)
    print("\n💡 The LLM now considers recent news when generating insights!")
    print("   This provides more context-aware recommendations.")
    
else:
    print("\n❌ LLM not available. Please set OPENAI_API_KEY in .env file")
    print("\nTo test this feature:")
    print("1. Add your OpenAI API key to .env")
    print("2. Run: python scripts/test_news_llm.py")
