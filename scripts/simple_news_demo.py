"""
Simplified News Sentiment Demo - Shows the concept clearly
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Sample news articles (real-world examples of what we'd fetch)
SAMPLE_ARTICLES = [
    {
        'title': 'Pune Metro Line 3 Approved: Hinjewadi to Get Direct Connectivity',
        'content': '''The Maharashtra government has approved Pune Metro Line 3, which will connect 
        Hinjewadi IT Park directly to the city center via Wakad and Baner. Real estate experts predict 
        a 15-20% surge in property prices along the metro corridor. Several premium builders are 
        planning new residential projects in anticipation of improved connectivity.''',
        'source': 'Times of India',
        'date': '2026-01-15'
    },
    {
        'title': 'Koregaon Park Faces Waterlogging Crisis During Monsoon',
        'content': '''Residents of Koregaon Park are frustrated with recurring waterlogging issues. 
        The PMC has failed to address drainage problems, causing property damage and reducing 
        the area\'s appeal to new buyers. Real estate agents report a 10% decline in inquiries 
        for properties in the affected zones.''',
        'source': 'Pune Mirror',
        'date': '2026-01-12'
    },
    {
        'title': 'Baner Emerges as Top Investment Destination in Pune',
        'content': '''Baner has emerged as the most sought-after locality for real estate investment, 
        thanks to its proximity to IT hubs, excellent connectivity, and growing social infrastructure. 
        Property prices have seen a healthy 8% appreciation over the last year, with strong demand 
        from young professionals and IT workers.''',
        'source': 'Economic Times',
        'date': '2026-01-10'
    },
    {
        'title': 'RERA Cracks Down on Delayed Projects in Pune',
        'content': '''MahaRERA has issued notices to 15 builders in Pune for project delays and 
        violations. Areas like Wagholi and Kharadi are most affected, with buyers facing 
        2-3 year delays in possession. This has impacted buyer confidence, with many preferring 
        ready-to-move properties over under-construction units.''',
        'source': 'Hindustan Times',
        'date': '2026-01-08'
    }
]

PUNE_LOCALITIES = [
    "Hinjewadi", "Baner", "Wakad", "Koregaon Park", "Kharadi", 
    "Wagholi", "Viman Nagar", "Kalyani Nagar"
]

def extract_localities(text):
    """Find which localities are mentioned"""
    found = []
    text_lower = text.lower()
    for locality in PUNE_LOCALITIES:
        if locality.lower() in text_lower:
            found.append(locality)
    return found

def categorize_news(text):
    """Categorize the news article"""
    text_lower = text.lower()
    if any(word in text_lower for word in ['metro', 'road', 'connectivity', 'infrastructure']):
        return 'Infrastructure'
    elif any(word in text_lower for word in ['price', 'expensive', 'affordable', 'investment']):
        return 'Pricing & Investment'
    elif any(word in text_lower for word in ['rera', 'delayed', 'violation', 'regulation']):
        return 'Regulations & Compliance'
    elif any(word in text_lower for word in ['waterlogging', 'drainage', 'crisis', 'problem']):
        return 'Infrastructure Issues'
    return 'General Market'

def analyze_demo():
    """Run the demo"""
    analyzer = SentimentIntensityAnalyzer()
    
    print("\n" + "="*80)
    print("📰 NEWS ARTICLE SENTIMENT ANALYSIS - WORKING DEMO")
    print("="*80 + "\n")
    
    locality_sentiments = {}
    
    for i, article in enumerate(SAMPLE_ARTICLES, 1):
        print(f"\n{'='*80}")
        print(f"ARTICLE #{i}: {article['title']}")
        print(f"{'='*80}")
        print(f"Source: {article['source']} | Date: {article['date']}")
        print(f"\nContent Preview:")
        print(article['content'][:200] + "...")
        
        # Extract localities
        localities = extract_localities(article['content'])
        print(f"\n🏘️  Localities Mentioned: {', '.join(localities) if localities else 'None'}")
        
        # Analyze sentiment
        sentiment_scores = analyzer.polarity_scores(article['content'])
        compound = sentiment_scores['compound']
        
        if compound >= 0.05:
            sentiment_label = "📈 POSITIVE"
            color = "green"
        elif compound <= -0.05:
            sentiment_label = "📉 NEGATIVE"
            color = "red"
        else:
            sentiment_label = "➡️  NEUTRAL"
            color = "yellow"
        
        print(f"📊 Sentiment: {sentiment_label} (Score: {compound:.3f})")
        
        # Categorize
        category = categorize_news(article['content'])
        print(f"🏷️  Category: {category}")
        
        # Store for locality aggregation
        for locality in localities:
            if locality not in locality_sentiments:
                locality_sentiments[locality] = {
                    'scores': [],
                    'articles': []
                }
            locality_sentiments[locality]['scores'].append(compound)
            locality_sentiments[locality]['articles'].append({
                'title': article['title'],
                'sentiment': compound,
                'category': category
            })
    
    # Show locality-level aggregation
    print("\n\n" + "="*80)
    print("📊 LOCALITY-LEVEL SENTIMENT SUMMARY")
    print("="*80)
    
    for locality, data in sorted(locality_sentiments.items()):
        avg_sentiment = sum(data['scores']) / len(data['scores'])
        
        if avg_sentiment >= 0.05:
            trend = "📈 POSITIVE"
        elif avg_sentiment <= -0.05:
            trend = "📉 NEGATIVE"
        else:
            trend = "➡️  NEUTRAL"
        
        print(f"\n🏘️  {locality.upper()}")
        print(f"   └─ News Articles: {len(data['articles'])}")
        print(f"   └─ Average Sentiment: {avg_sentiment:.3f} {trend}")
        print(f"   └─ Recent Headlines:")
        for article in data['articles']:
            emoji = "📈" if article['sentiment'] > 0 else "📉" if article['sentiment'] < 0 else "➡️"
            print(f"      {emoji} {article['title'][:70]}...")
    
    # Show how this integrates with existing system
    print("\n\n" + "="*80)
    print("💡 INTEGRATION WITH YOUR EXISTING SYSTEM")
    print("="*80)
    print("""
How this enhances your broker_sentiment platform:

1. CURRENT: You have synthetic sentiment data
   ENHANCED: Add real-world news sentiment as a new data source

2. CURRENT: sentiment_analyzer.py analyzes buyer reviews
   ENHANCED: Also analyze news articles for market signals

3. CURRENT: Your ML model uses sentiment scores
   ENHANCED: Add 'news_sentiment_score' as a new feature

4. CURRENT: Dashboard shows property-level sentiment
   ENHANCED: Show "Recent News" section per locality with impact scores

5. EXAMPLE FEATURE ENGINEERING:
   - news_sentiment_30d: Average news sentiment (last 30 days)
   - news_volume_7d: Number of news mentions (last 7 days)
   - news_infrastructure_score: Positive infrastructure news impact
   - news_risk_score: Negative news (RERA issues, delays, etc.)

6. DASHBOARD ENHANCEMENT:
   - Add "News Feed" tab
   - Show news alerts (e.g., "New metro line approved in Hinjewadi!")
   - Display news sentiment trend charts
   - Highlight properties in localities with positive news momentum
    """)
    
    print("="*80)
    print("✅ Demo Complete! Ready to integrate into your system?")
    print("="*80 + "\n")

if __name__ == "__main__":
    analyze_demo()
