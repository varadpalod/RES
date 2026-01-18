"""
Demo: News Article Sentiment Analysis
Fetches real news articles about Pune real estate and analyzes sentiment
"""
import feedparser
import re
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PUNE_LOCALITIES

class NewsArticleSentiment:
    """Quick demo of news sentiment analysis"""
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.localities = PUNE_LOCALITIES
        
        # RSS feeds for Pune news
        self.feeds = [
            "https://timesofindia.indiatimes.com/rssfeeds/4118874.cms",  # Pune Times
            "https://timesofindia.indiatimes.com/rssfeeds/-2128816549.cms",  # Real Estate
        ]
    
    def fetch_articles(self, max_articles=10):
        """Fetch recent articles from RSS feeds"""
        print("📰 Fetching news articles...\n")
        articles = []
        
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:max_articles]:
                    # Filter for Pune and real estate keywords
                    content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                    
                    if self._is_relevant(content):
                        articles.append({
                            'title': entry.get('title', 'No title'),
                            'summary': entry.get('summary', ''),
                            'link': entry.get('link', ''),
                            'published': entry.get('published', ''),
                            'content': content
                        })
            except Exception as e:
                print(f"⚠️  Error fetching feed {feed_url}: {e}")
        
        return articles
    
    def _is_relevant(self, text):
        """Check if article is relevant to Pune real estate"""
        text_lower = text.lower()
        
        # Real estate keywords
        re_keywords = ['property', 'real estate', 'housing', 'apartment', 
                       'builder', 'construction', 'rera', 'developer', 
                       'residential', 'metro', 'infrastructure']
        
        # Must mention Pune or a Pune locality
        pune_mentioned = 'pune' in text_lower or any(
            locality.lower() in text_lower for locality in self.localities
        )
        
        # Must have real estate context
        re_mentioned = any(keyword in text_lower for keyword in re_keywords)
        
        return pune_mentioned and re_mentioned
    
    def extract_localities(self, text):
        """Extract mentioned localities from text"""
        mentioned = []
        text_lower = text.lower()
        
        for locality in self.localities:
            if locality.lower() in text_lower:
                mentioned.append(locality)
        
        return mentioned
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        scores = self.analyzer.polarity_scores(text)
        
        # Classify sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'POSITIVE 📈'
        elif scores['compound'] <= -0.05:
            sentiment = 'NEGATIVE 📉'
        else:
            sentiment = 'NEUTRAL ➡️'
        
        return {
            'sentiment': sentiment,
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def categorize_article(self, text):
        """Categorize article by topic"""
        text_lower = text.lower()
        
        categories = {
            'Infrastructure': ['metro', 'road', 'highway', 'bridge', 'connectivity', 'transport'],
            'Pricing': ['price', 'expensive', 'affordable', 'cost', 'rates', 'cheaper'],
            'Regulations': ['rera', 'policy', 'government', 'tax', 'regulation', 'permission'],
            'Development': ['launch', 'project', 'developer', 'builder', 'construction'],
            'Market': ['demand', 'sales', 'market', 'investment', 'growth', 'decline']
        }
        
        for category, keywords in categories.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return 'General'
    
    def run_demo(self):
        """Run the complete demo"""
        print("=" * 80)
        print("🏘️  NEWS ARTICLE SENTIMENT ANALYSIS DEMO")
        print("=" * 80)
        print()
        
        # Fetch articles
        articles = self.fetch_articles(max_articles=20)
        
        if not articles:
            print("❌ No relevant articles found. Using sample article...\n")
            articles = [self._get_sample_article()]
        
        print(f"✅ Found {len(articles)} relevant articles\n")
        print("=" * 80)
        
        # Analyze each article
        locality_sentiments = {}
        
        for i, article in enumerate(articles, 1):
            print(f"\n📄 ARTICLE #{i}")
            print(f"Title: {article['title'][:100]}...")
            print(f"Published: {article['published']}")
            print(f"Link: {article['link'][:60]}...")
            
            # Extract localities
            localities = self.extract_localities(article['content'])
            print(f"Localities Mentioned: {', '.join(localities) if localities else 'None specific'}")
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(article['content'])
            print(f"Sentiment: {sentiment['sentiment']} (Score: {sentiment['compound']:.3f})")
            
            # Categorize
            category = self.categorize_article(article['content'])
            print(f"Category: {category}")
            
            # Store locality-level sentiment
            for locality in localities:
                if locality not in locality_sentiments:
                    locality_sentiments[locality] = []
                locality_sentiments[locality].append({
                    'title': article['title'],
                    'sentiment_score': sentiment['compound'],
                    'category': category
                })
            
            print("-" * 80)
        
        # Show locality-level aggregation
        if locality_sentiments:
            print("\n" + "=" * 80)
            print("📊 LOCALITY-LEVEL SENTIMENT AGGREGATION")
            print("=" * 80)
            
            for locality, sentiments in sorted(locality_sentiments.items()):
                avg_sentiment = sum(s['sentiment_score'] for s in sentiments) / len(sentiments)
                
                if avg_sentiment >= 0.05:
                    trend = "📈 POSITIVE"
                elif avg_sentiment <= -0.05:
                    trend = "📉 NEGATIVE"
                else:
                    trend = "➡️  NEUTRAL"
                
                print(f"\n🏘️  {locality}")
                print(f"   Articles: {len(sentiments)}")
                print(f"   Avg Sentiment: {avg_sentiment:.3f} {trend}")
                print(f"   Topics: {', '.join(set(s['category'] for s in sentiments))}")
        
        print("\n" + "=" * 80)
        print("✅ Demo Complete!")
        print("=" * 80)
        print("\n💡 This shows how news articles can enhance your sentiment analysis")
        print("   by providing real-world signals about localities, infrastructure,")
        print("   pricing trends, and market sentiment.\n")
    
    def _get_sample_article(self):
        """Get a sample article if RSS fails"""
        return {
            'title': 'Pune Metro Line 3 to Connect Hinjewadi with City Center',
            'summary': 'The Maharashtra government approved the new metro line connecting Hinjewadi IT Park with the city center. This is expected to boost real estate demand in areas like Wakad, Baner, and Hinjewadi. Property prices in these localities are expected to see a 15-20% surge. Developers are planning new residential projects along the metro corridor.',
            'link': 'https://example.com/pune-metro',
            'published': datetime.now().strftime('%Y-%m-%d'),
            'content': 'Pune Metro Line 3 to Connect Hinjewadi with City Center. The Maharashtra government approved the new metro line connecting Hinjewadi IT Park with the city center. This is expected to boost real estate demand in areas like Wakad, Baner, and Hinjewadi. Property prices in these localities are expected to see a 15-20% surge. Developers are planning new residential projects along the metro corridor.'
        }


if __name__ == "__main__":
    demo = NewsArticleSentiment()
    demo.run_demo()
