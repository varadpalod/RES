"""
News Fetcher Module
Retrieves recent news articles about Pune localities to provide context for LLM insights
Includes 6-hour caching to improve performance
"""
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import re
import json
import os

class NewsFetcher:
    """Fetches and filters news articles for Pune real estate"""
    
    # Cache duration: 6 hours
    CACHE_DURATION_HOURS = 6
    
    def __init__(self, cache_dir: str = "outputs"):
        # RSS feeds for Pune news from reputable sources
        # All feeds are filtered for Pune localities + real estate keywords
        self.feeds = [
            # Times of India
            "https://timesofindia.indiatimes.com/rssfeeds/4118874.cms",  # Pune Times
            "https://timesofindia.indiatimes.com/rssfeeds/-2128816549.cms",  # Real Estate
            
            # Hindustan Times
            "https://www.hindustantimes.com/feeds/rss/pune-news/rssfeed.xml",  # Pune News
            "https://www.hindustantimes.com/feeds/rss/real-estate/rssfeed.xml",  # Real Estate
            
            # Indian Express  
            "https://indianexpress.com/section/cities/pune/feed/",  # Pune City News
            
            # Economic Times
            "https://economictimes.indiatimes.com/wealth/real-estate/rssfeeds/74647611.cms",  # Real Estate
            
            # Pune Mirror (Bennett Coleman - Times Group)
            "https://punemirror.com/rss-feeds",  # Pune News
        ]
        
        # Real estate keywords
        self.re_keywords = [
            'property', 'real estate', 'housing', 'apartment', 'builder',
            'construction', 'rera', 'developer', 'residential', 'metro',
            'infrastructure', 'price', 'investment', 'project'
        ]
        
        # Cache setup
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "news_cache.json")
        self._cache = None
        self._cache_timestamp = None
        self._load_cache()
    
    def get_locality_news(self, locality: str, max_articles: int = 5, days_back: int = 30) -> List[Dict]:
        """
        Fetch recent news articles mentioning a specific locality
        Uses 6-hour cache to improve performance
        
        Args:
            locality: Name of the Pune locality
            max_articles: Maximum number of articles to return
            days_back: How many days back to search
            
        Returns:
            List of article dictionaries with title, summary, link, date
        """
        # Check cache first
        if self._is_cache_valid():
            cached_articles = self._cache.get('locality_news', {}).get(locality, [])
            if cached_articles:
                return cached_articles[:max_articles]
        
        # Cache miss or expired - fetch fresh news
        all_articles = self._fetch_locality_news(locality, days_back)
        
        # Update cache
        self._update_locality_cache(locality, all_articles)
        
        # Sort by date (most recent first) and limit
        all_articles.sort(key=lambda x: x['date'], reverse=True)
        return all_articles[:max_articles]
    
    def _fetch_locality_news(self, locality: str, days_back: int = 30) -> List[Dict]:
        """Internal method to fetch news from RSS feeds"""
        all_articles = []
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Get content
                    content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                    
                    # Check if relevant to locality and real estate
                    if self._is_relevant_to_locality(content, locality):
                        # Parse date
                        try:
                            pub_date = datetime(*entry.published_parsed[:6])
                            if pub_date < cutoff_date:
                                continue
                        except:
                            pub_date = None
                        
                        all_articles.append({
                            'title': entry.get('title', 'No title'),
                            'summary': self._clean_summary(entry.get('summary', '')),
                            'link': entry.get('link', ''),
                            'date': pub_date.strftime('%Y-%m-%d') if pub_date else 'Recent',
                            'content': content
                        })
            except Exception as e:
                print(f"Warning: Could not fetch feed {feed_url}: {e}")
                continue
        
        return all_articles
    
    def get_general_pune_news(self, max_articles: int = 3) -> List[Dict]:
        """
        Get general Pune real estate news (not locality-specific)
        Uses 6-hour cache to improve performance
        
        Args:
            max_articles: Maximum number of articles
            
        Returns:
            List of article dictionaries
        """
        # Check cache first
        if self._is_cache_valid():
            cached_general = self._cache.get('general_news', [])
            if cached_general:
                return cached_general[:max_articles]
        
        # Cache miss or expired - fetch fresh news
        articles = []
        
        for feed_url in self.feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    content = f"{entry.get('title', '')} {entry.get('summary', '')}"
                    
                    # Check if relevant to Pune real estate
                    if self._is_pune_real_estate(content):
                        articles.append({
                            'title': entry.get('title', 'No title'),
                            'summary': self._clean_summary(entry.get('summary', '')),
                            'link': entry.get('link', ''),
                            'date': 'Recent'
                        })
                        
                        if len(articles) >= max_articles:
                            break
            except Exception as e:
                continue
        
        # Update cache
        self._update_general_cache(articles)
        
        return articles[:max_articles]
    
    def _is_relevant_to_locality(self, text: str, locality: str) -> bool:
        """Check if text is relevant to the locality and real estate"""
        text_lower = text.lower()
        locality_lower = locality.lower()
        
        # Must mention the locality
        if locality_lower not in text_lower:
            return False
        
        # Must have real estate context
        has_re_keyword = any(keyword in text_lower for keyword in self.re_keywords)
        
        return has_re_keyword
    
    def _is_pune_real_estate(self, text: str) -> bool:
        """Check if text is about Pune real estate"""
        text_lower = text.lower()
        
        # Must mention Pune
        if 'pune' not in text_lower:
            return False
        
        # Must have real estate context
        has_re_keyword = any(keyword in text_lower for keyword in self.re_keywords)
        
        return has_re_keyword
    
    def _clean_summary(self, summary: str) -> str:
        """Clean HTML tags and extra whitespace from summary"""
        # Remove HTML tags
        clean = re.sub(r'<[^>]+>', '', summary)
        # Remove extra whitespace
        clean = ' '.join(clean.split())
        # Limit length
        if len(clean) > 300:
            clean = clean[:297] + '...'
        return clean
    
    def format_news_for_llm(self, articles: List[Dict]) -> str:
        """
        Format news articles as context for LLM prompts
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Formatted string summarizing news
        """
        if not articles:
            return "No recent news articles available."
        
        formatted = "Recent News Context:\n"
        for i, article in enumerate(articles, 1):
            formatted += f"{i}. [{article['date']}] {article['title']}\n"
            if article.get('summary'):
                formatted += f"   → {article['summary']}\n"
        
        return formatted.strip()
    
    def _load_cache(self):
        """Load cache from disk if available"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self._cache = cache_data.get('data', {})
                    timestamp_str = cache_data.get('timestamp')
                    if timestamp_str:
                        self._cache_timestamp = datetime.fromisoformat(timestamp_str)
        except Exception as e:
            print(f"Warning: Could not load news cache: {e}")
            self._cache = {'locality_news': {}, 'general_news': []}
            self._cache_timestamp = None
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': self._cache
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            self._cache_timestamp = datetime.now()
        except Exception as e:
            print(f"Warning: Could not save news cache: {e}")
    
    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid (less than 6 hours old)"""
        if self._cache is None or self._cache_timestamp is None:
            return False
        
        age = datetime.now() - self._cache_timestamp
        return age < timedelta(hours=self.CACHE_DURATION_HOURS)
    
    def _update_locality_cache(self, locality: str, articles: List[Dict]):
        """Update cache for a specific locality"""
        if self._cache is None:
            self._cache = {'locality_news': {}, 'general_news': []}
        
        if 'locality_news' not in self._cache:
            self._cache['locality_news'] = {}
        
        self._cache['locality_news'][locality] = articles
        self._save_cache()
    
    def _update_general_cache(self, articles: List[Dict]):
        """Update cache for general news"""
        if self._cache is None:
            self._cache = {'locality_news': {}, 'general_news': []}
        
        self._cache['general_news'] = articles
        self._save_cache()
    
    def clear_cache(self):
        """Manually clear the cache (useful for testing or forcing refresh)"""
        self._cache = {'locality_news': {}, 'general_news': []}
        self._cache_timestamp = None
        if os.path.exists(self.cache_file):
            os.remove(self.cache_file)
        print("News cache cleared")


if __name__ == "__main__":
    # Test the news fetcher
    fetcher = NewsFetcher()
    
    print("Testing News Fetcher...\n")
    print("=" * 80)
    
    # Test locality-specific news
    locality = "Hinjewadi"
    print(f"Fetching news for {locality}...\n")
    news = fetcher.get_locality_news(locality, max_articles=3)
    
    if news:
        print(f"Found {len(news)} articles:\n")
        print(fetcher.format_news_for_llm(news))
    else:
        print(f"No recent news found for {locality}")
    
    print("\n" + "=" * 80)
    print("\nFetching general Pune real estate news...\n")
    general_news = fetcher.get_general_pune_news(max_articles=2)
    
    if general_news:
        print(fetcher.format_news_for_llm(general_news))
    else:
        print("No general news found")
