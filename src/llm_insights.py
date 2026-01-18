"""
LLM Insights Generator Module
Uses OpenAI to generate real-time analysis and recommendations
Enhanced with real-world news context for better insights
"""
import os
import openai
from typing import Dict, Optional, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import news fetcher
try:
    from src.news_fetcher import NewsFetcher
    NEWS_AVAILABLE = True
except ImportError:
    NEWS_AVAILABLE = False
    print("Warning: News fetcher not available")

class LLMInsights:
    """Generates AI-powered insights using OpenAI"""
    
    def __init__(self, enable_news_context: bool = True):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if self.api_key:
            # Check for OpenRouter key
            if self.api_key.startswith("sk-or-v1"):
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url="https://openrouter.ai/api/v1"
                )
                self.model = "openai/gpt-4o-mini" # OpenRouter model ID
                print("Using OpenRouter (gpt-4o-mini)")
            else:
                self.client = openai.OpenAI(api_key=self.api_key)
                self.model = "gpt-4o-mini" # OpenAI model ID
                print("Using OpenAI (gpt-4o-mini)")
            
            self.available = True
        else:
            self.available = False
            print("Warning: OPENAI_API_KEY not found. AI insights disabled.")
        
        # Initialize news fetcher if enabled
        self.news_fetcher = None
        if enable_news_context and NEWS_AVAILABLE:
            try:
                self.news_fetcher = NewsFetcher()
                print("News context enabled")
            except Exception as e:
                print(f"Warning: Could not initialize news fetcher: {e}")
    
    def analyze_locality(self, locality_data: Dict, use_news: bool = True) -> str:
        """
        Generate comprehensive analysis for a locality
        
        Args:
            locality_data: Dictionary containing metrics for the locality
            use_news: Whether to include news context (default: True)
            
        Returns:
            Markdown formatted analysis string
        """
        if not self.available:
            return "AI Analysis unavailable. Please add OPENAI_API_KEY."
        
        # Fetch recent news for the locality
        news_context = ""
        if use_news and self.news_fetcher:
            try:
                locality_name = locality_data.get('area', '')
                news_articles = self.news_fetcher.get_locality_news(locality_name, max_articles=3)
                if news_articles:
                    news_context = f"""
        
        Latest News Context (Last 30 Days):
        {self.news_fetcher.format_news_for_llm(news_articles)}
        
        NOTE: Consider these recent developments in your analysis.
        """
            except Exception as e:
                print(f"Warning: Could not fetch news: {e}")
            
        prompt = f"""
        Act as an expert Real Estate Investment Consultant for Pune, India. 
        Analyze the following locality data and provide a professional investment thesis.
        
        Locality: {locality_data.get('area')}
        
        Key Metrics:
        - Price: ₹{locality_data.get('price', 0):,.0f}
        - Rental Yield: {locality_data.get('avg_rental_yield', 0):.2f}%
        - Predicted Demand Score: {locality_data.get('predicted_demand', 0):.1f}/100
        - Investment Score: {locality_data.get('investment_score', 0):.1f}/100
        - Market Sentiment: {locality_data.get('overall_sentiment', 0):.2f} (-1 to +1)
        - RERA Compliance: {locality_data.get('rera_compliance_pct', 0):.0f}%
        - Min Income Needed: ₹{locality_data.get('min_annual_income', 0):,.0f}/year
        - Distance to IT Park: {locality_data.get('it_park_distance_km', 0):.1f} km
        - Distance to Metro: {locality_data.get('metro_distance_km', 0):.1f} km{news_context}
        
        Provide a structured analysis in Markdown format with the following sections:
        
        ### 🏙️ Investment Thesis
        [A paragraph analyzing the potential of this locality. Is it undervalued? Overpriced? A hidden gem? Factor in recent news if available.]
        
        ### 🎯 Who Should Buy?
        [Specific buyer personas based on price and income data. e.g. Young techies, Senior investors, Families]
        
        ### ⚠️ Risk Factors
        [Analyze risks based on sentiment and metrics. Consider recent news events if available. e.g. If rental yield is low but price is high, warn about ROI.]
        
        ### 🏗️ Builder Strategy
        [Specific advice for a builder planning a project here. What configuration (1/2/3 BHK) and amenities would work best? Consider market trends from news.]
        
        Keep the tone professional, data-driven, yet concise. Use emojis for readability.
        """
        
        try:
            # Prepare optional params for OpenRouter
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a Real Estate Expert Analyst."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 800
            }
            
            # Add headers if using OpenRouter
            if "openrouter" in str(self.client.base_url):
                kwargs["extra_headers"] = {
                    "HTTP-Referer": "http://localhost:8501", # Site URL
                    "X-Title": "Pune Real Estate Intelligence" # App Name
                }

            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating AI analysis: {str(e)}"

    def generate_detailed_strategy(self, locality_data: Dict, use_news: bool = True) -> Dict:
        """
        Generate structured strategy (actions, target, pricing) for a locality
        
        Args:
            locality_data: Dictionary containing metrics for the locality
            use_news: Whether to include news context (default: True)
            
        Returns:
            Dictionary with keys: 'action_items', 'target_segment', 'pricing_strategy'
            Returns None if generation fails.
        """
        if not self.available:
            return None
        
        # Fetch recent news for the locality
        news_context = ""
        if use_news and self.news_fetcher:
            try:
                locality_name = locality_data.get('area', '')
                news_articles = self.news_fetcher.get_locality_news(locality_name, max_articles=3)
                if news_articles:
                    news_context = f"""
        
        Latest News (Last 30 Days):
        {self.news_fetcher.format_news_for_llm(news_articles)}
        """
            except Exception as e:
                print(f"Warning: Could not fetch news: {e}")
            
        prompt = f"""
        Act as a Real Estate Development Consultant for Pune. 
        You are advising a Builder on potential new residential projects.
        
        Analyze the locality data below and provide 3 specific outputs in JSON format.
        
        Locality: {locality_data.get('area')}
        
        Market Data:
        - Price: ₹{locality_data.get('price', 0):,.0f} (Avg Market Price)
        - Yield: {locality_data.get('avg_rental_yield', 0):.2f}%
        - Demand Score: {locality_data.get('predicted_demand', 0):.1f}/100
        - Investment Score: {locality_data.get('investment_score', 0):.1f}/100
        - Sentiment: {locality_data.get('overall_sentiment', 0):.2f}
        
        Existing Supply Context:
        - Avg Property Age: {locality_data.get('avg_property_age', 5):.1f} years
        - Avg Unit Size: {locality_data.get('avg_square_feet', 1000):.0f} sqft
        - Most Common Config: {locality_data.get('most_common_config', 2)} BHK
        - RERA Compliance: {locality_data.get('rera_compliance_pct', 0):.0f}%{news_context}
        
        Outputs required:
        1. "action_items": List of 3 specific, strategic actions for a NEW PROJECT. 
           - Focus on gaps in the market. (e.g. "Existing supply is old (10y+), launch modern smart homes" or "Avg size is small, launch spacious luxury units").
           - Factor in recent news developments if available.
           - Use emojis.
        2. "target_segment": Detailed buyer persona for THIS specific project type.
        3. "pricing_strategy": Specific pricing advice to enter the market (Premium vs Competitive). Consider market trends from news.
        
        Return ONLY valid JSON. No markdown formatting, no code blocks.
        {{
            "action_items": ["...", "...", "..."],
            "target_segment": "...",
            "pricing_strategy": "..."
        }}
        """
        
        try:
            # Prepare optional params for OpenRouter
            kwargs = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a Real Estate Strategist that outputs only JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500,
                "response_format": {"type": "json_object"}
            }
            
            # Add headers if using OpenRouter
            if "openrouter" in str(self.client.base_url):
                kwargs["extra_headers"] = {
                    "HTTP-Referer": "http://localhost:8501",
                    "X-Title": "Pune Real Estate Intelligence"
                }

            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
            
            import json
            return json.loads(content)
            
        except Exception as e:
            print(f"Error generating strategy: {e}")
            return None

if __name__ == "__main__":
    # Test
    llm = LLMInsights()
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
        'metro_distance_km': 2.0
    }
    if llm.available:
        print(llm.analyze_locality(sample_data))
