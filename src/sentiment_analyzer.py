"""
Sentiment Analysis Module
Multi-dimensional sentiment extraction for real estate domain
"""
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SENTIMENT_CATEGORIES, GEMINI_API_KEY, USE_GEMINI

# Try to import VADER
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")

# Try to import Gemini
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


@dataclass
class SentimentScore:
    """Container for multi-dimensional sentiment scores"""
    price_perception: float  # -1 (overpriced) to 1 (good value)
    infrastructure_satisfaction: float  # -1 (poor) to 1 (excellent)
    investment_confidence: float  # -1 (risky) to 1 (promising)
    buying_urgency: float  # 0 (no rush) to 1 (urgent)
    overall_sentiment: float  # -1 (negative) to 1 (positive)
    confidence: float  # 0 to 1 (how confident we are in the analysis)
    
    def to_dict(self) -> Dict:
        return {
            'price_perception': round(self.price_perception, 3),
            'infrastructure_satisfaction': round(self.infrastructure_satisfaction, 3),
            'investment_confidence': round(self.investment_confidence, 3),
            'buying_urgency': round(self.buying_urgency, 3),
            'overall_sentiment': round(self.overall_sentiment, 3),
            'confidence': round(self.confidence, 3)
        }


class SentimentAnalyzer:
    """
    Multi-dimensional sentiment analyzer for real estate text
    Uses Gemini AI when available, falls back to VADER + keyword analysis
    """
    
    def __init__(self, use_ai: bool = True):
        self.use_ai = use_ai and USE_GEMINI and GEMINI_AVAILABLE
        self.categories = SENTIMENT_CATEGORIES
        
        # Initialize VADER
        if VADER_AVAILABLE:
            self.vader = SentimentIntensityAnalyzer()
        else:
            self.vader = None
        
        # Initialize Gemini if available and configured
        if self.use_ai:
            try:
                genai.configure(api_key=GEMINI_API_KEY)
                self.model = genai.GenerativeModel('gemini-1.5-flash')
                print("✓ Gemini AI initialized for sentiment analysis")
            except Exception as e:
                print(f"⚠ Gemini initialization failed: {e}. Using VADER fallback.")
                self.use_ai = False
                self.model = None
        else:
            self.model = None
    
    def analyze(self, text: str) -> SentimentScore:
        """
        Analyze text for multi-dimensional sentiment
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentScore with all dimensions
        """
        if not text or len(text.strip()) < 10:
            return SentimentScore(0, 0, 0, 0.5, 0, 0.1)
        
        # Try AI-based analysis first
        if self.use_ai and self.model:
            try:
                return self._analyze_with_ai(text)
            except Exception as e:
                print(f"AI analysis failed: {e}. Falling back to VADER.")
        
        # Fallback to keyword + VADER analysis
        return self._analyze_with_keywords(text)
    
    def _analyze_with_ai(self, text: str) -> SentimentScore:
        """Use Gemini AI for nuanced sentiment extraction"""
        
        prompt = f"""Analyze this real estate text and provide sentiment scores.

Text: "{text}"

Provide scores from -1 to 1 for each dimension (use decimals like 0.5, -0.3):
1. price_perception: -1 (overpriced/expensive) to 1 (affordable/good value)
2. infrastructure_satisfaction: -1 (poor infrastructure) to 1 (excellent infrastructure)
3. investment_confidence: -1 (risky/bad investment) to 1 (promising/great ROI)
4. buying_urgency: 0 (no rush/wait) to 1 (buy now/urgent)
5. overall_sentiment: -1 (very negative) to 1 (very positive)

Respond ONLY with a JSON object like:
{{"price_perception": 0.3, "infrastructure_satisfaction": 0.5, "investment_confidence": 0.7, "buying_urgency": 0.4, "overall_sentiment": 0.5}}"""
        
        response = self.model.generate_content(prompt)
        
        # Parse the JSON response
        import json
        response_text = response.text.strip()
        
        # Extract JSON from response (handle markdown code blocks)
        if "```" in response_text:
            json_match = re.search(r'\{[^}]+\}', response_text)
            if json_match:
                response_text = json_match.group()
        
        scores = json.loads(response_text)
        
        return SentimentScore(
            price_perception=float(scores.get('price_perception', 0)),
            infrastructure_satisfaction=float(scores.get('infrastructure_satisfaction', 0)),
            investment_confidence=float(scores.get('investment_confidence', 0)),
            buying_urgency=max(0, float(scores.get('buying_urgency', 0.5))),  # Urgency is 0-1
            overall_sentiment=float(scores.get('overall_sentiment', 0)),
            confidence=0.9  # High confidence for AI analysis
        )
    
    def _analyze_with_keywords(self, text: str) -> SentimentScore:
        """Fallback analysis using VADER and keyword matching"""
        
        text_lower = text.lower()
        
        # Get base sentiment from VADER
        if self.vader:
            vader_scores = self.vader.polarity_scores(text)
            base_sentiment = vader_scores['compound']
        else:
            base_sentiment = 0
        
        # Calculate domain-specific scores
        price_score = self._calculate_domain_score(text_lower, 'price_perception')
        infra_score = self._calculate_domain_score(text_lower, 'infrastructure_satisfaction')
        invest_score = self._calculate_domain_score(text_lower, 'investment_confidence')
        urgency_score = self._calculate_urgency_score(text_lower)
        
        # If domain scores are neutral, use base sentiment
        if price_score == 0:
            price_score = base_sentiment * 0.5
        if infra_score == 0:
            infra_score = base_sentiment * 0.5
        if invest_score == 0:
            invest_score = base_sentiment * 0.5
        
        return SentimentScore(
            price_perception=max(-1, min(1, price_score)),
            infrastructure_satisfaction=max(-1, min(1, infra_score)),
            investment_confidence=max(-1, min(1, invest_score)),
            buying_urgency=max(0, min(1, urgency_score)),
            overall_sentiment=base_sentiment,
            confidence=0.6  # Lower confidence for keyword-based analysis
        )
    
    def _calculate_domain_score(self, text: str, category: str) -> float:
        """Calculate score for a specific domain category"""
        
        if category not in self.categories:
            return 0
        
        cat_config = self.categories[category]
        pos_keywords = cat_config.get('positive_keywords', [])
        neg_keywords = cat_config.get('negative_keywords', [])
        
        pos_count = sum(1 for kw in pos_keywords if kw in text)
        neg_count = sum(1 for kw in neg_keywords if kw in text)
        
        total = pos_count + neg_count
        if total == 0:
            return 0
        
        # Score from -1 to 1
        score = (pos_count - neg_count) / total
        return score
    
    def _calculate_urgency_score(self, text: str) -> float:
        """Calculate buying urgency score (0 to 1)"""
        
        urgency_config = self.categories.get('buying_urgency', {})
        pos_keywords = urgency_config.get('positive_keywords', [])
        neg_keywords = urgency_config.get('negative_keywords', [])
        
        pos_count = sum(1 for kw in pos_keywords if kw in text)
        neg_count = sum(1 for kw in neg_keywords if kw in text)
        
        # Base urgency is 0.5 (neutral)
        urgency = 0.5
        
        if pos_count > 0:
            urgency += 0.15 * pos_count
        if neg_count > 0:
            urgency -= 0.15 * neg_count
        
        return max(0, min(1, urgency))
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentScore]:
        """
        Analyze multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentScores
        """
        return [self.analyze(text) for text in texts]


def analyze_corpus(corpus: List[Dict]) -> List[Dict]:
    """
    Analyze a corpus of sentiment records
    
    Args:
        corpus: List of records with 'text' field
        
    Returns:
        Corpus with added sentiment scores
    """
    analyzer = SentimentAnalyzer(use_ai=False)  # Use VADER for batch processing
    
    for record in corpus:
        text = record.get('text', '')
        scores = analyzer.analyze(text)
        record['sentiment_scores'] = scores.to_dict()
    
    print(f"✓ Analyzed {len(corpus)} records")
    return corpus


if __name__ == "__main__":
    # Test the sentiment analyzer
    analyzer = SentimentAnalyzer(use_ai=False)
    
    test_texts = [
        "Hinjewadi is becoming too crowded and properties are overpriced! Traffic is unbearable.",
        "Great investment opportunity in Baner! Metro line coming soon, prices will appreciate.",
        "Koregaon Park offers excellent value for money with world-class infrastructure.",
        "Wait before buying in Wakad. Market is oversupplied and prices might drop.",
        "Book now! Limited units available at pre-launch prices in Kharadi!"
    ]
    
    print("Sentiment Analysis Tests:")
    print("=" * 80)
    
    for text in test_texts:
        scores = analyzer.analyze(text)
        print(f"\nText: {text[:70]}...")
        print(f"  Price Perception:     {scores.price_perception:+.2f}")
        print(f"  Infrastructure:       {scores.infrastructure_satisfaction:+.2f}")
        print(f"  Investment Confidence:{scores.investment_confidence:+.2f}")
        print(f"  Buying Urgency:       {scores.buying_urgency:.2f}")
        print(f"  Overall Sentiment:    {scores.overall_sentiment:+.2f}")
        print(f"  Confidence:           {scores.confidence:.2f}")
