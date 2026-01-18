"""
Text Processing Module
Handles text cleaning, normalization, and locality matching
"""
import re
from typing import List, Optional, Dict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PUNE_LOCALITIES

try:
    from rapidfuzz import fuzz, process
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Using exact matching only.")


class TextProcessor:
    """Text cleaning and normalization for real estate sentiment analysis"""
    
    def __init__(self, localities: List[str] = None):
        self.localities = localities or PUNE_LOCALITIES
        self.locality_lower = {loc.lower(): loc for loc in self.localities}
        
        # Common abbreviations and aliases
        self.locality_aliases = {
            "pcmc": "Pimpri-Chinchwad",
            "pimpri": "Pimpri-Chinchwad",
            "chinchwad": "Pimpri-Chinchwad",
            "koregaon": "Koregaon Park",
            "kp": "Koregaon Park",
            "kn": "Kalyani Nagar",
            "kalyani": "Kalyani Nagar",
            "viman": "Viman Nagar",
            "nibm": "NIBM Road",
            "sb road": "Senapati Bapat Road",
            "fc road": "Fergusson College Road",
            "jm road": "Jungli Maharaj Road"
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text for analysis
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase for processing
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers that are standalone (keep numbers with context)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Clean up
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_localities(self, text: str) -> List[str]:
        """
        Extract mentioned localities from text
        
        Args:
            text: Input text
            
        Returns:
            List of matched locality names (standardized)
        """
        found_localities = []
        text_lower = text.lower()
        
        # First check exact matches and aliases
        for alias, standard in self.locality_aliases.items():
            if alias in text_lower:
                if standard not in found_localities:
                    found_localities.append(standard)
        
        # Check for locality names
        for loc_lower, loc_standard in self.locality_lower.items():
            if loc_lower in text_lower:
                if loc_standard not in found_localities:
                    found_localities.append(loc_standard)
        
        return found_localities
    
    def match_locality(self, text: str, threshold: int = 80) -> Optional[str]:
        """
        Fuzzy match a locality name from text
        
        Args:
            text: Text that might contain a locality name
            threshold: Minimum match score (0-100)
            
        Returns:
            Best matching locality name or None
        """
        # First try exact extraction
        localities = self.extract_localities(text)
        if localities:
            return localities[0]
        
        # Try fuzzy matching if available
        if FUZZY_AVAILABLE:
            result = process.extractOne(
                text, 
                self.localities,
                scorer=fuzz.partial_ratio,
                score_cutoff=threshold
            )
            if result:
                return result[0]
        
        return None
    
    def standardize_locality(self, locality: str) -> str:
        """
        Standardize a locality name to match dataset format
        
        Args:
            locality: Input locality name (any format)
            
        Returns:
            Standardized locality name
        """
        if not locality:
            return ""
        
        locality_lower = locality.lower().strip()
        
        # Check aliases first
        if locality_lower in self.locality_aliases:
            return self.locality_aliases[locality_lower]
        
        # Check exact match
        if locality_lower in self.locality_lower:
            return self.locality_lower[locality_lower]
        
        # Fuzzy match
        matched = self.match_locality(locality)
        return matched if matched else locality.title()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Simple tokenization for text
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Clean first
        cleaned = self.clean_text(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', cleaned)
        
        return tokens
    
    def remove_stopwords(self, tokens: List[str], custom_stopwords: List[str] = None) -> List[str]:
        """
        Remove common stopwords from token list
        
        Args:
            tokens: List of tokens
            custom_stopwords: Additional stopwords to remove
            
        Returns:
            Filtered token list
        """
        # Basic English stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'need',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'what', 'which', 'who', 'whom', 'whose', 'where',
            'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'not', 'only', 'same', 'so',
            'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there'
        }
        
        if custom_stopwords:
            stopwords.update(custom_stopwords)
        
        return [t for t in tokens if t.lower() not in stopwords]


def preprocess_corpus(texts: List[str]) -> List[Dict]:
    """
    Preprocess a corpus of texts for sentiment analysis
    
    Args:
        texts: List of raw text strings
        
    Returns:
        List of processed text dictionaries
    """
    processor = TextProcessor()
    processed = []
    
    for text in texts:
        cleaned = processor.clean_text(text)
        localities = processor.extract_localities(text)
        tokens = processor.tokenize(text)
        
        processed.append({
            'original': text,
            'cleaned': cleaned,
            'localities': localities,
            'tokens': tokens,
            'token_count': len(tokens)
        })
    
    return processed


if __name__ == "__main__":
    # Test the text processor
    processor = TextProcessor()
    
    test_texts = [
        "Hinjewadi is becoming too crowded! Traffic is unbearable during peak hours.",
        "Looking to buy in Koregaon Park or KP area. Great infrastructure!",
        "PCMC has better value than Pune proper. Less traffic too.",
        "Visit https://example.com for more info about Baner properties @email.com"
    ]
    
    print("Text Processing Tests:")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {processor.clean_text(text)}")
        print(f"Localities: {processor.extract_localities(text)}")
        print(f"Tokens: {processor.tokenize(text)[:10]}...")
