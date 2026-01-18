"""
Configuration module for Sentiment-Aware Real Estate Intelligence System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============================================================================
# DATA FILES
# ============================================================================
PROPERTY_DATA_FILE = DATA_DIR / "pune_house_prices.csv"
SENTIMENT_CORPUS_FILE = DATA_DIR / "sentiment_corpus.json"
PROCESSED_DATA_FILE = OUTPUT_DIR / "processed_data.csv"
LOCALITY_SENTIMENT_FILE = OUTPUT_DIR / "locality_sentiment.csv"
MODEL_FILE = MODEL_DIR / "demand_predictor.joblib"

# ============================================================================
# API CONFIGURATION
# ============================================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_AI = bool(OPENAI_API_KEY)

# Gemini API configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
USE_GEMINI = bool(GEMINI_API_KEY)

# ============================================================================
# PUNE LOCALITIES (from dataset)
# ============================================================================
PUNE_LOCALITIES = [
    "Koregaon Park",
    "Hinjewadi",
    "Kharadi"
]

# ============================================================================
# SENTIMENT ANALYSIS CONFIGURATION
# ============================================================================
SENTIMENT_CATEGORIES = {
    "price_perception": {
        "positive_keywords": ["affordable", "good value", "reasonable", "worth", "bargain", "undervalued"],
        "negative_keywords": ["overpriced", "expensive", "costly", "unaffordable", "inflated", "steep"]
    },
    "infrastructure_satisfaction": {
        "positive_keywords": ["metro", "highway", "roads", "water supply", "electricity", "developed", "connectivity"],
        "negative_keywords": ["traffic", "congestion", "waterlogging", "power cuts", "underdeveloped", "poor roads"]
    },
    "investment_confidence": {
        "positive_keywords": ["appreciation", "growth", "returns", "IT hub", "development", "promising", "potential"],
        "negative_keywords": ["risky", "stagnant", "declining", "bubble", "oversupply", "slow"]
    },
    "buying_urgency": {
        "positive_keywords": ["now", "hurry", "limited", "selling fast", "high demand", "book now", "opportunity"],
        "negative_keywords": ["wait", "later", "no rush", "plenty available", "oversupply"]
    }
}

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": 42,
    "test_size": 0.2
}

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================
DASHBOARD_CONFIG = {
    "page_title": "Pune Real Estate Sentiment Intelligence",
    "page_icon": "🏠",
    "layout": "wide"
}

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================
def validate_config():
    """
    Validate configuration and environment setup.
    Returns tuple of (is_valid, warnings, errors)
    """
    warnings = []
    errors = []
    
    # Check directories exist
    for dir_name, dir_path in [("Data", DATA_DIR), ("Output", OUTPUT_DIR), ("Model", MODEL_DIR)]:
        if not dir_path.exists():
            dir_path.mkdir(exist_ok=True)
            warnings.append(f"{dir_name} directory created: {dir_path}")
    
    # Check for property data
    if not PROPERTY_DATA_FILE.exists():
        enhanced_file = DATA_DIR / "pune_house_prices_enhanced.csv"
        if not enhanced_file.exists():
            warnings.append(
                f"⚠️ Property data file not found: {PROPERTY_DATA_FILE}\n"
                "   Download from: https://www.kaggle.com/datasets/rohanchatse/pune-house-prices\n"
                "   Or see SETUP.md for instructions"
            )
    
    # Check API keys (optional)
    if not GEMINI_API_KEY:
        warnings.append(
            "ℹ️ No Gemini API key found. Using VADER sentiment analysis as fallback.\n"
            "   To enable AI: Add GEMINI_API_KEY to .env or Streamlit secrets"
        )
    
    is_valid = len(errors) == 0
    return is_valid, warnings, errors


def print_validation_results():
    """Print configuration validation results"""
    is_valid, warnings, errors = validate_config()
    
    if errors:
        print("\n❌ Configuration Errors:")
        for error in errors:
            print(f"  {error}")
    
    if warnings:
        print("\nℹ️ Configuration Warnings:")
        for warning in warnings:
            print(f"  {warning}")
    
    if is_valid and not warnings:
        print("\n✅ Configuration validated successfully")
    
    return is_valid

