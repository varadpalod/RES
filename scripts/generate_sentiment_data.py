"""
Synthetic Sentiment Data Generator
Creates realistic sentiment data for Pune real estate localities
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, PUNE_LOCALITIES, SENTIMENT_CORPUS_FILE


# Templates for different sentiment types
NEWS_TEMPLATES = {
    "positive_price": [
        "{locality} emerges as most affordable premium locality in Pune with excellent value proposition",
        "Property prices in {locality} offer better value compared to neighboring areas, attract first-time buyers",
        "Real estate experts recommend {locality} for budget-conscious investors seeking quality homes",
        "{locality} offers competitive pricing without compromising on amenities and connectivity"
    ],
    "negative_price": [
        "Property prices in {locality} see steep 20% hike, buyers express concerns over affordability",
        "{locality} real estate becoming increasingly unaffordable for middle-class families",
        "Overpriced properties in {locality} lead to slowing sales momentum",
        "Buyers hesitant as {locality} property rates touch all-time highs"
    ],
    "positive_infra": [
        "Metro line extension to {locality} boosts real estate sentiment significantly",
        "{locality} witnesses major infrastructure upgrade with new flyovers and road widening",
        "Smart city initiatives transform {locality} into modern residential hub",
        "New IT parks and commercial developments near {locality} improve connectivity"
    ],
    "negative_infra": [
        "Traffic congestion in {locality} reaches critical levels during peak hours",
        "Waterlogging issues plague {locality} during monsoon season, residents frustrated",
        "Poor road conditions in {locality} despite high property taxes, residents complain",
        "{locality} faces frequent power cuts and water supply issues"
    ],
    "positive_investment": [
        "Property values in {locality} appreciate 15% year-on-year, investors rejoice",
        "{locality} real estate delivers consistent returns, becomes investor favorite",
        "Major IT companies setting up offices near {locality} signal growth potential",
        "{locality} ranked among top 5 emerging investment destinations in Pune"
    ],
    "negative_investment": [
        "Real estate market in {locality} shows signs of stagnation, investors worried",
        "Oversupply of apartments in {locality} leads to declining rental yields",
        "{locality} property market faces uncertainty amid changing buyer preferences",
        "Delayed projects in {locality} shake investor confidence in the area"
    ],
    "high_urgency": [
        "Limited inventory in {locality} creates urgency among homebuyers",
        "Fast-selling projects in {locality} - buyers advised to book early",
        "{locality} pre-launch offers seeing unprecedented demand from NRIs",
        "Last few units available at old rates in {locality} premium projects"
    ],
    "low_urgency": [
        "Abundant inventory in {locality} gives buyers time to make informed decisions",
        "No rush to buy in {locality} as new projects set to launch next quarter",
        "{locality} market stabilizes, experts advise buyers to wait for festive offers"
    ]
}

REVIEW_TEMPLATES = {
    "positive": [
        "Living in {locality} for 3 years now. Best decision ever! Great connectivity and peaceful environment.",
        "{locality} has transformed completely. Metro, malls, hospitals - everything within reach. Highly recommend!",
        "Bought a 2BHK in {locality} in 2020. Value has appreciated significantly. Happy with my investment.",
        "The social infrastructure in {locality} is excellent. Good schools, parks, and shopping options.",
        "{locality} offers the perfect work-life balance. Close to IT hubs yet away from the chaos."
    ],
    "negative": [
        "Traffic in {locality} has become unbearable. What used to be a 15-min commute now takes an hour.",
        "Overrated area. {locality} properties are overpriced for what you get. Better options elsewhere.",
        "Water supply issues in {locality} are frustrating. Have to rely on tankers during summer.",
        "Noise pollution in {locality} has increased due to ongoing construction everywhere.",
        "Maintenance costs in {locality} societies are ridiculously high. Not worth it."
    ],
    "mixed": [
        "{locality} is good but getting crowded. Infrastructure hasn't kept pace with development.",
        "Great location but prices in {locality} are getting out of hand. Mixed feelings overall.",
        "{locality} has potential but needs better public transport connectivity.",
        "Peaceful locality but {locality} lacks quality healthcare facilities nearby."
    ]
}

SOCIAL_MEDIA_TEMPLATES = {
    "positive": [
        "Just moved to {locality}! The vibe here is amazing 🏠✨ #PuneRealEstate #NewHome",
        "{locality} morning walks hit different! Best area to live in Pune hands down 🌅",
        "Property prices may be high but {locality} quality of life is unmatched 💯",
        "Finally invested in {locality}! Excited for the appreciation 📈 #RealEstateInvestment"
    ],
    "negative": [
        "Stuck in {locality} traffic again 😤 This area needs serious infrastructure work",
        "Power cut in {locality} for the 4th time this week. Paying premium rates for what? 😡",
        "{locality} is overhyped and overpriced. Don't fall for the marketing 🚫",
        "Regretting my {locality} investment. Market is completely stagnant here 📉"
    ],
    "neutral": [
        "Checking out properties in {locality} this weekend. Any recommendations? 🤔",
        "How's the water supply situation in {locality}? Planning to buy there.",
        "{locality} vs Hinjewadi for IT professionals - thoughts? #PuneHomes"
    ]
}


def generate_sentiment_record(locality: str, source_type: str, sentiment_type: str) -> Dict:
    """Generate a single sentiment record"""
    
    # Select template based on source and sentiment
    if source_type == "news":
        templates = NEWS_TEMPLATES.get(sentiment_type, NEWS_TEMPLATES["positive_investment"])
    elif source_type == "review":
        templates = REVIEW_TEMPLATES.get(sentiment_type.split("_")[0], REVIEW_TEMPLATES["mixed"])
    else:  # social_media
        templates = SOCIAL_MEDIA_TEMPLATES.get(sentiment_type.split("_")[0], SOCIAL_MEDIA_TEMPLATES["neutral"])
    
    text = random.choice(templates).format(locality=locality)
    
    # Generate random date within last 6 months
    days_ago = random.randint(0, 180)
    date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
    
    return {
        "id": f"{source_type[:3]}_{locality[:3]}_{random.randint(1000, 9999)}",
        "text": text,
        "locality": locality,
        "source_type": source_type,
        "date": date,
        "sentiment_category": sentiment_type
    }


def generate_corpus(num_records: int = 600) -> List[Dict]:
    """
    Generate complete sentiment corpus
    
    Args:
        num_records: Target number of records to generate
        
    Returns:
        List of sentiment records
    """
    corpus = []
    
    source_types = ["news", "review", "social_media"]
    
    # Sentiment distribution (more mixed/realistic)
    sentiment_types = {
        "news": [
            "positive_price", "negative_price",
            "positive_infra", "negative_infra", 
            "positive_investment", "negative_investment",
            "high_urgency", "low_urgency"
        ],
        "review": ["positive", "negative", "mixed"],
        "social_media": ["positive", "negative", "neutral"]
    }
    
    # Weight localities (some more popular than others)
    locality_weights = {
        "Hinjewadi": 1.5,
        "Koregaon Park": 1.4,
        "Baner": 1.3,
        "Viman Nagar": 1.3,
        "Kharadi": 1.2,
        "Wakad": 1.2,
        "Pimpri-Chinchwad": 1.1
    }
    
    records_per_locality = num_records // len(PUNE_LOCALITIES)
    
    for locality in PUNE_LOCALITIES:
        weight = locality_weights.get(locality, 1.0)
        locality_records = int(records_per_locality * weight)
        
        for _ in range(locality_records):
            source = random.choice(source_types)
            sentiment = random.choice(sentiment_types[source])
            record = generate_sentiment_record(locality, source, sentiment)
            corpus.append(record)
    
    random.shuffle(corpus)
    
    print(f"✓ Generated {len(corpus)} sentiment records")
    print(f"  Sources: {dict(pd.DataFrame(corpus)['source_type'].value_counts())}")
    
    return corpus


def save_corpus(corpus: List[Dict], file_path: Path = None) -> None:
    """Save corpus to JSON file"""
    path = file_path or SENTIMENT_CORPUS_FILE
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved corpus to {path}")


# Import pandas here only for the value_counts in generate_corpus
import pandas as pd


if __name__ == "__main__":
    print("Generating synthetic Pune real estate sentiment data...")
    corpus = generate_corpus(600)
    save_corpus(corpus)
    
    # Show sample
    print("\nSample records:")
    for record in corpus[:5]:
        print(f"  [{record['source_type']}] {record['locality']}: {record['text'][:80]}...")
