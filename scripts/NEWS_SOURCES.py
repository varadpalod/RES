"""
News Sources Summary
Current configuration of all RSS feeds for Pune real estate news
"""

CONFIGURED_NEWS_SOURCES = {
    "Times of India": {
        "feeds": 2,
        "urls": [
            "https://timesofindia.indiatimes.com/rssfeeds/4118874.cms",
            "https://timesofindia.indiatimes.com/rssfeeds/-2128816549.cms"
        ],
        "coverage": "Pune local news + Real estate nationwide",
        "reputation": "⭐⭐⭐⭐⭐ (India's largest English daily)"
    },
    "Hindustan Times": {
        "feeds": 2,
        "urls": [
            "https://www.hindustantimes.com/feeds/rss/pune-news/rssfeed.xml",
            "https://www.hindustantimes.com/feeds/rss/real-estate/rssfeed.xml"
        ],
        "coverage": "Pune local news + Real estate nationwide",
        "reputation": "⭐⭐⭐⭐⭐ (Leading English daily)"
    },
    "Indian Express": {
        "feeds": 1,
        "urls": [
            "https://indianexpress.com/section/cities/pune/feed/"
        ],
        "coverage": "Pune city news (filtered for real estate)",
        "reputation": "⭐⭐⭐⭐⭐ (Trusted journalism)"
    },
    "Economic Times": {
        "feeds": 1,
        "urls": [
            "https://economictimes.indiatimes.com/wealth/real-estate/rssfeeds/74647611.cms"
        ],
        "coverage": "Real estate & property market (filtered for Pune)",
        "reputation": "⭐⭐⭐⭐⭐ (India's #1 business daily)"
    },
    "Pune Mirror": {
        "feeds": 1,
        "urls": [
            "https://punemirror.com/rss-feeds"
        ],
        "coverage": "Pune-specific local news",
        "reputation": "⭐⭐⭐⭐ (Pune-focused publication)"
    }
}

TOTAL_FEEDS = 7

FILTERING_CRITERIA = {
    "Geographic": [
        "Must mention 'Pune' or specific Pune localities",
        "Localities: Hinjewadi, Baner, Wakad, Koregaon Park, Kharadi, etc."
    ],
    "Topical": [
        "Real estate keywords: property, housing, apartment, builder, construction",
        "Infrastructure keywords: metro, road, connectivity, development",
        "Market keywords: price, investment, RERA, developer"
    ],
    "Temporal": [
        "Last 30 days by default (configurable)",
        "6-hour cache to reduce redundant fetches"
    ]
}

# All sources are:
# ✅ Reputable national/regional publications
# ✅ Filtered for Pune localities only
# ✅ Focused on real estate topics
# ✅ Cached for performance (6-hour refresh)
# ✅ No dummy data - all real RSS feeds
