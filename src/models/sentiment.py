"""
FinPulse — Sentiment Intelligence Engine
========================================
Module: src/models/sentiment.py

Responsibilities:
    - Multi-source RSS news fetching (Yahoo Finance + Google News).
    - AI-powered sentiment scoring via Hugging Face FinBERT.
    - Fallback keyword-based scoring system.

Author: FinPulse Team
Created: 2026-04-23
"""

import logging
import json
import requests
import feedparser
from datetime import datetime

logger = logging.getLogger(__name__)

# Hugging Face Inference API Config
HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"

def fetch_news_rss(ticker: str, company_name: str) -> list[dict]:
    """
    Fetch news from Yahoo Finance and Google News RSS feeds.
    Returns a deduplicated list of headlines.
    """
    headlines = []
    seen_titles = set()
    
    # RSS Feed URLs
    feeds = [
        f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
        f"https://news.google.com/rss/search?q={company_name}+stock&hl=en&gl=US&ceid=US:en"
    ]
    
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                title = entry.get("title", "").strip()
                if not title or title.lower() in seen_titles:
                    continue
                
                source = entry.get("source", {}).get("title", "Unknown Source")
                if source == "Unknown Source" and "Yahoo" in url:
                    source = "Yahoo Finance"
                elif source == "Unknown Source" and "google" in url:
                    source = "Google News"

                headlines.append({
                    "title": title,
                    "source": source,
                    "published": entry.get("published", ""),
                    "link": entry.get("link", "")
                })
                seen_titles.add(title.lower())
        except Exception as e:
            logger.warning(f"Failed to fetch RSS feed {url}: {e}")
            
    return headlines

def score_headlines(headlines: list[dict]) -> list[dict]:
    """
    Score headlines using Hugging Face FinBERT.
    Falls back to keyword scoring on API error.
    """
    if not headlines:
        return []

    titles = [h['title'] for h in headlines]
    
    try:
        response = requests.post(
            HF_API_URL,
            headers={"Content-Type": "application/json"},
            json={"inputs": titles, "options": {"wait_for_model": True}},
            timeout=8
        )
        
        if response.status_code == 200:
            results = response.json()
            # Handle both single and multiple result formats
            if isinstance(results[0], dict):
                results = [results] # Wrap single result list
                
            for i, res in enumerate(results):
                # res is a list of [{'label': 'positive', 'score': 0.9}, ...]
                best = max(res, key=lambda x: x['score'])
                label = best['label'].lower()
                
                # Map labels to scores
                score_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                headlines[i]['sentiment_label'] = label.capitalize()
                headlines[i]['sentiment_score'] = score_map.get(label, 0)
            return headlines
            
    except Exception as e:
        logger.error(f"HF Inference API error: {e}. Falling back to keywords.")
        
    # Fallback keyword scorer
    bullish_words = {'beat', 'surge', 'rally', 'growth', 'profit', 'high', 'gain', 'upgrade', 'buy'}
    bearish_words = {'miss', 'crash', 'fall', 'loss', 'cut', 'low', 'drop', 'downgrade', 'sell'}
    
    for h in headlines:
        text = h['title'].lower()
        pos_count = sum(1 for w in bullish_words if w in text)
        neg_count = sum(1 for w in bearish_words if w in text)
        
        if pos_count > neg_count:
            h['sentiment_label'] = 'Positive'
            h['sentiment_score'] = 1
        elif neg_count > pos_count:
            h['sentiment_label'] = 'Negative'
            h['sentiment_score'] = -1
        else:
            h['sentiment_label'] = 'Neutral'
            h['sentiment_score'] = 0
            
    return headlines
