from typing import Dict, List, Optional, Tuple
from loguru import logger
import openai
from newsapi import NewsApiClient
from datetime import datetime, timedelta
import json
import time

from config.config import AI_CONFIG, TRADING_CONFIG

class AIAnalyzer:
    def __init__(self):
        # Initialize OpenAI
        openai.api_key = AI_CONFIG["openai_api_key"]
        
        # Initialize NewsAPI
        self.newsapi = NewsApiClient(api_key=AI_CONFIG["news_api_key"])
        
        # Cache for sentiment analysis
        self.sentiment_cache = {}
        self.cache_duration = timedelta(hours=1)
    
    def get_news(self, symbol: str, hours_back: int = 24) -> List[Dict]:
        """Fetch relevant news articles for a trading pair."""
        try:
            # Convert forex pair to search terms
            base_currency = symbol[:3]
            quote_currency = symbol[3:]
            
            # Create search query
            query = f"({base_currency} OR {quote_currency}) AND (forex OR currency OR economy)"
            
            # Get news from NewsAPI
            from_time = (datetime.now() - timedelta(hours=hours_back)).strftime("%Y-%m-%dT%H:%M:%S")
            
            articles = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='relevancy',
                from_param=from_time,
                page_size=10
            )
            
            if not articles or articles.get('status') != 'ok':
                logger.error("Failed to fetch news articles")
                return []
            
            return [{
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'source': article.get('source', {}).get('name', ''),
                'published_at': article.get('publishedAt', ''),
                'url': article.get('url', '')
            } for article in articles.get('articles', [])]
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    async def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of text using OpenAI."""
        try:
            # Check cache first
            cache_key = hash(text)
            if cache_key in self.sentiment_cache:
                timestamp, score = self.sentiment_cache[cache_key]
                if datetime.now() - timestamp < self.cache_duration:
                    return score
            
            # Prepare prompt for GPT
            prompt = f"""Analyze the sentiment of the following forex market related text and rate it on a scale from -1 (extremely bearish) to 1 (extremely bullish), where 0 is neutral. Only respond with the numerical score.

Text: {text}

Score:"""
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a forex market sentiment analyzer. Analyze the given text and provide a sentiment score between -1 (bearish) and 1 (bullish)."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=10
            )
            
            # Extract score from response
            try:
                score = float(response.choices[0].message.content.strip())
                score = max(-1.0, min(1.0, score))  # Ensure score is between -1 and 1
            except ValueError:
                logger.error("Failed to parse sentiment score")
                score = 0.0
            
            # Cache the result
            self.sentiment_cache[cache_key] = (datetime.now(), score)
            
            return score
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0
    
    async def get_market_sentiment(self, symbol: str) -> Tuple[float, List[Dict]]:
        """Get overall market sentiment for a symbol."""
        try:
            # Fetch recent news
            articles = self.get_news(symbol)
            if not articles:
                return 0.0, []
            
            # Analyze sentiment for each article
            total_sentiment = 0.0
            analyzed_articles = []
            
            for article in articles:
                # Combine title and description for analysis
                text = f"{article['title']} {article['description']}"
                sentiment = await self.analyze_sentiment(text)
                
                total_sentiment += sentiment
                analyzed_articles.append({
                    **article,
                    'sentiment': sentiment
                })
            
            # Calculate average sentiment
            avg_sentiment = total_sentiment / len(articles)
            
            return avg_sentiment, analyzed_articles
            
        except Exception as e:
            logger.error(f"Error getting market sentiment: {str(e)}")
            return 0.0, []
    
    def detect_high_impact_events(self, articles: List[Dict]) -> bool:
        """Detect if there are any high-impact events in the news."""
        try:
            high_impact_keywords = [
                'fed', 'fomc', 'rate decision', 'nfp', 'non-farm payroll',
                'gdp', 'cpi', 'inflation', 'central bank', 'interest rate'
            ]
            
            for article in articles:
                title_lower = article['title'].lower()
                desc_lower = article['description'].lower()
                
                if any(keyword in title_lower or keyword in desc_lower 
                      for keyword in high_impact_keywords):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting high impact events: {str(e)}")
            return False
    
    def get_volatility_adjustment(self, articles: List[Dict]) -> float:
        """Calculate volatility adjustment factor based on news impact."""
        try:
            if self.detect_high_impact_events(articles):
                # Increase volatility expectations for high impact news
                return 1.5
            
            # Calculate based on sentiment deviation
            sentiments = [article.get('sentiment', 0) for article in articles]
            if not sentiments:
                return 1.0
            
            # Calculate standard deviation of sentiments
            import numpy as np
            sentiment_std = np.std(sentiments)
            
            # Adjust volatility based on sentiment deviation
            # Higher deviation suggests more volatile conditions
            volatility_factor = 1.0 + (sentiment_std * 0.5)
            
            return min(2.0, max(1.0, volatility_factor))
            
        except Exception as e:
            logger.error(f"Error calculating volatility adjustment: {str(e)}")
            return 1.0 