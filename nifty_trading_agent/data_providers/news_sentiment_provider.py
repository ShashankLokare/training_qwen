"""
News Sentiment Provider for Nifty Trading Agent
Analyzes news articles and generates sentiment scores for stocks
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
import re
import json

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import nltk
    nltk.download('punkt', quiet=True)
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

from ..utils.logging_utils import get_logger
from ..utils.io_utils import save_json_file, load_json_file

logger = get_logger(__name__)

class NewsSentimentProvider:
    """
    Provides news sentiment analysis for Indian stocks using multiple sentiment analysis techniques.
    """

    def __init__(self, cache_dir: str = "data/news_data", cache_expiry_hours: int = 6):
        """
        Initialize the news sentiment provider

        Args:
            cache_dir: Directory to store cached data
            cache_expiry_hours: Cache expiry time in hours (default: 6 hours for news)
        """
        from pathlib import Path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_expiry_hours = cache_expiry_hours
        self.cache_expiry_seconds = cache_expiry_hours * 3600

        # Initialize sentiment analyzers
        self.vader_analyzer = None
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()

        # Positive and negative word lists for fallback sentiment analysis
        self.positive_words = self._load_sentiment_words('positive')
        self.negative_words = self._load_sentiment_words('negative')

        logger.info(f"NewsSentimentProvider initialized with cache_dir: {cache_dir}")

    def get_recent_news(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """
        Get recent news articles for a stock symbol

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            lookback_days: Number of days to look back

        Returns:
            DataFrame with news articles and metadata
        """
        logger.info(f"Fetching recent news for {symbol}")

        # Check cache first
        cache_key = f"{symbol}_news_{lookback_days}d"
        cached_data = self._load_from_cache(cache_key)
        if cached_data is not None:
            logger.info(f"Loaded news data from cache for {symbol}")
            return pd.DataFrame(cached_data)

        try:
            # For demo purposes, generate synthetic news data
            # In production, this would integrate with news APIs (Moneycontrol, Economic Times, etc.)
            news_data = self._generate_synthetic_news(symbol, lookback_days)

            # Cache the data
            self._save_to_cache(cache_key, news_data.to_dict('records'))

            logger.info(f"Successfully fetched {len(news_data)} news articles for {symbol}")
            return news_data

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")
            return pd.DataFrame()

    def get_sentiment_score(self, symbol: str, lookback_days: int = 30) -> float:
        """
        Get overall sentiment score for a stock symbol

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            Sentiment score between -1 (very negative) and +1 (very positive)
        """
        logger.info(f"Calculating sentiment score for {symbol}")

        # Check cache first
        cache_key = f"{symbol}_sentiment_{lookback_days}d"
        cached_score = self._load_from_cache(cache_key)
        if cached_score is not None:
            logger.info(f"Loaded sentiment score from cache for {symbol}")
            return cached_score

        try:
            # Get news data
            news_data = self.get_recent_news(symbol, lookback_days)

            if news_data.empty:
                return 0.0

            # Calculate sentiment scores for each article
            sentiment_scores = []
            for _, article in news_data.iterrows():
                title = article.get('title', '')
                content = article.get('content', '')

                # Combine title and content for analysis
                text = f"{title} {content}"

                score = self._analyze_sentiment(text)
                sentiment_scores.append(score)

            # Calculate weighted average (more recent articles have higher weight)
            if sentiment_scores:
                weights = [1.0 / (i + 1) for i in range(len(sentiment_scores))]  # Recent articles weighted more
                weights = [w / sum(weights) for w in weights]  # Normalize weights

                overall_sentiment = sum(score * weight for score, weight in zip(sentiment_scores, weights))
            else:
                overall_sentiment = 0.0

            # Cache the result
            self._save_to_cache(cache_key, overall_sentiment)

            logger.info(f"Calculated sentiment score {overall_sentiment:.3f} for {symbol}")
            return overall_sentiment

        except Exception as e:
            logger.error(f"Error calculating sentiment for {symbol}: {e}")
            return 0.0

    def get_sentiment_trend(self, symbol: str, lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get sentiment trend analysis over time

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with sentiment trend analysis
        """
        logger.info(f"Analyzing sentiment trend for {symbol}")

        try:
            news_data = self.get_recent_news(symbol, lookback_days)

            if news_data.empty:
                return {'trend': 'neutral', 'change': 0.0, 'volatility': 0.0}

            # Group by date and calculate daily sentiment
            daily_sentiment = []
            for date in news_data['date'].unique():
                day_news = news_data[news_data['date'] == date]
                day_scores = []

                for _, article in day_news.iterrows():
                    text = f"{article.get('title', '')} {article.get('content', '')}"
                    score = self._analyze_sentiment(text)
                    day_scores.append(score)

                if day_scores:
                    daily_sentiment.append({
                        'date': date,
                        'sentiment': sum(day_scores) / len(day_scores)
                    })

            if len(daily_sentiment) < 2:
                return {'trend': 'neutral', 'change': 0.0, 'volatility': 0.0}

            # Calculate trend
            sentiments = [day['sentiment'] for day in daily_sentiment]
            first_half = sentiments[:len(sentiments)//2]
            second_half = sentiments[len(sentiments)//2:]

            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            change = avg_second - avg_first

            # Determine trend
            if change > 0.1:
                trend = 'improving'
            elif change < -0.1:
                trend = 'deteriorating'
            else:
                trend = 'stable'

            # Calculate sentiment volatility
            import statistics
            volatility = statistics.stdev(sentiments) if len(sentiments) > 1 else 0.0

            return {
                'trend': trend,
                'change': change,
                'volatility': volatility,
                'avg_sentiment': sum(sentiments) / len(sentiments),
                'data_points': len(daily_sentiment)
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment trend for {symbol}: {e}")
            return {'trend': 'neutral', 'change': 0.0, 'volatility': 0.0}

    def get_sector_sentiment(self, sector: str, lookback_days: int = 30) -> float:
        """
        Get sentiment score for an entire sector

        Args:
            sector: Sector name (e.g., 'IT', 'Banking')
            lookback_days: Number of days to analyze

        Returns:
            Sector-wide sentiment score
        """
        logger.info(f"Calculating sector sentiment for {sector}")

        # Define stocks by sector
        sector_stocks = self._get_sector_stocks(sector)

        if not sector_stocks:
            return 0.0

        # Get sentiment for each stock in sector
        sector_sentiments = []
        for stock in sector_stocks:
            try:
                sentiment = self.get_sentiment_score(stock, lookback_days)
                sector_sentiments.append(sentiment)
            except Exception as e:
                logger.warning(f"Error getting sentiment for {stock}: {e}")
                continue

        if not sector_sentiments:
            return 0.0

        # Return average sector sentiment
        return sum(sector_sentiments) / len(sector_sentiments)

    def _analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of a text using multiple techniques

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1 and +1
        """
        if not text or not text.strip():
            return 0.0

        scores = []

        # Method 1: VADER Sentiment Analysis
        if self.vader_analyzer:
            try:
                vader_score = self.vader_analyzer.polarity_scores(text)['compound']
                scores.append(vader_score)
            except Exception as e:
                logger.debug(f"VADER analysis failed: {e}")

        # Method 2: TextBlob Sentiment Analysis
        try:
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            scores.append(textblob_score)
        except Exception as e:
            logger.debug(f"TextBlob analysis failed: {e}")

        # Method 3: Simple word list analysis (fallback)
        word_score = self._simple_sentiment_analysis(text)
        scores.append(word_score)

        # Return average of all available methods
        if scores:
            return sum(scores) / len(scores)
        else:
            return 0.0

    def _simple_sentiment_analysis(self, text: str) -> float:
        """
        Simple sentiment analysis using positive/negative word lists

        Args:
            text: Text to analyze

        Returns:
            Sentiment score between -1 and +1
        """
        if not text:
            return 0.0

        # Clean and tokenize text
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)

        total_words = len(words)

        if total_words == 0:
            return 0.0

        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / total_words

        # Normalize to [-1, 1] range
        return max(-1.0, min(1.0, sentiment_score * 10))  # Multiply by 10 for stronger signal

    def _load_sentiment_words(self, sentiment_type: str) -> set:
        """
        Load sentiment word lists

        Args:
            sentiment_type: 'positive' or 'negative'

        Returns:
            Set of sentiment words
        """
        # Basic sentiment word lists (can be expanded)
        if sentiment_type == 'positive':
            return {
                'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                'outstanding', 'brilliant', 'superb', 'marvelous', 'incredible',
                'profitable', 'profitable', 'growth', 'increase', 'rise', 'gain',
                'bullish', 'strong', 'robust', 'healthy', 'positive', 'upbeat',
                'optimistic', 'confident', 'successful', 'profitable', 'winning'
            }
        elif sentiment_type == 'negative':
            return {
                'bad', 'terrible', 'awful', 'horrible', 'dreadful', 'abysmal',
                'poor', 'weak', 'decline', 'fall', 'drop', 'loss', 'bearish',
                'negative', 'pessimistic', 'worried', 'concerned', 'troubled',
                'disappointing', 'unsatisfactory', 'problematic', 'difficult',
                'challenging', 'struggling', 'declining', 'falling', 'dropping'
            }
        else:
            return set()

    def _get_sector_stocks(self, sector: str) -> List[str]:
        """
        Get list of stocks for a sector

        Args:
            sector: Sector name

        Returns:
            List of stock symbols in the sector
        """
        sector_map = {
            'IT': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
            'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'],
            'Energy': ['RELIANCE.NS'],
            'FMCG': ['HINDUNILVR.NS', 'ITC.NS'],
            'Construction': ['LT.NS']
        }

        return sector_map.get(sector, [])

    def _generate_synthetic_news(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """
        Generate synthetic news data for demonstration

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to generate news for

        Returns:
            DataFrame with synthetic news articles
        """
        import random

        # News templates with different sentiment tones
        news_templates = {
            'positive': [
                "{company} reports strong Q4 results, beating estimates by 15%",
                "{company} shares surge after positive analyst upgrades",
                "{company} announces major new contract worth ₹{amount} crores",
                "Institutional investors increase stake in {company}",
                "{company} expands operations with new facility in {city}",
                "{company} receives industry recognition for innovation excellence"
            ],
            'negative': [
                "{company} faces regulatory scrutiny over compliance issues",
                "{company} reports lower than expected quarterly performance",
                "{company} shares fall amid broader market correction",
                "Analysts downgrade {company} citing margin pressures",
                "{company} announces delay in major project delivery",
                "Key executive departure at {company} raises concerns"
            ],
            'neutral': [
                "{company} maintains dividend payout despite market conditions",
                "{company} announces board meeting to discuss quarterly results",
                "{company} participates in industry conference",
                "{company} updates corporate governance policies",
                "{company} completes employee training program",
                "{company} announces minor organizational changes"
            ]
        }

        # Company name mapping
        company_names = {
            'RELIANCE.NS': 'Reliance Industries',
            'TCS.NS': 'Tata Consultancy Services',
            'HDFCBANK.NS': 'HDFC Bank',
            'ICICIBANK.NS': 'ICICI Bank',
            'INFY.NS': 'Infosys',
            'HINDUNILVR.NS': 'Hindustan Unilever',
            'ITC.NS': 'ITC Limited',
            'KOTAKBANK.NS': 'Kotak Mahindra Bank',
            'LT.NS': 'Larsen & Toubro',
            'AXISBANK.NS': 'Axis Bank'
        }

        company_name = company_names.get(symbol, symbol.replace('.NS', ''))

        news_articles = []

        # Generate news for each day
        current_date = datetime.now()

        for i in range(lookback_days):
            news_date = current_date - timedelta(days=i)

            # Generate 0-3 articles per day
            num_articles = random.randint(0, 3)

            for j in range(num_articles):
                # Choose sentiment type (weighted towards neutral/positive for demo)
                sentiment_weights = {'positive': 0.4, 'neutral': 0.5, 'negative': 0.1}
                sentiment_type = random.choices(
                    list(sentiment_weights.keys()),
                    weights=list(sentiment_weights.values())
                )[0]

                # Select random template
                template = random.choice(news_templates[sentiment_type])

                # Fill in template
                article_text = template.format(
                    company=company_name,
                    amount=random.randint(500, 5000),
                    city=random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad'])
                )

                # Generate title from first part of article
                title = article_text.split(',')[0] + ' - Business News'

                # Add some content
                content = f"{article_text}. This development comes at a time when the company is focusing on expanding its market presence and improving operational efficiency. Industry analysts are closely watching the stock performance following this announcement."

                news_articles.append({
                    'date': news_date.strftime('%Y-%m-%d'),
                    'symbol': symbol,
                    'title': title,
                    'content': content,
                    'sentiment_type': sentiment_type,
                    'source': random.choice(['Economic Times', 'Business Standard', 'Moneycontrol', 'CNBC TV18']),
                    'url': f"https://news.example.com/{symbol.lower().replace('.ns', '')}/{i}_{j}"
                })

        return pd.DataFrame(news_articles)

    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"

        if not cache_file.exists():
            return None

        # Check if cache is expired
        if not self._is_cache_valid(cache_file):
            cache_file.unlink()
            return None

        try:
            return load_json_file(str(cache_file))
        except Exception as e:
            logger.warning(f"Error loading cache for {cache_key}: {e}")
            return None

    def _save_to_cache(self, cache_key: str, data: Any) -> None:
        """Save data to cache"""
        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            save_json_file(data, str(cache_file))
        except Exception as e:
            logger.warning(f"Error saving cache for {cache_key}: {e}")

    def _is_cache_valid(self, cache_file: str) -> bool:
        """Check if cache file is still valid"""
        from ..utils.io_utils import get_file_modification_time
        import time

        mtime = get_file_modification_time(cache_file)
        if mtime is None:
            return False

        current_time = time.time()
        return (current_time - mtime) <= self.cache_expiry_seconds
