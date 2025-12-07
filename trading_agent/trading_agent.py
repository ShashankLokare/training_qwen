"""
Indian Market Trading Agent
Main agent that fetches Nifty50 data and analyzes stocks for consistent upward momentum
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime

# Import local modules
from data_fetcher import IndianMarketDataFetcher
from stock_analyzer import StockAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndianTradingAgent:
    """
    AI-powered trading agent for Indian stock market analysis

    Features:
    - Fetches Nifty50 opening/closing values
    - Identifies stocks with consistent 2-5% gains over 5 days
    - Provides comprehensive market analysis
    """

    def __init__(self):
        """Initialize the trading agent"""
        self.data_fetcher = IndianMarketDataFetcher()
        self.analyzer = StockAnalyzer()
        logger.info("Indian Trading Agent initialized")

    def get_nifty50_summary(self, days: int = 30) -> Dict:
        """
        Get comprehensive Nifty50 market summary

        Args:
            days: Number of days of historical data to analyze

        Returns:
            Dictionary with Nifty50 analysis
        """
        logger.info(f"Fetching Nifty50 data for the last {days} days")

        # Fetch Nifty50 data
        nifty_data = self.data_fetcher.fetch_nifty50_data(days)

        if nifty_data.empty:
            return {
                'error': 'Unable to fetch Nifty50 data',
                'current_level': self.data_fetcher.get_current_nifty_level()
            }

        # Calculate key metrics
        current_level = nifty_data['Close'].iloc[-1]
        previous_close = nifty_data['Close'].iloc[-2] if len(nifty_data) > 1 else current_level

        # Performance metrics
        daily_change = ((current_level - previous_close) / previous_close) * 100

        # Period performance
        period_high = nifty_data['High'].max()
        period_low = nifty_data['Low'].min()
        period_return = ((current_level - nifty_data['Open'].iloc[0]) / nifty_data['Open'].iloc[0]) * 100

        # Volume analysis
        avg_volume = nifty_data['Volume'].mean()

        summary = {
            'current_level': current_level,
            'daily_change_pct': daily_change,
            'period_high': period_high,
            'period_low': period_low,
            'period_return_pct': period_return,
            'average_volume': avg_volume,
            'data_points': len(nifty_data),
            'analysis_period_days': days,
            'last_updated': datetime.now().isoformat()
        }

        logger.info(".2f")
        return summary

    def find_consistent_gainers(self, min_gain: float = 2.0, max_gain: float = 5.0,
                               days: int = 5, max_stocks: int = 50) -> Dict:
        """
        Find stocks with consistent gains in the specified range

        Args:
            min_gain: Minimum daily gain percentage
            max_gain: Maximum daily gain percentage
            days: Number of days to analyze
            max_stocks: Maximum number of stocks to analyze

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing stocks for consistent {min_gain}-{max_gain}% gains over {days} days")

        # Get list of stocks to analyze
        stock_symbols = self.data_fetcher.get_nifty50_stocks_list()
        stock_symbols = stock_symbols[:max_stocks]  # Limit for performance

        # Fetch data for all stocks
        stocks_data = self.data_fetcher.fetch_multiple_stocks(stock_symbols, days + 5)  # Extra days for analysis

        if not stocks_data:
            return {'error': 'Unable to fetch stock data', 'consistent_gainers': []}

        # Find consistent gainers
        consistent_gainers = self.analyzer.find_consistent_gainers(
            stocks_data, min_gain, max_gain, days
        )

        # Get detailed summaries for top performers
        top_performers = consistent_gainers[:10]  # Top 10
        detailed_analysis = []

        for stock in top_performers:
            symbol = stock['symbol']
            if symbol in stocks_data:
                summary = self.analyzer.get_stock_summary(symbol, stocks_data[symbol])
                detailed_analysis.append(summary)

        result = {
            'analysis_criteria': {
                'min_daily_gain_pct': min_gain,
                'max_daily_gain_pct': max_gain,
                'analysis_period_days': days,
                'stocks_analyzed': len(stocks_data)
            },
            'consistent_gainers_count': len(consistent_gainers),
            'top_performers': detailed_analysis,
            'all_consistent_gainers': consistent_gainers,
            'analysis_timestamp': datetime.now().isoformat()
        }

        logger.info(f"Found {len(consistent_gainers)} stocks meeting consistency criteria")
        return result

    def get_market_momentum_ranking(self, days: int = 5) -> Dict:
        """
        Get ranking of all analyzed stocks by momentum

        Args:
            days: Number of days to analyze momentum

        Returns:
            Dictionary with momentum rankings
        """
        logger.info(f"Ranking stocks by momentum over {days} days")

        # Get stock data
        stock_symbols = self.data_fetcher.get_nifty50_stocks_list()
        stocks_data = self.data_fetcher.fetch_multiple_stocks(stock_symbols, days + 5)

        if not stocks_data:
            return {'error': 'Unable to fetch stock data', 'momentum_ranking': []}

        # Rank by momentum
        momentum_ranking = self.analyzer.rank_stocks_by_momentum(stocks_data, days)

        # Get top 20
        top_momentum = momentum_ranking[:20]

        result = {
            'analysis_period_days': days,
            'stocks_analyzed': len(stocks_data),
            'top_momentum_stocks': top_momentum,
            'momentum_timestamp': datetime.now().isoformat()
        }

        logger.info(f"Generated momentum ranking for {len(momentum_ranking)} stocks")
        return result

    def get_comprehensive_market_analysis(self, days: int = 5) -> Dict:
        """
        Get comprehensive market analysis including Nifty50 and stock momentum

        Args:
            days: Number of days for analysis

        Returns:
            Complete market analysis report
        """
        logger.info("Generating comprehensive market analysis")

        # Get Nifty50 summary
        nifty_summary = self.get_nifty50_summary(days * 2)  # More data for Nifty

        # Find consistent gainers
        gainers_analysis = self.find_consistent_gainers(days=days)

        # Get momentum ranking
        momentum_analysis = self.get_market_momentum_ranking(days)

        # Market sentiment analysis
        sentiment = self._analyze_market_sentiment(gainers_analysis, momentum_analysis)

        comprehensive_report = {
            'report_title': 'Indian Market Daily Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'analysis_period_days': days,
            'nifty50_summary': nifty_summary,
            'consistent_gainers': gainers_analysis,
            'momentum_ranking': momentum_analysis,
            'market_sentiment': sentiment,
            'recommendations': self._generate_recommendations(gainers_analysis, sentiment)
        }

        logger.info("Comprehensive market analysis completed")
        return comprehensive_report

    def _analyze_market_sentiment(self, gainers_analysis: Dict, momentum_analysis: Dict) -> str:
        """
        Analyze overall market sentiment based on analysis results

        Args:
            gainers_analysis: Results from consistent gainers analysis
            momentum_analysis: Results from momentum analysis

        Returns:
            Market sentiment assessment
        """
        gainers_count = gainers_analysis.get('consistent_gainers_count', 0)
        total_stocks = gainers_analysis.get('analysis_criteria', {}).get('stocks_analyzed', 1)

        gainers_ratio = gainers_count / total_stocks

        if gainers_ratio > 0.3:
            sentiment = "BULLISH"
            confidence = "HIGH"
        elif gainers_ratio > 0.15:
            sentiment = "MODERATELY BULLISH"
            confidence = "MEDIUM"
        elif gainers_ratio > 0.05:
            sentiment = "NEUTRAL"
            confidence = "LOW"
        else:
            sentiment = "BEARISH"
            confidence = "HIGH"

        return f"{sentiment} (Confidence: {confidence}, {gainers_count}/{total_stocks} stocks showing consistent gains)"

    def _generate_recommendations(self, gainers_analysis: Dict, sentiment: str) -> List[str]:
        """
        Generate investment recommendations based on analysis

        Args:
            gainers_analysis: Analysis results
            sentiment: Market sentiment

        Returns:
            List of recommendations
        """
        recommendations = []

        top_performers = gainers_analysis.get('top_performers', [])

        if top_performers:
            recommendations.append(f"Consider monitoring these top performers: {', '.join([s['symbol'] for s in top_performers[:3]])}")

        if "BULLISH" in sentiment:
            recommendations.append("Market shows bullish momentum - consider increasing exposure to consistent gainers")
        elif "BEARISH" in sentiment:
            recommendations.append("Market shows bearish signals - focus on risk management and defensive stocks")
        else:
            recommendations.append("Market is neutral - focus on high-quality stocks with consistent performance")

        recommendations.append("Always perform thorough due diligence before making investment decisions")
        recommendations.append("Consider consulting with a financial advisor for personalized advice")

        return recommendations

    def run_daily_analysis(self) -> Dict:
        """
        Run complete daily market analysis

        Returns:
            Complete daily analysis report
        """
        return self.get_comprehensive_market_analysis(days=5)
