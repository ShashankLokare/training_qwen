# Indian Market Trading Agent 🤖📈

An AI-powered trading agent for analyzing the Indian stock market, specifically designed to identify equities with consistent upward momentum and provide comprehensive market insights.

## 🚀 Features

- **Nifty50 Data Fetching**: Real-time and historical Nifty50 opening/closing values
- **Consistent Gainers Analysis**: Identify stocks with steady 2-5% daily gains over 5 days
- **Market Momentum Ranking**: Rank stocks by momentum scores
- **Comprehensive Market Analysis**: Complete market sentiment and recommendations
- **Command Line Interface**: Easy-to-use CLI for various analysis modes
- **JSON Export**: Export results for further processing

## 📊 What It Does

### 1. Nifty50 Analysis
- Fetches current and historical Nifty50 levels
- Calculates daily changes, period highs/lows, and volume analysis
- Provides market performance metrics

### 2. Stock Momentum Analysis
- Analyzes Nifty50 stocks for consistent daily gains (2-5% range)
- Requires at least 60% of days to show gains in the specified range
- Filters for stocks with at least 3 consistent gain days

### 3. Market Sentiment
- Bullish/Neutral/Bearish market sentiment based on consistent gainers ratio
- Investment recommendations based on market conditions

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Internet connection for market data

### Setup
```bash
# Clone or navigate to the trading_agent directory
cd trading_agent

# Install dependencies
pip install -r requirements.txt

# Run the demo
python demo.py
```

## 📖 Usage

### Command Line Interface

```bash
# Basic usage - comprehensive analysis
python -m trading_agent.cli

# Get Nifty50 summary only
python -m trading_agent.cli --nifty

# Find consistent gainers
python -m trading_agent.cli --gainers --min-gain 2.5 --max-gain 4.0

# Get momentum ranking
python -m trading_agent.cli --momentum

# Comprehensive analysis with custom parameters
python -m trading_agent.cli --comprehensive --days 7 --max-stocks 30

# Export results as JSON
python -m trading_agent.cli --output json --comprehensive
```

### Python API Usage

```python
from trading_agent import IndianTradingAgent

# Initialize agent
agent = IndianTradingAgent()

# Get Nifty50 summary
nifty_data = agent.get_nifty50_summary(days=30)
print(f"Nifty50 Level: {nifty_data['current_level']}")

# Find consistent gainers (2-5% over 5 days)
gainers = agent.find_consistent_gainers(min_gain=2.0, max_gain=5.0, days=5)
print(f"Found {gainers['consistent_gainers_count']} consistent gainers")

# Get comprehensive analysis
analysis = agent.get_comprehensive_market_analysis(days=5)
print(f"Market Sentiment: {analysis['market_sentiment']}")
```

## 📈 Analysis Methodology

### Consistent Gainers Criteria
A stock is considered a "consistent gainer" if:
- At least 60% of the analysis period days show gains between 2-5%
- At least 3 days meet the gain criteria
- The stock demonstrates steady upward momentum without extreme volatility

### Momentum Score Calculation
```
Momentum Score = (Cumulative Return %) / (Daily Volatility %)
```
- Higher scores indicate better risk-adjusted performance
- Balances returns against price volatility

### Market Sentiment Logic
- **Bullish (High Confidence)**: >30% of stocks show consistent gains
- **Moderately Bullish (Medium)**: 15-30% of stocks show consistent gains
- **Neutral (Low)**: 5-15% of stocks show consistent gains
- **Bearish (High)**: <5% of stocks show consistent gains

## 📁 Project Structure

```
trading_agent/
├── __init__.py              # Package initialization
├── trading_agent.py         # Main agent class
├── data_fetcher.py          # Market data fetching utilities
├── stock_analyzer.py        # Stock analysis algorithms
├── cli.py                   # Command line interface
├── demo.py                  # Demo script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎯 Key Components

### IndianTradingAgent
Main interface class that orchestrates all analysis functions.

### IndianMarketDataFetcher
Handles data retrieval from Yahoo Finance for Indian market indices and stocks.

### StockAnalyzer
Implements momentum analysis, consistency checking, and ranking algorithms.

## ⚠️ Important Notes

### Data Source
- Uses Yahoo Finance (yfinance library) for market data
- NSE symbols use `.NS` suffix (e.g., `RELIANCE.NS`)
- Data availability depends on market hours and Yahoo Finance service

### Limitations
- Analysis is based on historical data and technical indicators
- Past performance doesn't guarantee future results
- Not financial advice - use for educational purposes only
- Market data may have delays or inaccuracies

### Risk Disclaimer
This tool is for educational and research purposes only. Always consult with qualified financial advisors before making investment decisions. The authors are not responsible for any financial losses incurred through the use of this tool.

## 🔧 Configuration

### Custom Stock Lists
Modify `NIFTY50_STOCKS` in `data_fetcher.py` to analyze different stocks:

```python
NIFTY50_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS",  # Add your preferred stocks
    # ... more symbols
]
```

### Analysis Parameters
Adjust analysis criteria in the CLI or API calls:
- `min_gain` / `max_gain`: Daily gain percentage range
- `days`: Analysis period
- `max_stocks`: Limit analysis to top N stocks for performance

## 📊 Sample Output

```
============================================================
NIFTY50 MARKET SUMMARY
============================================================
Current Level: 24,156.80
Daily Change: +1.23%
Period High: 24,234.56
Period Low: 23,890.12
Period Return: +2.45%
Average Volume: 345,678,901
Data Points: 30
Analysis Period: 30 days

============================================================
CONSISTENT GAINERS ANALYSIS
============================================================
Analysis Criteria: 2.0-5.0% daily gains over 5 days
Stocks Analyzed: 15
Consistent Gainers Found: 4

🏆 TOP 4 PERFORMERS:
 1. RELIANCE.NS        ₹2,543.60  (+3.2%)  4/5 days  +12.3%
 2. TCS.NS            ₹3,421.80  (+2.8%)  4/5 days  +11.7%
 3. INFY.NS           ₹1,856.40  (+3.1%)  3/5 days   +9.8%
 4. HDFCBANK.NS       ₹1,623.20  (+2.5%)  3/5 days   +8.2%

============================================================
MARKET SENTIMENT & RECOMMENDATIONS
============================================================
📊 Sentiment: MODERATELY BULLISH (Confidence: MEDIUM, 4/15 stocks showing consistent gains)

💡 RECOMMENDATIONS:
1. Consider monitoring these top performers: RELIANCE.NS, TCS.NS, INFY.NS
2. Market shows bullish momentum - consider increasing exposure to consistent gainers
3. Always perform thorough due diligence before making investment decisions
4. Consider consulting with a financial advisor for personalized advice
```

## 🚀 Future Enhancements

- [ ] Integration with NSE API for real-time data
- [ ] Technical indicators (RSI, MACD, Bollinger Bands)
- [ ] Portfolio optimization suggestions
- [ ] Risk assessment metrics
- [ ] Alert system for price movements
- [ ] Web dashboard interface
- [ ] Machine learning predictions

## 📄 License

This project is for educational purposes only. See individual component licenses for details.

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the demo script: `python demo.py`
2. Review the CLI help: `python -m trading_agent.cli --help`
3. Check Yahoo Finance data availability

---

**⚠️ Disclaimer**: This is not financial advice. Use at your own risk. Always do your own research and consult professionals before investing.**
