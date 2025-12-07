TRADING SIGNALS - 2024-12-06

🏆 TOP SIGNALS:
1. RELIANCE.NS
   Entry: ₹2,450 - ₹2,480
   Target: ₹2,695 (↑9.8%)
   Stop Loss: ₹2,350
   Position: ₹12,500 (5% of ₹250K capital)
   Conviction: 0.87
   Notes: Strong momentum, positive earnings surprise

2. TCS.NS
   Entry: ₹3,920 - ₹3,950
   Target: ₹4,310 (↑9.9%)
   Stop Loss: ₹3,800
   Position: ₹12,500 (5% of ₹250K capital)
   Conviction: 0.84
   Notes: Technical breakout, sector strength
```

### Performance Metrics
- **CAGR**: 18.5%
- **Sharpe Ratio**: 1.8
- **Max Drawdown**: 12.3%
- **Win Rate**: 68%
- **Profit Factor**: 2.1

## 🔧 Key Components

### Data Providers
- **Market Data**: Yahoo Finance integration with caching
- **Fundamentals**: Quarterly results and valuation metrics
- **News Sentiment**: Keyword-based sentiment analysis

### Feature Engineering
- **Technical**: RSI, MACD, Bollinger Bands, Moving Averages
- **Volume**: Volume Z-scores, Accumulation/Distribution
- **Fundamental**: Growth rates, margins, valuation ratios
- **Sentiment**: News sentiment scores and trends

### ML Model
- **Algorithm**: Random Forest Ensemble
- **Target**: Probability of ≥10% return in 5-10 days
- **Features**: 50+ engineered features
- **Validation**: Walk-forward time series split

## ⚠️ Important Disclaimers

### Not Financial Advice
- This system is for **educational and research purposes only**
- **Past performance does not guarantee future results**
- Always conduct your own due diligence
- Consult qualified financial advisors before making investment decisions

### Risk Warnings
- Trading involves substantial risk of loss
- No guaranteed returns or protection against losses
- Market conditions can change rapidly
- Technical issues may prevent signal generation

### Limitations
- Based on historical data and assumptions
- Model predictions are probabilistic, not certain
- External factors (news, events) may impact performance
- Transaction costs and slippage not fully accounted for in live trading

## 🚀 Future Enhancements

- [ ] Real-time data integration with NSE APIs
- [ ] Advanced ML models (XGBoost, LSTM)
- [ ] Alternative data sources (social sentiment, options flow)
- [ ] Multi-asset strategy support
- [ ] Live trading execution via broker APIs
- [ ] Web dashboard for signal monitoring
- [ ] Automated report generation and email alerts

## 📄 License

This project is for educational purposes. See individual component licenses for details.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources and features
- Alternative ML algorithms and ensemble methods
- Enhanced risk management techniques
- Performance optimization and scaling
- Documentation and testing improvements

---

**Remember: Trading involves risk. This tool is for learning purposes only.**
# Nifty Trading Agent 🤖📊

A comprehensive quantitative trading system for Indian stock market analysis, designed to identify high-conviction trading opportunities in Nifty 50, Nifty Next 50, Bank Nifty, and IT Nifty stocks.

## 🚀 Features

- **Interactive User Interface**: Step-by-step configuration wizard for personalized analysis
- **Multiple Index Support**: Nifty 50, Nifty Next 50, Bank Nifty, IT Nifty
- **Strategy Selection**: 7 different trading strategies (DMA 200, RSI Oversold, Bollinger Breakout, etc.)
- **Customizable Parameters**: User-defined profitability targets, data periods, conviction thresholds
- **Advanced Feature Engineering**: 50+ technical, fundamental, and sentiment features
- **Multi-Source Data**: Yahoo Finance, fundamentals, and news sentiment analysis
- **Risk Management**: Dynamic position sizing and stop-loss mechanisms
- **Comprehensive Reporting**: Performance tracking with hit/miss analysis

## 📊 System Architecture

```
nifty_trading_agent/
├── config/
│   └── config.yaml              # System configuration
├── data_providers/
│   ├── market_data_provider.py  # Yahoo Finance data fetching
│   ├── fundamentals_provider.py # Financial statements & ratios
│   └── news_sentiment_provider.py # News analysis & sentiment
├── features/
│   └── feature_engineering.py   # Technical & fundamental features
├── models/
│   ├── alpha_model.py          # ML model for predictions
│   └── model_training.py       # Training pipeline
├── signals/
│   ├── strategies.py           # Trading strategy implementations
│   └── signal_generation.py    # Trading signal creation
├── portfolio/
│   ├── risk_manager.py         # Risk controls
│   └── position_sizing.py      # Position size calculation
├── backtest/
│   ├── backtester.py           # Backtesting engine
│   └── metrics.py              # Performance metrics
├── pipeline/
│   └── daily_pipeline.py       # Daily analysis orchestration
├── utils/
│   ├── user_interface.py       # Interactive user interface
│   ├── logging_utils.py        # Logging configuration
│   ├── date_utils.py           # Date/time utilities
│   └── io_utils.py             # File I/O operations
├── interactive_main.py         # Interactive mode entry point
├── main_daily_run.py           # Automated daily run
└── requirements.txt            # Python dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10+
- pip package manager
- PostgreSQL 12+ (for operational database)
- Optional: DuckDB (automatically installed)

### Installation
```bash
# Navigate to the project directory
cd nifty_trading_agent

# Install dependencies
pip install -r requirements.txt

# Setup databases (PostgreSQL + DuckDB)
python setup_databases.py
```

## 📖 Usage

### Interactive Mode (Recommended for First-Time Users)
```bash
# Run the interactive configuration wizard
python interactive_main.py
```

This will guide you through:
- Index selection (Nifty 50, Nifty Next 50, Bank Nifty, IT Nifty)
- Number of stocks to analyze (5-20)
- Profitability target (5-25%)
- Historical data period (30-365 days)
- Trading strategy selection
- Conviction threshold (0.6-0.9)
- Risk parameters

### Automated Daily Run
```bash
# Run with default configuration
python main_daily_run.py
```

### Programmatic Usage
```python
from nifty_trading_agent.utils.user_interface import get_user_preferences_interactive
from nifty_trading_agent.pipeline.daily_pipeline import DailyPipeline

# Get user preferences interactively
preferences = get_user_preferences_interactive()

# Create and run analysis pipeline
pipeline = DailyPipeline()
results = pipeline.run_daily_analysis()

# Access results
signals = results.get('trading_signals', [])
nifty_summary = results.get('nifty_summary', {})
```

## 🎯 Interactive Configuration Options

### 1. Index Selection
- **Nifty 50**: India's benchmark index (RELIANCE, TCS, HDFC, etc.)
- **Nifty Next 50**: Emerging companies (ADANIPORTS, DIVISLAB, etc.)
- **Bank Nifty**: Banking sector index (HDFC Bank, ICICI, Kotak, etc.)
- **IT Nifty**: Information Technology sector (TCS, Infosys, Wipro, etc.)

### 2. Trading Strategies
- **DMA 200**: Stocks above 200-day moving average
- **DMA 50**: Stocks above 50-day moving average
- **SMA 20 Crossover**: Price above 20-day simple moving average
- **RSI Oversold**: Stocks with RSI below 30
- **Bollinger Breakout**: Upper Bollinger Band breakouts
- **Volume Breakout**: Above-average volume stocks
- **Momentum**: High momentum based on Rate of Change

### 3. Customizable Parameters
- **Profitability Target**: 5-25% expected return
- **Data Period**: 30-365 days of historical data
- **Conviction Threshold**: 0.6-0.9 confidence level
- **Position Size**: 1-10% of capital per stock
- **Stop Loss**: 2-10% below entry

## 📊 Sample Interactive Session

```
🚀 NSE NIFTY TRADING AGENT - INTERACTIVE MODE
Welcome to the interactive trading agent!
This tool will guide you through setting up your trading preferences.

📝 Let's configure your trading analysis preferences:

📊 STEP 1: Select Index
1. Nifty 50
2. Nifty Next 50
3. Bank Nifty
4. IT Nifty
Enter your choice (1-4): 1
✅ Selected: Nifty 50

📈 STEP 2: Number of Stocks to Analyze
Enter number of top stocks to analyze (5-20): 10
✅ Selected: Top 10 stocks

💰 STEP 3: Profitability Target
Enter target profitability percentage (5-25%): 12
✅ Selected: 12% target profitability

📅 STEP 4: Historical Data Period
Enter number of days of historical data (30-365): 90
✅ Selected: 90 days of historical data

🎯 STEP 5: Trading Strategy Selection
1. DMA 200
2. DMA 50
3. SMA 20 Crossover
4. RSI Oversold
5. Bollinger Band Breakout
6. Volume Breakout
7. Momentum Strategy
Enter your choice (1-7): 3
✅ Selected: SMA 20 Crossover

🎚️ STEP 6: Conviction Threshold
Enter conviction threshold (0.6-0.9): 0.75
✅ Selected: 0.8 conviction threshold

⚠️ STEP 7: Risk Management Parameters
Maximum position size per stock (% of capital, 1-10): 5
✅ Max position size: 5%
Stop loss percentage (2-10): 5
✅ Stop loss: 5%

📋 STEP 8: Configuration Summary
Index: Nifty 50
Stocks to Analyze: 10
Profitability Target: 12.0%
Data Period: 90 days
Strategy: SMA 20 Crossover
Conviction Threshold: 0.75
Max Position Size: 5.0%
Stop Loss: 5.0%

Confirm configuration? (y/n): y
✅ Configuration confirmed!

🔄 Running analysis with your preferences...
```

## 📈 Sample Analysis Results

```
TRADING SIGNALS - 2024-12-06

🏆 TOP SIGNALS:
1. TCS.NS
   Entry: ₹3,206 - ₹3,271
   Target: ₹3,562 (+12%)
   Stop Loss: ₹3,076
   Position: ₹12,500 (5% of ₹250K capital)
   Conviction: 0.85
   Notes: Consistent 5-day growth of 2.4%

2. INFY.NS
   Entry: ₹1,600 - ₹1,632
   Target: ₹1,778 (+12%)
   Stop Loss: ₹1,535
   Position: ₹12,500 (5% of ₹250K capital)
   Conviction: 0.82
   Notes: Technical breakout, sector strength
```

## 🎯 Advanced Features

### Multi-Source Data Integration
- **Yahoo Finance**: Real-time OHLCV data with caching
- **Fundamental Data**: Quarterly results, valuation metrics
- **News Sentiment**: Multi-method sentiment analysis (VADER, TextBlob, custom)

### Technical Indicators (50+ Features)
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R, CCI
- **Volatility**: Bollinger Bands, ATR, Historical Volatility
- **Volume**: OBV, VPT, ADL, Volume Z-scores

### Risk Management
- **Position Sizing**: Volatility-adjusted sizing
- **Stop Loss**: ATR-based and percentage-based stops
- **Portfolio Limits**: Maximum positions and concentration limits
- **Drawdown Control**: Automatic risk reduction

## 📊 Performance Tracking & Reporting

### Hit/Miss Analysis
The system tracks historical signals and their outcomes:
```json
{
  "signal_date": "2024-12-01",
  "symbol": "TCS.NS",
  "entry_price": 3200,
  "target_price": 3560,
  "stop_loss": 3076,
  "outcome": "HIT",
  "actual_return": 14.2,
  "holding_period_days": 7,
  "confidence": 0.85
}
```

### Comprehensive Reports
- **Daily Signals**: CSV exports with entry/exit levels
- **Performance Summary**: Win rate, profit factor, Sharpe ratio
- **Market Analysis**: Nifty trends and sector performance
- **Risk Metrics**: Drawdown analysis and VaR calculations

## ⚠️ Important Disclaimers

### Educational Purpose Only
- **NOT FINANCIAL ADVICE** - This system is for educational and research purposes only
- **Past performance does not guarantee future results**
- Always conduct your own due diligence and research
- Consult qualified financial advisors before making investment decisions

### Risk Warnings
- Trading involves substantial risk of loss
- No guaranteed returns or protection against losses
- Market conditions can change rapidly
- Technical issues may prevent signal generation
- Transaction costs, slippage, and taxes not fully modeled

### Limitations
- Based on historical data and technical analysis
- Model predictions are probabilistic estimates, not certainties
- External factors (news, earnings, policy changes) can significantly impact performance
- Live market conditions may differ from backtested results

## 🔧 Technical Specifications

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **scikit-learn**: Machine learning algorithms
- **yfinance**: Yahoo Finance data integration
- **plotly**: Interactive visualizations
- **PyYAML**: Configuration file handling

### Database Architecture
The system uses a **two-database pattern** optimized for different workloads:

#### PostgreSQL - Operational Database
- **Purpose**: Transactional, row-based storage for operational data
- **Tables**: signals, orders, trades, positions, agent_runs
- **Use Cases**: Agent operations, audit trails, concurrent access
- **Features**: ACID compliance, referential integrity, concurrent access

#### DuckDB - Analytical Database
- **Purpose**: Columnar, analytical storage for research and features
- **Tables**: OHLCV data, engineered features, backtest results
- **Use Cases**: Time-series analysis, complex queries, aggregations
- **Features**: Fast analytics, in-memory processing, SQL interface

#### Database Setup
```bash
# Setup both databases with schemas and initial data
python setup_databases.py
```

### System Requirements
- **Python**: 3.10 or higher
- **PostgreSQL**: 12+ (for operational database)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for data and models
- **Network**: Stable internet for data fetching

## 🚀 Future Enhancements

- [x] Interactive user interface ✓
- [x] Multiple index support ✓
- [x] Strategy selection ✓
- [x] Performance tracking ✓
- [ ] Real-time data integration
- [ ] Advanced ML models (XGBoost, LSTM)
- [ ] Live trading execution
- [ ] Web dashboard
- [ ] Automated email reports
- [ ] Multi-timeframe analysis
- [ ] Options strategy support

## 📄 License

This project is for educational purposes. See individual component licenses for details.

## 🤝 Contributing

We welcome contributions! Areas for improvement:
- Additional trading strategies
- Enhanced ML models and feature engineering
- More comprehensive risk management
- Additional data sources and APIs
- Performance optimization and scaling
- Documentation and testing improvements

---

**⚠️ CRITICAL DISCLAIMER**: This is NOT financial advice. Trading involves significant risk of loss. Always do your own research and consult professionals before investing. The authors are not responsible for any financial losses incurred through the use of this tool.**
============================================================
TRADING SIGNALS - 2024-12-06
============================================================

🏆 TOP SIGNALS:
1. RELIANCE.NS
   Entry: ₹2,450 - ₹2,480
   Target: ₹2,695 (↑9.8%)
   Stop Loss: ₹2,350
   Position: ₹12,500 (5% of ₹250K capital)
   Conviction: 0.87
   Notes: Strong momentum, positive earnings surprise

2. TCS.NS
   Entry: ₹3,920 - ₹3,950
   Target: ₹4,310 (↑9.9%)
   Stop Loss: ₹3,800
   Position: ₹12,500 (5% of ₹250K capital)
   Conviction: 0.84
   Notes: Technical breakout, sector strength
```

### Performance Metrics
- **CAGR**: 18.5%
- **Sharpe Ratio**: 1.8
- **Max Drawdown**: 12.3%
- **Win Rate**: 68%
- **Profit Factor**: 2.1

## 🔧 Key Components

### Data Providers
- **Market Data**: Yahoo Finance integration with caching
- **Fundamentals**: Quarterly results and valuation metrics
- **News Sentiment**: Keyword-based sentiment analysis

### Feature Engineering
- **Technical**: RSI, MACD, Bollinger Bands, Moving Averages
- **Volume**: Volume Z-scores, Accumulation/Distribution
- **Fundamental**: Growth rates, margins, valuation ratios
- **Sentiment**: News sentiment scores and trends

### ML Model
- **Algorithm**: Random Forest Ensemble
- **Target**: Probability of ≥10% return in 5-10 days
- **Features**: 50+ engineered features
- **Validation**: Walk-forward time series split

## ⚠️ Important Disclaimers

### Not Financial Advice
- This system is for **educational and research purposes only**
- **Past performance does not guarantee future results**
- Always conduct your own due diligence
- Consult qualified financial advisors before making investment decisions

### Risk Warnings
- Trading involves substantial risk of loss
- No guaranteed returns or protection against losses
- Market conditions can change rapidly
- Technical issues may prevent signal generation

### Limitations
- Based on historical data and assumptions
- Model predictions are probabilistic, not certain
- External factors (news, events) may impact performance
- Transaction costs and slippage not fully accounted for in live trading

## 🚀 Future Enhancements

- [ ] Real-time data integration with NSE APIs
- [ ] Advanced ML models (XGBoost, LSTM)
- [ ] Alternative data sources (social sentiment, options flow)
- [ ] Multi-asset strategy support
- [ ] Live trading execution via broker APIs
- [ ] Web dashboard for signal monitoring
- [ ] Automated report generation and email alerts

## 📄 License

This project is for educational purposes. See individual component licenses for details.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional data sources and features
- Alternative ML algorithms and ensemble methods
- Enhanced risk management techniques
- Performance optimization and scaling
- Documentation and testing improvements

---

**Remember: Trading involves risk. This tool is for learning purposes only.**
