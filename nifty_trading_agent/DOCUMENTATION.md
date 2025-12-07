# NSE Nifty Trading Agent - Complete Documentation

## 🎯 **Overview**

The **NSE Nifty Trading Agent** is a comprehensive quantitative trading system designed for Indian stock market analysis. It combines advanced technical analysis, machine learning models, and multi-source data integration to generate high-conviction trading signals for Nifty 50, Nifty Next 50, Bank Nifty, and IT Nifty stocks.

### **Key Features**
- **Interactive User Interface** - 8-step guided configuration wizard
- **Multiple Index Support** - 4 major Indian market indices
- **7 Trading Strategies** - DMA 200, RSI, Bollinger Breakout, Momentum, etc.
- **50+ Technical Features** - RSI, MACD, Bollinger Bands, Moving Averages, etc.
- **Production Database** - PostgreSQL + DuckDB architecture
- **Agentic AI Ready** - Clean function interfaces for automated agents
- **Risk Management** - Dynamic position sizing and stop-loss mechanisms
- **Performance Tracking** - Comprehensive backtesting and analysis

---

## 🏗️ **System Architecture**

### **Core Components**

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
├── signals/
│   ├── strategies.py           # Trading strategy implementations
│   └── signal_generation.py    # Trading signal creation
├── pipeline/
│   └── daily_pipeline.py       # Daily analysis orchestration
├── utils/
│   ├── user_interface.py       # Interactive configuration wizard
│   ├── db_postgres.py         # PostgreSQL connection management
│   ├── postgres_tools.py      # PostgreSQL high-level functions
│   ├── db_duckdb.py           # DuckDB connection management
│   └── duckdb_tools.py        # DuckDB high-level functions
├── interactive_main.py        # Interactive mode entry point
├── main_daily_run.py          # Automated daily run
└── setup_databases.py         # Database initialization
```

### **Database Architecture**

The system uses a **two-database pattern** optimized for different workloads:

#### **PostgreSQL - Operational Database**
- **Purpose**: Transactional, row-based storage for operational data
- **Use Cases**: Agent operations, audit trails, concurrent access
- **Tables**:
  - `symbols` - Stock universe with metadata
  - `daily_signals` - Trading signals with conviction scores
  - `orders` - Order management system
  - `trades` - Trade execution tracking
  - `positions` - Position management
  - `agent_runs` - Agent execution logging
  - `config_overrides` - Dynamic configuration

#### **DuckDB - Analytical Database**
- **Purpose**: Columnar, analytical storage for research and features
- **Use Cases**: Time-series analysis, complex queries, aggregations
- **Tables**:
  - `ohlcv_nifty` - Historical OHLCV data
  - `features_nifty` - Engineered features (50+ indicators)
  - `backtest_results` - Strategy performance curves

---

## 🚀 **Installation & Setup**

### **Prerequisites**
- **Python**: 3.10 or higher
- **PostgreSQL**: 12+ (optional - system works in DuckDB-only mode)
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB for data and models

### **Step 1: Clone & Install**
```bash
# Navigate to the project directory
cd nifty_trading_agent

# Install dependencies
pip install -r requirements.txt

# Install database dependencies
pip install psycopg2-binary duckdb
```

### **Step 2: Database Setup**
```bash
# Setup databases (PostgreSQL + DuckDB)
python setup_databases.py
```

This creates:
- PostgreSQL schema (if available) with all operational tables
- DuckDB analytical database with OHLCV, features, and backtest tables
- Initial symbol data for 20 Nifty stocks

### **Step 3: Configuration**
Edit `config/config.yaml` to customize:
- Database connection settings
- Universe definition and parameters
- Risk management settings
- Model hyperparameters

---

## 🎯 **Usage Guide**

### **Interactive Mode (Recommended)**

Run the interactive configuration wizard:
```bash
python interactive_main.py
```

#### **8-Step Configuration Process**

1. **Index Selection**
   - Nifty 50: India's benchmark index (RELIANCE, TCS, HDFC, etc.)
   - Nifty Next 50: Emerging companies (ADANIPORTS, DIVISLAB, etc.)
   - Bank Nifty: Banking sector index
   - IT Nifty: Information Technology sector

2. **Stock Universe**
   - Number of stocks to analyze (5-20)

3. **Profitability Target**
   - Expected return percentage (5-25%)

4. **Data Period**
   - Historical data days (30-365)

5. **Trading Strategy**
   - DMA 200: Stocks above 200-day moving average
   - DMA 50: Stocks above 50-day moving average
   - SMA 20 Crossover: Price above 20-day simple moving average
   - RSI Oversold: Stocks with RSI below 30
   - Bollinger Breakout: Upper Bollinger Band breakouts
   - Volume Breakout: Above-average volume stocks
   - Momentum: High momentum based on Rate of Change

6. **Conviction Threshold**
   - Confidence level (0.6-0.9)

7. **Risk Parameters**
   - Max position size per stock (1-10% of capital)
   - Stop loss percentage (2-10%)

8. **Confirmation**
   - Review and confirm configuration

### **Automated Daily Run**
```bash
# Run with default configuration
python main_daily_run.py
```

### **Programmatic Usage**
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

---

## 📊 **Trading Strategies**

### **1. DMA 200 Strategy**
- **Logic**: Stocks trading above their 200-day moving average
- **Signal**: Buy when price > 200-day MA
- **Rationale**: Long-term uptrend confirmation
- **Parameters**: MA period = 200 days

### **2. DMA 50 Strategy**
- **Logic**: Stocks trading above their 50-day moving average
- **Signal**: Buy when price > 50-day MA
- **Rationale**: Medium-term trend strength
- **Parameters**: MA period = 50 days

### **3. SMA 20 Crossover Strategy**
- **Logic**: Price above 20-day simple moving average
- **Signal**: Buy on crossover above SMA 20
- **Rationale**: Short-term momentum confirmation
- **Parameters**: MA period = 20 days

### **4. RSI Oversold Strategy**
- **Logic**: RSI below 30 (oversold condition)
- **Signal**: Buy when RSI < 30
- **Rationale**: Potential reversal from oversold levels
- **Parameters**: RSI period = 14, threshold = 30

### **5. Bollinger Band Breakout Strategy**
- **Logic**: Price breaks above upper Bollinger Band
- **Signal**: Buy on breakout above upper band
- **Rationale**: Strong momentum and volatility expansion
- **Parameters**: BB period = 20, standard deviation = 2

### **6. Volume Breakout Strategy**
- **Logic**: Volume significantly above average
- **Signal**: Buy when volume > 1.5x 20-day average
- **Rationale**: Institutional interest and accumulation
- **Parameters**: Volume multiplier = 1.5

### **7. Momentum Strategy**
- **Logic**: High rate of change over recent period
- **Signal**: Buy when ROC > threshold
- **Rationale**: Strong price momentum continuation
- **Parameters**: ROC period = 20, threshold = 5%

---

## 🔧 **Technical Features (50+ Indicators)**

### **Trend Indicators**
- Simple Moving Averages (5, 10, 20, 50, 100, 200 periods)
- Exponential Moving Averages (5, 10, 20, 50 periods)
- MACD (Moving Average Convergence Divergence)
- ADX (Average Directional Index)

### **Momentum Indicators**
- RSI (Relative Strength Index, 14-period)
- Stochastic Oscillator (%K, %D)
- Williams %R
- CCI (Commodity Channel Index)

### **Volatility Indicators**
- Bollinger Bands (20-period, 2 std dev)
- ATR (Average True Range, 14-period)
- Historical Volatility
- Bollinger Band Width

### **Volume Indicators**
- OBV (On Balance Volume)
- Volume Rate of Change
- Volume Z-Score (20-period)
- Accumulation/Distribution Line

### **Price Action**
- Returns (1-day, 3-day, 5-day, 10-day, 20-day)
- Price to Moving Average ratios
- Support/Resistance levels
- Pivot Points

### **Fundamental Features**
- Quarterly EPS Growth
- Quarterly Revenue Growth
- PE Ratio (Price to Earnings)
- PB Ratio (Price to Book)

### **Sentiment Features**
- News sentiment scores (short-term)
- News sentiment scores (medium-term)
- Sentiment trend analysis

---

## 🎯 **Signal Generation Process**

### **1. Data Collection**
- Fetch OHLCV data from Yahoo Finance
- Load fundamental data and ratios
- Gather news sentiment data
- Cache data for performance

### **2. Feature Engineering**
- Compute 50+ technical indicators
- Calculate fundamental ratios
- Generate sentiment scores
- Handle missing data and outliers

### **3. Strategy Evaluation**
- Apply selected trading strategy
- Calculate conviction scores
- Generate entry/exit levels
- Apply risk management rules

### **4. Signal Filtering**
- Apply conviction threshold (0.6-0.9)
- Limit signals per day (max 5)
- Ensure diversification
- Validate signal quality

### **5. Position Sizing**
- Calculate position size based on risk
- Apply volatility adjustments
- Consider correlation constraints
- Set stop-loss levels

### **6. Database Storage**
- Store signals in PostgreSQL
- Log agent run in audit trail
- Cache analytical data in DuckDB

---

## 📈 **Database Operations**

### **PostgreSQL Operations (Agent Tools)**

```python
from utils.postgres_tools import (
    record_signals, fetch_signals_for_date, create_order_from_signal,
    get_open_positions, log_agent_run, update_agent_run
)

# Record trading signals
signal_ids = record_signals([
    {
        'symbol': 'TCS.NS',
        'signal_date': datetime.now().date(),
        'entry_low': 3200,
        'entry_high': 3250,
        'target_price': 3560,
        'stop_loss': 3076,
        'position_size_pct': 0.05,
        'conviction': 0.85,
        'notes': 'Strong momentum signal',
        'model_version': 'v1.0'
    }
])

# Create order from signal
order_id = create_order_from_signal(signal_ids[0], 'BUY', 100, 3225.0)

# Log agent execution
run_id = log_agent_run('SignalAgent', 'daily_pipeline', 'SUCCESS', {
    'signals_generated': 2,
    'processing_time': 45.2
})

# Get open positions
positions = get_open_positions()
```

### **DuckDB Operations (Analytics Tools)**

```python
from utils.duckdb_tools import (
    store_ohlcv, load_ohlcv, store_features, load_features_for_training,
    store_backtest_results, load_backtest_results
)

# Store OHLCV data
success = store_ohlcv(df_ohlcv_data)

# Load historical data
ohlcv_data = load_ohlcv(['TCS.NS', 'INFY.NS'], '2024-01-01', '2024-12-31')

# Store engineered features
success = store_features(df_features)

# Load features for training
training_data = load_features_for_training('2020-01-01', '2024-01-01')

# Store backtest results
success = store_backtest_results('dma200_strategy', 'run_001', df_equity_curve)

# Load backtest results
backtest_results = load_backtest_results('dma200_strategy')
```

---

## 📊 **Output & Reporting**

### **Interactive Results**
```
================================================================================
📊 ANALYSIS RESULTS
================================================================================
📈 Index: Nifty 50
🎯 Strategy: SMA 20 Crossover
💰 Target Return: 12.0%

🎯 TRADING SIGNALS GENERATED:
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

### **File Outputs**
- **JSON Report**: `reports/interactive_analysis_YYYYMMDD_HHMMSS.json`
- **CSV Signals**: `reports/signals_YYYYMMDD_HHMMSS.csv`
- **Logs**: `logs/trading_agent.log`

### **Database Storage**
- **Signals**: Stored in PostgreSQL `daily_signals` table
- **Audit Trail**: Agent runs logged in `agent_runs` table
- **Analytics**: OHLCV and features cached in DuckDB

---

## 🎲 **Agentic AI Integration**

### **Specialized Agents**

#### **DataAgent**
- Fetches and stores market data
- Maintains data quality and freshness
- Handles API rate limits and errors

```python
# DataAgent workflow
data_agent = DataAgent()
data_agent.fetch_ohlcv(symbols, start_date, end_date)
data_agent.update_fundamentals()
data_agent.refresh_cache()
```

#### **FeatureAgent**
- Computes technical indicators
- Engineers fundamental features
- Maintains feature consistency

```python
# FeatureAgent workflow
feature_agent = FeatureAgent()
feature_agent.compute_technical_features(symbols)
feature_agent.compute_fundamental_features()
feature_agent.validate_features()
```

#### **SignalAgent**
- Generates trading signals
- Applies conviction thresholds
- Manages signal lifecycle

```python
# SignalAgent workflow
signal_agent = SignalAgent()
signals = signal_agent.generate_signals(strategy_config)
signal_ids = signal_agent.record_signals(signals)
signal_agent.update_signal_status(signal_ids)
```

#### **RiskAgent**
- Monitors position sizes
- Enforces risk limits
- Manages stop-loss orders

```python
# RiskAgent workflow
risk_agent = RiskAgent()
risk_agent.validate_positions()
risk_agent.adjust_stop_losses()
risk_agent.check_risk_limits()
```

#### **ExecutionAgent**
- Places orders through broker APIs
- Monitors order execution
- Updates position tracking

```python
# ExecutionAgent workflow
execution_agent = ExecutionAgent()
order_ids = execution_agent.place_orders(signals)
execution_agent.monitor_orders(order_ids)
execution_agent.update_positions()
```

#### **SupervisorAgent**
- Coordinates agent activities
- Monitors system health
- Handles error recovery

```python
# SupervisorAgent workflow
supervisor = SupervisorAgent()
supervisor.orchestrate_daily_run()
supervisor.monitor_agent_health()
supervisor.handle_failures()
```

### **Agent Communication**
- **Message Queue**: Agents communicate via database tables
- **Status Updates**: Real-time status tracking in `agent_runs`
- **Error Handling**: Centralized error logging and recovery
- **Configuration**: Dynamic config updates via `config_overrides`

---

## ⚠️ **Important Disclaimers**

### **Educational Purpose Only**
- **NOT FINANCIAL ADVICE** - This system is for educational and research purposes only
- **Past performance does not guarantee future results**
- Always conduct your own due diligence and research
- Consult qualified financial advisors before making investment decisions

### **Risk Warnings**
- Trading involves substantial risk of loss
- No guaranteed returns or protection against losses
- Market conditions can change rapidly
- Technical issues may prevent signal generation
- Transaction costs, slippage, and taxes not fully modeled

### **Limitations**
- Based on historical data and technical analysis
- Model predictions are probabilistic estimates, not certainties
- External factors (news, earnings, policy changes) can significantly impact performance
- Live market conditions may differ from backtested results

---

## 🔧 **Configuration Reference**

### **Database Settings**
```yaml
databases:
  postgres:
    host: "localhost"
    port: 5432
    dbname: "nifty_trading"
    user: "trader"
    password: "secret123"
    connection_pool_size: 10

  duckdb:
    path: "data/nifty_analytics.duckdb"
    memory_limit: "2GB"
```

### **Universe Definition**
```yaml
universe:
  tickers:
    - "RELIANCE.NS"
    - "TCS.NS"
    - "HDFCBANK.NS"
    - "INFY.NS"
    # ... more stocks
```

### **Signal Parameters**
```yaml
signal_thresholds:
  min_predicted_return_pct: 10.0
  min_conviction: 0.8
  max_signals_per_day: 5
```

### **Risk Settings**
```yaml
risk_settings:
  max_position_pct: 0.05
  max_open_positions: 5
  stop_loss_atr_multiplier: 1.5
  risk_per_trade_pct: 1.0
```

---

## 🚀 **Advanced Usage**

### **Custom Strategy Development**
```python
from signals.strategies import TradingStrategy

class CustomStrategy(TradingStrategy):
    def __init__(self):
        super().__init__(
            name="Custom Strategy",
            description="Your custom trading logic",
            parameters={'param1': value1, 'param2': value2}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        # Your strategy logic here
        return conviction_score

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        # Return detailed signal information
        return signal_details
```

### **Backtesting Custom Strategies**
```python
from backtest.backtester import Backtester

backtester = Backtester(
    strategy=CustomStrategy(),
    start_date='2020-01-01',
    end_date='2024-12-31',
    initial_capital=1000000
)

results = backtester.run_backtest()
backtester.generate_report()
```

### **Model Training & Validation**
```python
from models.model_training import ModelTrainer

trainer = ModelTrainer()
trainer.train_model(
    features=['rsi_14', 'macd', 'bb_width'],
    target='forward_10d_ret_pct',
    train_start='2020-01-01',
    train_end='2023-12-31'
)

trainer.validate_model(
    validation_start='2024-01-01',
    validation_end='2024-12-31'
)
```

---

## 📊 **Performance Metrics**

### **Signal Quality Metrics**
- **Win Rate**: Percentage of profitable signals
- **Profit Factor**: Gross profits / Gross losses
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Annual return / Maximum drawdown

### **System Health Metrics**
- **Signal Generation Rate**: Signals per day
- **Data Freshness**: Age of latest data
- **Processing Time**: Analysis pipeline duration
- **Error Rate**: System failure frequency

### **Risk Metrics**
- **Portfolio Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Maximum expected loss
- **Stress Test Results**: Performance under adverse conditions

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Database Connection Issues**
```bash
# Test PostgreSQL connection
python -c "from utils.db_postgres import test_connection; test_connection()"

# Test DuckDB connection
python -c "from utils.db_duckdb import test_connection; test_connection()"

# Reinitialize databases
python setup_databases.py
```

#### **Data Fetching Issues**
```bash
# Clear data cache
rm -rf data/price_data/*
rm -rf data/fundamentals_data/*

# Check Yahoo Finance access
python -c "import yfinance as yf; print(yf.Ticker('TCS.NS').info)"
```

#### **Memory Issues**
```yaml
# Increase DuckDB memory limit in config.yaml
duckdb:
  memory_limit: "4GB"  # Increase from 2GB
```

#### **Signal Generation Issues**
```bash
# Check feature computation
python -c "
from features.feature_engineering import FeatureEngineer
fe = FeatureEngineer()
features = fe.compute_features('TCS.NS')
print(features.head())
"
```

### **Logs & Debugging**
```bash
# Check application logs
tail -f logs/trading_agent.log

# Check database setup logs
tail -f logs/db_setup.log

# Enable debug logging in config.yaml
logging:
  level: "DEBUG"
```

---

## 📈 **Extending the System**

### **Adding New Data Sources**
```python
# Create new data provider
class CustomDataProvider:
    def fetch_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Your data fetching logic
        return df

# Register in data_providers/__init__.py
from .custom_data_provider import CustomDataProvider
```

### **Adding New Features**
```python
# Extend FeatureEngineer class
def compute_custom_feature(self, data: pd.DataFrame) -> pd.Series:
    # Your feature computation logic
    return custom_feature

# Add to feature computation pipeline
features['custom_feature'] = self.compute_custom_feature(data)
```

### **Adding New Strategies**
```python
# Implement new strategy class
class NewStrategy(TradingStrategy):
    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        # Strategy evaluation logic
        return score

# Register in strategies.py
STRATEGIES['new_strategy'] = NewStrategy
```

---

## 📄 **API Reference**

### **Core Classes**

#### **DailyPipeline**
```python
class DailyPipeline:
    def __init__(self, config: Optional[Dict] = None)
    def run_daily_analysis(self, preferences: Dict[str, Any]) -> Dict[str, Any]
    def fetch_market_data(self, symbols: List[str]) -> pd.DataFrame
    def generate_signals(self, data: pd.DataFrame, strategy: str) -> List[Dict]
    def calculate_positions(self, signals: List[Dict]) -> List[Dict]
```

#### **UserInterface**
```python
class UserInterface:
    def collect_user_preferences(self) -> Dict[str, Any]
    def _select_index(self) -> Dict[str, Any]
    def _select_num_stocks(self) -> int
    def _select_profitability(self) -> float
    def _select_strategy(self) -> Dict[str, Any]
    def _select_conviction_threshold(self) -> float
    def _select_risk_params(self) -> Dict[str, Any]
```

#### **TradingStrategy** (Base Class)
```python
class TradingStrategy:
    def __init__(self, name: str, description: str, parameters: Dict[str, Any])
    def evaluate(self, data: pd.DataFrame, symbol: str) -> float
    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]
```

### **Database Functions**

#### **PostgreSQL Tools**
- `record_signals(signals: List[Dict]) -> List[int]`
- `fetch_signals_for_date(date) -> List[Dict]`
- `create_order_from_signal(signal_id, side, quantity, price) -> int`
- `get_open_positions() -> List[Dict]`
- `log_agent_run(agent_name, run_type, status, meta) -> int`
- `update_agent_run(run_id, status, meta) -> bool`

#### **DuckDB Tools**
- `store_ohlcv(df_ohlcv) -> bool`
- `load_ohlcv(symbols, start_date, end_date) -> pd.DataFrame`
- `store_features(df_features) -> bool`
- `load_features_for_training(start_date, end_date) -> pd.DataFrame`
- `store_backtest_results(strategy_id, run_id, df_equity) -> bool`
- `load_backtest_results(strategy_id) -> pd.DataFrame`

---

## 🎯 **Best Practices**

### **System Usage**
1. **Regular Monitoring**: Check logs and system health daily
2. **Data Quality**: Validate data sources and feature computation
3. **Risk Management**: Never exceed recommended position sizes
4. **Performance Tracking**: Maintain detailed records of all trades

### **Development**
1. **Version Control**: Use git for all code changes
2. **Testing**: Add unit tests for new features
3. **Documentation**: Update documentation for API changes
4. **Code Quality**: Follow PEP 8 standards and add type hints

### **Production Deployment**
1. **Environment Setup**: Use separate environments for dev/staging/prod
2. **Monitoring**: Implement comprehensive logging and alerting
3. **Backup**: Regular database backups and disaster recovery
4. **Security**: Secure database credentials and API keys

---

## 🤝 **Support & Contributing**

### **Getting Help**
- **Documentation**: Check this comprehensive guide first
- **Logs**: Review application logs for error details
- **Debugging**: Use the troubleshooting section above
- **Community**: Join discussions and share experiences

### **Contributing**
1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Submit** a pull request with detailed description

### **Areas for Contribution**
- Additional trading strategies
- New technical indicators and features
- Alternative data sources
- Performance optimizations
- Enhanced risk management
- Web dashboard development
- Mobile app interface

---

## 📋 **Changelog**

### **Version 1.0.0** (Current)
- ✅ Interactive user interface with 8-step wizard
- ✅ Multiple index support (Nifty 50, Next 50, Bank Nifty, IT Nifty)
- ✅ 7 trading strategies with conviction scoring
- ✅ 50+ technical and fundamental features
- ✅ Production database architecture (PostgreSQL + DuckDB)
- ✅ Agentic AI integration with clean function interfaces
- ✅ Comprehensive risk management and position sizing
- ✅ Performance tracking and backtesting capabilities
- ✅ Real-time data integration with Yahoo Finance
- ✅ Advanced reporting and signal export

---

## ⚖️ **License**

This project is for educational purposes. See individual component licenses for details.

---

**⚠️ CRITICAL DISCLAIMER**: This is NOT financial advice. Trading involves significant risk of loss. Always do your own research and consult professionals before investing. The authors are not responsible for any financial losses incurred through the use of this tool.**

**Remember: This system is for educational and research purposes only. Past performance does not guarantee future results.**
