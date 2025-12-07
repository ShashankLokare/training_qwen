"""
Feature Engineering for Nifty Trading Agent
Creates technical, fundamental, and sentiment features for ML models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging

from ..utils.logging_utils import get_logger
from ..utils.date_utils import get_trading_days

logger = get_logger(__name__)

class FeatureEngineer:
    """
    Creates comprehensive feature set for stock prediction models.

    Features include:
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Price momentum and volatility measures
    - Volume-based features
    - Fundamental ratios and growth metrics
    - Sentiment scores and trends
    """

    def __init__(self):
        """Initialize the feature engineer"""
        logger.info("FeatureEngineer initialized")

    def create_feature_matrix(self, symbol: str, market_data: pd.DataFrame,
                            fundamentals: Optional[Dict] = None,
                            sentiment_data: Optional[Dict] = None,
                            lookback_periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create comprehensive feature matrix for a stock

        Args:
            symbol: Stock symbol
            market_data: OHLCV DataFrame with Date index
            fundamentals: Dictionary with fundamental data
            sentiment_data: Dictionary with sentiment data
            lookback_periods: List of periods for technical indicators

        Returns:
            DataFrame with features and target variables
        """
        logger.info(f"Creating feature matrix for {symbol}")

        if lookback_periods is None:
            lookback_periods = [5, 10, 20, 50, 100, 200]

        try:
            # Ensure data is sorted by date
            market_data = market_data.sort_index()

            # Create technical features
            tech_features = self._create_technical_features(market_data, lookback_periods)

            # Create volume features
            volume_features = self._create_volume_features(market_data, lookback_periods)

            # Create momentum features
            momentum_features = self._create_momentum_features(market_data, lookback_periods)

            # Create volatility features
            volatility_features = self._create_volatility_features(market_data, lookback_periods)

            # Combine all technical features
            features_df = pd.concat([
                tech_features,
                volume_features,
                momentum_features,
                volatility_features
            ], axis=1)

            # Add fundamental features if available
            if fundamentals:
                fund_features = self._create_fundamental_features(fundamentals)
                # Repeat fundamental features for each date
                fund_features_expanded = pd.DataFrame(
                    [fund_features] * len(features_df),
                    index=features_df.index
                )
                features_df = pd.concat([features_df, fund_features_expanded], axis=1)

            # Add sentiment features if available
            if sentiment_data:
                sentiment_features = self._create_sentiment_features(sentiment_data)
                sentiment_features_expanded = pd.DataFrame(
                    [sentiment_features] * len(features_df),
                    index=features_df.index
                )
                features_df = pd.concat([features_df, sentiment_features_expanded], axis=1)

            # Create target variables (future returns)
            target_features = self._create_target_features(market_data)
            features_df = pd.concat([features_df, target_features], axis=1)

            # Add symbol identifier
            features_df['symbol'] = symbol

            # Remove any rows with NaN values
            features_df = features_df.dropna()

            logger.info(f"Created feature matrix with {len(features_df)} rows and {len(features_df.columns)} features for {symbol}")
            return features_df

        except Exception as e:
            logger.error(f"Error creating feature matrix for {symbol}: {e}")
            return pd.DataFrame()

    def _create_technical_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Create technical indicator features

        Args:
            data: OHLCV DataFrame
            periods: List of periods for indicators

        Returns:
            DataFrame with technical features
        """
        features = pd.DataFrame(index=data.index)

        try:
            # Moving averages
            for period in periods:
                features[f'sma_{period}'] = data['Close'].rolling(window=period).mean()
                features[f'ema_{period}'] = data['Close'].ewm(span=period).mean()

            # RSI (Relative Strength Index)
            for period in [14, 21]:
                features[f'rsi_{period}'] = self._calculate_rsi(data['Close'], period)

            # MACD (Moving Average Convergence Divergence)
            features['macd_line'], features['macd_signal'], features['macd_hist'] = self._calculate_macd(data['Close'])

            # Bollinger Bands
            for period in [20, 50]:
                sma = data['Close'].rolling(window=period).mean()
                std = data['Close'].rolling(window=period).std()
                features[f'bb_upper_{period}'] = sma + (std * 2)
                features[f'bb_lower_{period}'] = sma - (std * 2)
                features[f'bb_middle_{period}'] = sma
                features[f'bb_width_{period}'] = (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}']) / features[f'bb_middle_{period}']

                # Bollinger Band position
                features[f'bb_position_{period}'] = (data['Close'] - features[f'bb_lower_{period}']) / (features[f'bb_upper_{period}'] - features[f'bb_lower_{period}'])

            # Stochastic Oscillator
            features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(data)

            # Williams %R
            features['williams_r'] = self._calculate_williams_r(data)

            # Commodity Channel Index (CCI)
            features['cci'] = self._calculate_cci(data)

        except Exception as e:
            logger.warning(f"Error creating technical features: {e}")

        return features

    def _create_volume_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Create volume-based features

        Args:
            data: OHLCV DataFrame
            periods: List of periods for volume analysis

        Returns:
            DataFrame with volume features
        """
        features = pd.DataFrame(index=data.index)

        try:
            # Volume moving averages
            for period in periods:
                features[f'volume_sma_{period}'] = data['Volume'].rolling(window=period).mean()

            # Volume ratio (current volume vs average)
            features['volume_ratio_20'] = data['Volume'] / data['Volume'].rolling(window=20).mean()

            # On-Balance Volume (OBV)
            features['obv'] = self._calculate_obv(data)

            # Volume Price Trend (VPT)
            features['vpt'] = self._calculate_vpt(data)

            # Accumulation/Distribution Line
            features['adl'] = self._calculate_adl(data)

            # Volume Z-score
            for period in [20, 50]:
                volume_mean = data['Volume'].rolling(window=period).mean()
                volume_std = data['Volume'].rolling(window=period).std()
                features[f'volume_zscore_{period}'] = (data['Volume'] - volume_mean) / volume_std

        except Exception as e:
            logger.warning(f"Error creating volume features: {e}")

        return features

    def _create_momentum_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Create momentum-based features

        Args:
            data: OHLCV DataFrame
            periods: List of periods for momentum analysis

        Returns:
            DataFrame with momentum features
        """
        features = pd.DataFrame(index=data.index)

        try:
            # Price returns
            features['return_1d'] = data['Close'].pct_change(1)
            features['return_3d'] = data['Close'].pct_change(3)
            features['return_5d'] = data['Close'].pct_change(5)
            features['return_10d'] = data['Close'].pct_change(10)
            features['return_20d'] = data['Close'].pct_change(20)

            # Rate of change
            for period in periods:
                if period > 1:
                    features[f'roc_{period}'] = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)

            # Momentum indicators
            features['momentum_10'] = data['Close'] - data['Close'].shift(10)
            features['momentum_20'] = data['Close'] - data['Close'].shift(20)

            # Price velocity (rate of change of momentum)
            features['price_velocity'] = features['return_1d'] - features['return_1d'].shift(1)

        except Exception as e:
            logger.warning(f"Error creating momentum features: {e}")

        return features

    def _create_volatility_features(self, data: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """
        Create volatility-based features

        Args:
            data: OHLCV DataFrame
            periods: List of periods for volatility analysis

        Returns:
            DataFrame with volatility features
        """
        features = pd.DataFrame(index=data.index)

        try:
            # Average True Range (ATR)
            features['atr'] = self._calculate_atr(data)

            # Historical volatility
            for period in periods:
                returns = data['Close'].pct_change()
                features[f'volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252)  # Annualized

            # Parkinson's volatility (high-low based)
            features['parkinson_vol'] = np.sqrt((1/(4*np.log(2))) * ((np.log(data['High']/data['Low']))**2))

            # Garman-Klass volatility
            features['garman_klass_vol'] = np.sqrt(
                0.5 * np.log(data['High']/data['Low'])**2 -
                (2*np.log(2)-1) * np.log(data['Close']/data['Open'])**2
            )

            # True Range
            features['true_range'] = self._calculate_true_range(data)

        except Exception as e:
            logger.warning(f"Error creating volatility features: {e}")

        return features

    def _create_fundamental_features(self, fundamentals: Dict[str, Any]) -> Dict[str, float]:
        """
        Create fundamental-based features

        Args:
            fundamentals: Dictionary with fundamental data

        Returns:
            Dictionary with fundamental features
        """
        features = {}

        try:
            # Valuation ratios
            features['pe_ratio'] = fundamentals.get('pe_ratio', 0)
            features['pb_ratio'] = fundamentals.get('pb_ratio', 0)
            features['div_yield'] = fundamentals.get('dividend_yield', 0)

            # Profitability ratios
            features['roe'] = fundamentals.get('roe', 0)
            features['roce'] = fundamentals.get('roce', 0)
            features['net_margin'] = fundamentals.get('net_margin', 0)
            features['gross_margin'] = fundamentals.get('gross_margin', 0)

            # Financial health
            features['debt_to_equity'] = fundamentals.get('debt_to_equity', 0)
            features['current_ratio'] = fundamentals.get('current_ratio', 0)

            # Growth indicators (placeholder - would need historical data)
            features['revenue_growth_1y'] = 0.0  # Would be calculated from quarterly data
            features['eps_growth_1y'] = 0.0

            # Market data
            features['market_cap'] = fundamentals.get('market_cap', 0)
            features['beta'] = fundamentals.get('beta', 1.0)

        except Exception as e:
            logger.warning(f"Error creating fundamental features: {e}")

        return features

    def _create_sentiment_features(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Create sentiment-based features

        Args:
            sentiment_data: Dictionary with sentiment data

        Returns:
            Dictionary with sentiment features
        """
        features = {}

        try:
            # Current sentiment score
            features['sentiment_score'] = sentiment_data.get('sentiment_score', 0.0)

            # Sentiment trend
            trend_data = sentiment_data.get('trend', {})
            features['sentiment_trend'] = 1 if trend_data.get('trend') == 'improving' else (-1 if trend_data.get('trend') == 'deteriorating' else 0)
            features['sentiment_volatility'] = trend_data.get('volatility', 0.0)

            # Sentiment momentum (change over time)
            features['sentiment_change'] = trend_data.get('change', 0.0)

        except Exception as e:
            logger.warning(f"Error creating sentiment features: {e}")

        return features

    def _create_target_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for supervised learning

        Args:
            data: OHLCV DataFrame

        Returns:
            DataFrame with target variables
        """
        targets = pd.DataFrame(index=data.index)

        try:
            # Future returns at different horizons
            targets['target_return_5d'] = data['Close'].shift(-5) / data['Close'] - 1
            targets['target_return_10d'] = data['Close'].shift(-10) / data['Close'] - 1
            targets['target_return_20d'] = data['Close'].shift(-20) / data['Close'] - 1

            # Binary classification targets (achieve >= 10% return)
            targets['target_up_10pct_5d'] = (targets['target_return_5d'] >= 0.10).astype(int)
            targets['target_up_10pct_10d'] = (targets['target_return_10d'] >= 0.10).astype(int)

            # Multi-class targets (up, flat, down)
            targets['target_direction_5d'] = pd.cut(
                targets['target_return_5d'],
                bins=[-np.inf, -0.02, 0.02, np.inf],
                labels=[-1, 0, 1]
            ).astype(int)

        except Exception as e:
            logger.warning(f"Error creating target features: {e}")

        return targets

    # Technical indicator calculation methods

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except:
            return pd.Series(index=prices.index)

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        try:
            ema_fast = prices.ewm(span=fast).mean()
            ema_slow = prices.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_hist = macd_line - macd_signal
            return macd_line, macd_signal, macd_hist
        except:
            return pd.Series(index=prices.index), pd.Series(index=prices.index), pd.Series(index=prices.index)

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        try:
            lowest_low = data['Low'].rolling(window=k_period).min()
            highest_high = data['High'].rolling(window=k_period).max()
            stoch_k = 100 * ((data['Close'] - lowest_low) / (highest_high - lowest_low))
            stoch_d = stoch_k.rolling(window=d_period).mean()
            return stoch_k, stoch_d
        except:
            return pd.Series(index=data.index), pd.Series(index=data.index)

    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        try:
            highest_high = data['High'].rolling(window=period).max()
            lowest_low = data['Low'].rolling(window=period).min()
            williams_r = -100 * ((highest_high - data['Close']) / (highest_high - lowest_low))
            return williams_r
        except:
            return pd.Series(index=data.index)

    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        try:
            typical_price = (data['High'] + data['Low'] + data['Close']) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad_tp = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad_tp)
            return cci
        except:
            return pd.Series(index=data.index)

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        try:
            obv = pd.Series(index=data.index, dtype=float)
            obv.iloc[0] = data['Volume'].iloc[0]

            for i in range(1, len(data)):
                if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] + data['Volume'].iloc[i]
                elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
                    obv.iloc[i] = obv.iloc[i-1] - data['Volume'].iloc[i]
                else:
                    obv.iloc[i] = obv.iloc[i-1]

            return obv
        except:
            return pd.Series(index=data.index)

    def _calculate_vpt(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        try:
            price_change = data['Close'].pct_change()
            vpt = (price_change * data['Volume']).cumsum()
            return vpt
        except:
            return pd.Series(index=data.index)

    def _calculate_adl(self, data: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line"""
        try:
            money_flow_multiplier = ((data['Close'] - data['Low']) - (data['High'] - data['Close'])) / (data['High'] - data['Low'])
            money_flow_volume = money_flow_multiplier * data['Volume']
            adl = money_flow_volume.cumsum()
            return adl
        except:
            return pd.Series(index=data.index)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift(1))
            low_close = np.abs(data['Low'] - data['Close'].shift(1))

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
        except:
            return pd.Series(index=data.index)

    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        try:
            high_low = data['High'] - data['Low']
            high_close = np.abs(data['High'] - data['Close'].shift(1))
            low_close = np.abs(data['Low'] - data['Close'].shift(1))

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return true_range
        except:
            return pd.Series(index=data.index)
