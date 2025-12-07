"""
Trading Strategies for Nifty Trading Agent
Implementation of various technical analysis strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class TradingStrategy:
    """
    Base class for trading strategies
    """

    def __init__(self, name: str, description: str, parameters: Dict[str, Any]):
        """
        Initialize trading strategy

        Args:
            name: Strategy name
            description: Strategy description
            parameters: Strategy parameters
        """
        self.name = name
        self.description = description
        self.parameters = parameters

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """
        Evaluate the strategy for a given stock

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Strategy score (higher = better signal)
        """
        raise NotImplementedError("Subclasses must implement evaluate method")

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Get detailed signal information

        Args:
            data: OHLCV DataFrame
            symbol: Stock symbol

        Returns:
            Dictionary with signal details
        """
        raise NotImplementedError("Subclasses must implement get_signal_details method")

class DMA200Strategy(TradingStrategy):
    """DMA 200 - Stocks above 200-day moving average"""

    def __init__(self):
        super().__init__(
            name="DMA 200",
            description="Stocks trading above 200-day moving average",
            parameters={'ma_period': 200, 'condition': 'above'}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate DMA 200 strategy"""
        if len(data) < 200:
            return 0.0

        ma_200 = data['Close'].rolling(window=200).mean()
        current_price = data['Close'].iloc[-1]
        ma_value = ma_200.iloc[-1]

        if pd.isna(ma_value):
            return 0.0

        # Calculate how far above/below MA the stock is
        distance_pct = (current_price - ma_value) / ma_value

        # Return positive score if above MA, negative if below
        return max(0.0, distance_pct * 10)  # Scale for scoring

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get DMA 200 signal details"""
        if len(data) < 200:
            return {'valid': False, 'reason': 'Insufficient data'}

        ma_200 = data['Close'].rolling(window=200).mean()
        current_price = data['Close'].iloc[-1]
        ma_value = ma_200.iloc[-1]

        if pd.isna(ma_value):
            return {'valid': False, 'reason': 'MA calculation failed'}

        is_above = current_price > ma_value
        distance_pct = abs(current_price - ma_value) / ma_value

        return {
            'valid': True,
            'is_above_ma': is_above,
            'ma_value': ma_value,
            'current_price': current_price,
            'distance_pct': distance_pct,
            'signal_strength': distance_pct if is_above else 0.0
        }

class DMA50Strategy(TradingStrategy):
    """DMA 50 - Stocks above 50-day moving average"""

    def __init__(self):
        super().__init__(
            name="DMA 50",
            description="Stocks trading above 50-day moving average",
            parameters={'ma_period': 50, 'condition': 'above'}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate DMA 50 strategy"""
        if len(data) < 50:
            return 0.0

        ma_50 = data['Close'].rolling(window=50).mean()
        current_price = data['Close'].iloc[-1]
        ma_value = ma_50.iloc[-1]

        if pd.isna(ma_value):
            return 0.0

        distance_pct = (current_price - ma_value) / ma_value
        return max(0.0, distance_pct * 10)

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get DMA 50 signal details"""
        if len(data) < 50:
            return {'valid': False, 'reason': 'Insufficient data'}

        ma_50 = data['Close'].rolling(window=50).mean()
        current_price = data['Close'].iloc[-1]
        ma_value = ma_50.iloc[-1]

        if pd.isna(ma_value):
            return {'valid': False, 'reason': 'MA calculation failed'}

        is_above = current_price > ma_value
        distance_pct = abs(current_price - ma_value) / ma_value

        return {
            'valid': True,
            'is_above_ma': is_above,
            'ma_value': ma_value,
            'current_price': current_price,
            'distance_pct': distance_pct,
            'signal_strength': distance_pct if is_above else 0.0
        }

class SMA20Strategy(TradingStrategy):
    """SMA 20 Crossover - Stocks above 20-day SMA"""

    def __init__(self):
        super().__init__(
            name="SMA 20 Crossover",
            description="Stocks with price above 20-day simple moving average",
            parameters={'ma_period': 20, 'condition': 'above'}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate SMA 20 strategy"""
        if len(data) < 20:
            return 0.0

        sma_20 = data['Close'].rolling(window=20).mean()
        current_price = data['Close'].iloc[-1]
        sma_value = sma_20.iloc[-1]

        if pd.isna(sma_value):
            return 0.0

        # Check for recent crossover
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        prev_sma = sma_20.iloc[-2] if len(sma_20) > 1 else sma_value

        # Recent crossover bonus
        crossover_bonus = 0.0
        if prev_price <= prev_sma and current_price > sma_value:
            crossover_bonus = 2.0  # Bullish crossover

        distance_pct = (current_price - sma_value) / sma_value
        return max(0.0, distance_pct * 5) + crossover_bonus

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get SMA 20 signal details"""
        if len(data) < 20:
            return {'valid': False, 'reason': 'Insufficient data'}

        sma_20 = data['Close'].rolling(window=20).mean()
        current_price = data['Close'].iloc[-1]
        sma_value = sma_20.iloc[-1]

        if pd.isna(sma_value):
            return {'valid': False, 'reason': 'SMA calculation failed'}

        # Check for crossover
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        prev_sma = sma_20.iloc[-2] if len(sma_20) > 1 else sma_value

        crossover_occurred = prev_price <= prev_sma and current_price > sma_value
        is_above = current_price > sma_value
        distance_pct = abs(current_price - sma_value) / sma_value

        return {
            'valid': True,
            'is_above_sma': is_above,
            'sma_value': sma_value,
            'current_price': current_price,
            'distance_pct': distance_pct,
            'crossover_occurred': crossover_occurred,
            'signal_strength': (distance_pct + (2.0 if crossover_occurred else 0.0)) if is_above else 0.0
        }

class RSIOversoldStrategy(TradingStrategy):
    """RSI Oversold - Stocks with RSI below 30"""

    def __init__(self):
        super().__init__(
            name="RSI Oversold",
            description="Stocks with RSI below 30 (potentially oversold)",
            parameters={'rsi_threshold': 30, 'condition': 'below'}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate RSI oversold strategy"""
        if len(data) < 14:
            return 0.0

        # Calculate RSI
        rsi = self._calculate_rsi(data['Close'], 14)
        current_rsi = rsi.iloc[-1]

        if pd.isna(current_rsi):
            return 0.0

        threshold = self.parameters['rsi_threshold']

        # Score based on how oversold the stock is
        if current_rsi < threshold:
            oversold_score = (threshold - current_rsi) / threshold  # Normalized 0-1
            return oversold_score * 10  # Scale for consistency
        else:
            return 0.0

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get RSI oversold signal details"""
        if len(data) < 14:
            return {'valid': False, 'reason': 'Insufficient data'}

        rsi = self._calculate_rsi(data['Close'], 14)
        current_rsi = rsi.iloc[-1]

        if pd.isna(current_rsi):
            return {'valid': False, 'reason': 'RSI calculation failed'}

        threshold = self.parameters['rsi_threshold']
        is_oversold = current_rsi < threshold

        return {
            'valid': True,
            'rsi_value': current_rsi,
            'threshold': threshold,
            'is_oversold': is_oversold,
            'oversold_extent': threshold - current_rsi if is_oversold else 0.0,
            'signal_strength': (threshold - current_rsi) / threshold if is_oversold else 0.0
        }

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

class BollingerBreakoutStrategy(TradingStrategy):
    """Bollinger Band Breakout - Stocks breaking above upper band"""

    def __init__(self):
        super().__init__(
            name="Bollinger Band Breakout",
            description="Stocks breaking above upper Bollinger Band",
            parameters={'bb_period': 20, 'bb_std': 2}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate Bollinger breakout strategy"""
        if len(data) < 20:
            return 0.0

        # Calculate Bollinger Bands
        period = self.parameters['bb_period']
        std_dev = self.parameters['bb_std']

        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        current_price = data['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]

        if pd.isna(current_upper):
            return 0.0

        # Check for breakout
        if current_price > current_upper:
            # Calculate breakout strength
            breakout_pct = (current_price - current_upper) / current_upper
            return breakout_pct * 20  # Scale for scoring
        else:
            return 0.0

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get Bollinger breakout signal details"""
        if len(data) < 20:
            return {'valid': False, 'reason': 'Insufficient data'}

        period = self.parameters['bb_period']
        std_dev = self.parameters['bb_std']

        sma = data['Close'].rolling(window=period).mean()
        std = data['Close'].rolling(window=period).std()

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        current_price = data['Close'].iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        current_middle = sma.iloc[-1]

        if pd.isna(current_upper) or pd.isna(current_middle):
            return {'valid': False, 'reason': 'BB calculation failed'}

        breakout_occurred = current_price > current_upper
        breakdown_occurred = current_price < current_lower

        # Calculate band position
        band_width = (current_upper - current_lower) / current_middle
        position_in_band = (current_price - current_lower) / (current_upper - current_lower)

        return {
            'valid': True,
            'breakout_occurred': breakout_occurred,
            'breakdown_occurred': breakdown_occurred,
            'current_price': current_price,
            'upper_band': current_upper,
            'middle_band': current_middle,
            'lower_band': current_lower,
            'band_width': band_width,
            'position_in_band': position_in_band,
            'breakout_strength': (current_price - current_upper) / current_upper if breakout_occurred else 0.0,
            'signal_strength': (current_price - current_upper) / current_upper * 20 if breakout_occurred else 0.0
        }

class VolumeBreakoutStrategy(TradingStrategy):
    """Volume Breakout - Stocks with above-average volume"""

    def __init__(self):
        super().__init__(
            name="Volume Breakout",
            description="Stocks with above-average volume",
            parameters={'volume_multiplier': 1.5}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate volume breakout strategy"""
        if len(data) < 20:
            return 0.0

        # Calculate average volume
        avg_volume = data['Volume'].rolling(window=20).mean()
        current_volume = data['Volume'].iloc[-1]
        avg_vol_value = avg_volume.iloc[-1]

        if pd.isna(avg_vol_value) or avg_vol_value == 0:
            return 0.0

        # Check volume multiplier
        multiplier = self.parameters['volume_multiplier']
        volume_ratio = current_volume / avg_vol_value

        if volume_ratio >= multiplier:
            # Score based on how much above average
            excess_volume = volume_ratio - 1.0
            return excess_volume * 5  # Scale for scoring
        else:
            return 0.0

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get volume breakout signal details"""
        if len(data) < 20:
            return {'valid': False, 'reason': 'Insufficient data'}

        avg_volume = data['Volume'].rolling(window=20).mean()
        current_volume = data['Volume'].iloc[-1]
        avg_vol_value = avg_volume.iloc[-1]

        if pd.isna(avg_vol_value) or avg_vol_value == 0:
            return {'valid': False, 'reason': 'Volume calculation failed'}

        volume_ratio = current_volume / avg_vol_value
        multiplier = self.parameters['volume_multiplier']
        breakout_occurred = volume_ratio >= multiplier

        return {
            'valid': True,
            'current_volume': current_volume,
            'avg_volume': avg_vol_value,
            'volume_ratio': volume_ratio,
            'multiplier_threshold': multiplier,
            'breakout_occurred': breakout_occurred,
            'excess_volume_pct': (volume_ratio - 1.0) * 100,
            'signal_strength': (volume_ratio - 1.0) * 5 if breakout_occurred else 0.0
        }

class MomentumStrategy(TradingStrategy):
    """Momentum Strategy - High momentum stocks based on ROC"""

    def __init__(self):
        super().__init__(
            name="Momentum Strategy",
            description="High momentum stocks based on Rate of Change",
            parameters={'momentum_period': 20, 'threshold': 0.05}
        )

    def evaluate(self, data: pd.DataFrame, symbol: str) -> float:
        """Evaluate momentum strategy"""
        period = self.parameters['momentum_period']
        threshold = self.parameters['threshold']

        if len(data) < period:
            return 0.0

        # Calculate Rate of Change (ROC)
        roc = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        current_roc = roc.iloc[-1]

        if pd.isna(current_roc):
            return 0.0

        # Check if momentum is above threshold
        if current_roc > threshold:
            # Score based on momentum strength
            return current_roc * 10  # Scale for scoring
        else:
            return 0.0

    def get_signal_details(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Get momentum signal details"""
        period = self.parameters['momentum_period']
        threshold = self.parameters['threshold']

        if len(data) < period:
            return {'valid': False, 'reason': 'Insufficient data'}

        roc = (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
        current_roc = roc.iloc[-1]

        if pd.isna(current_roc):
            return {'valid': False, 'reason': 'ROC calculation failed'}

        has_momentum = current_roc > threshold

        return {
            'valid': True,
            'current_roc': current_roc,
            'threshold': threshold,
            'has_momentum': has_momentum,
            'momentum_strength': current_roc,
            'period_days': period,
            'signal_strength': current_roc * 10 if has_momentum else 0.0
        }

class StrategyFactory:
    """Factory class for creating trading strategies"""

    @staticmethod
    def create_strategy(strategy_key: str) -> TradingStrategy:
        """
        Create a strategy instance based on key

        Args:
            strategy_key: Strategy identifier

        Returns:
            Strategy instance
        """
        strategies = {
            'dma200': DMA200Strategy,
            'dma50': DMA50Strategy,
            'sma20': SMA20Strategy,
            'rsi_oversold': RSIOversoldStrategy,
            'bollinger_breakout': BollingerBreakoutStrategy,
            'volume_breakout': VolumeBreakoutStrategy,
            'momentum': MomentumStrategy
        }

        strategy_class = strategies.get(strategy_key)
        if strategy_class:
            return strategy_class()
        else:
            raise ValueError(f"Unknown strategy: {strategy_key}")

    @staticmethod
    def get_available_strategies() -> Dict[str, Dict[str, Any]]:
        """
        Get information about all available strategies

        Returns:
            Dictionary with strategy information
        """
        return {
            'dma200': {
                'name': 'DMA 200',
                'description': 'Stocks above 200-day moving average',
                'class': DMA200Strategy
            },
            'dma50': {
                'name': 'DMA 50',
                'description': 'Stocks above 50-day moving average',
                'class': DMA50Strategy
            },
            'sma20': {
                'name': 'SMA 20 Crossover',
                'description': 'Stocks with price above 20-day SMA',
                'class': SMA20Strategy
            },
            'rsi_oversold': {
                'name': 'RSI Oversold',
                'description': 'Stocks with RSI below 30 (oversold)',
                'class': RSIOversoldStrategy
            },
            'bollinger_breakout': {
                'name': 'Bollinger Band Breakout',
                'description': 'Stocks breaking above upper Bollinger Band',
                'class': BollingerBreakoutStrategy
            },
            'volume_breakout': {
                'name': 'Volume Breakout',
                'description': 'Stocks with above-average volume',
                'class': VolumeBreakoutStrategy
            },
            'momentum': {
                'name': 'Momentum Strategy',
                'description': 'High momentum stocks based on ROC',
                'class': MomentumStrategy
            }
        }
