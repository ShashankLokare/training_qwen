#!/usr/bin/env python3
"""
V3 Position Sizing Module for Nifty Trading Agent
Dynamic, volatility-aware position sizing with conviction adjustments
"""

import sys
import numpy as np
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class PositionSizerV3:
    """
    V3 Enhanced position sizing with volatility adjustment and conviction scaling
    """

    def __init__(self,
                 base_risk_per_trade: float = 0.01,  # 1% of capital per trade
                 max_position_pct: float = 0.05,     # Max 5% per position
                 min_position_pct: float = 0.005,    # Min 0.5% per position
                 volatility_window: int = 20,        # 20-day volatility window
                 regime_adjustment: bool = True):    # Adjust for market regime

        self.base_risk_per_trade = base_risk_per_trade
        self.max_position_pct = max_position_pct
        self.min_position_pct = min_position_pct
        self.volatility_window = volatility_window
        self.regime_adjustment = regime_adjustment

        # Volatility scaling parameters
        self.vol_target = 0.15  # Target annual volatility (15%)
        self.vol_floor = 0.05   # Minimum volatility assumption (5%)
        self.vol_ceiling = 0.40 # Maximum volatility assumption (40%)

        # Conviction multipliers
        self.conviction_multipliers = {
            'low': 0.5,      # 50% of base size
            'medium': 1.0,   # 100% of base size
            'high': 1.5      # 150% of base size
        }

        # Regime multipliers
        self.regime_multipliers = {
            'bull': 1.2,     # Increase size in bull markets
            'bear': 0.8,     # Reduce size in bear markets
            'sideways': 1.0, # Normal size in sideways markets
            'high_vol': 0.6  # Significantly reduce in high vol
        }

        logger.info("PositionSizerV3 initialized")

    def compute_position_size(self,
                            capital: float,
                            recent_volatility: float,
                            conviction: str = 'medium',
                            regime: str = 'sideways',
                            symbol: str = None) -> float:
        """
        Compute position size as fraction of capital using Kelly criterion with adjustments

        Args:
            capital: Total portfolio capital
            recent_volatility: Recent realized volatility (annualized)
            conviction: Conviction level ('low', 'medium', 'high')
            regime: Current market regime
            symbol: Trading symbol (for logging)

        Returns:
            Position size as fraction of capital (0.0 to max_position_pct)
        """
        # Clamp volatility to reasonable bounds
        vol_adj = np.clip(recent_volatility, self.vol_floor, self.vol_ceiling)

        # Base Kelly position size
        # Kelly = (expected_return / variance) but simplified to volatility adjustment
        kelly_size = (self.vol_target / vol_adj) * self.base_risk_per_trade

        # Apply conviction multiplier
        conviction_mult = self.conviction_multipliers.get(conviction, 1.0)
        kelly_size *= conviction_mult

        # Apply regime multiplier
        if self.regime_adjustment:
            regime_mult = self.regime_multipliers.get(regime, 1.0)
            kelly_size *= regime_mult

        # Clamp to reasonable bounds
        position_size = np.clip(kelly_size, self.min_position_pct, self.max_position_pct)

        logger.debug(f"Position size for {symbol or 'unknown'}: {position_size:.4f} "
                    f"(vol={vol_adj:.3f}, conviction={conviction}, regime={regime})")

        return position_size

    def compute_portfolio_heat(self, current_positions: Dict[str, float],
                             max_heat: float = 0.8) -> float:
        """
        Compute current portfolio heat (exposure)

        Args:
            current_positions: Dict of symbol -> position_size
            max_heat: Maximum allowed portfolio heat

        Returns:
            Current portfolio heat (0.0 to 1.0+)
        """
        total_exposure = sum(abs(size) for size in current_positions.values())
        portfolio_heat = total_exposure / max_heat

        return portfolio_heat

    def adjust_for_portfolio_heat(self, position_size: float,
                                current_heat: float,
                                max_heat: float = 0.8) -> float:
        """
        Adjust position size based on current portfolio heat

        Args:
            position_size: Proposed position size
            current_heat: Current portfolio heat
            max_heat: Maximum allowed heat

        Returns:
            Adjusted position size
        """
        if current_heat >= max_heat:
            # Don't add new positions if at max heat
            return 0.0
        elif current_heat > max_heat * 0.8:
            # Reduce size when approaching max heat
            heat_factor = 1.0 - (current_heat / max_heat)
            return position_size * heat_factor
        else:
            return position_size

    def get_recent_volatility(self, symbol: str, current_date: str,
                            lookback_days: int = 20) -> float:
        """
        Calculate recent realized volatility for a symbol

        Args:
            symbol: Trading symbol
            current_date: Current date (YYYY-MM-DD)
            lookback_days: Days to look back for volatility calculation

        Returns:
            Annualized volatility or default value if insufficient data
        """
        try:
            from utils.db_duckdb import execute_query

            # Get historical prices
            start_date = (pd.to_datetime(current_date) - pd.Timedelta(days=lookback_days * 2)).strftime('%Y-%m-%d')

            query = f"""
            SELECT close
            FROM ohlcv_nifty
            WHERE symbol = '{symbol}'
            AND date BETWEEN '{start_date}' AND '{current_date}'
            ORDER BY date
            """

            result = execute_query(query)
            if not result or len(result) < 10:
                logger.warning(f"Insufficient data for {symbol} volatility calculation")
                return self.vol_target  # Return target volatility as default

            prices = [row[0] for row in result]

            # Calculate daily returns
            returns = np.diff(np.log(prices))

            # Annualize volatility
            daily_vol = np.std(returns)
            annualized_vol = daily_vol * np.sqrt(252)  # Trading days per year

            return annualized_vol

        except Exception as e:
            logger.error(f"Error calculating volatility for {symbol}: {e}")
            return self.vol_target

    def estimate_conviction(self, model_probability: float,
                          signal_strength: float = 1.0) -> str:
        """
        Estimate conviction level from model outputs

        Args:
            model_probability: Model prediction probability (0-1)
            signal_strength: Additional signal strength indicator

        Returns:
            Conviction level ('low', 'medium', 'high')
        """
        combined_score = (model_probability * signal_strength)

        if combined_score >= 0.8:
            return 'high'
        elif combined_score >= 0.6:
            return 'medium'
        else:
            return 'low'

    def get_optimal_portfolio_allocation(self,
                                       signals: Dict[str, Dict[str, Any]],
                                       capital: float,
                                       current_date: str,
                                       current_positions: Dict[str, float] = None) -> Dict[str, float]:
        """
        Compute optimal portfolio allocation for multiple signals

        Args:
            signals: Dict of symbol -> signal_info with keys:
                    'probability', 'regime', 'conviction' (optional)
            capital: Total portfolio capital
            current_date: Current date for volatility calculation
            current_positions: Current positions dict

        Returns:
            Dict of symbol -> position_size
        """
        if current_positions is None:
            current_positions = {}

        allocations = {}

        for symbol, signal_info in signals.items():
            # Get recent volatility
            volatility = self.get_recent_volatility(symbol, current_date, self.volatility_window)

            # Get conviction level
            conviction = signal_info.get('conviction', 'medium')
            if 'conviction' not in signal_info:
                # Estimate from probability if not provided
                probability = signal_info.get('probability', 0.5)
                conviction = self.estimate_conviction(probability)

            # Get regime
            regime = signal_info.get('regime', 'sideways')

            # Compute base position size
            position_size = self.compute_position_size(
                capital=capital,
                recent_volatility=volatility,
                conviction=conviction,
                regime=regime,
                symbol=symbol
            )

            # Adjust for portfolio heat
            current_heat = self.compute_portfolio_heat(current_positions)
            position_size = self.adjust_for_portfolio_heat(position_size, current_heat)

            allocations[symbol] = position_size

        logger.info(f"Computed allocations for {len(allocations)} signals")
        return allocations

class RiskManagerV3:
    """
    V3 Risk management with dynamic stops and position limits
    """

    def __init__(self,
                 base_stop_loss: float = 0.05,     # 5% base stop loss
                 trailing_stop_activation: float = 0.10,  # Activate trailing at 10% profit
                 max_holding_days: int = 30,       # Max holding period
                 max_portfolio_heat: float = 0.8): # Max portfolio exposure

        self.base_stop_loss = base_stop_loss
        self.trailing_stop_activation = trailing_stop_activation
        self.max_holding_days = max_holding_days
        self.max_portfolio_heat = max_portfolio_heat

        # Regime-based stop adjustments
        self.regime_stop_multipliers = {
            'bull': 1.2,     # Wider stops in bull markets
            'bear': 0.8,     # Tighter stops in bear markets
            'sideways': 1.0, # Normal stops in sideways
            'high_vol': 1.5  # Much wider stops in high vol
        }

        logger.info("RiskManagerV3 initialized")

    def compute_dynamic_stop(self, entry_price: float, regime: str,
                           volatility: float) -> Tuple[float, float]:
        """
        Compute dynamic stop loss and take profit levels

        Args:
            entry_price: Entry price
            regime: Current market regime
            volatility: Current volatility

        Returns:
            Tuple of (stop_loss_price, take_profit_price)
        """
        # Adjust stop loss by regime
        regime_mult = self.regime_stop_multipliers.get(regime, 1.0)
        adjusted_stop = self.base_stop_loss * regime_mult

        # Further adjust by volatility (wider stops in high vol)
        vol_mult = min(max(volatility / 0.15, 0.5), 2.0)  # 0.5x to 2x based on vol
        final_stop = adjusted_stop * vol_mult

        # Compute prices
        stop_loss_price = entry_price * (1 - final_stop)
        take_profit_price = entry_price * (1 + final_stop * 2)  # 2:1 reward:risk

        return stop_loss_price, take_profit_price

    def should_exit_position(self, entry_date: str, current_date: str,
                           entry_price: float, current_price: float,
                           trailing_stop_price: float = None) -> Tuple[bool, str]:
        """
        Determine if position should be exited

        Args:
            entry_date: Position entry date
            current_date: Current date
            entry_price: Entry price
            current_price: Current price
            trailing_stop_price: Current trailing stop price

        Returns:
            Tuple of (should_exit, reason)
        """
        # Check max holding period
        entry_dt = pd.to_datetime(entry_date)
        current_dt = pd.to_datetime(current_date)
        holding_days = (current_dt - entry_dt).days

        if holding_days > self.max_holding_days:
            return True, f"Max holding period exceeded ({holding_days} days)"

        # Check trailing stop
        if trailing_stop_price and current_price <= trailing_stop_price:
            return True, f"Trailing stop hit at {trailing_stop_price:.2f}"

        # Check if position is deep in loss (emergency exit)
        loss_pct = (entry_price - current_price) / entry_price
        if loss_pct > 0.15:  # 15% loss
            return True, f"Emergency exit: {loss_pct:.1%} loss"

        return False, "Hold position"

    def update_trailing_stop(self, entry_price: float, current_price: float,
                           trailing_stop_price: float = None) -> float:
        """
        Update trailing stop price based on current profit

        Args:
            entry_price: Original entry price
            current_price: Current price
            trailing_stop_price: Current trailing stop price

        Returns:
            Updated trailing stop price
        """
        profit_pct = (current_price - entry_price) / entry_price

        # Activate trailing stop if profit >= activation threshold
        if profit_pct >= self.trailing_stop_activation:
            # Calculate new trailing stop: lock in some profit
            trail_distance = profit_pct * 0.7  # Trail at 70% of profit
            new_stop = entry_price * (1 + profit_pct - trail_distance)

            # Only update if higher than current stop
            if trailing_stop_price is None or new_stop > trailing_stop_price:
                return new_stop

        return trailing_stop_price

def demo_position_sizing():
    """Demo function for V3 position sizing"""
    print("📊 V3 POSITION SIZING - DEMO")
    print("=" * 35)

    # Initialize position sizer
    sizer = PositionSizerV3()

    # Test scenarios
    test_cases = [
        {'volatility': 0.10, 'conviction': 'low', 'regime': 'sideways', 'desc': 'Low vol, low conviction'},
        {'volatility': 0.20, 'conviction': 'medium', 'regime': 'bull', 'desc': 'High vol, medium conviction, bull'},
        {'volatility': 0.30, 'conviction': 'high', 'regime': 'high_vol', 'desc': 'Very high vol, high conviction, high vol regime'},
    ]

    capital = 100000

    for case in test_cases:
        size = sizer.compute_position_size(
            capital=capital,
            recent_volatility=case['volatility'],
            conviction=case['conviction'],
            regime=case['regime']
        )

        dollar_amount = size * capital
        print(f"{case['desc']}:")
        print(".2f")
        print(".0f")
        print()

    # Test risk manager
    print("🛡️  RISK MANAGEMENT DEMO")
    print("=" * 25)

    risk_mgr = RiskManagerV3()

    # Test stop calculation
    entry_price = 1000
    stop_price, tp_price = risk_mgr.compute_dynamic_stop(
        entry_price=entry_price,
        regime='bull',
        volatility=0.15
    )

    print(f"Entry: ${entry_price}")
    print(".2f")
    print(".2f")
    print(".1f")

if __name__ == "__main__":
    demo_position_sizing()
