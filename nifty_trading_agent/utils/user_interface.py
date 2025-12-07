"""
User Interface utilities for Nifty Trading Agent
Interactive CLI for user input and configuration
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_utils import get_logger

logger = get_logger(__name__)

class UserInterface:
    """
    Interactive user interface for collecting trading parameters and preferences
    """

    def __init__(self):
        """Initialize the user interface"""
        self.available_indices = {
            'nifty50': {
                'name': 'Nifty 50',
                'symbol': '^NSEI',
                'description': 'India\'s benchmark stock market index',
                'sample_stocks': 10  # For demo, limit to 10 stocks
            },
            'nifty_next50': {
                'name': 'Nifty Next 50',
                'symbol': '^NSMIDCP',
                'description': 'Next tier of 50 companies after Nifty 50',
                'sample_stocks': 10
            },
            'bank_nifty': {
                'name': 'Bank Nifty',
                'symbol': '^NSEBANK',
                'description': 'Banking sector index',
                'sample_stocks': 8
            },
            'it_nifty': {
                'name': 'IT Nifty',
                'symbol': '^NSIT',
                'description': 'Information Technology sector index',
                'sample_stocks': 6
            }
        }

        self.available_strategies = {
            'dma200': {
                'name': 'DMA 200',
                'description': 'Stocks above 200-day moving average',
                'parameters': {'ma_period': 200, 'condition': 'above'}
            },
            'dma50': {
                'name': 'DMA 50',
                'description': 'Stocks above 50-day moving average',
                'parameters': {'ma_period': 50, 'condition': 'above'}
            },
            'sma20': {
                'name': 'SMA 20 Crossover',
                'description': 'Stocks with price above 20-day SMA',
                'parameters': {'ma_period': 20, 'condition': 'above'}
            },
            'rsi_oversold': {
                'name': 'RSI Oversold',
                'description': 'Stocks with RSI below 30 (oversold)',
                'parameters': {'rsi_threshold': 30, 'condition': 'below'}
            },
            'bollinger_breakout': {
                'name': 'Bollinger Band Breakout',
                'description': 'Stocks breaking above upper Bollinger Band',
                'parameters': {'bb_period': 20, 'bb_std': 2}
            },
            'volume_breakout': {
                'name': 'Volume Breakout',
                'description': 'Stocks with above-average volume',
                'parameters': {'volume_multiplier': 1.5}
            },
            'momentum': {
                'name': 'Momentum Strategy',
                'description': 'High momentum stocks based on ROC',
                'parameters': {'momentum_period': 20, 'threshold': 0.05}
            }
        }

    def collect_user_preferences(self) -> Dict[str, Any]:
        """
        Interactive collection of user preferences for trading analysis

        Returns:
            Dictionary with all user preferences
        """
        print("\n" + "="*60)
        print("🤖 NSE NIFTY TRADING AGENT - SETUP")
        print("="*60)
        print("Configure your trading analysis parameters\n")

        preferences = {}

        # Step 1: Select Index
        preferences['index'] = self._select_index()

        # Step 2: Select number of stocks
        preferences['num_stocks'] = self._select_num_stocks()

        # Step 3: Select profitability target
        preferences['profitability_pct'] = self._select_profitability()

        # Step 4: Select data period
        preferences['data_days'] = self._select_data_period()

        # Step 5: Select strategy
        preferences['strategy'] = self._select_strategy()

        # Step 6: Select conviction threshold
        preferences['conviction_threshold'] = self._select_conviction_threshold()

        # Step 7: Risk parameters
        preferences['risk_params'] = self._select_risk_params()

        # Step 8: Confirm configuration
        self._confirm_configuration(preferences)

        return preferences

    def _select_index(self) -> Dict[str, Any]:
        """Select the index to analyze"""
        print("📊 STEP 1: Select Index")
        print("-" * 30)

        for i, (key, info) in enumerate(self.available_indices.items(), 1):
            print(f"{i}. {info['name']}")
            print(f"   {info['description']}")
            print(f"   Sample stocks: {info['sample_stocks']}")
            print()

        while True:
            try:
                choice = input("Enter your choice (1-4): ").strip()
                index_num = int(choice)

                if 1 <= index_num <= len(self.available_indices):
                    selected_key = list(self.available_indices.keys())[index_num - 1]
                    selected_info = self.available_indices[selected_key].copy()

                    print(f"✅ Selected: {selected_info['name']}\n")
                    return selected_info

                else:
                    print("❌ Invalid choice. Please select a valid option.")

            except ValueError:
                print("❌ Please enter a valid number.")

    def _select_num_stocks(self) -> int:
        """Select number of stocks to analyze"""
        print("📈 STEP 2: Number of Stocks to Analyze")
        print("-" * 40)

        while True:
            try:
                num_stocks = input("Enter number of top stocks to analyze (5-20): ").strip()
                num = int(num_stocks)

                if 5 <= num <= 20:
                    print(f"✅ Selected: Top {num} stocks\n")
                    return num
                else:
                    print("❌ Please enter a number between 5 and 20.")

            except ValueError:
                print("❌ Please enter a valid number.")

    def _select_profitability(self) -> float:
        """Select profitability target"""
        print("💰 STEP 3: Profitability Target")
        print("-" * 35)

        while True:
            try:
                pct_str = input("Enter target profitability percentage (5-25%): ").strip()
                # Remove % if present
                pct_str = pct_str.replace('%', '')
                pct = float(pct_str)

                if 5.0 <= pct <= 25.0:
                    print(f"✅ Selected: {pct}% target profitability\n")
                    return pct
                else:
                    print("❌ Please enter a percentage between 5% and 25%.")

            except ValueError:
                print("❌ Please enter a valid percentage.")

    def _select_data_period(self) -> int:
        """Select data period in days"""
        print("📅 STEP 4: Historical Data Period")
        print("-" * 38)

        while True:
            try:
                days_str = input("Enter number of days of historical data (30-365): ").strip()
                days = int(days_str)

                if 30 <= days <= 365:
                    print(f"✅ Selected: {days} days of historical data\n")
                    return days
                else:
                    print("❌ Please enter a number between 30 and 365 days.")

            except ValueError:
                print("❌ Please enter a valid number.")

    def _select_strategy(self) -> Dict[str, Any]:
        """Select trading strategy"""
        print("🎯 STEP 5: Trading Strategy Selection")
        print("-" * 40)

        for i, (key, strategy) in enumerate(self.available_strategies.items(), 1):
            print(f"{i}. {strategy['name']}")
            print(f"   {strategy['description']}")
            print()

        while True:
            try:
                choice = input("Enter your choice (1-7): ").strip()
                strategy_num = int(choice)

                if 1 <= strategy_num <= len(self.available_strategies):
                    selected_key = list(self.available_strategies.keys())[strategy_num - 1]
                    selected_strategy = self.available_strategies[selected_key].copy()

                    print(f"✅ Selected: {selected_strategy['name']}\n")
                    return selected_strategy

                else:
                    print("❌ Invalid choice. Please select a valid option.")

            except ValueError:
                print("❌ Please enter a valid number.")

    def _select_conviction_threshold(self) -> float:
        """Select conviction threshold"""
        print("🎚️ STEP 6: Conviction Threshold")
        print("-" * 32)
        print("Conviction represents the model's confidence in the prediction.")
        print("Higher thresholds = fewer but more reliable signals")
        print()

        while True:
            try:
                threshold_str = input("Enter conviction threshold (0.6-0.9): ").strip()
                threshold = float(threshold_str)

                if 0.6 <= threshold <= 0.9:
                    print(f"✅ Selected: {threshold:.1f} conviction threshold\n")
                    return threshold
                else:
                    print("❌ Please enter a value between 0.6 and 0.9.")

            except ValueError:
                print("❌ Please enter a valid number.")

    def _select_risk_params(self) -> Dict[str, Any]:
        """Select risk management parameters"""
        print("⚠️ STEP 7: Risk Management Parameters")
        print("-" * 40)

        risk_params = {}

        # Maximum position size
        while True:
            try:
                pos_size_str = input("Maximum position size per stock (% of capital, 1-10): ").strip()
                pos_size = float(pos_size_str)

                if 1.0 <= pos_size <= 10.0:
                    risk_params['max_position_pct'] = pos_size / 100.0
                    print(f"✅ Max position size: {pos_size}%")
                    break
                else:
                    print("❌ Please enter a percentage between 1% and 10%.")

            except ValueError:
                print("❌ Please enter a valid percentage.")

        # Stop loss percentage
        while True:
            try:
                stop_loss_str = input("Stop loss percentage (2-10): ").strip()
                stop_loss = float(stop_loss_str)

                if 2.0 <= stop_loss <= 10.0:
                    risk_params['stop_loss_pct'] = stop_loss / 100.0
                    print(f"✅ Stop loss: {stop_loss}%")
                    break
                else:
                    print("❌ Please enter a percentage between 2% and 10%.")

            except ValueError:
                print("❌ Please enter a valid percentage.")

        print()
        return risk_params

    def _confirm_configuration(self, preferences: Dict[str, Any]) -> None:
        """Display and confirm the final configuration"""
        print("📋 STEP 8: Configuration Summary")
        print("-" * 35)

        print(f"📊 Index: {preferences['index']['name']}")
        print(f"📈 Stocks to Analyze: {preferences['num_stocks']}")
        print(f"💰 Profitability Target: {preferences['profitability_pct']}%")
        print(f"📅 Data Period: {preferences['data_days']} days")
        print(f"🎯 Strategy: {preferences['strategy']['name']}")
        print(f"🎚️ Conviction Threshold: {preferences['conviction_threshold']:.1f}")
        print(f"⚠️ Max Position Size: {preferences['risk_params']['max_position_pct']*100:.1f}%")
        print(f"🛡️ Stop Loss: {preferences['risk_params']['stop_loss_pct']*100:.1f}%")
        print()

        while True:
            confirm = input("Confirm configuration? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                print("✅ Configuration confirmed!\n")
                break
            elif confirm in ['n', 'no']:
                print("❌ Configuration cancelled. Restarting setup...\n")
                # Recursive call to restart
                new_preferences = self.collect_user_preferences()
                preferences.clear()
                preferences.update(new_preferences)
                break
            else:
                print("❌ Please enter 'y' for yes or 'n' for no.")

def get_user_preferences_interactive() -> Dict[str, Any]:
    """
    Convenience function to get user preferences interactively

    Returns:
        User preferences dictionary
    """
    ui = UserInterface()
    return ui.collect_user_preferences()

def validate_user_preferences(preferences: Dict[str, Any]) -> bool:
    """
    Validate user preferences

    Args:
        preferences: User preferences dictionary

    Returns:
        True if valid, False otherwise
    """
    required_keys = [
        'index', 'num_stocks', 'profitability_pct', 'data_days',
        'strategy', 'conviction_threshold', 'risk_params'
    ]

    for key in required_keys:
        if key not in preferences:
            logger.error(f"Missing required preference: {key}")
            return False

    # Validate ranges
    if not (5 <= preferences['num_stocks'] <= 20):
        logger.error("Number of stocks must be between 5 and 20")
        return False

    if not (5.0 <= preferences['profitability_pct'] <= 25.0):
        logger.error("Profitability percentage must be between 5% and 25%")
        return False

    if not (30 <= preferences['data_days'] <= 365):
        logger.error("Data days must be between 30 and 365")
        return False

    if not (0.6 <= preferences['conviction_threshold'] <= 0.9):
        logger.error("Conviction threshold must be between 0.6 and 0.9")
        return False

    return True
