"""
NSE Nifty Trading Agent Framework
A comprehensive quantitative trading system for Indian markets
"""

__version__ = "1.0.0"
__author__ = "Quantitative Trading Framework"
__description__ = "AI-powered trading agent for NSE Nifty stocks"

from .pipeline.daily_pipeline import DailyPipeline
from .models.alpha_model import AlphaModel
from .signals.signal_generation import SignalGenerator

__all__ = [
    'DailyPipeline',
    'AlphaModel',
    'SignalGenerator'
]
