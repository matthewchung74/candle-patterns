"""
Candle Patterns - Momentum Pattern Detection
======================================================

Detects momentum day trading patterns:
- Micro Pullback (long)
- VwapBounce (long)
- ParabolicExhaustion (short)

Usage:
    from candle_patterns import MicroPullback

    detector = MicroPullback()
    result = detector.detect(bars_df, vwap_series)

    if result.detected:
        print(f"Entry: {result.entry_price}, Stop: {result.stop_price}")
"""

from .base import PatternResult, PatternDetector, ExitSignal
from .micro_pullback import MicroPullback
from .parabolic_exhaustion import ParabolicExhaustion
from .vwap_bounce import VwapBounce

# Trailing stop module exports
from .trailing import (
    calculate_trailing_stop,
    TrailingStopState,
    TrailingStopConfig,
    TrailingStopResult,
)

__version__ = "0.1.0"
__all__ = [
    "PatternResult",
    "PatternDetector",
    "ExitSignal",
    # Trailing stop
    "calculate_trailing_stop",
    "TrailingStopState",
    "TrailingStopConfig",
    "TrailingStopResult",
    # Pattern detectors
    "MicroPullback",
    "ParabolicExhaustion",
    "VwapBounce",
]
