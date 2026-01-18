"""
Candle Patterns - Momentum Pattern Detection
======================================================

Detects momentum day trading patterns:
- Micro Pullback
- Bull Flag
- VWAP Break
- Opening Range Retest
- ABCD (Harmonic)

Usage:
    from candle_patterns import MicroPullback, BullFlag, VWAPBreak, ABCD

    detector = MicroPullback()
    result = detector.detect(bars_df, vwap_series)

    if result.detected:
        print(f"Entry: {result.entry_price}, Stop: {result.stop_price}")
"""

from .base import PatternResult, PatternDetector, ExitSignal, TrailingStopResult
from .micro_pullback import MicroPullback
from .bull_flag import BullFlag
from .vwap_break import VWAPBreak
from .opening_range_retest import OpeningRangeRetest
from .abcd import ABCD

__version__ = "0.1.0"
__all__ = [
    "PatternResult",
    "PatternDetector",
    "ExitSignal",
    "TrailingStopResult",
    "MicroPullback",
    "BullFlag",
    "VWAPBreak",
    "OpeningRangeRetest",
    "ABCD",
]
