"""
Candle Patterns - Ross Cameron Style Pattern Detection
======================================================

Detects momentum day trading patterns:
- Micro Pullback
- Bull Flag
- VWAP Break

Usage:
    from candle_patterns import MicroPullback, BullFlag, VWAPBreak

    detector = MicroPullback()
    result = detector.detect(bars_df, vwap_series)

    if result.detected:
        print(f"Entry: {result.entry_price}, Stop: {result.stop_price}")
"""

from .base import PatternResult, PatternDetector, ExitSignal
from .micro_pullback import MicroPullback
from .bull_flag import BullFlag
from .vwap_break import VWAPBreak

__version__ = "0.1.0"
__all__ = [
    "PatternResult",
    "PatternDetector",
    "ExitSignal",
    "MicroPullback",
    "BullFlag",
    "VWAPBreak",
]
