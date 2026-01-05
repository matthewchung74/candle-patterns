"""
Test Fixtures for Candle Patterns
=================================

Contains example bar data for testing pattern detection.
Each fixture is based on real market scenarios.
"""

from .micro_pullback_fixtures import (
    MICRO_PULLBACK_VALID,
    MICRO_PULLBACK_TOO_DEEP,
    MICRO_PULLBACK_NO_PRIOR_MOVE,
)
from .bull_flag_fixtures import (
    BULL_FLAG_VALID,
    BULL_FLAG_NO_BREAKOUT,
)
from .vwap_break_fixtures import (
    VWAP_BREAK_VALID,
    VWAP_HOLD_VALID,
)

__all__ = [
    "MICRO_PULLBACK_VALID",
    "MICRO_PULLBACK_TOO_DEEP",
    "MICRO_PULLBACK_NO_PRIOR_MOVE",
    "BULL_FLAG_VALID",
    "BULL_FLAG_NO_BREAKOUT",
    "VWAP_BREAK_VALID",
    "VWAP_HOLD_VALID",
]
