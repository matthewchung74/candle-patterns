"""
Reversal Pattern Test Fixtures
==============================

OHLCV DataFrames for testing reversal pattern detection.

Patterns tested:
- Shooting star
- Bearish engulfing
- Evening star
- Volume climax

All fixtures are designed to test boundary conditions.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _make_bars(data: list[dict]) -> pd.DataFrame:
    """Helper to create DataFrame from bar data."""
    df = pd.DataFrame(data)
    df["timestamp"] = pd.date_range(
        start=datetime(2024, 1, 15, 9, 30),
        periods=len(data),
        freq="1min"
    )
    return df


# =============================================================================
# SHOOTING STAR FIXTURES
# =============================================================================

# PASS: Valid shooting star at HOD after uptrend
REVERSAL_PASS_SHOOTING_STAR = _make_bars([
    # Initial uptrend (green bars)
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    # Small consolidation
    {"open": 1.25, "high": 1.27, "low": 1.23, "close": 1.26, "volume": 12000},
    {"open": 1.26, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 11000},
    {"open": 1.27, "high": 1.29, "low": 1.25, "close": 1.28, "volume": 10000},
    # Shooting star: long upper wick, body in lower third
    # Open=1.28, High=1.40 (big wick), Low=1.27, Close=1.29
    # Upper wick = 0.11, Body = 0.01, Ratio = 11x
    # Body position = (1.28-1.27)/(1.40-1.27) = 0.077 = 7.7% (in lower third)
    {"open": 1.28, "high": 1.40, "low": 1.27, "close": 1.29, "volume": 25000},
])

# FAIL: Shooting star but no prior uptrend
REVERSAL_FAIL_SHOOTING_STAR_NO_UPTREND = _make_bars([
    # Sideways/down movement (not extended from open)
    {"open": 1.00, "high": 1.02, "low": 0.98, "close": 1.01, "volume": 8000},
    {"open": 1.20, "high": 1.22, "low": 1.15, "close": 1.16, "volume": 10000},
    {"open": 1.16, "high": 1.18, "low": 1.14, "close": 1.15, "volume": 12000},
    {"open": 1.15, "high": 1.17, "low": 1.12, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.16, "low": 1.11, "close": 1.12, "volume": 18000},
    {"open": 1.12, "high": 1.14, "low": 1.10, "close": 1.11, "volume": 20000},
    # Small bounce
    {"open": 1.11, "high": 1.14, "low": 1.10, "close": 1.13, "volume": 12000},
    {"open": 1.13, "high": 1.15, "low": 1.12, "close": 1.14, "volume": 11000},
    {"open": 1.14, "high": 1.16, "low": 1.13, "close": 1.15, "volume": 10000},
    # Shooting star shape but no uptrend (also not extended - close 1.16 vs open 1.00 = 16%)
    {"open": 1.15, "high": 1.26, "low": 1.14, "close": 1.16, "volume": 25000},
])

# FAIL: Upper wick ratio too small
REVERSAL_FAIL_SHOOTING_STAR_SMALL_WICK = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    {"open": 1.25, "high": 1.27, "low": 1.23, "close": 1.26, "volume": 12000},
    {"open": 1.26, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 11000},
    {"open": 1.27, "high": 1.29, "low": 1.25, "close": 1.28, "volume": 10000},
    # Small upper wick (not a shooting star)
    # Upper wick = 0.02, Body = 0.04, Ratio = 0.5x (needs 2x)
    {"open": 1.28, "high": 1.34, "low": 1.27, "close": 1.32, "volume": 25000},
])


# =============================================================================
# BEARISH ENGULFING FIXTURES
# =============================================================================

# PASS: Valid bearish engulfing near HOD
REVERSAL_PASS_BEARISH_ENGULFING = _make_bars([
    # Uptrend to HOD
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    # Small consolidation
    {"open": 1.25, "high": 1.27, "low": 1.23, "close": 1.26, "volume": 12000},
    {"open": 1.26, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 11000},
    # Green candle (will be engulfed)
    # Body: 1.27 -> 1.31 (green)
    {"open": 1.27, "high": 1.32, "low": 1.26, "close": 1.31, "volume": 15000},
    # Red candle engulfing: opens above 1.31, closes below 1.27
    {"open": 1.33, "high": 1.34, "low": 1.23, "close": 1.24, "volume": 30000},
])

# FAIL: Red candle doesn't fully engulf
REVERSAL_FAIL_BEARISH_ENGULF_PARTIAL = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    {"open": 1.25, "high": 1.27, "low": 1.23, "close": 1.26, "volume": 12000},
    {"open": 1.26, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 11000},
    # Green candle
    {"open": 1.27, "high": 1.32, "low": 1.26, "close": 1.31, "volume": 15000},
    # Red candle but doesn't engulf bottom (closes at 1.28, above 1.27 open)
    {"open": 1.32, "high": 1.33, "low": 1.27, "close": 1.28, "volume": 25000},
])


# =============================================================================
# EVENING STAR FIXTURES
# =============================================================================

# PASS: Valid evening star pattern
REVERSAL_PASS_EVENING_STAR = _make_bars([
    # Uptrend
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    {"open": 1.25, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 15000},
    # Bar 1 of evening star: Strong green candle (body > 50% of range)
    # Range = 0.08, Body = 0.05 = 62.5% (good)
    {"open": 1.27, "high": 1.35, "low": 1.26, "close": 1.32, "volume": 18000},
    # Bar 2: Small body (doji-like) at top
    # Range = 0.04, Body = 0.01 = 25% (small, good)
    {"open": 1.33, "high": 1.36, "low": 1.32, "close": 1.34, "volume": 10000},
    # Bar 3: Strong red, closes below Bar 1 midpoint ((1.27+1.32)/2 = 1.295)
    {"open": 1.33, "high": 1.34, "low": 1.22, "close": 1.24, "volume": 28000},
])

# FAIL: Middle bar body too large
REVERSAL_FAIL_EVENING_STAR_LARGE_MIDDLE = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    {"open": 1.25, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 15000},
    # Bar 1: Strong green
    {"open": 1.27, "high": 1.35, "low": 1.26, "close": 1.32, "volume": 18000},
    # Bar 2: Large body (not doji-like) - Body > 30% of range
    # Range = 0.06, Body = 0.04 = 66% (too large)
    {"open": 1.32, "high": 1.37, "low": 1.31, "close": 1.36, "volume": 15000},
    # Bar 3: Red
    {"open": 1.35, "high": 1.36, "low": 1.22, "close": 1.24, "volume": 28000},
])

# FAIL: Final bar doesn't close below midpoint
REVERSAL_FAIL_EVENING_STAR_HIGH_CLOSE = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 10000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 12000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 15000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 18000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 20000},
    {"open": 1.25, "high": 1.28, "low": 1.24, "close": 1.27, "volume": 15000},
    # Bar 1: Green (midpoint = (1.27+1.32)/2 = 1.295)
    {"open": 1.27, "high": 1.35, "low": 1.26, "close": 1.32, "volume": 18000},
    # Bar 2: Small body
    {"open": 1.33, "high": 1.36, "low": 1.32, "close": 1.34, "volume": 10000},
    # Bar 3: Red but closes above midpoint (1.30 > 1.295)
    {"open": 1.33, "high": 1.34, "low": 1.28, "close": 1.30, "volume": 20000},
])


# =============================================================================
# VOLUME CLIMAX FIXTURES
# =============================================================================

# PASS: Volume climax with red reversal candle
REVERSAL_PASS_VOLUME_CLIMAX = _make_bars([
    # Buildup with normal volume (~10k avg)
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 9000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 10000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 11000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 12000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 11000},
    {"open": 1.25, "high": 1.30, "low": 1.24, "close": 1.29, "volume": 10000},
    {"open": 1.29, "high": 1.34, "low": 1.28, "close": 1.33, "volume": 10000},
    # Volume climax at HOD: 40000 > 3x avg (10000)
    # Red candle with climax volume
    {"open": 1.34, "high": 1.38, "low": 1.28, "close": 1.30, "volume": 40000},
    # Follow through red
    {"open": 1.30, "high": 1.32, "low": 1.25, "close": 1.26, "volume": 25000},
])

# PASS: Volume climax with topping tail (green but rejection)
REVERSAL_PASS_VOLUME_CLIMAX_TOPPING_TAIL = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 9000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 10000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 11000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 12000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 11000},
    {"open": 1.25, "high": 1.30, "low": 1.24, "close": 1.29, "volume": 10000},
    {"open": 1.29, "high": 1.34, "low": 1.28, "close": 1.33, "volume": 10000},
    # Volume climax with topping tail (green but long upper wick)
    # Upper wick = 0.07, Body = 0.02, Ratio = 3.5x (topping tail)
    {"open": 1.33, "high": 1.42, "low": 1.32, "close": 1.35, "volume": 45000},
    # Red follow through
    {"open": 1.34, "high": 1.36, "low": 1.27, "close": 1.28, "volume": 25000},
])

# FAIL: High volume but not at HOD
REVERSAL_FAIL_VOLUME_CLIMAX_NOT_HOD = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 9000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 10000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 11000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 12000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 11000},
    # HOD here at 1.50 - significantly higher than where volume climax occurs
    {"open": 1.25, "high": 1.50, "low": 1.24, "close": 1.45, "volume": 15000},
    # Pullback from HOD
    {"open": 1.45, "high": 1.46, "low": 1.35, "close": 1.38, "volume": 12000},
    # High volume but NOT at HOD (HOD was 1.50, current high is 1.36)
    # Distance from HOD: (1.50-1.36)/1.50 = 9.3% (> 5% threshold)
    {"open": 1.38, "high": 1.36, "low": 1.28, "close": 1.30, "volume": 45000},
    {"open": 1.30, "high": 1.32, "low": 1.26, "close": 1.28, "volume": 20000},
])

# FAIL: High volume but no reversal confirmation
REVERSAL_FAIL_VOLUME_CLIMAX_NO_REVERSAL = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 9000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 10000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 11000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 12000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 11000},
    {"open": 1.25, "high": 1.30, "low": 1.24, "close": 1.29, "volume": 10000},
    {"open": 1.29, "high": 1.34, "low": 1.28, "close": 1.33, "volume": 10000},
    # High volume but green candle with small wick (continuation, not reversal)
    {"open": 1.33, "high": 1.42, "low": 1.32, "close": 1.40, "volume": 45000},
    # More green continuation
    {"open": 1.40, "high": 1.45, "low": 1.39, "close": 1.44, "volume": 30000},
])


# =============================================================================
# EXTENSION REQUIREMENT FIXTURES
# =============================================================================

# FAIL: Not extended enough from open (<20%)
REVERSAL_FAIL_NOT_EXTENDED = _make_bars([
    # Only 15% extension from open
    {"open": 1.00, "high": 1.02, "low": 0.99, "close": 1.01, "volume": 8000},
    {"open": 1.01, "high": 1.03, "low": 1.00, "close": 1.02, "volume": 10000},
    {"open": 1.02, "high": 1.04, "low": 1.01, "close": 1.03, "volume": 12000},
    {"open": 1.03, "high": 1.06, "low": 1.02, "close": 1.05, "volume": 15000},
    {"open": 1.05, "high": 1.08, "low": 1.04, "close": 1.07, "volume": 18000},
    {"open": 1.07, "high": 1.10, "low": 1.06, "close": 1.09, "volume": 20000},
    {"open": 1.09, "high": 1.12, "low": 1.08, "close": 1.11, "volume": 18000},
    {"open": 1.11, "high": 1.14, "low": 1.10, "close": 1.13, "volume": 16000},
    {"open": 1.13, "high": 1.15, "low": 1.12, "close": 1.14, "volume": 14000},
    # Shooting star shape but only 15% from open (1.15/1.00 = 15%)
    {"open": 1.14, "high": 1.24, "low": 1.13, "close": 1.15, "volume": 25000},
])


# =============================================================================
# MULTIPLE PATTERNS (for testing priority)
# =============================================================================

# This fixture has both shooting star AND volume climax
# Volume climax should take priority
REVERSAL_PASS_MULTI_PATTERN = _make_bars([
    {"open": 1.00, "high": 1.03, "low": 0.99, "close": 1.02, "volume": 8000},
    {"open": 1.02, "high": 1.05, "low": 1.01, "close": 1.04, "volume": 9000},
    {"open": 1.04, "high": 1.10, "low": 1.03, "close": 1.09, "volume": 10000},
    {"open": 1.09, "high": 1.15, "low": 1.08, "close": 1.14, "volume": 11000},
    {"open": 1.14, "high": 1.20, "low": 1.13, "close": 1.19, "volume": 12000},
    {"open": 1.19, "high": 1.26, "low": 1.18, "close": 1.25, "volume": 11000},
    {"open": 1.25, "high": 1.30, "low": 1.24, "close": 1.29, "volume": 10000},
    {"open": 1.29, "high": 1.34, "low": 1.28, "close": 1.33, "volume": 10000},
    # Shooting star WITH volume climax: both patterns present
    {"open": 1.33, "high": 1.45, "low": 1.32, "close": 1.34, "volume": 50000},
    {"open": 1.33, "high": 1.35, "low": 1.28, "close": 1.29, "volume": 25000},
])
