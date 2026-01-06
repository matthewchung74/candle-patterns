"""
Micro Pullback Test Fixtures
============================

Real-world example bars for testing micro pullback detection.
"""

import pandas as pd
from datetime import datetime, timedelta

# Helper to create bar data
def _make_bars(data: list) -> pd.DataFrame:
    """Create DataFrame from OHLCV tuples."""
    base_time = datetime(2025, 1, 15, 9, 30)
    rows = []
    for i, (o, h, l, c, v) in enumerate(data):
        rows.append({
            "timestamp": base_time + timedelta(minutes=i),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
    return pd.DataFrame(rows)


# =============================================================================
# VALID MICRO PULLBACK
# =============================================================================
# Pattern: Stock surges 15% with 4 green candles, pulls back 5% with 2 red
# candles, then bounces on green candle.
#
# Designed to pass R:R check (min 2.0):
# - Prior move: 15% → estimated target +15% from entry
# - Pullback: 5% with tight stop → good R:R
#
# Visual:
#   Bar 1-4: Strong move from $4.00 to $4.60 (15% gain, 4 green)
#   Bar 5-6: Shallow pullback to $4.40 (4% dip, 2 red)
#   Bar 7: Bounce candle (entry candle, green)

MICRO_PULLBACK_VALID = _make_bars([
    # Strong prior move (4 green candles, ~15% move)
    (4.00, 4.12, 3.98, 4.10, 150000),  # Bar 1: +2.5%
    (4.10, 4.25, 4.08, 4.22, 180000),  # Bar 2: +2.9%
    (4.22, 4.40, 4.20, 4.38, 200000),  # Bar 3: +3.8%
    (4.38, 4.62, 4.36, 4.60, 220000),  # Bar 4: +5.0% (high of move: 4.62)

    # Shallow pullback (2 red candles, ~5% dip from high)
    (4.60, 4.61, 4.45, 4.48, 100000),  # Bar 5: -2.6% (red)
    (4.48, 4.50, 4.38, 4.40, 80000),   # Bar 6: -1.8% (red, pullback low: 4.38)

    # Entry candle - bounce from pullback
    (4.40, 4.55, 4.38, 4.52, 250000),  # Bar 7: green bounce (entry ~4.41)
])

# Expected result:
# - detected: True
# - entry_price: ~4.41 (open + 0.01)
# - stop_price: ~4.23 (pullback low 4.38 - 0.15)
# - risk: ~0.18 ($4.41 - $4.23)
# - target: ~4.41 + 15% = ~5.07
# - R:R: (5.07 - 4.41) / 0.18 = ~3.7 (passes 2.0 min)


# =============================================================================
# MICRO PULLBACK - TOO DEEP
# =============================================================================
# Pattern: Good prior move, but pullback is too deep (25% vs max 20%)
# Should NOT detect as valid micro pullback.

MICRO_PULLBACK_TOO_DEEP = _make_bars([
    # Strong prior move (15% gain)
    (4.00, 4.12, 3.98, 4.10, 150000),
    (4.10, 4.25, 4.08, 4.22, 180000),
    (4.22, 4.40, 4.20, 4.38, 200000),
    (4.38, 4.62, 4.36, 4.60, 220000),  # High: 4.62

    # Deep pullback (25% dip from high - exceeds 20% max)
    # 4.62 * 0.75 = 3.465, so low needs to be ~3.46
    (4.60, 4.61, 4.00, 4.05, 120000),  # Bar 5: big red
    (4.05, 4.10, 3.45, 3.50, 100000),  # Bar 6: low: 3.45 = 25% from 4.62

    # Attempted bounce
    (3.50, 3.80, 3.48, 3.75, 200000),  # Bar 7: green bounce attempt
])

# Expected result:
# - detected: False
# - reason: "Pullback too deep: 25.x% > 20%"


# =============================================================================
# MICRO PULLBACK - NO PRIOR MOVE
# =============================================================================
# Pattern: Only 2 green candles before pullback (need 3+)
# Should NOT detect as valid micro pullback.

MICRO_PULLBACK_NO_PRIOR_MOVE = _make_bars([
    # Some noise before
    (4.95, 4.98, 4.92, 4.96, 100000),  # Bar 1: noise

    # Weak prior move (only 2 green candles)
    (4.96, 5.08, 4.95, 5.07, 150000),  # Bar 2: green
    (5.07, 5.18, 5.05, 5.16, 180000),  # Bar 3: green (only 2 green)

    # Pullback
    (5.16, 5.17, 5.10, 5.12, 80000),   # Bar 4: red
    (5.12, 5.14, 5.08, 5.09, 70000),   # Bar 5: red

    # Entry attempt
    (5.09, 5.20, 5.08, 5.19, 150000),  # Bar 6: green, new high
])

# Expected result:
# - detected: False
# - reason: "Prior move too short: 2 green candles < 3"


# =============================================================================
# MICRO PULLBACK - PULLBACK TOO LONG
# =============================================================================
# Pattern: Good prior move, but 4 red candles in pullback (max is 2)

MICRO_PULLBACK_TOO_LONG = _make_bars([
    # Strong prior move
    (5.00, 5.08, 4.98, 5.07, 150000),
    (5.07, 5.18, 5.05, 5.16, 180000),
    (5.16, 5.28, 5.14, 5.26, 200000),
    (5.26, 5.42, 5.24, 5.40, 220000),

    # Long pullback (4 red candles - too many)
    (5.40, 5.41, 5.35, 5.36, 90000),
    (5.36, 5.37, 5.32, 5.33, 85000),
    (5.33, 5.34, 5.29, 5.30, 80000),
    (5.30, 5.31, 5.27, 5.28, 75000),  # 4th red candle

    # Entry attempt
    (5.28, 5.45, 5.27, 5.43, 200000),
])

# Expected result:
# - detected: False
# - reason: "Pullback too long: 4 candles > 2"


# =============================================================================
# MICRO PULLBACK - WAITING FOR ENTRY
# =============================================================================
# Pattern: Valid setup but last candle is still red (no entry yet)

MICRO_PULLBACK_WAITING = _make_bars([
    # Strong prior move
    (5.00, 5.08, 4.98, 5.07, 150000),
    (5.07, 5.18, 5.05, 5.16, 180000),
    (5.16, 5.28, 5.14, 5.26, 200000),
    (5.26, 5.42, 5.24, 5.40, 220000),

    # Pullback in progress
    (5.40, 5.41, 5.32, 5.34, 100000),
    (5.34, 5.36, 5.29, 5.30, 80000),  # Last bar is still red
])

# Expected result:
# - detected: False
# - reason: "Last candle is red - waiting for green entry candle"
