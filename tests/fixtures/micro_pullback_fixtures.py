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
# Pattern: Stock surges 8% with 4 green candles, pulls back 2% with 2 red
# candles, then makes new high on green candle.
#
# Visual:
#   Bar 1-4: Strong move from $5.00 to $5.40 (8% gain, 4 green)
#   Bar 5-6: Shallow pullback to $5.30 (2% dip, 2 red)
#   Bar 7: New high at $5.45 (entry candle, green)

MICRO_PULLBACK_VALID = _make_bars([
    # Strong prior move (4 green candles, ~8% move)
    (5.00, 5.08, 4.98, 5.07, 150000),  # Bar 1: +1.4%
    (5.07, 5.18, 5.05, 5.16, 180000),  # Bar 2: +1.8%
    (5.16, 5.28, 5.14, 5.26, 200000),  # Bar 3: +1.9%
    (5.26, 5.42, 5.24, 5.40, 220000),  # Bar 4: +2.7% (high of move)

    # Shallow pullback (2 red candles, ~2% dip)
    (5.40, 5.41, 5.32, 5.34, 100000),  # Bar 5: -1.1% (red)
    (5.34, 5.36, 5.29, 5.30, 80000),   # Bar 6: -0.7% (red, pullback low)

    # Entry candle - new high
    (5.30, 5.45, 5.28, 5.43, 250000),  # Bar 7: NEW HIGH (green, entry)
])

# Expected result:
# - detected: True
# - entry_price: ~5.43 (above prior high of 5.42)
# - stop_price: ~5.14 (below pullback low of 5.29)
# - prior_move_pct: ~8%
# - pullback_pct: ~2%


# =============================================================================
# MICRO PULLBACK - TOO DEEP
# =============================================================================
# Pattern: Good prior move, but pullback is too deep (5% vs max 3%)
# Should NOT detect as valid micro pullback.

MICRO_PULLBACK_TOO_DEEP = _make_bars([
    # Strong prior move
    (5.00, 5.08, 4.98, 5.07, 150000),
    (5.07, 5.18, 5.05, 5.16, 180000),
    (5.16, 5.28, 5.14, 5.26, 200000),
    (5.26, 5.42, 5.24, 5.40, 220000),  # High: 5.42

    # Deep pullback (5% dip - too deep for micro)
    (5.40, 5.41, 5.20, 5.22, 120000),  # Bar 5: -3.3% (red)
    (5.22, 5.24, 5.12, 5.15, 100000),  # Bar 6: -1.3% (red, low: 5.12 = 5.5% from high)

    # Attempted new high
    (5.15, 5.44, 5.14, 5.42, 200000),  # Bar 7: recovery attempt
])

# Expected result:
# - detected: False
# - reason: "Pullback too deep: 5.5% > 3%"


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
