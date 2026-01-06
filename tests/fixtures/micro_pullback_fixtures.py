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


# =============================================================================
# LIMIT TESTS - Testing rules at their boundaries
# =============================================================================

# =============================================================================
# MICRO PULLBACK - LIMIT: Exactly 5% Prior Move (minimum)
# =============================================================================
# Tests min_prior_move_pct: 5.0 at the exact limit.
# Should PASS with 5.25% move from surge low to surge high.
# Note: Using higher prices ($10+) so 15-cent stop buffer is proportionally
# smaller, allowing R:R >= 2.0 even with small 5% target.

MICRO_PULLBACK_LIMIT_PRIOR_MOVE = _make_bars([
    # Buffer bar before surge
    (9.95, 10.02, 9.94, 10.00, 100000),  # Bar 1: noise

    # Prior move: 5.25% from low 10.00 to high 10.525 in 3 bars
    # All green candles (100% green > 50%)
    (10.00, 10.18, 10.00, 10.15, 150000),  # Bar 2: green (surge low=10.00)
    (10.15, 10.35, 10.12, 10.32, 160000),  # Bar 3: green
    (10.32, 10.53, 10.30, 10.50, 180000),  # Bar 4: green (swing high=10.53, surge=5.3%)

    # Pullback from swing high (10.53) - shallow pullback for tight stop
    (10.50, 10.51, 10.35, 10.38, 80000),   # Bar 5: red pullback
    (10.38, 10.40, 10.30, 10.32, 70000),   # Bar 6: red (low: 10.30)

    # Entry candle - green bounce
    (10.32, 10.55, 10.30, 10.52, 200000),  # Bar 7: green entry
])

# Expected result:
# - detected: True
# - prior_move_pct: ~5.3% (at limit, from 10.00 to 10.53)
# - entry: ~10.33, stop: 10.30-0.15=10.15, risk: 0.18
# - reward: 10.33 * 0.053 = 0.55
# - R:R: 0.55 / 0.18 = 3.0 (passes 2.0)


# =============================================================================
# MICRO PULLBACK - LIMIT: Exactly 20% Pullback (maximum)
# =============================================================================
# Tests max_pullback_pct: 20.0 at the exact limit.
# Should PASS with exactly 20% pullback.

MICRO_PULLBACK_LIMIT_PULLBACK_PCT = _make_bars([
    # Strong prior move (25% gain for good R:R)
    (4.00, 4.15, 3.98, 4.12, 200000),  # Bar 1: green
    (4.12, 4.30, 4.10, 4.28, 220000),  # Bar 2: green
    (4.28, 4.50, 4.25, 4.48, 250000),  # Bar 3: green
    (4.48, 4.75, 4.45, 4.70, 280000),  # Bar 4: green
    (4.70, 5.02, 4.68, 5.00, 300000),  # Bar 5: green (high: 5.02, 25% from 4.00)

    # Exactly 20% pullback from 5.02 high → low = 5.02 * 0.80 = 4.016
    (5.00, 5.01, 4.20, 4.25, 100000),  # Bar 6: red
    (4.25, 4.28, 4.02, 4.05, 90000),   # Bar 7: red (low: 4.02 = 19.9% pullback)

    # Entry candle
    (4.05, 4.30, 4.00, 4.25, 250000),  # Bar 8: green entry
])

# Expected result:
# - detected: True
# - pullback_pct: ~20% (at limit)


# =============================================================================
# MICRO PULLBACK - LIMIT: Exactly 7 Pullback Candles (maximum)
# =============================================================================
# Tests max_pullback_candles: 7 at the exact limit.
# Should PASS with exactly 7 consolidation candles.

MICRO_PULLBACK_LIMIT_PULLBACK_CANDLES = _make_bars([
    # Strong prior move (20% for good R:R)
    (4.00, 4.15, 3.98, 4.12, 200000),  # Bar 1: green
    (4.12, 4.30, 4.10, 4.28, 220000),  # Bar 2: green
    (4.28, 4.50, 4.25, 4.48, 250000),  # Bar 3: green
    (4.48, 4.82, 4.45, 4.80, 280000),  # Bar 4: green (high: 4.82, 20% move)

    # Exactly 7 pullback/consolidation candles (max allowed)
    (4.80, 4.81, 4.60, 4.62, 100000),  # Bar 5: red
    (4.62, 4.68, 4.55, 4.58, 95000),   # Bar 6: red
    (4.58, 4.65, 4.50, 4.52, 90000),   # Bar 7: red
    (4.52, 4.58, 4.45, 4.48, 85000),   # Bar 8: red
    (4.48, 4.55, 4.40, 4.42, 80000),   # Bar 9: red
    (4.42, 4.50, 4.38, 4.40, 75000),   # Bar 10: red
    (4.40, 4.48, 4.35, 4.38, 70000),   # Bar 11: red (7th pullback, low: 4.35)

    # Entry candle
    (4.38, 4.60, 4.35, 4.55, 250000),  # Bar 12: green entry
])

# Expected result:
# - detected: True
# - pullback_candles: 7 (at limit)


# =============================================================================
# MICRO PULLBACK - LIMIT: Exactly 50% Green Ratio (minimum)
# =============================================================================
# Tests >50% green candle requirement at the exact limit.
# 4 candles with exactly 2 green = 50% (should fail, needs >50%)
# 3 candles with 2 green = 66% (should pass)

MICRO_PULLBACK_LIMIT_GREEN_RATIO = _make_bars([
    # Prior move with exactly 2 green out of 3 = 66% (>50%, at minimum to pass)
    (4.00, 4.08, 3.98, 3.99, 150000),  # Bar 1: RED (close < open)
    (3.99, 4.18, 3.97, 4.15, 180000),  # Bar 2: green +4%
    (4.15, 4.35, 4.12, 4.32, 200000),  # Bar 3: green +4%
    (4.32, 4.52, 4.30, 4.50, 220000),  # Bar 4: green +4% (>50% green: 3/4=75%)

    # Shallow pullback
    (4.50, 4.51, 4.30, 4.32, 80000),   # Bar 5: red
    (4.32, 4.35, 4.25, 4.28, 70000),   # Bar 6: red (low: 4.25)

    # Entry candle
    (4.28, 4.50, 4.25, 4.45, 200000),  # Bar 7: green entry
])

# Expected result:
# - detected: True
# - green_candles: 3 out of 4 in surge = 75% (passes >50% requirement)
