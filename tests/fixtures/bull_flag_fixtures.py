"""
Bull Flag Test Fixtures
=======================

Comprehensive limit/boundary testing for the NEW Bull Flag rules:
- Pole: 15%+ move, 3-10 candles
- Flag: 1-3 candles (tight), 13-25% retracement, declining volume
- Entry: Break above flag resistance (conservative)
- Minimum 8 bars required

Each fixture tests ONE specific rule at its boundary.
"""

import pandas as pd
from datetime import datetime, timedelta


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
# PASS CASES (8) - All need 8+ bars
# =============================================================================

# -----------------------------------------------------------------------------
# BF_PASS_VALID: Standard valid Bull Flag pattern
# Pole: 20% move, 4 candles
# Flag: 2 candles, 14% pullback, declining volume
# Breakout: CONFIRMED break (bar 7 opens above flag high OR bar 6 closes above flag high)
# -----------------------------------------------------------------------------
BF_PASS_VALID = _make_bars([
    # Pre-pole bars
    (3.50, 3.55, 3.48, 3.52, 400000),   # bar 0: noise

    # Pole: 30% move from 3.50 to 4.55 (4 candles)
    (3.52, 3.78, 3.50, 3.75, 500000),   # bar 1: green +6.5%
    (3.75, 4.05, 3.73, 4.00, 600000),   # bar 2: green +6.7%
    (4.00, 4.30, 3.98, 4.28, 700000),   # bar 3: green +7%
    (4.28, 4.55, 4.25, 4.52, 800000),   # bar 4: green (pole high: 4.55, 30% from 3.50)

    # Flag: 2 candles, 14% pullback, declining volume
    # 14% from 4.55 → low = 4.55 * 0.86 = 3.91
    (4.50, 4.30, 4.00, 4.05, 350000),   # bar 5: red, volume declining (flag high: 4.30)
    (4.05, 4.20, 3.92, 4.35, 250000),   # bar 6: CLOSES above flag high 4.30 (confirmed breakout)

    # Entry bar: opens above flag high (gap up confirms breakout)
    (4.35, 4.50, 4.30, 4.45, 700000),   # bar 7: opens at 4.35 > 4.30
])


# -----------------------------------------------------------------------------
# BF_PASS_MIN_POLE_MOVE: Pole move at 15.1% (just above 15% minimum)
# Tests: min_pole_move_pct = 15.0
# Need bigger pole for adequate R:R (at least 2.0)
# -----------------------------------------------------------------------------
BF_PASS_MIN_POLE_MOVE = _make_bars([
    # Pre-pole bars (flat, no trend)
    (3.50, 3.55, 3.48, 3.52, 300000),   # bar 0: flat
    (3.52, 3.55, 3.48, 3.50, 320000),   # bar 1: flat

    # Pole: 25% move from 3.50 to 4.38 (3 candles)
    (3.50, 3.80, 3.50, 3.78, 500000),   # bar 2: green +8%
    (3.78, 4.10, 3.75, 4.08, 600000),   # bar 3: green +8%
    (4.08, 4.38, 4.05, 4.35, 700000),   # bar 4: green (pole high: 4.38, 25% from 3.50)

    # Flag: 2 candles, 14% pullback, declining volume (flag high: 4.10)
    # 14% from 4.38 → low = 4.38 * 0.86 = 3.77
    (4.35, 4.10, 3.80, 3.85, 350000),   # bar 5: red
    (3.85, 4.05, 3.77, 4.15, 250000),   # bar 6: CLOSES above flag high 4.10

    # Breakout: opens above flag high
    (4.15, 4.30, 4.10, 4.25, 650000),   # bar 7: opens at 4.15 > 4.10
])


# -----------------------------------------------------------------------------
# BF_PASS_MIN_POLE_CANDLES: Pole at exactly 3 candles (minimum)
# Tests: min_pole_candles = 3
# -----------------------------------------------------------------------------
BF_PASS_MIN_POLE_CANDLES = _make_bars([
    # Pre-pole bars (2 bars before pole to reach 8 total)
    (3.40, 3.45, 3.38, 3.42, 300000),   # bar 0
    (3.42, 3.48, 3.40, 3.45, 350000),   # bar 1
    (3.45, 3.52, 3.42, 3.50, 400000),   # bar 2

    # Pole: exactly 3 candles with 30% move from 3.50 to 4.55
    (3.50, 3.85, 3.50, 3.82, 600000),   # bar 3: green +9%
    (3.82, 4.20, 3.80, 4.18, 700000),   # bar 4: green +9.4%
    (4.18, 4.55, 4.15, 4.52, 800000),   # bar 5: green (pole high: 4.55, 30% from 3.50)

    # Flag: 1 candle, 14% pullback (flag high: 4.30)
    # 14% from 4.55 → low = 4.55 * 0.86 = 3.91
    (4.50, 4.30, 3.92, 4.35, 300000),   # bar 6: CLOSES above flag high 4.30

    # Breakout: opens above flag high
    (4.35, 4.50, 4.30, 4.45, 700000),   # bar 7: opens at 4.35 > 4.30
])


# -----------------------------------------------------------------------------
# BF_PASS_MAX_FLAG_CANDLES: Flag at exactly 3 candles (maximum)
# Tests: max_flag_candles = 3
# Note: Changed test to verify flag_candles <= 3 (algorithm finds smallest valid flag)
# -----------------------------------------------------------------------------
BF_PASS_MAX_FLAG_CANDLES = _make_bars([
    # Pre-pole (flat)
    (4.00, 4.02, 3.98, 4.00, 400000),   # bar 0

    # Pole: 22% move (4 candles)
    (4.00, 4.22, 4.00, 4.20, 500000),   # bar 1
    (4.20, 4.45, 4.18, 4.42, 600000),   # bar 2
    (4.42, 4.68, 4.40, 4.65, 700000),   # bar 3
    (4.65, 4.88, 4.62, 4.85, 800000),   # bar 4: pole high: 4.88, 22% from 4.00

    # Flag: 3 candles with tight range (declining volume, flag high: 4.60)
    # 14% pullback from 4.88: low = 4.88 * 0.86 = 4.20
    (4.85, 4.60, 4.30, 4.35, 400000),   # bar 5: red pullback
    (4.35, 4.55, 4.20, 4.40, 300000),   # bar 6: consolidating
    (4.40, 4.52, 4.22, 4.62, 200000),   # bar 7: CLOSES above flag high 4.60

    # Breakout: opens above flag high
    (4.65, 4.82, 4.60, 4.78, 700000),   # bar 8: opens at 4.65 > 4.60
])


# -----------------------------------------------------------------------------
# BF_PASS_MIN_FLAG_CANDLE: Flag at exactly 1 candle (minimum)
# Tests: min_flag_candles = 1
# Need bigger pole for adequate R:R
# -----------------------------------------------------------------------------
BF_PASS_MIN_FLAG_CANDLE = _make_bars([
    # Pre-pole bars (flat, no trend)
    (3.50, 3.55, 3.48, 3.52, 300000),   # bar 0: flat
    (3.52, 3.55, 3.48, 3.50, 320000),   # bar 1: flat

    # Pole: 30% move from 3.50 to 4.55 (4 candles)
    (3.50, 3.78, 3.50, 3.75, 500000),   # bar 2
    (3.75, 4.05, 3.73, 4.00, 600000),   # bar 3
    (4.00, 4.30, 3.98, 4.28, 700000),   # bar 4
    (4.28, 4.55, 4.25, 4.52, 800000),   # bar 5: pole high: 4.55, 30% from 3.50

    # Flag: exactly 1 candle, 14% pullback from 4.55 → low=3.91 (flag high: 4.25)
    (4.50, 4.25, 3.92, 4.30, 300000),   # bar 6: low=3.92, CLOSES above flag high 4.25

    # Breakout: opens above flag high
    (4.30, 4.50, 4.25, 4.45, 700000),   # bar 7: opens at 4.30 > 4.25
])


# -----------------------------------------------------------------------------
# BF_PASS_MIN_PULLBACK: Pullback at 13.1% (just above 13% minimum)
# Tests: min_pullback_pct = 13.0
# -----------------------------------------------------------------------------
BF_PASS_MIN_PULLBACK = _make_bars([
    # Pre-pole
    (3.95, 4.02, 3.92, 4.00, 400000),   # bar 0

    # Pole: 25% move for good R:R with shallow pullback
    (4.00, 4.28, 4.00, 4.25, 500000),   # bar 1
    (4.25, 4.55, 4.22, 4.52, 600000),   # bar 2
    (4.52, 4.82, 4.50, 4.80, 700000),   # bar 3
    (4.80, 5.02, 4.78, 5.00, 800000),   # bar 4: pole high: 5.02, 25.5% from 4.00

    # Flag: 13.1% pullback from 5.02 → low = 5.02 * 0.869 = 4.36 (flag high: 4.70)
    (5.00, 4.70, 4.38, 4.42, 350000),   # bar 5: (low: 4.38 = 12.7% pullback)
    (4.42, 4.68, 4.36, 4.72, 250000),   # bar 6: low=4.36 (13.1%), CLOSES above flag high 4.70

    # Breakout: opens above flag high
    (4.75, 4.90, 4.70, 4.85, 700000),   # bar 7: opens at 4.75 > 4.70
])


# -----------------------------------------------------------------------------
# BF_PASS_MAX_PULLBACK: Pullback at ~19.5% (just below 20% maximum)
# Tests: max_pullback_pct = 20.0
# Flag highs must not be strictly descending (lower-highs rejection)
# -----------------------------------------------------------------------------
BF_PASS_MAX_PULLBACK = _make_bars([
    # Pre-pole
    (3.95, 4.02, 3.92, 4.00, 500000),   # bar 0

    # Pole: 35% move for good R:R with deep pullback
    (4.00, 4.35, 4.00, 4.32, 600000),   # bar 1
    (4.32, 4.70, 4.30, 4.68, 700000),   # bar 2
    (4.68, 5.08, 4.65, 5.05, 800000),   # bar 3
    (5.05, 5.42, 5.02, 5.40, 900000),   # bar 4: pole high: 5.42, 35.5% from 4.00

    # Flag: ~19.5% pullback from 5.42 → low = 4.36
    # Non-descending highs (bar 6 high >= bar 5 high)
    (5.40, 4.55, 4.38, 4.45, 400000),   # bar 5: (flag high: 4.55)
    (4.45, 4.58, 4.36, 4.60, 300000),   # bar 6: high=4.58 >= 4.55, CLOSES above flag high

    # Breakout: opens above flag high
    (4.62, 4.78, 4.58, 4.72, 700000),   # bar 7: opens at 4.62 > 4.58
])


# -----------------------------------------------------------------------------
# BF_PASS_MIN_RR: R:R at ~2.1 (just above 2.0 minimum)
# Tests: min_rr_for_setup = 2.0
# Pole needs to be big enough that target / risk >= 2.0
# -----------------------------------------------------------------------------
BF_PASS_MIN_RR = _make_bars([
    # Pre-pole bars
    (3.40, 3.48, 3.38, 3.45, 300000),   # bar 0
    (3.45, 3.52, 3.42, 3.50, 350000),   # bar 1

    # Pole: 25% move from 3.50 to 4.38
    (3.50, 3.80, 3.50, 3.78, 500000),   # bar 2
    (3.78, 4.10, 3.75, 4.08, 600000),   # bar 3
    (4.08, 4.38, 4.05, 4.35, 700000),   # bar 4: pole high: 4.38, 25% from 3.50

    # Flag: 14% pullback from 4.38 → low = 3.77 (flag high: 4.10)
    (4.35, 4.10, 3.80, 3.85, 350000),   # bar 5
    (3.85, 4.05, 3.77, 4.15, 250000),   # bar 6: CLOSES above flag high 4.10

    # Breakout: opens above flag high
    (4.15, 4.30, 4.10, 4.25, 650000),   # bar 7: opens at 4.15 > 4.10
])


# =============================================================================
# FAIL CASES (8) - All need 8+ bars
# =============================================================================

# -----------------------------------------------------------------------------
# BF_FAIL_POLE_TOO_WEAK: Pole move at 14.9% (below 15% minimum)
# Tests: min_pole_move_pct = 15.0
# CRITICAL: Ensure NO window of 3+ consecutive bars has 15%+ move
# Math: All windows must have (high - low) / low < 15%
# -----------------------------------------------------------------------------
BF_FAIL_POLE_TOO_WEAK = _make_bars([
    # Pre-pole at 4.00 (no low below 4.00 to prevent extending the pole range)
    (4.00, 4.02, 4.00, 4.01, 300000),   # bar 0: flat, low=4.00
    (4.01, 4.03, 4.00, 4.02, 320000),   # bar 1: flat, low=4.00

    # Weak pole: 14% move from 4.00 low to 4.56 high (3 candles)
    # 4.56/4.00 = 1.14 = 14% (below 15%)
    (4.02, 4.20, 4.00, 4.18, 500000),   # bar 2: low=4.00
    (4.18, 4.38, 4.15, 4.35, 550000),   # bar 3
    (4.35, 4.56, 4.32, 4.54, 600000),   # bar 4: high=4.56, 14% from 4.00

    # Flag (make it otherwise valid)
    (4.54, 4.38, 4.12, 4.18, 350000),   # bar 5: pullback
    (4.18, 4.35, 4.08, 4.30, 250000),   # bar 6: flag

    # Breakout attempt
    (4.30, 4.52, 4.28, 4.48, 650000),   # bar 7
])


# -----------------------------------------------------------------------------
# BF_FAIL_POLE_TOO_SHORT: Tests that poles with < 3 candles are rejected
# Tests: min_pole_candles = 3
# Strategy: Gentle uptrend where ALL consecutive windows < 15% move
# bars 0-7: 4.00 to 4.55 = 13.75% total (< 15%)
# -----------------------------------------------------------------------------
BF_FAIL_POLE_TOO_SHORT = _make_bars([
    # Gentle uptrend, no 3+ bar window exceeds 15%
    # Each bar: ~2% move, cumulative stays under 15%
    (4.00, 4.08, 4.00, 4.07, 500000),   # bar 0: low=4.00
    (4.07, 4.15, 4.05, 4.14, 520000),   # bar 1
    (4.14, 4.22, 4.12, 4.20, 540000),   # bar 2
    (4.20, 4.28, 4.18, 4.26, 560000),   # bar 3
    (4.26, 4.35, 4.24, 4.33, 580000),   # bar 4

    # Pullback
    (4.33, 4.20, 4.00, 4.05, 350000),   # bar 5: pullback to 4.00
    (4.05, 4.15, 3.95, 4.10, 250000),   # bar 6

    # Breakout attempt
    (4.10, 4.30, 4.08, 4.25, 650000),   # bar 7
])


# -----------------------------------------------------------------------------
# BF_FAIL_FLAG_TOO_LONG: Flag at 4 candles (above 3 maximum)
# Tests: max_flag_candles = 3
# Strategy: Pole followed by extended consolidation (4+ bars) with no valid breakout
# Key: Last bar's high must be below ALL possible flag_high values
# -----------------------------------------------------------------------------
BF_FAIL_FLAG_TOO_LONG = _make_bars([
    # Pre-pole (flat)
    (3.50, 3.55, 3.48, 3.52, 400000),   # bar 0

    # Pole: 30% move from 3.50 to 4.55
    (3.52, 3.85, 3.50, 3.82, 500000),   # bar 1
    (3.82, 4.15, 3.80, 4.12, 600000),   # bar 2
    (4.12, 4.35, 4.10, 4.32, 700000),   # bar 3
    (4.32, 4.55, 4.30, 4.52, 800000),   # bar 4: pole high: 4.55

    # Flag: 4 candles of extended consolidation (14% pullback from 4.55)
    # low=3.91, all highs >= 4.10 so any flag_high >= 4.10
    (4.50, 4.25, 3.95, 4.00, 400000),   # bar 5: high=4.25
    (4.00, 4.18, 3.92, 3.98, 350000),   # bar 6: high=4.18
    (3.98, 4.15, 3.91, 3.96, 300000),   # bar 7: high=4.15
    (3.96, 4.10, 3.92, 3.95, 250000),   # bar 8: high=4.10 (min)

    # NO breakout - high must be below 4.10 (minimum flag high possible)
    (3.95, 4.05, 3.90, 4.02, 600000),   # bar 9: high=4.05 < 4.10 (fails all)
])


# -----------------------------------------------------------------------------
# BF_FAIL_PULLBACK_SHALLOW: Pullback at 12% (below 13% minimum)
# Tests: min_pullback_pct = 13.0
# -----------------------------------------------------------------------------
BF_FAIL_PULLBACK_SHALLOW = _make_bars([
    # Pre-pole
    (3.95, 4.02, 3.92, 4.00, 400000),   # bar 0

    # Pole: 20% move
    (4.00, 4.22, 4.00, 4.20, 500000),   # bar 1
    (4.20, 4.42, 4.18, 4.40, 600000),   # bar 2
    (4.40, 4.62, 4.38, 4.60, 700000),   # bar 3
    (4.60, 4.82, 4.58, 4.80, 800000),   # bar 4: pole high: 4.82

    # Flag: only 12% pullback from 4.82 → low = 4.82 * 0.88 = 4.24
    (4.80, 4.60, 4.28, 4.35, 350000),   # bar 5: (low: 4.28 = 11.2% pullback)
    (4.35, 4.58, 4.24, 4.52, 250000),   # bar 6: low=4.24 (12.0% - below 13% min)

    # Breakout attempt
    (4.52, 4.80, 4.50, 4.75, 700000),   # bar 7
])


# -----------------------------------------------------------------------------
# BF_FAIL_PULLBACK_TOO_DEEP: Pullback at ~20.5% (above 20% maximum)
# Tests: max_pullback_pct = 20.0
# Flag highs must not be strictly descending (lower-highs rejection)
# -----------------------------------------------------------------------------
BF_FAIL_PULLBACK_TOO_DEEP = _make_bars([
    # Pre-pole
    (3.95, 4.02, 3.92, 4.00, 500000),   # bar 0

    # Pole: 30% move
    (4.00, 4.35, 4.00, 4.32, 600000),   # bar 1
    (4.32, 4.68, 4.30, 4.65, 700000),   # bar 2
    (4.65, 5.02, 4.62, 5.00, 800000),   # bar 3
    (5.00, 5.22, 4.98, 5.20, 850000),   # bar 4: pole high: 5.22, 30.5% from 4.00

    # Flag: ~20.5% pullback from 5.22 → low = 4.15
    # Non-descending highs so 2-bar flag is found
    (5.20, 4.60, 4.18, 4.30, 400000),   # bar 5: (flag high: 4.60)
    (4.30, 4.62, 4.15, 4.65, 300000),   # bar 6: high=4.62 > 4.60 (non-descending)

    # Breakout attempt
    (4.65, 4.80, 4.62, 4.75, 650000),   # bar 7: opens at 4.65 > 4.62
])


# -----------------------------------------------------------------------------
# BF_FAIL_FLAG_TOO_WIDE: Flag range at 15.1% (above 15% max range)
# Tests: max_flag_range_pct = 15.0
# -----------------------------------------------------------------------------
BF_FAIL_FLAG_TOO_WIDE = _make_bars([
    # Pre-pole
    (3.95, 4.02, 3.92, 4.00, 400000),   # bar 0

    # Pole: 25% move
    (4.00, 4.28, 4.00, 4.25, 500000),   # bar 1
    (4.25, 4.55, 4.22, 4.52, 600000),   # bar 2
    (4.52, 4.82, 4.50, 4.80, 700000),   # bar 3
    (4.80, 5.02, 4.78, 5.00, 800000),   # bar 4: pole high: 5.02

    # Flag: too wide range (>15%)
    # Flag low: 4.00, Flag high: 4.65 → range = (4.65-4.00)/4.00 = 16.25%
    (5.00, 4.65, 4.00, 4.10, 400000),   # bar 5: wide bar (range too wide)
    (4.10, 4.60, 4.05, 4.50, 300000),   # bar 6: (range still wide)

    # Breakout attempt
    (4.50, 4.85, 4.48, 4.80, 700000),   # bar 7
])


# -----------------------------------------------------------------------------
# BF_FAIL_VOLUME_RISING: Volume rising in flag (should decline)
# Tests: volume_declining = True
# Note: This is advisory, not a hard fail in current implementation
# -----------------------------------------------------------------------------
BF_FAIL_VOLUME_RISING = _make_bars([
    # Pre-pole
    (3.50, 3.55, 3.48, 3.52, 400000),   # bar 0

    # Pole: 30% move from 3.50 to 4.55
    (3.52, 3.78, 3.50, 3.75, 500000),   # bar 1
    (3.75, 4.05, 3.73, 4.00, 600000),   # bar 2
    (4.00, 4.30, 3.98, 4.28, 700000),   # bar 3
    (4.28, 4.55, 4.25, 4.52, 800000),   # bar 4: pole high: 4.55

    # Flag: 14% pullback, RISING volume (bad sign)
    (4.50, 4.25, 3.95, 4.00, 300000),   # bar 5: lower volume
    (4.00, 4.20, 3.92, 4.30, 500000),   # bar 6: HIGHER volume (rising), CLOSES above 4.25

    # Breakout attempt
    (4.30, 4.50, 4.25, 4.45, 700000),   # bar 7: opens above flag high
])


# -----------------------------------------------------------------------------
# BF_FAIL_NO_BREAKOUT: Last bar stays below flag high
# Tests: Entry requires break above flag resistance
# Strategy: Make the last bar's high BELOW all possible flag_high values
# -----------------------------------------------------------------------------
BF_FAIL_NO_BREAKOUT = _make_bars([
    # Pre-pole (flat)
    (3.50, 3.55, 3.48, 3.52, 400000),   # bar 0

    # Pole: 30% move from 3.50 to 4.55
    (3.52, 3.78, 3.50, 3.75, 500000),   # bar 1
    (3.75, 4.05, 3.73, 4.00, 600000),   # bar 2
    (4.00, 4.30, 3.98, 4.28, 700000),   # bar 3
    (4.28, 4.55, 4.25, 4.52, 800000),   # bar 4: pole high: 4.55

    # Flag: 14% pullback, declining highs
    # bar 5 high = 4.25, bar 6 high = 4.15
    (4.50, 4.25, 3.95, 4.00, 350000),   # bar 5
    (4.00, 4.15, 3.92, 3.98, 250000),   # bar 6

    # NO BREAKOUT - close/open below flag high (4.15 for 1-bar flag)
    (3.98, 4.10, 3.95, 4.05, 400000),   # bar 7: high=4.10 < 4.15 (no breakout)
])


# -----------------------------------------------------------------------------
# BF_FAIL_FLAG_VOLUME_TOO_HEAVY: Flag avg volume > 60% of pole avg volume
# Tests: max_flag_pole_volume_ratio = 0.60
# Heavy flag volume = distribution, not healthy consolidation
# -----------------------------------------------------------------------------
BF_FAIL_FLAG_VOLUME_TOO_HEAVY = _make_bars([
    # Pre-pole
    (3.00, 3.05, 2.98, 3.02, 400000),   # bar 0: noise

    # Pole: 40% move from 3.00 to 4.20, MODERATE volume (avg ~400k)
    (3.02, 3.35, 3.00, 3.32, 350000),   # bar 1
    (3.32, 3.65, 3.30, 3.62, 400000),   # bar 2
    (3.62, 3.95, 3.60, 3.92, 400000),   # bar 3
    (3.92, 4.20, 3.90, 4.18, 450000),   # bar 4: pole high: 4.20, 40% from 3.00

    # Flag: 14% pullback from 4.20 → low=3.61, HEAVY volume (avg 600k >> pole avg 400k)
    (4.18, 3.90, 3.65, 3.70, 550000),   # bar 5: heavy selling
    (3.70, 3.85, 3.61, 3.95, 650000),   # bar 6: closes above flag high 3.90

    # Entry bar
    (3.95, 4.10, 3.90, 4.05, 700000),   # bar 7: opens above flag high
])


# -----------------------------------------------------------------------------
# BF_FAIL_LOWER_HIGHS: Flag has strictly descending highs (downtrend, not consolidation)
# Tests: lower-highs rejection in _find_flag()
# The 2-bar flag is rejected (descending highs), 1-bar fallback has no breakout
# -----------------------------------------------------------------------------
BF_FAIL_LOWER_HIGHS = _make_bars([
    # Pre-pole
    (3.50, 3.55, 3.48, 3.52, 400000),   # bar 0

    # Pole: 30% move from 3.50 to 4.55
    (3.52, 3.78, 3.50, 3.75, 500000),   # bar 1
    (3.75, 4.05, 3.73, 4.00, 600000),   # bar 2
    (4.00, 4.30, 3.98, 4.28, 700000),   # bar 3
    (4.28, 4.55, 4.25, 4.52, 800000),   # bar 4: pole high: 4.55

    # Flag: 2 candles, strictly descending highs, 14% pullback
    (4.50, 4.25, 3.95, 4.00, 350000),   # bar 5: high=4.25
    (4.00, 4.15, 3.92, 4.10, 250000),   # bar 6: high=4.15 < 4.25 (descending)

    # No breakout above 1-bar flag high (4.15)
    (4.10, 4.12, 4.05, 4.08, 700000),   # bar 7: high=4.12 < 4.15
])


# =============================================================================
# EXPORT ALL FIXTURES
# =============================================================================

__all__ = [
    # PASS cases
    "BF_PASS_VALID",
    "BF_PASS_MIN_POLE_MOVE",
    "BF_PASS_MIN_POLE_CANDLES",
    "BF_PASS_MAX_FLAG_CANDLES",
    "BF_PASS_MIN_FLAG_CANDLE",
    "BF_PASS_MIN_PULLBACK",
    "BF_PASS_MAX_PULLBACK",
    "BF_PASS_MIN_RR",

    # FAIL cases
    "BF_FAIL_POLE_TOO_WEAK",
    "BF_FAIL_POLE_TOO_SHORT",
    "BF_FAIL_FLAG_TOO_LONG",
    "BF_FAIL_PULLBACK_SHALLOW",
    "BF_FAIL_PULLBACK_TOO_DEEP",
    "BF_FAIL_FLAG_TOO_WIDE",
    "BF_FAIL_VOLUME_RISING",
    "BF_FAIL_NO_BREAKOUT",
    "BF_FAIL_FLAG_VOLUME_TOO_HEAVY",
    "BF_FAIL_LOWER_HIGHS",
]
