"""
Micro Pullback Test Fixtures
============================

Comprehensive limit/boundary testing for the NEW Micro Pullback rules:
- Prior move: 5-15% (>15% routes to Bull Flag)
- Pullback: ≤12% retracement
- Duration: ≤2 candles
- Entry: First green candle after pullback (aggressive)
- Minimum 6 bars required

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
# PASS CASES (7)
# =============================================================================

# -----------------------------------------------------------------------------
# MP_PASS_VALID: Standard valid pattern
# Prior move ~10%, pullback ~8%, green entry
# -----------------------------------------------------------------------------
MP_PASS_VALID = _make_bars([
    # Surge: 10% move from 10.00 to 11.00 (3 green candles)
    (10.00, 10.35, 9.98, 10.30, 200000),  # green +3%
    (10.30, 10.65, 10.28, 10.60, 220000),  # green +2.9%
    (10.60, 11.02, 10.58, 11.00, 250000),  # green +3.8% (swing high: 11.02)

    # Pullback: 8% from 11.02 → low ~10.14 (2 candles)
    (11.00, 11.01, 10.50, 10.55, 100000),  # red
    (10.55, 10.58, 10.14, 10.18, 80000),   # red (low: 10.14 = 8% pullback)

    # Entry: green bounce
    (10.18, 10.50, 10.15, 10.45, 200000),  # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_PASS_MIN_PRIOR_MOVE: Prior move at 5.1% (just above 5% minimum)
# Tests: min_prior_move_pct = 5.0
# Need 6+ bars for pattern detection
# Note: Algorithm finds smallest valid window first, so even 2-bar window needs 5%+
# -----------------------------------------------------------------------------
MP_PASS_MIN_PRIOR_MOVE = _make_bars([
    # Surge: Make sure even 2-bar window has 5.1%+ move
    # bars 1-2 need: low to high >= 5%
    # If bar1 low = 10.00 and bar2 high = 10.51, move = 5.1%
    (10.00, 10.05, 10.00, 10.02, 150000),  # flat/noise bar
    (10.02, 10.08, 10.00, 10.05, 160000),  # bar 1: low = 10.00
    (10.05, 10.52, 10.02, 10.50, 180000),  # bar 2: high = 10.52, 5.2% from 10.00 (swing high)

    # Shallow pullback: ~4% (well under 12%)
    (10.50, 10.51, 10.12, 10.15, 80000),   # red
    (10.15, 10.18, 10.10, 10.12, 75000),   # red (low: 10.10)

    # Entry: green bounce
    (10.12, 10.40, 10.08, 10.35, 180000),  # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_PASS_MAX_PRIOR_MOVE: Prior move at 14.9% (just below 15% maximum)
# Tests: max_prior_move_pct = 15.0
# Note: Even 2-bar window must show ~14.9% to test boundary properly
# -----------------------------------------------------------------------------
MP_PASS_MAX_PRIOR_MOVE = _make_bars([
    # Surge: Make 2-bar window have 14.9% move
    # bars 1-2: low = 10.00, high = 11.49 → 14.9%
    (10.00, 10.05, 10.00, 10.02, 200000),  # flat bar
    (10.02, 10.10, 10.00, 10.08, 220000),  # bar 1: low = 10.00
    (10.08, 11.49, 10.05, 11.45, 280000),  # bar 2: high = 11.49, move = 14.9% (swing high)

    # Shallow pullback: 5%
    (11.45, 11.46, 10.92, 10.95, 100000),  # red
    (10.95, 10.98, 10.90, 10.92, 90000),   # red (low: 10.90)

    # Entry: green bounce
    (10.92, 11.30, 10.88, 11.25, 220000),  # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_PASS_MAX_PULLBACK: Pullback at 11.9% (just below 12% maximum)
# Tests: max_pullback_pct = 12.0
# -----------------------------------------------------------------------------
MP_PASS_MAX_PULLBACK = _make_bars([
    # Surge: 10% move for good R:R
    (10.00, 10.35, 10.00, 10.32, 200000),  # green
    (10.32, 10.68, 10.30, 10.65, 220000),  # green
    (10.65, 11.00, 10.62, 10.98, 250000),  # green (high: 11.00)

    # Pullback: ~11.8% from 11.00 → low = 9.70
    (10.98, 10.99, 10.20, 10.25, 100000),  # red
    (10.25, 10.28, 9.70, 9.72, 90000),     # red (low: 9.70 = 11.8% from 11.00)

    # Entry: green bounce
    (9.72, 10.30, 9.68, 10.25, 200000),    # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_PASS_MAX_DURATION: Pullback at exactly 2 candles (maximum)
# Tests: max_pullback_candles = 2
# -----------------------------------------------------------------------------
MP_PASS_MAX_DURATION = _make_bars([
    # Surge: 10% move (3 green candles)
    (10.00, 10.35, 10.00, 10.32, 200000),  # green
    (10.32, 10.68, 10.30, 10.65, 220000),  # green
    (10.65, 11.02, 10.62, 11.00, 250000),  # green (high: 11.02)

    # Pullback: exactly 2 candles (at limit), shallow 5%
    (11.00, 11.01, 10.70, 10.72, 100000),  # red candle 1
    (10.72, 10.75, 10.45, 10.47, 95000),   # red candle 2 (low: 10.45 = 5% from 11.02)

    # Entry: green bounce
    (10.47, 10.80, 10.45, 10.75, 200000),  # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_PASS_MIN_GREEN_RATIO: Green ratio at 60% (3/5 green, just above 50%)
# Tests: >50% green candles requirement
# -----------------------------------------------------------------------------
MP_PASS_MIN_GREEN_RATIO = _make_bars([
    # Surge: 5 candles with 3 green (60% > 50%)
    # Net move: 10.00 to 10.82 = 8.2%
    (10.00, 10.08, 9.98, 9.99, 150000),    # RED (close < open)
    (9.99, 10.28, 9.97, 10.25, 180000),    # green +2.6%
    (10.25, 10.22, 10.18, 10.20, 160000),  # RED
    (10.20, 10.55, 10.18, 10.52, 200000),  # green +3.1%
    (10.52, 10.82, 10.50, 10.80, 220000),  # green +2.7% (high: 10.82)

    # Shallow pullback
    (10.80, 10.81, 10.45, 10.48, 90000),   # red (low: 10.45 = 3.4% from 10.82)

    # Entry: green bounce
    (10.48, 10.75, 10.45, 10.70, 180000),  # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_PASS_MIN_RR: R:R at ~2.1 (just above 2.0 minimum)
# Tests: min_rr_for_setup = 2.0
# -----------------------------------------------------------------------------
MP_PASS_MIN_RR = _make_bars([
    # Surge: 8% move (gives decent target)
    (10.00, 10.30, 10.00, 10.28, 150000),  # green
    (10.28, 10.55, 10.25, 10.52, 160000),  # green
    (10.52, 10.82, 10.50, 10.80, 180000),  # green (high: 10.82, 8.2% from 10.00)

    # Shallow pullback to get right R:R
    (10.80, 10.81, 10.30, 10.32, 80000),   # red
    (10.32, 10.35, 10.22, 10.25, 75000),   # red (low: 10.22)

    # Entry: green bounce
    (10.25, 10.55, 10.20, 10.50, 180000),  # GREEN entry
])


# =============================================================================
# FAIL CASES (7)
# =============================================================================

# -----------------------------------------------------------------------------
# MP_FAIL_BELOW_MIN_PRIOR: Prior move at 4.9% (below 5% minimum)
# Tests: min_prior_move_pct = 5.0
# -----------------------------------------------------------------------------
MP_FAIL_BELOW_MIN_PRIOR = _make_bars([
    # Surge: 4.9% move (below 5% minimum) from 10.00 to 10.49
    (10.00, 10.18, 10.00, 10.15, 150000),  # green
    (10.15, 10.32, 10.12, 10.30, 160000),  # green
    (10.30, 10.49, 10.28, 10.47, 180000),  # green (high: 10.49, 4.9% from 10.00)

    # Shallow pullback
    (10.47, 10.48, 10.20, 10.22, 80000),   # red
    (10.22, 10.25, 10.15, 10.18, 75000),   # red

    # Entry attempt
    (10.18, 10.40, 10.15, 10.35, 180000),  # green
])


# -----------------------------------------------------------------------------
# MP_FAIL_ABOVE_MAX_PRIOR: Prior move at 15.2% (above 15% maximum)
# Tests: max_prior_move_pct = 15.0
# Should route to Bull Flag instead
# Note: Even smallest valid 2-bar window must have >15% move
# -----------------------------------------------------------------------------
MP_FAIL_ABOVE_MAX_PRIOR = _make_bars([
    # Surge: Make even 2-bar window have 15.2% move
    # bars 1-2: low = 10.00, high = 11.52 → 15.2%
    (10.00, 10.05, 10.00, 10.02, 200000),  # flat bar
    (10.02, 10.10, 10.00, 10.08, 220000),  # bar 1: low = 10.00
    (10.08, 11.52, 10.05, 11.50, 280000),  # bar 2: high = 11.52, move = 15.2% (swing high)

    # Shallow pullback
    (11.50, 11.51, 11.00, 11.05, 100000),  # red
    (11.05, 11.08, 10.98, 11.02, 90000),   # red

    # Entry attempt
    (11.02, 11.40, 11.00, 11.35, 220000),  # green
])


# -----------------------------------------------------------------------------
# MP_FAIL_PULLBACK_TOO_DEEP: Pullback at 12.5% (above 12% maximum)
# Tests: max_pullback_pct = 12.0
# -----------------------------------------------------------------------------
MP_FAIL_PULLBACK_TOO_DEEP = _make_bars([
    # Surge: 10% move
    (10.00, 10.35, 10.00, 10.32, 200000),  # green
    (10.32, 10.68, 10.30, 10.65, 220000),  # green
    (10.65, 11.00, 10.62, 10.98, 250000),  # green (high: 11.00)

    # Deep pullback: 12.5% from 11.00 → low = 9.625
    (10.98, 10.99, 10.20, 10.25, 100000),  # red
    (10.25, 10.28, 9.62, 9.65, 90000),     # red (low: 9.62 = 12.5% from 11.00)

    # Entry attempt
    (9.65, 10.10, 9.60, 10.05, 200000),    # green
])


# -----------------------------------------------------------------------------
# MP_FAIL_DURATION_TOO_LONG: Pullback at 3 candles (above 2 maximum)
# Tests: max_pullback_candles = 2
# -----------------------------------------------------------------------------
MP_FAIL_DURATION_TOO_LONG = _make_bars([
    # Surge: 10% move
    (10.00, 10.35, 10.00, 10.32, 200000),  # green
    (10.32, 10.68, 10.30, 10.65, 220000),  # green
    (10.65, 11.02, 10.62, 11.00, 250000),  # green (high: 11.02)

    # Pullback: 3 candles (above 2 limit)
    (11.00, 11.01, 10.75, 10.78, 100000),  # red candle 1
    (10.78, 10.80, 10.68, 10.70, 95000),   # red candle 2
    (10.70, 10.72, 10.55, 10.58, 90000),   # red candle 3 (exceeds limit)

    # Entry attempt
    (10.58, 10.85, 10.55, 10.80, 200000),  # green
])


# -----------------------------------------------------------------------------
# MP_FAIL_GREEN_RATIO_LOW: Green ratio at 40% (2/5 green, below 50%)
# Tests: >50% green candles requirement in surge
# This fixture ensures NO valid surge window exists
# -----------------------------------------------------------------------------
MP_FAIL_GREEN_RATIO_LOW = _make_bars([
    # Series of bars where no 2-10 bar window has both 5%+ move AND >50% green
    (10.00, 10.05, 9.95, 9.98, 150000),    # RED
    (9.98, 10.00, 9.88, 9.90, 140000),     # RED
    (9.90, 10.10, 9.88, 10.05, 180000),    # green (small)
    (10.05, 10.08, 9.95, 9.98, 160000),    # RED
    (9.98, 10.02, 9.90, 9.92, 140000),     # RED
    (9.92, 10.12, 9.90, 10.08, 200000),    # green (small)

    # Even looking at all 6 bars: 2 green / 6 = 33% < 50%
    # And net move is only ~0.8%
    (10.08, 10.15, 10.02, 10.10, 180000),  # GREEN entry
])


# -----------------------------------------------------------------------------
# MP_FAIL_LAST_BAR_RED: Last bar is red (no entry signal)
# Tests: Entry candle must be green
# -----------------------------------------------------------------------------
MP_FAIL_LAST_BAR_RED = _make_bars([
    # Surge: 10% move
    (10.00, 10.35, 10.00, 10.32, 200000),  # green
    (10.32, 10.68, 10.30, 10.65, 220000),  # green
    (10.65, 11.02, 10.62, 11.00, 250000),  # green (high: 11.02)

    # Pullback
    (11.00, 11.01, 10.60, 10.62, 100000),  # red
    (10.62, 10.65, 10.50, 10.52, 90000),   # red (low: 10.50)

    # NO entry - last bar is RED
    (10.52, 10.55, 10.40, 10.42, 80000),   # RED (no entry)
])


# -----------------------------------------------------------------------------
# MP_FAIL_RR_TOO_LOW: R:R below 2.0 minimum
# Tests: min_rr_for_setup = 2.0
# Small prior move with wide stop = bad R:R
# -----------------------------------------------------------------------------
MP_FAIL_RR_TOO_LOW = _make_bars([
    # Surge: 5.1% move (minimum)
    (10.00, 10.18, 10.00, 10.15, 150000),  # green
    (10.15, 10.35, 10.12, 10.32, 160000),  # green
    (10.32, 10.51, 10.30, 10.50, 180000),  # green (high: 10.51, 5.1% from 10.00)

    # Deep pullback (still within 12%) creates wide stop
    # 11% pullback from 10.51 = low of 9.35
    (10.50, 10.51, 9.50, 9.52, 80000),     # red (big drop)
    (9.52, 9.55, 9.35, 9.38, 75000),       # red (low: 9.35 = 11% pullback)

    # Entry: entry at 9.39, stop at 9.35 - 0.09 = 9.26
    # Risk = 9.39 - 9.26 = 0.13
    # Reward = 5.1% * 9.39 = 0.48
    # R:R = 0.48 / 0.13 = 3.7 - still too high!
    #
    # Need to make the risk larger. With stop buffer = 1% of price:
    # stop = 9.35 - 0.09 = 9.26
    # entry slightly above = 9.39
    # But the algo uses stop_buffer = max(1% of price, 3 cents)
    # For $9.35, 1% = 9.35 cents, so buffer = 9.35 cents
    # stop = 9.35 - 0.09 = 9.26
    # If entry is at open+1cent = 9.39
    # risk = 9.39 - 9.26 = 0.13 = 13 cents
    # reward = 5.1% * 9.39 = 0.48
    # R:R = 0.48/0.13 = 3.7
    #
    # To get R:R < 2, need risk > reward/2 = 0.24
    # So we need entry - stop > 0.24
    # If stop = pullback_low - 1%, and entry = open + 1 cent
    # Then we need the entry bar to open much higher than pullback low
    # Let's have entry bar open at 9.50, so entry = 9.51
    # stop = 9.35 - 0.09 = 9.26
    # risk = 9.51 - 9.26 = 0.25
    # reward = 5.1% * 9.51 = 0.49
    # R:R = 0.49/0.25 = 1.96 < 2.0 - FAIL!
    (9.50, 9.70, 9.48, 9.65, 180000),      # GREEN entry (opens at 9.50)
])


# =============================================================================
# EXPORT ALL FIXTURES
# =============================================================================

__all__ = [
    # PASS cases
    "MP_PASS_VALID",
    "MP_PASS_MIN_PRIOR_MOVE",
    "MP_PASS_MAX_PRIOR_MOVE",
    "MP_PASS_MAX_PULLBACK",
    "MP_PASS_MAX_DURATION",
    "MP_PASS_MIN_GREEN_RATIO",
    "MP_PASS_MIN_RR",

    # FAIL cases
    "MP_FAIL_BELOW_MIN_PRIOR",
    "MP_FAIL_ABOVE_MAX_PRIOR",
    "MP_FAIL_PULLBACK_TOO_DEEP",
    "MP_FAIL_DURATION_TOO_LONG",
    "MP_FAIL_GREEN_RATIO_LOW",
    "MP_FAIL_LAST_BAR_RED",
    "MP_FAIL_RR_TOO_LOW",
]
