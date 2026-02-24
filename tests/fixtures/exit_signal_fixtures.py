"""
Exit Signal Test Fixtures
=========================

Test data for exit signal detection (topping tail, jackknife, etc.)
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
# TOPPING TAIL - VALID
# =============================================================================
# Pattern: Long upper wick (12x body), body in lower 7% of range, in profit.
# Entry at $10.00, bar shows rejection at highs.
#
# Calculations:
# - body_top = 10.20, body_bottom = 10.15, body_size = 0.05
# - upper_wick = 10.80 - 10.20 = 0.60
# - upper_wick_ratio = 0.60 / 0.05 = 12x (>= 2x) ✓
# - candle_range = 10.80 - 10.10 = 0.70
# - body_position = (10.15 - 10.10) / 0.70 = 0.07 (<= 0.33) ✓
# - close (10.20) > entry (10.00) ✓

TOPPING_TAIL_VALID = {
    "entry_price": 10.00,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.05, 9.98, 10.02, 100000),
        # Post-entry bar with topping tail
        (10.15, 10.80, 10.10, 10.20, 150000),
    ]),
}

# Expected result:
# - ExitSignal with signal_type="topping_tail", triggered=True


# =============================================================================
# TOPPING TAIL - REJECTED: Upper wick too small
# =============================================================================
# Pattern: Upper wick is only 1.5x body (needs >= 2x to trigger).
# Should NOT trigger topping tail exit.
#
# Calculations:
# - body_top = 10.30, body_bottom = 10.20, body_size = 0.10
# - upper_wick = 10.45 - 10.30 = 0.15
# - upper_wick_ratio = 0.15 / 0.10 = 1.5x (< 2x) ✗

TOPPING_TAIL_WICK_TOO_SMALL = {
    "entry_price": 10.00,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.05, 9.98, 10.02, 100000),
        # Bar with short upper wick (1.5x body)
        (10.20, 10.45, 10.18, 10.30, 150000),
    ]),
}

# Expected result:
# - No topping_tail signal (upper wick ratio < 2x)


# =============================================================================
# TOPPING TAIL - REJECTED: Body not in lower third
# =============================================================================
# Pattern: Body is in middle of candle, not lower third.
# Should NOT trigger topping tail exit.
#
# Calculations:
# - body_top = 10.50, body_bottom = 10.40, body_size = 0.10
# - upper_wick = 10.70 - 10.50 = 0.20
# - upper_wick_ratio = 0.20 / 0.10 = 2x ✓
# - candle_range = 10.70 - 10.20 = 0.50
# - body_position = (10.40 - 10.20) / 0.50 = 0.40 (> 0.33) ✗

TOPPING_TAIL_BODY_NOT_LOW = {
    "entry_price": 10.00,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.05, 9.98, 10.02, 100000),
        # Bar with body in middle (position = 0.40)
        (10.40, 10.70, 10.20, 10.50, 150000),
    ]),
}

# Expected result:
# - No topping_tail signal (body_position > 0.33)


# =============================================================================
# TOPPING TAIL - REJECTED: Not in profit
# =============================================================================
# Pattern: Perfect topping tail shape, but close is below entry price.
# Should NOT trigger (we only want to exit when in profit).
#
# Calculations:
# - body_top = 10.02, body_bottom = 9.97, body_size = 0.05
# - upper_wick = 10.50 - 10.02 = 0.48
# - upper_wick_ratio = 0.48 / 0.05 = 9.6x ✓
# - candle_range = 10.50 - 9.95 = 0.55
# - body_position = (9.97 - 9.95) / 0.55 = 0.04 ✓
# - close (10.02) > entry (10.05)? NO ✗

TOPPING_TAIL_NOT_IN_PROFIT = {
    "entry_price": 10.05,
    "bars": _make_bars([
        # Entry bar
        (10.05, 10.08, 10.02, 10.06, 100000),
        # Topping tail pattern but close below entry
        (9.97, 10.50, 9.95, 10.02, 150000),
    ]),
}

# Expected result:
# - No topping_tail signal (close <= entry_price)


# =============================================================================
# TOPPING TAIL - LIMIT: Just above 2x wick ratio (minimum)
# =============================================================================
# Tests upper_wick_ratio >= 2.0 near the limit.
# Uses 2.05x ratio to avoid floating point precision issues.
#
# Calculations:
# - body_top = 10.30, body_bottom = 10.20, body_size = 0.10
# - upper_wick = 10.505 - 10.30 = 0.205
# - upper_wick_ratio = 0.205 / 0.10 = 2.05x (just above limit) ✓
# - candle_range = 10.505 - 10.18 = 0.325
# - body_position = (10.20 - 10.18) / 0.325 = 0.0615 ✓
# - close (10.30) > entry (10.00) ✓

TOPPING_TAIL_LIMIT_WICK_RATIO = {
    "entry_price": 10.00,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.05, 9.98, 10.02, 100000),
        # Bar with ~2.05x upper wick ratio (just above 2.0 minimum)
        (10.20, 10.505, 10.18, 10.30, 150000),
    ]),
}

# Expected result:
# - ExitSignal with signal_type="topping_tail", triggered=True


# =============================================================================
# TOPPING TAIL - LIMIT: Body at ~0.32 position (near maximum)
# =============================================================================
# Tests body_position <= 0.33 near the limit.
# Uses 0.32 to avoid floating point precision issues.
#
# Design: range = 1.00, so (body_bottom - low) = 0.32
# low = 10.00, body_bottom = 10.32, body_top = 10.42, high = 11.00
#
# Calculations:
# - body_size = 10.42 - 10.32 = 0.10
# - upper_wick = 11.00 - 10.42 = 0.58
# - upper_wick_ratio = 0.58 / 0.10 = 5.8x ✓
# - candle_range = 11.00 - 10.00 = 1.00
# - body_position = (10.32 - 10.00) / 1.00 = 0.32 (just below 0.33 limit) ✓
# - close (10.42) > entry (10.00) ✓

TOPPING_TAIL_LIMIT_BODY_POSITION = {
    "entry_price": 10.00,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.05, 9.98, 10.02, 100000),
        # Bar with body_position at ~0.32 (just below 0.33 limit)
        (10.32, 11.00, 10.00, 10.42, 150000),
    ]),
}

# Expected result:
# - ExitSignal with signal_type="topping_tail", triggered=True


# =============================================================================
# STOP HIT - VALID: Low hits stop exactly
# =============================================================================
# Condition: low <= stop_price
# Stop at $9.50, low hits exactly $9.50

STOP_HIT_VALID_EXACT = {
    "stop_price": 9.50,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bar - low hits stop exactly
        (10.05, 10.15, 9.50, 9.60, 120000),
    ]),
}


# =============================================================================
# STOP HIT - VALID: Low goes below stop
# =============================================================================
# Stop at $9.50, low goes to $9.40 (below stop)

STOP_HIT_VALID_BELOW = {
    "stop_price": 9.50,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bar - low goes below stop
        (10.05, 10.08, 9.40, 9.45, 120000),
    ]),
}


# =============================================================================
# STOP HIT - REJECTED: Low stays above stop
# =============================================================================
# Stop at $9.50, low only goes to $9.60 (above stop)

STOP_HIT_ABOVE_STOP = {
    "stop_price": 9.50,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bar - low stays above stop
        (10.05, 10.20, 9.60, 10.15, 120000),
    ]),
}


# =============================================================================
# STOP HIT - LIMIT: Low barely misses stop
# =============================================================================
# Stop at $9.50, low at $9.51 (just above stop - should NOT trigger)

STOP_HIT_LIMIT_MISS = {
    "stop_price": 9.50,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bar - low barely misses stop
        (10.05, 10.15, 9.51, 10.10, 120000),
    ]),
}


# =============================================================================
# STOP HIT - MULTI-BAR: Stop hit on second bar
# =============================================================================
# Stop at $9.50, first bar stays above, second bar hits stop

STOP_HIT_SECOND_BAR = {
    "stop_price": 9.50,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar - stays above stop
        (10.05, 10.20, 9.70, 10.15, 120000),
        # Second post-entry bar - hits stop
        (10.15, 10.18, 9.50, 9.55, 110000),
    ]),
}


# =============================================================================
# VOLUME DECLINE - VALID: Volume drops to 40%
# =============================================================================
# Entry volume 100k, last 3 bars avg 40k (40% < 50%)

VOLUME_DECLINE_VALID = {
    "entry_idx": 0,
    "bars": _make_bars([
        # Entry bar - 100k volume
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bars with declining volume AND stalling price
        # recent_bars[0].open = 10.05, recent_bars[-1].close = 10.00
        (10.05, 10.08, 9.98, 10.02, 45000),
        (10.02, 10.05, 9.95, 9.98, 38000),
        (9.98, 10.02, 9.92, 10.00, 37000),
    ]),
}


# =============================================================================
# VOLUME DECLINE - REJECTED: Not enough bars
# =============================================================================
# Only 2 bars after entry (needs 3)

VOLUME_DECLINE_NOT_ENOUGH_BARS = {
    "entry_idx": 0,
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Only 2 post-entry bars
        (10.05, 10.15, 10.00, 10.10, 30000),
        (10.10, 10.18, 10.05, 10.12, 30000),
    ]),
}


# =============================================================================
# VOLUME DECLINE - REJECTED: Volume stays high (60%)
# =============================================================================
# Entry volume 100k, last 3 bars avg 60k (60% > 50%)

VOLUME_DECLINE_STAYS_HIGH = {
    "entry_idx": 0,
    "bars": _make_bars([
        # Entry bar - 100k volume
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bars with moderate volume (avg = 60k)
        (10.05, 10.15, 10.00, 10.10, 65000),
        (10.10, 10.18, 10.05, 10.12, 58000),
        (10.12, 10.20, 10.08, 10.15, 57000),
    ]),
}


# =============================================================================
# VOLUME DECLINE - LIMIT: Volume at exactly 50% (should NOT trigger)
# =============================================================================
# Entry volume 100k, last 3 bars avg 50k (50% is NOT < 50%)

VOLUME_DECLINE_LIMIT_AT_50 = {
    "entry_idx": 0,
    "bars": _make_bars([
        # Entry bar - 100k volume
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bars with exactly 50% avg volume
        (10.05, 10.15, 10.00, 10.10, 50000),
        (10.10, 10.18, 10.05, 10.12, 50000),
        (10.12, 10.20, 10.08, 10.15, 50000),
    ]),
}


# =============================================================================
# VOLUME DECLINE - LIMIT: Volume at 49% (should trigger)
# =============================================================================
# Entry volume 100k, last 3 bars avg 49k (49% < 50%)

VOLUME_DECLINE_LIMIT_AT_49 = {
    "entry_idx": 0,
    "bars": _make_bars([
        # Entry bar - 100k volume
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bars with 49% avg volume AND stalling price
        # recent_bars[0].open = 10.05, recent_bars[-1].close = 10.01
        (10.05, 10.08, 9.98, 10.02, 49000),
        (10.02, 10.05, 9.95, 9.98, 49000),
        (9.98, 10.03, 9.95, 10.01, 49000),
    ]),
}


# =============================================================================
# VOLUME DECLINE - REJECTED: Low volume but price still rising
# =============================================================================
# Entry volume 100k, last 3 bars avg 30k but price making new highs.
# Should NOT trigger — stock is still running.

VOLUME_DECLINE_PRICE_RISING = {
    "entry_idx": 0,
    "bars": _make_bars([
        # Entry bar - 100k volume
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Post-entry bars with low volume BUT price rising
        # recent_bars[0].open = 10.10, recent_bars[-1].close = 10.40
        (10.10, 10.20, 10.05, 10.18, 30000),
        (10.18, 10.30, 10.15, 10.28, 28000),
        (10.28, 10.45, 10.25, 10.40, 32000),
    ]),
}


# =============================================================================
# JACKKNIFE - VALID: All conditions met
# =============================================================================
# Conditions:
# 1. curr["high"] > prev["high"] (new high)
# 2. curr["close"] < prev["low"] (closes below prior low)
# 3. curr["close"] < curr["open"] (red candle)

JACKKNIFE_VALID = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar (will be "prev")
        (10.05, 10.20, 10.00, 10.15, 120000),
        # Jackknife bar: new high (10.25 > 10.20), closes below prev low (9.95 < 10.00), red
        (10.15, 10.25, 9.90, 9.95, 150000),
    ]),
}


# =============================================================================
# JACKKNIFE - REJECTED: Not enough bars
# =============================================================================
# Only 1 bar in post_entry (needs 2)

JACKKNIFE_NOT_ENOUGH_BARS = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # Only 1 post-entry bar
        (10.05, 10.20, 9.90, 9.95, 120000),
    ]),
}


# =============================================================================
# JACKKNIFE - REJECTED: No new high
# =============================================================================
# High doesn't exceed prior high

JACKKNIFE_NO_NEW_HIGH = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar
        (10.05, 10.20, 10.00, 10.15, 120000),
        # No new high (10.18 < 10.20), but closes below and is red
        (10.15, 10.18, 9.90, 9.95, 150000),
    ]),
}


# =============================================================================
# JACKKNIFE - REJECTED: Doesn't close below prior low
# =============================================================================
# Makes new high but closes above prior low

JACKKNIFE_ABOVE_PRIOR_LOW = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar (low = 10.00)
        (10.05, 10.20, 10.00, 10.15, 120000),
        # New high but closes at 10.02 (above prior low 10.00)
        (10.15, 10.25, 10.01, 10.02, 150000),
    ]),
}


# =============================================================================
# JACKKNIFE - REJECTED: Green candle
# =============================================================================
# New high, closes below prior low, but green (close > open)

JACKKNIFE_GREEN_CANDLE = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar (low = 10.00)
        (10.05, 10.20, 10.00, 10.15, 120000),
        # New high, closes below prior low, but GREEN (9.98 > 9.90)
        (9.90, 10.25, 9.85, 9.98, 150000),
    ]),
}


# =============================================================================
# JACKKNIFE - LIMIT: High exactly equals prior high (should NOT trigger)
# =============================================================================
# Condition is curr["high"] > prev["high"], not >=

JACKKNIFE_LIMIT_EQUAL_HIGH = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar (high = 10.20)
        (10.05, 10.20, 10.00, 10.15, 120000),
        # High exactly equals (10.20 == 10.20), closes below, red
        (10.15, 10.20, 9.90, 9.95, 150000),
    ]),
}


# =============================================================================
# JACKKNIFE - LIMIT: Close exactly at prior low (should NOT trigger)
# =============================================================================
# Condition is curr["close"] < prev["low"], not <=

JACKKNIFE_LIMIT_EQUAL_LOW = {
    "bars": _make_bars([
        # Entry bar
        (10.00, 10.10, 9.95, 10.05, 100000),
        # First post-entry bar (low = 10.00)
        (10.05, 10.20, 10.00, 10.15, 120000),
        # New high, close exactly at prior low (10.00 == 10.00), red
        (10.15, 10.25, 9.95, 10.00, 150000),
    ]),
}


# =============================================================================
# MACD CROSS - Helper to generate bars with specific price pattern
# =============================================================================
def _make_macd_bars(base_prices: list, volumes: list = None) -> pd.DataFrame:
    """
    Create bars that will produce specific MACD behavior.

    Args:
        base_prices: List of close prices (needs ~40+ for stable MACD)
        volumes: Optional list of volumes (defaults to 100000)
    """
    if volumes is None:
        volumes = [100000] * len(base_prices)

    data = []
    for i, close in enumerate(base_prices):
        # Create OHLC with small range around close
        open_p = close * 0.998
        high = close * 1.002
        low = close * 0.996
        vol = volumes[i] if i < len(volumes) else 100000
        data.append((open_p, high, low, close, vol))

    return _make_bars(data)


# =============================================================================
# MACD CROSS - VALID: Bearish crossover after entry
# =============================================================================
# Price rises then falls, causing MACD to cross below signal
# Need ~40 bars for stable MACD, entry near end, then decline

MACD_CROSS_VALID = {
    "entry_idx": 35,
    "bars": _make_macd_bars(
        # 35 bars of gradual rise (builds positive MACD)
        [10.0 + i * 0.05 for i in range(35)] +
        # Entry point at bar 35 (price = 11.70)
        [11.70] +
        # 5 bars of sharp decline (causes bearish cross)
        [11.50, 11.20, 10.90, 10.60, 10.30]
    ),
}


# =============================================================================
# MACD CROSS - REJECTED: Not enough bars for MACD
# =============================================================================
# Only 20 bars (need ~35 for stable MACD calculation)

MACD_CROSS_NOT_ENOUGH_BARS = {
    "entry_idx": 15,
    "bars": _make_macd_bars([10.0 + i * 0.02 for i in range(20)]),
}


# =============================================================================
# MACD CROSS - REJECTED: Not enough bars after entry
# =============================================================================
# 40 bars total but entry at bar 39 (only 1 bar after entry, need 2)

MACD_CROSS_NOT_ENOUGH_AFTER_ENTRY = {
    "entry_idx": 39,
    "bars": _make_macd_bars([10.0 + i * 0.03 for i in range(41)]),
}


# =============================================================================
# MACD CROSS - REJECTED: MACD stays above signal
# =============================================================================
# Continued uptrend - MACD stays bullish

MACD_CROSS_STAYS_BULLISH = {
    "entry_idx": 35,
    "bars": _make_macd_bars(
        # Continued uptrend - MACD stays above signal
        [10.0 + i * 0.05 for i in range(45)]
    ),
}


# =============================================================================
# MACD CROSS - REJECTED: Bullish cross (wrong direction)
# =============================================================================
# Price falls then rises - MACD crosses ABOVE signal (bullish, not bearish)

MACD_CROSS_BULLISH_CROSS = {
    "entry_idx": 35,
    "bars": _make_macd_bars(
        # 35 bars of gradual decline (builds negative MACD)
        [12.0 - i * 0.05 for i in range(35)] +
        # Entry at bar 35 (price = 10.30)
        [10.30] +
        # 5 bars of sharp rise (causes bullish cross - wrong direction)
        [10.50, 10.80, 11.10, 11.40, 11.70]
    ),
}


# =============================================================================
# MACD CROSS - LIMIT: MACD equals signal then crosses below
# =============================================================================
# MACD touches signal line then goes below (should trigger)

MACD_CROSS_LIMIT_EQUALS_THEN_BELOW = {
    "entry_idx": 35,
    "bars": _make_macd_bars(
        # Build up positive MACD
        [10.0 + i * 0.04 for i in range(35)] +
        # Plateau (MACD approaches signal)
        [11.35, 11.35, 11.35] +
        # Slight decline (MACD crosses below)
        [11.30, 11.20, 11.10]
    ),
}


# =============================================================================
# VWAP CROSS - Helper to create bars with VWAP
# =============================================================================
def _make_vwap_bars(closes: list, vwap_values: list, volumes: list = None) -> tuple:
    """
    Create bars DataFrame and VWAP series for testing.

    Args:
        closes: List of close prices
        vwap_values: List of VWAP values (same length as closes)
        volumes: Optional list of volumes (defaults to 100000)

    Returns:
        Tuple of (bars DataFrame, vwap Series)
    """
    if volumes is None:
        volumes = [100000] * len(closes)

    data = []
    for i, close in enumerate(closes):
        open_p = close * 0.998
        high = close * 1.002
        low = close * 0.996
        vol = volumes[i] if i < len(volumes) else 100000
        data.append((open_p, high, low, close, vol))

    bars = _make_bars(data)
    vwap = pd.Series(vwap_values)
    return bars, vwap


# =============================================================================
# VWAP CROSS - VALID: Price crosses below VWAP (long exit)
# =============================================================================
# For longs: price closing below VWAP indicates losing institutional support.
# Entry at bar 0 with price above VWAP, then price drops below VWAP.
#
# Bar 0: close=10.20, vwap=10.00 (above VWAP - ok)
# Bar 1: close=10.15, vwap=10.05 (above VWAP - ok)
# Bar 2: close=9.90, vwap=10.00 (below VWAP - trigger!)

VWAP_CROSS_VALID = {
    "entry_idx": 0,
    "bars": _make_vwap_bars(
        closes=[10.20, 10.15, 9.90],
        vwap_values=[10.00, 10.05, 10.00],
    )[0],
    "vwap": _make_vwap_bars(
        closes=[10.20, 10.15, 9.90],
        vwap_values=[10.00, 10.05, 10.00],
    )[1],
}

# Expected: ExitSignal with signal_type="vwap_cross", triggered=True


# =============================================================================
# VWAP CROSS - REJECTED: Not enough bars after entry
# =============================================================================
# Entry at bar 0, only 1 bar after entry (need at least confirmation_bars + 1)
# With default confirmation_bars=1, we need at least 2 bars after entry.

VWAP_CROSS_NOT_ENOUGH_AFTER_ENTRY = {
    "entry_idx": 0,
    "bars": _make_vwap_bars(
        closes=[10.20, 9.90],  # Only 1 bar after entry
        vwap_values=[10.00, 10.00],
    )[0],
    "vwap": _make_vwap_bars(
        closes=[10.20, 9.90],
        vwap_values=[10.00, 10.00],
    )[1],
}

# Expected: No vwap_cross signal (not enough bars)


# =============================================================================
# VWAP CROSS - REJECTED: Price stays above VWAP (for longs)
# =============================================================================
# Price never crosses below VWAP - no exit signal.

VWAP_CROSS_STAYS_ABOVE = {
    "entry_idx": 0,
    "bars": _make_vwap_bars(
        closes=[10.20, 10.25, 10.30, 10.35],
        vwap_values=[10.00, 10.05, 10.10, 10.15],
    )[0],
    "vwap": _make_vwap_bars(
        closes=[10.20, 10.25, 10.30, 10.35],
        vwap_values=[10.00, 10.05, 10.10, 10.15],
    )[1],
}

# Expected: No vwap_cross signal (price stays above VWAP)


# =============================================================================
# VWAP CROSS - LIMIT: Price equals VWAP then goes below
# =============================================================================
# Tests boundary condition: close == vwap is NOT adverse (need close < vwap).
# Bar 1: close equals VWAP (not adverse)
# Bar 2: close below VWAP (adverse - should trigger)

VWAP_CROSS_LIMIT_EQUALS_THEN_BELOW = {
    "entry_idx": 0,
    "bars": _make_vwap_bars(
        closes=[10.20, 10.00, 9.95],
        vwap_values=[10.00, 10.00, 10.00],
    )[0],
    "vwap": _make_vwap_bars(
        closes=[10.20, 10.00, 9.95],
        vwap_values=[10.00, 10.00, 10.00],
    )[1],
}

# Expected: ExitSignal with signal_type="vwap_cross" on bar 2


# =============================================================================
# VWAP CROSS - SHORT: Price crosses above VWAP (short exit)
# =============================================================================
# For shorts: price closing above VWAP indicates losing institutional pressure.
# Entry at bar 0 with price below VWAP, then price rises above VWAP.

VWAP_CROSS_SHORT_VALID = {
    "entry_idx": 0,
    "direction": "short",
    "bars": _make_vwap_bars(
        closes=[9.80, 9.85, 10.10],
        vwap_values=[10.00, 9.95, 10.00],
    )[0],
    "vwap": _make_vwap_bars(
        closes=[9.80, 9.85, 10.10],
        vwap_values=[10.00, 9.95, 10.00],
    )[1],
}

# Expected: ExitSignal with signal_type="vwap_cross", triggered=True
