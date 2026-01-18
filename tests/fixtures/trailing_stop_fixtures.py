"""
Trailing Stop Test Fixtures
============================

Test data for 2-bar low trailing stop with dynamic buffer.

Rules tested:
- Activation: after +1R profit OR after partial taken
- Buffer: max(spread × 2, ATR(14) × 0.1)
- 2-bar low trailing (for longs)
- Never lower the stop

Each fixture includes entry_idx, entry_price, original_stop for context.
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
# ACTIVATION TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# TRAIL_NOT_ACTIVATED: Trade at +0.5R, should NOT activate trailing
# Entry: $10.00, Stop: $9.50 (risk = $0.50), 1R target = $10.50
# Current high: $10.25 = 0.5R
# -----------------------------------------------------------------------------
TRAIL_NOT_ACTIVATED = _make_bars([
    # Pre-entry bars (for ATR calculation)
    (9.80, 9.90, 9.70, 9.85, 100000),
    (9.85, 9.95, 9.75, 9.90, 110000),
    (9.90, 10.00, 9.80, 9.95, 120000),
    (9.95, 10.05, 9.85, 10.00, 130000),

    # Entry bar (idx 4)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Post-entry: rises to +0.5R only
    (10.05, 10.15, 10.00, 10.10, 140000),
    (10.10, 10.25, 10.05, 10.20, 130000),  # High = 10.25 = 0.5R
    (10.20, 10.22, 10.12, 10.18, 120000),
])
TRAIL_NOT_ACTIVATED_ENTRY_IDX = 4
TRAIL_NOT_ACTIVATED_ENTRY_PRICE = 10.00
TRAIL_NOT_ACTIVATED_ORIGINAL_STOP = 9.50


# -----------------------------------------------------------------------------
# TRAIL_ACTIVATED_1R: Trade at +1R, should activate trailing
# Entry: $10.00, Stop: $9.50 (risk = $0.50), 1R = $10.50
# Current high: $10.55 = 1.1R
# -----------------------------------------------------------------------------
TRAIL_ACTIVATED_1R = _make_bars([
    # Pre-entry bars
    (9.80, 9.90, 9.70, 9.85, 100000),
    (9.85, 9.95, 9.75, 9.90, 110000),
    (9.90, 10.00, 9.80, 9.95, 120000),
    (9.95, 10.05, 9.85, 10.00, 130000),

    # Entry bar (idx 4)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Post-entry: rises to +1.1R
    (10.05, 10.20, 10.00, 10.15, 160000),
    (10.15, 10.35, 10.10, 10.30, 170000),
    (10.30, 10.55, 10.25, 10.50, 180000),  # High = 10.55 = 1.1R
    (10.50, 10.52, 10.40, 10.45, 150000),  # 2-bar low: min(10.25, 10.40) = 10.25
])
TRAIL_ACTIVATED_1R_ENTRY_IDX = 4
TRAIL_ACTIVATED_1R_ENTRY_PRICE = 10.00
TRAIL_ACTIVATED_1R_ORIGINAL_STOP = 9.50


# -----------------------------------------------------------------------------
# TRAIL_ACTIVATED_PARTIAL: Trade at +0.5R but partial taken, should activate
# Entry: $10.00, Stop: $9.50 (risk = $0.50)
# Current high: $10.25 = 0.5R, but partial_taken=True
# -----------------------------------------------------------------------------
TRAIL_ACTIVATED_PARTIAL = _make_bars([
    # Pre-entry bars
    (9.80, 9.90, 9.70, 9.85, 100000),
    (9.85, 9.95, 9.75, 9.90, 110000),
    (9.90, 10.00, 9.80, 9.95, 120000),
    (9.95, 10.05, 9.85, 10.00, 130000),

    # Entry bar (idx 4)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Post-entry: only +0.5R
    (10.05, 10.15, 10.00, 10.10, 140000),
    (10.10, 10.25, 10.05, 10.20, 130000),  # High = 10.25 = 0.5R
    (10.20, 10.22, 10.12, 10.18, 120000),  # 2-bar low: min(10.05, 10.12) = 10.05
])
TRAIL_ACTIVATED_PARTIAL_ENTRY_IDX = 4
TRAIL_ACTIVATED_PARTIAL_ENTRY_PRICE = 10.00
TRAIL_ACTIVATED_PARTIAL_ORIGINAL_STOP = 9.50


# =============================================================================
# TRAILING BEHAVIOR TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# TRAIL_RAISES_STOP: Trailing stop should be higher than original
# Entry: $10.00, Stop: $9.50
# After 1R reached, 2-bar low should raise stop above original
# -----------------------------------------------------------------------------
TRAIL_RAISES_STOP = _make_bars([
    # Pre-entry bars (need 14+ for ATR)
    (9.50, 9.60, 9.45, 9.55, 100000),
    (9.55, 9.65, 9.50, 9.60, 100000),
    (9.60, 9.70, 9.55, 9.65, 100000),
    (9.65, 9.75, 9.60, 9.70, 100000),
    (9.70, 9.80, 9.65, 9.75, 100000),
    (9.75, 9.85, 9.70, 9.80, 100000),
    (9.80, 9.90, 9.75, 9.85, 100000),
    (9.85, 9.95, 9.80, 9.90, 100000),
    (9.90, 10.00, 9.85, 9.95, 100000),
    (9.95, 10.05, 9.90, 10.00, 100000),

    # Entry bar (idx 10)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Post-entry: strong move up to 1.5R
    (10.05, 10.30, 10.02, 10.25, 180000),  # Low: 10.02
    (10.25, 10.55, 10.20, 10.50, 200000),  # Low: 10.20, High: 10.55 = 1.1R
    (10.50, 10.75, 10.45, 10.70, 220000),  # Low: 10.45, High: 10.75 = 1.5R
    (10.70, 10.72, 10.60, 10.65, 180000),  # Low: 10.60
    # 2-bar low = min(10.45, 10.60) = 10.45 >> original stop 9.50
])
TRAIL_RAISES_STOP_ENTRY_IDX = 10
TRAIL_RAISES_STOP_ENTRY_PRICE = 10.00
TRAIL_RAISES_STOP_ORIGINAL_STOP = 9.50


# -----------------------------------------------------------------------------
# TRAIL_NEVER_LOWERS: Even if price drops, stop should not lower
# Entry: $10.00, Stop: $9.50
# After reaching 1.5R, price pulls back but stop stays at highest level
# -----------------------------------------------------------------------------
TRAIL_NEVER_LOWERS = _make_bars([
    # Pre-entry bars (need 14+ for ATR)
    (9.50, 9.60, 9.45, 9.55, 100000),
    (9.55, 9.65, 9.50, 9.60, 100000),
    (9.60, 9.70, 9.55, 9.65, 100000),
    (9.65, 9.75, 9.60, 9.70, 100000),
    (9.70, 9.80, 9.65, 9.75, 100000),
    (9.75, 9.85, 9.70, 9.80, 100000),
    (9.80, 9.90, 9.75, 9.85, 100000),
    (9.85, 9.95, 9.80, 9.90, 100000),
    (9.90, 10.00, 9.85, 9.95, 100000),
    (9.95, 10.05, 9.90, 10.00, 100000),

    # Entry bar (idx 10)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Post-entry: move up to 1.5R
    (10.05, 10.30, 10.02, 10.25, 180000),  # Low: 10.02
    (10.25, 10.55, 10.20, 10.50, 200000),  # Low: 10.20
    (10.50, 10.75, 10.45, 10.70, 220000),  # Low: 10.45, High: 10.75 = 1.5R

    # Now pullback - lows get lower but stop should stay at previous high
    (10.70, 10.72, 10.30, 10.35, 150000),  # Low: 10.30 (pullback)
    (10.35, 10.40, 10.15, 10.20, 140000),  # Low: 10.15 (deeper pullback)
    # 2-bar low = min(10.30, 10.15) = 10.15
    # But stop should NOT go below the previous trailing level
])
TRAIL_NEVER_LOWERS_ENTRY_IDX = 10
TRAIL_NEVER_LOWERS_ENTRY_PRICE = 10.00
TRAIL_NEVER_LOWERS_ORIGINAL_STOP = 9.50


# =============================================================================
# BUFFER CALCULATION TESTS
# =============================================================================

# -----------------------------------------------------------------------------
# TRAIL_WIDE_SPREAD: Wide spread should result in larger buffer
# Spread = $0.10, so buffer = max(0.10 × 2, ATR × 0.1) = max(0.20, ~0.05) = 0.20
# -----------------------------------------------------------------------------
TRAIL_WIDE_SPREAD = _make_bars([
    # Pre-entry with small ATR (tight ranges ~$0.05)
    (10.00, 10.03, 9.98, 10.02, 100000),
    (10.02, 10.05, 10.00, 10.04, 100000),
    (10.04, 10.07, 10.02, 10.05, 100000),
    (10.05, 10.08, 10.03, 10.06, 100000),
    (10.06, 10.09, 10.04, 10.07, 100000),
    (10.07, 10.10, 10.05, 10.08, 100000),
    (10.08, 10.11, 10.06, 10.09, 100000),
    (10.09, 10.12, 10.07, 10.10, 100000),
    (10.10, 10.13, 10.08, 10.11, 100000),
    (10.11, 10.14, 10.09, 10.12, 100000),
    (10.12, 10.15, 10.10, 10.13, 100000),
    (10.13, 10.16, 10.11, 10.14, 100000),
    (10.14, 10.17, 10.12, 10.15, 100000),
    (10.15, 10.18, 10.13, 10.16, 100000),

    # Entry bar (idx 14)
    (10.16, 10.25, 10.14, 10.22, 150000),

    # Post-entry: move to 1R+
    (10.22, 10.45, 10.20, 10.40, 160000),  # Low: 10.20
    (10.40, 10.70, 10.38, 10.65, 170000),  # Low: 10.38, reaches 1R+
    (10.65, 10.72, 10.55, 10.60, 150000),  # Low: 10.55
    # 2-bar low = min(10.38, 10.55) = 10.38
])
TRAIL_WIDE_SPREAD_ENTRY_IDX = 14
TRAIL_WIDE_SPREAD_ENTRY_PRICE = 10.16
TRAIL_WIDE_SPREAD_ORIGINAL_STOP = 9.66  # $0.50 risk


# -----------------------------------------------------------------------------
# TRAIL_HIGH_ATR: High ATR should result in larger buffer
# ATR ~$1.00, so buffer = max(spread × 2, 1.00 × 0.1) = max(0.02, 0.10) = 0.10
# -----------------------------------------------------------------------------
TRAIL_HIGH_ATR = _make_bars([
    # Pre-entry with large ATR (wide ranges ~$1.00)
    (9.00, 9.50, 8.50, 9.30, 100000),
    (9.30, 9.80, 8.80, 9.50, 100000),
    (9.50, 10.00, 9.00, 9.70, 100000),
    (9.70, 10.20, 9.20, 10.00, 100000),
    (10.00, 10.50, 9.50, 10.30, 100000),
    (10.30, 10.80, 9.80, 10.50, 100000),
    (10.50, 11.00, 10.00, 10.70, 100000),
    (10.70, 11.20, 10.20, 11.00, 100000),
    (11.00, 11.50, 10.50, 11.20, 100000),
    (11.20, 11.70, 10.70, 11.50, 100000),
    (11.50, 12.00, 11.00, 11.70, 100000),
    (11.70, 12.20, 11.20, 12.00, 100000),
    (12.00, 12.50, 11.50, 12.20, 100000),
    (12.20, 12.70, 11.70, 12.50, 100000),

    # Entry bar (idx 14)
    (12.50, 13.00, 12.40, 12.80, 150000),

    # Post-entry: move to 1R+
    (12.80, 13.50, 12.70, 13.30, 160000),  # Low: 12.70
    (13.30, 14.00, 13.20, 13.80, 170000),  # Low: 13.20, reaches 1R+
    (13.80, 14.20, 13.60, 14.00, 150000),  # Low: 13.60
    # 2-bar low = min(13.20, 13.60) = 13.20
])
TRAIL_HIGH_ATR_ENTRY_IDX = 14
TRAIL_HIGH_ATR_ENTRY_PRICE = 12.50
TRAIL_HIGH_ATR_ORIGINAL_STOP = 11.50  # $1.00 risk


# =============================================================================
# EDGE CASES
# =============================================================================

# -----------------------------------------------------------------------------
# TRAIL_INSUFFICIENT_BARS: Not enough bars after entry
# Need at least 2 bars after entry for trailing
# -----------------------------------------------------------------------------
TRAIL_INSUFFICIENT_BARS = _make_bars([
    (9.80, 9.90, 9.70, 9.85, 100000),
    (9.85, 9.95, 9.75, 9.90, 110000),
    (9.90, 10.00, 9.80, 9.95, 120000),
    (9.95, 10.05, 9.85, 10.00, 130000),

    # Entry bar (idx 4)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Only 1 bar after entry
    (10.05, 10.60, 10.00, 10.55, 160000),  # Strong move to 1R+ but only 1 bar
])
TRAIL_INSUFFICIENT_BARS_ENTRY_IDX = 4
TRAIL_INSUFFICIENT_BARS_ENTRY_PRICE = 10.00
TRAIL_INSUFFICIENT_BARS_ORIGINAL_STOP = 9.50


# -----------------------------------------------------------------------------
# TRAIL_SHORT_POSITION: Trailing for short position (2-bar high)
# Entry: $10.00 short, Stop: $10.50 (risk = $0.50), 1R = $9.50
# -----------------------------------------------------------------------------
TRAIL_SHORT_POSITION = _make_bars([
    # Pre-entry bars
    (10.50, 10.60, 10.40, 10.55, 100000),
    (10.55, 10.65, 10.45, 10.60, 100000),
    (10.60, 10.70, 10.50, 10.65, 100000),
    (10.65, 10.75, 10.55, 10.70, 100000),
    (10.70, 10.80, 10.60, 10.75, 100000),
    (10.75, 10.85, 10.65, 10.80, 100000),
    (10.80, 10.90, 10.70, 10.85, 100000),
    (10.85, 10.95, 10.75, 10.90, 100000),
    (10.90, 11.00, 10.80, 10.95, 100000),
    (10.95, 11.05, 10.85, 11.00, 100000),

    # Entry bar (idx 10) - shorting at $10.00
    (10.05, 10.10, 9.95, 10.00, 150000),

    # Post-entry: price drops (good for shorts)
    (10.00, 10.05, 9.70, 9.75, 160000),  # High: 10.05
    (9.75, 9.80, 9.45, 9.50, 170000),    # High: 9.80, Low = 9.45 = 1.1R profit
    (9.50, 9.55, 9.40, 9.45, 150000),    # High: 9.55
    # 2-bar high = max(9.80, 9.55) = 9.80
])
TRAIL_SHORT_POSITION_ENTRY_IDX = 10
TRAIL_SHORT_POSITION_ENTRY_PRICE = 10.00
TRAIL_SHORT_POSITION_ORIGINAL_STOP = 10.50
