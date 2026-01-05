"""
Bull Flag Test Fixtures
=======================

Real-world example bars for testing bull flag detection.
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
# VALID BULL FLAG
# =============================================================================
# Pattern:
#   Pole: Strong 25% move over 5 candles ($4.00 -> $5.00)
#   Flag: 15% retracement over 4 candles with declining volume
#   Breakout: New high above flag resistance
#
# Visual:
#           ___
#          /   \___  <- Flag (consolidation, declining volume)
#         /        \___
#        /             \
#       /               → BREAKOUT
#      / <- Pole
#     /

BULL_FLAG_VALID = _make_bars([
    # The Pole - Strong 25% move (5 candles) from 4.00 to 5.00
    (4.00, 4.18, 3.98, 4.15, 500000),   # Bar 1: +3.8%
    (4.15, 4.35, 4.12, 4.32, 600000),   # Bar 2: +4.1%
    (4.32, 4.55, 4.30, 4.52, 700000),   # Bar 3: +4.6%
    (4.52, 4.78, 4.50, 4.75, 800000),   # Bar 4: +5.1%
    (4.75, 5.05, 4.72, 5.00, 900000),   # Bar 5: +5.3% (pole high: 5.05)

    # The Flag - 15% retracement with declining volume (4 candles)
    # Pole moved ~1.05 (4.00 to 5.05), 15% retrace = 0.16, so low ~4.50
    (5.00, 5.02, 4.60, 4.65, 400000),   # Bar 6: pullback starts
    (4.65, 4.70, 4.50, 4.55, 300000),   # Bar 7: more pullback
    (4.55, 4.62, 4.48, 4.52, 250000),   # Bar 8: flag low (4.48)
    (4.52, 4.68, 4.50, 4.65, 200000),   # Bar 9: consolidating (flag high: 4.68)

    # Breakout - New high above flag resistance
    (4.65, 5.00, 4.62, 4.95, 850000),   # Bar 10: BREAKOUT above 4.68
])

# Expected result:
# - detected: True
# - entry_price: ~4.81 (above flag high of 4.80)
# - stop_price: ~4.55 (below flag low of 4.60)
# - pole_move_pct: ~25%
# - pullback_pct: ~15%
# - volume_declining: True


# =============================================================================
# BULL FLAG - NO BREAKOUT YET
# =============================================================================
# Valid pole and flag, but last candle doesn't break resistance

BULL_FLAG_NO_BREAKOUT = _make_bars([
    # The Pole
    (4.00, 4.18, 3.98, 4.15, 500000),
    (4.15, 4.35, 4.12, 4.32, 600000),
    (4.32, 4.55, 4.30, 4.52, 700000),
    (4.52, 4.78, 4.50, 4.75, 800000),
    (4.75, 5.05, 4.72, 5.00, 900000),

    # The Flag (deeper pullback to satisfy 10-25% requirement)
    (5.00, 5.02, 4.60, 4.65, 400000),
    (4.65, 4.70, 4.50, 4.55, 300000),
    (4.55, 4.62, 4.48, 4.52, 250000),
    (4.52, 4.68, 4.50, 4.65, 200000),  # Flag high: 4.68

    # No breakout - stays within flag
    (4.65, 4.67, 4.55, 4.60, 180000),   # Bar 10: no breakout
])

# Expected result:
# - detected: False
# - reason: "No breakout yet: 4.79 <= flag high 4.80"


# =============================================================================
# BULL FLAG - PULLBACK TOO DEEP
# =============================================================================
# Good pole, but flag retraces 30% (max is 25%)

BULL_FLAG_TOO_DEEP = _make_bars([
    # The Pole - 25% move
    (4.00, 4.18, 3.98, 4.15, 500000),
    (4.15, 4.35, 4.12, 4.32, 600000),
    (4.32, 4.55, 4.30, 4.52, 700000),
    (4.52, 4.78, 4.50, 4.75, 800000),
    (4.75, 5.05, 4.72, 5.00, 900000),  # Pole high: 5.05

    # Deep pullback - 30% retracement
    (5.00, 5.02, 4.60, 4.65, 400000),
    (4.65, 4.70, 4.40, 4.45, 350000),
    (4.45, 4.55, 4.30, 4.35, 300000),  # Flag low: 4.30 (30% pullback)
    (4.35, 4.50, 4.32, 4.48, 280000),

    # Breakout attempt
    (4.48, 4.85, 4.45, 4.80, 500000),
])

# Expected result:
# - detected: False
# - reason: "Pullback too deep: 30% > 25%"


# =============================================================================
# BULL FLAG - POLE TOO WEAK
# =============================================================================
# Pole only moves 15% (minimum is 20%)

BULL_FLAG_WEAK_POLE = _make_bars([
    # Weak Pole - only 15% move
    (4.00, 4.10, 3.98, 4.08, 300000),
    (4.08, 4.20, 4.05, 4.18, 350000),
    (4.18, 4.35, 4.15, 4.32, 400000),
    (4.32, 4.50, 4.28, 4.45, 450000),
    (4.45, 4.62, 4.42, 4.60, 480000),  # 15% move only

    # Flag
    (4.60, 4.62, 4.45, 4.50, 250000),
    (4.50, 4.55, 4.40, 4.42, 200000),
    (4.42, 4.48, 4.38, 4.45, 180000),

    # Breakout
    (4.45, 4.70, 4.43, 4.68, 400000),
])

# Expected result:
# - detected: False
# - reason: "No valid pole found before flag"
