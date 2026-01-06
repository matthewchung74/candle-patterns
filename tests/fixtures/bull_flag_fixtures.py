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


# =============================================================================
# LIMIT TESTS - Testing rules at their boundaries
# =============================================================================

# =============================================================================
# BULL FLAG - LIMIT: Exactly 20% Pole Move (minimum)
# =============================================================================
# Tests min_pole_move_pct: 20.0 at the exact limit.
# Should PASS with 20.5% pole move.
# Note: Flag must be BELOW pole high with tight range (<15%).
# Note: Pole search loop goes up to pole_end_idx+1, so we need the move
# achievable within that range. Bar 1's low set to 4.00 for 20.5% move in 4 bars.

BULL_FLAG_LIMIT_POLE_MOVE = _make_bars([
    # Pole: 20.5% move from low 4.00 to high 4.82 in bars 1-4
    # Bar 0 is buffer, bar 1 has the low that defines pole start
    (4.00, 4.12, 4.00, 4.10, 400000),   # Bar 0: buffer (not in pole range)
    (4.10, 4.28, 4.00, 4.25, 450000),   # Bar 1: low 4.00 defines pole low
    (4.25, 4.45, 4.22, 4.42, 500000),   # Bar 2
    (4.42, 4.62, 4.40, 4.60, 550000),   # Bar 3
    (4.60, 4.82, 4.58, 4.80, 600000),   # Bar 4: pole high 4.82 (20.5% from 4.00)

    # Flag: Tight consolidation BELOW pole high, 12% pullback
    # Flag range: 4.24 to 4.50 = 6% (tight)
    (4.80, 4.50, 4.35, 4.40, 300000),   # Bar 5: gap down into flag
    (4.40, 4.48, 4.25, 4.35, 250000),   # Bar 6: pullback continues
    (4.35, 4.50, 4.24, 4.45, 200000),   # Bar 7: flag low (4.24), flag high (4.50)

    # Breakout: Must break above flag high (4.50)
    (4.45, 4.70, 4.42, 4.65, 500000),   # Bar 8: BREAKOUT above 4.50
])

# Expected result:
# - detected: True
# - pole_move_pct: ~20.5% (at limit, from 4.00 to 4.82)


# =============================================================================
# BULL FLAG - LIMIT: Just Over 10% Pullback (minimum)
# =============================================================================
# Tests min_pullback_pct: 10.0 - uses strict `<` so need >10%.
# Should PASS with 10.5% pullback (just over limit).
# Note: Flag must be BELOW pole high with tight range.

BULL_FLAG_LIMIT_PULLBACK_SHALLOW = _make_bars([
    # Strong pole: 30% move for good R:R
    (4.00, 4.18, 4.00, 4.15, 500000),   # Bar 1
    (4.15, 4.42, 4.12, 4.40, 600000),   # Bar 2
    (4.40, 4.70, 4.38, 4.68, 700000),   # Bar 3
    (4.68, 5.00, 4.65, 4.98, 800000),   # Bar 4
    (4.98, 5.22, 4.95, 5.20, 850000),   # Bar 5 (pole high: 5.22, 30.5% from 4.00)

    # Flag: 10.5% pullback, tight consolidation BELOW pole high
    # 10.5% from 5.22 = low of 4.67, tight range 4.67 to 4.85 = 4%
    (5.20, 4.90, 4.75, 4.80, 400000),   # Bar 6: gap down into flag
    (4.80, 4.85, 4.70, 4.75, 350000),   # Bar 7: consolidating
    (4.75, 4.85, 4.67, 4.80, 300000),   # Bar 8: flag low (4.67), flag high (4.85)

    # Breakout: Must break above flag high (4.85)
    (4.80, 5.00, 4.78, 4.95, 600000),   # Bar 9: BREAKOUT above 4.85
])

# Expected result:
# - detected: True
# - pullback_pct: ~10.5% (just over minimum limit)


# =============================================================================
# BULL FLAG - LIMIT: Just Under 25% Pullback (maximum)
# =============================================================================
# Tests max_pullback_pct: 25.0 at the exact limit.
# Should PASS with 24.5% pullback (just under limit).
# Note: Flag must be BELOW pole high with tight range (<15%).

BULL_FLAG_LIMIT_PULLBACK_DEEP = _make_bars([
    # Very strong pole: 40% move for R:R with deep pullback
    (4.00, 4.22, 4.00, 4.20, 600000),   # Bar 1
    (4.20, 4.52, 4.18, 4.50, 700000),   # Bar 2
    (4.50, 4.88, 4.48, 4.85, 800000),   # Bar 3
    (4.85, 5.28, 4.82, 5.25, 900000),   # Bar 4
    (5.25, 5.62, 5.22, 5.60, 950000),   # Bar 5 (pole high: 5.62, 40.5% from 4.00)

    # Flag: 24.5% pullback, tight consolidation BELOW pole high
    # 24.5% from 5.62 = low of 4.24, tight range 4.24 to 4.65 = 9.6%
    (5.60, 4.65, 4.40, 4.50, 400000),   # Bar 6: gap down into flag
    (4.50, 4.60, 4.30, 4.40, 350000),   # Bar 7: more pullback
    (4.40, 4.65, 4.24, 4.55, 300000),   # Bar 8: flag low (4.24), flag high (4.65)

    # Breakout: Must break above flag high (4.65)
    (4.55, 4.90, 4.52, 4.85, 650000),   # Bar 9: BREAKOUT above 4.65
])

# Expected result:
# - detected: True
# - pullback_pct: ~24.5% (just under maximum limit)


# =============================================================================
# BULL FLAG - LIMIT: Minimum Candle Counts (3 pole + 3 flag)
# =============================================================================
# Tests min_pole_candles: 3 and min_flag_candles: 3 at their minimums.
# Note: min_bars_required is 8. Flag must be BELOW pole high.
# Should PASS with exactly 3 pole + 3 flag + 2 extra = 8 bars total.

BULL_FLAG_LIMIT_MIN_CANDLES = _make_bars([
    # Extra bar for buffer (not part of pole)
    (3.95, 4.02, 3.92, 4.00, 400000),   # Bar 1: lead-in bar

    # Pole: Exactly 3 candles with 25% move
    (4.00, 4.35, 4.00, 4.32, 600000),   # Bar 2: +8%
    (4.32, 4.68, 4.30, 4.65, 700000),   # Bar 3: +7.6%
    (4.65, 5.02, 4.62, 5.00, 800000),   # Bar 4: +7.5% (pole high: 5.02, ~25.5% from 4.00)

    # Flag: Exactly 3 candles, tight consolidation BELOW pole high
    # Flag range: 4.30 to 4.60 = 7% (tight)
    (5.00, 4.60, 4.45, 4.50, 350000),   # Bar 5: gap down into flag
    (4.50, 4.55, 4.35, 4.42, 280000),   # Bar 6: more pullback
    (4.42, 4.60, 4.30, 4.55, 220000),   # Bar 7: flag low (4.30), flag high (4.60)

    # Breakout: Must break above flag high (4.60)
    (4.55, 4.85, 4.52, 4.80, 550000),   # Bar 8: BREAKOUT above 4.60 (total 8 bars)
])

# Expected result:
# - detected: True
# - pole_candles: 3 (at minimum)
# - flag_candles: 3 (at minimum)
