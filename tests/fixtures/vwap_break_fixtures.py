"""
VWAP Break Test Fixtures
========================

Real-world example bars for testing VWAP break detection.
"""

import pandas as pd
from datetime import datetime, timedelta


def _make_bars_with_vwap(data: list) -> tuple:
    """
    Create DataFrame and VWAP series from data.

    Args:
        data: List of (open, high, low, close, volume, vwap) tuples

    Returns:
        Tuple of (bars_df, vwap_series)
    """
    base_time = datetime(2025, 1, 15, 9, 30)
    rows = []
    vwap_values = []

    for i, (o, h, l, c, v, vw) in enumerate(data):
        rows.append({
            "timestamp": base_time + timedelta(minutes=i),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
        vwap_values.append(vw)

    df = pd.DataFrame(rows)
    vwap = pd.Series(vwap_values, name="vwap")

    return df, vwap


# =============================================================================
# VALID VWAP BREAK
# =============================================================================
# Pattern:
#   Bars 1-5: Trading below VWAP
#   Bar 6: Breaks above VWAP with volume spike
#   Bar 7: Closes above VWAP (confirmation)
#
#   VWAP line: ~~~~~~~~~ 5.20 ~~~~~~~~~
#                     /
#   Price:    ___/\__/   <- Break above VWAP
#

VWAP_BREAK_VALID = _make_bars_with_vwap([
    # Trading below VWAP (5 bars)
    (5.10, 5.15, 5.05, 5.08, 100000, 5.20),  # Bar 1: below VWAP
    (5.08, 5.12, 5.02, 5.05, 90000, 5.19),   # Bar 2: below VWAP
    (5.05, 5.18, 5.00, 5.10, 110000, 5.18),  # Bar 3: tested but failed
    (5.10, 5.14, 5.06, 5.08, 95000, 5.18),   # Bar 4: below VWAP
    (5.08, 5.16, 5.05, 5.15, 120000, 5.17),  # Bar 5: approaching VWAP

    # Break above VWAP with volume spike
    (5.15, 5.28, 5.14, 5.25, 350000, 5.17),  # Bar 6: BREAK! Volume spike
    (5.25, 5.35, 5.22, 5.32, 300000, 5.18),  # Bar 7: Closes above VWAP
])

# Expected result:
# - detected: True
# - pattern_name: "VWAPBreak"
# - entry_price: ~5.19 (above VWAP)
# - stop_price: ~5.07 (below VWAP with buffer)
# - volume_confirmation: True


# =============================================================================
# VALID VWAP HOLD
# =============================================================================
# Pattern: Price trades below VWAP, breaks above, pulls back to VWAP,
# holds as support, continues up
#
#              ___/
#   VWAP: ~~~~~|~~~~~  <- VWAP acts as support
#             \/
#            pullback touches VWAP, holds, bounces

VWAP_HOLD_VALID = _make_bars_with_vwap([
    # First: Trading below VWAP (5 consecutive bars with close < VWAP)
    (5.05, 5.10, 5.02, 5.06, 100000, 5.20),  # Bar 1: 5.06 < 5.20 ✓
    (5.06, 5.12, 5.03, 5.08, 110000, 5.19),  # Bar 2: 5.08 < 5.19 ✓
    (5.08, 5.14, 5.05, 5.10, 120000, 5.18),  # Bar 3: 5.10 < 5.18 ✓
    (5.10, 5.16, 5.08, 5.12, 130000, 5.17),  # Bar 4: 5.12 < 5.17 ✓
    (5.12, 5.18, 5.10, 5.14, 140000, 5.16),  # Bar 5: 5.14 < 5.16 ✓

    # Break above VWAP
    (5.14, 5.28, 5.12, 5.25, 350000, 5.16),  # Bar 6: 5.25 > 5.16 BREAK!

    # Pullback to VWAP, holds as support
    (5.25, 5.26, 5.16, 5.18, 150000, 5.17),  # Bar 7: low 5.16 touches VWAP 5.17
    (5.18, 5.32, 5.17, 5.30, 280000, 5.18),  # Bar 8: GREEN, closes above VWAP
])

# Expected result:
# - detected: True
# - pattern_name: "VWAPHold"
# - entry_price: ~5.20 (above VWAP)
# - stop_price: ~5.09 (below touch low)


# =============================================================================
# VWAP BREAK - NO PERIOD BELOW
# =============================================================================
# Stock is already trading above VWAP, no setup

VWAP_BREAK_ALREADY_ABOVE = _make_bars_with_vwap([
    (5.30, 5.35, 5.28, 5.32, 100000, 5.20),  # Already above
    (5.32, 5.40, 5.30, 5.38, 120000, 5.21),  # Above
    (5.38, 5.45, 5.35, 5.42, 130000, 5.22),  # Above
    (5.42, 5.50, 5.40, 5.48, 150000, 5.23),  # Above
    (5.48, 5.55, 5.45, 5.52, 160000, 5.24),  # Above
    (5.52, 5.60, 5.50, 5.58, 170000, 5.25),  # Above (need 6 bars)
])

# Expected result:
# - detected: False
# - reason: "No period of trading below VWAP found"


# =============================================================================
# VWAP BREAK - NO VOLUME SPIKE
# =============================================================================
# Breaks VWAP but without conviction (low volume)

VWAP_BREAK_WEAK_VOLUME = _make_bars_with_vwap([
    # Trading below VWAP
    (5.10, 5.15, 5.05, 5.08, 100000, 5.20),
    (5.08, 5.12, 5.02, 5.05, 90000, 5.19),
    (5.05, 5.18, 5.00, 5.10, 110000, 5.18),
    (5.10, 5.14, 5.06, 5.08, 95000, 5.18),
    (5.08, 5.16, 5.05, 5.15, 100000, 5.17),

    # Break above VWAP but LOW volume (no spike)
    (5.15, 5.22, 5.14, 5.20, 80000, 5.17),   # Break but weak volume
    (5.20, 5.25, 5.18, 5.23, 90000, 5.18),   # Still low volume
])

# Expected result:
# - detected: True (break is still valid)
# - volume_confirmation: False (no spike)
# - confidence: Lower due to lack of volume


# =============================================================================
# VWAP BREAK - FAILED HOLD
# =============================================================================
# Pulls back to VWAP but fails to hold, breaks down

VWAP_FAILED_HOLD = _make_bars_with_vwap([
    # Initial move above VWAP
    (5.15, 5.25, 5.12, 5.22, 200000, 5.10),
    (5.22, 5.35, 5.20, 5.32, 250000, 5.12),

    # Pullback to VWAP
    (5.32, 5.33, 5.18, 5.20, 150000, 5.15),
    (5.20, 5.22, 5.14, 5.16, 120000, 5.16),  # Touches VWAP

    # FAILS to hold - breaks below VWAP
    (5.16, 5.17, 5.08, 5.10, 180000, 5.16),  # RED, closes BELOW VWAP
    (5.10, 5.12, 5.02, 5.05, 200000, 5.15),  # Continues down
])

# Expected result:
# - detected: False (for VWAPHold)
# - reason: Next bar after touch is red and below VWAP
