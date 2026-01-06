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
#   Bars 1-5: Trading below VWAP (deeper lows for R:R compliance)
#   Bar 6: Breaks above VWAP with volume spike
#   Bar 7: Closes above VWAP (confirmation)
#
# Designed to pass R:R check (min 2.0):
# - recapture_move = current_vwap - below_period_low >= 2x risk
# - With VWAP ~5.18, low ~4.85, recapture = 0.33, risk = 0.12, R:R = 2.75
#
#   VWAP line: ~~~~~~~~~ 5.18 ~~~~~~~~~
#                         /
#   Price:    ___/\__/\__/   <- Break above VWAP
#

VWAP_BREAK_VALID = _make_bars_with_vwap([
    # Trading below VWAP (5 bars) - deeper lows for R:R compliance
    (5.05, 5.10, 4.92, 4.98, 100000, 5.20),  # Bar 1: below VWAP, low 4.92
    (4.98, 5.02, 4.88, 4.94, 90000, 5.19),   # Bar 2: below VWAP, low 4.88
    (4.94, 5.08, 4.85, 4.98, 110000, 5.18),  # Bar 3: key low 4.85
    (4.98, 5.05, 4.90, 5.00, 95000, 5.18),   # Bar 4: below VWAP
    (5.00, 5.12, 4.95, 5.08, 120000, 5.17),  # Bar 5: approaching VWAP

    # Break above VWAP with volume spike
    (5.08, 5.30, 5.06, 5.25, 350000, 5.17),  # Bar 6: BREAK! Volume spike
    (5.25, 5.40, 5.22, 5.35, 300000, 5.18),  # Bar 7: Closes above VWAP
])

# Expected result:
# - detected: True
# - pattern_name: "VWAPBreak"
# - entry_price: ~5.20 (VWAP + 0.02)
# - stop_price: ~5.08 (VWAP - 0.10)
# - R:R: (5.18 - 4.85) / 0.12 = 2.75 ✓
# - volume_confirmation: True


# =============================================================================
# VALID VWAP HOLD
# =============================================================================
# Pattern: Price trades below VWAP, breaks above, pulls back to VWAP,
# holds as support, continues up
#
# Note: Test accepts either VWAPBreak or VWAPHold detection.
# Design with deep lows to ensure R:R >= 2.0 for VWAPBreak detection.
#
#              ___/
#   VWAP: ~~~~~|~~~~~  <- VWAP acts as support
#             \/
#            pullback touches VWAP, holds, bounces

VWAP_HOLD_VALID = _make_bars_with_vwap([
    # Trading below VWAP (5 bars) - deeper lows for R:R compliance
    (5.00, 5.05, 4.88, 4.92, 100000, 5.20),  # Bar 1: below VWAP, low 4.88
    (4.92, 5.00, 4.85, 4.90, 110000, 5.19),  # Bar 2: below VWAP, low 4.85 (key)
    (4.90, 5.02, 4.86, 4.95, 120000, 5.18),  # Bar 3: below VWAP
    (4.95, 5.08, 4.90, 5.00, 130000, 5.17),  # Bar 4: below VWAP
    (5.00, 5.12, 4.95, 5.08, 140000, 5.16),  # Bar 5: approaching VWAP

    # Break above VWAP
    (5.08, 5.28, 5.05, 5.25, 350000, 5.16),  # Bar 6: BREAK!

    # Pullback to VWAP, holds as support
    (5.25, 5.26, 5.16, 5.18, 150000, 5.17),  # Bar 7: low 5.16 touches VWAP 5.17
    (5.18, 5.35, 5.17, 5.32, 280000, 5.18),  # Bar 8: GREEN bounce, closes above VWAP
])

# Expected result:
# - detected: True
# - pattern_name: "VWAPBreak" or "VWAPHold" (test accepts either)
# - R:R for VWAPBreak: (5.18 - 4.85) / 0.12 = 2.75 ✓
# - entry_price: ~5.20 (above VWAP)
# - stop_price: ~5.08 (VWAP - 0.10)


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


# =============================================================================
# LIMIT TESTS - Testing rules at their boundaries
# =============================================================================

# =============================================================================
# VWAP BREAK - LIMIT: Exactly 5 Bars Below VWAP (minimum)
# =============================================================================
# Tests min_time_below_minutes: 5 at the exact limit.
# Should PASS with exactly 5 bars below VWAP.

VWAP_BREAK_LIMIT_BARS_BELOW = _make_bars_with_vwap([
    # Exactly 5 bars below VWAP (minimum required)
    (5.00, 5.05, 4.85, 4.90, 100000, 5.20),  # Bar 1: below VWAP
    (4.90, 4.98, 4.82, 4.88, 110000, 5.19),  # Bar 2: below VWAP
    (4.88, 4.95, 4.78, 4.85, 120000, 5.18),  # Bar 3: below VWAP, low 4.78
    (4.85, 4.92, 4.80, 4.90, 130000, 5.17),  # Bar 4: below VWAP
    (4.90, 5.10, 4.88, 5.05, 140000, 5.16),  # Bar 5: below VWAP (5 bars total)

    # Break above VWAP with volume spike
    (5.05, 5.30, 5.02, 5.25, 400000, 5.16),  # Bar 6: BREAK! (2.8x volume)
    (5.25, 5.40, 5.22, 5.35, 350000, 5.17),  # Bar 7: Closes above VWAP
])

# Expected result:
# - detected: True
# - bars_below_vwap: 5 (at minimum)
# - R:R: (5.17 - 4.78) / 0.12 = 3.25 ✓


# =============================================================================
# VWAP BREAK - LIMIT: Exactly 2.0x Volume Spike (minimum)
# =============================================================================
# Tests volume_spike_on_break: 2.0 at the exact limit.
# Should have volume_confirmation: True with exactly 2.0x average.

VWAP_BREAK_LIMIT_VOLUME_SPIKE = _make_bars_with_vwap([
    # Trading below VWAP (5 bars with ~100k avg volume)
    (5.00, 5.05, 4.85, 4.92, 100000, 5.20),  # Bar 1
    (4.92, 4.98, 4.80, 4.88, 100000, 5.19),  # Bar 2
    (4.88, 4.95, 4.78, 4.85, 100000, 5.18),  # Bar 3: low 4.78
    (4.85, 4.92, 4.82, 4.90, 100000, 5.17),  # Bar 4
    (4.90, 5.10, 4.88, 5.05, 100000, 5.16),  # Bar 5

    # Break with exactly 2.0x volume (avg = ~100k, so need 200k)
    # But avg includes all bars, so with 7 bars:
    # avg = (100+100+100+100+100+200+200)/7 = 142k, need 284k for spike
    # Let's make break bar exactly 2.0x of pre-break average
    (5.05, 5.28, 5.02, 5.24, 200000, 5.16),  # Bar 6: BREAK! (2.0x of 100k avg)
    (5.24, 5.38, 5.20, 5.35, 180000, 5.17),  # Bar 7: Continues above VWAP
])

# Expected result:
# - detected: True
# - volume_confirmation: True (at limit)


# =============================================================================
# VWAP BREAK - LIMIT: R:R Exactly 2.0 (minimum)
# =============================================================================
# Tests min_rr_for_setup: 2.0 at the exact limit.
# Designed so recapture_move / risk = exactly ~2.0.

VWAP_BREAK_LIMIT_RR = _make_bars_with_vwap([
    # Trading below VWAP with controlled low for R:R = 2.0
    # entry = VWAP + 0.02 = 5.18 + 0.02 = 5.20
    # stop = VWAP - 0.10 = 5.18 - 0.10 = 5.08
    # risk = 5.20 - 5.08 = 0.12
    # For R:R = 2.0, need recapture_move = 0.24
    # recapture_move = current_vwap - below_period_low = 5.18 - low
    # low = 5.18 - 0.24 = 4.94
    (5.05, 5.10, 4.95, 4.98, 100000, 5.20),  # Bar 1: below VWAP, low 4.95
    (4.98, 5.02, 4.94, 4.96, 110000, 5.19),  # Bar 2: below VWAP, low 4.94 (key)
    (4.96, 5.00, 4.95, 4.98, 120000, 5.18),  # Bar 3: below VWAP
    (4.98, 5.08, 4.96, 5.02, 130000, 5.18),  # Bar 4: below VWAP
    (5.02, 5.12, 5.00, 5.08, 140000, 5.17),  # Bar 5: below VWAP

    # Break above VWAP
    (5.08, 5.28, 5.05, 5.24, 350000, 5.17),  # Bar 6: BREAK!
    (5.24, 5.35, 5.20, 5.32, 300000, 5.18),  # Bar 7: current_vwap = 5.18
])

# Expected result:
# - detected: True
# - R:R: (5.18 - 4.94) / 0.12 = 2.0 (at minimum limit)


# =============================================================================
# VWAP BREAK - LIMIT: Close Barely Above VWAP
# =============================================================================
# Tests that close > VWAP is sufficient (even by $0.01).
# Should PASS with close just barely above VWAP.

VWAP_BREAK_LIMIT_CLOSE_ABOVE = _make_bars_with_vwap([
    # Trading below VWAP with deep lows for R:R
    (5.00, 5.05, 4.85, 4.90, 100000, 5.20),  # Bar 1
    (4.90, 4.98, 4.80, 4.88, 110000, 5.19),  # Bar 2
    (4.88, 4.95, 4.75, 4.85, 120000, 5.18),  # Bar 3: low 4.75 for R:R
    (4.85, 4.92, 4.80, 4.90, 130000, 5.17),  # Bar 4
    (4.90, 5.08, 4.88, 5.05, 140000, 5.16),  # Bar 5

    # Break where close is BARELY above VWAP
    (5.05, 5.20, 5.02, 5.17, 350000, 5.16),  # Bar 6: close 5.17 > VWAP 5.16 ✓
    (5.17, 5.22, 5.15, 5.19, 300000, 5.18),  # Bar 7: close 5.19 > VWAP 5.18 (barely)
])

# Expected result:
# - detected: True
# - Close is only $0.01 above VWAP (at limit)
