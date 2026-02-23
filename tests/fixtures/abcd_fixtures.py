"""
ABCD Pattern Test Fixtures
==========================

Test data for ABCD harmonic pattern detection.
Fixtures create controlled swing point scenarios.

Naming convention matches BullFlag/MicroPullback:
- ABCD_PASS_* for valid patterns
- ABCD_FAIL_* for invalid patterns
- ABCD_FILTER_* for direction filter tests
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def _make_abcd_bars(
    a_price: float,
    b_price: float,
    c_price: float,
    current_price: float,
    direction: str = "long",
    swing_lookback: int = 3,
) -> pd.DataFrame:
    """
    Create bars with specific A, B, C swing points and current price approaching D.

    For bullish: A=low, B=high, C=higher low, D=projected high
    For bearish: A=high, B=low, C=lower high, D=projected low
    """
    bars = []
    base_time = datetime(2024, 1, 15, 9, 30)

    if direction == "long":
        # Bullish ABCD: A=low, B=high, C=higher low
        # Pre-A: descending to A
        pre_a = [a_price * 1.02, a_price * 1.015, a_price * 1.01]

        # A swing low - needs to be lower than surrounding bars
        a_bar_low = a_price
        a_bar_high = a_price * 1.005

        # AB leg (upward) - progressively higher
        ab_count = 4
        ab_step = (b_price - a_price) / ab_count
        ab_prices = [a_price + ab_step * (i+1) for i in range(ab_count-1)]

        # B swing high - needs to be higher than surrounding bars
        b_bar_high = b_price
        b_bar_low = b_price * 0.995

        # BC leg (downward retracement) - progressively lower
        bc_count = 3
        bc_step = (b_price - c_price) / bc_count
        bc_prices = [b_price - bc_step * (i+1) for i in range(bc_count-1)]

        # C swing low - needs to be lower than surrounding bars
        c_bar_low = c_price
        c_bar_high = c_price * 1.005

        # CD leg (upward toward D) - current price approaching D
        cd_count = 3
        cd_prices = []
        cd_step = (current_price - c_price) / (cd_count + 1)
        for i in range(cd_count):
            cd_prices.append(c_price + cd_step * (i+1))

        # Build the price sequence
        # Pre-A bars (3 bars - need lookback)
        for i, p in enumerate(pre_a):
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.998, "high": p * 1.002, "low": p * 0.996, "close": p,
                "volume": 100000,
            })

        # A bar (swing low)
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": a_bar_high, "high": a_bar_high * 1.001, "low": a_bar_low, "close": a_bar_low * 1.002,
            "volume": 120000,
        })

        # AB move bars
        for p in ab_prices:
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.995, "high": p * 1.003, "low": p * 0.994, "close": p,
                "volume": 100000,
            })

        # B bar (swing high)
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": b_bar_low, "high": b_bar_high, "low": b_bar_low * 0.999, "close": b_bar_high * 0.998,
            "volume": 150000,
        })

        # Post-B bars (need lookback bars that are lower)
        for i in range(swing_lookback):
            p = b_price * (0.99 - i * 0.005)
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 1.002, "high": p * 1.004, "low": p * 0.998, "close": p,
                "volume": 80000,
            })

        # BC move bars (excluding post-B lookback bars already added)
        for p in bc_prices:
            if p < b_price * 0.98:  # Only add if lower than post-B
                bars.append({
                    "timestamp": base_time + timedelta(minutes=len(bars)),
                    "open": p * 1.002, "high": p * 1.004, "low": p * 0.998, "close": p,
                    "volume": 70000,
                })

        # C bar (swing low)
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": c_bar_high, "high": c_bar_high * 1.001, "low": c_bar_low, "close": c_bar_low * 1.003,
            "volume": 90000,
        })

        # Post-C bars (need lookback bars that are higher)
        for i in range(swing_lookback):
            p = c_price * (1.01 + i * 0.005)
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.998, "high": p * 1.003, "low": p * 0.996, "close": p,
                "volume": 85000,
            })

        # CD move bars
        for p in cd_prices:
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.998, "high": p * 1.002, "low": p * 0.996, "close": p,
                "volume": 95000,
            })

        # Current bar
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": current_price * 0.998, "high": current_price * 1.002,
            "low": current_price * 0.996, "close": current_price,
            "volume": 110000,
        })

    else:
        # Bearish ABCD: A=high, B=low, C=lower high
        # Pre-A: ascending to A
        pre_a = [a_price * 0.98, a_price * 0.985, a_price * 0.99]

        # A swing high
        a_bar_high = a_price
        a_bar_low = a_price * 0.995

        # AB leg (downward)
        ab_count = 4
        ab_step = (a_price - b_price) / ab_count
        ab_prices = [a_price - ab_step * (i+1) for i in range(ab_count-1)]

        # B swing low
        b_bar_low = b_price
        b_bar_high = b_price * 1.005

        # BC leg (upward retracement)
        bc_count = 3
        bc_step = (c_price - b_price) / bc_count
        bc_prices = [b_price + bc_step * (i+1) for i in range(bc_count-1)]

        # C swing high (lower than A)
        c_bar_high = c_price
        c_bar_low = c_price * 0.995

        # CD leg (downward toward D)
        cd_count = 3
        cd_prices = []
        cd_step = (c_price - current_price) / (cd_count + 1)
        for i in range(cd_count):
            cd_prices.append(c_price - cd_step * (i+1))

        # Build bars similar to bullish but inverted
        for i, p in enumerate(pre_a):
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.998, "high": p * 1.004, "low": p * 0.996, "close": p,
                "volume": 100000,
            })

        # A bar (swing high)
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": a_bar_low, "high": a_bar_high, "low": a_bar_low * 0.999, "close": a_bar_high * 0.998,
            "volume": 120000,
        })

        # Post-A bars (need lookback bars that are lower highs)
        for i in range(swing_lookback):
            p = a_price * (0.99 - i * 0.005)
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 1.002, "high": p * 1.003, "low": p * 0.998, "close": p,
                "volume": 80000,
            })

        # AB move bars
        for p in ab_prices:
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 1.005, "high": p * 1.006, "low": p * 0.997, "close": p,
                "volume": 100000,
            })

        # B bar (swing low)
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": b_bar_high, "high": b_bar_high * 1.001, "low": b_bar_low, "close": b_bar_low * 1.002,
            "volume": 150000,
        })

        # Post-B bars (need lookback bars that are higher lows)
        for i in range(swing_lookback):
            p = b_price * (1.01 + i * 0.005)
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.998, "high": p * 1.004, "low": p * 0.997, "close": p,
                "volume": 80000,
            })

        # BC move bars
        for p in bc_prices:
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 0.998, "high": p * 1.004, "low": p * 0.996, "close": p,
                "volume": 70000,
            })

        # C bar (swing high, lower than A)
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": c_bar_low, "high": c_bar_high, "low": c_bar_low * 0.999, "close": c_bar_high * 0.998,
            "volume": 90000,
        })

        # Post-C bars (need lookback bars that are lower highs)
        for i in range(swing_lookback):
            p = c_price * (0.99 - i * 0.005)
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 1.002, "high": p * 1.003, "low": p * 0.998, "close": p,
                "volume": 85000,
            })

        # CD move bars
        for p in cd_prices:
            bars.append({
                "timestamp": base_time + timedelta(minutes=len(bars)),
                "open": p * 1.002, "high": p * 1.004, "low": p * 0.997, "close": p,
                "volume": 95000,
            })

        # Current bar
        bars.append({
            "timestamp": base_time + timedelta(minutes=len(bars)),
            "open": current_price * 1.002, "high": current_price * 1.004,
            "low": current_price * 0.997, "close": current_price,
            "volume": 110000,
        })

    return pd.DataFrame(bars)


# =============================================================================
# PASS FIXTURES
# =============================================================================

# Valid bullish ABCD with 50% BC retracement
# A=$10.00, B=$11.50 (AB=15%), C=$10.75 (BC=50%), D=$12.25 projected
ABCD_PASS_BULLISH_VALID = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=11.50,
        c_price=10.75,
        current_price=12.00,
        direction="long",
    ),
    "description": "Valid bullish ABCD with 50% BC retracement",
}

# Valid bearish ABCD with 50% BC retracement
# A=$20.00, B=$17.00 (AB=15%), C=$18.50 (BC=50%), D=$15.50 projected
ABCD_PASS_BEARISH_VALID = {
    "bars": _make_abcd_bars(
        a_price=20.00,
        b_price=17.00,
        c_price=18.50,
        current_price=16.00,
        direction="short",
    ),
    "description": "Valid bearish ABCD with 50% BC retracement",
}

# Valid bullish ABCD with ideal 61.8% Fibonacci retracement
# A=$10.00, B=$12.00 (AB=20%), C=$10.76 (BC=62%), D=$12.76 projected
ABCD_PASS_618_RETRACEMENT = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=10.76,  # 61.8% retracement
        current_price=12.50,
        direction="long",
    ),
    "description": "Valid bullish ABCD with ideal 61.8% retracement",
}

# Valid bullish ABCD with BC retracement at 39% (just above 38.2% minimum)
# A=$10.00, B=$12.00 (AB=$2.00), C=$11.22 (BC=$0.78 = 39%)
ABCD_PASS_MIN_BC_RETRACEMENT = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=11.22,  # 39% retracement
        current_price=13.00,
        direction="long",
    ),
    "description": "BC retracement at 39% (just above 38.2% min)",
}

# Valid bullish ABCD with BC retracement at 78% (just below 78.6% maximum)
# A=$10.00, B=$12.00 (AB=$2.00), C=$10.44 (BC=$1.56 = 78%)
# CD = $12.20 - $10.44 = $1.76 = 88% of AB → passes 80% min completion
ABCD_PASS_MAX_BC_RETRACEMENT = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=10.44,  # 78% retracement
        current_price=12.20,
        direction="long",
    ),
    "description": "BC retracement at 78% (just below 78.6% max)",
}

# Valid bullish ABCD with CD/AB ratio at ~81% (just above 80% min completion)
# A=$10.00, B=$12.00 (AB=$2.00), C=$11.00 (BC=50%), current=$12.62 (CD=$1.62=81%)
ABCD_PASS_MIN_CD_AB_RATIO = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=11.00,
        current_price=12.62,  # CD = $1.62 = 81% of AB
        direction="long",
    ),
    "description": "CD/AB ratio at 81% (just above 80% min completion)",
}

# Valid bullish ABCD with CD/AB ratio at ~120% (approaching max)
# A=$10.00, B=$12.00 (AB=$2.00), C=$11.00, current=$13.40 (CD=$2.40=120%)
ABCD_PASS_MAX_CD_AB_RATIO = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=11.00,
        current_price=13.40,  # CD = $2.40 = 120% of AB
        direction="long",
    ),
    "description": "CD/AB ratio at 120% (approaching 125% max)",
}


# =============================================================================
# FAIL FIXTURES
# =============================================================================

# BC retracement too shallow (25% < 38.2% min)
# A=$10.00, B=$12.00, C=$11.50 (BC=$0.50 = 25%)
ABCD_FAIL_BC_TOO_SHALLOW = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=11.50,  # Only 25% retracement
        current_price=13.00,
        direction="long",
    ),
    "description": "BC retracement 25% < 38.2% minimum",
}

# BC retracement too deep (90% > 78.6% max)
# A=$10.00, B=$12.00, C=$10.20 (BC=$1.80 = 90%)
ABCD_FAIL_BC_TOO_DEEP = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=10.20,  # 90% retracement - too deep
        current_price=11.50,
        direction="long",
    ),
    "description": "BC retracement 90% > 78.6% maximum",
}

# C not higher than A (bullish pattern invalid)
# A=$10.00, B=$12.00, C=$9.50 (below A)
ABCD_FAIL_C_NOT_HIGHER = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=9.50,  # Below A - invalid
        current_price=11.00,
        direction="long",
    ),
    "description": "C below A invalidates bullish ABCD",
}

# CD leg not developed enough (< 80% of AB)
# A=$10.00, B=$12.00, C=$11.00, current=$11.40 (CD=$0.40 = 20%)
ABCD_FAIL_CD_NOT_DEVELOPED = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=12.00,
        c_price=11.00,
        current_price=11.40,  # Only 20% of AB move
        direction="long",
    ),
    "description": "CD only 20% of AB, need at least 80%",
}

# AB leg too small (< 1% move) - flat price action
ABCD_FAIL_AB_TOO_SMALL = {
    "bars": (lambda: pd.DataFrame([{
        "timestamp": datetime(2024, 1, 15, 9, 30) + timedelta(minutes=i),
        "open": 10.00 * (1 + 0.002 * np.sin(i * 0.5)),
        "high": 10.00 * (1 + 0.002 * np.sin(i * 0.5)) * 1.002,
        "low": 10.00 * (1 + 0.002 * np.sin(i * 0.5)) * 0.998,
        "close": 10.00 * (1 + 0.002 * np.sin(i * 0.5)),
        "volume": 100000,
    } for i in range(20)]))(),
    "description": "All price movements < 1% minimum leg size",
}

# Insufficient bars (< 10 minimum)
ABCD_FAIL_INSUFFICIENT_BARS = {
    "bars": pd.DataFrame([{
        "timestamp": datetime(2024, 1, 15, 9, 30) + timedelta(minutes=i),
        "open": 10.00, "high": 10.05, "low": 9.95, "close": 10.02,
        "volume": 100000,
    } for i in range(5)]),
    "description": "Only 5 bars, need at least 10",
}

# No swing points (flat price action)
ABCD_FAIL_NO_SWING_POINTS = {
    "bars": pd.DataFrame([{
        "timestamp": datetime(2024, 1, 15, 9, 30) + timedelta(minutes=i),
        "open": 10.00, "high": 10.01, "low": 9.99, "close": 10.00,
        "volume": 100000,
    } for i in range(20)]),
    "description": "Flat price action, no swing points",
}


# =============================================================================
# DIRECTION FILTER FIXTURES
# =============================================================================

# Bearish pattern present but filter is 'long'
ABCD_FILTER_BULLISH_ONLY = {
    "bars": _make_abcd_bars(
        a_price=20.00,
        b_price=17.00,
        c_price=18.50,
        current_price=16.00,
        direction="short",
    ),
    "config": {"direction_filter": "long"},
    "description": "Bearish pattern not detected when filter is 'long'",
}

# Bullish pattern present but filter is 'short'
ABCD_FILTER_BEARISH_ONLY = {
    "bars": _make_abcd_bars(
        a_price=10.00,
        b_price=11.50,
        c_price=10.75,
        current_price=12.00,
        direction="long",
    ),
    "config": {"direction_filter": "short"},
    "description": "Bullish pattern not detected when filter is 'short'",
}
