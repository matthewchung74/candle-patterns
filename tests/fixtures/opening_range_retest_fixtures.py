"""
Opening Range Retest Test Fixtures
==================================

Synthetic bars for testing the OpeningRangeRetest detector.
"""

import pandas as pd
from datetime import datetime, timedelta


def _make_bars(data: list, start_time: datetime) -> pd.DataFrame:
    """Create DataFrame from OHLCV tuples."""
    rows = []
    for i, (o, h, l, c, v) in enumerate(data):
        rows.append({
            "timestamp": start_time + timedelta(minutes=i),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
    return pd.DataFrame(rows)


# =============================================================================
# VALID ORB RETEST (LONG) - BREAKOUT, RETEST, ENGULFING
# =============================================================================

_ORB_VALID_LONG = [
    # Opening Range (9:30-9:34) ORH=100.5, ORL=99.5
    (100.0, 100.2, 99.7, 100.1, 100000),
    (100.1, 100.5, 99.8, 100.4, 100000),  # OR high = 100.5
    (100.4, 100.5, 99.9, 100.0, 100000),
    (100.0, 100.3, 99.6, 100.0, 100000),
    (100.0, 100.4, 99.5, 99.9, 100000),   # OR low = 99.5

    # Post-OR drift (9:35-9:36)
    (99.9, 100.2, 99.9, 100.1, 90000),
    (100.0, 100.3, 99.9, 100.2, 90000),

    # Breakout bar (close > ORH)
    (100.4, 101.0, 100.3, 100.9, 150000),

    # Retest + Bullish Engulfing:
    # - Low touches ORH (100.5)
    # - Body larger than prior bar's body
    # - Close above prior high
    (100.6, 101.25, 100.45, 101.20, 180000),
]

OPENING_RANGE_RETEST_VALID = _make_bars(
    _ORB_VALID_LONG,
    datetime(2026, 1, 6, 9, 30),
)


# =============================================================================
# ORB RETEST - NO RETEST (STAYS ABOVE)
# =============================================================================

_ORB_NO_RETEST = [
    # Opening Range
    (100.0, 100.2, 99.7, 100.1, 100000),
    (100.1, 100.5, 99.8, 100.4, 100000),
    (100.4, 100.5, 99.9, 100.0, 100000),
    (100.0, 100.3, 99.6, 100.0, 100000),
    (100.0, 100.4, 99.5, 99.9, 100000),

    # Breakout and stay above, never retests ORH
    (100.5, 101.1, 100.5, 101.0, 140000),
    (101.0, 101.2, 100.9, 101.1, 120000),
    (101.1, 101.3, 101.0, 101.2, 120000),
]

OPENING_RANGE_RETEST_NO_RETEST = _make_bars(
    _ORB_NO_RETEST,
    datetime(2026, 1, 6, 9, 30),
)


# =============================================================================
# ORB RETEST - FAKEOUT (CLOSES BACK INSIDE)
# =============================================================================

_ORB_FAKEOUT = [
    # Opening Range
    (100.0, 100.2, 99.7, 100.1, 100000),
    (100.1, 100.5, 99.8, 100.4, 100000),
    (100.4, 100.5, 99.9, 100.0, 100000),
    (100.0, 100.3, 99.6, 100.0, 100000),
    (100.0, 100.4, 99.5, 99.9, 100000),

    # Breakout close above ORH
    (100.4, 101.0, 100.3, 100.9, 150000),
    # Next bar closes back inside range (fakeout)
    (100.9, 100.95, 99.9, 100.0, 150000),
    # Even if later retest happens, setup should be invalid
    (99.9, 100.6, 99.8, 100.5, 150000),
]

OPENING_RANGE_RETEST_FAKEOUT = _make_bars(
    _ORB_FAKEOUT,
    datetime(2026, 1, 6, 9, 30),
)


# =============================================================================
# ORB RETEST - OUTSIDE WINDOW (PAST 11:00)
# =============================================================================

_ORB_OUTSIDE_WINDOW = _ORB_VALID_LONG + [
    (100.9, 101.0, 100.8, 100.95, 90000),
] * 95  # Extend to push end past 11:00

OPENING_RANGE_RETEST_OUTSIDE_WINDOW = _make_bars(
    _ORB_OUTSIDE_WINDOW,
    datetime(2026, 1, 6, 9, 30),
)
