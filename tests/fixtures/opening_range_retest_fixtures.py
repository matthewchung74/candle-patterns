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
# VALID ORB RETEST (LONG)
# =============================================================================

_ORB_VALID_DATA = [
    # Opening Range (9:30-9:34)
    (100.0, 100.2, 99.7, 100.1, 100000),
    (100.1, 100.5, 99.8, 100.4, 100000),  # OR high = 100.5
    (100.4, 100.5, 99.9, 100.0, 100000),
    (100.0, 100.3, 99.6, 100.0, 100000),
    (100.0, 100.4, 99.5, 99.9, 100000),   # OR low = 99.5

    # Post-OR drift
    (99.9, 100.2, 99.9, 100.1, 90000),

    # Breakout with displacement + FVG (strong body)
    (100.7, 101.2, 100.7, 101.15, 200000),

    # Drift above OR
    (101.0, 101.1, 100.9, 101.0, 110000),
    (101.0, 101.1, 100.9, 101.0, 110000),
    (101.0, 101.1, 100.9, 101.0, 110000),

    # Retest (last bar) within OR high zone
    (100.9, 100.8, 100.4, 100.6, 150000),
]

OPENING_RANGE_RETEST_VALID = _make_bars(
    _ORB_VALID_DATA,
    datetime(2026, 1, 6, 9, 30),
)


# =============================================================================
# ORB RETEST - NO RETEST
# =============================================================================

_ORB_NO_RETEST_DATA = _ORB_VALID_DATA[:-1] + [
    # Last bar does not retest zone
    (101.0, 101.2, 100.9, 101.1, 120000),
]

OPENING_RANGE_RETEST_NO_RETEST = _make_bars(
    _ORB_NO_RETEST_DATA,
    datetime(2026, 1, 6, 9, 30),
)


# =============================================================================
# ORB RETEST - OUTSIDE WINDOW
# =============================================================================

_ORB_OUTSIDE_WINDOW = _ORB_VALID_DATA + [
    # Extend beyond 11:00 AM ET
    (100.9, 101.0, 100.8, 100.9, 90000),
] * 95  # Push last bar past 11:00

OPENING_RANGE_RETEST_OUTSIDE_WINDOW = _make_bars(
    _ORB_OUTSIDE_WINDOW,
    datetime(2026, 1, 6, 9, 30),
)
