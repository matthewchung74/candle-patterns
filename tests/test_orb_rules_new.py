"""
ORB rule coverage tests (1-min bar sequences).

These are narrow, bar-based tests to exercise key ORB rules:
- Range timing (9:30-9:35 only)
- Breakout + retest + confirmation (bullish/bearish)
- Same-side fakeout reset
- No retest rejection
- Window/kill time
"""

import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from candle_patterns import OpeningRangeRetest

ET = ZoneInfo("America/New_York")


def make_bar(ts, o, h, l, c, v=100_000):
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def make_df(start, bars):
    rows = []
    for i, bar in enumerate(bars):
        # Allow passing full dicts or tuples
        if isinstance(bar, dict):
            rows.append(bar)
        else:
            o, h, l, c, v = bar
            rows.append(make_bar(start + timedelta(minutes=i), o, h, l, c, v))
    return pd.DataFrame(rows)


class TestORBRuleCoverage:
    def setup_method(self):
        # Enable ORB (disabled by default) and disable optional filters for deterministic behavior
        self.det = OpeningRangeRetest({
            "enabled": True,
            "trend_alignment": False,
            "fakeout_filter": True,
            "choppy_filter": False,
            "confirmation_filter": False,
            "require_clean_breakout_bar": False,
        })
        self.start = datetime(2025, 1, 15, 9, 30, tzinfo=ET)

    def test_long_breakout_retest_engulfing(self):
        """Bullish breakout, retest touch, and strong bullish retest bar -> detected."""
        bars = make_df(self.start, [
            (100.0, 100.2, 99.7, 100.1, 100000),
            (100.1, 100.5, 99.8, 100.4, 100000),  # ORH=100.5
            (100.4, 100.5, 99.9, 100.0, 100000),
            (100.0, 100.3, 99.6, 100.0, 100000),
            (100.0, 100.4, 99.5, 99.9, 100000),   # ORL=99.5
            (100.0, 100.2, 99.9, 100.1, 90000),   # 9:35
            # Breakout close > ORH + disp
            (100.4, 101.0, 100.3, 100.8, 150000),
            # Pullback bar: low touches retest zone (ORH area)
            (100.7, 100.85, 100.5, 100.6, 100000),
            # Confirmation bar: bullish, closes above ORH
            (100.6, 101.1, 100.55, 101.0, 180000),
            # Entry candle (current bar)
            (100.95, 101.15, 100.9, 101.1, 160000),
        ])

        result = self.det.detect(bars)
        assert result.detected is True, result.reason
        assert result.entry_price is not None
        assert result.stop_price is not None

    def test_bear_breakout_retest_pinbar(self):
        """Bearish breakout, retest touch, bearish pinbar-style retest -> detected."""
        bars = make_df(self.start, [
            (100.0, 100.2, 99.7, 100.1, 100000),
            (100.1, 100.5, 99.8, 100.4, 100000),  # ORH=100.5
            (100.4, 100.5, 99.9, 100.0, 100000),
            (100.0, 100.3, 99.6, 100.0, 100000),
            (100.0, 100.4, 99.5, 99.9, 100000),   # ORL=99.5
            (100.0, 100.2, 99.9, 100.1, 90000),   # 9:35
            # Breakout close < ORL - disp (disp=0.2 -> need close < 99.3)
            (99.4, 99.5, 99.1, 99.2, 160000),
            # Pullback bar: high touches retest zone (ORL area)
            (99.25, 99.5, 99.2, 99.35, 100000),
            # Confirmation bar: bearish, closes below ORL
            (99.35, 99.55, 99.1, 99.15, 150000),
            # Entry candle (current bar)
            (99.2, 99.25, 99.05, 99.1, 140000),
        ])

        result = self.det.detect(bars)
        assert result.detected is True, result.reason
        assert result.entry_price is not None
        assert result.stop_price is not None

    def test_fakeout_resets(self):
        """Breakout then close back inside range should reject when fakeout_filter enabled."""
        bars = make_df(self.start, [
            (100.0, 100.2, 99.7, 100.1, 100000),
            (100.1, 100.5, 99.8, 100.4, 100000),  # ORH=100.5
            (100.4, 100.5, 99.9, 100.0, 100000),
            (100.0, 100.3, 99.6, 100.0, 100000),
            (100.0, 100.4, 99.5, 99.9, 100000),
            # Breakout
            (100.4, 101.0, 100.3, 100.8, 150000),
            # Close back inside range -> fakeout
            (100.7, 100.8, 99.9, 100.0, 150000),
            # Retest-like bar should not rescue the setup
            (99.9, 100.6, 99.8, 100.5, 150000),
        ])

        result = self.det.detect(bars)
        assert result.detected is False

    def test_no_retest_rejected(self):
        """Breakout with no pullback to OR level should reject."""
        bars = make_df(self.start, [
            (100.0, 100.2, 99.7, 100.1, 100000),
            (100.1, 100.5, 99.8, 100.4, 100000),
            (100.4, 100.5, 99.9, 100.0, 100000),
            (100.0, 100.3, 99.6, 100.0, 100000),
            (100.0, 100.4, 99.5, 99.9, 100000),
            # Breakout and drift above, never retest
            (100.4, 101.0, 100.3, 100.8, 150000),
            (100.9, 101.2, 100.8, 101.1, 140000),  # low > zone_high (100.7) -> no retest
            (101.1, 101.3, 101.0, 101.2, 120000),
            (101.2, 101.4, 101.1, 101.3, 110000),
        ])

        result = self.det.detect(bars)
        assert result.detected is False

    def test_range_incomplete_rejected(self):
        """Fewer than 5 bars before 9:35 should reject."""
        bars = make_df(self.start, [
            (100.0, 100.2, 99.7, 100.1, 100000),
            (100.1, 100.5, 99.8, 100.4, 100000),
            (100.4, 100.5, 99.9, 100.0, 100000),
            (100.0, 100.3, 99.6, 100.0, 100000),
        ])
        result = self.det.detect(bars)
        assert result.detected is False
        assert "insufficient" in (result.reason or "").lower()

    def test_outside_window_rejected(self):
        """Bars past the 90-minute window should reject."""
        # Reuse fixture already extended past 11:00
        from tests.fixtures.opening_range_retest_fixtures import OPENING_RANGE_RETEST_OUTSIDE_WINDOW

        result = self.det.detect(OPENING_RANGE_RETEST_OUTSIDE_WINDOW)
        assert result.detected is False
        assert "window" in (result.reason or "").lower()

    def test_confirmation_bar_no_longer_requires_zone_touch(self):
        """With basic confirmation, a reclaim without zone dip is allowed."""
        from tests.fixtures.opening_range_retest_fixtures import OPENING_RANGE_RETEST_CONFIRM_NO_TOUCH

        result = self.det.detect(OPENING_RANGE_RETEST_CONFIRM_NO_TOUCH)
        assert result.detected is True, result.reason
