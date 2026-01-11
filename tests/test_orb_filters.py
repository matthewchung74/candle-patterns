"""
Unit tests for ORB Filter Logic
================================

Tests for:
1. fakeout_filter - Same-side fakeout detection
2. choppy_filter - Consolidation/chop detection
3. confirmation_filter - Entry bar quality

Each filter has 3 test scenarios designed to expose potential bugs.
"""

import pytest
import pandas as pd
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from candle_patterns.opening_range_retest import OpeningRangeRetest


def create_bar(ts: datetime, o: float, h: float, l: float, c: float, v: int = 10000) -> dict:
    """Helper to create a single bar."""
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def create_bars_df(bars: list) -> pd.DataFrame:
    """Convert list of bar dicts to DataFrame."""
    return pd.DataFrame(bars)


def make_session_date(year=2025, month=11, day=10) -> datetime:
    """Create a session date for testing."""
    return datetime(year, month, day)


# =============================================================================
# FAKEOUT FILTER TESTS
# =============================================================================

class TestFakeoutFilter:
    """
    Fakeout filter checks if price closes back inside OR within N bars after breakout.

    Current implementation (lines 226-237):
    - Checks bars 1 to fakeout_bars (default 2) after breakout
    - For longs: triggers if check_bar["close"] < or_high
    - For shorts: triggers if check_bar["close"] > or_low

    Potential bugs:
    1. Only checks first N bars, not entire path to retest
    2. Breakout at end of data passes silently
    3. Default value in get() is True but config default is False
    """

    @pytest.fixture
    def detector(self):
        """Create detector with fakeout_filter enabled."""
        det = OpeningRangeRetest()
        det.config["fakeout_filter"] = True
        det.config["fakeout_bars"] = 2
        det.config["require_clean_breakout_bar"] = False  # Disable to isolate fakeout test
        det.config["choppy_filter"] = False
        det.config["confirmation_filter"] = False
        det.config["trend_alignment"] = False  # Disable trend check
        return det

    def test_fakeout_immediate_failure_detected(self, detector):
        """
        Scenario 1: Breakout at 9:40, bar at 9:41 closes back below OR high.
        Expected: Should trigger fakeout and reject the setup.

        OR: 9:30-9:35, high=100, low=99
        Breakout bar (9:40): closes at 100.30 (above OR high + displacement)
        Fakeout bar (9:41): closes at 99.80 (below OR high)
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # Opening range (9:30-9:34)
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),  # OR bar 1
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),   # OR bar 2
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),   # OR bar 3
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),   # OR bar 4
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),   # OR bar 5
            # Post-OR
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.30, 99.50),   # 9:35
            create_bar(base + timedelta(minutes=6), 99.50, 99.70, 99.40, 99.60),   # 9:36
            create_bar(base + timedelta(minutes=7), 99.60, 99.80, 99.50, 99.70),   # 9:37
            create_bar(base + timedelta(minutes=8), 99.70, 99.90, 99.60, 99.80),   # 9:38
            create_bar(base + timedelta(minutes=9), 99.80, 100.00, 99.70, 99.90),  # 9:39
            # Breakout with displacement (9:40) - closes above OR high + disp
            # OR range = 1.00, disp = max(1.00*0.20, 0.05) = 0.20
            # Need close > 100 + 0.20 = 100.20
            create_bar(base + timedelta(minutes=10), 99.90, 100.35, 99.85, 100.30),  # 9:40 breakout
            # Fakeout bar - closes back below OR high
            create_bar(base + timedelta(minutes=11), 100.30, 100.35, 99.70, 99.80),  # 9:41 FAKEOUT
            # Entry bar (would be retest)
            create_bar(base + timedelta(minutes=12), 99.80, 100.10, 99.75, 100.05),  # 9:42
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # Should NOT detect pattern due to fakeout
        assert not result.detected, f"Should reject due to fakeout, got: {result.reason}"
        assert "fakeout" in (result.reason or "").lower(), \
            f"Reason should mention fakeout: {result.reason}"

    def test_fakeout_no_failure_passes(self, detector):
        """
        Scenario 2: Breakout at 9:40, bars 9:41 and 9:42 both close above OR high.
        Expected: Should NOT trigger fakeout, pattern can proceed.

        OR: 9:30-9:35, high=100, low=99
        Breakout bar (9:40): closes at 100.30
        Bar 9:41: closes at 100.25 (above OR high)
        Bar 9:42: closes at 100.20 (above OR high)
        Retest bar (9:43): pulls back to zone and reclaims
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # Opening range (9:30-9:34)
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Post-OR buildup
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.30, 99.50),
            create_bar(base + timedelta(minutes=6), 99.50, 99.70, 99.40, 99.60),
            create_bar(base + timedelta(minutes=7), 99.60, 99.80, 99.50, 99.70),
            create_bar(base + timedelta(minutes=8), 99.70, 99.90, 99.60, 99.80),
            create_bar(base + timedelta(minutes=9), 99.80, 100.00, 99.70, 99.90),
            # Breakout with displacement (9:40)
            create_bar(base + timedelta(minutes=10), 99.90, 100.35, 99.85, 100.30),
            # No fakeout - stays above OR high
            create_bar(base + timedelta(minutes=11), 100.30, 100.40, 100.15, 100.25),  # Above ORH
            create_bar(base + timedelta(minutes=12), 100.25, 100.30, 100.10, 100.20),  # Above ORH
            # Retest bar - pulls back to zone (zone = ORH +/- 20% of range = 99.80 to 100.20)
            # Must: touch zone (low <= 100.20), reclaim (high > 100), bullish (close > open)
            create_bar(base + timedelta(minutes=13), 100.05, 100.15, 99.85, 100.10),  # Retest
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # Should detect pattern (fakeout filter should pass)
        assert result.detected, f"Should pass fakeout filter, got: {result.reason}"

    def test_fakeout_bar3_failure_not_caught_BUG(self, detector):
        """
        Scenario 3 (BUG): Breakout at 9:40, bars 9:41-9:42 OK, bar 9:43 closes below OR high.
        Expected bug: Fakeout filter only checks bars 1-2, so bar 3 failure is NOT caught.

        This exposes a gap: fakeout_bars=2 only checks immediate bars after breakout,
        but price could fail on bar 3+ and still pass the fakeout filter.

        OR: high=100, low=99
        Breakout (9:40): 100.30
        Bar 9:41: 100.25 (above ORH) - PASSES fakeout check
        Bar 9:42: 100.20 (above ORH) - PASSES fakeout check
        Bar 9:43: 99.70 (below ORH) - NOT CHECKED by fakeout filter!
        Bar 9:44: Attempts retest entry

        The setup should be invalid (price closed back in OR), but fakeout filter misses it.
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # Opening range
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Post-OR
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.30, 99.50),
            create_bar(base + timedelta(minutes=6), 99.50, 99.70, 99.40, 99.60),
            create_bar(base + timedelta(minutes=7), 99.60, 99.80, 99.50, 99.70),
            create_bar(base + timedelta(minutes=8), 99.70, 99.90, 99.60, 99.80),
            create_bar(base + timedelta(minutes=9), 99.80, 100.00, 99.70, 99.90),
            # Breakout (9:40)
            create_bar(base + timedelta(minutes=10), 99.90, 100.35, 99.85, 100.30),
            # Bar 1 after breakout - OK (above ORH)
            create_bar(base + timedelta(minutes=11), 100.30, 100.40, 100.15, 100.25),
            # Bar 2 after breakout - OK (above ORH)
            create_bar(base + timedelta(minutes=12), 100.25, 100.30, 100.10, 100.20),
            # Bar 3 - CLOSES BELOW OR HIGH (fakeout filter doesn't check this!)
            create_bar(base + timedelta(minutes=13), 100.20, 100.25, 99.65, 99.70),
            # Retest attempt - price went back above ORH
            create_bar(base + timedelta(minutes=14), 99.70, 100.15, 99.65, 100.10),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # BUG: Current implementation PASSES this because fakeout_bars=2 only checks bars 1-2
        # This test documents the bug - it SHOULD reject but currently accepts
        # If this test fails (i.e., result.detected is False), the bug was fixed!

        print(f"Result: detected={result.detected}, reason={result.reason}")

        # This is documenting expected behavior - adjust based on whether bug is fixed
        if result.detected:
            pytest.fail(
                "BUG CONFIRMED: Fakeout filter missed bar 3 closing below OR high. "
                "Filter only checks fakeout_bars (2) immediately after breakout."
            )


# =============================================================================
# CHOPPY FILTER TESTS
# =============================================================================

class TestChoppyFilter:
    """
    Choppy filter rejects setups where pre-breakout bars show consolidation.

    Current implementation (lines 239-277):
    Triggers if 2 of 3 conditions met:
    (a) body_pct < 40% for >= 60% of bars
    (b) alternating colors >= 50% of bars
    (c) net drift from first to last close < 20% of OR range

    Potential bugs:
    1. Requires >= 3 bars, otherwise passes silently
    2. Drift measures first→last, not total path (up then down = low drift)
    3. lookback starts at max(0, breakout_idx - lookback), may include OR bars
    """

    @pytest.fixture
    def detector(self):
        """Create detector with choppy_filter enabled."""
        det = OpeningRangeRetest()
        det.config["choppy_filter"] = True
        det.config["choppy_lookback_bars"] = 6
        det.config["fakeout_filter"] = False
        det.config["require_clean_breakout_bar"] = False
        det.config["confirmation_filter"] = False
        det.config["trend_alignment"] = False  # Disable trend check
        return det

    def test_choppy_clear_trend_passes(self, detector):
        """
        Scenario 1: Pre-breakout bars show clear uptrend with large bodies.
        Expected: Should NOT trigger choppy filter.

        6 bars before breakout:
        - All green (no alternating)
        - Bodies are 60%+ of range (not small)
        - Drift is 0.50 (50% of OR range)
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        # OR: high=100, low=99, range=1.00
        bars = [
            # Opening range
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Clear uptrend - 6 bars before breakout (9:35-9:40)
            # All green, large bodies, strong drift up
            create_bar(base + timedelta(minutes=5), 99.40, 99.55, 99.35, 99.50),   # Green, body=0.10/0.20=50%
            create_bar(base + timedelta(minutes=6), 99.50, 99.70, 99.45, 99.65),   # Green, body=0.15/0.25=60%
            create_bar(base + timedelta(minutes=7), 99.65, 99.85, 99.60, 99.80),   # Green, body=0.15/0.25=60%
            create_bar(base + timedelta(minutes=8), 99.80, 100.00, 99.75, 99.95),  # Green, body=0.15/0.25=60%
            create_bar(base + timedelta(minutes=9), 99.95, 100.10, 99.90, 100.05), # Green, body=0.10/0.20=50%
            create_bar(base + timedelta(minutes=10), 100.05, 100.20, 100.00, 100.15), # Green
            # Breakout (9:41)
            create_bar(base + timedelta(minutes=11), 100.15, 100.40, 100.10, 100.35),
            # Post-breakout holds
            create_bar(base + timedelta(minutes=12), 100.35, 100.45, 100.25, 100.30),
            create_bar(base + timedelta(minutes=13), 100.30, 100.35, 100.20, 100.25),
            # Retest
            create_bar(base + timedelta(minutes=14), 100.05, 100.20, 99.90, 100.15),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # Should pass choppy filter (clear trend)
        if not result.detected:
            reason = result.reason or ""
            assert "choppy" not in reason.lower(), f"Failed due to choppy filter: {reason}"

    def test_choppy_consolidation_rejected(self, detector):
        """
        Scenario 2: Pre-breakout bars show consolidation.
        Expected: Should trigger choppy filter and reject.

        6 bars before breakout:
        - Alternating colors (red, green, red, green, red, green)
        - Small bodies (< 40% of bar range)
        - Low drift (starts and ends near same price)
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        # OR: high=100, low=99, range=1.00
        bars = [
            # Opening range
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Choppy consolidation - 6 bars (9:35-9:40)
            # Alternating colors, small bodies, low drift
            # Body < 40% means body/range < 0.40
            # Range = 0.20, body needs to be < 0.08
            create_bar(base + timedelta(minutes=5), 99.52, 99.60, 99.45, 99.48),   # Red, body=0.04, range=0.15, 27%
            create_bar(base + timedelta(minutes=6), 99.48, 99.58, 99.42, 99.55),   # Green, body=0.07, range=0.16, 44%
            create_bar(base + timedelta(minutes=7), 99.55, 99.62, 99.48, 99.50),   # Red, body=0.05, range=0.14, 36%
            create_bar(base + timedelta(minutes=8), 99.50, 99.60, 99.45, 99.57),   # Green, body=0.07, range=0.15, 47%
            create_bar(base + timedelta(minutes=9), 99.57, 99.65, 99.50, 99.52),   # Red, body=0.05, range=0.15, 33%
            create_bar(base + timedelta(minutes=10), 99.52, 99.62, 99.48, 99.58),  # Green, body=0.06, range=0.14, 43%
            # Breakout (9:41) - finally breaks out
            create_bar(base + timedelta(minutes=11), 99.58, 100.35, 99.55, 100.30),
            # Post breakout
            create_bar(base + timedelta(minutes=12), 100.30, 100.40, 100.20, 100.25),
            create_bar(base + timedelta(minutes=13), 100.25, 100.30, 100.15, 100.20),
            # Retest
            create_bar(base + timedelta(minutes=14), 100.05, 100.18, 99.90, 100.12),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # Should reject due to choppy filter
        assert not result.detected, f"Should reject choppy pattern, got: {result.reason}"
        assert "choppy" in (result.reason or "").lower(), \
            f"Reason should mention choppy: {result.reason}"

    def test_choppy_less_than_3_bars_passes_BUG(self, detector):
        """
        Scenario 3 (BUG): Breakout happens immediately after OR (only 2 bars before breakout).
        Expected bug: Filter requires >= 3 bars, so it passes silently.

        This means early breakouts (e.g., 9:37) bypass the choppy filter entirely,
        even if those 2 bars are extremely choppy.
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        # OR: high=100, low=99
        bars = [
            # Opening range
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Only 2 choppy bars before breakout (9:35-9:36)
            create_bar(base + timedelta(minutes=5), 99.52, 99.60, 99.45, 99.48),   # Red, choppy
            create_bar(base + timedelta(minutes=6), 99.48, 99.58, 99.42, 99.55),   # Green, choppy
            # Breakout at 9:37 (only 2 bars to check, filter needs >= 3)
            create_bar(base + timedelta(minutes=7), 99.55, 100.35, 99.50, 100.30),
            # Post breakout
            create_bar(base + timedelta(minutes=8), 100.30, 100.40, 100.20, 100.25),
            create_bar(base + timedelta(minutes=9), 100.25, 100.30, 100.15, 100.20),
            # Retest
            create_bar(base + timedelta(minutes=10), 100.05, 100.18, 99.90, 100.12),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # BUG: Choppy filter should catch this but passes due to < 3 bars
        # Document what current behavior is
        reason = result.reason or ""

        if result.detected:
            pytest.fail(
                "BUG CONFIRMED: Choppy filter bypassed because only 2 bars before breakout. "
                "Filter requires >= 3 bars (line 245: if len(pre_breakout) >= 3)."
            )
        else:
            # Rejected for some other reason - still document what happened
            print(f"Rejected but not for choppy: {reason}")


# =============================================================================
# CONFIRMATION FILTER TESTS
# =============================================================================

class TestConfirmationFilter:
    """
    Confirmation filter requires entry bar to show strong rejection.

    Current implementation (lines 379-420):
    For longs, must pass ONE of:
    - Hammer: lower_wick >= 2x body AND upper_wick <= body
    - Strong bullish: body_pct >= 60% AND close > ORH + 10% of OR range

    Potential bugs:
    1. Hammer requires upper_wick <= body, very strict for small bodies
    2. Strong bullish requires close > ORH + 10% range, rejects closes at ORH
    3. Zero-body (doji) edge case: lower_wick >= 0 (true), upper_wick <= 0 (only if no upper wick)
    """

    @pytest.fixture
    def detector(self):
        """Create detector with confirmation_filter enabled."""
        det = OpeningRangeRetest()
        det.config["confirmation_filter"] = True
        det.config["fakeout_filter"] = False
        det.config["choppy_filter"] = False
        det.config["require_clean_breakout_bar"] = False
        det.config["trend_alignment"] = False  # Disable trend check
        return det

    def test_confirmation_hammer_passes(self, detector):
        """
        Scenario 1: Entry bar is a clear hammer (long lower wick, small body, tiny upper wick).
        Expected: Should pass confirmation filter.

        Hammer requirements:
        - lower_wick >= 2x body
        - upper_wick <= body

        Bar: O=100.05, H=100.10, L=99.85, C=100.08
        - Body = 0.03 (100.08 - 100.05)
        - Lower wick = 0.20 (100.05 - 99.85) = 6.7x body ✓
        - Upper wick = 0.02 (100.10 - 100.08) < body 0.03 ✓
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        # OR: high=100, low=99, range=1.00
        bars = [
            # Opening range
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Build up
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),
            create_bar(base + timedelta(minutes=7), 99.70, 99.90, 99.65, 99.85),
            create_bar(base + timedelta(minutes=8), 99.85, 100.00, 99.80, 99.95),
            create_bar(base + timedelta(minutes=9), 99.95, 100.10, 99.90, 100.05),
            # Breakout
            create_bar(base + timedelta(minutes=10), 100.05, 100.40, 100.00, 100.35),
            # Post breakout holds
            create_bar(base + timedelta(minutes=11), 100.35, 100.45, 100.25, 100.30),
            create_bar(base + timedelta(minutes=12), 100.30, 100.35, 100.20, 100.25),
            # Hammer retest: O=100.05, H=100.10, L=99.85, C=100.08
            # Body=0.03, lower_wick=0.20, upper_wick=0.02
            # lower_wick >= 2*body (0.20 >= 0.06) ✓
            # upper_wick <= body (0.02 <= 0.03) ✓
            create_bar(base + timedelta(minutes=13), 100.05, 100.10, 99.85, 100.08),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # Should pass confirmation filter
        if not result.detected:
            reason = result.reason or ""
            assert "confirmation" not in reason.lower() and "hammer" not in reason.lower(), \
                f"Failed due to confirmation filter: {reason}"

    def test_confirmation_strong_bullish_passes(self, detector):
        """
        Scenario 2: Entry bar is strong bullish (large body, closes well above OR high).
        Expected: Should pass confirmation filter.

        Strong bullish requirements:
        - body_pct >= 60%
        - close > ORH + 10% of OR range = 100 + 0.10 = 100.10

        Bar: O=99.95, H=100.20, L=99.90, C=100.18
        - Range = 0.30, Body = 0.23, body_pct = 77% ✓
        - Close 100.18 > 100.10 ✓
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # Opening range
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Build up
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),
            create_bar(base + timedelta(minutes=7), 99.70, 99.90, 99.65, 99.85),
            create_bar(base + timedelta(minutes=8), 99.85, 100.00, 99.80, 99.95),
            create_bar(base + timedelta(minutes=9), 99.95, 100.10, 99.90, 100.05),
            # Breakout
            create_bar(base + timedelta(minutes=10), 100.05, 100.40, 100.00, 100.35),
            # Post breakout
            create_bar(base + timedelta(minutes=11), 100.35, 100.45, 100.25, 100.30),
            create_bar(base + timedelta(minutes=12), 100.30, 100.35, 100.20, 100.25),
            # Strong bullish retest: O=99.95, H=100.20, L=99.90, C=100.18
            # Body = 0.23, Range = 0.30, body_pct = 77%
            # Close 100.18 > 100.10 (ORH + 10% range)
            create_bar(base + timedelta(minutes=13), 99.95, 100.20, 99.90, 100.18),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        if not result.detected:
            reason = result.reason or ""
            assert "confirmation" not in reason.lower() and "strong" not in reason.lower(), \
                f"Failed due to confirmation filter: {reason}"

    def test_confirmation_good_candle_at_orh_rejected_BUG(self, detector):
        """
        Scenario 3 (BUG): Entry bar is a strong green candle closing just above OR high.
        Expected bug: Rejected because close is not > ORH + 10% range.

        This is a valid confirmation - a strong green candle that reclaims above OR high.
        But the filter requires close > ORH + 0.10 * range (100.10).

        Bar: O=99.92, H=100.05, L=99.88, C=100.02
        - High = 100.05 > ORH (100) ✓ - passes high check
        - Close = 100.02 > Open = 99.92 ✓ - bullish
        - Body = 0.10, Range = 0.17, body_pct = 59%
        - Close = 100.02, needs > 100.10 for "strong bullish" ✗

        Is this a hammer?
        - lower_wick = 0.04 (99.92 - 99.88)
        - upper_wick = 0.03 (100.05 - 100.02)
        - body = 0.10
        - lower_wick >= 2*body? 0.04 >= 0.20? NO
        - upper_wick <= body? 0.03 <= 0.10? YES (but lower_wick fails)
        - Not a hammer

        So this perfectly valid candle fails both confirmation checks!
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # Opening range - high=100, low=99
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Build up
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),
            create_bar(base + timedelta(minutes=7), 99.70, 99.90, 99.65, 99.85),
            create_bar(base + timedelta(minutes=8), 99.85, 100.00, 99.80, 99.95),
            create_bar(base + timedelta(minutes=9), 99.95, 100.10, 99.90, 100.05),
            # Breakout
            create_bar(base + timedelta(minutes=10), 100.05, 100.40, 100.00, 100.35),
            # Post breakout
            create_bar(base + timedelta(minutes=11), 100.35, 100.45, 100.25, 100.30),
            create_bar(base + timedelta(minutes=12), 100.30, 100.35, 100.20, 100.25),
            # Strong green candle reclaiming ORH: O=99.92, H=100.05, L=99.88, C=100.02
            # High > ORH ✓, Close > Open ✓, but fails confirmation filter
            # Not a hammer (lower_wick too short), not "strong" (close not > ORH+10%)
            create_bar(base + timedelta(minutes=13), 99.92, 100.05, 99.88, 100.02),
        ]

        df = create_bars_df(bars)
        result = detector.detect(df)

        # BUG: This valid candle should arguably pass, but fails confirmation filter
        reason = result.reason or ""

        if result.detected:
            # If it passes, the filter might have other logic we missed
            print(f"Unexpectedly passed: {result.details}")
        else:
            if "confirmation" in reason.lower() or "hammer" in reason.lower() or "strong" in reason.lower() or "weak" in reason.lower():
                pytest.fail(
                    f"BUG CONFIRMED: Valid green candle (59% body, above ORH) rejected. "
                    f"Reason: {reason}. "
                    f"Filter requires either hammer OR close > ORH + 10% range."
                )
            else:
                # Rejected for another reason
                print(f"Rejected for different reason: {reason}")


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
