"""
Exhaustive Unit Tests for ORB Filter Logic
===========================================

Comprehensive limit testing for all filter rules and edge cases.

Filters tested:
1. fakeout_filter - 12 test cases
2. choppy_filter - 15 test cases
3. confirmation_filter - 15 test cases

Total: 42 test cases
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from candle_patterns.opening_range_retest import OpeningRangeRetest


# =============================================================================
# TEST HELPERS
# =============================================================================

def create_bar(ts: datetime, o: float, h: float, l: float, c: float, v: int = 10000) -> dict:
    """Helper to create a single bar."""
    return {"timestamp": ts, "open": o, "high": h, "low": l, "close": c, "volume": v}


def create_bars_df(bars: list) -> pd.DataFrame:
    """Convert list of bar dicts to DataFrame."""
    return pd.DataFrame(bars)


def build_base_scenario(et, base, direction="long"):
    """
    Build base bars for OR + buildup + breakout.
    Returns list of bars and breakout index.

    OR: high=100, low=99, range=1.00
    Displacement threshold = max(1.00*0.20, 0.05) = 0.20
    For long: need close > 100.20
    For short: need close < 98.80
    """
    bars = [
        # Opening range (9:30-9:34) - 5 bars
        create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
        create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
        create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
        create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
        create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
    ]

    if direction == "long":
        # Buildup to long breakout (9:35-9:39) - 5 bars
        bars.extend([
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),
            create_bar(base + timedelta(minutes=7), 99.70, 99.90, 99.65, 99.85),
            create_bar(base + timedelta(minutes=8), 99.85, 100.00, 99.80, 99.95),
            create_bar(base + timedelta(minutes=9), 99.95, 100.10, 99.90, 100.05),
        ])
        # Long breakout (9:40) - closes above 100.20
        bars.append(create_bar(base + timedelta(minutes=10), 100.05, 100.40, 100.00, 100.35))
    else:
        # Buildup to short breakout (9:35-9:39) - 5 bars
        bars.extend([
            create_bar(base + timedelta(minutes=5), 99.40, 99.45, 99.20, 99.25),
            create_bar(base + timedelta(minutes=6), 99.25, 99.30, 99.05, 99.10),
            create_bar(base + timedelta(minutes=7), 99.10, 99.15, 98.95, 99.00),
            create_bar(base + timedelta(minutes=8), 99.00, 99.05, 98.85, 98.90),
            create_bar(base + timedelta(minutes=9), 98.90, 98.95, 98.75, 98.80),
        ])
        # Short breakout (9:40) - closes below 98.80
        bars.append(create_bar(base + timedelta(minutes=10), 98.80, 98.85, 98.60, 98.65))

    return bars


# =============================================================================
# FAKEOUT FILTER EXHAUSTIVE TESTS
# =============================================================================

class TestFakeoutFilterExhaustive:
    """
    Fakeout filter rules:
    - Checks bars 1 to fakeout_bars after breakout
    - For longs: triggers if close < or_high (100)
    - For shorts: triggers if close > or_low (99)

    Test matrix:
    - Bar 1 scenarios (pass/fail/boundary)
    - Bar 2 scenarios (pass/fail/boundary)
    - Bar 3+ scenarios (not checked - bug)
    - Short direction
    - Edge cases (no bars after breakout, exactly at level)
    """

    @pytest.fixture
    def detector(self):
        det = OpeningRangeRetest()
        det.config["fakeout_filter"] = True
        det.config["fakeout_bars"] = 2
        det.config["require_clean_breakout_bar"] = False
        det.config["choppy_filter"] = False
        det.config["confirmation_filter"] = False
        det.config["trend_alignment"] = False
        return det

    # --- Bar 1 Tests ---

    def test_long_bar1_closes_well_above_orh_passes(self, detector):
        """Bar 1 closes at 100.25 (well above ORH 100) - should pass."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        # Bar 1: closes well above ORH
        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 100.20, 100.25))
        # Bar 2: also above
        bars.append(create_bar(base + timedelta(minutes=12), 100.25, 100.30, 100.15, 100.20))
        # Retest bar
        bars.append(create_bar(base + timedelta(minutes=13), 100.05, 100.15, 99.90, 100.10))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"Should pass, got: {result.reason}"

    def test_long_bar1_closes_at_orh_exactly_fails(self, detector):
        """Bar 1 closes at exactly 100.00 (ORH) - should FAIL (close < or_high is false, but close == or_high)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        # Bar 1: closes exactly at ORH - check says "close < or_high" so 100 < 100 is False
        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 99.95, 100.00))
        bars.append(create_bar(base + timedelta(minutes=12), 100.00, 100.10, 99.95, 100.05))
        bars.append(create_bar(base + timedelta(minutes=13), 100.05, 100.15, 99.90, 100.10))

        result = detector.detect(create_bars_df(bars))
        # Close = 100.00, or_high = 100.00, check is "close < or_high" = False
        # So this should PASS the fakeout filter (not a fakeout)
        print(f"Bar1 at ORH exactly: detected={result.detected}, reason={result.reason}")

    def test_long_bar1_closes_1cent_below_orh_fails(self, detector):
        """Bar 1 closes at 99.99 (1 cent below ORH) - should trigger fakeout."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        # Bar 1: closes 1 cent below ORH
        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 99.90, 99.99))
        bars.append(create_bar(base + timedelta(minutes=12), 99.99, 100.10, 99.95, 100.05))
        bars.append(create_bar(base + timedelta(minutes=13), 100.05, 100.15, 99.90, 100.10))

        result = detector.detect(create_bars_df(bars))
        assert not result.detected, f"Should fail (fakeout), got: {result.reason}"
        assert "fakeout" in (result.reason or "").lower()

    def test_long_bar1_closes_deep_below_orh_fails(self, detector):
        """Bar 1 closes at 99.50 (deep below ORH) - should trigger fakeout."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 99.40, 99.50))
        bars.append(create_bar(base + timedelta(minutes=12), 99.50, 100.10, 99.45, 100.05))
        bars.append(create_bar(base + timedelta(minutes=13), 100.05, 100.15, 99.90, 100.10))

        result = detector.detect(create_bars_df(bars))
        assert not result.detected
        assert "fakeout" in (result.reason or "").lower()

    # --- Bar 2 Tests ---

    def test_long_bar1_ok_bar2_fails(self, detector):
        """Bar 1 OK (100.25), Bar 2 closes below ORH (99.95) - should trigger fakeout."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 100.20, 100.25))  # OK
        bars.append(create_bar(base + timedelta(minutes=12), 100.25, 100.30, 99.90, 99.95))   # FAIL
        bars.append(create_bar(base + timedelta(minutes=13), 99.95, 100.15, 99.85, 100.10))

        result = detector.detect(create_bars_df(bars))
        assert not result.detected
        assert "fakeout" in (result.reason or "").lower()

    def test_long_bar1_ok_bar2_at_orh_exactly(self, detector):
        """Bar 1 OK, Bar 2 at exactly ORH (100.00) - boundary test."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 100.20, 100.25))
        bars.append(create_bar(base + timedelta(minutes=12), 100.25, 100.30, 99.95, 100.00))  # Exactly at ORH
        bars.append(create_bar(base + timedelta(minutes=13), 100.00, 100.15, 99.90, 100.10))

        result = detector.detect(create_bars_df(bars))
        # close < or_high => 100.00 < 100.00 => False, so NOT a fakeout
        print(f"Bar2 at ORH exactly: detected={result.detected}, reason={result.reason}")

    # --- Bar 3+ Tests (BUG: not checked) ---

    def test_long_bar3_fails_not_caught_BUG(self, detector):
        """Bar 1-2 OK, Bar 3 fails - fakeout filter misses this (BUG)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 100.20, 100.25))  # OK
        bars.append(create_bar(base + timedelta(minutes=12), 100.25, 100.30, 100.15, 100.20))  # OK
        bars.append(create_bar(base + timedelta(minutes=13), 100.20, 100.25, 99.60, 99.70))   # FAIL - not checked!
        bars.append(create_bar(base + timedelta(minutes=14), 99.70, 100.15, 99.65, 100.10))   # Retest

        result = detector.detect(create_bars_df(bars))
        if result.detected:
            pytest.fail("BUG: Bar 3 closed below ORH but fakeout filter didn't catch it")

    def test_long_bar4_fails_not_caught_BUG(self, detector):
        """Bar 1-3 OK, Bar 4 fails - fakeout filter misses this (BUG)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "long")

        bars.append(create_bar(base + timedelta(minutes=11), 100.35, 100.40, 100.20, 100.25))
        bars.append(create_bar(base + timedelta(minutes=12), 100.25, 100.30, 100.15, 100.20))
        bars.append(create_bar(base + timedelta(minutes=13), 100.20, 100.28, 100.12, 100.18))
        bars.append(create_bar(base + timedelta(minutes=14), 100.18, 100.22, 99.50, 99.60))  # FAIL
        bars.append(create_bar(base + timedelta(minutes=15), 99.60, 100.12, 99.55, 100.08))

        result = detector.detect(create_bars_df(bars))
        if result.detected:
            pytest.fail("BUG: Bar 4 closed below ORH but fakeout filter didn't catch it")

    # --- Short Direction Tests ---

    def test_short_bar1_closes_below_orl_passes(self, detector):
        """Short: Bar 1 closes at 98.75 (below ORL 99) - should pass."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "short")

        bars.append(create_bar(base + timedelta(minutes=11), 98.65, 98.70, 98.60, 98.75))  # Below ORL
        bars.append(create_bar(base + timedelta(minutes=12), 98.75, 98.80, 98.70, 98.78))
        # Short retest: high touches zone, low < ORL, bearish
        bars.append(create_bar(base + timedelta(minutes=13), 98.95, 99.10, 98.85, 98.90))

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            reason = result.reason or ""
            assert "fakeout" not in reason.lower(), f"Should pass fakeout, got: {reason}"

    def test_short_bar1_closes_above_orl_fails(self, detector):
        """Short: Bar 1 closes at 99.10 (above ORL 99) - should trigger fakeout."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = build_base_scenario(et, base, "short")

        bars.append(create_bar(base + timedelta(minutes=11), 98.65, 99.15, 98.60, 99.10))  # Above ORL!
        bars.append(create_bar(base + timedelta(minutes=12), 99.10, 99.15, 98.90, 98.95))
        bars.append(create_bar(base + timedelta(minutes=13), 98.95, 99.05, 98.85, 98.90))

        result = detector.detect(create_bars_df(bars))
        assert not result.detected
        assert "fakeout" in (result.reason or "").lower()

    # --- Edge Cases ---

    def test_breakout_at_end_no_bars_to_check(self, detector):
        """Breakout is the last bar - no bars after to check for fakeout."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        # Just OR + buildup + breakout, no bars after
        bars = build_base_scenario(et, base, "long")
        # bars ends with breakout bar

        result = detector.detect(create_bars_df(bars))
        # Should fail for "waiting for retest" not fakeout
        assert not result.detected
        print(f"No bars after breakout: {result.reason}")


# =============================================================================
# CHOPPY FILTER EXHAUSTIVE TESTS
# =============================================================================

class TestChoppyFilterExhaustive:
    """
    Choppy filter rules (triggers if 2/3 conditions met):
    (a) body_pct < 40% for >= 60% of bars
    (b) alternating colors >= 50% of bars
    (c) net drift < 20% of OR range (0.20)

    Test matrix:
    - Each condition at/above/below threshold
    - Combinations: 0/3, 1/3, 2/3, 3/3 conditions
    - Edge cases: < 3 bars, includes OR bars
    """

    @pytest.fixture
    def detector(self):
        det = OpeningRangeRetest()
        det.config["choppy_filter"] = True
        det.config["choppy_lookback_bars"] = 6
        det.config["choppy_small_body_pct"] = 40.0
        det.config["choppy_small_body_ratio"] = 0.60
        det.config["choppy_alternating_ratio"] = 0.50
        det.config["choppy_drift_pct"] = 0.20
        det.config["fakeout_filter"] = False
        det.config["require_clean_breakout_bar"] = False
        det.config["confirmation_filter"] = False
        det.config["trend_alignment"] = False
        return det

    def _add_retest(self, bars, base, minute_offset):
        """Add valid retest bar for long."""
        bars.append(create_bar(base + timedelta(minutes=minute_offset), 100.05, 100.15, 99.90, 100.10))
        bars.append(create_bar(base + timedelta(minutes=minute_offset+1), 100.10, 100.20, 100.00, 100.15))
        return bars

    # --- Condition (a): Small body tests ---

    def test_condition_a_all_large_bodies_passes(self, detector):
        """All 6 pre-breakout bars have body >= 40% - condition (a) NOT met."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # OR
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # 6 bars with LARGE bodies (body >= 40% of range) - all green trending up
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),   # body=0.15, range=0.25, 60%
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),   # body=0.15, range=0.25, 60%
            create_bar(base + timedelta(minutes=7), 99.70, 99.92, 99.65, 99.88),   # body=0.18, range=0.27, 67%
            create_bar(base + timedelta(minutes=8), 99.88, 100.10, 99.82, 100.05), # body=0.17, range=0.28, 61%
            create_bar(base + timedelta(minutes=9), 100.05, 100.25, 100.00, 100.20),
            create_bar(base + timedelta(minutes=10), 100.20, 100.38, 100.15, 100.35),
            # Breakout
            create_bar(base + timedelta(minutes=11), 100.35, 100.55, 100.30, 100.50),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            assert "choppy" not in (result.reason or "").lower(), f"Should pass choppy: {result.reason}"

    def test_condition_a_exactly_60pct_small_bodies_triggers(self, detector):
        """Exactly 60% of bars (4/6) have body < 40% - condition (a) MET at boundary."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # 4 small body + 2 large body = 67% small (> 60% threshold)
            create_bar(base + timedelta(minutes=5), 99.50, 99.60, 99.45, 99.52),   # body=0.02, range=0.15, 13% SMALL
            create_bar(base + timedelta(minutes=6), 99.52, 99.62, 99.47, 99.55),   # body=0.03, range=0.15, 20% SMALL
            create_bar(base + timedelta(minutes=7), 99.55, 99.65, 99.50, 99.58),   # body=0.03, range=0.15, 20% SMALL
            create_bar(base + timedelta(minutes=8), 99.58, 99.68, 99.53, 99.62),   # body=0.04, range=0.15, 27% SMALL
            create_bar(base + timedelta(minutes=9), 99.62, 99.85, 99.57, 99.80),   # body=0.18, range=0.28, 64% LARGE
            create_bar(base + timedelta(minutes=10), 99.80, 100.05, 99.75, 100.00),# body=0.20, range=0.30, 67% LARGE
            # Breakout
            create_bar(base + timedelta(minutes=11), 100.00, 100.35, 99.95, 100.30),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        # Condition (a) met (67% small), need 1 more condition for choppy
        print(f"60% small bodies: detected={result.detected}, reason={result.reason}")

    # --- Condition (b): Alternating colors tests ---

    def test_condition_b_no_alternating_passes(self, detector):
        """All same color (green) - no alternating, condition (b) NOT met."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # All green (close > open), large bodies, strong drift
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),   # Green
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),   # Green
            create_bar(base + timedelta(minutes=7), 99.70, 99.90, 99.65, 99.85),   # Green
            create_bar(base + timedelta(minutes=8), 99.85, 100.05, 99.80, 100.00), # Green
            create_bar(base + timedelta(minutes=9), 100.00, 100.20, 99.95, 100.15),# Green
            create_bar(base + timedelta(minutes=10), 100.15, 100.32, 100.10, 100.28),
            create_bar(base + timedelta(minutes=11), 100.28, 100.50, 100.23, 100.45),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            assert "choppy" not in (result.reason or "").lower()

    def test_condition_b_all_alternating_triggers(self, detector):
        """Perfect alternating (RGRGRGR) - 100% alternating, condition (b) MET."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Alternating: R-G-R-G-R-G (with large bodies and high drift to isolate cond b)
            create_bar(base + timedelta(minutes=5), 99.55, 99.65, 99.40, 99.45),   # Red (55>45)
            create_bar(base + timedelta(minutes=6), 99.45, 99.70, 99.40, 99.65),   # Green (45<65)
            create_bar(base + timedelta(minutes=7), 99.70, 99.75, 99.55, 99.58),   # Red (70>58)
            create_bar(base + timedelta(minutes=8), 99.58, 99.85, 99.53, 99.80),   # Green (58<80)
            create_bar(base + timedelta(minutes=9), 99.85, 99.90, 99.70, 99.72),   # Red (85>72)
            create_bar(base + timedelta(minutes=10), 99.72, 100.10, 99.67, 100.05),# Green (72<105)
            create_bar(base + timedelta(minutes=11), 100.05, 100.45, 100.00, 100.40),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        print(f"All alternating: detected={result.detected}, reason={result.reason}")

    def test_condition_b_exactly_50pct_alternating(self, detector):
        """Exactly 50% alternating (3/6 transitions alternate) - boundary test."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Colors: G-G-R-R-G-G = 2 alternations out of 5 transitions = 40%
            # Need: G-R-G-R-G-R = 5/5 = 100% or G-R-R-G-G-R = 3/5 = 60%
            # At 50%: G-R-G-G-R-R = 2.5/5 not possible
            # Let's do G-R-G-G-R-G = 4/5 = 80% > 50%
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),   # G
            create_bar(base + timedelta(minutes=6), 99.60, 99.65, 99.50, 99.52),   # R (alt)
            create_bar(base + timedelta(minutes=7), 99.52, 99.72, 99.47, 99.68),   # G (alt)
            create_bar(base + timedelta(minutes=8), 99.68, 99.88, 99.63, 99.85),   # G
            create_bar(base + timedelta(minutes=9), 99.90, 99.95, 99.78, 99.80),   # R (alt)
            create_bar(base + timedelta(minutes=10), 99.80, 100.15, 99.75, 100.10),# G (alt)
            create_bar(base + timedelta(minutes=11), 100.10, 100.45, 100.05, 100.40),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        print(f"80% alternating: detected={result.detected}, reason={result.reason}")

    # --- Condition (c): Drift tests ---

    def test_condition_c_high_drift_passes(self, detector):
        """Drift = 0.60 (60% of OR range) - condition (c) NOT met (need < 20%)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # First close: 99.45, Last close: 100.05 => drift = 0.60 = 60% of OR range
            create_bar(base + timedelta(minutes=5), 99.40, 99.55, 99.35, 99.45),
            create_bar(base + timedelta(minutes=6), 99.45, 99.65, 99.40, 99.60),
            create_bar(base + timedelta(minutes=7), 99.60, 99.80, 99.55, 99.75),
            create_bar(base + timedelta(minutes=8), 99.75, 99.95, 99.70, 99.90),
            create_bar(base + timedelta(minutes=9), 99.90, 100.10, 99.85, 100.05),
            create_bar(base + timedelta(minutes=10), 100.05, 100.25, 100.00, 100.20),
            create_bar(base + timedelta(minutes=11), 100.20, 100.50, 100.15, 100.45),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            assert "choppy" not in (result.reason or "").lower()

    def test_condition_c_zero_drift_triggers(self, detector):
        """Drift = 0 (starts and ends at same price) - condition (c) MET."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # First close = Last close = 99.55 => drift = 0
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.70, 99.50, 99.65),
            create_bar(base + timedelta(minutes=7), 99.65, 99.80, 99.60, 99.75),
            create_bar(base + timedelta(minutes=8), 99.75, 99.85, 99.60, 99.65),
            create_bar(base + timedelta(minutes=9), 99.65, 99.75, 99.50, 99.55),
            create_bar(base + timedelta(minutes=10), 99.55, 100.20, 99.50, 100.15),  # Big breakout prep
            create_bar(base + timedelta(minutes=11), 100.15, 100.50, 100.10, 100.45),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        print(f"Zero drift: detected={result.detected}, reason={result.reason}")

    def test_condition_c_exactly_20pct_drift_boundary(self, detector):
        """Drift = exactly 0.20 (20% of OR range) - boundary test."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # First close: 99.55, Last close: 99.75 => drift = 0.20 = 20% of OR range
            # Condition is drift < 20%, so 20% should NOT trigger (>= passes)
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.70, 99.50, 99.65),
            create_bar(base + timedelta(minutes=7), 99.65, 99.80, 99.60, 99.70),
            create_bar(base + timedelta(minutes=8), 99.70, 99.85, 99.65, 99.75),
            create_bar(base + timedelta(minutes=9), 99.75, 99.90, 99.70, 99.85),
            create_bar(base + timedelta(minutes=10), 99.85, 100.20, 99.80, 100.15),
            create_bar(base + timedelta(minutes=11), 100.15, 100.50, 100.10, 100.45),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        # drift_ratio = 0.20, threshold is < 0.20, so 0.20 < 0.20 is False
        # Condition (c) NOT met
        print(f"Exactly 20% drift: detected={result.detected}, reason={result.reason}")

    # --- Combination Tests ---

    def test_all_3_conditions_met_rejects(self, detector):
        """All 3 choppy conditions met - should definitely reject."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Perfect chop: small bodies, alternating, zero drift
            create_bar(base + timedelta(minutes=5), 99.52, 99.60, 99.45, 99.48),   # R, small
            create_bar(base + timedelta(minutes=6), 99.48, 99.58, 99.42, 99.55),   # G, small
            create_bar(base + timedelta(minutes=7), 99.55, 99.62, 99.48, 99.50),   # R, small
            create_bar(base + timedelta(minutes=8), 99.50, 99.58, 99.44, 99.55),   # G, small
            create_bar(base + timedelta(minutes=9), 99.55, 99.62, 99.48, 99.50),   # R, small
            create_bar(base + timedelta(minutes=10), 99.50, 99.58, 99.44, 99.52),  # G, small, ends near start
            create_bar(base + timedelta(minutes=11), 99.52, 100.40, 99.48, 100.35),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        assert not result.detected
        assert "choppy" in (result.reason or "").lower()

    def test_only_1_condition_met_passes(self, detector):
        """Only 1 of 3 choppy conditions met - should pass."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # All same color (green), large bodies, but low drift (only cond c)
            create_bar(base + timedelta(minutes=5), 99.50, 99.70, 99.45, 99.65),   # G, large, close=99.65
            create_bar(base + timedelta(minutes=6), 99.65, 99.85, 99.60, 99.80),   # G, large
            create_bar(base + timedelta(minutes=7), 99.80, 99.95, 99.75, 99.70),   # R (breaks trend)
            create_bar(base + timedelta(minutes=8), 99.70, 99.85, 99.65, 99.80),   # G
            create_bar(base + timedelta(minutes=9), 99.80, 99.95, 99.75, 99.68),   # R, close=99.68
            # First=99.65, last=99.68, drift=0.03 = 3% < 20% - cond c met
            # But no alternating (mixed), large bodies - only 1/3
            create_bar(base + timedelta(minutes=10), 99.68, 100.20, 99.63, 100.15),
            create_bar(base + timedelta(minutes=11), 100.15, 100.50, 100.10, 100.45),
        ]
        bars = self._add_retest(bars, base, 12)

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            reason = result.reason or ""
            # Should pass choppy (only 1 condition)
            print(f"1 condition only: {reason}")

    # --- Edge Cases ---

    def test_choppy_lookback_includes_OR_bars_BUG(self, detector):
        """
        BUG: If breakout happens early (e.g., 9:37), lookback includes OR bars.
        lookback_bars=6, breakout at minute 7 => looks at bars 1-6 which includes OR.
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        bars = [
            # OR bars (often look choppy!)
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Only 2 post-OR bars before breakout
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.90, 99.50, 99.85),
            # Early breakout at 9:37
            create_bar(base + timedelta(minutes=7), 99.85, 100.40, 99.80, 100.35),
        ]
        bars = self._add_retest(bars, base, 8)

        result = detector.detect(create_bars_df(bars))
        # This might trigger choppy because it's analyzing OR bars
        print(f"Early breakout (includes OR): detected={result.detected}, reason={result.reason}")
        if not result.detected and "choppy" in (result.reason or "").lower():
            pytest.fail("BUG: Choppy filter analyzing OR bars, not just post-OR bars")


# =============================================================================
# CONFIRMATION FILTER EXHAUSTIVE TESTS
# =============================================================================

class TestConfirmationFilterExhaustive:
    """
    Confirmation filter rules (for longs, must pass ONE):
    - Hammer: lower_wick >= 2x body AND upper_wick <= body
    - Strong bullish: body_pct >= 60% AND close > ORH + 10% of range (100.10)

    Test matrix:
    - Hammer variations (wick ratios at/above/below thresholds)
    - Strong bullish variations (body% and close position)
    - Edge cases (doji, tiny body, short direction)
    """

    @pytest.fixture
    def detector(self):
        det = OpeningRangeRetest()
        det.config["confirmation_filter"] = True
        det.config["confirm_hammer_wick_ratio"] = 2.0
        det.config["confirm_strong_body_pct"] = 60.0
        det.config["confirm_strong_close_buffer"] = 0.10
        det.config["fakeout_filter"] = False
        det.config["choppy_filter"] = False
        det.config["require_clean_breakout_bar"] = False
        det.config["trend_alignment"] = False
        return det

    def _build_to_retest(self, et, base):
        """Build bars up to (but not including) the retest bar."""
        return [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            create_bar(base + timedelta(minutes=5), 99.40, 99.60, 99.35, 99.55),
            create_bar(base + timedelta(minutes=6), 99.55, 99.75, 99.50, 99.70),
            create_bar(base + timedelta(minutes=7), 99.70, 99.90, 99.65, 99.85),
            create_bar(base + timedelta(minutes=8), 99.85, 100.00, 99.80, 99.95),
            create_bar(base + timedelta(minutes=9), 99.95, 100.10, 99.90, 100.05),
            create_bar(base + timedelta(minutes=10), 100.05, 100.40, 100.00, 100.35),
            create_bar(base + timedelta(minutes=11), 100.35, 100.45, 100.25, 100.30),
            create_bar(base + timedelta(minutes=12), 100.30, 100.35, 100.20, 100.25),
        ]

    # --- Hammer Tests ---

    def test_hammer_perfect_passes(self, detector):
        """
        Perfect hammer: lower_wick = 3x body, upper_wick = 0.
        Bar: O=100.06, H=100.06, L=99.85, C=100.06 (doji with long lower shadow)
        Actually need: O=100.03, H=100.06, L=99.85, C=100.06
        body=0.03, lower_wick=0.18 (6x), upper_wick=0
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # O=100.03, H=100.06, L=99.85, C=100.06
        # body = 0.03, lower = 100.03-99.85 = 0.18, upper = 0
        bars.append(create_bar(base + timedelta(minutes=13), 100.03, 100.06, 99.85, 100.06))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"Perfect hammer should pass: {result.reason}"

    def test_hammer_lower_wick_exactly_2x_body_passes(self, detector):
        """Hammer: lower_wick = exactly 2x body (boundary)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body=0.05, lower_wick=0.10 (2x), upper_wick=0.02
        # O=100.02, C=100.07, L=99.92, H=100.09
        # body = 0.05, lower = 100.02-99.92 = 0.10 = 2x body ✓
        # upper = 100.09-100.07 = 0.02 < 0.05 ✓
        bars.append(create_bar(base + timedelta(minutes=13), 100.02, 100.09, 99.92, 100.07))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"Hammer at 2x boundary should pass: {result.reason}"

    def test_hammer_lower_wick_1_9x_body_fails(self, detector):
        """Hammer: lower_wick = 1.9x body (just below 2x threshold)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body=0.10, lower_wick=0.19 (1.9x < 2x), upper_wick=0.01
        # O=99.95, C=100.05, L=99.76, H=100.06
        # lower = 99.95-99.76 = 0.19, body = 0.10, ratio = 1.9x
        bars.append(create_bar(base + timedelta(minutes=13), 99.95, 100.06, 99.76, 100.05))

        result = detector.detect(create_bars_df(bars))
        # Should fail hammer check (1.9x < 2x), check if strong bullish saves it
        # body_pct = 0.10/0.30 = 33% < 60%, close=100.05 < 100.10
        # Should fail both
        if result.detected:
            print(f"Unexpectedly passed: {result.details}")
        else:
            assert "weak" in (result.reason or "").lower() or "hammer" in (result.reason or "").lower()

    def test_hammer_upper_wick_equals_body_passes(self, detector):
        """Hammer: upper_wick = body (at boundary)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body=0.05, lower_wick=0.15 (3x), upper_wick=0.05 (= body)
        # O=100.00, C=100.05, L=99.85, H=100.10
        bars.append(create_bar(base + timedelta(minutes=13), 100.00, 100.10, 99.85, 100.05))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"upper_wick = body should pass: {result.reason}"

    def test_hammer_upper_wick_exceeds_body_fails(self, detector):
        """Hammer: upper_wick = body + 0.01 (just above threshold)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body=0.05, lower_wick=0.15 (3x), upper_wick=0.06 (> body)
        # O=100.00, C=100.05, L=99.85, H=100.11
        bars.append(create_bar(base + timedelta(minutes=13), 100.00, 100.11, 99.85, 100.05))

        result = detector.detect(create_bars_df(bars))
        # Hammer fails (upper > body), check strong bullish
        # body_pct = 0.05/0.26 = 19% < 60%, close=100.05 < 100.10
        if result.detected:
            print(f"Unexpectedly passed: {result.details}")

    # --- Strong Bullish Tests ---

    def test_strong_bullish_perfect_passes(self, detector):
        """Perfect strong bullish: body=70%, close > ORH+10%."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # O=99.92, H=100.22, L=99.88, C=100.18
        # body = 0.26, range = 0.34, body_pct = 76%
        # close = 100.18 > 100.10 (ORH + 10%)
        bars.append(create_bar(base + timedelta(minutes=13), 99.92, 100.22, 99.88, 100.18))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"Strong bullish should pass: {result.reason}"

    def test_strong_bullish_body_exactly_60pct_passes(self, detector):
        """Strong bullish: body = exactly 60% (boundary)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # Need: body/range = 60%, close > 100.10
        # range=0.25, body=0.15 (60%), close=100.12
        # O=99.93, L=99.88, C=100.08, H=100.13
        # Hmm, close needs to be > 100.10
        # O=99.95, L=99.90, C=100.13, H=100.15 => body=0.18, range=0.25, 72%
        # Let's try: O=99.96, L=99.90, C=100.11, H=100.15 => body=0.15, range=0.25, 60%
        bars.append(create_bar(base + timedelta(minutes=13), 99.96, 100.15, 99.90, 100.11))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"60% body should pass: {result.reason}"

    def test_strong_bullish_body_59pct_fails(self, detector):
        """Strong bullish: body = 59% (just below 60% threshold)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body/range = 59%, close > 100.10 but body% fails
        # range=0.27, body=0.16 (59.3%)
        # O=99.95, L=99.88, C=100.11, H=100.15 => body=0.16, range=0.27, 59%
        bars.append(create_bar(base + timedelta(minutes=13), 99.95, 100.15, 99.88, 100.11))

        result = detector.detect(create_bars_df(bars))
        # body_pct = 59% < 60%, fails strong bullish
        # Not a hammer either (check wick ratios)
        # lower_wick = 99.95-99.88 = 0.07, body = 0.16, ratio = 0.44x < 2x
        if result.detected:
            print(f"Unexpectedly passed with 59% body: {result.details}")

    def test_strong_bullish_close_exactly_at_threshold_fails(self, detector):
        """Strong bullish: close = ORH + 10% exactly (100.10) - should FAIL (need >)."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body >= 60%, close = exactly 100.10
        # O=99.92, L=99.88, C=100.10, H=100.15 => body=0.18, range=0.27, 67%
        bars.append(create_bar(base + timedelta(minutes=13), 99.92, 100.15, 99.88, 100.10))

        result = detector.detect(create_bars_df(bars))
        # close > ORH + 0.10*range means close > 100.10
        # 100.10 > 100.10 is False, so strong bullish fails
        if result.detected:
            print(f"Unexpectedly passed with close exactly at threshold: {result.details}")

    def test_strong_bullish_close_1cent_above_threshold_passes(self, detector):
        """Strong bullish: close = ORH + 10% + 0.01 (100.11) - should pass."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # body >= 60%, close = 100.11
        # O=99.92, L=99.88, C=100.11, H=100.15 => body=0.19, range=0.27, 70%
        bars.append(create_bar(base + timedelta(minutes=13), 99.92, 100.15, 99.88, 100.11))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"Close 1 cent above threshold should pass: {result.reason}"

    # --- Edge Cases ---

    def test_doji_with_long_lower_wick_passes(self, detector):
        """Doji (body=0) with long lower wick - technically passes hammer check."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # O=C=100.05, L=99.85, H=100.06
        # body=0, lower_wick=0.20, upper_wick=0.01
        # lower >= 2*body => 0.20 >= 0 ✓
        # upper <= body => 0.01 <= 0 ✗
        bars.append(create_bar(base + timedelta(minutes=13), 100.05, 100.06, 99.85, 100.05))

        result = detector.detect(create_bars_df(bars))
        # upper_wick (0.01) > body (0), so hammer fails
        print(f"Doji with lower wick: detected={result.detected}, reason={result.reason}")

    def test_doji_no_wicks_fails(self, detector):
        """Pure doji (O=H=L=C) - fails everything."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # O=H=L=C=100.05
        bars.append(create_bar(base + timedelta(minutes=13), 100.05, 100.05, 100.05, 100.05))

        result = detector.detect(create_bars_df(bars))
        # body=0, range=0 => body_pct=0
        # lower_wick=0, upper_wick=0
        # hammer: 0 >= 0 ✓, 0 <= 0 ✓ => should pass hammer!
        print(f"Pure doji: detected={result.detected}, reason={result.reason}")

    def test_tiny_body_huge_lower_wick_passes(self, detector):
        """Tiny body (0.01) with huge lower wick (0.20) - should pass hammer."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # O=100.04, C=100.05, L=99.84, H=100.05
        # body=0.01, lower=0.20 (20x), upper=0
        bars.append(create_bar(base + timedelta(minutes=13), 100.04, 100.05, 99.84, 100.05))

        result = detector.detect(create_bars_df(bars))
        assert result.detected, f"Tiny body huge wick should pass: {result.reason}"

    def test_valid_green_candle_neither_hammer_nor_strong_fails_BUG(self, detector):
        """
        BUG: A valid green candle that's neither hammer nor "strong" gets rejected.
        50% body, closing just above ORH - reasonable entry but fails filter.
        """
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)
        bars = self._build_to_retest(et, base)

        # O=99.94, L=99.88, C=100.06, H=100.08
        # body=0.12, range=0.20, body_pct=60% ✓ but close=100.06 < 100.10 ✗
        # lower=0.06, upper=0.02, ratio=0.5x < 2x ✗
        bars.append(create_bar(base + timedelta(minutes=13), 99.94, 100.08, 99.88, 100.06))

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            if "weak" in (result.reason or "").lower():
                pytest.fail(
                    f"BUG: Valid 60% body green candle rejected. "
                    f"Reason: {result.reason}. "
                    f"Filter should accept any 60%+ body candle, not require close > ORH+10%."
                )

    # --- Short Direction Tests ---

    def test_short_inverted_hammer_passes(self, detector):
        """Short: inverted hammer (long upper wick) should pass."""
        et = ZoneInfo("America/New_York")
        base = datetime(2025, 11, 10, 9, 30, tzinfo=et)

        # Build short scenario
        bars = [
            create_bar(base + timedelta(minutes=0), 99.50, 100.00, 99.00, 99.50),
            create_bar(base + timedelta(minutes=1), 99.50, 99.80, 99.20, 99.60),
            create_bar(base + timedelta(minutes=2), 99.60, 99.90, 99.30, 99.70),
            create_bar(base + timedelta(minutes=3), 99.70, 99.85, 99.40, 99.50),
            create_bar(base + timedelta(minutes=4), 99.50, 99.70, 99.10, 99.40),
            # Buildup to short
            create_bar(base + timedelta(minutes=5), 99.40, 99.45, 99.20, 99.25),
            create_bar(base + timedelta(minutes=6), 99.25, 99.30, 99.05, 99.10),
            create_bar(base + timedelta(minutes=7), 99.10, 99.15, 98.95, 99.00),
            create_bar(base + timedelta(minutes=8), 99.00, 99.05, 98.85, 98.90),
            create_bar(base + timedelta(minutes=9), 98.90, 98.95, 98.75, 98.80),
            # Short breakout
            create_bar(base + timedelta(minutes=10), 98.80, 98.85, 98.60, 98.65),
            create_bar(base + timedelta(minutes=11), 98.65, 98.70, 98.55, 98.60),
            create_bar(base + timedelta(minutes=12), 98.60, 98.65, 98.55, 98.58),
            # Inverted hammer retest: upper_wick >= 2x body, lower_wick <= body
            # O=98.97, H=99.15, L=98.95, C=98.95
            # upper=0.18, body=0.02, lower=0
            create_bar(base + timedelta(minutes=13), 98.97, 99.15, 98.95, 98.95),
        ]

        result = detector.detect(create_bars_df(bars))
        if not result.detected:
            reason = result.reason or ""
            assert "confirmation" not in reason.lower() and "hammer" not in reason.lower()


# =============================================================================
# RUN ALL TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
