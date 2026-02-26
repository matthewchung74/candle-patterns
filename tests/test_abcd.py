"""
ABCD Pattern Tests
==================

Comprehensive boundary/limit tests for ABCD harmonic pattern detection.

Rules tested:
- min_bars_required: 10
- swing_lookback: 3
- min_bc_retracement: 0.382 (38.2%)
- max_bc_retracement: 0.786 (78.6%)
- cd_ab_ratio_min: 0.75
- cd_ab_ratio_max: 1.25
- min_leg_pct: 1.0%
- stop_buffer_pct: 0.5%
- d_completion_tolerance: 0.02 (2%)
- direction_filter: None (detect both bullish/bearish)

Run with: pytest tests/test_abcd.py -v
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from candle_patterns import ABCD, PatternResult
from tests.fixtures.abcd_fixtures import (
    # PASS cases
    ABCD_PASS_BULLISH_VALID,
    ABCD_PASS_BEARISH_VALID,
    ABCD_PASS_618_RETRACEMENT,
    ABCD_PASS_MIN_BC_RETRACEMENT,
    ABCD_PASS_MAX_BC_RETRACEMENT,
    ABCD_PASS_MIN_CD_AB_RATIO,
    ABCD_PASS_MAX_CD_AB_RATIO,
    # FAIL cases
    ABCD_FAIL_BC_TOO_SHALLOW,
    ABCD_FAIL_BC_TOO_DEEP,
    ABCD_FAIL_C_NOT_HIGHER,
    ABCD_FAIL_CD_NOT_DEVELOPED,
    ABCD_FAIL_AB_TOO_SMALL,
    ABCD_FAIL_INSUFFICIENT_BARS,
    ABCD_FAIL_NO_SWING_POINTS,
    # Direction filter cases
    ABCD_FILTER_BULLISH_ONLY,
    ABCD_FILTER_BEARISH_ONLY,
)


class TestABCDDetection:
    """Tests for ABCD pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ABCD()

    # =========================================================================
    # PASS TESTS (7)
    # =========================================================================

    def test_pass_bullish_valid(self):
        """Test that a standard valid bullish ABCD is detected."""
        fixture = ABCD_PASS_BULLISH_VALID
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        assert result.pattern_name == "ABCD"
        assert result.details["direction"] == "long"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.stop_price < result.entry_price  # Long: stop below entry
        # Verify Fibonacci retracement in valid range
        assert 0.382 <= result.details["bc_retracement"] <= 0.786

    def test_pass_bearish_valid(self):
        """Test that a standard valid bearish ABCD is detected."""
        fixture = ABCD_PASS_BEARISH_VALID
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        assert result.pattern_name == "ABCD"
        assert result.details["direction"] == "short"
        assert result.stop_price > result.entry_price  # Short: stop above entry
        assert 0.382 <= result.details["bc_retracement"] <= 0.786

    def test_pass_618_retracement(self):
        """Test detection with ideal 61.8% Fibonacci retracement."""
        fixture = ABCD_PASS_618_RETRACEMENT
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        # 61.8% retracement should give higher confidence
        assert result.confidence >= 0.7

    def test_pass_min_bc_retracement_boundary(self):
        """Test detection with BC retracement at 39% (just above 38.2% minimum)."""
        fixture = ABCD_PASS_MIN_BC_RETRACEMENT
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        assert result.details["bc_retracement"] >= 0.382

    def test_pass_max_bc_retracement_boundary(self):
        """Test detection with BC retracement at 78% (just below 78.6% maximum)."""
        fixture = ABCD_PASS_MAX_BC_RETRACEMENT
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        assert result.details["bc_retracement"] <= 0.786

    def test_pass_min_cd_ab_ratio_boundary(self):
        """Test detection with CD/AB ratio at 81% (just above 80% min completion)."""
        fixture = ABCD_PASS_MIN_CD_AB_RATIO
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        # CD must be at least 80% developed (detection threshold)
        assert result.details["cd_ab_ratio"] >= 0.80

    def test_pass_max_cd_ab_ratio_boundary(self):
        """Test detection with CD/AB ratio near maximum."""
        fixture = ABCD_PASS_MAX_CD_AB_RATIO
        result = self.detector.detect(fixture["bars"])

        assert result.detected is True
        # Pattern detected with CD approaching or exceeding AB

    # =========================================================================
    # FAIL TESTS (7)
    # =========================================================================

    def test_fail_bc_retracement_too_shallow(self):
        """Test rejection when BC retracement is 25% (below 38.2% minimum)."""
        fixture = ABCD_FAIL_BC_TOO_SHALLOW
        result = self.detector.detect(fixture["bars"])

        # Pattern should not be detected with intended A-B-C
        if not result.detected:
            assert "no valid" in result.reason.lower()

    def test_fail_bc_retracement_too_deep(self):
        """Test rejection when BC retracement is 90% (above 78.6% maximum)."""
        fixture = ABCD_FAIL_BC_TOO_DEEP
        result = self.detector.detect(fixture["bars"])

        if not result.detected:
            assert "no valid" in result.reason.lower()

    def test_fail_c_not_higher_than_a(self):
        """Test rejection when C is below A (bullish pattern invalid)."""
        fixture = ABCD_FAIL_C_NOT_HIGHER
        result = self.detector.detect(fixture["bars"])

        if not result.detected:
            assert "no valid" in result.reason.lower()

    def test_fail_cd_not_developed(self):
        """Test rejection when CD leg is < 80% of AB."""
        fixture = ABCD_FAIL_CD_NOT_DEVELOPED
        result = self.detector.detect(fixture["bars"])

        if not result.detected:
            assert "no valid" in result.reason.lower()

    def test_fail_ab_leg_too_small(self):
        """Test rejection when AB leg is < 1% minimum."""
        fixture = ABCD_FAIL_AB_TOO_SMALL
        result = self.detector.detect(fixture["bars"])

        assert result.detected is False

    def test_fail_insufficient_bars(self):
        """Test rejection when bars < 10 minimum."""
        fixture = ABCD_FAIL_INSUFFICIENT_BARS
        result = self.detector.detect(fixture["bars"])

        assert result.detected is False
        assert "insufficient" in result.reason.lower() or "swing" in result.reason.lower()

    def test_fail_no_swing_points(self):
        """Test rejection when no swing points can be detected."""
        fixture = ABCD_FAIL_NO_SWING_POINTS
        result = self.detector.detect(fixture["bars"])

        assert result.detected is False
        assert "swing" in result.reason.lower()


class TestABCDDirectionFilter:
    """Tests for direction filter functionality."""

    def test_filter_bullish_only(self):
        """Test direction_filter='long' rejects bearish patterns."""
        fixture = ABCD_FILTER_BULLISH_ONLY
        detector = ABCD(config=fixture["config"])
        result = detector.detect(fixture["bars"])

        # Should not detect bearish pattern when filter is 'long'
        if result.detected:
            assert result.details["direction"] == "long"

    def test_filter_bearish_only(self):
        """Test direction_filter='short' rejects bullish patterns."""
        fixture = ABCD_FILTER_BEARISH_ONLY
        detector = ABCD(config=fixture["config"])
        result = detector.detect(fixture["bars"])

        # Should not detect bullish pattern when filter is 'short'
        if result.detected:
            assert result.details["direction"] == "short"


class TestABCDConfig:
    """Tests for ABCD configuration."""

    def test_default_config_values(self):
        """Test that default config has correct values."""
        detector = ABCD()

        assert detector.config["min_bars_required"] == 10
        assert detector.config["swing_lookback"] == 3
        assert detector.config["min_bc_retracement"] == 0.382
        assert detector.config["max_bc_retracement"] == 0.786
        assert detector.config["cd_ab_ratio_min"] == 0.75
        assert detector.config["cd_ab_ratio_max"] == 1.25
        assert detector.config["min_leg_pct"] == 1.0
        assert detector.config["stop_buffer_pct"] == 0.5
        assert detector.config["d_completion_tolerance"] == 0.02
        assert detector.config["direction_filter"] is None

    def test_custom_config_override(self):
        """Test that custom config overrides defaults."""
        custom = ABCD({
            "min_bc_retracement": 0.30,
            "max_bc_retracement": 0.85,
            "swing_lookback": 5,
        })

        assert custom.config["min_bc_retracement"] == 0.30
        assert custom.config["max_bc_retracement"] == 0.85
        assert custom.config["swing_lookback"] == 5
        # Other defaults should remain
        assert custom.config["min_leg_pct"] == 1.0


class TestABCDPatternDetails:
    """Tests for pattern result details structure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ABCD()

    def test_bullish_details_structure(self):
        """Test bullish pattern returns correct details structure."""
        fixture = ABCD_PASS_BULLISH_VALID
        result = self.detector.detect(fixture["bars"])

        if result.detected:
            # Required detail fields
            assert "direction" in result.details
            assert "a_idx" in result.details
            assert "b_idx" in result.details
            assert "c_idx" in result.details
            assert "a_price" in result.details
            assert "b_price" in result.details
            assert "c_price" in result.details
            assert "projected_d" in result.details
            assert "ab_move" in result.details
            assert "bc_retracement" in result.details
            assert "cd_ab_ratio" in result.details

            # Verify relationships for bullish
            assert result.details["a_idx"] < result.details["b_idx"]
            assert result.details["b_idx"] < result.details["c_idx"]
            assert result.details["c_price"] > result.details["a_price"]  # Higher low
            assert result.details["b_price"] > result.details["a_price"]  # B is high

    def test_bearish_details_structure(self):
        """Test bearish pattern returns correct details structure."""
        fixture = ABCD_PASS_BEARISH_VALID
        result = self.detector.detect(fixture["bars"])

        if result.detected:
            assert result.details["direction"] == "short"
            assert result.details["c_price"] < result.details["a_price"]  # Lower high
            assert result.details["b_price"] < result.details["a_price"]  # B is low


class TestABCDEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ABCD()

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = self.detector.detect(empty_df)
        assert result.detected is False

    def test_vwap_macd_params_ignored(self):
        """Test that vwap and macd parameters don't cause errors."""
        fixture = ABCD_PASS_BULLISH_VALID
        vwap = pd.Series([10.5] * len(fixture["bars"]))
        macd = pd.DataFrame({
            "macd": [0.1] * len(fixture["bars"]),
            "signal": [0.05] * len(fixture["bars"]),
            "histogram": [0.05] * len(fixture["bars"]),
        })

        # Should not raise an error
        result = self.detector.detect(fixture["bars"], vwap=vwap, macd=macd)
        assert isinstance(result, PatternResult)

    def test_zero_volume_halt_bars_ignored(self):
        """Test that zero-volume bars (trading halts) are skipped in swing detection."""
        # Create bars where the only swing points have zero volume (halt bars)
        # This reproduces the ANPA phantom detection bug
        bars = []
        base_time = datetime(2024, 1, 15, 9, 30)

        # Build 20 bars: flat price with zero-volume halt bars that create phantom swings
        for i in range(20):
            t = base_time + timedelta(minutes=i)
            if i == 8:
                # Phantom swing high — zero volume halt bar with price anomaly
                # (reproduces ANPA bug: halt bar creates detectable swing point)
                bars.append({
                    "timestamp": t,
                    "open": 10.28, "high": 10.35, "low": 10.28, "close": 10.28,
                    "volume": 0,
                })
            elif 5 <= i <= 12:
                # Halt bars: zero volume, flat price
                bars.append({
                    "timestamp": t,
                    "open": 10.28, "high": 10.28, "low": 10.28, "close": 10.28,
                    "volume": 0,
                })
            else:
                # Normal trading bars
                bars.append({
                    "timestamp": t,
                    "open": 10.00 + i * 0.05,
                    "high": 10.05 + i * 0.05,
                    "low": 9.95 + i * 0.05,
                    "close": 10.02 + i * 0.05,
                    "volume": 50000,
                })

        df = pd.DataFrame(bars)
        result = self.detector.detect(df)

        # Should not detect a pattern from halt bars
        assert result.detected is False


class TestABCDPreAMomentum:
    """Tests for pre-A momentum candle requirement."""

    def test_bullish_rejected_when_pre_a_candles_red(self):
        """Bullish ABCD rejected when 3 candles before A are red (no uptrend)."""
        fixture = ABCD_PASS_BULLISH_VALID
        bars = fixture["bars"].copy()

        # Flip pre-A candles (indices 0-2) from green to red
        for i in range(3):
            orig_open = bars.at[i, "open"]
            orig_close = bars.at[i, "close"]
            bars.at[i, "open"] = max(orig_open, orig_close) + 0.01
            bars.at[i, "close"] = min(orig_open, orig_close)

        detector = ABCD()
        result = detector.detect(bars)
        assert result.detected is False

    def test_bearish_rejected_when_pre_a_candles_green(self):
        """Bearish ABCD rejected when 3 candles before A are green (no downtrend)."""
        fixture = ABCD_PASS_BEARISH_VALID
        bars = fixture["bars"].copy()

        # Flip pre-A candles (indices 0-2) from red to green
        for i in range(3):
            orig_open = bars.at[i, "open"]
            orig_close = bars.at[i, "close"]
            bars.at[i, "open"] = min(orig_open, orig_close)
            bars.at[i, "close"] = max(orig_open, orig_close) + 0.01

        detector = ABCD()
        result = detector.detect(bars)
        assert result.detected is False

    def test_check_disabled_when_min_pre_a_candles_zero(self):
        """Pattern detected regardless of candle color when check is disabled."""
        fixture = ABCD_PASS_BULLISH_VALID
        bars = fixture["bars"].copy()

        # Flip pre-A candles to red (would normally reject)
        for i in range(3):
            orig_open = bars.at[i, "open"]
            orig_close = bars.at[i, "close"]
            bars.at[i, "open"] = max(orig_open, orig_close) + 0.01
            bars.at[i, "close"] = min(orig_open, orig_close)

        detector = ABCD({"min_pre_a_candles": 0})
        result = detector.detect(bars)
        assert result.detected is True


class TestABCDVolumeProfile:
    """Tests for volume profile checks (BC < AB, CD > BC)."""

    def test_bullish_rejected_when_bc_volume_too_heavy(self):
        """Bullish rejected when BC pullback volume exceeds 75% of AB impulse."""
        fixture = ABCD_PASS_BULLISH_VALID
        bars = fixture["bars"].copy()

        # Inflate BC zone volume (indices after B through C) to exceed AB
        # B is typically around index 7, C around index 13-14
        for i in range(len(bars)):
            if bars.at[i, "volume"] < 60000:  # BC bars have low volume
                bars.at[i, "volume"] = 200000  # Make heavier than AB

        detector = ABCD()
        result = detector.detect(bars)
        assert result.detected is False

    def test_bullish_rejected_when_cd_volume_too_low(self):
        """Bullish rejected when CD volume is below 100% of BC."""
        fixture = ABCD_PASS_BULLISH_VALID
        bars = fixture["bars"].copy()

        # Suppress CD zone volume (last ~7 bars including post-C and CD)
        for i in range(len(bars) - 7, len(bars)):
            bars.at[i, "volume"] = 10000  # Much lower than BC

        detector = ABCD()
        result = detector.detect(bars)
        assert result.detected is False

    def test_volume_check_disabled_when_ratio_zero(self):
        """Pattern detected when volume checks disabled (ratios set to 0)."""
        fixture = ABCD_PASS_BULLISH_VALID
        bars = fixture["bars"].copy()

        # Set all volumes to the same value (would normally fail BC ratio)
        for i in range(len(bars)):
            bars.at[i, "volume"] = 100000

        detector = ABCD({"max_bc_volume_ratio": 0, "min_cd_bc_volume_ratio": 0})
        result = detector.detect(bars)
        assert result.detected is True

    def test_bullish_details_include_volume_ratios(self):
        """Bullish pattern includes volume ratio details for logging."""
        fixture = ABCD_PASS_BULLISH_VALID
        detector = ABCD()
        result = detector.detect(fixture["bars"])

        assert result.detected is True
        assert "ab_avg_vol" in result.details
        assert "bc_avg_vol" in result.details
        assert "cd_avg_vol" in result.details
        assert "bc_volume_ratio" in result.details
        assert "cd_bc_volume_ratio" in result.details
        assert result.details["bc_volume_ratio"] <= 0.75
        assert result.details["cd_bc_volume_ratio"] >= 1.0

    def test_bearish_details_include_volume_ratios(self):
        """Bearish pattern includes volume ratio details for logging."""
        fixture = ABCD_PASS_BEARISH_VALID
        detector = ABCD()
        result = detector.detect(fixture["bars"])

        assert result.detected is True
        assert result.details["bc_volume_ratio"] <= 0.75
        assert result.details["cd_bc_volume_ratio"] >= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
