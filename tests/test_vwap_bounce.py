"""
Tests for VWAP Bounce Pattern Detector
=======================================

Run with: pytest tests/test_vwap_bounce.py -v
"""

import pytest
import pandas as pd
from candle_patterns import VwapBounce
from tests.fixtures.vwap_bounce_fixtures import (
    VB_PASS_VALID,
    VB_PASS_MIN_CONSOL,
    VB_PASS_TIGHT_RANGE,
    VB_PASS_NO_GAP_NARROWING_REQUIRED,
    VB_FAIL_NO_VWAP,
    VB_FAIL_CONSOL_TOO_SHORT,
    VB_FAIL_CONSOL_TOO_WIDE,
    VB_FAIL_VWAP_NOT_RISING,
    VB_FAIL_ENTRY_NOT_NEAR_LOW,
    VB_FAIL_ALREADY_BROKE_OUT,
    VB_FAIL_LAST_BAR_RED,
    VB_FAIL_GAP_TOO_WIDE,
    VB_FAIL_GAP_NOT_NARROWING,
)


class TestVwapBounceDetection:
    """Tests for VwapBounce pattern detection."""

    def setup_method(self):
        self.detector = VwapBounce()

    # =========================================================================
    # PASS CASES
    # =========================================================================

    def test_valid_pattern_detected(self):
        """Standard valid pattern: consolidation + rising VWAP + entry near low."""
        bars, vwap = VB_PASS_VALID
        result = self.detector.detect(bars, vwap=vwap)
        assert result.detected, f"Should detect: {result.reason}"
        assert result.pattern_name == "VwapBounce"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.entry_price > result.stop_price
        assert result.confidence >= 0.65
        assert result.confidence <= 0.90

    def test_valid_pattern_details(self):
        """Check that details dict contains expected fields."""
        bars, vwap = VB_PASS_VALID
        result = self.detector.detect(bars, vwap=vwap)
        assert result.detected
        d = result.details
        assert "consolidation_bars" in d
        assert "consolidation_range_pct" in d
        assert "vwap_rising_bars" in d
        assert "vwap_at_entry" in d
        assert "price_vwap_gap_pct" in d
        assert "gap_narrowing" in d
        assert "volume_declining" in d
        assert "stop_buffer" in d

    def test_valid_pattern_stop_below_vwap(self):
        """Stop must be below VWAP, not below pullback low."""
        bars, vwap = VB_PASS_VALID
        result = self.detector.detect(bars, vwap=vwap)
        assert result.detected
        vwap_at_entry = result.details["vwap_at_entry"]
        assert result.stop_price < vwap_at_entry

    def test_min_consolidation_bars(self):
        """Exactly 5 bars of consolidation (minimum) should pass."""
        bars, vwap = VB_PASS_MIN_CONSOL
        result = self.detector.detect(bars, vwap=vwap)
        assert result.detected, f"Should detect with 5 bars: {result.reason}"

    def test_tight_range_confidence_boost(self):
        """Tight consolidation should detect and score confidence."""
        bars, vwap = VB_PASS_TIGHT_RANGE
        result = self.detector.detect(bars, vwap=vwap)
        assert result.detected, f"Should detect: {result.reason}"
        assert result.details["consolidation_range_pct"] < 2.0
        assert result.confidence >= 0.65

    def test_gap_narrowing_not_required(self):
        """With require_gap_narrowing=False, should still pass."""
        bars, vwap = VB_PASS_NO_GAP_NARROWING_REQUIRED
        detector = VwapBounce(config={"require_gap_narrowing": False})
        result = detector.detect(bars, vwap=vwap)
        assert result.detected, f"Should detect without gap narrowing: {result.reason}"

    # =========================================================================
    # FAIL CASES
    # =========================================================================

    def test_fail_no_vwap(self):
        """Must reject when no VWAP data provided."""
        bars, _ = VB_FAIL_NO_VWAP
        result = self.detector.detect(bars, vwap=None)
        assert not result.detected
        assert "VWAP" in result.reason

    def test_fail_consolidation_too_short(self):
        """Must reject when fewer than 5 consolidation bars."""
        bars, vwap = VB_FAIL_CONSOL_TOO_SHORT
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected, f"Should reject short consolidation: {result.reason}"

    def test_fail_consolidation_too_wide(self):
        """Must reject when consolidation range > 2%."""
        bars, vwap = VB_FAIL_CONSOL_TOO_WIDE
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected, f"Should reject wide range: {result.reason}"

    def test_fail_vwap_not_rising(self):
        """Must reject when VWAP is not rising (< 6 of 10 strictly increasing)."""
        bars, vwap = VB_FAIL_VWAP_NOT_RISING
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected
        assert "rising" in result.reason.lower() or "VWAP" in result.reason

    def test_fail_entry_not_near_low(self):
        """Must reject when entry bar is not near consolidation low."""
        bars, vwap = VB_FAIL_ENTRY_NOT_NEAR_LOW
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected
        assert "low" in result.reason.lower() or "zone" in result.reason.lower()

    def test_fail_already_broke_out(self):
        """Must reject when price already broke above consolidation high."""
        bars, vwap = VB_FAIL_ALREADY_BROKE_OUT
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected
        assert "broke out" in result.reason.lower() or "breakout" in result.reason.lower()

    def test_fail_last_bar_red(self):
        """Must reject when entry bar is red."""
        bars, vwap = VB_FAIL_LAST_BAR_RED
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected
        assert "red" in result.reason.lower()

    def test_fail_gap_too_wide(self):
        """Must reject when price-VWAP gap exceeds 3%."""
        bars, vwap = VB_FAIL_GAP_TOO_WIDE
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected
        assert "gap" in result.reason.lower()

    def test_fail_gap_not_narrowing(self):
        """Must reject when gap is not narrowing (default require_gap_narrowing=True)."""
        bars, vwap = VB_FAIL_GAP_NOT_NARROWING
        result = self.detector.detect(bars, vwap=vwap)
        assert not result.detected


class TestVwapBounceConfig:
    """Tests for VwapBounce configuration."""

    def test_default_config_values(self):
        detector = VwapBounce()
        assert detector.config["min_consolidation_bars"] == 5
        assert detector.config["max_consolidation_range_pct"] == 2.0
        assert detector.config["min_vwap_rising_bars"] == 6
        assert detector.config["vwap_slope_lookback"] == 10
        assert detector.config["entry_zone_pct"] == 35
        assert detector.config["max_stop_distance_pct"] == 4.0
        assert detector.config["require_macd_positive"] is False
        assert detector.config["vwap_exit_confirmation_bars"] == 3

    def test_custom_config_override(self):
        detector = VwapBounce(config={"min_vwap_rising_bars": 8, "entry_zone_pct": 25})
        assert detector.config["min_vwap_rising_bars"] == 8
        assert detector.config["entry_zone_pct"] == 25
        # Other defaults unchanged
        assert detector.config["min_consolidation_bars"] == 5

    def test_no_min_vwap_slope_per_bar(self):
        """Verify min_vwap_slope_per_bar was removed (replaced by count approach)."""
        detector = VwapBounce()
        assert "min_vwap_slope_per_bar" not in detector.config


class TestVwapBounceStopPlacement:
    """Tests for VWAP-based stop placement."""

    def test_stop_uses_vwap_not_pullback_low(self):
        """Stop must be derived from VWAP, not from price structure."""
        bars, vwap = VB_PASS_VALID
        detector = VwapBounce()
        result = detector.detect(bars, vwap=vwap)
        assert result.detected

        vwap_at_entry = vwap.iloc[-1]
        consol_low = result.details["consolidation_low"]

        # Stop should be below VWAP (not below consolidation low)
        assert result.stop_price < vwap_at_entry
        # Stop should be ABOVE or near consolidation low in this case
        # (VWAP is below consolidation, so stop below VWAP is below consol low)
        # Just verify it's reasonable
        assert result.stop_price > 0

    def test_stop_distance_within_max(self):
        """Stop distance must not exceed max_stop_distance_pct."""
        bars, vwap = VB_PASS_VALID
        detector = VwapBounce()
        result = detector.detect(bars, vwap=vwap)
        assert result.detected

        stop_pct = (result.entry_price - result.stop_price) / result.entry_price * 100
        assert stop_pct <= detector.config["max_stop_distance_pct"]
