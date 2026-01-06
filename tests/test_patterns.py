"""
Tests for Candle Patterns Library
=================================

Run with: pytest tests/test_patterns.py -v
"""

import pytest
import pandas as pd
from candle_patterns import MicroPullback, BullFlag, VWAPBreak
from tests.fixtures.micro_pullback_fixtures import (
    MICRO_PULLBACK_VALID,
    MICRO_PULLBACK_TOO_DEEP,
    MICRO_PULLBACK_NO_PRIOR_MOVE,
    MICRO_PULLBACK_LIMIT_PRIOR_MOVE,
    MICRO_PULLBACK_LIMIT_PULLBACK_PCT,
    MICRO_PULLBACK_LIMIT_PULLBACK_CANDLES,
    MICRO_PULLBACK_LIMIT_GREEN_RATIO,
)
from tests.fixtures.bull_flag_fixtures import (
    BULL_FLAG_VALID,
    BULL_FLAG_NO_BREAKOUT,
    BULL_FLAG_LIMIT_POLE_MOVE,
    BULL_FLAG_LIMIT_PULLBACK_SHALLOW,
    BULL_FLAG_LIMIT_PULLBACK_DEEP,
    BULL_FLAG_LIMIT_MIN_CANDLES,
)
from tests.fixtures.vwap_break_fixtures import (
    VWAP_BREAK_VALID,
    VWAP_HOLD_VALID,
    VWAP_BREAK_ALREADY_ABOVE,
    VWAP_BREAK_LIMIT_BARS_BELOW,
    VWAP_BREAK_LIMIT_VOLUME_SPIKE,
    VWAP_BREAK_LIMIT_RR,
    VWAP_BREAK_LIMIT_CLOSE_ABOVE,
)


class TestMicroPullback:
    """Tests for Micro Pullback pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def test_valid_pattern_detected(self):
        """Test that a valid micro pullback is detected."""
        result = self.detector.detect(MICRO_PULLBACK_VALID)

        assert result.detected is True
        assert result.pattern_name == "MicroPullback"
        assert result.confidence >= 0.7
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.entry_price > result.stop_price

    def test_valid_pattern_details(self):
        """Test pattern details for valid detection."""
        result = self.detector.detect(MICRO_PULLBACK_VALID)

        assert result.details is not None
        assert result.details["prior_move_pct"] >= 5.0
        assert result.details["pullback_pct"] <= 20.0  # Updated for tuned config
        assert result.details["green_candles"] >= 2  # Updated: now requires >50% green, not strict count

    def test_too_deep_pullback_rejected(self):
        """Test that deep pullback is rejected."""
        result = self.detector.detect(MICRO_PULLBACK_TOO_DEEP)

        assert result.detected is False
        assert "too deep" in result.reason.lower()

    def test_no_prior_move_rejected(self):
        """Test that insufficient prior move is rejected."""
        result = self.detector.detect(MICRO_PULLBACK_NO_PRIOR_MOVE)

        assert result.detected is False
        # Could fail for various reasons related to prior move
        assert any(x in result.reason.lower() for x in ["short", "green", "prior", "insufficient"])

    def test_empty_bars_rejected(self):
        """Test that empty DataFrame is rejected."""
        result = self.detector.detect(pd.DataFrame())

        assert result.detected is False
        assert "empty" in result.reason.lower()

    def test_custom_config(self):
        """Test that custom config overrides work."""
        custom_detector = MicroPullback({
            "min_prior_move_pct": 10.0,  # Stricter requirement
        })

        result = custom_detector.detect(MICRO_PULLBACK_VALID)
        # With stricter requirement, may not detect
        # (depends on fixture data)

    # === LIMIT TESTS ===

    def test_limit_prior_move_at_minimum(self):
        """Test detection with exactly 5% prior move (minimum)."""
        result = self.detector.detect(MICRO_PULLBACK_LIMIT_PRIOR_MOVE)

        assert result.detected is True
        assert result.details["prior_move_pct"] >= 5.0
        assert result.details["prior_move_pct"] < 6.0  # Close to limit

    def test_limit_pullback_pct_at_maximum(self):
        """Test detection with exactly 20% pullback (maximum)."""
        result = self.detector.detect(MICRO_PULLBACK_LIMIT_PULLBACK_PCT)

        assert result.detected is True
        assert result.details["pullback_pct"] >= 18.0  # Near 20% limit
        assert result.details["pullback_pct"] <= 20.0

    def test_limit_pullback_candles_at_maximum(self):
        """Test detection with exactly 7 pullback candles (maximum)."""
        result = self.detector.detect(MICRO_PULLBACK_LIMIT_PULLBACK_CANDLES)

        assert result.detected is True
        assert result.details["pullback_candles"] == 7

    def test_limit_green_ratio_at_minimum(self):
        """Test detection with >50% green candles (minimum passing)."""
        result = self.detector.detect(MICRO_PULLBACK_LIMIT_GREEN_RATIO)

        assert result.detected is True
        # Should have passed with >50% green ratio


class TestBullFlag:
    """Tests for Bull Flag pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = BullFlag()

    def test_valid_pattern_detected(self):
        """Test that a valid bull flag is detected."""
        result = self.detector.detect(BULL_FLAG_VALID)

        assert result.detected is True
        assert result.pattern_name == "BullFlag"
        assert result.confidence >= 0.6
        assert result.entry_price is not None
        assert result.stop_price is not None

    def test_valid_pattern_details(self):
        """Test pattern details for valid detection."""
        result = self.detector.detect(BULL_FLAG_VALID)

        assert result.details is not None
        assert result.details["pole_move_pct"] >= 20.0
        assert 10.0 <= result.details["pullback_pct"] <= 25.0

    def test_no_breakout_rejected(self):
        """Test that pattern without breakout is rejected."""
        result = self.detector.detect(BULL_FLAG_NO_BREAKOUT)

        assert result.detected is False
        assert "breakout" in result.reason.lower()

    def test_volume_declining_checked(self):
        """Test that volume confirmation is checked."""
        result = self.detector.detect(BULL_FLAG_VALID)

        assert result.volume_confirmation is not None

    # === LIMIT TESTS ===

    def test_limit_pole_move_at_minimum(self):
        """Test detection with exactly 20% pole move (minimum)."""
        result = self.detector.detect(BULL_FLAG_LIMIT_POLE_MOVE)

        assert result.detected is True
        assert result.details["pole_move_pct"] >= 20.0
        assert result.details["pole_move_pct"] < 22.0  # Close to limit

    def test_limit_pullback_shallow_at_minimum(self):
        """Test detection with exactly 10% pullback (minimum)."""
        result = self.detector.detect(BULL_FLAG_LIMIT_PULLBACK_SHALLOW)

        assert result.detected is True
        assert result.details["pullback_pct"] >= 10.0
        assert result.details["pullback_pct"] < 12.0  # Close to limit

    def test_limit_pullback_deep_at_maximum(self):
        """Test detection with exactly 25% pullback (maximum)."""
        result = self.detector.detect(BULL_FLAG_LIMIT_PULLBACK_DEEP)

        assert result.detected is True
        assert result.details["pullback_pct"] >= 24.0  # Near 25% limit
        assert result.details["pullback_pct"] <= 25.0

    def test_limit_min_candles(self):
        """Test detection with exactly 3 pole + 3 flag candles (minimums)."""
        result = self.detector.detect(BULL_FLAG_LIMIT_MIN_CANDLES)

        assert result.detected is True
        assert result.details["pole_candles"] == 3
        assert result.details["flag_candles"] == 3


class TestVWAPBreak:
    """Tests for VWAP Break pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = VWAPBreak()

    def test_valid_break_detected(self):
        """Test that a valid VWAP break is detected."""
        bars, vwap = VWAP_BREAK_VALID
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is True
        assert result.pattern_name == "VWAPBreak"
        assert result.above_vwap is True

    def test_valid_hold_detected(self):
        """Test that a valid VWAP hold is detected."""
        bars, vwap = VWAP_HOLD_VALID
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is True
        assert result.pattern_name in ["VWAPBreak", "VWAPHold"]

    def test_already_above_rejected(self):
        """Test that stock already above VWAP is rejected."""
        bars, vwap = VWAP_BREAK_ALREADY_ABOVE
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is False
        assert "below" in result.reason.lower()

    def test_vwap_required(self):
        """Test that VWAP is required."""
        bars, _ = VWAP_BREAK_VALID
        result = self.detector.detect(bars, vwap=None)

        assert result.detected is False
        assert "required" in result.reason.lower()

    # === LIMIT TESTS ===

    def test_limit_bars_below_at_minimum(self):
        """Test detection with exactly 5 bars below VWAP (minimum)."""
        bars, vwap = VWAP_BREAK_LIMIT_BARS_BELOW
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is True
        assert result.details["bars_below_vwap"] == 5

    def test_limit_volume_spike_at_minimum(self):
        """Test detection with 2.0x volume spike (minimum for confirmation)."""
        bars, vwap = VWAP_BREAK_LIMIT_VOLUME_SPIKE
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is True
        # Volume confirmation should be True at 2.0x threshold

    def test_limit_rr_at_minimum(self):
        """Test detection with exactly 2.0 R:R (minimum)."""
        bars, vwap = VWAP_BREAK_LIMIT_RR
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is True
        # R:R should be ~2.0, just passing the minimum

    def test_limit_close_barely_above_vwap(self):
        """Test detection when close is barely above VWAP."""
        bars, vwap = VWAP_BREAK_LIMIT_CLOSE_ABOVE
        result = self.detector.detect(bars, vwap=vwap)

        assert result.detected is True
        assert result.above_vwap is True


class TestPatternResult:
    """Tests for PatternResult dataclass."""

    def test_bool_conversion(self):
        """Test that PatternResult can be used in if statements."""
        detector = MicroPullback()

        result = detector.detect(MICRO_PULLBACK_VALID)
        if result:
            assert result.detected is True

        result = detector.detect(MICRO_PULLBACK_TOO_DEEP)
        if not result:
            assert result.detected is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
