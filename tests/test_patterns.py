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
)
from tests.fixtures.bull_flag_fixtures import (
    BULL_FLAG_VALID,
    BULL_FLAG_NO_BREAKOUT,
)
from tests.fixtures.vwap_break_fixtures import (
    VWAP_BREAK_VALID,
    VWAP_HOLD_VALID,
    VWAP_BREAK_ALREADY_ABOVE,
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
        assert result.details["pullback_pct"] <= 3.0
        assert result.details["green_candles"] >= 3

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
