"""
Micro Pullback Pattern Tests
============================

Comprehensive boundary/limit tests for Micro Pullback pattern detection.

Rules tested:
- min_prior_move_pct: 5.0%
- max_prior_move_pct: 15.0%
- max_pullback_pct: 12.0%
- max_pullback_candles: 2
- >50% green candles in surge
- Entry candle must be green
- min_rr_for_setup: 2.0

Run with: pytest tests/test_micro_pullback.py -v
"""

import pytest
from candle_patterns import MicroPullback
from tests.fixtures.micro_pullback_fixtures import (
    # PASS cases
    MP_PASS_VALID,
    MP_PASS_MIN_PRIOR_MOVE,
    MP_PASS_MAX_PRIOR_MOVE,
    MP_PASS_MAX_PULLBACK,
    MP_PASS_MAX_DURATION,
    MP_PASS_MIN_GREEN_RATIO,
    MP_PASS_MIN_RR,
    # FAIL cases
    MP_FAIL_BELOW_MIN_PRIOR,
    MP_FAIL_ABOVE_MAX_PRIOR,
    MP_FAIL_PULLBACK_TOO_DEEP,
    MP_FAIL_DURATION_TOO_LONG,
    MP_FAIL_GREEN_RATIO_LOW,
    MP_FAIL_LAST_BAR_RED,
    MP_FAIL_RR_TOO_LOW,
)


class TestMicroPullbackDetection:
    """Tests for Micro Pullback pattern detection with NEW rules."""

    def setup_method(self):
        """Set up test fixtures with NEW config."""
        self.detector = MicroPullback()

    # =========================================================================
    # PASS TESTS (7)
    # =========================================================================

    def test_valid_pattern_detected(self):
        """Test that a standard valid pattern is detected."""
        result = self.detector.detect(MP_PASS_VALID)

        assert result.detected is True
        assert result.pattern_name == "MicroPullback"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.entry_price > result.stop_price
        # Verify details
        assert 5.0 <= result.details["prior_move_pct"] <= 15.0
        assert result.details["pullback_pct"] <= 12.0
        assert result.details["pullback_candles"] <= 2

    def test_pass_min_prior_move_boundary(self):
        """Test detection with prior move at 5.1% (just above 5% minimum)."""
        result = self.detector.detect(MP_PASS_MIN_PRIOR_MOVE)

        assert result.detected is True
        assert result.details["prior_move_pct"] >= 5.0
        assert result.details["prior_move_pct"] < 6.0  # Close to minimum

    def test_pass_max_prior_move_boundary(self):
        """Test detection with prior move at 14.9% (just below 15% maximum)."""
        result = self.detector.detect(MP_PASS_MAX_PRIOR_MOVE)

        assert result.detected is True
        assert result.details["prior_move_pct"] >= 14.0
        assert result.details["prior_move_pct"] <= 15.0  # Just under max

    def test_pass_max_pullback_boundary(self):
        """Test detection with pullback at 11.9% (just below 12% maximum)."""
        result = self.detector.detect(MP_PASS_MAX_PULLBACK)

        assert result.detected is True
        assert result.details["pullback_pct"] <= 12.0

    def test_pass_max_duration_boundary(self):
        """Test detection with exactly 2 pullback candles (maximum)."""
        result = self.detector.detect(MP_PASS_MAX_DURATION)

        assert result.detected is True
        assert result.details["pullback_candles"] == 2

    def test_pass_min_green_ratio_boundary(self):
        """Test detection with 60% green ratio (just above 50% minimum)."""
        result = self.detector.detect(MP_PASS_MIN_GREEN_RATIO)

        assert result.detected is True
        # Pattern detected means green ratio was >50%

    def test_pass_min_rr_boundary(self):
        """Test detection with R:R just above 2.0 minimum."""
        result = self.detector.detect(MP_PASS_MIN_RR)

        assert result.detected is True
        # Pattern detected means R:R >= 2.0

    # =========================================================================
    # FAIL TESTS (7)
    # =========================================================================

    def test_fail_below_min_prior_move(self):
        """Test rejection when prior move is 4.9% (below 5% minimum)."""
        result = self.detector.detect(MP_FAIL_BELOW_MIN_PRIOR)

        assert result.detected is False
        # Should fail on surge validation

    def test_fail_above_max_prior_move(self):
        """Test rejection when prior move is 15.1% (above 15% maximum)."""
        result = self.detector.detect(MP_FAIL_ABOVE_MAX_PRIOR)

        assert result.detected is False
        assert "too large" in result.reason.lower() or "bull flag" in result.reason.lower()

    def test_fail_pullback_too_deep(self):
        """Test rejection when pullback is 12.1% (above 12% maximum)."""
        result = self.detector.detect(MP_FAIL_PULLBACK_TOO_DEEP)

        assert result.detected is False
        assert "deep" in result.reason.lower()

    def test_fail_duration_too_long(self):
        """Test rejection when pullback is 3 candles (above 2 maximum)."""
        result = self.detector.detect(MP_FAIL_DURATION_TOO_LONG)

        assert result.detected is False
        assert "long" in result.reason.lower()

    def test_fail_green_ratio_too_low(self):
        """Test rejection when green ratio is 40% (below 50%)."""
        result = self.detector.detect(MP_FAIL_GREEN_RATIO_LOW)

        assert result.detected is False
        # Should fail on surge validation (not enough green)

    def test_fail_last_bar_red(self):
        """Test rejection when last bar is red (no entry signal)."""
        result = self.detector.detect(MP_FAIL_LAST_BAR_RED)

        assert result.detected is False
        assert "red" in result.reason.lower()

    def test_fail_rr_too_low(self):
        """Test rejection when R:R is 1.9 (below 2.0 minimum)."""
        result = self.detector.detect(MP_FAIL_RR_TOO_LOW)

        assert result.detected is False
        # May fail on R:R or other criteria


class TestMicroPullbackConfig:
    """Tests for Micro Pullback configuration."""

    def test_default_config_values(self):
        """Test that default config has correct NEW values."""
        detector = MicroPullback()

        assert detector.config["min_prior_move_pct"] == 5.0
        assert detector.config["max_prior_move_pct"] == 15.0
        assert detector.config["max_pullback_pct"] == 12.0
        assert detector.config["max_pullback_candles"] == 2
        assert detector.config["entry"] == "first_green_after_pullback"

    def test_custom_config_override(self):
        """Test that custom config overrides defaults."""
        custom = MicroPullback({
            "min_prior_move_pct": 8.0,
            "max_pullback_pct": 10.0,
        })

        assert custom.config["min_prior_move_pct"] == 8.0
        assert custom.config["max_pullback_pct"] == 10.0
        # Other defaults should remain
        assert custom.config["max_prior_move_pct"] == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
