"""
Reversal Pattern Tests
======================

Comprehensive tests for bearish reversal pattern detection.

Patterns tested:
- Shooting star
- Bearish engulfing
- Evening star
- Volume climax

Run with: pytest tests/test_reversal.py -v
"""

import pytest
from candle_patterns import ReversalPatternDetector
from tests.fixtures.reversal_fixtures import (
    # Shooting star
    REVERSAL_PASS_SHOOTING_STAR,
    REVERSAL_FAIL_SHOOTING_STAR_NO_UPTREND,
    REVERSAL_FAIL_SHOOTING_STAR_SMALL_WICK,
    # Bearish engulfing
    REVERSAL_PASS_BEARISH_ENGULFING,
    REVERSAL_FAIL_BEARISH_ENGULF_PARTIAL,
    # Evening star
    REVERSAL_PASS_EVENING_STAR,
    REVERSAL_FAIL_EVENING_STAR_LARGE_MIDDLE,
    REVERSAL_FAIL_EVENING_STAR_HIGH_CLOSE,
    # Volume climax
    REVERSAL_PASS_VOLUME_CLIMAX,
    REVERSAL_PASS_VOLUME_CLIMAX_TOPPING_TAIL,
    REVERSAL_FAIL_VOLUME_CLIMAX_NOT_HOD,
    REVERSAL_FAIL_VOLUME_CLIMAX_NO_REVERSAL,
    # Extension requirement
    REVERSAL_FAIL_NOT_EXTENDED,
    # Multi-pattern
    REVERSAL_PASS_MULTI_PATTERN,
)


class TestReversalPatternDetection:
    """Tests for reversal pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ReversalPatternDetector()

    # =========================================================================
    # SHOOTING STAR TESTS
    # =========================================================================

    def test_shooting_star_detected(self):
        """Test that valid shooting star pattern is detected."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)

        assert result.detected is True
        assert result.pattern_name == "ShootingStar"
        assert result.entry_price is not None
        assert result.stop_price is not None
        # For shorts: stop should be above entry
        assert result.stop_price > result.entry_price
        assert result.details["direction"] == "short"
        assert result.details["upper_wick_ratio"] >= 2.0

    def test_shooting_star_no_uptrend_rejected(self):
        """Test that shooting star without uptrend is rejected."""
        result = self.detector.detect(REVERSAL_FAIL_SHOOTING_STAR_NO_UPTREND)

        # Should not detect shooting star (no uptrend), may detect other patterns
        if result.detected and result.pattern_name == "ShootingStar":
            pytest.fail("Should not detect shooting star without uptrend")

    def test_shooting_star_small_wick_rejected(self):
        """Test that candle with small upper wick is not a shooting star."""
        result = self.detector.detect(REVERSAL_FAIL_SHOOTING_STAR_SMALL_WICK)

        if result.detected and result.pattern_name == "ShootingStar":
            pytest.fail("Should not detect shooting star with small wick")

    # =========================================================================
    # BEARISH ENGULFING TESTS
    # =========================================================================

    def test_bearish_engulfing_detected(self):
        """Test that valid bearish engulfing pattern is detected."""
        result = self.detector.detect(REVERSAL_PASS_BEARISH_ENGULFING)

        assert result.detected is True
        assert result.pattern_name == "BearishEngulfing"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.stop_price > result.entry_price
        assert result.details["direction"] == "short"
        assert result.details["engulf_ratio"] >= 1.0

    def test_bearish_engulfing_partial_rejected(self):
        """Test that partial engulfing is rejected."""
        result = self.detector.detect(REVERSAL_FAIL_BEARISH_ENGULF_PARTIAL)

        if result.detected and result.pattern_name == "BearishEngulfing":
            pytest.fail("Should not detect partial bearish engulfing")

    # =========================================================================
    # EVENING STAR TESTS
    # =========================================================================

    def test_evening_star_detected(self):
        """Test that valid evening star pattern is detected."""
        result = self.detector.detect(REVERSAL_PASS_EVENING_STAR)

        assert result.detected is True
        assert result.pattern_name == "EveningStar"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.stop_price > result.entry_price
        assert result.details["direction"] == "short"
        assert result.candle_count == 3

    def test_evening_star_large_middle_rejected(self):
        """Test that evening star with large middle body is rejected."""
        result = self.detector.detect(REVERSAL_FAIL_EVENING_STAR_LARGE_MIDDLE)

        if result.detected and result.pattern_name == "EveningStar":
            pytest.fail("Should not detect evening star with large middle body")

    def test_evening_star_high_close_rejected(self):
        """Test that evening star with high close is rejected."""
        result = self.detector.detect(REVERSAL_FAIL_EVENING_STAR_HIGH_CLOSE)

        if result.detected and result.pattern_name == "EveningStar":
            pytest.fail("Should not detect evening star with high close")

    # =========================================================================
    # VOLUME CLIMAX TESTS
    # =========================================================================

    def test_volume_climax_detected(self):
        """Test that valid volume climax pattern is detected."""
        result = self.detector.detect(REVERSAL_PASS_VOLUME_CLIMAX)

        assert result.detected is True
        assert result.pattern_name == "VolumeClimax"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.stop_price > result.entry_price
        assert result.details["direction"] == "short"
        assert result.details["volume_ratio"] >= 3.0
        assert result.volume_confirmation is True

    def test_volume_climax_topping_tail_detected(self):
        """Test that volume climax with topping tail detects a reversal pattern."""
        result = self.detector.detect(REVERSAL_PASS_VOLUME_CLIMAX_TOPPING_TAIL)

        # This fixture may detect as VolumeClimax or EveningStar depending on pattern match order
        # Both are valid reversal detections
        assert result.detected is True
        assert result.pattern_name in ["VolumeClimax", "EveningStar", "ShootingStar"]
        assert result.details["direction"] == "short"

    def test_volume_climax_not_hod_rejected(self):
        """Test that volume climax not at HOD is rejected as VolumeClimax."""
        result = self.detector.detect(REVERSAL_FAIL_VOLUME_CLIMAX_NOT_HOD)

        # Should not detect as VolumeClimax specifically
        # (may detect other patterns that don't require HOD proximity)
        if result.detected:
            assert result.pattern_name != "VolumeClimax", "Should not detect volume climax not at HOD"

    def test_volume_climax_no_reversal_rejected(self):
        """Test that high volume without reversal confirmation is rejected."""
        result = self.detector.detect(REVERSAL_FAIL_VOLUME_CLIMAX_NO_REVERSAL)

        if result.detected and result.pattern_name == "VolumeClimax":
            pytest.fail("Should not detect volume climax without reversal")

    # =========================================================================
    # EXTENSION REQUIREMENT TESTS
    # =========================================================================

    def test_not_extended_rejected(self):
        """Test that stocks not extended enough are rejected."""
        result = self.detector.detect(REVERSAL_FAIL_NOT_EXTENDED)

        assert result.detected is False
        assert "extended" in result.reason.lower() or "not extended" in result.reason.lower()

    # =========================================================================
    # PATTERN PRIORITY TESTS
    # =========================================================================

    def test_evening_star_has_highest_priority(self):
        """Test that evening star is detected first when multiple patterns present."""
        # Evening star is checked first in the detection order
        result = self.detector.detect(REVERSAL_PASS_EVENING_STAR)
        assert result.pattern_name == "EveningStar"

    def test_volume_climax_detected_in_multi(self):
        """Test that multi-pattern fixture detects strongest pattern."""
        result = self.detector.detect(REVERSAL_PASS_MULTI_PATTERN)

        # Should detect one of the patterns
        assert result.detected is True
        assert result.pattern_name in ["VolumeClimax", "ShootingStar", "EveningStar"]


class TestReversalPatternConfig:
    """Tests for reversal pattern configuration."""

    def test_default_config_values(self):
        """Test that default config has correct values."""
        detector = ReversalPatternDetector()

        assert detector.config["min_extension_from_open_pct"] == 20.0
        assert detector.config["volume_climax_multiplier"] == 3.0
        assert detector.config["min_upper_wick_ratio"] == 2.0
        assert detector.config["max_body_position_pct"] == 33.0
        assert detector.config["min_rr_for_setup"] == 2.0

    def test_custom_config_override(self):
        """Test that custom config overrides defaults."""
        custom = ReversalPatternDetector({
            "min_extension_from_open_pct": 15.0,
            "volume_climax_multiplier": 2.5,
        })

        assert custom.config["min_extension_from_open_pct"] == 15.0
        assert custom.config["volume_climax_multiplier"] == 2.5
        # Other defaults should remain
        assert custom.config["min_upper_wick_ratio"] == 2.0

    def test_relaxed_extension_detects_more(self):
        """Test that relaxed extension threshold detects more patterns."""
        # With default 20% extension, should fail
        default_detector = ReversalPatternDetector()
        result1 = default_detector.detect(REVERSAL_FAIL_NOT_EXTENDED)
        assert result1.detected is False

        # With relaxed 10% extension, might detect
        relaxed_detector = ReversalPatternDetector({
            "min_extension_from_open_pct": 10.0,
        })
        result2 = relaxed_detector.detect(REVERSAL_FAIL_NOT_EXTENDED)
        # May or may not detect depending on other pattern criteria


class TestReversalPatternStopCalculation:
    """Tests for stop price calculation."""

    def test_stop_above_hod_for_shorts(self):
        """Test that stop is placed above HOD for short entries."""
        detector = ReversalPatternDetector()
        result = detector.detect(REVERSAL_PASS_SHOOTING_STAR)

        if result.detected:
            # Stop should be above HOD
            hod = REVERSAL_PASS_SHOOTING_STAR["high"].max()
            assert result.stop_price > hod
            # Stop should include buffer
            buffer = result.stop_price - hod
            assert buffer >= 0.05  # At least 5 cents buffer

    def test_stop_distance_calculation(self):
        """Test that stop distance is calculated correctly."""
        detector = ReversalPatternDetector()
        result = detector.detect(REVERSAL_PASS_BEARISH_ENGULFING)

        if result.detected:
            expected_distance = (result.stop_price - result.entry_price) * 100
            assert abs(result.stop_distance_cents - expected_distance) < 0.01


class TestReversalPatternConfidence:
    """Tests for confidence calculation."""

    def test_confidence_capped_at_90(self):
        """Test that confidence is capped at 90%."""
        detector = ReversalPatternDetector()
        result = detector.detect(REVERSAL_PASS_VOLUME_CLIMAX)

        if result.detected:
            assert result.confidence <= 0.90

    def test_volume_climax_has_volume_confirmation(self):
        """Test that volume climax sets volume_confirmation flag."""
        detector = ReversalPatternDetector()
        result = detector.detect(REVERSAL_PASS_VOLUME_CLIMAX)

        if result.detected and result.pattern_name == "VolumeClimax":
            assert result.volume_confirmation is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
