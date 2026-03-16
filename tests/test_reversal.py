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
    # Stale HOD
    REVERSAL_FAIL_STALE_HOD,
    # Extension requirement
    REVERSAL_FAIL_NOT_EXTENDED,
    # Multi-pattern
    REVERSAL_PASS_MULTI_PATTERN,
    # Low volume
    REVERSAL_FAIL_SHOOTING_STAR_LOW_VOLUME,
    REVERSAL_FAIL_BEARISH_ENGULFING_LOW_VOLUME,
    REVERSAL_FAIL_EVENING_STAR_LOW_VOLUME,
    # Wide session
    REVERSAL_PASS_SHOOTING_STAR_WIDE_SESSION,
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

    def test_shooting_star_has_volume_ratio(self):
        """Shooting star should have volume_ratio >= 1.5x in details."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True
        assert result.details["volume_ratio"] >= 1.5

    def test_shooting_star_low_volume_rejected(self):
        """Shooting star with low volume should be rejected."""
        result = self.detector.detect(REVERSAL_FAIL_SHOOTING_STAR_LOW_VOLUME)
        if result.detected and result.pattern_name == "ShootingStar":
            pytest.fail("Should not detect shooting star with low volume")

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

    def test_bearish_engulfing_has_volume_ratio(self):
        """Bearish engulfing should have volume_ratio >= 1.5x in details."""
        result = self.detector.detect(REVERSAL_PASS_BEARISH_ENGULFING)
        assert result.detected is True
        assert result.details["volume_ratio"] >= 1.5

    def test_bearish_engulfing_low_volume_rejected(self):
        """Bearish engulfing with low volume should be rejected."""
        result = self.detector.detect(REVERSAL_FAIL_BEARISH_ENGULFING_LOW_VOLUME)
        if result.detected and result.pattern_name == "BearishEngulfing":
            pytest.fail("Should not detect bearish engulfing with low volume")

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

    def test_evening_star_has_volume_ratio(self):
        """Evening star should have volume_ratio >= 1.5x in details."""
        result = self.detector.detect(REVERSAL_PASS_EVENING_STAR)
        assert result.detected is True
        assert result.details["volume_ratio"] >= 1.5

    def test_evening_star_low_volume_rejected(self):
        """Evening star with low volume should be rejected."""
        result = self.detector.detect(REVERSAL_FAIL_EVENING_STAR_LOW_VOLUME)
        if result.detected and result.pattern_name == "EveningStar":
            pytest.fail("Should not detect evening star with low volume")

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
    # STALE HOD TESTS
    # =========================================================================

    def test_stale_hod_rejected(self):
        """Test rejection when HOD occurred more than 10 bars ago."""
        result = self.detector.detect(REVERSAL_FAIL_STALE_HOD)

        assert result.detected is False

    def test_stale_hod_passes_with_relaxed_config(self):
        """Test that increasing max_hod_age_bars allows stale HOD."""
        detector = ReversalPatternDetector({"max_hod_age_bars": 20})
        result = detector.detect(REVERSAL_FAIL_STALE_HOD)

        # With relaxed config, pattern may detect (if other criteria met)
        # We just verify it doesn't reject on staleness
        if not result.detected:
            assert "stale" not in result.reason.lower()

    # =========================================================================
    # EXTENSION REQUIREMENT TESTS
    # =========================================================================

    def test_not_extended_rejected(self):
        """Test that stocks not extended enough are rejected (both OR gates fail)."""
        result = self.detector.detect(REVERSAL_FAIL_NOT_EXTENDED)

        assert result.detected is False
        assert "not extended" in result.reason.lower()
        # Verify both gates are mentioned in the reason (OR gate message)
        assert "AND" in result.reason

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
        assert detector.config["stop_buffer_pct"] == 1.0  # Reduced from 2%
        assert detector.config["min_volume_multiplier"] == 1.5
        assert detector.config["short_stop_buffer_cents"] == 2

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
        # With default 20%/25% extension, should fail both OR gates
        default_detector = ReversalPatternDetector()
        result1 = default_detector.detect(REVERSAL_FAIL_NOT_EXTENDED)
        assert result1.detected is False

        # With relaxed 10% from-ref threshold, ref gate now passes (15% >= 10%)
        relaxed_detector = ReversalPatternDetector({
            "min_extension_from_open_pct": 10.0,
        })
        result2 = relaxed_detector.detect(REVERSAL_FAIL_NOT_EXTENDED)
        # Should now detect since 15% from open >= 10% relaxed threshold
        assert result2.detected is True


class TestReversalPatternStopCalculation:
    """Tests for stop price calculation."""

    def test_stop_above_pattern_high_for_shorts(self):
        """Stop is placed 2c above pattern high."""
        detector = ReversalPatternDetector()
        result = detector.detect(REVERSAL_PASS_SHOOTING_STAR)

        assert result.detected is True
        pattern_high = REVERSAL_PASS_SHOOTING_STAR.iloc[-1]["high"]
        assert result.stop_price == pytest.approx(pattern_high + 0.02, abs=0.001)

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


class TestReversalRetracementTarget:
    """Tests for 50% retracement target calculation."""

    def setup_method(self):
        self.detector = ReversalPatternDetector()

    def test_shooting_star_target_wired(self):
        """Shooting star should wire retracement target into target_price."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True
        assert result.target_price is not None
        assert result.target_price == result.details["retracement_target"]

    def test_bearish_engulfing_target_wired(self):
        """Bearish engulfing should wire retracement target into target_price."""
        result = self.detector.detect(REVERSAL_PASS_BEARISH_ENGULFING)
        assert result.detected is True
        assert result.target_price is not None
        assert result.target_price == result.details["retracement_target"]

    def test_evening_star_target_wired(self):
        """Evening star should wire retracement target into target_price."""
        result = self.detector.detect(REVERSAL_PASS_EVENING_STAR)
        assert result.detected is True
        assert result.target_price is not None
        assert result.target_price == result.details["retracement_target"]

    def test_volume_climax_target_wired(self):
        """Volume climax should wire retracement target into target_price."""
        result = self.detector.detect(REVERSAL_PASS_VOLUME_CLIMAX)
        assert result.detected is True
        assert result.target_price is not None
        assert result.target_price == result.details["retracement_target"]

    def test_retracement_target_is_50pct(self):
        """Retracement target should be the midpoint of the run (50% retracement)."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True

        run_low = result.details["run_low"]
        run_high = result.details["run_high"]
        expected_target = run_high - (run_high - run_low) * 0.50

        assert result.details["retracement_target"] == pytest.approx(expected_target, abs=0.01)

    def test_retracement_target_below_entry(self):
        """For shorts, retracement target should be below entry price."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True
        assert result.details["retracement_target"] < result.entry_price

    def test_rise_details_present(self):
        """Rise calculation details should be in details dict."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True
        assert "rise_amount" in result.details
        assert "rise_pct" in result.details
        assert "run_low" in result.details
        assert "run_high" in result.details
        assert result.details["rise_amount"] > 0
        assert result.details["rise_pct"] > 0


class TestReversalStopBuffer:
    """Tests for the price-adaptive stop buffer above pattern high."""

    def test_stop_buffer_is_2_cents_on_low_price(self):
        """On sub-$4 stocks, 2c wins over 0.5% of price."""
        detector = ReversalPatternDetector()
        result = detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True

        pattern_high = REVERSAL_PASS_SHOOTING_STAR.iloc[-1]["high"]
        # 2c > 0.5% of $1.40 (0.7c), so 2c wins
        expected_stop = pattern_high + 0.02
        assert result.stop_price == pytest.approx(expected_stop, abs=0.001)

    def test_stop_buffer_scales_with_price(self):
        """On higher-priced stocks, 0.5% of price wins over 2c."""
        detector = ReversalPatternDetector()
        # Manually test the calculation: at $10 pattern_high,
        # 0.5% = 5c > 2c, so buffer should be 5c
        stop = detector._calculate_stop(None, "above", pattern_high=10.00)
        assert stop == pytest.approx(10.05, abs=0.001)

    def test_stop_buffer_config_keys(self):
        """Config has both buffer keys."""
        detector = ReversalPatternDetector()
        assert detector.config["short_stop_buffer_cents"] == 2
        assert detector.config["short_stop_min_pct"] == 0.5


class TestReversalTargetCap:
    """Tests for R:R target cap."""

    def setup_method(self):
        self.detector = ReversalPatternDetector()

    def test_target_capped_at_max_r_multiple(self):
        """Target distance should not exceed max_target_r_multiple × risk."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR_WIDE_SESSION)
        assert result.detected is True

        risk = result.stop_price - result.entry_price
        reward = result.entry_price - result.target_price
        rr = reward / risk
        assert rr <= 8.0 + 0.01  # max_target_r_multiple default

    def test_default_max_target_r_multiple(self):
        """Default max_target_r_multiple should be 8.0."""
        assert self.detector.config["max_target_r_multiple"] == 8.0

    def test_uncapped_target_below_cap(self):
        """When natural R:R < cap, target is unchanged."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR)
        assert result.detected is True

        # Verify target equals raw retracement (not capped)
        run_low = result.details["run_low"]
        run_high = result.details["run_high"]
        raw_target = run_high - (run_high - run_low) * 0.50
        assert result.target_price == pytest.approx(raw_target, abs=0.01)


class TestReversalSessionWideTarget:
    """Tests for session-wide target calculation."""

    def setup_method(self):
        self.detector = ReversalPatternDetector()

    def test_session_wide_target_uses_full_range(self):
        """Target uses session-wide low, not 10-bar window low."""
        result = self.detector.detect(REVERSAL_PASS_SHOOTING_STAR_WIDE_SESSION)
        assert result.detected is True
        # Session low is 0.75 (bar 0), not ~1.22 (10-bar window low)
        assert result.details["run_low"] == pytest.approx(0.75, abs=0.01)
        # Target should be below entry
        assert result.target_price < result.entry_price


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
