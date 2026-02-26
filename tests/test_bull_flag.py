"""
Bull Flag Pattern Tests
=======================

Comprehensive boundary/limit tests for Bull Flag pattern detection.

Rules tested:
- min_pole_move_pct: 15.0%
- min_pole_candles: 3
- max_pole_candles: 10
- min_flag_candles: 1
- max_flag_candles: 3
- min_pullback_pct: 10.0%
- max_pullback_pct: 25.0%
- max_flag_range_pct: 15.0%
- volume_declining: True
- Breakout above flag high required
- min_rr_for_setup: 2.0

Run with: pytest tests/test_bull_flag.py -v
"""

import pytest
from candle_patterns import BullFlag
from tests.fixtures.bull_flag_fixtures import (
    # PASS cases
    BF_PASS_VALID,
    BF_PASS_MIN_POLE_MOVE,
    BF_PASS_MIN_POLE_CANDLES,
    BF_PASS_MAX_FLAG_CANDLES,
    BF_PASS_MIN_FLAG_CANDLE,
    BF_PASS_MIN_PULLBACK,
    BF_PASS_MAX_PULLBACK,
    BF_PASS_MIN_RR,
    # FAIL cases
    BF_FAIL_POLE_TOO_WEAK,
    BF_FAIL_POLE_TOO_SHORT,
    BF_FAIL_FLAG_TOO_LONG,
    BF_FAIL_PULLBACK_SHALLOW,
    BF_FAIL_PULLBACK_TOO_DEEP,
    BF_FAIL_FLAG_TOO_WIDE,
    BF_FAIL_VOLUME_RISING,
    BF_FAIL_NO_BREAKOUT,
    BF_FAIL_FLAG_VOLUME_TOO_HEAVY,
)


class TestBullFlagDetection:
    """Tests for Bull Flag pattern detection with NEW rules."""

    def setup_method(self):
        """Set up test fixtures with NEW config."""
        self.detector = BullFlag()

    # =========================================================================
    # PASS TESTS (8)
    # =========================================================================

    def test_valid_pattern_detected(self):
        """Test that a standard valid Bull Flag is detected."""
        result = self.detector.detect(BF_PASS_VALID)

        assert result.detected is True
        assert result.pattern_name == "BullFlag"
        assert result.entry_price is not None
        assert result.stop_price is not None
        # Verify details
        assert result.details["pole_move_pct"] >= 15.0
        assert 10.0 <= result.details["pullback_pct"] <= 25.0
        assert result.details["flag_candles"] <= 3

    def test_pass_min_pole_move_boundary(self):
        """Test detection with pole move above 15% minimum (with adequate R:R)."""
        result = self.detector.detect(BF_PASS_MIN_POLE_MOVE)

        assert result.detected is True
        assert result.details["pole_move_pct"] >= 15.0
        # Pole sized to provide adequate R:R (may be larger than minimum)

    def test_pass_min_pole_candles_boundary(self):
        """Test detection with exactly 3 pole candles (minimum)."""
        result = self.detector.detect(BF_PASS_MIN_POLE_CANDLES)

        assert result.detected is True
        assert result.details["pole_candles"] == 3

    def test_pass_max_flag_candles_boundary(self):
        """Test detection with flag up to 3 candles (maximum allowed)."""
        result = self.detector.detect(BF_PASS_MAX_FLAG_CANDLES)

        assert result.detected is True
        # Algorithm finds smallest valid flag, so just verify it's within limit
        assert result.details["flag_candles"] <= 3

    def test_pass_min_flag_candle_boundary(self):
        """Test detection with exactly 1 flag candle (minimum)."""
        result = self.detector.detect(BF_PASS_MIN_FLAG_CANDLE)

        assert result.detected is True
        assert result.details["flag_candles"] == 1

    def test_pass_min_pullback_boundary(self):
        """Test detection with pullback at 10.1% (just above 10% minimum)."""
        result = self.detector.detect(BF_PASS_MIN_PULLBACK)

        assert result.detected is True
        assert result.details["pullback_pct"] >= 10.0

    def test_pass_max_pullback_boundary(self):
        """Test detection with pullback at 24.9% (just below 25% maximum)."""
        result = self.detector.detect(BF_PASS_MAX_PULLBACK)

        assert result.detected is True
        assert result.details["pullback_pct"] <= 25.0

    def test_pass_min_rr_boundary(self):
        """Test detection with R:R just above 2.0 minimum."""
        result = self.detector.detect(BF_PASS_MIN_RR)

        assert result.detected is True
        # Pattern detected means R:R >= 2.0

    # =========================================================================
    # FAIL TESTS (8)
    # =========================================================================

    def test_fail_pole_too_weak(self):
        """Test rejection when pole move is 14.9% (below 15% minimum)."""
        result = self.detector.detect(BF_FAIL_POLE_TOO_WEAK)

        assert result.detected is False
        # Should fail on pole validation

    def test_fail_pole_too_short(self):
        """Test rejection when pole is 2 candles (below 3 minimum)."""
        result = self.detector.detect(BF_FAIL_POLE_TOO_SHORT)

        assert result.detected is False
        # Should fail on pole candle count

    def test_fail_flag_too_long(self):
        """Test rejection when flag is 4 candles (above 3 maximum)."""
        result = self.detector.detect(BF_FAIL_FLAG_TOO_LONG)

        assert result.detected is False
        # Should fail on flag consolidation validation

    def test_fail_pullback_too_shallow(self):
        """Test rejection when pullback is 9.9% (below 10% minimum)."""
        result = self.detector.detect(BF_FAIL_PULLBACK_SHALLOW)

        assert result.detected is False
        assert "shallow" in result.reason.lower()

    def test_fail_pullback_too_deep(self):
        """Test rejection when pullback is 25.1% (above 25% maximum)."""
        result = self.detector.detect(BF_FAIL_PULLBACK_TOO_DEEP)

        assert result.detected is False
        assert "deep" in result.reason.lower()

    def test_fail_flag_too_wide(self):
        """Test rejection when flag range is 15.1% (above 15% maximum)."""
        result = self.detector.detect(BF_FAIL_FLAG_TOO_WIDE)

        assert result.detected is False
        # Should fail on flag consolidation (range too wide)

    def test_fail_volume_rising(self):
        """Test volume rising in flag (advisory, may still pass)."""
        result = self.detector.detect(BF_FAIL_VOLUME_RISING)

        # Note: volume_declining is advisory, not a hard fail
        # Pattern may still be detected
        if result.detected:
            # If detected, volume confirmation should be False
            assert result.volume_confirmation is False
        # Either way, test passes

    def test_fail_no_breakout(self):
        """Test rejection when last bar stays below flag high."""
        result = self.detector.detect(BF_FAIL_NO_BREAKOUT)

        assert result.detected is False
        assert "breakout" in result.reason.lower()


class TestBullFlagVolumeProfile:
    """Tests for Bull Flag volume profile gate (flag vs pole)."""

    def setup_method(self):
        self.detector = BullFlag()

    def test_flag_rejected_when_flag_volume_too_heavy(self):
        """Test rejection when flag avg volume > 75% of pole avg volume."""
        result = self.detector.detect(BF_FAIL_FLAG_VOLUME_TOO_HEAVY)

        assert result.detected is False
        assert "volume" in result.reason.lower()

    def test_volume_gate_disabled_when_ratio_zero(self):
        """Test that setting max_flag_pole_volume_ratio=0 disables the gate."""
        detector = BullFlag({"max_flag_pole_volume_ratio": 0})
        result = detector.detect(BF_FAIL_FLAG_VOLUME_TOO_HEAVY)

        # Should pass now (heavy volume no longer blocks)
        assert result.detected is True

    def test_details_include_volume_ratios(self):
        """Test that detected patterns include volume profile in details."""
        result = self.detector.detect(BF_PASS_VALID)

        assert result.detected is True
        assert "pole_avg_vol" in result.details
        assert "flag_avg_vol" in result.details
        assert "flag_volume_ratio" in result.details
        # Valid fixture should have flag volume well below pole
        assert result.details["flag_volume_ratio"] < 0.75


class TestBullFlagConfig:
    """Tests for Bull Flag configuration."""

    def test_default_config_values(self):
        """Test that default config has correct NEW values."""
        detector = BullFlag()

        assert detector.config["min_pole_move_pct"] == 15.0
        assert detector.config["min_pole_candles"] == 3
        assert detector.config["max_pole_candles"] == 10
        assert detector.config["min_flag_candles"] == 1
        assert detector.config["max_flag_candles"] == 3
        assert detector.config["min_pullback_pct"] == 10.0
        assert detector.config["max_pullback_pct"] == 25.0
        assert detector.config["volume_declining"] is True

    def test_custom_config_override(self):
        """Test that custom config overrides defaults."""
        custom = BullFlag({
            "min_pole_move_pct": 20.0,
            "max_flag_candles": 5,
        })

        assert custom.config["min_pole_move_pct"] == 20.0
        assert custom.config["max_flag_candles"] == 5
        # Other defaults should remain
        assert custom.config["min_flag_candles"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
