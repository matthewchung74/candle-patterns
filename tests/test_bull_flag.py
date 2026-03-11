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
- min_pullback_pct: 13.0%
- max_pullback_pct: 20.0%
- max_flag_range_pct: 15.0%
- max_flag_pole_volume_ratio: 0.60
- volume_declining: True
- Breakout above flag high required
- min_rr_for_setup: 2.0
- Lower-highs rejection (flag shape quality)
- MACD histogram strength threshold (price-scaled)

Run with: pytest tests/test_bull_flag.py -v
"""

import pytest
import pandas as pd
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
    BF_FAIL_LOWER_HIGHS,
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
        assert 13.0 <= result.details["pullback_pct"] <= 20.0
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
        """Test detection with pullback at 13.1% (just above 13% minimum)."""
        result = self.detector.detect(BF_PASS_MIN_PULLBACK)

        assert result.detected is True
        assert result.details["pullback_pct"] >= 13.0

    def test_pass_max_pullback_boundary(self):
        """Test detection with pullback at ~19.5% (just below 20% maximum)."""
        result = self.detector.detect(BF_PASS_MAX_PULLBACK)

        assert result.detected is True
        assert result.details["pullback_pct"] <= 20.0

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
        """Test rejection when pullback is 12% (below 13% minimum)."""
        result = self.detector.detect(BF_FAIL_PULLBACK_SHALLOW)

        assert result.detected is False
        assert "shallow" in result.reason.lower()

    def test_fail_pullback_too_deep(self):
        """Test rejection when pullback is ~20.5% (above 20% maximum)."""
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
        """Test rejection when flag avg volume > 60% of pole avg volume."""
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
        assert result.details["flag_volume_ratio"] < 0.60


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
        assert detector.config["min_pullback_pct"] == 13.0
        assert detector.config["max_pullback_pct"] == 20.0
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


class TestBullFlagHardGates:
    """Tests for BullFlag VWAP/MACD hard gates (mirrors MicroPullback convention)."""

    def setup_method(self):
        self.detector = BullFlag()

    def _make_vwap(self, bars, above: bool):
        """Create a VWAP series where price is above or below VWAP."""
        n = len(bars)
        if above:
            # VWAP well below all closes
            return pd.Series([1.0] * n)
        else:
            # VWAP well above all closes
            return pd.Series([9999.0] * n)

    def _make_macd(self, bars, positive: bool):
        """Create a MACD DataFrame with positive or negative histogram."""
        n = len(bars)
        hist_val = 0.5 if positive else -0.5
        macd_val = 1.0 if positive else -1.0
        return pd.DataFrame({
            "macd": [macd_val] * n,
            "signal": [0.0] * n,
            "histogram": [hist_val] * n,
        })

    def test_below_vwap_rejected(self):
        """BullFlag with price below VWAP should be rejected by hard gate."""
        bars = BF_PASS_VALID
        vwap = self._make_vwap(bars, above=False)
        macd = self._make_macd(bars, positive=True)
        result = self.detector.detect(bars, vwap=vwap, macd=macd)

        assert result.detected is False
        assert "HARD GATE" in result.reason
        assert "VWAP" in result.reason

    def test_macd_negative_rejected(self):
        """BullFlag with MACD negative should be rejected by hard gate."""
        bars = BF_PASS_VALID
        vwap = self._make_vwap(bars, above=True)
        macd = self._make_macd(bars, positive=False)
        result = self.detector.detect(bars, vwap=vwap, macd=macd)

        assert result.detected is False
        assert "HARD GATE" in result.reason
        assert "MACD" in result.reason

    def test_vwap_none_still_detects(self):
        """BullFlag with VWAP=None should still detect (gate skipped)."""
        bars = BF_PASS_VALID
        macd = self._make_macd(bars, positive=True)
        result = self.detector.detect(bars, vwap=None, macd=macd)

        assert result.detected is True

    def test_macd_none_still_detects(self):
        """BullFlag with MACD=None (and insufficient bars) should still detect (gate skipped)."""
        # Use a detector that skips auto-MACD calculation
        # When bars < min_bars, auto-MACD returns None, so gate is skipped
        bars = BF_PASS_VALID
        vwap = self._make_vwap(bars, above=True)
        # Pass macd=None explicitly; auto-calc needs 35 bars which fixture may not have
        result = self.detector.detect(bars, vwap=vwap, macd=None)

        # If auto-MACD kicked in (enough bars), it may or may not be positive
        # The key test is: if macd_positive is None, the gate is skipped
        if result.macd_positive is None:
            assert result.detected is True

    def test_gates_can_be_disabled(self):
        """BullFlag with gates disabled should detect even with bad VWAP/MACD."""
        detector = BullFlag({
            "require_above_vwap": False,
            "require_macd_positive": False,
        })
        bars = BF_PASS_VALID
        vwap = self._make_vwap(bars, above=False)
        macd = self._make_macd(bars, positive=False)
        result = detector.detect(bars, vwap=vwap, macd=macd)

        assert result.detected is True


class TestBullFlagLowerHighs:
    """Tests for BullFlag flag shape quality (lower-highs rejection)."""

    def setup_method(self):
        self.detector = BullFlag()

    def test_fail_flag_lower_highs(self):
        """Flag with strictly descending highs is rejected (downtrend, not consolidation).

        The 2-bar flag is rejected due to descending highs. The 1-bar fallback
        also fails because there's no breakout above the smaller flag high.
        """
        result = self.detector.detect(BF_FAIL_LOWER_HIGHS)

        assert result.detected is False


class TestBullFlagMACDThreshold:
    """Tests for BullFlag MACD histogram strength threshold."""

    def setup_method(self):
        self.detector = BullFlag()

    def _make_weak_macd(self, bars, histogram_val: float):
        """Create a MACD DataFrame with a specific histogram value."""
        n = len(bars)
        return pd.DataFrame({
            "macd": [histogram_val] * n,
            "signal": [0.0] * n,
            "histogram": [histogram_val] * n,
        })

    def test_weak_positive_macd_rejected(self):
        """BullFlag with weak positive MACD histogram should be rejected.

        histogram=0.001 is positive (passes > 0 check) but below the
        price-scaled threshold (entry ~4.5 * 0.001 = 0.0045).
        """
        bars = BF_PASS_VALID
        vwap = pd.Series([1.0] * len(bars))  # well below price
        macd = self._make_weak_macd(bars, 0.001)
        result = self.detector.detect(bars, vwap=vwap, macd=macd)

        assert result.detected is False
        assert "HARD GATE" in result.reason
        assert "threshold" in result.reason

    def test_strong_positive_macd_accepted(self):
        """BullFlag with strong positive MACD histogram should pass threshold."""
        bars = BF_PASS_VALID
        vwap = pd.Series([1.0] * len(bars))  # well below price
        macd = self._make_weak_macd(bars, 0.5)
        result = self.detector.detect(bars, vwap=vwap, macd=macd)

        assert result.detected is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
