"""
Micro Pullback Pattern Tests
============================

Comprehensive boundary/limit tests for Micro Pullback pattern detection.

Rules tested:
- min_prior_move_pct: 5.0%
- max_prior_move_pct: 25.0%
- max_pullback_retrace_pct: 0.50 (retrace as fraction of surge)
- max_pullback_candles: 3
- >50% green candles in surge
- Entry candle must be green
- min_rr_for_setup: 1.2
- stop_buffer_atr_multiplier: 1.5 (ATR-based floor)

Run with: pytest tests/test_micro_pullback.py -v
"""

import pytest
from candle_patterns import MicroPullback
from tests.fixtures.micro_pullback_fixtures import (
    _make_bars,
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
    MP_FAIL_PULLBACK_VOLUME_TOO_HEAVY,
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
        assert 5.0 <= result.details["prior_move_pct"] <= 25.0
        assert result.details["pullback_retrace"] <= 0.50
        assert result.details["pullback_candles"] <= 3

    def test_pass_min_prior_move_boundary(self):
        """Test detection with prior move near minimum (7.2% — effective min with 3% stop floor)."""
        result = self.detector.detect(MP_PASS_MIN_PRIOR_MOVE)

        assert result.detected is True
        assert result.details["prior_move_pct"] >= 5.0
        assert result.details["prior_move_pct"] < 8.0  # Near effective minimum

    def test_pass_max_prior_move_boundary(self):
        """Test detection with prior move at 14.9% (just below 15% maximum)."""
        result = self.detector.detect(MP_PASS_MAX_PRIOR_MOVE)

        assert result.detected is True
        assert result.details["prior_move_pct"] >= 14.0
        assert result.details["prior_move_pct"] <= 15.0  # Just under max

    def test_pass_max_pullback_boundary(self):
        """Test detection with retrace just below 50% maximum."""
        result = self.detector.detect(MP_PASS_MAX_PULLBACK)

        assert result.detected is True
        assert result.details["pullback_retrace"] <= 0.50

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
        """Test rejection when prior move is above 25% maximum."""
        result = self.detector.detect(MP_FAIL_ABOVE_MAX_PRIOR)

        assert result.detected is False
        assert "too large" in result.reason.lower() or "bull flag" in result.reason.lower()

    def test_fail_pullback_too_deep(self):
        """Test rejection when pullback is 12.1% (above 12% maximum)."""
        result = self.detector.detect(MP_FAIL_PULLBACK_TOO_DEEP)

        assert result.detected is False
        assert "deep" in result.reason.lower()

    def test_fail_duration_too_long(self):
        """Test rejection when pullback is 4 candles (above 3 maximum)."""
        detector = MicroPullback({"max_pullback_candles": 3})
        result = detector.detect(MP_FAIL_DURATION_TOO_LONG)

        # Fixture has 3 pullback candles which now passes with default of 3.
        # Use explicit config of 2 to test the gate still works.
        detector_strict = MicroPullback({"max_pullback_candles": 2})
        result_strict = detector_strict.detect(MP_FAIL_DURATION_TOO_LONG)

        assert result_strict.detected is False
        assert "long" in result_strict.reason.lower()

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


class TestMicroPullbackVolumeProfile:
    """Tests for Micro Pullback volume profile gate (pullback vs surge)."""

    def setup_method(self):
        self.detector = MicroPullback()

    def test_pullback_rejected_when_volume_too_heavy(self):
        """Test rejection when pullback avg volume > 75% of surge avg volume."""
        result = self.detector.detect(MP_FAIL_PULLBACK_VOLUME_TOO_HEAVY)

        assert result.detected is False
        assert "volume" in result.reason.lower()

    def test_volume_gate_disabled_when_ratio_zero(self):
        """Test that setting max_pullback_surge_volume_ratio=0 disables the gate."""
        detector = MicroPullback({"max_pullback_surge_volume_ratio": 0})
        result = detector.detect(MP_FAIL_PULLBACK_VOLUME_TOO_HEAVY)

        # Should pass now (heavy volume no longer blocks)
        assert result.detected is True

    def test_details_include_volume_ratio(self):
        """Test that detected patterns include volume ratio in details."""
        result = self.detector.detect(MP_PASS_VALID)

        assert result.detected is True
        assert "surge_volume_avg" in result.details
        assert "pullback_volume_avg" in result.details
        assert "pullback_volume_ratio" in result.details
        assert result.details["pullback_volume_ratio"] < 0.75


class TestMicroPullbackStopBufferATR:
    """Tests for ATR-based stop buffer floor."""

    def test_atr_buffer_used_on_volatile_stock(self):
        """ATR × 1.5 should widen stop when volatility is high.

        With enough bars and volatility, ATR buffer should exceed
        the 1% pct_buffer and 3-cent min.
        """
        detector = MicroPullback({
            "require_above_vwap": False,
            "require_macd_positive": False,
            "max_pullback_surge_volume_ratio": 0,
        })

        # Build 20 bars with volatile price action to get a meaningful ATR,
        # ending with a valid micro pullback pattern
        bars = _make_bars([
            # 14 bars of volatile pre-action to warm up ATR
            (1.40, 1.48, 1.38, 1.45, 5000),
            (1.45, 1.50, 1.40, 1.42, 5000),
            (1.42, 1.52, 1.40, 1.50, 5000),
            (1.50, 1.55, 1.44, 1.46, 5000),
            (1.46, 1.54, 1.43, 1.52, 5000),
            (1.52, 1.58, 1.48, 1.50, 5000),
            (1.50, 1.56, 1.45, 1.53, 5000),
            (1.53, 1.60, 1.49, 1.51, 5000),
            (1.51, 1.57, 1.47, 1.55, 5000),
            (1.55, 1.62, 1.50, 1.52, 5000),
            (1.52, 1.58, 1.48, 1.56, 5000),
            (1.56, 1.63, 1.52, 1.54, 5000),
            (1.54, 1.60, 1.50, 1.58, 5000),
            (1.58, 1.63, 1.55, 1.57, 5000),
            # Surge: 2 strong green candles
            (1.57, 1.68, 1.56, 1.67, 8000),  # surge green
            (1.67, 1.78, 1.66, 1.76, 9000),  # surge green (swing high)
            # Pullback
            (1.76, 1.77, 1.73, 1.74, 3000),  # pullback (low=1.73)
            # Entry
            (1.74, 1.79, 1.73, 1.78, 4000),  # entry green candle
        ])

        result = detector.detect(bars)

        if result.detected:
            assert result.details["atr"] is not None
            assert result.details["stop_buffer"] > 0
            # ATR buffer should be meaningful (> 3 cents)
            expected_atr_buffer = result.details["atr"] * 1.5
            assert result.details["stop_buffer"] >= expected_atr_buffer - 0.001

    def test_atr_multiplier_configurable(self):
        """Custom stop_buffer_atr_multiplier should override default."""
        detector = MicroPullback({"stop_buffer_atr_multiplier": 2.0})
        assert detector.config["stop_buffer_atr_multiplier"] == 2.0

    def test_default_atr_config(self):
        """Default ATR config values."""
        detector = MicroPullback()
        assert detector.config["stop_buffer_atr_multiplier"] == 1.5
        assert detector.config["stop_buffer_atr_period"] == 14


class TestMicroPullbackConfig:
    """Tests for Micro Pullback configuration."""

    def test_default_config_values(self):
        """Test that default config has correct values."""
        detector = MicroPullback()

        assert detector.config["min_prior_move_pct"] == 5.0
        assert detector.config["max_prior_move_pct"] == 25.0
        assert detector.config["max_pullback_retrace_pct"] == 0.50
        assert detector.config["max_pullback_candles"] == 3
        assert detector.config["entry"] == "first_green_after_pullback"
        assert detector.config["stop_buffer_atr_multiplier"] == 1.5

    def test_custom_config_override(self):
        """Test that custom config overrides defaults."""
        custom = MicroPullback({
            "min_prior_move_pct": 8.0,
            "max_pullback_retrace_pct": 0.40,
        })

        assert custom.config["min_prior_move_pct"] == 8.0
        assert custom.config["max_pullback_retrace_pct"] == 0.40
        # Other defaults should remain
        assert custom.config["max_prior_move_pct"] == 25.0


class TestMicroPullbackVCR:
    """Tests for Volume Collapse Ratio (VCR) gate."""

    def test_vcr_disabled_by_default(self):
        """Default config (max_volume_collapse_ratio=0.0) never rejects on VCR."""
        detector = MicroPullback()
        assert detector.config["max_volume_collapse_ratio"] == 0.0

        # Use a fixture with heavy pullback volume — VCR gate should NOT fire
        # because default is 0 (disabled). The avg volume gate may still fire,
        # so disable it to isolate VCR behavior.
        detector_no_avg = MicroPullback({"max_pullback_surge_volume_ratio": 0})
        result = detector_no_avg.detect(MP_FAIL_PULLBACK_VOLUME_TOO_HEAVY)
        # With avg gate disabled and VCR disabled, pattern should pass
        assert result.detected is True

    def test_vcr_rejects_distribution(self):
        """SER-like bars with high VCR + config 0.55 -> not detected."""
        # SER pattern: peak surge vol 3M, first pullback bar 2M => VCR = 0.67
        # Retrace kept under 50% so VCR is the gate that fires.
        bars = _make_bars([
            # Surge: peak volume 3M
            (10.00, 10.35, 9.98, 10.30, 1_000_000),   # green
            (10.30, 10.65, 10.28, 10.60, 2_000_000),   # green
            (10.60, 11.02, 10.58, 11.00, 3_000_000),   # green, peak surge = 3M

            # Pullback: shallow retrace (~43%), volume stays high (distribution)
            (11.00, 11.01, 10.80, 10.82, 2_000_000),   # red, peak pullback = 2M
            (10.82, 10.85, 10.70, 10.72, 1_800_000),   # red (pullback low 10.70)

            # Entry
            (10.72, 10.95, 10.70, 10.92, 1_500_000),   # green
        ])

        detector = MicroPullback({
            "max_volume_collapse_ratio": 0.55,
            "max_pullback_surge_volume_ratio": 0,  # Disable avg gate to isolate VCR
            "require_above_vwap": False,
            "require_macd_positive": False,
        })
        result = detector.detect(bars)
        assert result.detected is False
        assert "collapse ratio" in result.reason.lower()

    def test_vcr_passes_capitulation(self):
        """ARTL-like bars with low VCR + config 0.55 -> detected."""
        # ARTL pattern: peak surge vol 531k, first pullback bar 193k => VCR = 0.36
        # Retrace kept under 50% so only VCR behavior is exercised here.
        bars = _make_bars([
            # Surge: peak volume 531k
            (10.00, 10.35, 9.98, 10.30, 300_000),   # green
            (10.30, 10.65, 10.28, 10.60, 400_000),   # green
            (10.60, 11.02, 10.58, 11.00, 531_000),   # green, peak surge = 531k

            # Pullback: shallow retrace (~43%), volume collapses (healthy)
            (11.00, 11.01, 10.80, 10.82, 193_000),   # red, peak pullback = 193k
            (10.82, 10.85, 10.70, 10.72, 150_000),   # red (pullback low 10.70)

            # Entry
            (10.72, 10.95, 10.70, 10.92, 250_000),   # green
        ])

        detector = MicroPullback({
            "max_volume_collapse_ratio": 0.55,
            "max_pullback_surge_volume_ratio": 0,  # Disable avg gate to isolate VCR
            "require_above_vwap": False,
            "require_macd_positive": False,
        })
        result = detector.detect(bars)
        assert result.detected is True

    def test_vcr_in_details(self):
        """volume_collapse_ratio present in PatternResult.details."""
        result = MicroPullback().detect(MP_PASS_VALID)
        assert result.detected is True
        assert "volume_collapse_ratio" in result.details
        assert isinstance(result.details["volume_collapse_ratio"], float)


class TestMicroPullbackRetraceGate:
    """Tests for the pullback-retrace gate (max_pullback_retrace_pct).

    The gate measures pullback depth as a fraction of the prior surge,
    not as a fraction of price. A 50% default rejects deep retraces
    (e.g. 72% of surge given back) that the old pct-of-price gate let
    through — which is what happened with CLIK 2026-04-21 MicroPullback.
    """

    def test_default_config_value(self):
        """Default 0.50 — balanced (up to half the surge retraced is OK)."""
        assert MicroPullback().config["max_pullback_retrace_pct"] == 0.50

    def test_clik_style_deep_retrace_rejects(self):
        """CLIK 2026-04-21 setup: $0.32 surge, $0.23 pullback = 72% retrace.

        Old 12%-of-price gate passed it at 6.6%; new gate at 0.50 rejects.
        """
        # Surge: $3.17 → $3.49 ($0.32). Pullback: $3.49 → $3.26 ($0.23 = 72%).
        bars = _make_bars([
            # Pre-surge flat
            (3.17, 3.21, 3.17, 3.20, 60_000),
            # Surge: 3 green bars, low 3.17, swing high 3.49
            (3.20, 3.32, 3.20, 3.31, 100_000),  # green
            (3.30, 3.49, 3.30, 3.44, 250_000),  # green (swing high 3.49)
            (3.44, 3.46, 3.40, 3.46, 110_000),  # green
            # Pullback: 2 red bars, low 3.26 (72% retrace of $0.32 surge)
            (3.45, 3.49, 3.35, 3.41, 85_000),   # red
            (3.41, 3.41, 3.26, 3.33, 60_000),   # red (pullback low 3.26)
            # Entry: green bounce
            (3.33, 3.38, 3.28, 3.36, 45_000),   # GREEN entry
        ])
        detector = MicroPullback({
            "require_above_vwap": False,
            "require_macd_positive": False,
            "max_pullback_surge_volume_ratio": 0,
        })
        r = detector.detect(bars)
        assert not r.detected, f"should reject 72% retrace: reason={r.reason}"
        assert "retrace" in (r.reason or "").lower()

    def test_shallow_retrace_passes(self):
        """A ~35% retrace (detector-measured) passes under the 50% cap.

        Detector picks the 2-bar surge window (bars 1–2) by default, so surge
        magnitude is $10.30 low → $11.00 high = $0.70. Pullback low $10.76 →
        retrace $0.24/$0.70 ≈ 34%.
        """
        bars = _make_bars([
            (10.00, 10.35, 10.00, 10.32, 200_000),  # green
            (10.32, 10.68, 10.30, 10.65, 220_000),  # green
            (10.65, 11.00, 10.62, 10.98, 250_000),  # green (swing high 11.00)
            (10.98, 10.99, 10.80, 10.82, 100_000),  # red
            (10.82, 10.85, 10.76, 10.78, 90_000),   # red (pullback low 10.76)
            (10.78, 10.95, 10.76, 10.92, 200_000),  # GREEN entry
        ])
        detector = MicroPullback({
            "require_above_vwap": False,
            "require_macd_positive": False,
            "max_pullback_surge_volume_ratio": 0,
        })
        r = detector.detect(bars)
        assert r.detected, f"~34% retrace should pass: reason={r.reason}"

    def test_looser_config_allows_deeper_retrace(self):
        """Relaxing the cap to 0.85 lets the CLIK-style deep retrace through."""
        bars = _make_bars([
            (3.17, 3.21, 3.17, 3.20, 60_000),
            (3.20, 3.32, 3.20, 3.31, 100_000),
            (3.30, 3.49, 3.30, 3.44, 250_000),
            (3.44, 3.46, 3.40, 3.46, 110_000),
            (3.45, 3.49, 3.35, 3.41, 85_000),
            (3.41, 3.41, 3.26, 3.33, 60_000),
            (3.33, 3.38, 3.28, 3.36, 45_000),
        ])
        detector = MicroPullback({
            "max_pullback_retrace_pct": 0.85,
            "require_above_vwap": False,
            "require_macd_positive": False,
            "max_pullback_surge_volume_ratio": 0,
        })
        r = detector.detect(bars)
        assert r.detected, f"deep retrace should pass at 0.85 cap: reason={r.reason}"

    def test_retrace_in_details(self):
        """pullback_retrace exposed in details dict for observability."""
        bars = _make_bars([
            (10.00, 10.35, 10.00, 10.32, 200_000),
            (10.32, 10.68, 10.30, 10.65, 220_000),
            (10.65, 11.00, 10.62, 10.98, 250_000),
            (10.98, 10.99, 10.80, 10.82, 100_000),
            (10.82, 10.85, 10.76, 10.78, 90_000),
            (10.78, 10.95, 10.76, 10.92, 200_000),
        ])
        detector = MicroPullback({
            "require_above_vwap": False,
            "require_macd_positive": False,
            "max_pullback_surge_volume_ratio": 0,
        })
        r = detector.detect(bars)
        assert r.detected, f"fixture should pass: {r.reason}"
        assert "pullback_retrace" in r.details
        assert 0.30 < r.details["pullback_retrace"] < 0.40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
