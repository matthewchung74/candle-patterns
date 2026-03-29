"""
Tests for ParabolicExhaustion pattern detector.
"""

import pytest
import pandas as pd

from candle_patterns import ParabolicExhaustion
from tests.fixtures.parabolic_exhaustion_fixtures import (
    PE_PASS_CLASSIC, PE_PASS_CLASSIC_PREV_CLOSE,
    PE_PASS_TOPPING_TAIL, PE_PASS_TOPPING_TAIL_PREV_CLOSE,
    PE_PASS_LOWER_HIGH, PE_PASS_LOWER_HIGH_PREV_CLOSE,
    PE_FAIL_NOT_EXTENDED, PE_FAIL_NOT_EXTENDED_PREV_CLOSE,
    PE_FAIL_NO_SURGE, PE_FAIL_NO_SURGE_PREV_CLOSE,
    PE_FAIL_LOW_VOLUME, PE_FAIL_LOW_VOLUME_PREV_CLOSE,
    PE_FAIL_NO_REJECTION, PE_FAIL_NO_REJECTION_PREV_CLOSE,
    PE_FAIL_STALE_HOD, PE_FAIL_STALE_HOD_PREV_CLOSE,
    PE_FAIL_ENTRY_GREEN, PE_FAIL_ENTRY_GREEN_PREV_CLOSE,
)


@pytest.fixture
def detector():
    return ParabolicExhaustion()


class TestDetectionPass:
    """Test cases where pattern should be detected."""

    def test_classic_blowoff_top(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.detected
        assert result.pattern_name == "ParabolicExhaustion"
        assert result.details["direction"] == "short"
        assert result.entry_price is not None
        assert result.stop_price is not None
        assert result.stop_price > result.entry_price  # Stop above entry for shorts

    def test_topping_tail_rejection(self, detector):
        result = detector.detect(PE_PASS_TOPPING_TAIL, prev_close=PE_PASS_TOPPING_TAIL_PREV_CLOSE)
        assert result.detected
        assert result.details["direction"] == "short"
        assert result.details["rejection_type"] == "topping_tail"

    def test_lower_high_rejection(self, detector):
        result = detector.detect(PE_PASS_LOWER_HIGH, prev_close=PE_PASS_LOWER_HIGH_PREV_CLOSE)
        assert result.detected
        assert result.details["direction"] == "short"
        assert result.details["rejection_type"] == "lower_high"

    def test_classic_details_populated(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.detected
        d = result.details
        assert d["extension_from_open_pct"] >= 25.0
        assert d["strong_green_count"] >= 3
        assert d["climax_volume_ratio"] >= 3.0
        assert d["pattern_high"] > 0
        assert d["atr"] is not None
        assert d["stop_buffer"] > 0

    def test_stop_above_hod(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.detected
        assert result.stop_price > result.details["hod"]


class TestDetectionFail:
    """Test cases where pattern should NOT be detected."""

    def test_not_extended(self, detector):
        result = detector.detect(PE_FAIL_NOT_EXTENDED, prev_close=PE_FAIL_NOT_EXTENDED_PREV_CLOSE)
        assert not result.detected
        assert "Not extended" in result.reason

    def test_no_surge(self, detector):
        result = detector.detect(PE_FAIL_NO_SURGE, prev_close=PE_FAIL_NO_SURGE_PREV_CLOSE)
        assert not result.detected
        # Should fail on surge (no strong green bars) or volume climax

    def test_low_volume(self, detector):
        result = detector.detect(PE_FAIL_LOW_VOLUME, prev_close=PE_FAIL_LOW_VOLUME_PREV_CLOSE)
        assert not result.detected
        assert "volume" in result.reason.lower() or "climax" in result.reason.lower()

    def test_no_rejection(self, detector):
        result = detector.detect(PE_FAIL_NO_REJECTION, prev_close=PE_FAIL_NO_REJECTION_PREV_CLOSE)
        assert not result.detected
        # Should fail on rejection or entry (last bar is green)

    def test_stale_hod(self, detector):
        result = detector.detect(PE_FAIL_STALE_HOD, prev_close=PE_FAIL_STALE_HOD_PREV_CLOSE)
        assert not result.detected
        assert "stale" in result.reason.lower() or "HOD" in result.reason

    def test_entry_bar_green(self, detector):
        result = detector.detect(PE_FAIL_ENTRY_GREEN, prev_close=PE_FAIL_ENTRY_GREEN_PREV_CLOSE)
        assert not result.detected
        assert "green" in result.reason.lower() or "red" in result.reason.lower()


class TestConfigOverrides:
    """Test that config overrides work correctly."""

    def test_stricter_extension_rejects(self):
        detector = ParabolicExhaustion(config={"min_extension_from_open_pct": 50.0})
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert not result.detected
        assert "Not extended" in result.reason

    def test_lower_volume_threshold_accepts(self):
        detector = ParabolicExhaustion(config={"volume_climax_multiplier": 1.0})
        result = detector.detect(PE_FAIL_LOW_VOLUME, prev_close=PE_FAIL_LOW_VOLUME_PREV_CLOSE)
        # With lower threshold, volume should pass (but may fail on other gates)
        # At minimum, the failure reason should NOT be about volume
        if not result.detected:
            assert "volume" not in result.reason.lower() or "climax" not in result.reason.lower()

    def test_default_config_values(self, detector):
        assert detector.config["min_extension_from_open_pct"] == 25.0
        assert detector.config["min_strong_green_bars"] == 3
        assert detector.config["volume_climax_multiplier"] == 3.0
        assert detector.config["max_hod_age_bars"] == 8
        assert detector.config["rejection_window"] == 3
        assert detector.config["max_entry_delay_bars"] == 3
        assert detector.config["min_bars_required"] == 15


class TestStopCalculation:
    """Test stop placement logic."""

    def test_stop_above_entry(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.detected
        assert result.stop_price > result.entry_price

    def test_stop_distance_in_range(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.detected
        stop_pct = (result.stop_price - result.entry_price) / result.entry_price * 100
        assert stop_pct <= 15.0  # max_stop_distance_pct
        assert result.stop_distance_cents >= 5  # min_stop_distance_cents


class TestConfidence:
    """Test confidence scoring."""

    def test_confidence_in_range(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.detected
        assert 0.65 <= result.confidence <= 0.90

    def test_confidence_capped_at_90(self, detector):
        result = detector.detect(PE_PASS_CLASSIC, prev_close=PE_PASS_CLASSIC_PREV_CLOSE)
        assert result.confidence <= 0.90


class TestEdgeCases:
    """Test edge cases."""

    def test_insufficient_bars(self):
        detector = ParabolicExhaustion()
        short_bars = pd.DataFrame({
            "open": [1.0] * 5,
            "high": [1.1] * 5,
            "low": [0.9] * 5,
            "close": [1.0] * 5,
            "volume": [1000] * 5,
        })
        result = detector.detect(short_bars, prev_close=0.5)
        assert not result.detected
        assert "Insufficient" in result.reason

    def test_no_prev_close_uses_first_open(self, detector):
        # Should still work using first bar's open as reference
        result = detector.detect(PE_PASS_CLASSIC, prev_close=None)
        # With prev_close=None, uses first bar open ($4.00) same as fixture prev_close
        assert result.detected

    def test_halt_bar_rejects(self, detector):
        """Halt bars (zero volume) should cause rejection."""
        from tests.fixtures.parabolic_exhaustion_fixtures import _make_bars
        bars = _make_bars([
            (3.98, 4.02, 3.95, 4.00, 45000),
            (4.00, 4.05, 3.95, 4.02, 50000),
            (4.02, 4.08, 3.98, 4.05, 48000),
            (4.05, 4.10, 4.00, 4.03, 52000),
            (4.03, 4.07, 3.99, 4.04, 47000),
            (4.04, 4.08, 4.00, 4.06, 51000),
            (4.06, 4.10, 4.02, 4.05, 49000),
            (4.05, 4.09, 4.01, 4.07, 53000),
            (4.07, 4.12, 4.03, 4.08, 50000),
            (4.08, 4.25, 4.06, 4.20, 80000),
            (4.20, 4.45, 4.18, 4.42, 0),       # HALT BAR
            (4.42, 4.75, 4.40, 4.70, 120000),
            (4.70, 5.10, 4.68, 5.05, 150000),
            (5.05, 5.55, 5.00, 5.40, 280000),
            (5.40, 5.42, 5.10, 5.15, 200000),
        ])
        result = detector.detect(bars, prev_close=4.00)
        assert not result.detected
        assert "Halt" in result.reason or "halt" in result.reason
