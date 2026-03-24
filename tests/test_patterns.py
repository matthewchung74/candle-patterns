"""
Tests for Candle Patterns Library
=================================

Core pattern tests are in dedicated files:
- test_micro_pullback.py - Micro Pullback pattern tests
- test_micro_pullback.py - Micro Pullback VCR tests

This file contains:
- Exit signal tests (topping tail, stop hit, volume decline, jackknife, MACD cross, VWAP cross)
- PatternResult tests

Run with: pytest tests/test_patterns.py -v
"""

import pytest
import pandas as pd
from candle_patterns import MicroPullback
from tests.fixtures.exit_signal_fixtures import (
    # Topping Tail
    TOPPING_TAIL_VALID,
    TOPPING_TAIL_WICK_TOO_SMALL,
    TOPPING_TAIL_BODY_NOT_LOW,
    TOPPING_TAIL_NOT_IN_PROFIT,
    TOPPING_TAIL_LIMIT_WICK_RATIO,
    TOPPING_TAIL_LIMIT_BODY_POSITION,
    # Stop Hit
    STOP_HIT_VALID_EXACT,
    STOP_HIT_VALID_BELOW,
    STOP_HIT_ABOVE_STOP,
    STOP_HIT_LIMIT_MISS,
    STOP_HIT_SECOND_BAR,
    # Volume Decline
    VOLUME_DECLINE_VALID,
    VOLUME_DECLINE_NOT_ENOUGH_BARS,
    VOLUME_DECLINE_STAYS_HIGH,
    VOLUME_DECLINE_LIMIT_AT_50,
    VOLUME_DECLINE_LIMIT_AT_49,
    VOLUME_DECLINE_PRICE_RISING,
    # Jackknife
    JACKKNIFE_VALID,
    JACKKNIFE_NOT_ENOUGH_BARS,
    JACKKNIFE_NO_NEW_HIGH,
    JACKKNIFE_ABOVE_PRIOR_LOW,
    JACKKNIFE_GREEN_CANDLE,
    JACKKNIFE_LIMIT_EQUAL_HIGH,
    JACKKNIFE_LIMIT_EQUAL_LOW,
    # MACD Cross
    MACD_CROSS_VALID,
    MACD_CROSS_NOT_ENOUGH_BARS,
    MACD_CROSS_NOT_ENOUGH_AFTER_ENTRY,
    MACD_CROSS_STAYS_BULLISH,
    MACD_CROSS_BULLISH_CROSS,
    MACD_CROSS_LIMIT_EQUALS_THEN_BELOW,
    # VWAP Cross
    VWAP_CROSS_VALID,
    VWAP_CROSS_NOT_ENOUGH_AFTER_ENTRY,
    VWAP_CROSS_STAYS_ABOVE,
    VWAP_CROSS_LIMIT_EQUALS_THEN_BELOW,
    VWAP_CROSS_SHORT_VALID,
)
class TestPatternResult:
    """Tests for PatternResult dataclass."""

    def test_bool_conversion_detected(self):
        """Test that detected PatternResult is truthy."""
        detector = MicroPullback()
        # Use empty DataFrame to get a not_detected result
        result = detector.detect(pd.DataFrame())

        # Not detected should be falsy
        assert not result
        assert result.detected is False

    def test_empty_bars_rejected(self):
        """Test that empty DataFrame is rejected."""
        detector = MicroPullback()
        result = detector.detect(pd.DataFrame())

        assert result.detected is False
        assert "empty" in result.reason.lower()


class TestStopHitExit:
    """Tests for stop hit exit signal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def _get_stop_signal(self, fixture):
        """Helper to check for stop hit signal."""
        bars = fixture["bars"]
        stop_price = fixture["stop_price"]
        post_entry = bars.iloc[1:]  # Skip entry bar
        return self.detector._check_stop_hit(post_entry, stop_price)

    def test_valid_stop_hit_exact(self):
        """Test that stop is triggered when low equals stop price."""
        signal = self._get_stop_signal(STOP_HIT_VALID_EXACT)

        assert signal is not None
        assert signal.signal_type == "stop_hit"
        assert signal.triggered is True

    def test_valid_stop_hit_below(self):
        """Test that stop is triggered when low goes below stop price."""
        signal = self._get_stop_signal(STOP_HIT_VALID_BELOW)

        assert signal is not None
        assert signal.signal_type == "stop_hit"
        assert signal.triggered is True

    def test_stop_not_hit_above(self):
        """Test that stop is NOT triggered when low stays above stop."""
        signal = self._get_stop_signal(STOP_HIT_ABOVE_STOP)

        assert signal is None

    def test_limit_stop_barely_missed(self):
        """Test that stop is NOT triggered when low barely misses."""
        signal = self._get_stop_signal(STOP_HIT_LIMIT_MISS)

        assert signal is None

    def test_stop_hit_on_second_bar(self):
        """Test that stop is detected on second bar."""
        signal = self._get_stop_signal(STOP_HIT_SECOND_BAR)

        assert signal is not None
        assert signal.signal_type == "stop_hit"
        assert signal.triggered is True


class TestVolumDeclineExit:
    """Tests for volume decline exit signal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def _get_volume_signal(self, fixture):
        """Helper to check for volume decline signal."""
        bars = fixture["bars"]
        entry_idx = fixture["entry_idx"]
        return self.detector._check_volume_decline(bars, entry_idx)

    def test_valid_volume_decline(self):
        """Test that volume decline triggers at 40%."""
        signal = self._get_volume_signal(VOLUME_DECLINE_VALID)

        assert signal is not None
        assert signal.signal_type == "volume_decline"
        assert signal.triggered is True

    def test_not_enough_bars_after_entry(self):
        """Test that signal is NOT triggered with < 3 bars after entry."""
        signal = self._get_volume_signal(VOLUME_DECLINE_NOT_ENOUGH_BARS)

        assert signal is None

    def test_volume_stays_high(self):
        """Test that signal is NOT triggered when volume stays at 60%."""
        signal = self._get_volume_signal(VOLUME_DECLINE_STAYS_HIGH)

        assert signal is None

    def test_limit_volume_at_50_percent(self):
        """Test that 50% volume does NOT trigger (needs < 50%)."""
        signal = self._get_volume_signal(VOLUME_DECLINE_LIMIT_AT_50)

        assert signal is None

    def test_limit_volume_at_49_percent(self):
        """Test that 49% volume triggers when price is stalling."""
        signal = self._get_volume_signal(VOLUME_DECLINE_LIMIT_AT_49)

        assert signal is not None
        assert signal.signal_type == "volume_decline"
        assert signal.triggered is True

    def test_low_volume_but_price_rising(self):
        """Test that low volume does NOT trigger when price is still rising."""
        signal = self._get_volume_signal(VOLUME_DECLINE_PRICE_RISING)

        assert signal is None  # Price rising = no exit despite low volume


class TestMACDCrossExit:
    """Tests for MACD cross exit signal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def _get_macd_signal(self, fixture):
        """Helper to check for MACD cross signal."""
        bars = fixture["bars"]
        entry_idx = fixture["entry_idx"]
        return self.detector._check_macd_cross(bars, entry_idx)

    def test_valid_macd_cross(self):
        """Test that bearish MACD crossover triggers."""
        signal = self._get_macd_signal(MACD_CROSS_VALID)

        assert signal is not None
        assert signal.signal_type == "macd_cross"
        assert signal.triggered is True

    def test_not_enough_bars_for_macd(self):
        """Test that signal is NOT triggered with < 35 bars."""
        signal = self._get_macd_signal(MACD_CROSS_NOT_ENOUGH_BARS)

        assert signal is None

    def test_not_enough_bars_after_entry(self):
        """Test that signal is NOT triggered with < 2 bars after entry."""
        signal = self._get_macd_signal(MACD_CROSS_NOT_ENOUGH_AFTER_ENTRY)

        assert signal is None

    def test_macd_stays_bullish(self):
        """Test that signal is NOT triggered when MACD stays above signal."""
        signal = self._get_macd_signal(MACD_CROSS_STAYS_BULLISH)

        assert signal is None

    def test_bullish_cross_wrong_direction(self):
        """Test that bullish cross (wrong direction) does NOT trigger."""
        signal = self._get_macd_signal(MACD_CROSS_BULLISH_CROSS)

        assert signal is None

    def test_limit_equals_then_below(self):
        """Test that MACD equaling signal then going below triggers."""
        signal = self._get_macd_signal(MACD_CROSS_LIMIT_EQUALS_THEN_BELOW)

        assert signal is not None
        assert signal.signal_type == "macd_cross"
        assert signal.triggered is True


class TestVWAPCrossExit:
    """Tests for VWAP cross exit signal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def _get_vwap_signal(self, fixture, direction="long"):
        """Helper to check for VWAP cross signal."""
        bars = fixture["bars"]
        vwap = fixture["vwap"]
        entry_idx = fixture["entry_idx"]
        return self.detector._check_vwap_cross(bars, entry_idx, vwap, direction)

    def test_valid_vwap_cross_long(self):
        """Test that price crossing below VWAP triggers exit for longs."""
        signal = self._get_vwap_signal(VWAP_CROSS_VALID)

        assert signal is not None
        assert signal.signal_type == "vwap_cross"
        assert signal.triggered is True
        assert "below VWAP" in signal.reason

    def test_not_enough_bars_after_entry(self):
        """Test that signal is NOT triggered with insufficient bars after entry."""
        signal = self._get_vwap_signal(VWAP_CROSS_NOT_ENOUGH_AFTER_ENTRY)

        assert signal is None

    def test_price_stays_above_vwap(self):
        """Test that signal is NOT triggered when price stays above VWAP."""
        signal = self._get_vwap_signal(VWAP_CROSS_STAYS_ABOVE)

        assert signal is None

    def test_limit_equals_then_below(self):
        """Test that price equaling VWAP then going below triggers."""
        signal = self._get_vwap_signal(VWAP_CROSS_LIMIT_EQUALS_THEN_BELOW)

        assert signal is not None
        assert signal.signal_type == "vwap_cross"
        assert signal.triggered is True

    def test_valid_vwap_cross_short(self):
        """Test that price crossing above VWAP triggers exit for shorts."""
        signal = self._get_vwap_signal(VWAP_CROSS_SHORT_VALID, direction="short")

        assert signal is not None
        assert signal.signal_type == "vwap_cross"
        assert signal.triggered is True
        assert "above VWAP" in signal.reason

    def test_vwap_in_check_exit_signals(self):
        """Test that VWAP cross is included when calling check_exit_signals with vwap."""
        fixture = VWAP_CROSS_VALID
        bars = fixture["bars"]
        vwap = fixture["vwap"]
        entry_idx = fixture["entry_idx"]

        signals = self.detector.check_exit_signals(
            bars=bars,
            entry_idx=entry_idx,
            entry_price=10.20,
            stop_price=9.50,
            direction="long",
            vwap=vwap,
        )

        vwap_signals = [s for s in signals if s.signal_type == "vwap_cross"]
        assert len(vwap_signals) == 1
        assert vwap_signals[0].triggered is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
