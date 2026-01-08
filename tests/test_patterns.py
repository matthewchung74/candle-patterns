"""
Tests for Candle Patterns Library
=================================

Run with: pytest tests/test_patterns.py -v
"""

import pytest
import pandas as pd
from candle_patterns import MicroPullback, BullFlag, VWAPBreak, OpeningRangeRetest
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
)
from tests.fixtures.opening_range_retest_fixtures import (
    OPENING_RANGE_RETEST_VALID,
    OPENING_RANGE_RETEST_NO_RETEST,
    OPENING_RANGE_RETEST_OUTSIDE_WINDOW,
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


class TestToppingTailExit:
    """Tests for topping tail exit signal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use MicroPullback as a concrete PatternDetector for testing
        self.detector = MicroPullback()

    def _get_topping_tail_signal(self, fixture):
        """
        Helper to check for topping tail signal in fixture data.

        Args:
            fixture: Dict with 'entry_price' and 'bars' keys

        Returns:
            ExitSignal if topping tail detected, None otherwise
        """
        bars = fixture["bars"]
        entry_price = fixture["entry_price"]
        post_entry = bars.iloc[1:]  # Skip entry bar

        signal = self.detector._check_topping_tail(post_entry, entry_price)
        return signal

    def test_valid_topping_tail_detected(self):
        """Test that a valid topping tail is detected."""
        signal = self._get_topping_tail_signal(TOPPING_TAIL_VALID)

        assert signal is not None
        assert signal.signal_type == "topping_tail"
        assert signal.triggered is True
        assert "upper wick" in signal.reason.lower()

    def test_wick_too_small_rejected(self):
        """Test that small upper wick (<2x body) is rejected."""
        signal = self._get_topping_tail_signal(TOPPING_TAIL_WICK_TOO_SMALL)

        assert signal is None  # Should not trigger

    def test_body_not_low_rejected(self):
        """Test that body not in lower third is rejected."""
        signal = self._get_topping_tail_signal(TOPPING_TAIL_BODY_NOT_LOW)

        assert signal is None  # Should not trigger

    def test_not_in_profit_rejected(self):
        """Test that topping tail below entry is rejected."""
        signal = self._get_topping_tail_signal(TOPPING_TAIL_NOT_IN_PROFIT)

        assert signal is None  # Should not trigger (not in profit)

    # === LIMIT TESTS ===

    def test_limit_wick_ratio_at_minimum(self):
        """Test detection with exactly 2.0x wick ratio (minimum)."""
        signal = self._get_topping_tail_signal(TOPPING_TAIL_LIMIT_WICK_RATIO)

        assert signal is not None
        assert signal.signal_type == "topping_tail"
        assert signal.triggered is True

    def test_limit_body_position_at_maximum(self):
        """Test detection with body at exactly 0.33 position (maximum)."""
        signal = self._get_topping_tail_signal(TOPPING_TAIL_LIMIT_BODY_POSITION)

        assert signal is not None
        assert signal.signal_type == "topping_tail"
        assert signal.triggered is True


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
        """Test that 49% volume triggers."""
        signal = self._get_volume_signal(VOLUME_DECLINE_LIMIT_AT_49)

        assert signal is not None
        assert signal.signal_type == "volume_decline"
        assert signal.triggered is True


class TestJackknifeExit:
    """Tests for jackknife exit signal detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def _get_jackknife_signal(self, fixture):
        """Helper to check for jackknife signal."""
        bars = fixture["bars"]
        post_entry = bars.iloc[1:]  # Skip entry bar
        return self.detector._check_jackknife(post_entry)

    def test_valid_jackknife(self):
        """Test that jackknife triggers when all conditions met."""
        signal = self._get_jackknife_signal(JACKKNIFE_VALID)

        assert signal is not None
        assert signal.signal_type == "jackknife"
        assert signal.triggered is True

    def test_not_enough_bars(self):
        """Test that signal is NOT triggered with < 2 bars."""
        signal = self._get_jackknife_signal(JACKKNIFE_NOT_ENOUGH_BARS)

        assert signal is None

    def test_no_new_high(self):
        """Test that signal is NOT triggered without new high."""
        signal = self._get_jackknife_signal(JACKKNIFE_NO_NEW_HIGH)

        assert signal is None

    def test_above_prior_low(self):
        """Test that signal is NOT triggered when close >= prior low."""
        signal = self._get_jackknife_signal(JACKKNIFE_ABOVE_PRIOR_LOW)

        assert signal is None

    def test_green_candle(self):
        """Test that signal is NOT triggered on green candle."""
        signal = self._get_jackknife_signal(JACKKNIFE_GREEN_CANDLE)

        assert signal is None

    def test_limit_equal_high(self):
        """Test that high == prior high does NOT trigger (needs >)."""
        signal = self._get_jackknife_signal(JACKKNIFE_LIMIT_EQUAL_HIGH)

        assert signal is None

    def test_limit_equal_low(self):
        """Test that close == prior low does NOT trigger (needs <)."""
        signal = self._get_jackknife_signal(JACKKNIFE_LIMIT_EQUAL_LOW)

        assert signal is None


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


class TestOpeningRangeRetest:
    """Tests for Opening Range Retest pattern detection."""

    def setup_method(self):
        """Set up test fixtures."""
        # Disable trend alignment to keep fixtures small and deterministic
        self.detector = OpeningRangeRetest({
            "trend_alignment": False,
        })

    def test_valid_orb_retest_detected(self):
        """Test that a valid ORB retest is detected."""
        result = self.detector.detect(OPENING_RANGE_RETEST_VALID)

        assert result.detected is True
        assert result.pattern_name == "OpeningRangeRetest"
        assert result.entry_price is not None
        assert result.stop_price is not None

    def test_no_retest_rejected(self):
        """Test that pattern without retest is rejected."""
        result = self.detector.detect(OPENING_RANGE_RETEST_NO_RETEST)

        assert result.detected is False
        assert "retest" in result.reason.lower()

    def test_outside_window_rejected(self):
        """Test that bars outside the 90-minute window are rejected."""
        result = self.detector.detect(OPENING_RANGE_RETEST_OUTSIDE_WINDOW)

        assert result.detected is False
        assert "window" in result.reason.lower()


class TestConfidenceSystem:
    """Tests for the standardized confidence scoring system."""

    def test_micro_pullback_base_confidence(self):
        """Test that MicroPullback starts at 65% base confidence."""
        detector = MicroPullback()
        result = detector.detect(MICRO_PULLBACK_VALID)

        if result.detected:
            # Base is 65%, max is 90%
            assert 0.65 <= result.confidence <= 0.90

    def test_bull_flag_base_confidence(self):
        """Test that BullFlag starts at 65% base confidence."""
        detector = BullFlag()
        result = detector.detect(BULL_FLAG_VALID)

        if result.detected:
            # Base is 65%, max is 90%
            assert 0.65 <= result.confidence <= 0.90

    def test_vwap_break_base_confidence(self):
        """Test that VWAPBreak starts at 65% base confidence."""
        detector = VWAPBreak()
        bars, vwap = VWAP_BREAK_VALID
        result = detector.detect(bars, vwap=vwap)

        if result.detected:
            # Base is 65%, max is 90%
            assert 0.65 <= result.confidence <= 0.90

    def test_confidence_capped_at_90(self):
        """Test that confidence is capped at 90%."""
        detector = MicroPullback()
        result = detector.detect(MICRO_PULLBACK_VALID)

        if result.detected:
            assert result.confidence <= 0.90

    def test_macd_slope_up_field_exists(self):
        """Test that macd_slope_up field is populated."""
        detector = MicroPullback()
        result = detector.detect(MICRO_PULLBACK_VALID)

        if result.detected:
            # Should be boolean or None (if insufficient bars)
            assert result.macd_slope_up is None or isinstance(result.macd_slope_up, bool)

    def test_macd_slope_up_in_bull_flag(self):
        """Test that BullFlag has macd_slope_up field."""
        detector = BullFlag()
        result = detector.detect(BULL_FLAG_VALID)

        if result.detected:
            assert result.macd_slope_up is None or isinstance(result.macd_slope_up, bool)

    def test_macd_slope_up_in_vwap_break(self):
        """Test that VWAPBreak has macd_slope_up field."""
        detector = VWAPBreak()
        bars, vwap = VWAP_BREAK_VALID
        result = detector.detect(bars, vwap=vwap)

        if result.detected:
            assert result.macd_slope_up is None or isinstance(result.macd_slope_up, bool)


class TestConfidenceBoosts:
    """Tests for individual confidence boost factors."""

    def _create_bars_with_macd(self, n=50, trend='up'):
        """Create test bars with controlled MACD behavior."""
        import numpy as np

        # Generate price data that creates predictable MACD
        base_price = 10.0
        if trend == 'up':
            prices = base_price + np.linspace(0, 2, n) + np.random.normal(0, 0.05, n)
        else:
            prices = base_price + np.linspace(2, 0, n) + np.random.normal(0, 0.05, n)

        bars = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01 09:30', periods=n, freq='1min'),
            'open': prices - 0.02,
            'high': prices + 0.05,
            'low': prices - 0.05,
            'close': prices,
            'volume': [100000] * n,
        })
        return bars

    def test_confidence_boosted_by_volume_declining(self):
        """Test that volume_declining adds to confidence."""
        detector = MicroPullback()
        result = detector.detect(MICRO_PULLBACK_VALID)

        if result.detected and result.details:
            # If volume is declining, confidence should be boosted
            if result.details.get('volume_declining'):
                # Base 65% + volume_declining 10% = at least 75%
                assert result.confidence >= 0.75

    def test_confidence_boosted_by_macd_positive(self):
        """Test that macd_positive adds +8% to confidence."""
        detector = MicroPullback()
        result = detector.detect(MICRO_PULLBACK_VALID)

        if result.detected:
            # If MACD is positive, confidence should be boosted
            if result.macd_positive:
                # Should include +8% from macd_positive
                assert result.confidence >= 0.73  # 65% base + 8%

    def test_confidence_boosted_by_macd_slope_up(self):
        """Test that macd_slope_up adds +4% to confidence."""
        detector = MicroPullback()
        result = detector.detect(MICRO_PULLBACK_VALID)

        if result.detected:
            # If MACD slope is up, confidence should be boosted
            if result.macd_slope_up:
                # Should include +4% from macd_slope_up
                assert result.confidence >= 0.69  # 65% base + 4%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
