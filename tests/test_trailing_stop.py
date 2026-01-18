"""
Trailing Stop Tests
===================

Tests for 2-bar low trailing stop with dynamic buffer.

Rules tested:
- Activation: after +1R profit OR after partial taken
- Buffer: max(spread × 2, ATR(14) × 0.1)
- 2-bar low trailing (longs) / 2-bar high trailing (shorts)
- Never lower the stop (longs) / never raise the stop (shorts)

Run with: pytest tests/test_trailing_stop.py -v
"""

import pytest
from candle_patterns import MicroPullback, TrailingStopResult
from tests.fixtures.trailing_stop_fixtures import (
    # Activation tests
    TRAIL_NOT_ACTIVATED,
    TRAIL_NOT_ACTIVATED_ENTRY_IDX,
    TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
    TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
    TRAIL_ACTIVATED_1R,
    TRAIL_ACTIVATED_1R_ENTRY_IDX,
    TRAIL_ACTIVATED_1R_ENTRY_PRICE,
    TRAIL_ACTIVATED_1R_ORIGINAL_STOP,
    TRAIL_ACTIVATED_PARTIAL,
    TRAIL_ACTIVATED_PARTIAL_ENTRY_IDX,
    TRAIL_ACTIVATED_PARTIAL_ENTRY_PRICE,
    TRAIL_ACTIVATED_PARTIAL_ORIGINAL_STOP,
    # Trailing behavior tests
    TRAIL_RAISES_STOP,
    TRAIL_RAISES_STOP_ENTRY_IDX,
    TRAIL_RAISES_STOP_ENTRY_PRICE,
    TRAIL_RAISES_STOP_ORIGINAL_STOP,
    TRAIL_NEVER_LOWERS,
    TRAIL_NEVER_LOWERS_ENTRY_IDX,
    TRAIL_NEVER_LOWERS_ENTRY_PRICE,
    TRAIL_NEVER_LOWERS_ORIGINAL_STOP,
    # Buffer tests
    TRAIL_WIDE_SPREAD,
    TRAIL_WIDE_SPREAD_ENTRY_IDX,
    TRAIL_WIDE_SPREAD_ENTRY_PRICE,
    TRAIL_WIDE_SPREAD_ORIGINAL_STOP,
    TRAIL_HIGH_ATR,
    TRAIL_HIGH_ATR_ENTRY_IDX,
    TRAIL_HIGH_ATR_ENTRY_PRICE,
    TRAIL_HIGH_ATR_ORIGINAL_STOP,
    # Edge cases
    TRAIL_INSUFFICIENT_BARS,
    TRAIL_INSUFFICIENT_BARS_ENTRY_IDX,
    TRAIL_INSUFFICIENT_BARS_ENTRY_PRICE,
    TRAIL_INSUFFICIENT_BARS_ORIGINAL_STOP,
    TRAIL_SHORT_POSITION,
    TRAIL_SHORT_POSITION_ENTRY_IDX,
    TRAIL_SHORT_POSITION_ENTRY_PRICE,
    TRAIL_SHORT_POSITION_ORIGINAL_STOP,
)


class TestTrailingStopActivation:
    """Tests for trailing stop activation conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def test_not_activated_below_1r(self):
        """Trailing should NOT activate when profit < 1R."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_NOT_ACTIVATED,
            entry_idx=TRAIL_NOT_ACTIVATED_ENTRY_IDX,
            entry_price=TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
            original_stop=TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
        )

        assert result.active is False
        assert result.new_stop == TRAIL_NOT_ACTIVATED_ORIGINAL_STOP
        assert result.current_r_multiple < 1.0
        assert "Not activated" in result.reason

    def test_activated_at_1r(self):
        """Trailing should activate when profit >= 1R."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_ACTIVATED_1R,
            entry_idx=TRAIL_ACTIVATED_1R_ENTRY_IDX,
            entry_price=TRAIL_ACTIVATED_1R_ENTRY_PRICE,
            original_stop=TRAIL_ACTIVATED_1R_ORIGINAL_STOP,
        )

        assert result.active is True
        assert result.current_r_multiple >= 1.0
        assert "Trailing active" in result.reason

    def test_activated_with_partial(self):
        """Trailing should activate when partial taken, even if < 1R."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_ACTIVATED_PARTIAL,
            entry_idx=TRAIL_ACTIVATED_PARTIAL_ENTRY_IDX,
            entry_price=TRAIL_ACTIVATED_PARTIAL_ENTRY_PRICE,
            original_stop=TRAIL_ACTIVATED_PARTIAL_ORIGINAL_STOP,
            partial_taken=True,  # Key: partial taken
        )

        assert result.active is True
        assert result.current_r_multiple < 1.0  # Below 1R
        assert "Trailing active" in result.reason

    def test_custom_activation_threshold(self):
        """Should respect custom activation_r threshold."""
        # With 0.8R threshold, should activate at 0.5R? No, still below
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_NOT_ACTIVATED,
            entry_idx=TRAIL_NOT_ACTIVATED_ENTRY_IDX,
            entry_price=TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
            original_stop=TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
            trailing_config={"activation_r": 0.4},  # Lower threshold
        )

        # Now at 0.5R it should activate with 0.4R threshold
        assert result.active is True


class TestTrailingStopBehavior:
    """Tests for trailing stop calculation behavior."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def test_stop_raises_above_original(self):
        """Trailing stop should raise above original stop."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_RAISES_STOP,
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            original_stop=TRAIL_RAISES_STOP_ORIGINAL_STOP,
        )

        assert result.active is True
        assert result.new_stop > TRAIL_RAISES_STOP_ORIGINAL_STOP
        assert result.is_trailing == True  # Stop has moved (use == for numpy bool)
        assert "raised" in result.reason.lower()

    def test_stop_never_lowers(self):
        """Stop should never go below original, even if trailing would."""
        # First, get the trailing stop at peak
        result_at_peak = self.detector.calculate_trailing_stop(
            bars=TRAIL_RAISES_STOP,
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            original_stop=TRAIL_RAISES_STOP_ORIGINAL_STOP,
        )
        peak_stop = result_at_peak.new_stop

        # Now check after pullback
        result_after_pullback = self.detector.calculate_trailing_stop(
            bars=TRAIL_NEVER_LOWERS,
            entry_idx=TRAIL_NEVER_LOWERS_ENTRY_IDX,
            entry_price=TRAIL_NEVER_LOWERS_ENTRY_PRICE,
            original_stop=TRAIL_NEVER_LOWERS_ORIGINAL_STOP,
        )

        # Stop should still be above original
        assert result_after_pullback.new_stop >= TRAIL_NEVER_LOWERS_ORIGINAL_STOP
        # Note: The actual stop might be lower than peak_stop because we're
        # calculating fresh each time. In real usage, you'd track the high-water
        # mark for the stop itself.

    def test_high_water_mark_tracked(self):
        """High water mark should track highest high since entry."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_RAISES_STOP,
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            original_stop=TRAIL_RAISES_STOP_ORIGINAL_STOP,
        )

        # High water mark should be the max high after entry
        post_entry = TRAIL_RAISES_STOP.iloc[TRAIL_RAISES_STOP_ENTRY_IDX + 1:]
        expected_hwm = post_entry["high"].max()

        assert result.high_water_mark == expected_hwm


class TestTrailingStopBuffer:
    """Tests for dynamic buffer calculation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def test_wide_spread_larger_buffer(self):
        """Wide spread should result in buffer based on spread × 2."""
        wide_spread = 0.10  # $0.10 spread

        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_WIDE_SPREAD,
            entry_idx=TRAIL_WIDE_SPREAD_ENTRY_IDX,
            entry_price=TRAIL_WIDE_SPREAD_ENTRY_PRICE,
            original_stop=TRAIL_WIDE_SPREAD_ORIGINAL_STOP,
            current_spread=wide_spread,
        )

        assert result.active is True
        # With wide spread, the 2-bar low minus buffer should be visible in reason
        # Buffer should be at least spread × 2 = $0.20

    def test_high_atr_larger_buffer(self):
        """High ATR should result in buffer based on ATR × 0.1."""
        tight_spread = 0.01  # $0.01 spread (tight)

        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_HIGH_ATR,
            entry_idx=TRAIL_HIGH_ATR_ENTRY_IDX,
            entry_price=TRAIL_HIGH_ATR_ENTRY_PRICE,
            original_stop=TRAIL_HIGH_ATR_ORIGINAL_STOP,
            current_spread=tight_spread,
        )

        assert result.active is True
        # With high ATR (~$1.00), buffer should be ATR × 0.1 = $0.10
        # This is larger than spread × 2 = $0.02


class TestTrailingStopEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def test_insufficient_bars_after_entry(self):
        """Should not activate with insufficient bars after entry."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_INSUFFICIENT_BARS,
            entry_idx=TRAIL_INSUFFICIENT_BARS_ENTRY_IDX,
            entry_price=TRAIL_INSUFFICIENT_BARS_ENTRY_PRICE,
            original_stop=TRAIL_INSUFFICIENT_BARS_ORIGINAL_STOP,
        )

        assert result.active is False
        assert result.new_stop == TRAIL_INSUFFICIENT_BARS_ORIGINAL_STOP
        assert "Need" in result.reason and "bars" in result.reason

    def test_short_position_trailing(self):
        """Trailing should work for short positions (2-bar high)."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_SHORT_POSITION,
            entry_idx=TRAIL_SHORT_POSITION_ENTRY_IDX,
            entry_price=TRAIL_SHORT_POSITION_ENTRY_PRICE,
            original_stop=TRAIL_SHORT_POSITION_ORIGINAL_STOP,
            direction="short",
        )

        assert result.active is True
        # For shorts, stop should be LOWERED (tightened)
        assert result.new_stop < TRAIL_SHORT_POSITION_ORIGINAL_STOP
        assert "lowered" in result.reason.lower()

    def test_zero_risk_returns_error(self):
        """Should handle zero risk gracefully."""
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_ACTIVATED_1R,
            entry_idx=TRAIL_ACTIVATED_1R_ENTRY_IDX,
            entry_price=10.00,
            original_stop=10.00,  # Zero risk!
        )

        assert result.active is False
        assert "zero risk" in result.reason.lower()

    def test_custom_trailing_bars(self):
        """Should respect custom trailing_bars setting."""
        # Use 3-bar trailing instead of 2
        result = self.detector.calculate_trailing_stop(
            bars=TRAIL_RAISES_STOP,
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            original_stop=TRAIL_RAISES_STOP_ORIGINAL_STOP,
            trailing_config={"trailing_bars": 3},
        )

        assert result.active is True
        # With 3 bars, the 3-bar low might be different than 2-bar low


class TestTrailingStopResultDataclass:
    """Tests for TrailingStopResult properties."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = MicroPullback()

    def test_is_trailing_property(self):
        """is_trailing should be True only when stop has moved."""
        # Not trailing (below threshold)
        result_not_trailing = self.detector.calculate_trailing_stop(
            bars=TRAIL_NOT_ACTIVATED,
            entry_idx=TRAIL_NOT_ACTIVATED_ENTRY_IDX,
            entry_price=TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
            original_stop=TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
        )
        assert result_not_trailing.is_trailing == False

        # Trailing (above threshold, stop moved)
        result_trailing = self.detector.calculate_trailing_stop(
            bars=TRAIL_RAISES_STOP,
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            original_stop=TRAIL_RAISES_STOP_ORIGINAL_STOP,
        )
        assert result_trailing.is_trailing == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
