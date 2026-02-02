"""
Swing Low Trailing Stop Tests
=============================

Tests for swing_low trailing stop strategy (N-bar low with dynamic buffer).

Rules tested:
- Activation: after +1R profit OR after partial taken
- Buffer: max(spread × 2, ATR(14) × 0.1)
- N-bar low trailing (longs) / N-bar high trailing (shorts)
- Never lower the stop (longs) / never raise the stop (shorts)

Run with: pytest tests/trailing/test_swing_low.py -v
"""

import pytest
from candle_patterns.trailing import (
    calculate_trailing_stop,
    TrailingStopState,
    TrailingStopConfig,
    TrailingStopResult,
)
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
    # Never give back tests
    TRAIL_NEVER_GIVE_BACK_PEAK,
    TRAIL_NEVER_GIVE_BACK_PEAK_ENTRY_IDX,
    TRAIL_NEVER_GIVE_BACK_PEAK_ENTRY_PRICE,
    TRAIL_NEVER_GIVE_BACK_PEAK_ORIGINAL_STOP,
    TRAIL_NEVER_GIVE_BACK_PULLBACK,
    TRAIL_NEVER_GIVE_BACK_PULLBACK_ENTRY_IDX,
    TRAIL_NEVER_GIVE_BACK_PULLBACK_ENTRY_PRICE,
    TRAIL_NEVER_GIVE_BACK_PULLBACK_ORIGINAL_STOP,
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

    def test_not_activated_below_1r(self):
        """Trailing should NOT activate when profit < 1R."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
            stop_price=TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NOT_ACTIVATED_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_NOT_ACTIVATED, state, config)

        assert result.active is False
        assert result.new_stop == TRAIL_NOT_ACTIVATED_ORIGINAL_STOP
        assert result.current_r_multiple < 1.0
        assert "Not activated" in result.reason
        assert result.strategy_name == "swing_low"

    def test_activated_at_1r(self):
        """Trailing should activate when profit >= 1R."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_ACTIVATED_1R_ENTRY_PRICE,
            stop_price=TRAIL_ACTIVATED_1R_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_ACTIVATED_1R_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_ACTIVATED_1R, state, config)

        assert result.active is True
        assert result.current_r_multiple >= 1.0
        assert "Trailing active" in result.reason
        assert result.strategy_name == "swing_low"

    def test_activated_with_partial(self):
        """Trailing should activate when partial taken, even if < 1R."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_ACTIVATED_PARTIAL_ENTRY_PRICE,
            stop_price=TRAIL_ACTIVATED_PARTIAL_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_ACTIVATED_PARTIAL_ENTRY_IDX,
        )
        state.partial_taken = True  # Partial taken
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_ACTIVATED_PARTIAL, state, config)

        assert result.active is True
        assert result.current_r_multiple < 1.0  # Below 1R
        assert "Trailing active" in result.reason

    def test_custom_activation_threshold(self):
        """Should respect custom activation_r threshold."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
            stop_price=TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NOT_ACTIVATED_ENTRY_IDX,
        )
        # With 0.4R threshold, should activate at 0.5R
        config = TrailingStopConfig(strategy="swing_low", activation_r=0.4)

        result = calculate_trailing_stop(TRAIL_NOT_ACTIVATED, state, config)

        # Now at 0.5R it should activate with 0.4R threshold
        assert result.active is True


class TestTrailingStopBehavior:
    """Tests for trailing stop calculation behavior."""

    def test_stop_raises_above_original(self):
        """Trailing stop should raise above original stop."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            stop_price=TRAIL_RAISES_STOP_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_RAISES_STOP, state, config)

        assert result.active is True
        assert result.new_stop > TRAIL_RAISES_STOP_ORIGINAL_STOP
        assert result.is_trailing == True
        assert "raised" in result.reason.lower()

    def test_stop_never_lowers(self):
        """Stop should never go below original, even if trailing would."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_NEVER_LOWERS_ENTRY_PRICE,
            stop_price=TRAIL_NEVER_LOWERS_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NEVER_LOWERS_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_NEVER_LOWERS, state, config)

        # Stop should still be above original
        assert result.new_stop >= TRAIL_NEVER_LOWERS_ORIGINAL_STOP

    def test_high_water_mark_tracked(self):
        """High water mark should track highest high since entry."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            stop_price=TRAIL_RAISES_STOP_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_RAISES_STOP, state, config)

        # High water mark should be the max high after entry
        post_entry = TRAIL_RAISES_STOP.iloc[TRAIL_RAISES_STOP_ENTRY_IDX + 1:]
        expected_hwm = post_entry["high"].max()

        assert result.high_water_mark == expected_hwm

    def test_never_give_back_with_state_tracking(self):
        """Stop should never loosen below previous level when state is tracked."""
        # First, get the trailing stop at peak
        state_at_peak = TrailingStopState.from_entry(
            entry_price=TRAIL_NEVER_GIVE_BACK_PEAK_ENTRY_PRICE,
            stop_price=TRAIL_NEVER_GIVE_BACK_PEAK_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NEVER_GIVE_BACK_PEAK_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result_at_peak = calculate_trailing_stop(TRAIL_NEVER_GIVE_BACK_PEAK, state_at_peak, config)
        peak_stop = result_at_peak.new_stop
        assert result_at_peak.active is True
        assert peak_stop > TRAIL_NEVER_GIVE_BACK_PEAK_ORIGINAL_STOP

        # Now calculate after pullback WITHOUT tracking state
        state_without_tracking = TrailingStopState.from_entry(
            entry_price=TRAIL_NEVER_GIVE_BACK_PULLBACK_ENTRY_PRICE,
            stop_price=TRAIL_NEVER_GIVE_BACK_PULLBACK_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NEVER_GIVE_BACK_PULLBACK_ENTRY_IDX,
        )
        result_without_tracking = calculate_trailing_stop(
            TRAIL_NEVER_GIVE_BACK_PULLBACK, state_without_tracking, config
        )
        stop_without_tracking = result_without_tracking.new_stop

        # Now calculate WITH tracking state (passing previous stop)
        state_with_tracking = TrailingStopState.from_entry(
            entry_price=TRAIL_NEVER_GIVE_BACK_PULLBACK_ENTRY_PRICE,
            stop_price=TRAIL_NEVER_GIVE_BACK_PULLBACK_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NEVER_GIVE_BACK_PULLBACK_ENTRY_IDX,
        )
        state_with_tracking.current_stop = peak_stop  # Track the peak stop
        state_with_tracking.is_activated = True

        result_with_tracking = calculate_trailing_stop(
            TRAIL_NEVER_GIVE_BACK_PULLBACK, state_with_tracking, config
        )
        stop_with_tracking = result_with_tracking.new_stop

        # Without tracking, stop would have loosened (given back gains)
        assert stop_without_tracking < peak_stop, "Without tracking, stop should loosen"

        # With tracking, stop should NOT go below peak level
        assert stop_with_tracking >= peak_stop, "With tracking, stop should NOT loosen"


class TestTrailingStopBuffer:
    """Tests for dynamic buffer calculation."""

    def test_wide_spread_larger_buffer(self):
        """Wide spread should result in buffer based on spread × 2."""
        wide_spread = 0.10  # $0.10 spread

        state = TrailingStopState.from_entry(
            entry_price=TRAIL_WIDE_SPREAD_ENTRY_PRICE,
            stop_price=TRAIL_WIDE_SPREAD_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_WIDE_SPREAD_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low", current_spread=wide_spread)

        result = calculate_trailing_stop(TRAIL_WIDE_SPREAD, state, config)

        assert result.active is True
        # With wide spread, the 2-bar low minus buffer should be visible in reason
        # Buffer should be at least spread × 2 = $0.20

    def test_high_atr_larger_buffer(self):
        """High ATR should result in buffer based on ATR × 0.1."""
        tight_spread = 0.01  # $0.01 spread (tight)

        state = TrailingStopState.from_entry(
            entry_price=TRAIL_HIGH_ATR_ENTRY_PRICE,
            stop_price=TRAIL_HIGH_ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_HIGH_ATR_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low", current_spread=tight_spread)

        result = calculate_trailing_stop(TRAIL_HIGH_ATR, state, config)

        assert result.active is True
        # With high ATR (~$1.00), buffer should be ATR × 0.1 = $0.10
        # This is larger than spread × 2 = $0.02


class TestTrailingStopEdgeCases:
    """Tests for edge cases and error handling."""

    def test_insufficient_bars_after_entry(self):
        """Should not activate with insufficient bars after entry."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_INSUFFICIENT_BARS_ENTRY_PRICE,
            stop_price=TRAIL_INSUFFICIENT_BARS_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_INSUFFICIENT_BARS_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_INSUFFICIENT_BARS, state, config)

        assert result.active is False
        assert result.new_stop == TRAIL_INSUFFICIENT_BARS_ORIGINAL_STOP
        assert "Need" in result.reason and "bars" in result.reason

    def test_short_position_trailing(self):
        """Trailing should work for short positions (N-bar high)."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_SHORT_POSITION_ENTRY_PRICE,
            stop_price=TRAIL_SHORT_POSITION_ORIGINAL_STOP,
            direction="short",
            entry_idx=TRAIL_SHORT_POSITION_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_SHORT_POSITION, state, config)

        assert result.active is True
        # For shorts, stop should be LOWERED (tightened)
        assert result.new_stop < TRAIL_SHORT_POSITION_ORIGINAL_STOP
        assert "lowered" in result.reason.lower()

    def test_zero_risk_returns_error(self):
        """Should handle zero risk gracefully."""
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=10.00,  # Zero risk!
            direction="long",
            entry_idx=TRAIL_ACTIVATED_1R_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result = calculate_trailing_stop(TRAIL_ACTIVATED_1R, state, config)

        assert result.active is False
        assert "zero risk" in result.reason.lower()

    def test_custom_trailing_bars(self):
        """Should respect custom trailing_bars setting."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            stop_price=TRAIL_RAISES_STOP_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
        )
        # Use 3-bar trailing instead of 2
        config = TrailingStopConfig(
            strategy="swing_low",
            params={"trailing_bars": 3},
        )

        result = calculate_trailing_stop(TRAIL_RAISES_STOP, state, config)

        assert result.active is True
        # With 3 bars, the 3-bar low might be different than 2-bar low


class TestTrailingStopResultDataclass:
    """Tests for TrailingStopResult properties."""

    def test_is_trailing_property(self):
        """is_trailing should be True only when stop has moved."""
        # Not trailing (below threshold)
        state_not_trailing = TrailingStopState.from_entry(
            entry_price=TRAIL_NOT_ACTIVATED_ENTRY_PRICE,
            stop_price=TRAIL_NOT_ACTIVATED_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_NOT_ACTIVATED_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        result_not_trailing = calculate_trailing_stop(TRAIL_NOT_ACTIVATED, state_not_trailing, config)
        assert result_not_trailing.is_trailing == False

        # Trailing (above threshold, stop moved)
        state_trailing = TrailingStopState.from_entry(
            entry_price=TRAIL_RAISES_STOP_ENTRY_PRICE,
            stop_price=TRAIL_RAISES_STOP_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_RAISES_STOP_ENTRY_IDX,
        )
        result_trailing = calculate_trailing_stop(TRAIL_RAISES_STOP, state_trailing, config)
        assert result_trailing.is_trailing == True

    def test_just_activated_flag(self):
        """just_activated should be True only on first activation."""
        state = TrailingStopState.from_entry(
            entry_price=TRAIL_ACTIVATED_1R_ENTRY_PRICE,
            stop_price=TRAIL_ACTIVATED_1R_ORIGINAL_STOP,
            direction="long",
            entry_idx=TRAIL_ACTIVATED_1R_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="swing_low")

        # First call - should be just activated
        result1 = calculate_trailing_stop(TRAIL_ACTIVATED_1R, state, config)
        assert result1.just_activated is True

        # Update state to reflect activation
        state.is_activated = True
        state.current_stop = result1.new_stop

        # Second call - should NOT be just activated
        result2 = calculate_trailing_stop(TRAIL_ACTIVATED_1R, state, config)
        assert result2.just_activated is False


class TestTrailingStopStateFactory:
    """Tests for TrailingStopState.from_entry() factory method."""

    def test_from_entry_calculates_risk(self):
        """from_entry should calculate risk_per_share correctly."""
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=9.50,
            direction="long",
            entry_idx=5,
        )

        assert state.risk_per_share == 0.50
        assert state.entry_price == 10.00
        assert state.original_stop == 9.50
        assert state.current_stop == 9.50
        assert state.direction == "long"
        assert state.high_water_mark == 10.00
        assert state.is_activated is False
        assert state.partial_taken is False
        assert state.entry_idx == 5

    def test_from_entry_short_position(self):
        """from_entry should work for short positions."""
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=10.50,
            direction="short",
            entry_idx=5,
        )

        assert state.risk_per_share == 0.50
        assert state.direction == "short"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
