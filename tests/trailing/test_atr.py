"""
ATR Trailing Stop Tests
=======================

Tests for ATR trailing stop strategy (trail N × ATR from high water mark).

Run with: pytest tests/trailing/test_atr.py -v
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from candle_patterns.trailing import (
    calculate_trailing_stop,
    TrailingStopState,
    TrailingStopConfig,
    TrailingStopResult,
)


def _make_bars(data: list) -> pd.DataFrame:
    """Create DataFrame from OHLCV tuples."""
    base_time = datetime(2025, 1, 15, 9, 30)
    rows = []
    for i, (o, h, l, c, v) in enumerate(data):
        rows.append({
            "timestamp": base_time + timedelta(minutes=i),
            "open": o,
            "high": h,
            "low": l,
            "close": c,
            "volume": v,
        })
    return pd.DataFrame(rows)


# Test fixtures for ATR strategy
# Entry: $10.00, Stop: $9.50, Risk = $0.50, 1R = $10.50

ATR_ACTIVATED_BARS = _make_bars([
    # Pre-entry bars for ATR calculation (need 14+ bars)
    # Using ~$0.10 range per bar for predictable ATR
    (9.50, 9.60, 9.45, 9.55, 100000),
    (9.55, 9.65, 9.50, 9.60, 100000),
    (9.60, 9.70, 9.55, 9.65, 100000),
    (9.65, 9.75, 9.60, 9.70, 100000),
    (9.70, 9.80, 9.65, 9.75, 100000),
    (9.75, 9.85, 9.70, 9.80, 100000),
    (9.80, 9.90, 9.75, 9.85, 100000),
    (9.85, 9.95, 9.80, 9.90, 100000),
    (9.90, 10.00, 9.85, 9.95, 100000),
    (9.95, 10.05, 9.90, 10.00, 100000),

    # Entry bar (idx 10)
    (10.00, 10.10, 9.95, 10.05, 150000),

    # Post-entry: strong move up to 1.5R
    (10.05, 10.30, 10.02, 10.25, 180000),
    (10.25, 10.55, 10.20, 10.50, 200000),  # 1.1R
    (10.50, 10.75, 10.45, 10.70, 220000),  # 1.5R
    (10.70, 10.80, 10.65, 10.75, 180000),
])
ATR_ENTRY_IDX = 10
ATR_ENTRY_PRICE = 10.00
ATR_ORIGINAL_STOP = 9.50


ATR_SHORT_BARS = _make_bars([
    # Pre-entry bars (need 14+ for ATR)
    (10.50, 10.60, 10.45, 10.55, 100000),
    (10.55, 10.65, 10.50, 10.60, 100000),
    (10.60, 10.70, 10.55, 10.65, 100000),
    (10.65, 10.75, 10.60, 10.70, 100000),
    (10.70, 10.80, 10.65, 10.75, 100000),
    (10.75, 10.85, 10.70, 10.80, 100000),
    (10.80, 10.90, 10.75, 10.85, 100000),
    (10.85, 10.95, 10.80, 10.90, 100000),
    (10.90, 11.00, 10.85, 10.95, 100000),
    (10.95, 11.05, 10.90, 11.00, 100000),
    (10.50, 10.60, 10.45, 10.55, 100000),
    (10.45, 10.55, 10.40, 10.50, 100000),
    (10.40, 10.50, 10.35, 10.45, 100000),
    (10.30, 10.40, 10.25, 10.35, 100000),
    (10.20, 10.30, 10.15, 10.25, 100000),
    (10.10, 10.20, 10.05, 10.15, 100000),

    # Entry bar (idx 16) - shorting at $10.00
    (10.05, 10.10, 9.95, 10.00, 150000),

    # Post-entry: price drops (good for shorts)
    (10.00, 10.05, 9.70, 9.75, 160000),
    (9.75, 9.80, 9.45, 9.50, 170000),  # 1.0R profit
    (9.50, 9.55, 9.40, 9.45, 150000),
])
ATR_SHORT_ENTRY_IDX = 16
ATR_SHORT_ENTRY_PRICE = 10.00
ATR_SHORT_ORIGINAL_STOP = 10.50


ATR_INSUFFICIENT_DATA = _make_bars([
    # Only 5 bars - not enough for ATR(14)
    (10.00, 10.10, 9.95, 10.05, 100000),
    (10.05, 10.15, 10.00, 10.10, 100000),
    (10.10, 10.60, 10.05, 10.55, 100000),  # Entry bar (idx 2)
    (10.55, 10.70, 10.50, 10.65, 100000),
    (10.65, 10.80, 10.60, 10.75, 100000),
])


class TestATRTrailingActivation:
    """Tests for ATR trailing stop activation."""

    def test_activated_at_1r(self):
        """ATR trailing should activate when profit >= 1R."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_ENTRY_PRICE,
            stop_price=ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="atr")

        result = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)

        assert result.active is True
        assert result.current_r_multiple >= 1.0
        assert result.strategy_name == "atr"
        assert "ATR trailing" in result.reason

    def test_not_activated_insufficient_atr_data(self):
        """Should not activate if insufficient data for ATR calculation."""
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=9.50,
            direction="long",
            entry_idx=2,
        )
        config = TrailingStopConfig(strategy="atr")

        result = calculate_trailing_stop(ATR_INSUFFICIENT_DATA, state, config)

        assert result.active is False
        assert "Insufficient data for ATR" in result.reason


class TestATRTrailingBehavior:
    """Tests for ATR trailing stop calculation behavior."""

    def test_trails_below_high_water_mark(self):
        """ATR trailing should trail N × ATR below high water mark."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_ENTRY_PRICE,
            stop_price=ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )
        config = TrailingStopConfig(
            strategy="atr",
            params={"atr_multiplier": 2.0},
        )

        result = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)

        assert result.active is True
        assert result.new_stop > ATR_ORIGINAL_STOP
        # Stop should be HWM - (2 × ATR)
        # HWM is ~10.80, ATR is ~0.10-0.15, so stop should be ~10.50-10.60
        assert result.new_stop < result.high_water_mark
        assert "HWM" in result.reason

    def test_custom_atr_multiplier(self):
        """Should respect custom ATR multiplier."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_ENTRY_PRICE,
            stop_price=ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )

        # Tight stop with 1.0 × ATR
        config_tight = TrailingStopConfig(
            strategy="atr",
            params={"atr_multiplier": 1.0},
        )
        result_tight = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config_tight)

        # Loose stop with 3.0 × ATR
        config_loose = TrailingStopConfig(
            strategy="atr",
            params={"atr_multiplier": 3.0},
        )
        result_loose = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config_loose)

        # Tight stop should be higher (closer to price)
        assert result_tight.new_stop > result_loose.new_stop

    def test_short_position(self):
        """ATR trailing should work for short positions."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_SHORT_ENTRY_PRICE,
            stop_price=ATR_SHORT_ORIGINAL_STOP,
            direction="short",
            entry_idx=ATR_SHORT_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="atr")

        result = calculate_trailing_stop(ATR_SHORT_BARS, state, config)

        assert result.active is True
        # For shorts, stop should be LOWERED (tightened)
        assert result.new_stop < ATR_SHORT_ORIGINAL_STOP
        assert "lowered" in result.reason.lower()
        assert "LWM" in result.reason  # Low water mark for shorts

    def test_stop_never_loosens(self):
        """Stop should never loosen when tracking state."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_ENTRY_PRICE,
            stop_price=ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="atr")

        # Get initial trailing stop
        result1 = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)
        first_stop = result1.new_stop

        # Update state
        state.current_stop = first_stop
        state.is_activated = True

        # Even with same bars, stop should not loosen
        result2 = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)

        assert result2.new_stop >= first_stop


class TestATRTrailingEdgeCases:
    """Tests for ATR trailing edge cases."""

    def test_zero_risk(self):
        """Should handle zero risk gracefully."""
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=10.00,  # Zero risk
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )
        config = TrailingStopConfig(strategy="atr")

        result = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)

        assert result.active is False
        assert "zero risk" in result.reason.lower()

    def test_custom_atr_period(self):
        """Should respect custom ATR period."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_ENTRY_PRICE,
            stop_price=ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )
        config = TrailingStopConfig(
            strategy="atr",
            params={"atr_period": 10},  # Shorter period
        )

        result = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)

        assert result.active is True
        assert result.strategy_name == "atr"


class TestStrategySelection:
    """Tests for strategy selection via config."""

    def test_invalid_strategy_raises(self):
        """Should raise ValueError for invalid strategy name."""
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=9.50,
            direction="long",
            entry_idx=0,
        )
        config = TrailingStopConfig(strategy="invalid_strategy")

        with pytest.raises(ValueError) as exc_info:
            calculate_trailing_stop(ATR_ACTIVATED_BARS, state, config)

        assert "Unknown trailing stop strategy" in str(exc_info.value)
        assert "invalid_strategy" in str(exc_info.value)

    def test_default_config_uses_swing_low(self):
        """Default config should use swing_low strategy."""
        state = TrailingStopState.from_entry(
            entry_price=ATR_ENTRY_PRICE,
            stop_price=ATR_ORIGINAL_STOP,
            direction="long",
            entry_idx=ATR_ENTRY_IDX,
        )
        # No config = defaults
        result = calculate_trailing_stop(ATR_ACTIVATED_BARS, state, None)

        assert result.strategy_name == "swing_low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
