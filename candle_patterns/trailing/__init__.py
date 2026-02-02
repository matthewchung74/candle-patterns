"""
Trailing Stop Module
====================

Multi-strategy trailing stop calculation for candle-patterns.

Strategies:
- swing_low: Trail to N-bar low minus buffer (default)
- atr: Trail N × ATR from high water mark

Usage:
    from candle_patterns.trailing import (
        calculate_trailing_stop,
        TrailingStopState,
        TrailingStopConfig,
        TrailingStopResult,
    )

    # Initialize state from entry
    state = TrailingStopState.from_entry(
        entry_price=10.00,
        stop_price=9.50,
        direction="long",
        entry_idx=5,
    )

    # Configure trailing (defaults to swing_low)
    config = TrailingStopConfig(
        strategy="swing_low",
        activation_r=1.0,
        current_spread=0.02,
    )

    # On each new bar, calculate trailing stop
    result = calculate_trailing_stop(bars, state, config)

    if result.active:
        # Update state for next iteration
        state.current_stop = result.new_stop
        state.high_water_mark = result.high_water_mark
        state.is_activated = True
"""

import pandas as pd
from typing import Callable

from .base import TrailingStopState, TrailingStopConfig, TrailingStopResult
from . import swing_low
from . import atr


# Strategy registry: maps strategy name to calculate function
STRATEGIES: dict[str, Callable[[pd.DataFrame, TrailingStopState, TrailingStopConfig], TrailingStopResult]] = {
    "swing_low": swing_low.calculate,
    "atr": atr.calculate,
}


def calculate_trailing_stop(
    bars: pd.DataFrame,
    state: TrailingStopState,
    config: TrailingStopConfig | None = None,
) -> TrailingStopResult:
    """
    Calculate trailing stop using the configured strategy.

    This is the main entry point for trailing stop calculation.

    Args:
        bars: OHLCV DataFrame including bars after entry
        state: Current trailing stop state (from TrailingStopState.from_entry())
        config: Trailing stop configuration (defaults to swing_low strategy)

    Returns:
        TrailingStopResult with new stop price and metadata

    Raises:
        ValueError: If strategy name is not recognized

    Example:
        # Initialize state once at entry
        state = TrailingStopState.from_entry(
            entry_price=10.00,
            stop_price=9.50,
            direction="long",
            entry_idx=5,
        )

        # Configure strategy
        config = TrailingStopConfig(strategy="swing_low")

        # Call on each new bar
        result = calculate_trailing_stop(bars, state, config)

        if result.active:
            # Update state for next call
            state.current_stop = result.new_stop
            state.high_water_mark = result.high_water_mark
            state.is_activated = True
    """
    if config is None:
        config = TrailingStopConfig()

    strategy_name = config.strategy
    if strategy_name not in STRATEGIES:
        raise ValueError(
            f"Unknown trailing stop strategy: '{strategy_name}'. "
            f"Available strategies: {list(STRATEGIES.keys())}"
        )

    strategy_fn = STRATEGIES[strategy_name]
    return strategy_fn(bars, state, config)


# Public exports
__all__ = [
    "calculate_trailing_stop",
    "TrailingStopState",
    "TrailingStopConfig",
    "TrailingStopResult",
    "STRATEGIES",
]
