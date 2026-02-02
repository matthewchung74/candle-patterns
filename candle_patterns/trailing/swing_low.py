"""
Swing Low Trailing Stop Strategy
================================

Trail stop to N-bar low (longs) or N-bar high (shorts) minus a dynamic buffer.
Best for momentum trades where you want to give the trade room to breathe.
"""

import pandas as pd
from typing import Optional

from .base import TrailingStopState, TrailingStopConfig, TrailingStopResult
from ..indicators.atr import get_current_atr


# Default strategy parameters
DEFAULT_PARAMS = {
    "trailing_bars": 2,          # Number of bars for low/high calculation
    "spread_multiplier": 2.0,    # Spread × N for buffer
    "atr_multiplier": 0.1,       # ATR × N for buffer
    "atr_period": 14,            # ATR lookback period
}


def calculate(
    bars: pd.DataFrame,
    state: TrailingStopState,
    config: TrailingStopConfig,
) -> TrailingStopResult:
    """
    Calculate swing low trailing stop.

    Uses N-bar low (longs) or N-bar high (shorts) with dynamic buffer.
    Buffer = max(spread × spread_multiplier, ATR × atr_multiplier)

    Args:
        bars: OHLCV DataFrame including bars after entry
        state: Current trailing stop state
        config: Trailing stop configuration

    Returns:
        TrailingStopResult with new stop price and metadata
    """
    # Merge default params with config params
    params = {**DEFAULT_PARAMS, **config.params}

    df = bars.copy().reset_index(drop=True)
    n = len(df)
    entry_idx = state.entry_idx

    # Calculate risk (R)
    risk = state.risk_per_share
    if risk == 0:
        return TrailingStopResult(
            active=False,
            new_stop=state.original_stop,
            original_stop=state.original_stop,
            high_water_mark=state.entry_price,
            current_r_multiple=0.0,
            reason="Invalid setup: zero risk",
            strategy_name="swing_low",
        )

    # Need enough bars after entry
    bars_after_entry = n - entry_idx - 1
    if bars_after_entry < config.min_bars_after_entry:
        return TrailingStopResult(
            active=False,
            new_stop=state.original_stop,
            original_stop=state.original_stop,
            high_water_mark=state.entry_price,
            current_r_multiple=0.0,
            reason=f"Need {config.min_bars_after_entry} bars after entry, have {bars_after_entry}",
            strategy_name="swing_low",
        )

    # Get post-entry bars (excluding entry bar for trailing calc)
    post_entry = df.iloc[entry_idx + 1:]

    # Calculate high water mark and current R-multiple
    if state.direction == "long":
        high_water_mark = post_entry["high"].max()
        current_profit = high_water_mark - state.entry_price
    else:
        # For shorts, track low water mark
        high_water_mark = post_entry["low"].min()
        current_profit = state.entry_price - high_water_mark

    current_r = current_profit / risk

    # Check activation conditions
    was_activated = state.is_activated
    is_activated = was_activated or (current_r >= config.activation_r)
    if config.activate_on_partial and state.partial_taken:
        is_activated = True

    just_activated = is_activated and not was_activated

    if not is_activated:
        return TrailingStopResult(
            active=False,
            new_stop=state.original_stop,
            original_stop=state.original_stop,
            high_water_mark=high_water_mark,
            current_r_multiple=current_r,
            reason=f"Not activated: {current_r:.2f}R < {config.activation_r}R threshold",
            strategy_name="swing_low",
        )

    # Calculate dynamic buffer
    # buffer = max(spread × multiplier, ATR × multiplier)
    spread_buffer = config.current_spread * params["spread_multiplier"]

    atr_value = get_current_atr(df, period=params["atr_period"])
    if atr_value is not None:
        atr_buffer = atr_value * params["atr_multiplier"]
    else:
        # Fallback if not enough data for ATR
        atr_buffer = 0.0

    buffer = max(spread_buffer, atr_buffer)

    # Calculate N-bar low/high trailing stop
    trailing_bars = params["trailing_bars"]

    # Get last N completed bars
    completed_bars = post_entry.iloc[-trailing_bars:] if len(post_entry) >= trailing_bars else post_entry

    if state.direction == "long":
        # N-bar low trailing for longs
        n_bar_low = completed_bars["low"].min()
        trailing_stop = n_bar_low - buffer

        # Never lower the stop - use max of original, current, and new trailing
        if config.never_loosen_stop:
            floor_stop = max(state.original_stop, state.current_stop)
            new_stop = max(floor_stop, trailing_stop)
        else:
            new_stop = max(state.original_stop, trailing_stop)
    else:
        # N-bar high trailing for shorts
        n_bar_high = completed_bars["high"].max()
        trailing_stop = n_bar_high + buffer

        # Never raise the stop (for shorts, higher stop is worse)
        if config.never_loosen_stop:
            ceiling_stop = min(state.original_stop, state.current_stop)
            new_stop = min(ceiling_stop, trailing_stop)
        else:
            new_stop = min(state.original_stop, trailing_stop)

    # Check if stop actually moved
    stop_moved = abs(new_stop - state.current_stop) > 0.0001

    # Build reason string
    if new_stop != state.original_stop:
        if state.direction == "long":
            reason = (
                f"Trailing active: {trailing_bars}-bar low ${n_bar_low:.2f} - "
                f"buffer ${buffer:.3f} = ${trailing_stop:.2f}, "
                f"stop raised to ${new_stop:.2f}"
            )
        else:
            reason = (
                f"Trailing active: {trailing_bars}-bar high ${n_bar_high:.2f} + "
                f"buffer ${buffer:.3f} = ${trailing_stop:.2f}, "
                f"stop lowered to ${new_stop:.2f}"
            )
    else:
        if state.direction == "long":
            reason = (
                f"Trailing active but stop unchanged: "
                f"trailing ${trailing_stop:.2f} < original ${state.original_stop:.2f}"
            )
        else:
            reason = (
                f"Trailing active but stop unchanged: "
                f"trailing ${trailing_stop:.2f} > original ${state.original_stop:.2f}"
            )

    return TrailingStopResult(
        active=True,
        new_stop=new_stop,
        original_stop=state.original_stop,
        high_water_mark=high_water_mark,
        current_r_multiple=current_r,
        reason=reason,
        strategy_name="swing_low",
        just_activated=just_activated,
        stop_moved=stop_moved,
    )
