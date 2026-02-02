"""
ATR Trailing Stop Strategy
==========================

Trail stop at high_water_mark - (N × ATR) for longs.
Best for volatile stocks where you want volatility-adjusted trailing.
"""

import pandas as pd

from .base import TrailingStopState, TrailingStopConfig, TrailingStopResult
from ..indicators.atr import get_current_atr


# Default strategy parameters
DEFAULT_PARAMS = {
    "atr_multiplier": 2.0,    # Trail N × ATR from high water mark
    "atr_period": 14,         # ATR lookback period
}


def calculate(
    bars: pd.DataFrame,
    state: TrailingStopState,
    config: TrailingStopConfig,
) -> TrailingStopResult:
    """
    Calculate ATR trailing stop.

    Trails at high_water_mark - (N × ATR) for longs.
    Trails at low_water_mark + (N × ATR) for shorts.

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
            strategy_name="atr",
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
            strategy_name="atr",
        )

    # Get ATR value
    atr_value = get_current_atr(df, period=params["atr_period"])
    if atr_value is None:
        return TrailingStopResult(
            active=False,
            new_stop=state.original_stop,
            original_stop=state.original_stop,
            high_water_mark=state.entry_price,
            current_r_multiple=0.0,
            reason=f"Insufficient data for ATR({params['atr_period']})",
            strategy_name="atr",
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
            strategy_name="atr",
        )

    # Calculate ATR trailing stop
    atr_distance = atr_value * params["atr_multiplier"]

    if state.direction == "long":
        # Trail below high water mark for longs
        trailing_stop = high_water_mark - atr_distance

        # Never lower the stop
        if config.never_loosen_stop:
            floor_stop = max(state.original_stop, state.current_stop)
            new_stop = max(floor_stop, trailing_stop)
        else:
            new_stop = max(state.original_stop, trailing_stop)
    else:
        # Trail above low water mark for shorts
        trailing_stop = high_water_mark + atr_distance

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
                f"ATR trailing: HWM ${high_water_mark:.2f} - "
                f"{params['atr_multiplier']}×ATR ${atr_distance:.2f} = ${trailing_stop:.2f}, "
                f"stop raised to ${new_stop:.2f}"
            )
        else:
            reason = (
                f"ATR trailing: LWM ${high_water_mark:.2f} + "
                f"{params['atr_multiplier']}×ATR ${atr_distance:.2f} = ${trailing_stop:.2f}, "
                f"stop lowered to ${new_stop:.2f}"
            )
    else:
        if state.direction == "long":
            reason = (
                f"ATR trailing active but stop unchanged: "
                f"trailing ${trailing_stop:.2f} < original ${state.original_stop:.2f}"
            )
        else:
            reason = (
                f"ATR trailing active but stop unchanged: "
                f"trailing ${trailing_stop:.2f} > original ${state.original_stop:.2f}"
            )

    return TrailingStopResult(
        active=True,
        new_stop=new_stop,
        original_stop=state.original_stop,
        high_water_mark=high_water_mark,
        current_r_multiple=current_r,
        reason=reason,
        strategy_name="atr",
        just_activated=just_activated,
        stop_moved=stop_moved,
    )
