"""
5-Minute Trend Confirmation for Entry Signals
==============================================

Confirms higher timeframe trend before taking 1-min setups.
Works in both premarket and regular hours (no VWAP limitations).
Supports both long and short directions.

Conditions for LONG entry:
1. At least 3 of last 4 candles are green (bullish trend)
2. Most recent close > high from 2-3 candles ago (actual breakout/progress)
3. No huge rejection wick on latest candle (not being rejected at highs)

Conditions for SHORT entry:
1. At least 3 of last 4 candles are red (bearish trend)
2. Most recent close < low from 2-3 candles ago (actual breakdown)
3. No huge rejection wick on latest candle (not being rejected at lows)

Early Session (<4 bars):
Uses simplified 2-bar check instead of skipping entirely.
"""

import pandas as pd
from typing import Tuple, Optional, Literal


def check_5min_trend_confirmation(
    bars_5min: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    min_trend_candles: int = 3,
    lookback_candles: int = 4,
    max_wick_ratio: float = 2.0,
) -> Tuple[bool, str]:
    """
    Check if 5-min trend supports entry in the given direction.

    Args:
        bars_5min: DataFrame with OHLCV data (5-min bars)
        direction: Trade direction - "long" or "short"
        min_trend_candles: Minimum candles in trend direction required (default: 3)
        lookback_candles: Number of candles to check (default: 4)
        max_wick_ratio: Max rejection wick as multiple of body (default: 2.0)

    Returns:
        tuple: (passed: bool, reason: str)
    """
    if bars_5min is None or len(bars_5min) == 0:
        return True, "No bars available - skipping check"

    # Early session fallback: use simplified 2-bar check
    if len(bars_5min) < lookback_candles:
        return _check_early_session_trend(bars_5min, direction)

    last_n = bars_5min.tail(lookback_candles)

    if direction == "long":
        return _check_long_trend(bars_5min, last_n, min_trend_candles, lookback_candles, max_wick_ratio)
    else:
        return _check_short_trend(bars_5min, last_n, min_trend_candles, lookback_candles, max_wick_ratio)


def _check_early_session_trend(
    bars: pd.DataFrame,
    direction: Literal["long", "short"],
) -> Tuple[bool, str]:
    """
    Simplified trend check for early session when <4 bars available.

    Requires:
    - At least 2 bars
    - Last bar in trend direction (green for long, red for short)
    - Current close making progress (above prior high for long, below prior low for short)
    """
    if len(bars) < 2:
        return True, "Only 1 bar - skipping early session check"

    last_bar = bars.iloc[-1]
    prior_bar = bars.iloc[-2]

    if direction == "long":
        # Last bar should be green
        if last_bar['close'] <= last_bar['open']:
            return False, f"Early session: last bar is red (need green for long)"

        # Close should be above prior bar's high
        if last_bar['close'] <= prior_bar['high']:
            return False, f"Early session: close ${last_bar['close']:.2f} <= prior high ${prior_bar['high']:.2f}"

        return True, "Early session trend confirmed (2-bar check)"

    else:  # short
        # Last bar should be red
        if last_bar['close'] >= last_bar['open']:
            return False, f"Early session: last bar is green (need red for short)"

        # Close should be below prior bar's low
        if last_bar['close'] >= prior_bar['low']:
            return False, f"Early session: close ${last_bar['close']:.2f} >= prior low ${prior_bar['low']:.2f}"

        return True, "Early session trend confirmed (2-bar check)"


def _check_long_trend(
    bars_5min: pd.DataFrame,
    last_n: pd.DataFrame,
    min_trend_candles: int,
    lookback_candles: int,
    max_wick_ratio: float,
) -> Tuple[bool, str]:
    """Check trend confirmation for long entry."""

    # Condition 1: At least min_trend_candles are green
    green_mask = last_n['close'] > last_n['open']
    green_count = green_mask.sum()

    if green_count < min_trend_candles:
        return False, f"Weak trend: only {green_count}/{lookback_candles} green candles on 5-min"

    # Condition 2: Current close > high from 2-3 bars ago (shows breakout progress)
    current_close = bars_5min['close'].iloc[-1]
    high_2_bars_ago = bars_5min['high'].iloc[-3] if len(bars_5min) >= 3 else 0
    high_3_bars_ago = bars_5min['high'].iloc[-4] if len(bars_5min) >= 4 else 0
    reference_high = max(high_2_bars_ago, high_3_bars_ago)

    if current_close <= reference_high:
        return False, f"No breakout: close ${current_close:.2f} <= prior high ${reference_high:.2f}"

    # Condition 3: No huge upper rejection wick on latest candle
    latest = bars_5min.iloc[-1]
    body = abs(latest['close'] - latest['open'])
    upper_wick = latest['high'] - max(latest['close'], latest['open'])

    if body > 0.001 and upper_wick > max_wick_ratio * body:
        return False, f"Rejection wick: upper wick ${upper_wick:.2f} > {max_wick_ratio}x body ${body:.2f}"

    return True, "5-min long trend confirmed"


def _check_short_trend(
    bars_5min: pd.DataFrame,
    last_n: pd.DataFrame,
    min_trend_candles: int,
    lookback_candles: int,
    max_wick_ratio: float,
) -> Tuple[bool, str]:
    """Check trend confirmation for short entry."""

    # Condition 1: At least min_trend_candles are red
    red_mask = last_n['close'] < last_n['open']
    red_count = red_mask.sum()

    if red_count < min_trend_candles:
        return False, f"Weak trend: only {red_count}/{lookback_candles} red candles on 5-min"

    # Condition 2: Current close < low from 2-3 bars ago (shows breakdown progress)
    current_close = bars_5min['close'].iloc[-1]
    low_2_bars_ago = bars_5min['low'].iloc[-3] if len(bars_5min) >= 3 else float('inf')
    low_3_bars_ago = bars_5min['low'].iloc[-4] if len(bars_5min) >= 4 else float('inf')
    reference_low = min(low_2_bars_ago, low_3_bars_ago)

    if current_close >= reference_low:
        return False, f"No breakdown: close ${current_close:.2f} >= prior low ${reference_low:.2f}"

    # Condition 3: No huge lower rejection wick on latest candle
    latest = bars_5min.iloc[-1]
    body = abs(latest['close'] - latest['open'])
    lower_wick = min(latest['close'], latest['open']) - latest['low']

    if body > 0.001 and lower_wick > max_wick_ratio * body:
        return False, f"Rejection wick: lower wick ${lower_wick:.2f} > {max_wick_ratio}x body ${body:.2f}"

    return True, "5-min short trend confirmed"


def is_green_candle(open_price: float, close_price: float) -> bool:
    """Check if candle is green (bullish)."""
    return close_price > open_price


def is_red_candle(open_price: float, close_price: float) -> bool:
    """Check if candle is red (bearish)."""
    return close_price < open_price


def is_doji(row: pd.Series, threshold: float = 0.2) -> bool:
    """
    Check if candle is a doji (indecision).

    A doji has a very small body relative to its range.

    Args:
        row: Series with open, high, low, close
        threshold: Max body/range ratio to be considered doji (default: 0.2 = 20%)

    Returns:
        bool: True if doji
    """
    body = abs(row['close'] - row['open'])
    range_size = row['high'] - row['low']

    if range_size < 0.001:
        return True  # No range = doji

    return (body / range_size) < threshold


def count_consecutive_dojis(bars: pd.DataFrame, threshold: float = 0.2) -> int:
    """
    Count consecutive dojis at the end of the dataframe.

    Args:
        bars: DataFrame with OHLCV data
        threshold: Doji threshold (default: 0.2)

    Returns:
        int: Number of consecutive dojis at end
    """
    count = 0
    for i in range(len(bars) - 1, -1, -1):
        if is_doji(bars.iloc[i], threshold):
            count += 1
        else:
            break
    return count


def check_candle_quality(
    bars: pd.DataFrame,
    max_consecutive_dojis: int = 1,
    doji_threshold: float = 0.2,
) -> Tuple[bool, str]:
    """
    Check if recent candle quality is good for entry.

    Rejects if too many consecutive dojis (indecision) before entry.

    Args:
        bars: DataFrame with OHLCV data (1-min bars typically)
        max_consecutive_dojis: Max dojis allowed at end (default: 1)
        doji_threshold: Body/range ratio threshold for doji (default: 0.2)

    Returns:
        tuple: (passed: bool, reason: str)
    """
    if bars is None or len(bars) < 2:
        return True, "Insufficient bars for candle quality check"

    consecutive_dojis = count_consecutive_dojis(bars.tail(5), doji_threshold)

    if consecutive_dojis > max_consecutive_dojis:
        return False, f"Weak momentum: {consecutive_dojis} consecutive dojis"

    return True, "Candle quality OK"
