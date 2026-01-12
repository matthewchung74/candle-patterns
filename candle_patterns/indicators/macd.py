"""
MACD Indicator Module
=====================
Moving Average Convergence Divergence (12, 26, 9 standard parameters).
"""

import pandas as pd
from typing import Tuple, Optional


# Default MACD parameters (standard)
DEFAULT_FAST = 12
DEFAULT_SLOW = 26
DEFAULT_SIGNAL = 9


def calculate_macd(
    data: pd.DataFrame,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL,
    column: str = "close"
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD line, signal line, and histogram.

    Args:
        data: OHLCV DataFrame
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)
        column: Price column to use

    Returns:
        Tuple[pd.Series, pd.Series, pd.Series]: (macd_line, signal_line, histogram)
    """
    prices = data[column]

    # MACD Line = Fast EMA - Slow EMA
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema

    # Signal Line = EMA of MACD Line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram = MACD Line - Signal Line
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def add_macd_to_dataframe(
    data: pd.DataFrame,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL
) -> pd.DataFrame:
    """
    Add MACD columns to DataFrame.

    Args:
        data: OHLCV DataFrame
        fast, slow, signal: MACD parameters

    Returns:
        pd.DataFrame: Original data with MACD columns added
    """
    result = data.copy()
    macd_line, signal_line, histogram = calculate_macd(data, fast, slow, signal)

    result["macd"] = macd_line
    result["macd_signal"] = signal_line
    result["macd_histogram"] = histogram

    return result


def macd_is_positive(
    data: pd.DataFrame,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL
) -> Optional[bool]:
    """
    Check if MACD histogram is positive (bullish momentum).

    Args:
        data: OHLCV DataFrame (minimum 35 bars recommended)

    Returns:
        bool: True if histogram > 0, None if insufficient data
    """
    # Need at least slow + signal bars for valid MACD
    min_bars = slow + signal  # 35 bars minimum

    if len(data) < min_bars:
        return None

    _, _, histogram = calculate_macd(data, fast, slow, signal)
    return histogram.iloc[-1] > 0


def macd_crossover(
    data: pd.DataFrame,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL
) -> Optional[str]:
    """
    Detect MACD crossover events.

    Args:
        data: OHLCV DataFrame

    Returns:
        str: "bullish" if MACD crosses above signal,
             "bearish" if MACD crosses below signal,
             None if no crossover or insufficient data
    """
    min_bars = slow + signal + 1
    if len(data) < min_bars:
        return None

    macd_line, signal_line, _ = calculate_macd(data, fast, slow, signal)

    # Current and previous positions
    current_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
    prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]

    if prev_diff <= 0 and current_diff > 0:
        return "bullish"
    elif prev_diff >= 0 and current_diff < 0:
        return "bearish"

    return None


def macd_histogram_slope(
    data: pd.DataFrame,
    lookback: int = 3,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    signal: int = DEFAULT_SIGNAL
) -> Optional[float]:
    """
    Calculate the slope of MACD histogram (momentum acceleration).

    Args:
        data: OHLCV DataFrame
        lookback: Number of bars for slope calculation

    Returns:
        float: Slope (positive = momentum increasing), None if insufficient data
    """
    min_bars = slow + signal + lookback
    if len(data) < min_bars:
        return None

    _, _, histogram = calculate_macd(data, fast, slow, signal)

    if len(histogram) < lookback:
        return None

    recent = histogram.iloc[-lookback:]
    return (recent.iloc[-1] - recent.iloc[0]) / lookback
