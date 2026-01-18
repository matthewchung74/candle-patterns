"""
ATR (Average True Range) Indicator Module
==========================================
Measures market volatility using the True Range concept.
"""

import pandas as pd
from typing import Optional


DEFAULT_PERIOD = 14


def true_range(data: pd.DataFrame) -> pd.Series:
    """
    Calculate True Range for each bar.

    True Range = max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )

    Args:
        data: OHLCV DataFrame with 'high', 'low', 'close' columns

    Returns:
        pd.Series: True Range values
    """
    high = data["high"]
    low = data["low"]
    close = data["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr


def calculate_atr(
    data: pd.DataFrame,
    period: int = DEFAULT_PERIOD,
) -> pd.Series:
    """
    Calculate Average True Range (ATR).

    Uses Wilder's smoothing method (exponential moving average).

    Args:
        data: OHLCV DataFrame with 'high', 'low', 'close' columns
        period: ATR period (default 14)

    Returns:
        pd.Series: ATR values (first `period` values will be NaN/warming up)
    """
    tr = true_range(data)

    # Wilder's smoothing: alpha = 1/period
    atr = tr.ewm(alpha=1/period, adjust=False).mean()

    return atr


def get_current_atr(
    data: pd.DataFrame,
    period: int = DEFAULT_PERIOD,
) -> Optional[float]:
    """
    Get the current (latest) ATR value.

    Args:
        data: OHLCV DataFrame
        period: ATR period (default 14)

    Returns:
        float: Current ATR value, or None if insufficient data
    """
    if len(data) < period + 1:
        return None

    atr = calculate_atr(data, period)
    return atr.iloc[-1]


def add_atr_to_dataframe(
    data: pd.DataFrame,
    period: int = DEFAULT_PERIOD,
    column_name: str = "atr",
) -> pd.DataFrame:
    """
    Add ATR column to DataFrame.

    Args:
        data: OHLCV DataFrame
        period: ATR period (default 14)
        column_name: Name for the ATR column

    Returns:
        pd.DataFrame: Original data with ATR column added
    """
    result = data.copy()
    result[column_name] = calculate_atr(data, period)
    return result
