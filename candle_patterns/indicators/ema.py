"""
EMA Indicator Module
====================
Exponential Moving Average calculation for common periods (9, 20, 200).
"""

import pandas as pd
from typing import Union, List, Optional


# Default EMA periods
DEFAULT_PERIODS = [9, 20, 200]


def calculate_ema(
    data: Union[pd.Series, pd.DataFrame],
    period: int,
    column: str = "close"
) -> pd.Series:
    """
    Calculate EMA for a given period.

    Args:
        data: Price data (Series or DataFrame with 'close' column)
        period: EMA period (e.g., 9, 20, 200)
        column: Column name if DataFrame provided

    Returns:
        pd.Series: EMA values
    """
    if isinstance(data, pd.DataFrame):
        prices = data[column]
    else:
        prices = data

    return prices.ewm(span=period, adjust=False).mean()


def calculate_all_emas(
    data: pd.DataFrame,
    periods: Optional[List[int]] = None,
    column: str = "close"
) -> pd.DataFrame:
    """
    Calculate multiple EMAs and add them to DataFrame.

    Args:
        data: OHLCV DataFrame
        periods: List of periods (default: [9, 20, 200])
        column: Price column to use

    Returns:
        pd.DataFrame: Original data with EMA columns added
    """
    if periods is None:
        periods = DEFAULT_PERIODS

    result = data.copy()

    for period in periods:
        result[f"ema_{period}"] = calculate_ema(data, period, column)

    return result


def price_above_ema(
    data: pd.DataFrame,
    period: int,
    column: str = "close"
) -> bool:
    """
    Check if current price is above EMA.

    Args:
        data: OHLCV DataFrame (most recent bar last)
        period: EMA period to check
        column: Price column

    Returns:
        bool: True if price > EMA
    """
    ema_col = f"ema_{period}"

    if ema_col not in data.columns:
        data = calculate_all_emas(data, [period], column)

    last_row = data.iloc[-1]
    return last_row[column] > last_row[ema_col]


def ema_slope(
    data: pd.DataFrame,
    period: int,
    lookback: int = 3
) -> float:
    """
    Calculate EMA slope over recent bars.

    Args:
        data: OHLCV DataFrame with EMA column
        period: EMA period
        lookback: Number of bars to measure slope

    Returns:
        float: Slope (positive = uptrend, negative = downtrend)
    """
    ema_col = f"ema_{period}"

    if ema_col not in data.columns:
        data = calculate_all_emas(data, [period])

    if len(data) < lookback:
        return 0.0

    recent = data[ema_col].iloc[-lookback:]
    return (recent.iloc[-1] - recent.iloc[0]) / lookback
