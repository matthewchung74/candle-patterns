"""
VWAP Indicator Module
=====================
Dual VWAP calculation:
- Premarket VWAP: Cumulative from 4 AM, used for 7:00-9:30 AM trades
- Regular VWAP: Resets at 9:30 AM, used for regular session trades
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional, Tuple


# Default time boundaries
PREMARKET_START = time(4, 0)
REGULAR_START = time(9, 30)


def calculate_vwap(
    data: pd.DataFrame,
    reset_time: Optional[time] = None
) -> pd.Series:
    """
    Calculate VWAP, optionally resetting at a specific time.

    Args:
        data: OHLCV DataFrame with 'high', 'low', 'close', 'volume'
              Optional 'timestamp' column for time-based reset
        reset_time: Time to reset VWAP cumulation (None = no reset)

    Returns:
        pd.Series: VWAP values
    """
    df = data.copy()

    # Sort by timestamp to ensure correct cumulative calculation
    # (handles out-of-order bars from historical backfill + live updates)
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Typical price = (H + L + C) / 3
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["tp_volume"] = df["typical_price"] * df["volume"]

    if reset_time is not None and "timestamp" in df.columns:
        # Create session groups based on reset time
        df["time_only"] = pd.to_datetime(df["timestamp"]).dt.time
        df["session"] = (df["time_only"] >= reset_time).astype(int).diff().fillna(0).cumsum()

        # Cumulative within each session
        df["cum_tp_volume"] = df.groupby("session")["tp_volume"].cumsum()
        df["cum_volume"] = df.groupby("session")["volume"].cumsum()
    else:
        # No reset - cumulative from start
        df["cum_tp_volume"] = df["tp_volume"].cumsum()
        df["cum_volume"] = df["volume"].cumsum()

    # VWAP = cumulative(TP * V) / cumulative(V)
    vwap = df["cum_tp_volume"] / df["cum_volume"]
    vwap = vwap.replace([np.inf, -np.inf], np.nan)

    return vwap


def calculate_premarket_vwap(
    data: pd.DataFrame,
    premarket_start: time = PREMARKET_START
) -> pd.Series:
    """
    Calculate premarket VWAP (cumulative from 4 AM ET, no reset).

    Args:
        data: OHLCV DataFrame
        premarket_start: Premarket start time (default 4:00 AM)

    Returns:
        pd.Series: Premarket VWAP values
    """
    from zoneinfo import ZoneInfo

    # Filter to only premarket bars (4 AM to 9:30 AM ET)
    if "timestamp" in data.columns:
        df = data.copy()
        # Convert to ET timezone for proper time comparison
        timestamps = pd.to_datetime(df["timestamp"])
        if timestamps.dt.tz is not None:
            timestamps_et = timestamps.dt.tz_convert(ZoneInfo("America/New_York"))
        else:
            timestamps_et = timestamps.dt.tz_localize("UTC").dt.tz_convert(ZoneInfo("America/New_York"))
        df["time_only_et"] = timestamps_et.dt.time

        end = REGULAR_START

        # Keep only premarket bars for VWAP calculation
        premarket_mask = (df["time_only_et"] >= premarket_start) & (df["time_only_et"] < end)
        premarket_data = df[premarket_mask].copy()

        if len(premarket_data) == 0:
            return pd.Series(index=data.index, dtype=float)

        # Calculate VWAP on premarket data only
        premarket_vwap = calculate_vwap(premarket_data, reset_time=None)

        # Map back to full index
        result = pd.Series(index=data.index, dtype=float)
        result.loc[premarket_data.index] = premarket_vwap
        return result

    # If no timestamp, calculate on all data
    return calculate_vwap(data, reset_time=None)


def calculate_regular_vwap(
    data: pd.DataFrame,
    regular_start: time = REGULAR_START
) -> pd.Series:
    """
    Calculate regular session VWAP (resets at 9:30 AM).

    Args:
        data: OHLCV DataFrame
        regular_start: Regular session start time (default 9:30 AM)

    Returns:
        pd.Series: Regular session VWAP values
    """
    return calculate_vwap(data, reset_time=regular_start)


def get_current_vwap(
    data: pd.DataFrame,
    current_time: Optional[datetime] = None
) -> Tuple[float, str]:
    """
    Get the appropriate VWAP value based on current time.

    Args:
        data: OHLCV DataFrame
        current_time: Current time (auto-detected from last bar if None)

    Returns:
        Tuple[float, str]: (VWAP value, VWAP type: "premarket" or "regular")
    """
    # Determine current time
    if current_time is None:
        if "timestamp" in data.columns:
            current_time = pd.to_datetime(data["timestamp"].iloc[-1])
        else:
            current_time = datetime.now()

    current_time_only = current_time.time() if isinstance(current_time, datetime) else current_time

    # Before 9:30 AM = premarket VWAP
    if current_time_only < REGULAR_START:
        vwap_series = calculate_premarket_vwap(data)
        vwap_type = "premarket"
    else:
        vwap_series = calculate_regular_vwap(data)
        vwap_type = "regular"

    # Get last valid value
    last_valid = vwap_series.dropna().iloc[-1] if len(vwap_series.dropna()) > 0 else np.nan

    return last_valid, vwap_type


def price_above_vwap(
    data: pd.DataFrame,
    current_time: Optional[datetime] = None
) -> bool:
    """
    Check if current price is above VWAP.

    Args:
        data: OHLCV DataFrame
        current_time: Current time (for VWAP selection)

    Returns:
        bool: True if close > VWAP
    """
    vwap_value, _ = get_current_vwap(data, current_time)

    if pd.isna(vwap_value):
        return False

    return data["close"].iloc[-1] > vwap_value
