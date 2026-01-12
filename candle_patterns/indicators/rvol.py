"""
Relative Volume (RVOL) Indicator Module
=======================================
Time-of-Day RVOL calculation - compares current volume to same time
historical average (premarket vs premarket, RTH vs RTH).
"""

import pandas as pd
from datetime import datetime, time
from typing import Optional, Dict, Tuple


# Default configuration
DEFAULT_LOOKBACK_DAYS = 10
DEFAULT_BUCKET_MINUTES = 5


def get_time_bucket(
    timestamp: datetime,
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES
) -> str:
    """
    Get time bucket string for a timestamp.

    Args:
        timestamp: Datetime
        bucket_minutes: Size of time buckets

    Returns:
        str: Time bucket string (e.g., "07:30")
    """
    minutes = timestamp.hour * 60 + timestamp.minute
    bucket_start = (minutes // bucket_minutes) * bucket_minutes
    hour = bucket_start // 60
    minute = bucket_start % 60
    return f"{hour:02d}:{minute:02d}"


def is_premarket(timestamp: datetime) -> bool:
    """Check if timestamp is in premarket (4 AM - 9:30 AM)."""
    t = timestamp.time() if isinstance(timestamp, datetime) else timestamp
    return time(4, 0) <= t < time(9, 30)


def is_regular_hours(timestamp: datetime) -> bool:
    """Check if timestamp is in regular trading hours (9:30 AM - 4 PM)."""
    t = timestamp.time() if isinstance(timestamp, datetime) else timestamp
    return time(9, 30) <= t < time(16, 0)


def calculate_historical_volume_profile(
    historical_bars: pd.DataFrame,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES,
    session: str = "premarket"
) -> Dict[str, float]:
    """
    Calculate average volume by time bucket from historical data.

    Args:
        historical_bars: Historical OHLCV with 'timestamp' and 'volume'
        lookback_days: Number of days to average
        bucket_minutes: Time bucket size
        session: "premarket" or "regular"

    Returns:
        dict: {time_bucket: avg_volume}
    """
    df = historical_bars.copy()

    if "timestamp" not in df.columns:
        return {}

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date

    # Filter by session
    if session == "premarket":
        df = df[df["timestamp"].apply(is_premarket)]
    else:
        df = df[df["timestamp"].apply(is_regular_hours)]

    if len(df) == 0:
        return {}

    # Get unique dates and limit to lookback
    unique_dates = sorted(df["date"].unique(), reverse=True)
    lookback_dates = unique_dates[:lookback_days]
    df = df[df["date"].isin(lookback_dates)]

    # Calculate time buckets
    df["time_bucket"] = df["timestamp"].apply(
        lambda x: get_time_bucket(x, bucket_minutes)
    )

    # Average volume by bucket
    avg_by_bucket = df.groupby("time_bucket")["volume"].mean().to_dict()

    return avg_by_bucket


def calculate_rvol_tod(
    current_bars: pd.DataFrame,
    historical_bars: pd.DataFrame,
    current_time: Optional[datetime] = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    bucket_minutes: int = DEFAULT_BUCKET_MINUTES
) -> Tuple[float, str]:
    """
    Calculate time-of-day relative volume.

    Args:
        current_bars: Today's OHLCV data
        historical_bars: Historical OHLCV for comparison
        current_time: Current timestamp (auto-detect if None)
        lookback_days: Number of historical days to average
        bucket_minutes: Time bucket size

    Returns:
        Tuple[float, str]: (RVOL ratio, session type)
    """
    if current_time is None:
        if "timestamp" in current_bars.columns:
            current_time = pd.to_datetime(current_bars["timestamp"].iloc[-1])
        else:
            current_time = datetime.now()

    # Determine session
    if is_premarket(current_time):
        session = "premarket"
    else:
        session = "regular"

    # Get historical volume profile
    volume_profile = calculate_historical_volume_profile(
        historical_bars,
        lookback_days=lookback_days,
        bucket_minutes=bucket_minutes,
        session=session
    )

    if not volume_profile:
        return 1.0, session  # Default to 1x if no history

    # Get current time bucket volume
    current_bucket = get_time_bucket(current_time, bucket_minutes)
    avg_volume = volume_profile.get(current_bucket)

    if avg_volume is None or avg_volume == 0:
        return 1.0, session

    # Sum today's volume up to current bucket
    today_bars = current_bars.copy()
    if "timestamp" in today_bars.columns:
        today_bars["timestamp"] = pd.to_datetime(today_bars["timestamp"])

        # Filter to same session
        if session == "premarket":
            today_bars = today_bars[today_bars["timestamp"].apply(is_premarket)]
        else:
            today_bars = today_bars[today_bars["timestamp"].apply(is_regular_hours)]

        # Get volume up to current bucket
        today_bars["time_bucket"] = today_bars["timestamp"].apply(
            lambda x: get_time_bucket(x, bucket_minutes)
        )
        bucket_volume = today_bars[
            today_bars["time_bucket"] == current_bucket
        ]["volume"].sum()
    else:
        # If no timestamp, use last bar's volume
        bucket_volume = today_bars["volume"].iloc[-1]

    rvol = bucket_volume / avg_volume
    return rvol, session


def calculate_cumulative_rvol(
    current_bars: pd.DataFrame,
    historical_bars: pd.DataFrame,
    session: str = "premarket"
) -> float:
    """
    Calculate cumulative RVOL for entire session so far.

    Args:
        current_bars: Today's OHLCV data
        historical_bars: Historical data
        session: "premarket" or "regular"

    Returns:
        float: Cumulative RVOL ratio
    """
    today = current_bars.copy()
    hist = historical_bars.copy()

    if "timestamp" not in today.columns:
        return 1.0

    today["timestamp"] = pd.to_datetime(today["timestamp"])
    hist["timestamp"] = pd.to_datetime(hist["timestamp"])

    # Filter by session
    if session == "premarket":
        session_filter = is_premarket
    else:
        session_filter = is_regular_hours

    today_session = today[today["timestamp"].apply(session_filter)]
    hist_session = hist[hist["timestamp"].apply(session_filter)]

    if len(today_session) == 0:
        return 1.0

    # Today's total volume
    today_volume = today_session["volume"].sum()

    # Historical average daily volume for same session
    hist_session["date"] = hist_session["timestamp"].dt.date
    daily_volumes = hist_session.groupby("date")["volume"].sum()

    if len(daily_volumes) == 0:
        return 1.0

    avg_daily_volume = daily_volumes.mean()

    if avg_daily_volume == 0:
        return 1.0

    return today_volume / avg_daily_volume
