"""
Technical Indicators Module
===========================
EMA, VWAP, MACD, RVOL, and ATR indicators.

Self-contained indicator calculations for pattern detection
and strategy integration.
"""

from .atr import (
    true_range,
    calculate_atr,
    get_current_atr,
    add_atr_to_dataframe,
)

from .ema import (
    calculate_ema,
    calculate_all_emas,
    price_above_ema,
    ema_slope,
)

from .vwap import (
    calculate_vwap,
    calculate_premarket_vwap,
    calculate_regular_vwap,
    get_current_vwap,
    price_above_vwap,
)

from .macd import (
    calculate_macd,
    add_macd_to_dataframe,
    macd_is_positive,
    macd_crossover,
    macd_histogram_slope,
)

from .rvol import (
    calculate_rvol_tod,
    calculate_cumulative_rvol,
    calculate_historical_volume_profile,
    is_premarket,
    is_regular_hours,
)

from .trend_confirmation import (
    check_5min_trend_confirmation,
    is_green_candle,
    is_red_candle,
    is_doji,
    count_consecutive_dojis,
    check_candle_quality,
)

__all__ = [
    # ATR
    "true_range",
    "calculate_atr",
    "get_current_atr",
    "add_atr_to_dataframe",
    # EMA
    "calculate_ema",
    "calculate_all_emas",
    "price_above_ema",
    "ema_slope",
    # VWAP
    "calculate_vwap",
    "calculate_premarket_vwap",
    "calculate_regular_vwap",
    "get_current_vwap",
    "price_above_vwap",
    # MACD
    "calculate_macd",
    "add_macd_to_dataframe",
    "macd_is_positive",
    "macd_crossover",
    "macd_histogram_slope",
    # RVOL
    "calculate_rvol_tod",
    "calculate_cumulative_rvol",
    "calculate_historical_volume_profile",
    "is_premarket",
    "is_regular_hours",
    # Trend Confirmation
    "check_5min_trend_confirmation",
    "is_green_candle",
    "is_red_candle",
    "is_doji",
    "count_consecutive_dojis",
    "check_candle_quality",
]
