"""
Base Pattern Detector
=====================

Abstract base class for all pattern detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class PatternResult:
    """Result of pattern detection."""

    detected: bool
    pattern_name: str
    confidence: float  # 0.0 to 1.0

    # Entry/Exit levels (None if not detected)
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    stop_distance_cents: Optional[float] = None

    # Pattern metadata
    pattern_start_idx: Optional[int] = None
    pattern_end_idx: Optional[int] = None
    candle_count: Optional[int] = None

    # Confirmation signals
    above_vwap: Optional[bool] = None
    macd_positive: Optional[bool] = None
    volume_confirmation: Optional[bool] = None

    # Debug info
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __bool__(self):
        """Allow using result directly in if statements."""
        return self.detected

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Calculate R:R if entry and stop are set."""
        if self.entry_price and self.stop_price and self.stop_distance_cents:
            # Assumes 2:1 minimum target
            return 2.0
        return None


class PatternDetector(ABC):
    """
    Abstract base class for pattern detection.

    All pattern detectors must implement:
    - detect(): Main detection method
    - validate_bars(): Ensure input data is valid

    Expected DataFrame columns:
    - timestamp: datetime
    - open: float
    - high: float
    - low: float
    - close: float
    - volume: int/float
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize pattern detector with optional config overrides.

        Args:
            config: Dictionary of pattern-specific parameters
        """
        self.config = self.default_config()
        if config:
            self.config.update(config)

    @abstractmethod
    def default_config(self) -> Dict[str, Any]:
        """Return default configuration for this pattern."""
        pass

    @abstractmethod
    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Detect pattern in the given bars.

        Args:
            bars: DataFrame with OHLCV data
            vwap: Optional VWAP series for confirmation
            macd: Optional MACD DataFrame with 'macd', 'signal', 'histogram'

        Returns:
            PatternResult with detection details
        """
        pass

    def validate_bars(self, bars: pd.DataFrame) -> bool:
        """
        Validate that bars DataFrame has required columns.

        Args:
            bars: DataFrame to validate

        Returns:
            True if valid, raises ValueError if not
        """
        required_columns = ["open", "high", "low", "close", "volume"]

        if bars is None or bars.empty:
            raise ValueError("Bars DataFrame is empty or None")

        missing = [col for col in required_columns if col not in bars.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        if len(bars) < self.config.get("min_bars_required", 5):
            raise ValueError(
                f"Insufficient bars: {len(bars)} < {self.config.get('min_bars_required', 5)}"
            )

        return True

    def is_green_candle(self, row: pd.Series) -> bool:
        """Check if candle is green (close > open)."""
        return row["close"] > row["open"]

    def is_red_candle(self, row: pd.Series) -> bool:
        """Check if candle is red (close < open)."""
        return row["close"] < row["open"]

    def candle_body_pct(self, row: pd.Series) -> float:
        """Calculate candle body as percentage of total range."""
        total_range = row["high"] - row["low"]
        if total_range == 0:
            return 0.0
        body = abs(row["close"] - row["open"])
        return (body / total_range) * 100

    def calculate_move_pct(self, start_price: float, end_price: float) -> float:
        """Calculate percentage move between two prices."""
        if start_price == 0:
            return 0.0
        return ((end_price - start_price) / start_price) * 100

    def calculate_macd(
        self,
        closes: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Optional[pd.DataFrame]:
        """
        Calculate MACD if enough bars available.

        Args:
            closes: Series of close prices
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            DataFrame with 'macd', 'signal', 'histogram' columns,
            or None if insufficient bars (< 35)
        """
        min_bars = slow + signal  # ~35 bars for stable values
        if len(closes) < min_bars:
            return None

        fast_ema = closes.ewm(span=fast, adjust=False).mean()
        slow_ema = closes.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram,
        })

    def not_detected(self, reason: str) -> PatternResult:
        """Helper to return a non-detected result with reason."""
        return PatternResult(
            detected=False,
            pattern_name=self.__class__.__name__,
            confidence=0.0,
            reason=reason,
        )
