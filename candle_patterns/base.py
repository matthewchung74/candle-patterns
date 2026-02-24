"""
Base Pattern Detector
=====================

Abstract base class for all pattern detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List, Dict, Any
import pandas as pd


@dataclass
class ExitSignal:
    """Exit signal detected during trade monitoring."""
    signal_type: str  # "macd_cross", "volume_decline", "jackknife", "stop_hit"
    triggered: bool
    reason: str
    bar_idx: Optional[int] = None
    price: Optional[float] = None


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
    macd_slope_up: Optional[bool] = None
    volume_confirmation: Optional[bool] = None

    # Exit signals (invalidation conditions)
    exit_signals: Optional[List[ExitSignal]] = None

    # Debug info
    reason: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __bool__(self):
        """Allow using result directly in if statements."""
        return self.detected

    def calc_risk_reward(self, target_price: float) -> Optional[float]:
        """
        Calculate R:R ratio for a given target price.

        Args:
            target_price: Profit target price

        Returns:
            Risk/reward ratio (e.g., 2.0 means 2:1)
        """
        if self.entry_price is None or self.stop_price is None:
            return None
        if self.entry_price == self.stop_price:
            return None

        risk = abs(self.entry_price - self.stop_price)
        reward = abs(target_price - self.entry_price)

        if risk == 0:
            return None
        return reward / risk


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

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 exit_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pattern detector with optional config overrides.

        Args:
            config: Dictionary of pattern-specific parameters
            exit_config: Pattern-specific exit/trailing stop configuration
        """
        self.config = self.default_config()
        if config:
            self.config.update(config)
        self.exit_config = exit_config  # Pattern-specific exit config (or None for global)

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

    @staticmethod
    def _bar_time(df: pd.DataFrame, idx: int) -> str:
        """Get HH:MM timestamp string for a bar index."""
        if "timestamp" in df.columns and 0 <= idx < len(df):
            ts = df.iloc[idx]["timestamp"]
            if hasattr(ts, "strftime"):
                return ts.strftime("%H:%M")
        return ""

    def not_detected(self, reason: str) -> PatternResult:
        """Helper to return a non-detected result with reason."""
        return PatternResult(
            detected=False,
            pattern_name=self.__class__.__name__,
            confidence=0.0,
            reason=reason,
        )

    def check_exit_signals(
        self,
        bars: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_price: float,
        direction: str = "long",
        current_time: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
        vwap: Optional[pd.Series] = None,
    ) -> List[ExitSignal]:
        """
        Check for exit/invalidation signals after entry.

        Call this on each new bar after entry to monitor for exits.

        Args:
            bars: OHLCV DataFrame including bars after entry
            entry_idx: Index of the entry bar
            entry_price: Price at which position was entered
            stop_price: Stop loss price
            direction: "long" or "short"
            current_time: Current bar time (for time-based exits)
            details: Pattern-specific details (e.g., OR levels for ORB)
            vwap: Optional VWAP series (same length as bars) for VWAP cross exit

        Returns:
            List of ExitSignal objects (empty if no exits triggered)
        """
        signals = []
        df = bars.copy().reset_index(drop=True)
        n = len(df)

        if entry_idx >= n - 1:
            return signals  # Need at least one bar after entry

        # Check bars from entry onwards (including entry bar for stop check)
        # Entry bar itself can violate stop if it gaps down or wicks through
        post_entry_with_entry = df.iloc[entry_idx:]
        post_entry = df.iloc[entry_idx + 1:]

        # 1. Stop hit check (direction-aware) - includes entry bar
        stop_signal = self._check_stop_hit(post_entry_with_entry, stop_price, direction)
        if stop_signal:
            signals.append(stop_signal)

        # 2. MACD crossover (direction-aware)
        macd_signal = self._check_macd_cross(df, entry_idx, direction)
        if macd_signal:
            signals.append(macd_signal)

        # 3. VWAP crossover (direction-aware)
        if vwap is not None:
            vwap_signal = self._check_vwap_cross(df, entry_idx, vwap, direction)
            if vwap_signal:
                signals.append(vwap_signal)

        # 4. Volume decline (weakness) - applies to both directions
        vol_signal = self._check_volume_decline(df, entry_idx)
        if vol_signal:
            signals.append(vol_signal)

        # 5. Jackknife/Bottoming rejection (direction-aware)
        rejection_signal = self._check_rejection(post_entry, direction)
        if rejection_signal:
            signals.append(rejection_signal)

        # 6. Topping/Bottoming tail (direction-aware)
        tail_signal = self._check_reversal_tail(post_entry, entry_price, direction)
        if tail_signal:
            signals.append(tail_signal)

        return signals

    def _check_stop_hit(
        self, post_entry: pd.DataFrame, stop_price: float, direction: str = "long"
    ) -> Optional[ExitSignal]:
        """Check if price hit stop loss (direction-aware)."""
        for idx, row in post_entry.iterrows():
            if direction == "short":
                # Shorts: stop hit when price goes UP through stop
                if row["high"] >= stop_price:
                    return ExitSignal(
                        signal_type="stop_hit",
                        triggered=True,
                        reason=f"Stop loss hit: high {row['high']:.2f} >= stop {stop_price:.2f}",
                        bar_idx=idx,
                        price=stop_price,
                    )
            else:
                # Longs: stop hit when price goes DOWN through stop
                if row["low"] <= stop_price:
                    return ExitSignal(
                        signal_type="stop_hit",
                        triggered=True,
                        reason=f"Stop loss hit: low {row['low']:.2f} <= stop {stop_price:.2f}",
                        bar_idx=idx,
                        price=stop_price,
                    )
        return None

    def _check_macd_cross(
        self, df: pd.DataFrame, entry_idx: int, direction: str = "long"
    ) -> Optional[ExitSignal]:
        """Check for adverse MACD crossover with confirmation bars (direction-aware).

        Instead of exiting immediately on cross, wait for N consecutive bars
        where MACD remains in adverse territory. This filters false signals.
        """
        macd = self.calculate_macd(df["close"])
        if macd is None:
            return None

        # Get confirmation bars from config (default: 1 = immediate exit)
        confirmation_bars = self.config.get("macd_exit_confirmation_bars", 1)

        # Need at least (confirmation_bars + 1) bars after entry
        if entry_idx >= len(df) - (confirmation_bars + 1):
            return None

        cross_bar_idx = None  # Bar where initial cross occurred
        consecutive_adverse = 0  # Count of consecutive adverse bars

        for i in range(entry_idx + 1, len(df)):
            if i < 1:
                continue

            curr_macd = macd.iloc[i]["macd"]
            curr_signal = macd.iloc[i]["signal"]

            if direction == "short":
                # For shorts: bullish (MACD > signal) is adverse
                is_adverse = curr_macd > curr_signal
            else:
                # For longs: bearish (MACD < signal) is adverse
                is_adverse = curr_macd < curr_signal

            if is_adverse:
                if cross_bar_idx is None:
                    # Check if this is a new cross (prev was not adverse)
                    if i > 0:
                        prev_macd = macd.iloc[i - 1]["macd"]
                        prev_signal = macd.iloc[i - 1]["signal"]
                        if direction == "short":
                            was_adverse = prev_macd > prev_signal
                        else:
                            was_adverse = prev_macd < prev_signal
                        if not was_adverse:
                            # This is the cross bar
                            cross_bar_idx = i
                            consecutive_adverse = 1
                else:
                    # Continuing adverse after initial cross
                    consecutive_adverse += 1

                # Check if we have enough confirmation
                if consecutive_adverse >= confirmation_bars:
                    if direction == "short":
                        reason = f"MACD crossed above signal line ({consecutive_adverse} bars confirmed)"
                    else:
                        reason = f"MACD crossed below signal line ({consecutive_adverse} bars confirmed)"
                    return ExitSignal(
                        signal_type="macd_cross",
                        triggered=True,
                        reason=reason,
                        bar_idx=i,
                        price=df.iloc[i]["close"],
                    )
            else:
                # MACD recovered - reset counter
                cross_bar_idx = None
                consecutive_adverse = 0

        return None

    def _check_vwap_cross(
        self,
        df: pd.DataFrame,
        entry_idx: int,
        vwap: pd.Series,
        direction: str = "long",
    ) -> Optional[ExitSignal]:
        """
        Check for adverse VWAP crossover with confirmation (direction-aware).

        For longs: exit when price closes below VWAP (losing institutional support)
        For shorts: exit when price closes above VWAP (losing institutional pressure)

        Uses confirmation bars from config to filter false signals.

        Args:
            df: OHLCV DataFrame
            entry_idx: Index of entry bar
            vwap: VWAP series (must be same length as df)
            direction: "long" or "short"

        Returns:
            ExitSignal if VWAP cross detected, None otherwise
        """
        if vwap is None or len(vwap) != len(df):
            return None

        # Get confirmation bars from config (default: 1 = immediate exit)
        confirmation_bars = self.config.get("vwap_exit_confirmation_bars", 1)

        # Need at least (confirmation_bars + 1) bars after entry
        if entry_idx >= len(df) - (confirmation_bars + 1):
            return None

        # Reset VWAP index to match df
        vwap = vwap.reset_index(drop=True)

        cross_bar_idx = None
        consecutive_adverse = 0

        for i in range(entry_idx + 1, len(df)):
            close = df.iloc[i]["close"]
            vwap_val = vwap.iloc[i]

            if direction == "short":
                # For shorts: price above VWAP is adverse
                is_adverse = close > vwap_val
            else:
                # For longs: price below VWAP is adverse
                is_adverse = close < vwap_val

            if is_adverse:
                if cross_bar_idx is None:
                    # Check if this is a new cross
                    if i > entry_idx:
                        prev_close = df.iloc[i - 1]["close"]
                        prev_vwap = vwap.iloc[i - 1]
                        if direction == "short":
                            was_adverse = prev_close > prev_vwap
                        else:
                            was_adverse = prev_close < prev_vwap
                        if not was_adverse:
                            # This is the cross bar
                            cross_bar_idx = i
                            consecutive_adverse = 1
                        else:
                            # Was already adverse, just count
                            consecutive_adverse += 1
                else:
                    consecutive_adverse += 1

                # Check if we have enough confirmation
                if consecutive_adverse >= confirmation_bars:
                    if direction == "short":
                        reason = f"VWAP cross: price {close:.2f} above VWAP {vwap_val:.2f} ({consecutive_adverse} bars)"
                    else:
                        reason = f"VWAP cross: price {close:.2f} below VWAP {vwap_val:.2f} ({consecutive_adverse} bars)"
                    return ExitSignal(
                        signal_type="vwap_cross",
                        triggered=True,
                        reason=reason,
                        bar_idx=i,
                        price=close,
                    )
            else:
                # Price recovered - reset counter
                cross_bar_idx = None
                consecutive_adverse = 0

        return None

    def _check_volume_decline(
        self, df: pd.DataFrame, entry_idx: int
    ) -> Optional[ExitSignal]:
        """
        Check for significant volume decline after entry.

        Declining volume on continuation = weakness, potential exit.
        Requires BOTH volume decline AND price stalling to avoid
        exiting positions that are still running on lighter volume.
        """
        if entry_idx >= len(df) - 3:
            return None  # Need at least 3 bars after entry

        entry_volume = df.iloc[entry_idx]["volume"]
        post_entry = df.iloc[entry_idx + 1:]

        if len(post_entry) < 3:
            return None

        # Check if last 3 bars have declining volume < 50% of entry
        recent_bars = post_entry.tail(3)
        recent_avg_vol = recent_bars["volume"].mean()

        if recent_avg_vol >= entry_volume * 0.5:
            return None

        # Volume is low — only exit if price is also stalling/declining
        # (don't exit if price is still making new highs on lighter volume)
        price_stalling = recent_bars.iloc[-1]["close"] <= recent_bars.iloc[0]["open"]

        if price_stalling:
            return ExitSignal(
                signal_type="volume_decline",
                triggered=True,
                reason=(
                    f"Volume declining: {recent_avg_vol:.0f} < 50% of entry vol "
                    f"{entry_volume:.0f}, price stalling"
                ),
                bar_idx=len(df) - 1,
                price=df.iloc[-1]["close"],
            )
        return None

    def _check_rejection(
        self, post_entry: pd.DataFrame, direction: str = "long"
    ) -> Optional[ExitSignal]:
        """
        Check for rejection pattern (direction-aware).

        For longs (jackknife): Price makes new high then reverses sharply,
        closing below the prior candle's low. Sign of trapped buyers.

        For shorts (bottoming): Price makes new low then reverses sharply,
        closing above the prior candle's high. Sign of trapped sellers.
        """
        if len(post_entry) < 2:
            return None

        for i in range(1, len(post_entry)):
            curr = post_entry.iloc[i]
            prev = post_entry.iloc[i - 1]

            if direction == "short":
                # Bottoming rejection (adverse for shorts):
                # 1. Current bar made a lower low (new low attempt)
                # 2. Current bar closes above prior bar's high (sharp rejection)
                # 3. Current bar is green (close > open)
                made_new_low = curr["low"] < prev["low"]
                closes_above_prior_high = curr["close"] > prev["high"]
                is_green = curr["close"] > curr["open"]

                if made_new_low and closes_above_prior_high and is_green:
                    return ExitSignal(
                        signal_type="bottoming_rejection",
                        triggered=True,
                        reason=f"Bottoming rejection: new low {curr['low']:.2f} then closed above prior high {prev['high']:.2f}",
                        bar_idx=post_entry.index[i],
                        price=curr["close"],
                    )
            else:
                # Jackknife rejection (adverse for longs):
                # 1. Current bar made a higher high (new high attempt)
                # 2. Current bar closes below prior bar's low (sharp rejection)
                # 3. Current bar is red (close < open)
                made_new_high = curr["high"] > prev["high"]
                closes_below_prior_low = curr["close"] < prev["low"]
                is_red = curr["close"] < curr["open"]

                if made_new_high and closes_below_prior_low and is_red:
                    return ExitSignal(
                        signal_type="jackknife",
                        triggered=True,
                        reason=f"Jackknife rejection: new high {curr['high']:.2f} then closed below prior low {prev['low']:.2f}",
                        bar_idx=post_entry.index[i],
                        price=curr["close"],
                    )
        return None

    def _check_reversal_tail(
        self, post_entry: pd.DataFrame, entry_price: float, direction: str = "long"
    ) -> Optional[ExitSignal]:
        """
        Check for reversal tail pattern (direction-aware).

        For longs (topping tail/shooting star): Long upper wick, body in lower third.
        Indicates rejection at highs - sellers pushing price down.

        For shorts (bottoming tail/hammer): Long lower wick, body in upper third.
        Indicates rejection at lows - buyers pushing price up.

        Only triggers when in profit (warning of potential reversal).
        """
        if len(post_entry) < 1:
            return None

        for i in range(len(post_entry)):
            bar = post_entry.iloc[i]

            high = bar["high"]
            low = bar["low"]
            open_price = bar["open"]
            close = bar["close"]

            # Calculate body and wicks
            body_top = max(open_price, close)
            body_bottom = min(open_price, close)
            body_size = body_top - body_bottom
            upper_wick = high - body_top
            lower_wick = body_bottom - low
            candle_range = high - low

            # Skip if no range (doji-like with no movement)
            if candle_range < 0.01:
                continue

            # Skip if body is zero (use small threshold)
            if body_size < 0.005:
                body_size = 0.005  # Prevent division by zero

            if direction == "short":
                # Bottoming tail (adverse for shorts): long lower wick, body in upper third
                lower_wick_ratio = lower_wick / body_size
                body_position = (body_bottom - low) / candle_range  # 0 = body at bottom, 1 = body at top

                is_bottoming_tail = (
                    lower_wick_ratio >= 2.0 and      # Long lower wick
                    body_position >= 0.67 and         # Body in upper third
                    close < entry_price               # We're in profit (price below entry for shorts)
                )

                if is_bottoming_tail:
                    return ExitSignal(
                        signal_type="bottoming_tail",
                        triggered=True,
                        reason=f"Bottoming tail: lower wick {lower_wick:.2f} ({lower_wick_ratio:.1f}x body), rejection at {low:.2f}",
                        bar_idx=post_entry.index[i],
                        price=close,
                    )
            else:
                # Topping tail (adverse for longs): long upper wick, body in lower third
                upper_wick_ratio = upper_wick / body_size
                body_position = (body_bottom - low) / candle_range

                is_topping_tail = (
                    upper_wick_ratio >= 2.0 and      # Long upper wick
                    body_position <= 0.33 and         # Body in lower third
                    close > entry_price               # We're in profit
                )

                if is_topping_tail:
                    return ExitSignal(
                        signal_type="topping_tail",
                        triggered=True,
                        reason=f"Topping tail: upper wick {upper_wick:.2f} ({upper_wick_ratio:.1f}x body), rejection at {high:.2f}",
                        bar_idx=post_entry.index[i],
                        price=close,
                    )

        return None
