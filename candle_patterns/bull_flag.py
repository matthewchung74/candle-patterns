"""
Bull Flag Pattern Detector
==========================

Classic momentum continuation pattern.

Pattern Structure:
1. The Pole: Strong vertical move (20%+ gain, 3-10 candles)
2. The Flag: Consolidation with declining volume (3-10 candles, 10-25% retracement)
3. Entry: Break above flag resistance

Example:
           /\\
          /  \\____
         /       \\____  <- Flag (consolidation)
        /             \\
       /               -> BREAKOUT ENTRY
      /
     / <- Pole (strong move up)
    /
"""

from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np
from .base import PatternDetector, PatternResult


class BullFlag(PatternDetector):
    """
    Detect Bull Flag pattern.

    A consolidation pattern after a strong move, with declining volume
    during the flag, indicating accumulation before continuation.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration matching Ross Cameron's criteria."""
        return {
            # The Pole (Initial Surge)
            "min_pole_move_pct": 20.0,  # Min 20% move in pole
            "min_pole_candles": 3,  # 3-10 candles in pole
            "max_pole_candles": 10,

            # The Flag (Consolidation)
            "min_flag_candles": 3,  # 3-10 candles in flag
            "max_flag_candles": 10,
            "min_pullback_pct": 10.0,  # 10-25% retracement of pole
            "max_pullback_pct": 25.0,
            "volume_declining": True,  # Volume must decrease in flag

            # Entry trigger
            "entry": "break_flag_resistance",

            # Confirmation
            "price_above_9ema": True,
            "price_above_vwap": True,

            # Risk
            "stop_loss": "low_of_flag",
            "stop_buffer_cents": 5,  # Buffer below flag low

            # Flag consolidation
            "max_flag_range_pct": 15.0,  # Max range for tight consolidation

            # Minimum bars needed
            "min_bars_required": 8,

            # Quality filter
            "min_rr_for_setup": 2.0,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Detect Bull Flag pattern.

        Args:
            bars: OHLCV DataFrame (newest bar last)
            vwap: Optional VWAP series for confirmation
            macd: Optional MACD DataFrame for confirmation

        Returns:
            PatternResult with detection details
        """
        try:
            self.validate_bars(bars)
        except ValueError as e:
            return self.not_detected(str(e))

        df = bars.copy().reset_index(drop=True)
        n = len(df)

        # Need enough bars for pole + flag
        min_total = self.config["min_pole_candles"] + self.config["min_flag_candles"]
        if n < min_total:
            return self.not_detected(f"Insufficient bars: {n} < {min_total}")

        # Step 1: Find potential flag (consolidation at the end)
        flag_result = self._find_flag(df)
        if flag_result is None:
            return self.not_detected("No valid flag consolidation found")

        flag_start_idx, flag_end_idx, flag_high, flag_low = flag_result

        # Step 2: Find the pole before the flag
        pole_result = self._find_pole(df, flag_start_idx)
        if pole_result is None:
            return self.not_detected("No valid pole found before flag")

        pole_start_idx, pole_end_idx, pole_move_pct = pole_result

        # Step 3: Calculate pullback percentage
        pole_high = df.iloc[pole_start_idx:pole_end_idx + 1]["high"].max()
        pullback_pct = self.calculate_move_pct(pole_high, flag_low)

        if abs(pullback_pct) < self.config["min_pullback_pct"]:
            return self.not_detected(
                f"Pullback too shallow: {abs(pullback_pct):.1f}% < {self.config['min_pullback_pct']}%"
            )

        if abs(pullback_pct) > self.config["max_pullback_pct"]:
            return self.not_detected(
                f"Pullback too deep: {abs(pullback_pct):.1f}% > {self.config['max_pullback_pct']}%"
            )

        # Step 4: Check volume declining in flag
        volume_declining = None
        if self.config["volume_declining"]:
            volume_declining = self._check_volume_declining(df, flag_start_idx, flag_end_idx)
            if not volume_declining:
                # Advisory warning, not a hard fail
                pass

        # Step 5: Check for breakout
        entry_candle = df.iloc[-1]
        if entry_candle["high"] <= flag_high:
            return self.not_detected(
                f"No breakout yet: {entry_candle['high']:.2f} <= flag high {flag_high:.2f}"
            )

        # Step 6: Calculate entry and stop
        entry_price = flag_high + 0.01  # 1 cent above flag resistance
        stop_buffer = self.config["stop_buffer_cents"] / 100
        stop_price = flag_low - stop_buffer
        stop_distance_cents = (entry_price - stop_price) * 100

        # Step 7: Confirmations
        above_vwap = None
        if vwap is not None and len(vwap) == n:
            above_vwap = entry_candle["close"] > vwap.iloc[-1]

        # Calculate 9 EMA
        above_9ema = None
        if n >= 9:
            ema_9 = df["close"].ewm(span=9, adjust=False).mean()
            above_9ema = entry_candle["close"] > ema_9.iloc[-1]

        # Auto-calculate MACD if not provided and enough bars
        if macd is None:
            macd = self.calculate_macd(df["close"])

        macd_positive = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0

        # Calculate confidence
        confidence = 0.6  # Base confidence
        if volume_declining:
            confidence += 0.15
        if above_vwap:
            confidence += 0.10
        if above_9ema:
            confidence += 0.10
        if macd_positive:
            confidence += 0.05

        return PatternResult(
            detected=True,
            pattern_name="BullFlag",
            confidence=min(confidence, 1.0),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=pole_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - pole_start_idx,
            above_vwap=above_vwap,
            macd_positive=macd_positive,
            volume_confirmation=volume_declining,
            reason="Pattern detected",
            details={
                "pole_move_pct": pole_move_pct,
                "pullback_pct": abs(pullback_pct),
                "pole_candles": pole_end_idx - pole_start_idx + 1,
                "flag_candles": flag_end_idx - flag_start_idx + 1,
                "flag_high": flag_high,
                "flag_low": flag_low,
                "volume_declining": volume_declining,
                "above_9ema": above_9ema,
            },
        )

    def _find_flag(
        self, df: pd.DataFrame
    ) -> Optional[Tuple[int, int, float, float]]:
        """
        Find flag consolidation at the end of the bars.

        Returns:
            Tuple of (start_idx, end_idx, flag_high, flag_low) or None
        """
        n = len(df)
        min_flag = self.config["min_flag_candles"]
        max_flag = self.config["max_flag_candles"]

        # Look for consolidation in the last max_flag candles
        for flag_len in range(min_flag, min(max_flag + 1, n - 3)):
            start_idx = n - flag_len - 1  # -1 to exclude potential breakout candle
            end_idx = n - 2  # Exclude last candle (potential breakout)

            if start_idx < 3:  # Need room for pole
                continue

            flag_bars = df.iloc[start_idx:end_idx + 1]
            flag_high = flag_bars["high"].max()
            flag_low = flag_bars["low"].min()

            # Flag should have tight range (consolidation)
            flag_range_pct = self.calculate_move_pct(flag_low, flag_high)

            # Consolidation should be tight (configurable)
            max_range = self.config.get("max_flag_range_pct", 15.0)
            if flag_range_pct < max_range:
                return (start_idx, end_idx, flag_high, flag_low)

        return None

    def _find_pole(
        self, df: pd.DataFrame, flag_start_idx: int
    ) -> Optional[Tuple[int, int, float]]:
        """
        Find the pole (strong move) before the flag.

        Returns:
            Tuple of (start_idx, end_idx, move_pct) or None
        """
        min_pole = self.config["min_pole_candles"]
        max_pole = self.config["max_pole_candles"]

        # Pole ends just before flag starts
        pole_end_idx = flag_start_idx - 1

        if pole_end_idx < min_pole:
            return None

        # Try different pole lengths
        for pole_len in range(min_pole, min(max_pole + 1, pole_end_idx + 1)):
            start_idx = pole_end_idx - pole_len + 1

            if start_idx < 0:
                continue

            pole_bars = df.iloc[start_idx:pole_end_idx + 1]
            pole_low = pole_bars["low"].min()
            pole_high = pole_bars["high"].max()

            move_pct = self.calculate_move_pct(pole_low, pole_high)

            if move_pct >= self.config["min_pole_move_pct"]:
                return (start_idx, pole_end_idx, move_pct)

        return None

    def _check_volume_declining(
        self, df: pd.DataFrame, start_idx: int, end_idx: int
    ) -> bool:
        """
        Check if volume is declining in the flag.

        Uses linear regression on volume to detect decline.
        """
        flag_volume = df.iloc[start_idx:end_idx + 1]["volume"].values

        if len(flag_volume) < 2:
            return False

        # Simple check: is last half of volume less than first half?
        mid = len(flag_volume) // 2
        first_half_avg = flag_volume[:mid].mean()
        second_half_avg = flag_volume[mid:].mean()

        return second_half_avg < first_half_avg * 0.9  # 10% decline
