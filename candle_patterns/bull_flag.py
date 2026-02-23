"""
Bull Flag Pattern Detector
==========================

Classic momentum continuation pattern.

Pattern Structure:
1. The Pole: Strong vertical move (15%+ gain, 3-10 candles)
2. The Flag: Tight consolidation with declining volume (1-3 candles, 10-25% retracement)
3. Entry: Break above flag resistance (conservative)

Note: Prior moves <15% with shallow pullbacks are handled by Micro Pullback pattern.

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
        """Default configuration for bull flag detection."""
        return {
            # The Pole (Initial Surge) - 15%+ to separate from Micro Pullback
            "min_pole_move_pct": 15.0,  # Min 15% move in pole
            "min_pole_candles": 3,  # 3-10 candles in pole
            "max_pole_candles": 10,

            # The Flag (Tight Consolidation) - 1-3 candles for quick resolution
            "min_flag_candles": 1,  # Min 1 candle in flag
            "max_flag_candles": 3,  # Max 3 candles (tight flag)
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
            "stop_buffer_pct": 0.5,  # 0.5% below flag low
            "stop_buffer_min_cents": 5,  # Minimum 5 cents buffer

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

        # Step 5: Check for breakout (no lookahead bias)
        # Use PREVIOUS bar's close to confirm breakout (not current bar's high)
        # This ensures we only detect pattern AFTER breakout is confirmed
        prev_bar = df.iloc[-2]  # Previous bar (complete)
        entry_candle = df.iloc[-1]  # Current bar (for confirmations only)

        # Breakout confirmed if previous bar CLOSED above flag high
        # OR current bar OPENED above flag high (gap up breakout)
        breakout_confirmed = (prev_bar["close"] > flag_high) or (entry_candle["open"] > flag_high)

        if not breakout_confirmed:
            return self.not_detected(
                f"No confirmed breakout: prev close {prev_bar['close']:.2f}, "
                f"curr open {entry_candle['open']:.2f} <= flag high {flag_high:.2f}"
            )

        # Step 6: Calculate entry and stop
        entry_price = flag_high + 0.01  # 1 cent above flag resistance

        # Validate entry_price is within reasonable range of current price
        # This prevents stale bar data from causing invalid signals
        current_price = entry_candle["close"]
        max_entry_deviation_pct = self.config.get("max_entry_deviation_pct", 5.0)
        if entry_price > current_price * (1 + max_entry_deviation_pct / 100):
            return self.not_detected(
                f"Entry price {entry_price:.2f} too far from current {current_price:.2f} "
                f"(>{max_entry_deviation_pct}% deviation - possible stale data)"
            )

        pct_buffer = flag_low * (self.config["stop_buffer_pct"] / 100)
        min_buffer = self.config["stop_buffer_min_cents"] / 100
        stop_buffer = max(pct_buffer, min_buffer)
        stop_price = flag_low - stop_buffer
        stop_distance_cents = (entry_price - stop_price) * 100

        # Step 6b: Enforce minimum R:R
        # Estimate profit target based on pole move (flag breakout often equals pole)
        estimated_target = entry_price + (pole_move_pct / 100 * entry_price)
        risk = entry_price - stop_price
        if risk > 0:
            estimated_rr = (estimated_target - entry_price) / risk
            min_rr = self.config.get("min_rr_for_setup", 2.0)
            if estimated_rr < min_rr:
                return self.not_detected(
                    f"R:R too low: {estimated_rr:.1f} < {min_rr}"
                )

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
        macd_slope_up = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0
            # 3-bar MACD slope: compare current MACD line to 3 bars ago
            if "macd" in macd.columns and len(macd) >= 4:
                macd_slope_up = macd.iloc[-1]["macd"] > macd.iloc[-4]["macd"]

        # Calculate confidence (standardized system)
        # Base: 65%, Cap: 90%, Gate: 80% (enforced in trade_engine)
        confidence = 0.65  # Base confidence
        if volume_declining:
            confidence += 0.10
        if above_vwap:
            confidence += 0.08
        if above_9ema:
            confidence += 0.06
        if macd_positive:
            confidence += 0.08
        if macd_slope_up:
            confidence += 0.04

        return PatternResult(
            detected=True,
            pattern_name="BullFlag",
            confidence=min(confidence, 0.90),  # Cap at 90%
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=pole_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - pole_start_idx,
            above_vwap=above_vwap,
            macd_positive=macd_positive,
            macd_slope_up=macd_slope_up,
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
