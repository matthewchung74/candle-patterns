"""
Micro Pullback Pattern Detector
===============================

Ross Cameron's favorite pattern for momentum stocks.

Pattern Structure:
1. Strong prior move (3+ green candles, 5%+ gain)
2. Very shallow pullback (1-2 red candles, max 3% retracement)
3. Entry on first candle making new high

Example:
    [GREEN][GREEN][GREEN][GREEN][red][red][GREEN→ENTRY]
         Prior Move (5%+)        Pullback   New High
"""

from typing import Optional, Dict, Any
import pandas as pd
from .base import PatternDetector, PatternResult


class MicroPullback(PatternDetector):
    """
    Detect Micro Pullback pattern.

    This is a shallow retracement after a strong move, indicating
    continuation rather than reversal.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration matching Ross Cameron's criteria."""
        return {
            # Prior move requirements
            "min_prior_move_pct": 5.0,  # Min 5% move before pullback
            "min_green_candles_prior": 3,  # At least 3 green candles in surge

            # Pullback requirements
            "max_pullback_candles": 2,  # Only 1-2 red candles
            "max_pullback_pct": 3.0,  # Max 3% retracement

            # Entry trigger
            "entry": "first_candle_new_high",

            # Confirmation (advisory, not hard gates)
            "price_above_vwap": True,
            "macd_positive": True,

            # Risk
            "stop_loss_cents": 15,  # Tighter stop for micro pullback

            # Minimum bars needed
            "min_bars_required": 6,

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
        Detect Micro Pullback pattern.

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

        # Work with a copy to avoid modifying original
        df = bars.copy().reset_index(drop=True)
        n = len(df)

        # Need at least 6 bars: 3 green + 2 pullback + 1 potential entry
        if n < 6:
            return self.not_detected(f"Insufficient bars: {n}")

        # Mark green/red candles
        df["is_green"] = df["close"] > df["open"]

        # Step 1: Find the pullback (look for recent red candles)
        pullback_end_idx = n - 1
        pullback_start_idx = None

        # Count consecutive red candles from the end (going backwards)
        # But the LAST bar should be green (potential entry)
        if df.iloc[-1]["is_green"]:
            # Last bar is green - potential entry candle
            # Look for pullback before it
            red_count = 0
            for i in range(n - 2, -1, -1):
                if not df.iloc[i]["is_green"]:  # Red candle
                    red_count += 1
                    if pullback_start_idx is None:
                        pullback_end_idx = i
                    pullback_start_idx = i
                else:
                    break  # Hit a green candle, pullback ends

            if red_count == 0:
                return self.not_detected("No pullback found (no red candles)")

            if red_count > self.config["max_pullback_candles"]:
                return self.not_detected(
                    f"Pullback too long: {red_count} candles > {self.config['max_pullback_candles']}"
                )
        else:
            # Last bar is red - pattern not complete yet
            return self.not_detected("Last candle is red - waiting for green entry candle")

        # Step 2: Find the prior move (green candles before pullback)
        if pullback_start_idx is None or pullback_start_idx < 3:
            return self.not_detected("Not enough bars before pullback")

        prior_move_end_idx = pullback_start_idx - 1
        prior_move_start_idx = None
        green_count = 0

        for i in range(prior_move_end_idx, -1, -1):
            if df.iloc[i]["is_green"]:
                green_count += 1
                prior_move_start_idx = i
            else:
                break  # Hit a red candle

        if green_count < self.config["min_green_candles_prior"]:
            return self.not_detected(
                f"Prior move too short: {green_count} green candles < {self.config['min_green_candles_prior']}"
            )

        # Step 3: Calculate prior move percentage
        move_start_price = df.iloc[prior_move_start_idx]["open"]
        move_end_price = df.iloc[prior_move_end_idx]["high"]
        prior_move_pct = self.calculate_move_pct(move_start_price, move_end_price)

        if prior_move_pct < self.config["min_prior_move_pct"]:
            return self.not_detected(
                f"Prior move too small: {prior_move_pct:.1f}% < {self.config['min_prior_move_pct']}%"
            )

        # Step 4: Calculate pullback percentage
        pullback_high = df.iloc[pullback_start_idx:pullback_end_idx + 1]["high"].max()
        pullback_low = df.iloc[pullback_start_idx:pullback_end_idx + 1]["low"].min()
        pullback_pct = self.calculate_move_pct(move_end_price, pullback_low)

        if abs(pullback_pct) > self.config["max_pullback_pct"]:
            return self.not_detected(
                f"Pullback too deep: {abs(pullback_pct):.1f}% > {self.config['max_pullback_pct']}%"
            )

        # Step 5: Check if entry candle makes new high
        entry_candle = df.iloc[-1]
        prior_high = df.iloc[prior_move_end_idx]["high"]

        if entry_candle["high"] <= prior_high:
            return self.not_detected(
                f"Entry candle not making new high: {entry_candle['high']:.2f} <= {prior_high:.2f}"
            )

        # Step 6: Calculate entry and stop
        entry_price = prior_high + 0.01  # 1 cent above prior high
        stop_price = pullback_low - (self.config["stop_loss_cents"] / 100)
        stop_distance_cents = (entry_price - stop_price) * 100

        # Step 7: Check pullback volume vs surge volume
        surge_volume = df.iloc[prior_move_start_idx:prior_move_end_idx + 1]["volume"].mean()
        pullback_volume = df.iloc[pullback_start_idx:pullback_end_idx + 1]["volume"].mean()
        volume_declining = pullback_volume < surge_volume

        # Step 8: Confirmations (advisory)
        above_vwap = None
        if vwap is not None and len(vwap) == n:
            above_vwap = entry_candle["close"] > vwap.iloc[-1]

        # Auto-calculate MACD if not provided and enough bars
        if macd is None:
            macd = self.calculate_macd(df["close"])

        macd_positive = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0

        # Calculate confidence based on confirmations
        confidence = 0.7  # Base confidence
        if volume_declining:
            confidence += 0.10
        if above_vwap:
            confidence += 0.10
        if macd_positive:
            confidence += 0.10

        return PatternResult(
            detected=True,
            pattern_name="MicroPullback",
            confidence=min(confidence, 1.0),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=prior_move_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - prior_move_start_idx,
            above_vwap=above_vwap,
            macd_positive=macd_positive,
            volume_confirmation=volume_declining,
            reason="Pattern detected",
            details={
                "prior_move_pct": prior_move_pct,
                "pullback_pct": abs(pullback_pct),
                "green_candles": green_count,
                "pullback_candles": pullback_end_idx - pullback_start_idx + 1,
                "prior_high": prior_high,
                "pullback_low": pullback_low,
                "surge_volume_avg": surge_volume,
                "pullback_volume_avg": pullback_volume,
                "volume_declining": volume_declining,
            },
        )
