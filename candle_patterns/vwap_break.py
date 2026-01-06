"""
VWAP Break Pattern Detector
===========================

Pattern based on price breaking above VWAP with volume confirmation.

Pattern Structure:
1. Stock trading below VWAP for a period
2. Strong break above VWAP with volume spike
3. Entry on close above VWAP or VWAP hold variant

Variants:
- VWAP Break: Direct break above VWAP
- VWAP Hold: Pullback to VWAP, holds as support, then continues
"""

from typing import Optional, Dict, Any
import pandas as pd
from .base import PatternDetector, PatternResult


class VWAPBreak(PatternDetector):
    """
    Detect VWAP Break pattern.

    VWAP (Volume Weighted Average Price) is a key institutional level.
    Breaking above it with conviction often leads to continuation.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            # Prior condition
            "prior_condition": "trading_below_vwap",
            "min_time_below_minutes": 5,  # Minimum 5 bars below VWAP

            # Entry trigger
            "entry": "break_above_vwap",
            "volume_spike_on_break": 2.0,  # 2x average volume
            "close_above_vwap": True,  # Candle must close above VWAP

            # VWAP Hold variant
            "vwap_hold_variant": {
                "enabled": True,
                "pullback_to_vwap": True,
                "holds_as_support": True,
                "entry": "first_green_after_hold",
            },

            # Confirmation
            "macd_positive": True,  # Advisory

            # Risk
            "stop_loss": "below_vwap",
            "stop_buffer_cents": 10,

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
        Detect VWAP Break pattern.

        Args:
            bars: OHLCV DataFrame (newest bar last)
            vwap: VWAP series (REQUIRED for this pattern)
            macd: Optional MACD DataFrame for confirmation

        Returns:
            PatternResult with detection details
        """
        try:
            self.validate_bars(bars)
        except ValueError as e:
            return self.not_detected(str(e))

        # VWAP is required for this pattern
        if vwap is None:
            return self.not_detected("VWAP series is required for VWAP Break pattern")

        df = bars.copy().reset_index(drop=True)
        n = len(df)

        if len(vwap) != n:
            return self.not_detected(f"VWAP length mismatch: {len(vwap)} != {n}")

        # Align VWAP with DataFrame
        df["vwap"] = vwap.values

        # Step 1: Check if there was a period below VWAP
        below_vwap_result = self._find_below_vwap_period(df)
        if below_vwap_result is None:
            return self.not_detected("No period of trading below VWAP found")

        below_start_idx, below_end_idx, bars_below = below_vwap_result

        # Step 2: Check for break above VWAP
        break_result = self._check_vwap_break(df, below_end_idx)
        if break_result is None:
            # Try VWAP Hold variant
            hold_result = self._check_vwap_hold(df, below_end_idx)
            if hold_result is None:
                return self.not_detected("No VWAP break or hold pattern found")
            else:
                return self._create_hold_result(df, hold_result, macd)

        break_idx, volume_spike = break_result

        # Step 3: Calculate entry and stop
        current_vwap = df.iloc[-1]["vwap"]
        entry_price = current_vwap + 0.02  # 2 cents above VWAP
        stop_price = current_vwap - (self.config["stop_buffer_cents"] / 100)
        stop_distance_cents = (entry_price - stop_price) * 100

        # Step 3b: Enforce minimum R:R
        # Estimate profit target based on recapture move (low below VWAP to current VWAP)
        below_period_low = df.iloc[below_start_idx:below_end_idx + 1]["low"].min()
        recapture_move = current_vwap - below_period_low
        estimated_target = entry_price + recapture_move  # Expect continuation of similar magnitude
        risk = entry_price - stop_price
        if risk > 0:
            estimated_rr = (estimated_target - entry_price) / risk
            min_rr = self.config.get("min_rr_for_setup", 2.0)
            if estimated_rr < min_rr:
                return self.not_detected(
                    f"R:R too low: {estimated_rr:.1f} < {min_rr}"
                )

        # Step 4: Confirmations
        # Auto-calculate MACD if not provided and enough bars
        if macd is None:
            macd = self.calculate_macd(df["close"])

        macd_positive = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0

        # Calculate confidence
        confidence = 0.65  # Base confidence
        if volume_spike:
            confidence += 0.20
        if macd_positive:
            confidence += 0.15

        return PatternResult(
            detected=True,
            pattern_name="VWAPBreak",
            confidence=min(confidence, 1.0),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=below_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - below_start_idx,
            above_vwap=True,  # By definition
            macd_positive=macd_positive,
            volume_confirmation=volume_spike,
            reason="VWAP Break detected",
            details={
                "bars_below_vwap": bars_below,
                "break_bar_idx": break_idx,
                "volume_spike": volume_spike,
                "current_vwap": current_vwap,
            },
        )

    def _find_below_vwap_period(
        self, df: pd.DataFrame
    ) -> Optional[tuple]:
        """
        Find the most recent period where price traded below VWAP.

        Returns:
            Tuple of (start_idx, end_idx, num_bars) or None
        """
        n = len(df)
        min_bars = self.config.get("min_time_below_minutes", 5)

        # Find all below-VWAP periods, return the most recent one
        periods = []
        below_count = 0
        below_start = None

        for i in range(n - 1):  # Exclude last bar (potential break bar)
            if df.iloc[i]["close"] < df.iloc[i]["vwap"]:
                if below_start is None:
                    below_start = i
                below_count += 1
            else:
                # End of a below-VWAP period
                if below_count >= min_bars:
                    periods.append((below_start, i - 1, below_count))
                below_start = None
                below_count = 0

        # Check if we ended below VWAP
        if below_count >= min_bars:
            periods.append((below_start, n - 2, below_count))

        # Return the most recent period (last in list)
        if periods:
            return periods[-1]

        return None

    def _check_vwap_break(
        self, df: pd.DataFrame, after_idx: int
    ) -> Optional[tuple]:
        """
        Check for a break above VWAP.

        Returns:
            Tuple of (break_idx, volume_spike) or None
        """
        n = len(df)

        # Calculate average volume
        avg_volume = df["volume"].mean()
        volume_threshold = avg_volume * self.config["volume_spike_on_break"]

        # Look for break in bars after the below period
        for i in range(after_idx + 1, n):
            bar = df.iloc[i]

            # Check if close is above VWAP
            if bar["close"] > bar["vwap"]:
                # Check for volume spike
                volume_spike = bar["volume"] >= volume_threshold

                if self.config["close_above_vwap"]:
                    return (i, volume_spike)

        return None

    def _check_vwap_hold(
        self, df: pd.DataFrame, after_idx: int
    ) -> Optional[dict]:
        """
        Check for VWAP Hold variant (pullback to VWAP, holds as support).

        Returns:
            Dict with hold details or None
        """
        if not self.config["vwap_hold_variant"]["enabled"]:
            return None

        n = len(df)

        # Look for pullback to VWAP that holds
        for i in range(after_idx + 1, n - 1):
            bar = df.iloc[i]
            next_bar = df.iloc[i + 1] if i + 1 < n else None

            # Check if low touches VWAP (within 0.5%)
            vwap = bar["vwap"]
            touch_threshold = vwap * 0.005  # 0.5%

            if abs(bar["low"] - vwap) <= touch_threshold:
                # VWAP touched, check if it held (next bar green and above VWAP)
                if next_bar is not None:
                    if (
                        next_bar["close"] > next_bar["open"]  # Green
                        and next_bar["close"] > next_bar["vwap"]  # Above VWAP
                    ):
                        return {
                            "touch_idx": i,
                            "entry_idx": i + 1,
                            "touch_low": bar["low"],
                            "vwap_at_touch": vwap,
                        }

        return None

    def _create_hold_result(
        self,
        df: pd.DataFrame,
        hold_result: dict,
        macd: Optional[pd.DataFrame],
    ) -> PatternResult:
        """Create PatternResult for VWAP Hold variant."""
        n = len(df)
        entry_idx = hold_result["entry_idx"]

        current_vwap = df.iloc[-1]["vwap"]
        entry_price = current_vwap + 0.02
        stop_price = hold_result["touch_low"] - 0.05
        stop_distance_cents = (entry_price - stop_price) * 100

        # Enforce minimum R:R for hold pattern
        # Estimate target based on bounce magnitude (VWAP to touch low distance)
        bounce_potential = current_vwap - hold_result["touch_low"]
        estimated_target = entry_price + bounce_potential
        risk = entry_price - stop_price
        if risk > 0:
            estimated_rr = (estimated_target - entry_price) / risk
            min_rr = self.config.get("min_rr_for_setup", 2.0)
            if estimated_rr < min_rr:
                return self.not_detected(
                    f"R:R too low for VWAP Hold: {estimated_rr:.1f} < {min_rr}"
                )

        # Auto-calculate MACD if not provided and enough bars
        if macd is None:
            macd = self.calculate_macd(df["close"])

        macd_positive = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0

        confidence = 0.70  # Higher confidence for hold pattern

        return PatternResult(
            detected=True,
            pattern_name="VWAPHold",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=hold_result["touch_idx"],
            pattern_end_idx=n - 1,
            candle_count=n - hold_result["touch_idx"],
            above_vwap=True,
            macd_positive=macd_positive,
            volume_confirmation=None,
            reason="VWAP Hold pattern detected",
            details={
                "variant": "hold",
                "touch_idx": hold_result["touch_idx"],
                "entry_idx": hold_result["entry_idx"],
                "vwap_at_touch": hold_result["vwap_at_touch"],
            },
        )
