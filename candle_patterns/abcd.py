"""
ABCD Pattern Detector
=====================

Harmonic pattern detector for identifying ABCD setups.

Pattern Structure (Bullish):
    A - Swing low (start of impulse)
    B - Swing high (end of AB leg)
    C - Higher low (BC retracement of 38.2%-78.6% of AB)
    D - Projected target where CD ≈ AB

Pattern Structure (Bearish):
    A - Swing high (start of impulse)
    B - Swing low (end of AB leg)
    C - Lower high (BC retracement of 38.2%-78.6% of AB)
    D - Projected target where CD ≈ AB

Entry: At D completion (price reaches projected D level)
Stop: Below C (bullish) or above C (bearish)

References:
- Fibonacci retracement levels for BC leg
- AB=CD harmonic relationship
"""

from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from .base import PatternDetector, PatternResult


class ABCD(PatternDetector):
    """
    Detect ABCD harmonic pattern.

    The ABCD pattern is a measured move pattern where the CD leg
    mirrors the AB leg, with a Fibonacci retracement at point C.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration for ABCD pattern detection."""
        return {
            # Minimum bars required for pattern detection
            "min_bars_required": 10,

            # Swing detection parameters
            "swing_lookback": 3,  # Bars to look back/forward for swing confirmation

            # BC retracement limits (Fibonacci)
            "min_bc_retracement": 0.382,  # Minimum 38.2% retracement
            "max_bc_retracement": 0.786,  # Maximum 78.6% retracement

            # CD leg requirements
            "cd_ab_ratio_min": 0.75,  # CD must be at least 75% of AB
            "cd_ab_ratio_max": 1.25,  # CD must be at most 125% of AB
            "cd_min_completion": 0.80,  # CD must be at least 80% developed

            # Minimum leg size (as percentage of price)
            "min_leg_pct": 1.0,  # AB leg must be at least 1% move

            # Stop buffer
            "stop_buffer_pct": 0.5,  # 0.5% below/above C for stop

            # Pattern completion tolerance
            "d_completion_tolerance": 0.02,  # 2% tolerance for D level

            # Direction filter (None = detect both, "long" or "short")
            "direction_filter": None,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Detect ABCD pattern in the given bars.

        Args:
            bars: OHLCV DataFrame (newest bar last)
            vwap: Optional VWAP series (unused, for interface compatibility)
            macd: Optional MACD DataFrame (unused, for interface compatibility)

        Returns:
            PatternResult with detection details including A/B/C indices
            and projected D level
        """
        try:
            self.validate_bars(bars)
        except ValueError as e:
            return self.not_detected(str(e))

        df = bars.copy().reset_index(drop=True)
        n = len(df)

        # Find swing points
        swing_highs = self._find_swing_highs(df)
        swing_lows = self._find_swing_lows(df)

        if len(swing_highs) < 1 or len(swing_lows) < 1:
            return self.not_detected("Insufficient swing points")

        # Try to find bullish ABCD (A=low, B=high, C=higher low)
        direction_filter = self.config.get("direction_filter")

        if direction_filter is None or direction_filter == "long":
            bullish_result = self._find_bullish_abcd(df, swing_highs, swing_lows)
            if bullish_result:
                return bullish_result

        # Try to find bearish ABCD (A=high, B=low, C=lower high)
        if direction_filter is None or direction_filter == "short":
            bearish_result = self._find_bearish_abcd(df, swing_highs, swing_lows)
            if bearish_result:
                return bearish_result

        return self.not_detected("No valid ABCD pattern found")

    def _find_swing_highs(self, df: pd.DataFrame) -> List[int]:
        """Find swing high indices."""
        lookback = self.config["swing_lookback"]
        swing_highs = []

        for i in range(lookback, len(df) - lookback):
            # Skip zero-volume bars (trading halts produce phantom swings)
            if df.iloc[i]["volume"] <= 0:
                continue

            is_swing = True
            current_high = df.iloc[i]["high"]

            # Check if higher than surrounding bars
            for j in range(1, lookback + 1):
                if df.iloc[i - j]["high"] >= current_high:
                    is_swing = False
                    break
                if df.iloc[i + j]["high"] >= current_high:
                    is_swing = False
                    break

            if is_swing:
                swing_highs.append(i)

        return swing_highs

    def _find_swing_lows(self, df: pd.DataFrame) -> List[int]:
        """Find swing low indices."""
        lookback = self.config["swing_lookback"]
        swing_lows = []

        for i in range(lookback, len(df) - lookback):
            # Skip zero-volume bars (trading halts produce phantom swings)
            if df.iloc[i]["volume"] <= 0:
                continue

            is_swing = True
            current_low = df.iloc[i]["low"]

            # Check if lower than surrounding bars
            for j in range(1, lookback + 1):
                if df.iloc[i - j]["low"] <= current_low:
                    is_swing = False
                    break
                if df.iloc[i + j]["low"] <= current_low:
                    is_swing = False
                    break

            if is_swing:
                swing_lows.append(i)

        return swing_lows

    def _find_bullish_abcd(
        self,
        df: pd.DataFrame,
        swing_highs: List[int],
        swing_lows: List[int],
    ) -> Optional[PatternResult]:
        """
        Find bullish ABCD pattern.

        Bullish: A=swing low, B=swing high, C=higher swing low
        Entry long at D (projected level where CD = AB)
        """
        n = len(df)

        # Need at least 2 swing lows and 1 swing high
        if len(swing_lows) < 2 or len(swing_highs) < 1:
            return None

        # Try combinations, prioritizing most recent patterns
        for c_idx in reversed(swing_lows):
            for b_idx in reversed(swing_highs):
                if b_idx >= c_idx:
                    continue  # B must be before C

                for a_idx in swing_lows:
                    if a_idx >= b_idx:
                        continue  # A must be before B

                    result = self._validate_bullish_abcd(df, a_idx, b_idx, c_idx)
                    if result:
                        return result

        return None

    def _find_bearish_abcd(
        self,
        df: pd.DataFrame,
        swing_highs: List[int],
        swing_lows: List[int],
    ) -> Optional[PatternResult]:
        """
        Find bearish ABCD pattern.

        Bearish: A=swing high, B=swing low, C=lower swing high
        Entry short at D (projected level where CD = AB)
        """
        n = len(df)

        # Need at least 2 swing highs and 1 swing low
        if len(swing_highs) < 2 or len(swing_lows) < 1:
            return None

        # Try combinations, prioritizing most recent patterns
        for c_idx in reversed(swing_highs):
            for b_idx in reversed(swing_lows):
                if b_idx >= c_idx:
                    continue  # B must be before C

                for a_idx in swing_highs:
                    if a_idx >= b_idx:
                        continue  # A must be before B

                    result = self._validate_bearish_abcd(df, a_idx, b_idx, c_idx)
                    if result:
                        return result

        return None

    def _validate_bullish_abcd(
        self,
        df: pd.DataFrame,
        a_idx: int,
        b_idx: int,
        c_idx: int,
    ) -> Optional[PatternResult]:
        """Validate a potential bullish ABCD pattern."""
        a_price = df.iloc[a_idx]["low"]
        b_price = df.iloc[b_idx]["high"]
        c_price = df.iloc[c_idx]["low"]

        # AB leg (upward move)
        ab_move = b_price - a_price

        # Check minimum leg size
        min_leg_pct = self.config["min_leg_pct"]
        if (ab_move / a_price) * 100 < min_leg_pct:
            return None

        # BC retracement (downward from B to C)
        bc_move = b_price - c_price

        # C must be above A (higher low)
        if c_price <= a_price:
            return None

        # Calculate BC retracement ratio
        bc_retracement = bc_move / ab_move if ab_move != 0 else 0

        # Validate BC retracement is within Fibonacci range
        min_ret = self.config["min_bc_retracement"]
        max_ret = self.config["max_bc_retracement"]
        if bc_retracement < min_ret or bc_retracement > max_ret:
            return None

        # Project D level (CD = AB, so D = C + AB)
        projected_d = c_price + ab_move

        # Check if current price is near projected D
        current_price = df.iloc[-1]["close"]
        d_tolerance = self.config["d_completion_tolerance"]
        d_range_low = projected_d * (1 - d_tolerance)
        d_range_high = projected_d * (1 + d_tolerance)

        # Pattern is "completing" if price is approaching or at D level
        # For detection, we check if current price >= C (moving toward D)
        if current_price < c_price:
            return None  # Price hasn't bounced from C yet

        # Calculate CD move so far
        cd_move = current_price - c_price
        cd_ab_ratio = cd_move / ab_move if ab_move != 0 else 0

        # Check if CD is within acceptable range of AB
        ratio_min = self.config["cd_ab_ratio_min"]
        ratio_max = self.config["cd_ab_ratio_max"]

        # Pattern detected if CD is developing (at least 80% of expected)
        min_completion = self.config.get("cd_min_completion", 0.80)
        if cd_ab_ratio < min_completion:
            return None  # CD leg not developed enough

        # Entry at current price (or projected D)
        entry_price = current_price if current_price < projected_d else projected_d

        # Stop below C with buffer
        stop_buffer = c_price * (self.config["stop_buffer_pct"] / 100)
        stop_price = c_price - stop_buffer

        # Calculate confidence based on how close CD is to AB
        # Perfect AB=CD gives highest confidence
        cd_ab_diff = abs(1.0 - cd_ab_ratio)
        confidence = max(0.5, min(0.95, 0.85 - cd_ab_diff * 0.5))

        # Adjust confidence based on BC retracement quality
        # Ideal retracement is 61.8%
        ideal_ret = 0.618
        ret_diff = abs(bc_retracement - ideal_ret)
        confidence -= ret_diff * 0.1

        return PatternResult(
            detected=True,
            pattern_name="ABCD",
            confidence=max(0.5, min(0.95, confidence)),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=(entry_price - stop_price) * 100,
            pattern_start_idx=a_idx,
            pattern_end_idx=len(df) - 1,
            candle_count=len(df) - a_idx,
            details={
                "direction": "long",
                "a_idx": a_idx,
                "b_idx": b_idx,
                "c_idx": c_idx,
                "a_price": a_price,
                "b_price": b_price,
                "c_price": c_price,
                "projected_d": projected_d,
                "ab_move": ab_move,
                "bc_retracement": bc_retracement,
                "cd_ab_ratio": cd_ab_ratio,
            },
        )

    def _validate_bearish_abcd(
        self,
        df: pd.DataFrame,
        a_idx: int,
        b_idx: int,
        c_idx: int,
    ) -> Optional[PatternResult]:
        """Validate a potential bearish ABCD pattern."""
        a_price = df.iloc[a_idx]["high"]
        b_price = df.iloc[b_idx]["low"]
        c_price = df.iloc[c_idx]["high"]

        # AB leg (downward move)
        ab_move = a_price - b_price

        # Check minimum leg size
        min_leg_pct = self.config["min_leg_pct"]
        if (ab_move / a_price) * 100 < min_leg_pct:
            return None

        # BC retracement (upward from B to C)
        bc_move = c_price - b_price

        # C must be below A (lower high)
        if c_price >= a_price:
            return None

        # Calculate BC retracement ratio
        bc_retracement = bc_move / ab_move if ab_move != 0 else 0

        # Validate BC retracement is within Fibonacci range
        min_ret = self.config["min_bc_retracement"]
        max_ret = self.config["max_bc_retracement"]
        if bc_retracement < min_ret or bc_retracement > max_ret:
            return None

        # Project D level (CD = AB, so D = C - AB)
        projected_d = c_price - ab_move

        # Check if current price is approaching D
        current_price = df.iloc[-1]["close"]

        # For bearish, price should be below C (moving toward D)
        if current_price > c_price:
            return None  # Price hasn't dropped from C yet

        # Calculate CD move so far
        cd_move = c_price - current_price
        cd_ab_ratio = cd_move / ab_move if ab_move != 0 else 0

        # Check if CD is developing (at least 80% of expected)
        min_completion = self.config.get("cd_min_completion", 0.80)
        if cd_ab_ratio < min_completion:
            return None  # CD leg not developed enough

        # Entry at current price (or projected D)
        entry_price = current_price if current_price > projected_d else projected_d

        # Stop above C with buffer
        stop_buffer = c_price * (self.config["stop_buffer_pct"] / 100)
        stop_price = c_price + stop_buffer

        # Calculate confidence
        cd_ab_diff = abs(1.0 - cd_ab_ratio)
        confidence = max(0.5, min(0.95, 0.85 - cd_ab_diff * 0.5))

        ideal_ret = 0.618
        ret_diff = abs(bc_retracement - ideal_ret)
        confidence -= ret_diff * 0.1

        return PatternResult(
            detected=True,
            pattern_name="ABCD",
            confidence=max(0.5, min(0.95, confidence)),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=(stop_price - entry_price) * 100,
            pattern_start_idx=a_idx,
            pattern_end_idx=len(df) - 1,
            candle_count=len(df) - a_idx,
            details={
                "direction": "short",
                "a_idx": a_idx,
                "b_idx": b_idx,
                "c_idx": c_idx,
                "a_price": a_price,
                "b_price": b_price,
                "c_price": c_price,
                "projected_d": projected_d,
                "ab_move": ab_move,
                "bc_retracement": bc_retracement,
                "cd_ab_ratio": cd_ab_ratio,
            },
        )
