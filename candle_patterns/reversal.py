"""
Reversal Pattern Detector
=========================

Detects bearish reversal patterns for short entry on extended stocks.

Pattern Structure:
1. Stock must be extended from open (>20% gain)
2. Stock must lack strong catalyst (opportunistic run-up)
3. Reversal pattern detected at or near high of day

Patterns Detected:
- Shooting Star: Long upper wick, small body in lower third (rejection at highs)
- Bearish Engulfing: Red candle fully engulfs prior green candle
- Evening Star: 3-bar pattern: green, small body (doji), red closes below green midpoint
- Volume Climax: Extreme volume (>3x avg) at highs with red reversal candle

Entry is on short side when pattern detected with sufficient confirmation.

Note: These patterns are for research/paper trading first. The system defaults
to long-only trading - short functionality must be explicitly enabled.
"""

from typing import Optional, Dict, Any
import pandas as pd
from .base import PatternDetector, PatternResult


class ReversalPatternDetector(PatternDetector):
    """
    Detect bearish reversal patterns for potential short entry.

    These patterns identify potential tops on extended stocks that
    have run up without strong fundamental catalyst.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration for reversal pattern detection.

        Tuned based on analysis of BOXL, TOPP, ELPW behavior at tops.
        """
        return {
            # Extension requirements (stock must be extended)
            "min_extension_from_open_pct": 20.0,  # Min 20% gain from open
            "min_extension_from_low_pct": 25.0,   # Min 25% gain from intraday low

            # Volume climax detection
            "volume_climax_multiplier": 3.0,  # Volume > 3x 20-bar avg
            "volume_avg_period": 20,           # Bars for average volume

            # Shooting star requirements
            "min_upper_wick_ratio": 2.0,       # Upper wick >= 2x body
            "max_body_position_pct": 33.0,     # Body must be in lower 33%

            # Bearish engulfing requirements
            "min_engulf_ratio": 1.0,           # Red body >= green body (fully engulfs)

            # Evening star requirements
            "max_middle_body_pct": 30.0,       # Middle candle body < 30% of range
            "min_close_below_midpoint": True,  # Final bar closes below bar[-3] midpoint

            # Hard gates
            "require_red_reversal": True,      # Must have red candle at/after top
            "require_volume_confirmation": False,  # Volume climax optional

            # Risk parameters
            "stop_buffer_pct": 2.0,            # Stop 2% above HOD
            "stop_buffer_min_cents": 5,        # Minimum 5 cents buffer

            # Quality filter
            "min_rr_for_setup": 2.0,           # Minimum R:R

            # Minimum bars needed
            "min_bars_required": 10,

            # Pattern-specific weights for confidence
            "shooting_star_weight": 0.85,
            "bearish_engulfing_weight": 0.80,
            "evening_star_weight": 0.90,
            "volume_climax_weight": 0.88,

            # MACD configuration
            "macd_exit_confirmation_bars": 1,
            "vwap_exit_confirmation_bars": 1,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Detect reversal pattern in the given bars.

        Checks for multiple bearish reversal patterns and returns
        the strongest match.

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

        if n < self.config["min_bars_required"]:
            return self.not_detected(f"Insufficient bars: {n}")

        # Step 1: Check if stock is extended
        open_price = df.iloc[0]["open"]
        current_price = df.iloc[-1]["close"]
        intraday_low = df["low"].min()
        intraday_high = df["high"].max()

        extension_from_open = self.calculate_move_pct(open_price, current_price)
        extension_from_low = self.calculate_move_pct(intraday_low, intraday_high)

        if extension_from_open < self.config["min_extension_from_open_pct"]:
            return self.not_detected(
                f"Not extended: {extension_from_open:.1f}% from open < {self.config['min_extension_from_open_pct']}%"
            )

        # Step 2: Check each reversal pattern (in order of strength)
        # Returns first match - patterns are mutually exclusive

        # Try evening star first (strongest, 3-bar pattern)
        evening_star_result = self._check_evening_star(df, vwap, macd)
        if evening_star_result.detected:
            return evening_star_result

        # Try volume climax (strong signal)
        volume_climax_result = self._check_volume_climax(df, vwap, macd)
        if volume_climax_result.detected:
            return volume_climax_result

        # Try shooting star
        shooting_star_result = self._check_shooting_star(df, vwap, macd)
        if shooting_star_result.detected:
            return shooting_star_result

        # Try bearish engulfing
        bearish_engulf_result = self._check_bearish_engulfing(df, vwap, macd)
        if bearish_engulf_result.detected:
            return bearish_engulf_result

        return self.not_detected("No reversal pattern detected")

    def _check_shooting_star(
        self,
        df: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Check for shooting star pattern.

        Shooting star characteristics:
        - Long upper wick (>= 2x body size)
        - Small body in lower 1/3 of candle range
        - Appears after uptrend (3+ green bars prior)
        - Best if at or near HOD
        """
        n = len(df)
        bar = df.iloc[-1]
        prev_bars = df.iloc[-4:-1] if n >= 4 else df.iloc[:-1]

        # Check for prior uptrend (3+ green bars)
        green_count = sum(1 for _, row in prev_bars.iterrows() if row["close"] > row["open"])
        if green_count < 2:
            return self.not_detected("No prior uptrend for shooting star")

        # Calculate candle metrics
        high = bar["high"]
        low = bar["low"]
        open_price = bar["open"]
        close = bar["close"]

        body_top = max(open_price, close)
        body_bottom = min(open_price, close)
        body_size = body_top - body_bottom
        upper_wick = high - body_top
        candle_range = high - low

        if candle_range < 0.01:
            return self.not_detected("No range in candle")

        if body_size < 0.005:
            body_size = 0.005  # Prevent division by zero

        # Check upper wick ratio
        upper_wick_ratio = upper_wick / body_size
        if upper_wick_ratio < self.config["min_upper_wick_ratio"]:
            return self.not_detected(
                f"Upper wick ratio {upper_wick_ratio:.1f}x < {self.config['min_upper_wick_ratio']}x"
            )

        # Check body position (must be in lower third)
        body_position_pct = ((body_bottom - low) / candle_range) * 100
        if body_position_pct > self.config["max_body_position_pct"]:
            return self.not_detected(
                f"Body position {body_position_pct:.0f}% > {self.config['max_body_position_pct']}%"
            )

        # Check if at/near HOD
        hod = df["high"].max()
        distance_from_hod_pct = ((hod - high) / hod) * 100 if hod > 0 else 100
        if distance_from_hod_pct > 3.0:  # Within 3% of HOD
            return self.not_detected(f"Shooting star not at HOD ({distance_from_hod_pct:.1f}% away)")

        # Calculate entry/stop for short
        entry_price = close  # Enter on close of shooting star
        stop_price = self._calculate_stop(df, "above")
        if stop_price <= entry_price:
            return self.not_detected(f"Invalid setup: stop ${stop_price:.2f} <= entry ${entry_price:.2f}")

        stop_distance_cents = (stop_price - entry_price) * 100

        # Build result
        confidence = self._calculate_confidence(
            pattern_weight=self.config["shooting_star_weight"],
            df=df,
            vwap=vwap,
            macd=macd,
        )

        return PatternResult(
            detected=True,
            pattern_name="ShootingStar",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=n - 4,
            pattern_end_idx=n - 1,
            candle_count=4,
            above_vwap=self._check_above_vwap(df, vwap),
            macd_positive=self._check_macd_positive(df, macd),
            reason="Shooting star reversal detected",
            details={
                "upper_wick_ratio": upper_wick_ratio,
                "body_position_pct": body_position_pct,
                "distance_from_hod_pct": distance_from_hod_pct,
                "green_bars_prior": green_count,
                "direction": "short",
            },
        )

    def _check_bearish_engulfing(
        self,
        df: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Check for bearish engulfing pattern.

        Bearish engulfing characteristics:
        - Prior bar is green
        - Current bar is red
        - Current bar's body fully contains prior bar's body
        - Current bar opens above prior close, closes below prior open
        """
        n = len(df)
        if n < 2:
            return self.not_detected("Need at least 2 bars")

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        # Prior bar must be green
        if not self.is_green_candle(prev):
            return self.not_detected("Prior bar is not green")

        # Current bar must be red
        if not self.is_red_candle(curr):
            return self.not_detected("Current bar is not red")

        # Current body must engulf prior body
        curr_body_top = curr["open"]  # Red candle: open > close
        curr_body_bottom = curr["close"]
        prev_body_top = prev["close"]  # Green candle: close > open
        prev_body_bottom = prev["open"]

        # Check engulfing condition
        engulfs_top = curr_body_top >= prev_body_top
        engulfs_bottom = curr_body_bottom <= prev_body_bottom

        if not (engulfs_top and engulfs_bottom):
            return self.not_detected(
                f"Red body doesn't engulf: curr [{curr_body_bottom:.2f}-{curr_body_top:.2f}] "
                f"vs prev [{prev_body_bottom:.2f}-{prev_body_top:.2f}]"
            )

        # Check if at/near HOD
        hod = df["high"].max()
        distance_from_hod_pct = ((hod - prev["high"]) / hod) * 100 if hod > 0 else 100
        if distance_from_hod_pct > 5.0:  # Within 5% of HOD
            return self.not_detected(f"Bearish engulfing not near HOD ({distance_from_hod_pct:.1f}% away)")

        # Calculate entry/stop
        entry_price = curr["close"]
        stop_price = self._calculate_stop(df, "above")
        if stop_price <= entry_price:
            return self.not_detected(f"Invalid setup: stop ${stop_price:.2f} <= entry ${entry_price:.2f}")

        stop_distance_cents = (stop_price - entry_price) * 100

        # Calculate body sizes for engulf ratio
        curr_body_size = abs(curr["open"] - curr["close"])
        prev_body_size = abs(prev["close"] - prev["open"])
        engulf_ratio = curr_body_size / prev_body_size if prev_body_size > 0 else 1.0

        confidence = self._calculate_confidence(
            pattern_weight=self.config["bearish_engulfing_weight"],
            df=df,
            vwap=vwap,
            macd=macd,
        )

        return PatternResult(
            detected=True,
            pattern_name="BearishEngulfing",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=n - 2,
            pattern_end_idx=n - 1,
            candle_count=2,
            above_vwap=self._check_above_vwap(df, vwap),
            macd_positive=self._check_macd_positive(df, macd),
            reason="Bearish engulfing pattern detected",
            details={
                "engulf_ratio": engulf_ratio,
                "distance_from_hod_pct": distance_from_hod_pct,
                "direction": "short",
            },
        )

    def _check_evening_star(
        self,
        df: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Check for evening star pattern.

        Evening star characteristics (3-bar pattern):
        - Bar[-3]: Strong green candle
        - Bar[-2]: Small body (doji-like), gaps up or at same level
        - Bar[-1]: Strong red candle, closes below Bar[-3] midpoint
        """
        n = len(df)
        if n < 3:
            return self.not_detected("Need at least 3 bars")

        bar1 = df.iloc[-3]  # First bar (green)
        bar2 = df.iloc[-2]  # Middle bar (small body)
        bar3 = df.iloc[-1]  # Final bar (red)

        # Bar 1 must be green with decent body
        if not self.is_green_candle(bar1):
            return self.not_detected("Bar[-3] is not green")

        bar1_range = bar1["high"] - bar1["low"]
        bar1_body_pct = self.candle_body_pct(bar1)
        if bar1_body_pct < 50:  # Body should be substantial
            return self.not_detected(f"Bar[-3] body too small: {bar1_body_pct:.0f}%")

        # Bar 2 must have small body (indecision)
        bar2_body_pct = self.candle_body_pct(bar2)
        if bar2_body_pct > self.config["max_middle_body_pct"]:
            return self.not_detected(
                f"Middle bar body too large: {bar2_body_pct:.0f}% > {self.config['max_middle_body_pct']}%"
            )

        # Bar 3 must be red
        if not self.is_red_candle(bar3):
            return self.not_detected("Bar[-1] is not red")

        # Bar 3 must close below Bar 1 midpoint
        if self.config["min_close_below_midpoint"]:
            bar1_midpoint = (bar1["open"] + bar1["close"]) / 2
            if bar3["close"] > bar1_midpoint:
                return self.not_detected(
                    f"Bar[-1] close {bar3['close']:.2f} above Bar[-3] midpoint {bar1_midpoint:.2f}"
                )

        # Check if at/near HOD
        hod = df["high"].max()
        pattern_high = max(bar1["high"], bar2["high"], bar3["high"])
        distance_from_hod_pct = ((hod - pattern_high) / hod) * 100 if hod > 0 else 100
        if distance_from_hod_pct > 3.0:
            return self.not_detected(f"Evening star not near HOD ({distance_from_hod_pct:.1f}% away)")

        # Calculate entry/stop
        entry_price = bar3["close"]
        stop_price = self._calculate_stop(df, "above")
        if stop_price <= entry_price:
            return self.not_detected(f"Invalid setup: stop ${stop_price:.2f} <= entry ${entry_price:.2f}")

        stop_distance_cents = (stop_price - entry_price) * 100

        confidence = self._calculate_confidence(
            pattern_weight=self.config["evening_star_weight"],
            df=df,
            vwap=vwap,
            macd=macd,
        )

        return PatternResult(
            detected=True,
            pattern_name="EveningStar",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=n - 3,
            pattern_end_idx=n - 1,
            candle_count=3,
            above_vwap=self._check_above_vwap(df, vwap),
            macd_positive=self._check_macd_positive(df, macd),
            reason="Evening star reversal detected",
            details={
                "bar1_body_pct": bar1_body_pct,
                "bar2_body_pct": bar2_body_pct,
                "distance_from_hod_pct": distance_from_hod_pct,
                "direction": "short",
            },
        )

    def _check_volume_climax(
        self,
        df: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Check for volume climax pattern.

        Volume climax characteristics:
        - Volume > 3x 20-bar average
        - Occurs at or near HOD
        - Followed by or accompanied by red candle/topping tail

        This pattern indicates exhaustion - all buyers are in.
        """
        n = len(df)
        avg_period = min(self.config["volume_avg_period"], n - 1)

        if avg_period < 5:
            return self.not_detected("Insufficient bars for volume average")

        # Calculate average volume (exclude last few bars and zero-volume halt bars)
        base = df.iloc[:-3] if n > 5 else df
        trading_bars = base[base["volume"] > 0]
        avg_volume = trading_bars["volume"].mean() if len(trading_bars) > 0 else 0

        # Check recent bars for volume climax
        for i in range(-3, 0):  # Check last 3 bars
            bar_idx = n + i
            if bar_idx < 0:
                continue

            bar = df.iloc[i]
            volume = bar["volume"]
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < self.config["volume_climax_multiplier"]:
                continue

            # Volume climax found - check for reversal confirmation
            # Either: bar is red, or has topping tail, or next bar is red

            is_red = self.is_red_candle(bar)
            has_topping_tail = self._has_topping_tail(bar)

            # Check next bar if exists
            next_bar_red = False
            if i < -1:  # Not the last bar
                next_bar = df.iloc[i + 1]
                next_bar_red = self.is_red_candle(next_bar)

            if not (is_red or has_topping_tail or next_bar_red):
                continue  # No reversal confirmation

            # Check if at/near HOD
            hod = df["high"].max()
            distance_from_hod_pct = ((hod - bar["high"]) / hod) * 100 if hod > 0 else 100
            if distance_from_hod_pct > 5.0:
                continue  # Not at HOD

            # Volume climax with reversal found
            entry_price = df.iloc[-1]["close"]  # Enter on current bar's close
            stop_price = self._calculate_stop(df, "above")
            if stop_price <= entry_price:
                return self.not_detected(f"Invalid setup: stop ${stop_price:.2f} <= entry ${entry_price:.2f}")

            stop_distance_cents = (stop_price - entry_price) * 100

            confidence = self._calculate_confidence(
                pattern_weight=self.config["volume_climax_weight"],
                df=df,
                vwap=vwap,
                macd=macd,
                volume_bonus=0.05,  # Extra confidence for volume climax
            )

            return PatternResult(
                detected=True,
                pattern_name="VolumeClimax",
                confidence=confidence,
                entry_price=entry_price,
                stop_price=stop_price,
                stop_distance_cents=stop_distance_cents,
                pattern_start_idx=bar_idx,
                pattern_end_idx=n - 1,
                candle_count=n - bar_idx,
                above_vwap=self._check_above_vwap(df, vwap),
                macd_positive=self._check_macd_positive(df, macd),
                volume_confirmation=True,
                reason="Volume climax reversal detected",
                details={
                    "volume_ratio": volume_ratio,
                    "avg_volume": avg_volume,
                    "climax_volume": volume,
                    "distance_from_hod_pct": distance_from_hod_pct,
                    "climax_bar_red": is_red,
                    "has_topping_tail": has_topping_tail,
                    "direction": "short",
                },
            )

        return self.not_detected(
            f"No volume climax (need >{self.config['volume_climax_multiplier']}x avg volume at HOD)"
        )

    def _has_topping_tail(self, bar: pd.Series) -> bool:
        """Check if bar has a topping tail (long upper wick, body in lower portion)."""
        high = bar["high"]
        low = bar["low"]
        body_top = max(bar["open"], bar["close"])
        body_bottom = min(bar["open"], bar["close"])

        candle_range = high - low
        if candle_range < 0.01:
            return False

        body_size = body_top - body_bottom
        if body_size < 0.005:
            body_size = 0.005

        upper_wick = high - body_top
        upper_wick_ratio = upper_wick / body_size
        body_position = (body_bottom - low) / candle_range

        return upper_wick_ratio >= 1.5 and body_position <= 0.4

    def _calculate_stop(self, df: pd.DataFrame, direction: str) -> float:
        """
        Calculate stop price for short entry.

        Args:
            df: OHLCV DataFrame
            direction: "above" for short entries (stop above HOD)

        Returns:
            Stop price
        """
        # Use recent bars (last 10) for stop placement, not full session HOD.
        # Pattern detection already requires the pattern to be near HOD,
        # so recent bars contain the relevant high.
        recent_bars = df.tail(10) if len(df) > 10 else df
        hod = recent_bars["high"].max()

        if direction == "above":
            # Stop above HOD for shorts
            stop_buffer_pct = self.config.get("stop_buffer_pct", 2.0)
            stop_buffer_min_cents = self.config.get("stop_buffer_min_cents", 5)

            pct_buffer = hod * (stop_buffer_pct / 100)
            min_buffer = stop_buffer_min_cents / 100
            stop_buffer = max(pct_buffer, min_buffer)

            return hod + stop_buffer
        else:
            # For long (not used in this detector)
            lod = df["low"].min()
            return lod * 0.98  # 2% below LOD

    def _calculate_confidence(
        self,
        pattern_weight: float,
        df: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
        volume_bonus: float = 0.0,
    ) -> float:
        """
        Calculate confidence score using standardized system.

        Base starts from pattern weight, then adds confirmations.
        """
        confidence = pattern_weight * 0.65 / 0.85  # Normalize to base ~65%

        # Add confirmations (same as long patterns but inverted meaning)
        # For shorts: ABOVE VWAP is good (more room to fall)
        if self._check_above_vwap(df, vwap):
            confidence += 0.06  # Room to fall to VWAP

        # For shorts: MACD turning negative is good
        if macd is not None:
            macd_data = self.calculate_macd(df["close"]) if macd is None else macd
            if macd_data is not None and len(macd_data) >= 2:
                hist_curr = macd_data.iloc[-1].get("histogram", 0)
                hist_prev = macd_data.iloc[-2].get("histogram", 0)
                if hist_curr < hist_prev:  # MACD weakening
                    confidence += 0.06

        # Volume bonus
        confidence += volume_bonus

        # Cap at 90%
        return min(confidence, 0.90)

    def _check_above_vwap(self, df: pd.DataFrame, vwap: Optional[pd.Series]) -> Optional[bool]:
        """Check if current price is above VWAP."""
        if vwap is None or len(vwap) != len(df):
            return None
        return df.iloc[-1]["close"] > vwap.iloc[-1]

    def _check_macd_positive(self, df: pd.DataFrame, macd: Optional[pd.DataFrame]) -> Optional[bool]:
        """Check if MACD histogram is positive."""
        if macd is None:
            macd = self.calculate_macd(df["close"])
        if macd is None or "histogram" not in macd.columns:
            return None
        return macd.iloc[-1]["histogram"] > 0
