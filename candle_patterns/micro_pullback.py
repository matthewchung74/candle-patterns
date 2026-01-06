"""
Micro Pullback Pattern Detector
===============================

Momentum continuation pattern for day trading stocks.
Tuned for Ross Cameron's trading style on volatile small caps.

Pattern Structure:
1. Strong prior surge (2+ candles with >50% green, 5%+ net gain)
2. Pullback/consolidation (configurable depth and length via config)
   - Default: up to max_pullback_candles bars, max_pullback_pct retracement
   - Ross's actual trades show 15-20% pullbacks on volatile names
3. Entry trigger (configurable):
   - "first_green_after_pullback": Enter on first green candle (Ross's style)
   - "first_candle_new_high": Wait for breakout to new high (conservative)

Detection uses flexible range-based logic:
- Finds swing high in lookback window
- Measures pullback depth from swing high (not strict consecutive candles)
- Validates entry_price > stop_price before accepting pattern

Example:
    [GREEN][GREEN][green][RED][red][green][GREEN→ENTRY]
         Prior Surge (5%+)     Consolidation    Bounce
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
        """Default configuration for micro pullback detection.

        Tuned for Ross Cameron's trading style on volatile small caps.
        Validated against labeled trades: SPRC, GORV, SBEV, HTOO.
        """
        return {
            # Prior move requirements
            "min_prior_move_pct": 5.0,  # Min 5% move before pullback
            "min_green_candles_prior": 2,  # At least 2 candles, >50% green

            # Pullback requirements (tuned for Ross's actual trades)
            # HTOO: 16.2% pullback, SBEV: 18.6% pullback
            "max_pullback_candles": 7,  # Allow up to 7 consolidation bars
            "max_pullback_pct": 20.0,  # Allow up to 20% retracement

            # Entry trigger - Ross's style
            # Options: "first_green_after_pullback" (aggressive) or "first_candle_new_high" (conservative)
            "entry": "first_green_after_pullback",

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
        Detect Micro Pullback pattern using flexible range-based logic.

        Instead of requiring strictly consecutive green/red candles,
        this uses net movement over windows to find surge + pullback.

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

        # Need at least 6 bars for pattern
        if n < 6:
            return self.not_detected(f"Insufficient bars: {n}")

        # Mark green/red candles
        df["is_green"] = df["close"] > df["open"]

        # Last bar should be green (potential entry candle)
        if not df.iloc[-1]["is_green"]:
            return self.not_detected("Last candle is red - waiting for green entry candle")

        # === FLEXIBLE APPROACH ===
        # Step 1: Find the pullback zone (recent consolidation/dip before entry)
        # Look for a zone where price pulled back from a recent high

        max_pullback_candles = self.config["max_pullback_candles"]

        # Find the recent swing high (highest high in last 15 candles, excluding last 1)
        lookback = min(15, n - 1)
        recent_bars = df.iloc[-(lookback + 1):-1]  # Exclude entry candle

        swing_high_idx_relative = recent_bars["high"].idxmax()
        swing_high = recent_bars.loc[swing_high_idx_relative, "high"]

        # Pullback zone is between swing high and entry candle
        pullback_start_idx = swing_high_idx_relative + 1
        pullback_end_idx = n - 2  # Just before entry candle

        if pullback_end_idx < pullback_start_idx:
            return self.not_detected("No pullback zone found after swing high")

        pullback_candle_count = pullback_end_idx - pullback_start_idx + 1

        # Check pullback length against config (no hidden buffer - config is source of truth)
        if pullback_candle_count > max_pullback_candles:
            return self.not_detected(
                f"Pullback too long: {pullback_candle_count} candles > {max_pullback_candles}"
            )

        # Step 2: Find the prior surge (the move UP to the swing high)
        # Look for net positive movement before the swing high
        min_green_prior = self.config["min_green_candles_prior"]

        # Search backward from swing high to find where surge started
        surge_end_idx = swing_high_idx_relative
        surge_start_idx = None

        # Look back up to 10 bars for the surge start
        for lookback_len in range(min_green_prior, min(11, surge_end_idx + 1)):
            test_start = surge_end_idx - lookback_len + 1
            if test_start < 0:
                break

            surge_window = df.iloc[test_start:surge_end_idx + 1]
            surge_low = surge_window["low"].min()
            surge_high = surge_window["high"].max()

            # Calculate net move from low to high in window
            net_move_pct = self.calculate_move_pct(surge_low, surge_high)

            # Count mostly-green candles (allow some red)
            green_count = surge_window["is_green"].sum()
            green_ratio = green_count / len(surge_window)

            # Accept if: net move >= min_prior_move AND mostly green (>50%)
            if net_move_pct >= self.config["min_prior_move_pct"] and green_ratio >= 0.5:
                surge_start_idx = test_start
                break

        if surge_start_idx is None:
            return self.not_detected(
                f"No valid surge found (need {self.config['min_prior_move_pct']}%+ move with >50% green candles)"
            )

        # Step 3: Calculate actual surge and pullback metrics
        surge_window = df.iloc[surge_start_idx:surge_end_idx + 1]
        surge_low = surge_window["low"].min()
        surge_high = surge_window["high"].max()
        prior_move_pct = self.calculate_move_pct(surge_low, surge_high)

        pullback_window = df.iloc[pullback_start_idx:pullback_end_idx + 1]
        pullback_low = pullback_window["low"].min()
        pullback_high = pullback_window["high"].max()

        # Pullback percentage from swing high to pullback low
        pullback_pct = self.calculate_move_pct(swing_high, pullback_low)

        if abs(pullback_pct) > self.config["max_pullback_pct"]:
            return self.not_detected(
                f"Pullback too deep: {abs(pullback_pct):.1f}% > {self.config['max_pullback_pct']}%"
            )

        # Step 4: Calculate stop price first (needed for entry validation)
        stop_price = pullback_low - (self.config["stop_loss_cents"] / 100)

        # Step 5: Entry trigger
        entry_candle = df.iloc[-1]
        entry_mode = self.config.get("entry", "first_green_after_pullback")

        if entry_mode == "first_candle_new_high":
            # Conservative: require break of swing high
            if entry_candle["high"] <= swing_high:
                return self.not_detected(
                    f"Entry candle not making new high: {entry_candle['high']:.2f} <= {swing_high:.2f}"
                )
            entry_price = swing_high + 0.01
        else:
            # Ross's style: enter on first green after pullback
            # Entry slightly above open of green candle
            entry_price = entry_candle["open"] + 0.01

        # Step 6: Validate entry > stop (critical safety check)
        # Reject pattern if entry would be at or below stop - invalid long setup
        if entry_price <= stop_price:
            return self.not_detected(
                f"Invalid setup: entry ${entry_price:.2f} <= stop ${stop_price:.2f}"
            )

        stop_distance_cents = (entry_price - stop_price) * 100

        # Step 7: Enforce minimum R:R
        estimated_target = entry_price + (prior_move_pct / 100 * entry_price)
        risk = entry_price - stop_price
        estimated_rr = (estimated_target - entry_price) / risk
        min_rr = self.config.get("min_rr_for_setup", 2.0)
        if estimated_rr < min_rr:
            return self.not_detected(
                f"R:R too low: {estimated_rr:.1f} < {min_rr}"
            )

        # Step 8: Volume confirmation
        surge_volume = surge_window["volume"].mean()
        pullback_volume = pullback_window["volume"].mean() if len(pullback_window) > 0 else 0
        volume_declining = pullback_volume < surge_volume

        # Step 9: Confirmations (advisory)
        above_vwap = None
        if vwap is not None and len(vwap) == n:
            above_vwap = entry_candle["close"] > vwap.iloc[-1]

        # Auto-calculate MACD if not provided
        if macd is None:
            macd = self.calculate_macd(df["close"])

        macd_positive = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0

        # Calculate confidence
        confidence = 0.7  # Base confidence
        if volume_declining:
            confidence += 0.10
        if above_vwap:
            confidence += 0.10
        if macd_positive:
            confidence += 0.10

        # Bonus confidence for tighter pullback
        if abs(pullback_pct) < 5.0:
            confidence += 0.05

        green_count = surge_window["is_green"].sum()

        return PatternResult(
            detected=True,
            pattern_name="MicroPullback",
            confidence=min(confidence, 1.0),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=surge_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - surge_start_idx,
            above_vwap=above_vwap,
            macd_positive=macd_positive,
            volume_confirmation=volume_declining,
            reason="Pattern detected",
            details={
                "prior_move_pct": prior_move_pct,
                "pullback_pct": abs(pullback_pct),
                "green_candles": int(green_count),
                "pullback_candles": pullback_candle_count,
                "swing_high": swing_high,
                "pullback_low": pullback_low,
                "surge_volume_avg": surge_volume,
                "pullback_volume_avg": pullback_volume,
                "volume_declining": volume_declining,
            },
        )
