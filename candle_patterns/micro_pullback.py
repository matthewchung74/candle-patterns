"""
Micro Pullback Pattern Detector
===============================

Momentum continuation pattern for day trading stocks.
Tuned for Ross Cameron's trading style on volatile small caps.

Pattern Structure:
1. Strong prior surge (2+ candles with >50% green, 5-15% net gain)
2. Shallow pullback/consolidation:
   - Max pullback depth: 12% retracement from swing high
   - Max duration: 6 candles
3. Entry trigger: First green candle after pullback (aggressive, Ross's style)

Note: Prior moves >15% are routed to Bull Flag pattern for deeper pullbacks.

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
            # Prior move requirements (5-15% range, >15% routes to Bull Flag)
            "min_prior_move_pct": 5.0,  # Min 5% move before pullback
            "max_prior_move_pct": 15.0,  # Max 15% (larger moves -> Bull Flag)
            "min_green_candles_prior": 2,  # At least 2 candles, >50% green

            # Shallow pullback limits (tighter than Bull Flag)
            "max_pullback_pct": 12.0,  # Max 12% retracement (shallow only)
            "max_pullback_candles": 6,  # Max 6 candles in pullback

            # Entry trigger - Ross's style (aggressive)
            "entry": "first_green_after_pullback",

            # Hard gates (reject pattern if not met)
            "require_above_vwap": True,   # HARD GATE: Must be above VWAP
            "require_macd_positive": True, # HARD GATE: MACD histogram must be > 0

            # Risk - percent-based stop buffer with minimum floor
            # Stop = pullback_low - max(stop_buffer_pct% of price, stop_buffer_min_cents)
            "stop_buffer_pct": 1.0,  # 1% below pullback low
            "stop_buffer_min_cents": 3,  # Minimum 3 cents buffer

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

        # Calculate pullback depth first (needed for two-tier candle limit)
        pullback_window = df.iloc[pullback_start_idx:pullback_end_idx + 1]
        pullback_low = pullback_window["low"].min()
        pullback_pct = abs(self.calculate_move_pct(swing_high, pullback_low))

        # Check pullback duration (simplified single limit)
        max_pullback_candles = self.config.get("max_pullback_candles", 6)
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

        # Step 3: Calculate actual surge metrics (pullback already calculated above)
        surge_window = df.iloc[surge_start_idx:surge_end_idx + 1]
        surge_low = surge_window["low"].min()
        surge_high = surge_window["high"].max()
        prior_move_pct = self.calculate_move_pct(surge_low, surge_high)

        # Check max prior move (larger moves should use Bull Flag)
        max_prior_move = self.config.get("max_prior_move_pct", 15.0)
        if prior_move_pct > max_prior_move:
            return self.not_detected(
                f"Prior move too large: {prior_move_pct:.1f}% > {max_prior_move}% (use Bull Flag)"
            )

        # Get pullback high (pullback_window, pullback_low, pullback_pct already calculated above)
        pullback_high = pullback_window["high"].max()

        # Check max pullback depth
        if pullback_pct > self.config["max_pullback_pct"]:
            return self.not_detected(
                f"Pullback too deep: {pullback_pct:.1f}% > {self.config['max_pullback_pct']}%"
            )

        # Step 4: Calculate stop price first (needed for entry validation)
        # Percent-based stop buffer with minimum floor
        stop_buffer_pct = self.config.get("stop_buffer_pct", 1.0)
        stop_buffer_min_cents = self.config.get("stop_buffer_min_cents", 3)

        pct_buffer = pullback_low * (stop_buffer_pct / 100)
        min_buffer = stop_buffer_min_cents / 100
        stop_buffer = max(pct_buffer, min_buffer)

        stop_price = pullback_low - stop_buffer

        # Step 5: Entry trigger (no lookahead bias)
        prev_bar = df.iloc[-2]  # Previous bar (complete)
        entry_candle = df.iloc[-1]  # Current bar
        entry_mode = self.config.get("entry", "first_green_after_pullback")

        if entry_mode == "first_candle_new_high":
            # Conservative: require CONFIRMED break of swing high
            # Use previous bar's close or current bar's open (not current bar's high)
            breakout_confirmed = (prev_bar["close"] > swing_high) or (entry_candle["open"] > swing_high)
            if not breakout_confirmed:
                return self.not_detected(
                    f"No confirmed new high: prev close {prev_bar['close']:.2f}, "
                    f"curr open {entry_candle['open']:.2f} <= swing high {swing_high:.2f}"
                )
            entry_price = swing_high + 0.01
        else:
            # Ross's style: enter on first green after pullback
            # Entry slightly above open of green candle (no lookahead - open is known)
            entry_price = entry_candle["open"] + 0.01

        # Step 6: Validate entry price is reasonable
        # Check entry > stop (critical safety check)
        if entry_price <= stop_price:
            return self.not_detected(
                f"Invalid setup: entry ${entry_price:.2f} <= stop ${stop_price:.2f}"
            )

        # Validate entry_price is within reasonable range of current price
        # This prevents stale bar data from causing invalid signals
        current_price = entry_candle["close"]
        max_entry_deviation_pct = self.config.get("max_entry_deviation_pct", 5.0)
        if entry_price > current_price * (1 + max_entry_deviation_pct / 100):
            return self.not_detected(
                f"Entry price {entry_price:.2f} too far from current {current_price:.2f} "
                f"(>{max_entry_deviation_pct}% deviation - possible stale data)"
            )

        stop_distance_cents = (entry_price - stop_price) * 100

        # Step 6b: Enforce minimum stop distance (prevents gap-down setups)
        min_stop_distance_cents = self.config.get("min_stop_distance_cents", 3)
        if stop_distance_cents < min_stop_distance_cents:
            return self.not_detected(
                f"Stop too tight: {stop_distance_cents:.1f}¢ < {min_stop_distance_cents}¢ min"
            )

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
        macd_slope_up = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0
            # 3-bar MACD slope: compare current MACD line to 3 bars ago
            if "macd" in macd.columns and len(macd) >= 4:
                macd_slope_up = macd.iloc[-1]["macd"] > macd.iloc[-4]["macd"]

        # Step 10: Hard gates (reject pattern if not met)
        # Note: Use == False (not 'is False') because numpy.bool != Python bool
        if self.config.get("require_above_vwap", True) and above_vwap == False:
            return self.not_detected("HARD GATE: Price below VWAP")

        if self.config.get("require_macd_positive", True) and macd_positive == False:
            return self.not_detected("HARD GATE: MACD histogram negative")

        # Step 10b: Minimum histogram strength threshold
        # Data shows: histogram < 0.01 = 38% WR, histogram > 0.1 = 80% WR
        min_histogram = self.config.get("min_histogram_threshold", 0.03)
        if min_histogram > 0 and macd is not None and "histogram" in macd.columns:
            current_histogram = macd.iloc[-1]["histogram"]
            if current_histogram < min_histogram:
                return self.not_detected(
                    f"HARD GATE: MACD histogram {current_histogram:.4f} below threshold {min_histogram}"
                )

        # Step 10c: Minimum price threshold
        # Data shows: $0-3 = 21% WR, $10+ = 71% WR
        min_price = self.config.get("min_price_threshold", 0.0)
        if min_price > 0 and entry_price < min_price:
            return self.not_detected(
                f"HARD GATE: Price ${entry_price:.2f} below threshold ${min_price:.2f}"
            )

        # Calculate confidence (standardized system)
        # Base: 65%, Cap: 90%, Gate: 80% (enforced in trade_engine)
        confidence = 0.65  # Base confidence
        if volume_declining:
            confidence += 0.10
        if above_vwap:
            confidence += 0.08
        if macd_positive:
            confidence += 0.08
        if macd_slope_up:
            confidence += 0.04
        # Bonus for tighter pullback
        if pullback_pct < 5.0:
            confidence += 0.06

        green_count = surge_window["is_green"].sum()

        return PatternResult(
            detected=True,
            pattern_name="MicroPullback",
            confidence=min(confidence, 0.90),  # Cap at 90%
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=surge_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - surge_start_idx,
            above_vwap=above_vwap,
            macd_positive=macd_positive,
            macd_slope_up=macd_slope_up,
            volume_confirmation=volume_declining,
            reason="Pattern detected",
            details={
                "prior_move_pct": prior_move_pct,
                "pullback_pct": pullback_pct,
                "green_candles": int(green_count),
                "pullback_candles": pullback_candle_count,
                "swing_high": swing_high,
                "pullback_low": pullback_low,
                "surge_volume_avg": surge_volume,
                "pullback_volume_avg": pullback_volume,
                "volume_declining": volume_declining,
            },
        )
