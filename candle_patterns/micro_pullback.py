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

Note: Prior moves >25% are not handled (too extended for micro pullback).

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
from .indicators.atr import get_current_atr


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
            # Prior move requirements (5-25% range)
            "min_prior_move_pct": 5.0,  # Min 5% move before pullback
            "max_prior_move_pct": 25.0,  # Max 25% (shallow pullbacks on big moves)
            "min_green_candles_prior": 2,  # At least 2 candles, >50% green

            # Shallow pullback limits
            "max_pullback_pct": 12.0,  # Max 12% retracement (shallow only)
            "max_pullback_candles": 3,  # Max 3 candles in pullback (micro = tight)

            # Entry trigger - Ross's style (aggressive)
            "entry": "first_green_after_pullback",

            # Hard gates (reject pattern if not met)
            "require_above_vwap": True,   # HARD GATE: Must be above VWAP
            "require_macd_positive": True, # HARD GATE: MACD histogram must be > 0

            # Risk - percent-based stop buffer with ATR floor
            # Stop = pullback_low - max(stop_buffer_pct% of price, stop_buffer_min_cents, ATR * multiplier)
            "stop_buffer_pct": 1.0,  # 1% below pullback low
            "stop_buffer_min_cents": 3,  # Minimum 3 cents buffer
            "stop_buffer_atr_multiplier": 1.5,  # ATR(14) × 1.5 floor (adapts to volatility)
            "stop_buffer_atr_period": 14,  # ATR lookback period

            # Minimum bars needed
            "min_bars_required": 6,

            # Volume profile: pullback avg volume must be lighter than surge avg volume.
            # Heavy pullback volume = distribution, not healthy consolidation.
            # Set to 0 to disable.
            "max_pullback_surge_volume_ratio": 0.75,

            # Volume collapse ratio: peak pullback bar volume / peak surge bar volume.
            # Low VCR = capitulation (healthy). High VCR = distribution (avoid).
            # Set to 0 to disable. Log-only when set to a value > 1.0.
            "max_volume_collapse_ratio": 0.0,  # Disabled by default

            # Quality filter (lowered from 2.0: with 3% stop floor, 5-6% micro-surges
            # on $10+ stocks produce estimated R:R ~1.5. Gate's bracket R:R is the real check.)
            "min_rr_for_setup": 1.2,

        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
        prev_close: Optional[float] = None,
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

        # Reject if any halt bar within pattern range (surge → entry)
        if self._has_halt_bar(df, surge_start_idx, n - 1):
            return self.not_detected("Halt bar within pattern")

        # Step 3: Calculate actual surge metrics (pullback already calculated above)
        surge_window = df.iloc[surge_start_idx:surge_end_idx + 1]
        surge_low = surge_window["low"].min()
        surge_high = surge_window["high"].max()
        prior_move_pct = self.calculate_move_pct(surge_low, surge_high)

        # Check max prior move (too extended for micro pullback)
        max_prior_move = self.config.get("max_prior_move_pct", 15.0)
        if prior_move_pct > max_prior_move:
            return self.not_detected(
                f"Prior move too large: {prior_move_pct:.1f}% > {max_prior_move}%"
            )

        # Get pullback high (pullback_window, pullback_low, pullback_pct already calculated above)
        pullback_high = pullback_window["high"].max()

        # Check max pullback depth
        if pullback_pct > self.config["max_pullback_pct"]:
            return self.not_detected(
                f"Pullback too deep: {pullback_pct:.1f}% > {self.config['max_pullback_pct']}%"
            )

        # Step 4: Calculate stop price first (needed for entry validation)
        # Percent-based stop buffer with ATR floor
        stop_buffer_pct = self.config.get("stop_buffer_pct", 1.0)
        stop_buffer_min_cents = self.config.get("stop_buffer_min_cents", 3)
        atr_multiplier = self.config.get("stop_buffer_atr_multiplier", 1.5)
        atr_period = self.config.get("stop_buffer_atr_period", 14)

        pct_buffer = pullback_low * (stop_buffer_pct / 100)
        min_buffer_cents = stop_buffer_min_cents / 100

        # ATR-based floor: adapts to actual volatility
        atr_value = get_current_atr(df, period=atr_period)
        atr_buffer = (atr_value * atr_multiplier) if atr_value is not None else 0.0

        stop_buffer = max(pct_buffer, min_buffer_cents, atr_buffer)

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
            # Entry at close + 1 cent — reflects realistic fill when signal fires at bar close
            entry_price = entry_candle["close"] + 0.01

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
        min_rr = self.config.get("min_rr_for_setup", 1.2)
        if estimated_rr < min_rr:
            return self.not_detected(
                f"R:R too low: {estimated_rr:.1f} < {min_rr}"
            )

        # Step 8: Volume profile gate — pullback must be lighter than surge
        surge_volume = self._avg_volume(df, surge_start_idx, surge_end_idx)
        pullback_volume = self._avg_volume(df, pullback_start_idx, pullback_end_idx)

        max_vol_ratio = self.config.get("max_pullback_surge_volume_ratio", 0.75)
        if max_vol_ratio > 0 and surge_volume > 0:
            pullback_volume_ratio = pullback_volume / surge_volume
            if pullback_volume_ratio > max_vol_ratio:
                return self.not_detected(
                    f"Pullback volume too heavy: {pullback_volume_ratio:.2f} > {max_vol_ratio} "
                    f"(pullback avg {pullback_volume:,.0f} vs surge avg {surge_volume:,.0f})"
                )

        # Step 8b: Volume collapse ratio (peak-to-peak, not average)
        max_vcr = self.config.get("max_volume_collapse_ratio", 0.0)
        peak_surge_vol = df.iloc[surge_start_idx:surge_end_idx + 1]["volume"].max()
        vcr_end = min(pullback_start_idx + 2, pullback_end_idx + 1)
        peak_pullback_vol = df.iloc[pullback_start_idx:vcr_end]["volume"].max() if vcr_end > pullback_start_idx else 0
        volume_collapse_ratio = (peak_pullback_vol / peak_surge_vol) if peak_surge_vol > 0 else 0.0

        if 0 < max_vcr <= 1.0 and volume_collapse_ratio > max_vcr:
            return self.not_detected(
                f"Volume collapse ratio {volume_collapse_ratio:.2f} > {max_vcr} "
                f"(peak pullback {peak_pullback_vol:,.0f} vs peak surge {peak_surge_vol:,.0f})"
            )

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

        # Step 10b: Minimum histogram strength threshold (price-scaled)
        # 0.1% of entry price so threshold scales with stock price.
        # $1 → 0.001, $10 → 0.01 (same as old default), $50 → 0.05
        min_histogram = max(entry_price * 0.001, 0.001)
        if macd is not None and "histogram" in macd.columns:
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

        # VWAP bounce observability metrics (log-only, for future pattern validation)
        vwap_rising_bars_10 = None
        consolidation_range_pct = None
        consolidation_bar_count = None
        price_vwap_gap_pct = None
        price_vwap_gap_pct_start = None

        if vwap is not None and len(vwap) == n:
            lookback = min(10, n - 1)
            if lookback >= 10:
                # Count strictly rising VWAP bars in last 10 (strict >, flat not counted)
                vwap_tail = vwap.iloc[-(lookback + 1):]
                vwap_rising_bars_10 = sum(
                    1 for i in range(1, len(vwap_tail))
                    if vwap_tail.iloc[i] > vwap_tail.iloc[i - 1]
                )
            # Price-VWAP gap at entry bar (denominator is close)
            entry_vwap = vwap.iloc[-1]
            if entry_vwap > 0:
                price_vwap_gap_pct = round(
                    (entry_candle["close"] - entry_vwap) / entry_candle["close"] * 100, 2
                )
            # Gap at start of VWAP lookback (for gap narrowing analysis)
            start_idx = max(0, n - 1 - lookback)
            start_vwap = vwap.iloc[start_idx]
            start_close = df.iloc[start_idx]["close"]
            if start_vwap > 0 and start_close > 0:
                price_vwap_gap_pct_start = round(
                    (start_close - start_vwap) / start_close * 100, 2
                )

        # Consolidation range over bars before surge start
        min_pre_surge_bars = 3
        if surge_start_idx >= min_pre_surge_bars:
            pre_start = max(0, surge_start_idx - 10)
            pre_surge_bars = df.iloc[pre_start:surge_start_idx]
            consolidation_bar_count = len(pre_surge_bars)
            range_high = pre_surge_bars["high"].max()
            range_low = pre_surge_bars["low"].min()
            avg_price = pre_surge_bars["close"].mean()
            if avg_price > 0:
                consolidation_range_pct = round(
                    (range_high - range_low) / avg_price * 100, 2
                )

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
                "swing_high_time": self._bar_time(df, swing_high_idx_relative),
                "pullback_low": pullback_low,
                "surge_volume_avg": round(surge_volume),
                "pullback_volume_avg": round(pullback_volume),
                "pullback_volume_ratio": round(pullback_volume / surge_volume, 2) if surge_volume > 0 else 0,
                "volume_collapse_ratio": round(volume_collapse_ratio, 2),
                "volume_declining": volume_declining,
                "atr": round(atr_value, 4) if atr_value is not None else None,
                "stop_buffer": round(stop_buffer, 4),
                # VWAP bounce observability (log-only)
                "vwap_rising_bars_10": vwap_rising_bars_10,
                "consolidation_range_pct": consolidation_range_pct,
                "consolidation_bar_count": consolidation_bar_count,
                "price_vwap_gap_pct": price_vwap_gap_pct,
                "price_vwap_gap_pct_start": price_vwap_gap_pct_start,
            },
        )
