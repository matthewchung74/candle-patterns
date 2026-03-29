"""
Parabolic Exhaustion Pattern Detector
======================================

Short-side pattern for small-cap momentum stocks. Detects blow-off tops
where a parabolic run exhausts buying pressure, signaled by a volume
climax bar and price rejection.

Pattern Structure:
1. Extension: stock 25%+ from open (or prev_close)
2. Parabolic surge: 3+ strong green bars (each gaining 1.5%+) running into HOD
3. Volume climax: a bar near HOD with 3x+ session average volume
4. Price rejection: red close, topping tail, or lower high within 3 bars of climax
5. Entry: first red bar after climax (must be current bar, no lookahead)
6. Stop: above HOD + ATR buffer

Why it works on small caps:
Low float means available shares get exhausted. When volume spikes at the top,
there's nobody left to buy. The transition from "everyone wants in" to "who's
left to buy?" happens in one bar.

Example:
    [+3%][+4%][+5%][CLIMAX 5x vol][RED→ENTRY]
         Parabolic run              Exhaustion
"""

from typing import Optional, Dict, Any
import pandas as pd
from .base import PatternDetector, PatternResult
from .indicators.atr import get_current_atr


class ParabolicExhaustion(PatternDetector):
    """
    Detect parabolic exhaustion (blow-off top) for short entry.

    Composite pattern: extension + surge momentum + volume climax + price rejection.
    """

    def default_config(self) -> Dict[str, Any]:
        return {
            # Extension gate (hard)
            "min_extension_from_open_pct": 25.0,

            # Surge requirements (hard)
            "min_strong_green_bars": 3,
            "strong_green_threshold_pct": 1.5,
            "max_surge_lookback": 15,

            # Volume climax (hard)
            "volume_climax_multiplier": 3.0,
            "climax_lookback_bars": 5,
            "hod_proximity_pct": 2.0,

            # HOD recency (hard)
            "max_hod_age_bars": 8,

            # Rejection (hard)
            "rejection_window": 3,
            "min_topping_tail_wick_ratio": 1.5,

            # Entry
            "max_entry_delay_bars": 3,

            # Stop placement
            "stop_buffer_pct": 0.5,
            "stop_buffer_min_cents": 5,
            "stop_buffer_atr_multiplier": 1.0,
            "stop_buffer_atr_period": 14,

            # Safety
            "min_stop_distance_cents": 5,
            "max_stop_distance_pct": 15.0,
            "min_bars_required": 15,

            # Exit config
            "macd_exit_confirmation_bars": 1,
            "vwap_exit_confirmation_bars": 1,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
        prev_close: Optional[float] = None,
    ) -> PatternResult:
        try:
            self.validate_bars(bars)
        except ValueError as e:
            return self.not_detected(str(e))

        df = bars.copy().reset_index(drop=True)
        n = len(df)

        # --- Phase 1: Reference price ---
        reference_price = prev_close if prev_close and prev_close > 0 else df.iloc[0]["open"]
        if reference_price <= 0:
            return self.not_detected("Invalid reference price")

        # Reject if any halt bar in recent bars
        lookback_start = max(0, n - self.config["max_surge_lookback"] - 5)
        if self._has_halt_bar(df, lookback_start, n - 1):
            return self.not_detected("Halt bar in recent bars")

        # --- Phase 2: Extension gate ---
        hod = df["high"].max()
        hod_idx = df["high"].idxmax()
        extension_pct = self.calculate_move_pct(reference_price, hod)

        min_ext = self.config["min_extension_from_open_pct"]
        if extension_pct < min_ext:
            return self.not_detected(
                f"Not extended: {extension_pct:.1f}% < {min_ext}%"
            )

        # --- Phase 3: HOD recency ---
        max_age = self.config["max_hod_age_bars"]
        hod_age = n - 1 - hod_idx
        if hod_age > max_age:
            return self.not_detected(
                f"HOD too stale: {hod_age} bars ago > {max_age} max"
            )

        # --- Phase 4: Parabolic surge detection ---
        surge_result = self._find_surge(df, hod_idx)
        if surge_result is None:
            return self.not_detected("No parabolic surge found")

        surge_start_idx, strong_green_count, surge_green_count = surge_result

        # --- Phase 5: Volume climax detection ---
        climax_result = self._find_volume_climax(df, hod_idx)
        if climax_result is None:
            return self.not_detected("No volume climax near HOD")

        climax_idx, climax_volume_ratio = climax_result

        # --- Phase 6: Price rejection ---
        rejection_result = self._check_rejection_after_climax(df, climax_idx)
        if rejection_result is None:
            return self.not_detected("No price rejection after volume climax")

        rejection_type = rejection_result

        # --- Phase 7: Entry trigger ---
        entry_bar = df.iloc[-1]
        if entry_bar["close"] >= entry_bar["open"]:
            return self.not_detected("Last bar is green — waiting for red entry bar")

        # Entry bar must be within max_entry_delay_bars of climax
        entry_delay = n - 1 - climax_idx
        max_delay = self.config["max_entry_delay_bars"]
        if entry_delay > max_delay:
            return self.not_detected(
                f"Entry too late: {entry_delay} bars after climax > {max_delay} max"
            )

        entry_price = entry_bar["close"]

        # --- Phase 8: Stop placement ---
        pattern_high = df.iloc[climax_idx:]["high"].max()
        atr_value = get_current_atr(df, period=self.config["stop_buffer_atr_period"])

        pct_buffer = pattern_high * (self.config["stop_buffer_pct"] / 100)
        min_buffer_cents = self.config["stop_buffer_min_cents"] / 100
        atr_buffer = (atr_value * self.config["stop_buffer_atr_multiplier"]) if atr_value else 0.0

        stop_buffer = max(pct_buffer, min_buffer_cents, atr_buffer)
        stop_price = pattern_high + stop_buffer

        # --- Phase 9: Safety checks ---
        if stop_price <= entry_price:
            return self.not_detected(
                f"Invalid: stop ${stop_price:.2f} <= entry ${entry_price:.2f}"
            )

        stop_distance_cents = (stop_price - entry_price) * 100

        min_stop_cents = self.config["min_stop_distance_cents"]
        if stop_distance_cents < min_stop_cents:
            return self.not_detected(
                f"Stop too tight: {stop_distance_cents:.1f}c < {min_stop_cents}c"
            )

        max_stop_pct = self.config["max_stop_distance_pct"]
        stop_pct = (stop_price - entry_price) / entry_price * 100
        if stop_pct > max_stop_pct:
            return self.not_detected(
                f"Stop too wide: {stop_pct:.1f}% > {max_stop_pct}%"
            )

        # --- Phase 10: Confidence scoring ---
        confidence = 0.65

        # Volume climax strength
        if climax_volume_ratio >= 5.0:
            confidence += 0.06

        # Red climax bar (strongest rejection)
        climax_bar = df.iloc[climax_idx]
        if climax_bar["close"] < climax_bar["open"]:
            confidence += 0.05

        # Topping tail on climax bar
        body_top = max(climax_bar["open"], climax_bar["close"])
        body_bottom = min(climax_bar["open"], climax_bar["close"])
        body_size = max(body_top - body_bottom, 0.005)
        upper_wick = climax_bar["high"] - body_top
        if upper_wick / body_size >= self.config["min_topping_tail_wick_ratio"]:
            confidence += 0.04

        # Below VWAP at entry
        below_vwap = None
        if vwap is not None and len(vwap) == n:
            vwap_reset = vwap.reset_index(drop=True)
            vwap_at_entry = vwap_reset.iloc[-1]
            if pd.notna(vwap_at_entry) and vwap_at_entry > 0:
                below_vwap = entry_price < vwap_at_entry
                if below_vwap:
                    confidence += 0.04

        # MACD declining
        macd_declining = None
        if macd is None:
            macd = self.calculate_macd(df["close"])
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            if n >= 2:
                curr_hist = macd.iloc[-1]["histogram"]
                prev_hist = macd.iloc[-2]["histogram"]
                macd_declining = curr_hist < prev_hist
                if macd_declining:
                    confidence += 0.04

        # Strong surge (4+ strong green bars)
        if strong_green_count >= 4:
            confidence += 0.03

        # Escalating volume in surge
        escalating_volume = self._check_escalating_volume(df, surge_start_idx, hod_idx)
        if escalating_volume:
            confidence += 0.03

        confidence = min(confidence, 0.90)

        # --- Phase 11: VWAP gap for details ---
        entry_vwap_gap_pct = None
        if vwap is not None and len(vwap) == n:
            vwap_reset = vwap.reset_index(drop=True)
            vwap_val = vwap_reset.iloc[-1]
            if pd.notna(vwap_val) and entry_price > 0:
                entry_vwap_gap_pct = round(
                    (entry_price - vwap_val) / entry_price * 100, 2
                )

        return PatternResult(
            detected=True,
            pattern_name="ParabolicExhaustion",
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=surge_start_idx,
            pattern_end_idx=n - 1,
            candle_count=n - 1 - surge_start_idx + 1,
            above_vwap=not below_vwap if below_vwap is not None else None,
            macd_positive=None,
            macd_slope_up=None if macd_declining is None else not macd_declining,
            volume_confirmation=escalating_volume,
            reason="Pattern detected",
            details={
                "direction": "short",
                "extension_from_open_pct": round(extension_pct, 2),
                "hod": round(hod, 4),
                "hod_idx": int(hod_idx),
                "hod_time": self._bar_time(df, hod_idx),
                "surge_start_idx": int(surge_start_idx),
                "surge_green_count": surge_green_count,
                "strong_green_count": strong_green_count,
                "climax_idx": int(climax_idx),
                "climax_volume_ratio": round(climax_volume_ratio, 2),
                "rejection_type": rejection_type,
                "pattern_high": round(pattern_high, 4),
                "below_vwap": below_vwap,
                "macd_declining": macd_declining,
                "escalating_volume": escalating_volume,
                "entry_vwap_gap_pct": entry_vwap_gap_pct,
                "atr": round(atr_value, 4) if atr_value else None,
                "stop_buffer": round(stop_buffer, 4),
            },
        )

    def _find_surge(self, df: pd.DataFrame, hod_idx: int):
        """Find the parabolic surge leading into HOD.

        Returns (surge_start_idx, strong_green_count, total_green_count) or None.
        """
        max_lookback = self.config["max_surge_lookback"]
        min_strong = self.config["min_strong_green_bars"]
        strong_threshold = self.config["strong_green_threshold_pct"]

        # Scan backward from HOD to find surge start
        earliest = max(0, hod_idx - max_lookback)

        best_start = None
        best_strong = 0
        best_green = 0

        for start in range(earliest, hod_idx):
            window = df.iloc[start:hod_idx + 1]
            strong_count = 0
            green_count = 0

            for i in range(len(window)):
                bar = window.iloc[i]
                if bar["close"] > bar["open"]:
                    green_count += 1
                    gain_pct = (bar["close"] - bar["open"]) / bar["open"] * 100
                    if gain_pct >= strong_threshold:
                        strong_count += 1

            if strong_count >= min_strong and (best_start is None or strong_count > best_strong):
                best_start = start
                best_strong = strong_count
                best_green = green_count

        if best_start is None:
            return None

        return best_start, best_strong, best_green

    def _find_volume_climax(self, df: pd.DataFrame, hod_idx: int):
        """Find volume climax bar near HOD.

        Returns (climax_idx, volume_ratio) or None.
        """
        n = len(df)
        lookback = self.config["climax_lookback_bars"]
        multiplier = self.config["volume_climax_multiplier"]
        hod_prox = self.config["hod_proximity_pct"]

        hod = df["high"].max()

        # Search window: bars around HOD
        search_start = max(0, hod_idx - lookback)
        search_end = min(n - 1, hod_idx + lookback)

        # Average volume excluding the search window
        if search_start > 0:
            avg_vol = self._avg_volume(df, 0, search_start - 1)
        else:
            avg_vol = self._avg_volume(df, 0, n - 1)

        if avg_vol <= 0:
            return None

        # Find highest volume bar in the search window
        best_idx = None
        best_ratio = 0.0

        for i in range(search_start, search_end + 1):
            bar = df.iloc[i]
            if bar["volume"] <= 0:
                continue

            ratio = bar["volume"] / avg_vol

            # Must be near HOD
            if bar["high"] < hod * (1 - hod_prox / 100):
                continue

            if ratio > best_ratio:
                best_idx = i
                best_ratio = ratio

        if best_idx is None or best_ratio < multiplier:
            return None

        return best_idx, best_ratio

    def _check_rejection_after_climax(self, df: pd.DataFrame, climax_idx: int):
        """Check for price rejection after volume climax.

        Returns rejection_type string or None.
        """
        n = len(df)
        window = self.config["rejection_window"]
        wick_ratio = self.config["min_topping_tail_wick_ratio"]

        climax_bar = df.iloc[climax_idx]

        # Check 1: Red close at climax
        if climax_bar["close"] < climax_bar["open"]:
            return "red_climax"

        # Check 2: Topping tail at climax
        body_top = max(climax_bar["open"], climax_bar["close"])
        body_bottom = min(climax_bar["open"], climax_bar["close"])
        body_size = max(body_top - body_bottom, 0.005)
        upper_wick = climax_bar["high"] - body_top

        if upper_wick / body_size >= wick_ratio:
            return "topping_tail"

        # Check bars after climax
        end = min(n, climax_idx + window + 1)
        for i in range(climax_idx + 1, end):
            bar = df.iloc[i]

            # Check 3: Lower high
            if bar["high"] < climax_bar["high"]:
                return "lower_high"

            # Check 4: Red bar after climax
            if bar["close"] < bar["open"]:
                return "red_after_climax"

        return None

    def _check_escalating_volume(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> bool:
        """Check if volume escalated during the surge (second half > first half)."""
        if end_idx - start_idx < 3:
            return False

        mid = start_idx + (end_idx - start_idx) // 2
        first_half_vol = self._avg_volume(df, start_idx, mid)
        second_half_vol = self._avg_volume(df, mid + 1, end_idx)

        return second_half_vol > first_half_vol if first_half_vol > 0 else False
