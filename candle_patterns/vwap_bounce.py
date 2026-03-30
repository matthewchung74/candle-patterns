"""
VWAP Bounce Pattern Detector
=============================

Consolidation entry pattern for day trading momentum stocks.
Detects institutional accumulation where VWAP is rising into
a tight consolidation range, and enters near the bottom of
the range before the breakout.

Pattern Structure:
1. Tight consolidation (5-15 bars, range < 2% of price)
2. VWAP strictly rising through consolidation (6+ of 10 bar-pairs)
3. VWAP-to-price gap narrowing (VWAP catching up)
4. Entry: green bar near bottom 35% of consolidation range
5. Gate: price has NOT broken out yet (close < consolidation high)
6. Stop: below VWAP at entry time (with ATR buffer)

Key difference from MicroPullback:
- No surge required (consolidation IS the setup)
- Enters BEFORE breakout (MicroPullback enters after)
- VWAP is the signal (slope + proximity), not just a gate
- Stop below VWAP, not below pullback low
- Zero overlap: no surge = MicroPullback can't fire

Example:
    [flat][flat][flat][flat][flat][GREEN→ENTRY]
         Consolidation (VWAP rising)    Entry near low
"""

from typing import Optional, Dict, Any
import pandas as pd
from .base import PatternDetector, PatternResult
from .indicators.atr import get_current_atr


class VwapBounce(PatternDetector):
    """
    Detect VWAP Bounce pattern.

    Rising VWAP into tight consolidation = sustained buying pressure
    absorbing supply. Enter at bottom of range before breakout.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration for VWAP bounce detection."""
        return {
            # Consolidation requirements
            "min_consolidation_bars": 5,
            "max_consolidation_bars": 15,
            "max_consolidation_range_pct": 2.0,  # (max_high - min_low) / avg_price
            "min_consolidation_range_cents": 5,   # Floor for cheap stocks (5 cents)

            # VWAP slope: strict > (flat does NOT count as rising)
            "vwap_slope_lookback": 10,
            "min_vwap_rising_bars": 6,  # 6 of 10 strictly rising

            # Proximity / gap narrowing
            "max_price_vwap_gap_pct": 3.0,  # Max avg distance during consolidation
            "require_gap_narrowing": True,   # First-half gap > second-half gap

            # Entry zone
            "entry_zone_pct": 35,  # Entry bar low must be in bottom 35% of range

            # Volume during consolidation
            "require_volume_declining": False,  # Second-half vol < first-half (log initially)

            # Stop placement (below VWAP)
            "stop_buffer_pct": 0.5,
            "stop_buffer_min_cents": 3,
            "stop_buffer_atr_multiplier": 1.5,
            "stop_buffer_atr_period": 14,

            # Safety
            "min_stop_distance_cents": 3,
            "max_stop_distance_pct": 4.0,
            "min_bars_required": 15,

            # MACD: boost only, not hard gate (VWAP slope is the momentum signal)
            "require_macd_positive": False,

            # Exit configuration (higher VWAP confirmation for near-VWAP entries)
            "macd_exit_confirmation_bars": 1,
            "vwap_exit_confirmation_bars": 3,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
        prev_close: Optional[float] = None,
    ) -> PatternResult:
        """
        Detect VWAP Bounce pattern.

        Scans backward from last bar to find consolidation with
        rising VWAP, then checks if entry bar is near consolidation low.

        Args:
            bars: OHLCV DataFrame (newest bar last)
            vwap: VWAP series (required, same length as bars)
            macd: Optional MACD DataFrame for confidence boost
            prev_close: Optional previous day's close

        Returns:
            PatternResult with detection details
        """
        try:
            self.validate_bars(bars)
        except ValueError as e:
            return self.not_detected(str(e))

        df = bars.copy().reset_index(drop=True)
        n = len(df)

        # Step 1: Require VWAP data (core signal, not optional)
        if vwap is None or len(vwap) != n:
            return self.not_detected("No VWAP data (required for VwapBounce)")

        vwap = vwap.reset_index(drop=True)
        valid_vwap_count = vwap.notna().sum()
        if valid_vwap_count < self.config["vwap_slope_lookback"] + self.config["min_consolidation_bars"]:
            return self.not_detected(
                f"Insufficient VWAP data: {valid_vwap_count} valid bars"
            )

        # Step 2: Last bar must be green (entry trigger)
        entry_candle = df.iloc[-1]
        if entry_candle["close"] <= entry_candle["open"]:
            return self.not_detected("Last candle is red — waiting for green entry candle")

        # Step 3: Check VWAP slope (strictly rising)
        lookback = self.config["vwap_slope_lookback"]
        min_rising = self.config["min_vwap_rising_bars"]

        if n < lookback + 1:
            return self.not_detected(f"Need {lookback + 1} bars for VWAP slope check")

        vwap_tail = vwap.iloc[-(lookback + 1):]
        rising_count = 0
        for i in range(1, len(vwap_tail)):
            v_curr = vwap_tail.iloc[i]
            v_prev = vwap_tail.iloc[i - 1]
            if pd.notna(v_curr) and pd.notna(v_prev) and v_curr > v_prev:
                rising_count += 1

        if rising_count < min_rising:
            return self.not_detected(
                f"VWAP not rising: {rising_count}/{lookback} bars strictly increasing "
                f"(need {min_rising})"
            )

        # Step 4: Find consolidation zone (scan backward from entry bar)
        min_consol = self.config["min_consolidation_bars"]
        max_consol = self.config["max_consolidation_bars"]
        max_range_pct = self.config["max_consolidation_range_pct"]
        min_range_cents = self.config["min_consolidation_range_cents"]

        consol_end_idx = n - 2  # Bar before entry
        consol_start_idx = None
        consol_high = None
        consol_low = None
        consol_range_pct = None

        # Try longest window first, shrink until range fits
        for length in range(min(max_consol, consol_end_idx + 1), min_consol - 1, -1):
            start = consol_end_idx - length + 1
            if start < 0:
                continue
            window = df.iloc[start:consol_end_idx + 1]
            w_high = window["high"].max()
            w_low = window["low"].min()
            avg_price = window["close"].mean()
            if avg_price <= 0:
                continue
            range_pct = (w_high - w_low) / avg_price * 100
            range_cents = (w_high - w_low) * 100

            if range_pct <= max_range_pct and range_cents >= min_range_cents:
                consol_start_idx = start
                consol_high = w_high
                consol_low = w_low
                consol_range_pct = round(range_pct, 2)
                break

        if consol_start_idx is None:
            return self.not_detected(
                f"No consolidation found ({min_consol}-{max_consol} bars, "
                f"range < {max_range_pct}%, min {min_range_cents}c)"
            )

        consol_bars = consol_end_idx - consol_start_idx + 1

        # Step 5: Check price-VWAP proximity during consolidation
        max_gap_pct = self.config["max_price_vwap_gap_pct"]
        consol_closes = df.iloc[consol_start_idx:consol_end_idx + 1]["close"]
        consol_vwaps = vwap.iloc[consol_start_idx:consol_end_idx + 1]

        # Filter out NaN VWAP bars
        valid_mask = consol_vwaps.notna()
        if valid_mask.sum() < min_consol:
            return self.not_detected("Insufficient valid VWAP during consolidation")

        avg_close = consol_closes[valid_mask].mean()
        avg_vwap = consol_vwaps[valid_mask].mean()
        if avg_close <= 0:
            return self.not_detected("Invalid price data")

        avg_gap_pct = abs(avg_close - avg_vwap) / avg_close * 100
        if avg_gap_pct > max_gap_pct:
            return self.not_detected(
                f"Price-VWAP gap too wide: {avg_gap_pct:.1f}% > {max_gap_pct}%"
            )

        # Step 6: Check gap narrowing (first-half avg gap > second-half avg gap)
        gap_narrowing = None
        gap_start_pct = None
        gap_end_pct = None
        if self.config["require_gap_narrowing"] and consol_bars >= 4:
            mid = consol_start_idx + consol_bars // 2
            first_half_closes = df.iloc[consol_start_idx:mid]["close"]
            first_half_vwaps = vwap.iloc[consol_start_idx:mid]
            second_half_closes = df.iloc[mid:consol_end_idx + 1]["close"]
            second_half_vwaps = vwap.iloc[mid:consol_end_idx + 1]

            fh_valid = first_half_vwaps.notna()
            sh_valid = second_half_vwaps.notna()

            if fh_valid.sum() > 0 and sh_valid.sum() > 0:
                fh_avg_close = first_half_closes[fh_valid].mean()
                fh_avg_vwap = first_half_vwaps[fh_valid].mean()
                sh_avg_close = second_half_closes[sh_valid].mean()
                sh_avg_vwap = second_half_vwaps[sh_valid].mean()

                gap_start_pct = round(
                    abs(fh_avg_close - fh_avg_vwap) / fh_avg_close * 100, 2
                ) if fh_avg_close > 0 else None
                gap_end_pct = round(
                    abs(sh_avg_close - sh_avg_vwap) / sh_avg_close * 100, 2
                ) if sh_avg_close > 0 else None

                if gap_start_pct is not None and gap_end_pct is not None:
                    gap_narrowing = gap_start_pct > gap_end_pct
                    if not gap_narrowing:
                        return self.not_detected(
                            f"Gap not narrowing: {gap_start_pct:.1f}% -> {gap_end_pct:.1f}%"
                        )

        # Step 7: Entry near consolidation low (bottom entry_zone_pct% of range)
        entry_zone_pct = self.config["entry_zone_pct"]
        consol_range = consol_high - consol_low
        if consol_range <= 0:
            return self.not_detected("Zero consolidation range")

        entry_zone_ceiling = consol_low + consol_range * (entry_zone_pct / 100)
        if entry_candle["low"] > entry_zone_ceiling:
            return self.not_detected(
                f"Entry bar not near consolidation low: low ${entry_candle['low']:.2f} "
                f"> zone ceiling ${entry_zone_ceiling:.2f} "
                f"(bottom {entry_zone_pct}% of ${consol_low:.2f}-${consol_high:.2f})"
            )

        # Step 8: No breakout yet (close must be below consolidation high)
        if entry_candle["close"] >= consol_high:
            return self.not_detected(
                f"Already broke out: close ${entry_candle['close']:.2f} "
                f">= consolidation high ${consol_high:.2f}"
            )

        # Step 9: Consolidation volume character
        volume_declining = None
        if consol_bars >= 4:
            mid = consol_start_idx + consol_bars // 2
            first_half_vol = self._avg_volume(df, consol_start_idx, mid - 1)
            second_half_vol = self._avg_volume(df, mid, consol_end_idx)
            if first_half_vol > 0:
                volume_declining = second_half_vol < first_half_vol

            if self.config.get("require_volume_declining") and volume_declining == False:
                return self.not_detected(
                    f"Volume rising during consolidation: "
                    f"first half avg {first_half_vol:,.0f} -> second half avg {second_half_vol:,.0f}"
                )

        # Step 10: Calculate stop price (below VWAP)
        vwap_at_entry = vwap.iloc[-1]
        if pd.isna(vwap_at_entry) or vwap_at_entry <= 0:
            return self.not_detected("No valid VWAP at entry bar")

        stop_buffer_pct = self.config["stop_buffer_pct"]
        stop_buffer_min_cents = self.config["stop_buffer_min_cents"]
        atr_multiplier = self.config["stop_buffer_atr_multiplier"]
        atr_period = self.config["stop_buffer_atr_period"]

        pct_buffer = vwap_at_entry * (stop_buffer_pct / 100)
        min_buffer_cents = stop_buffer_min_cents / 100
        atr_value = get_current_atr(df, period=atr_period)
        atr_buffer = (atr_value * atr_multiplier) if atr_value is not None else 0.0

        stop_buffer = max(pct_buffer, min_buffer_cents, atr_buffer)
        stop_price = vwap_at_entry - stop_buffer

        # Entry at close + 1 cent — reflects realistic fill when signal fires at bar close
        entry_price = entry_candle["close"] + 0.01

        # Step 11: Safety checks
        # Reject if any halt bar in pattern
        if self._has_halt_bar(df, consol_start_idx, n - 1):
            return self.not_detected("Halt bar within pattern")

        # Entry must be above stop
        if entry_price <= stop_price:
            return self.not_detected(
                f"Invalid setup: entry ${entry_price:.2f} <= stop ${stop_price:.2f}"
            )

        stop_distance_cents = (entry_price - stop_price) * 100

        # Minimum stop distance
        min_stop_cents = self.config["min_stop_distance_cents"]
        if stop_distance_cents < min_stop_cents:
            return self.not_detected(
                f"Stop too tight: {stop_distance_cents:.1f}c < {min_stop_cents}c min"
            )

        # Maximum stop distance
        max_stop_pct = self.config["max_stop_distance_pct"]
        stop_pct = (entry_price - stop_price) / entry_price * 100
        if stop_pct > max_stop_pct:
            return self.not_detected(
                f"Stop too wide: {stop_pct:.1f}% > {max_stop_pct}% max"
            )

        # Reject if consolidation window spans 9:30 ET VWAP reset
        if "timestamp" in df.columns:
            consol_timestamps = df.iloc[consol_start_idx:n]["timestamp"]
            for ts in consol_timestamps:
                if hasattr(ts, "hour") and hasattr(ts, "minute"):
                    # Check if any bar is at the 9:30 boundary
                    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                        from zoneinfo import ZoneInfo
                        ts_et = ts.astimezone(ZoneInfo("America/New_York"))
                    else:
                        ts_et = ts  # Assume already ET
                    h, m = ts_et.hour, ts_et.minute
                    # Check if window spans across 9:30
                    break  # Only need first and last
            # Compare first and last bar times
            first_ts = df.iloc[consol_start_idx].get("timestamp")
            last_ts = df.iloc[-1].get("timestamp")
            if first_ts is not None and last_ts is not None:
                if hasattr(first_ts, "hour"):
                    if hasattr(first_ts, "tzinfo") and first_ts.tzinfo is not None:
                        from zoneinfo import ZoneInfo
                        first_et = first_ts.astimezone(ZoneInfo("America/New_York"))
                        last_et = last_ts.astimezone(ZoneInfo("America/New_York"))
                    else:
                        first_et = first_ts
                        last_et = last_ts
                    first_minutes = first_et.hour * 60 + first_et.minute
                    last_minutes = last_et.hour * 60 + last_et.minute
                    reset_minutes = 9 * 60 + 30  # 9:30 ET
                    if first_minutes < reset_minutes <= last_minutes:
                        return self.not_detected(
                            "Consolidation spans 9:30 ET VWAP reset"
                        )

        # Step 12: MACD (confidence boost, not hard gate)
        if macd is None:
            macd = self.calculate_macd(df["close"])

        macd_positive = None
        macd_slope_up = None
        if macd is not None and "histogram" in macd.columns and len(macd) == n:
            macd_positive = macd.iloc[-1]["histogram"] > 0
            if "macd" in macd.columns and len(macd) >= 4:
                macd_slope_up = macd.iloc[-1]["macd"] > macd.iloc[-4]["macd"]

        if self.config.get("require_macd_positive") and macd_positive == False:
            return self.not_detected("HARD GATE: MACD histogram negative")

        # Step 13: Confidence scoring
        confidence = 0.65

        # Strong VWAP slope (8+ of 10 rising)
        if rising_count >= 8:
            confidence += 0.08

        # Gap narrowing confirmed
        if gap_narrowing:
            confidence += 0.06

        # MACD positive
        if macd_positive:
            confidence += 0.06

        # MACD slope up
        if macd_slope_up:
            confidence += 0.04

        # Tight consolidation (< 1% range)
        if consol_range_pct is not None and consol_range_pct < 1.0:
            confidence += 0.04

        # Volume declining during consolidation
        if volume_declining:
            confidence += 0.03

        # Price-VWAP gap at entry
        entry_vwap_gap_pct = round(
            (entry_candle["close"] - vwap_at_entry) / entry_candle["close"] * 100, 2
        ) if entry_candle["close"] > 0 else None

        return PatternResult(
            detected=True,
            pattern_name="VwapBounce",
            confidence=min(confidence, 0.90),
            entry_price=entry_price,
            stop_price=stop_price,
            stop_distance_cents=stop_distance_cents,
            pattern_start_idx=consol_start_idx,
            pattern_end_idx=n - 1,
            candle_count=consol_bars + 1,  # consolidation + entry
            above_vwap=entry_candle["close"] > vwap_at_entry if pd.notna(vwap_at_entry) else None,
            macd_positive=macd_positive,
            macd_slope_up=macd_slope_up,
            volume_confirmation=volume_declining,
            reason="Pattern detected",
            details={
                "consolidation_bars": consol_bars,
                "consolidation_range_pct": consol_range_pct,
                "consolidation_high": consol_high,
                "consolidation_low": consol_low,
                "vwap_rising_bars": rising_count,
                "vwap_slope_lookback": lookback,
                "vwap_at_entry": round(vwap_at_entry, 4),
                "price_vwap_gap_pct": entry_vwap_gap_pct,
                "gap_start_pct": gap_start_pct,
                "gap_end_pct": gap_end_pct,
                "gap_narrowing": gap_narrowing,
                "entry_zone_ceiling": round(entry_zone_ceiling, 4),
                "volume_declining": volume_declining,
                "atr": round(atr_value, 4) if atr_value is not None else None,
                "stop_buffer": round(stop_buffer, 4),
            },
        )
