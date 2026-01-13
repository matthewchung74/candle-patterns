"""
Opening Range Retest Pattern Detector
=====================================

Opening Range Retest (ORB) with displacement and retest entry.

Rules:
1. Define opening range from 9:30-9:35 ET (first 5 minutes).
2. Require a displacement breakout beyond the range.
3. If no FVG, require confirmation close beyond prior candle high/low.
4. Enter on retest of the range level within a configurable zone.
5. Only the first valid retest in the first 90 minutes (one-shot).
"""

from typing import Optional, Dict, Any, List
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

from .base import PatternDetector, PatternResult, ExitSignal


class OpeningRangeRetest(PatternDetector):
    """
    Detect Opening Range Retest (ORB) pattern.

    Designed for liquid indices (QQQ, SPY) during the first 90 minutes.
    """

    def default_config(self) -> Dict[str, Any]:
        """Default configuration for ORB retest detection."""
        return {
            # Time windows (ET)
            "opening_range_minutes": 5,
            "setup_window_minutes": 90,

            # Displacement requirements
            "displacement_or_pct": 0.20,  # 20% of OR range
            "displacement_min_cents": 5,  # Min 5 cents beyond OR level
            "min_body_pct": 50.0,  # Candle body must be >= 50% of range

            # Breakout trigger: "close" (conservative) or "high" (aggressive)
            # "close" = bar must close above OR high + displacement
            # "high" = bar high touching above OR high + displacement triggers breakout
            "breakout_trigger": "close",

            # Retest zone (percent of OR range)
            "retest_zone_or_pct": 0.20,  # 20% of OR range

            # FVG requirement: "strict", "preferred", "off"
            "fvg_requirement": "preferred",

            # One-shot rule: only first valid retest
            "one_shot": True,

            # Stop buffer (percentage of OR range - scales with volatility)
            "stop_buffer_or_pct": 0.15,  # 15% of OR range

            # Trend alignment (5-min EMA slope)
            "trend_alignment": True,
            "trend_timeframe_minutes": 5,
            "trend_ema_period": 9,
            "trend_lookback_bars": 3,

            # Liquidity filter (first N minutes volume)
            "min_opening_volume": 0,
            "opening_volume_minutes": 15,

            # Clean handle requirement: at least one bar in the "handle" (between breakout
            # and retest) must be completely above/below OR level.
            # For longs: low > OR high. For shorts: high < OR low.
            # This filters weak setups that wick through OR and immediately reverse.
            # NOTE: Disabled for now - test impact separately
            "require_clean_breakout_bar": False,

            # Same-side fakeout filter: if price closes back inside OR after breakout
            # and before retest, invalidate the setup
            # NOTE: Disabled by default - aggressive filter, may reduce valid signals
            "fakeout_filter": False,
            "fakeout_bars": 2,  # Legacy fallback (now check all bars to retest)

            # Opposite-side invalidation: if price crosses the other side of OR before retest, invalidate
            # Default False to allow setups even if both sides were probed early
            "invalidate_on_opposite_break": False,

            # Choppy filter: reject if pre-breakout bars show consolidation
            # Triggers if 2 of 3 conditions met:
            # (a) body_pct < 40% for >= 60% of bars
            # (b) alternating colors >= 50% of bars
            # (c) net drift from OR mid < 20% of OR range
            # NOTE: Tested - hurts P&L, kept disabled
            "choppy_filter": False,
            "choppy_lookback_bars": 6,  # 5-8 bars before breakout
            "choppy_small_body_pct": 40.0,
            "choppy_small_body_ratio": 0.60,  # 60% of bars must have small bodies
            "choppy_alternating_ratio": 0.50,  # 50% alternating colors
            "choppy_drift_pct": 0.20,  # Net drift < 20% of OR range

            # Confirmation quality: entry bar must show strong rejection
            # Either hammer-like (lower_wick >= 2x body, upper_wick <= body)
            # Or strong bullish (body >= 60%, close > ORH + 0.1*OR range)
            # NOTE: Disabled - too aggressive, filters many valid entries
            "confirmation_filter": False,
            "confirm_hammer_wick_ratio": 2.0,  # lower_wick >= 2x body
            "confirm_strong_body_pct": 60.0,
            "confirm_strong_close_buffer": 0.10,  # close > ORH + 10% of OR range

            # Retest confirmation (engulfing/pinbar) - always applied
            # Mode: "strict" (engulfing/pinbar) or "basic" (reclaim without pattern check)
            "confirmation_mode": "basic",
            "confirm_body_ratio": 0.8,  # body must be >= 80% of prior body for engulfing check
            "confirm_wick_ratio": 2.0,   # wick multiple for pinbar-style rejection

            # MACD histogram hard gate (filters weak momentum entries)
            # Set to 0 to disable
            "min_histogram_threshold": 0,

            # MACD exit confirmation (delay exit to avoid false signals)
            # Require N consecutive bars with MACD below signal before exiting
            "macd_exit_confirmation_bars": 2,

            # Disable MACD cross exit (often premature - let other exits handle it)
            "disable_macd_exit": False,

            # Pullback volume filter (filters weak bounce / strong selling)
            # Avg volume of RED bars during pullback must be < bounce bar volume
            # Healthy pullback = low volume selling, high volume bounce
            "require_healthy_pullback_volume": True,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
    ) -> PatternResult:
        """
        Detect Opening Range Retest pattern.

        Args:
            bars: OHLCV DataFrame (newest bar last)
            vwap: Optional VWAP series (unused)
            macd: Optional MACD DataFrame (unused)

        Returns:
            PatternResult with detection details
        """
        try:
            self.validate_bars(bars)
        except ValueError as e:
            return self.not_detected(str(e))

        if "timestamp" not in bars.columns:
            return self.not_detected("Missing timestamp column")

        df = bars.copy().reset_index(drop=True)
        et_tz = ZoneInfo("America/New_York")

        # Ensure timestamps are tz-aware in ET
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["ts_et"] = df["timestamp"].dt.tz_localize(et_tz)
        else:
            df["ts_et"] = df["timestamp"].dt.tz_convert(et_tz)

        current_time = df["ts_et"].iloc[-1]
        session_date = current_time.date()

        or_start = datetime.combine(session_date, time(9, 30), tzinfo=et_tz)
        or_end = or_start + timedelta(minutes=self.config["opening_range_minutes"])
        window_end = or_start + timedelta(minutes=self.config["setup_window_minutes"])

        if current_time < or_end:
            return self.not_detected("Waiting for opening range to form")
        if current_time > window_end:
            return self.not_detected("Outside 90-minute window")

        # Use only session-day bars within setup window
        df_day = df[(df["ts_et"] >= or_start) & (df["ts_et"] <= window_end)]
        if df_day.empty:
            return self.not_detected("No session bars in setup window")

        # Opening range bars (9:30-9:35)
        or_bars = df_day[(df_day["ts_et"] >= or_start) & (df_day["ts_et"] < or_end)]
        if len(or_bars) < self.config["opening_range_minutes"]:
            return self.not_detected("Insufficient opening range bars")

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        or_range = or_high - or_low
        if or_range <= 0:
            return self.not_detected("Invalid opening range")

        # Liquidity filter
        min_open_vol = self.config.get("min_opening_volume", 0)
        if min_open_vol > 0:
            vol_end = or_start + timedelta(minutes=self.config["opening_volume_minutes"])
            vol_bars = df_day[(df_day["ts_et"] >= or_start) & (df_day["ts_et"] < vol_end)]
            if vol_bars["volume"].sum() < min_open_vol:
                return self.not_detected("Insufficient opening volume")

        # Displacement threshold
        disp_pct = self.config["displacement_or_pct"]
        disp_cents = self.config["displacement_min_cents"] / 100
        disp_threshold = max(or_range * disp_pct, disp_cents)

        # Find first breakout candle with displacement
        breakout_idx = None
        direction = None
        fvg_found = False
        confirmed = False

        for idx in range(len(df_day)):
            row = df_day.iloc[idx]
            ts = row["ts_et"]
            if ts < or_end or ts > window_end:
                continue

            prev = df_day.iloc[idx - 1] if idx > 0 else None
            body_pct = self.candle_body_pct(row)

            # Breakout trigger: use "high"/"low" or "close" based on config
            use_high_trigger = self.config.get("breakout_trigger", "close") == "high"

            # Long breakout
            breakout_price = row["high"] if use_high_trigger else row["close"]
            if breakout_price > or_high + disp_threshold and body_pct >= self.config["min_body_pct"]:
                if prev is None:
                    continue
                fvg = prev["high"] < row["low"]
                if self.config["fvg_requirement"] == "strict" and not fvg:
                    continue
                if not fvg:
                    confirmed = row["close"] > prev["high"]
                    if not confirmed:
                        continue
                breakout_idx = idx
                direction = "long"
                fvg_found = fvg
                break

            # Short breakout
            breakout_price = row["low"] if use_high_trigger else row["close"]
            if breakout_price < or_low - disp_threshold and body_pct >= self.config["min_body_pct"]:
                if prev is None:
                    continue
                fvg = prev["low"] > row["high"]
                if self.config["fvg_requirement"] == "strict" and not fvg:
                    continue
                if not fvg:
                    confirmed = row["close"] < prev["low"]
                    if not confirmed:
                        continue
                breakout_idx = idx
                direction = "short"
                fvg_found = fvg
                break

        if breakout_idx is None:
            return self.not_detected("No displacement breakout")

        # Same-side fakeout filter: check ALL bars after breakout until retest bar
        if self.config.get("fakeout_filter", False):
            # Legacy fakeout_bars retained for compatibility, but we scan all bars
            for i in range(breakout_idx + 1, len(df_day)):
                check_bar = df_day.iloc[i]
                bars_after_breakout = i - breakout_idx
                if direction == "long" and check_bar["close"] < or_high:
                    return self.not_detected(
                        f"Same-side fakeout: bar {bars_after_breakout} closed below ORH"
                    )
                if direction == "short" and check_bar["close"] > or_low:
                    return self.not_detected(
                        f"Same-side fakeout: bar {bars_after_breakout} closed above ORL"
                    )

        # Choppy filter: reject if pre-breakout bars show consolidation
        if self.config.get("choppy_filter", False):
            lookback = self.config.get("choppy_lookback_bars", 6)
            or_end_idx = len(or_bars)  # First bar after OR formation
            start_idx = max(or_end_idx, breakout_idx - lookback)
            pre_breakout = df_day.iloc[start_idx:breakout_idx]

            if len(pre_breakout) >= 2:  # Need at least 2 bars to assess
                # Condition (a): small bodies
                small_body_pct = self.config.get("choppy_small_body_pct", 40.0)
                small_body_count = sum(
                    1 for _, bar in pre_breakout.iterrows()
                    if self.candle_body_pct(bar) < small_body_pct
                )
                small_body_ratio = small_body_count / len(pre_breakout)
                cond_a = small_body_ratio >= self.config.get("choppy_small_body_ratio", 0.60)

                # Condition (b): alternating colors
                colors = [bar["close"] > bar["open"] for _, bar in pre_breakout.iterrows()]
                alternating_count = sum(
                    1 for i in range(1, len(colors)) if colors[i] != colors[i-1]
                )
                alternating_ratio = alternating_count / max(1, len(colors) - 1)
                cond_b = alternating_ratio >= self.config.get("choppy_alternating_ratio", 0.50)

                # Condition (c): low drift from OR mid
                or_mid = (or_high + or_low) / 2
                first_close = pre_breakout.iloc[0]["close"]
                last_close = pre_breakout.iloc[-1]["close"]
                drift = abs(last_close - first_close)
                drift_ratio = drift / or_range if or_range > 0 else 0
                cond_c = drift_ratio < self.config.get("choppy_drift_pct", 0.20)

                # 2 of 3 conditions = choppy
                chop_count = sum([cond_a, cond_b, cond_c])
                if chop_count >= 2:
                    return self.not_detected(
                        f"Choppy pre-breakout: {chop_count}/3 conditions "
                        f"(small_body={cond_a}, alternating={cond_b}, low_drift={cond_c})"
                    )

        # Trend alignment (5-min EMA slope)
        if self.config.get("trend_alignment", False):
            trend_ok = self._trend_alignment_ok(df_day, direction)
            if not trend_ok:
                return self.not_detected("Trend alignment failed")

        # Optional opposite-side invalidation: by default, allow setups even if price briefly crossed the other side
        if self.config.get("invalidate_on_opposite_break", False):
            or_end_idx = len(or_bars)  # First bar after OR formation
            post_or_bars = df_day.iloc[or_end_idx:]
            for _, row in post_or_bars.iterrows():
                close = row["close"]
                if direction == "long" and close <= or_low:
                    return self.not_detected("Breakout invalidated: price broke below OR low before entry")
                if direction == "short" and close >= or_high:
                    return self.not_detected("Breakout invalidated: price broke above OR high before entry")

        # Check for clean handle (at least one bar completely above/below OR level)
        # between breakout and retest. This ensures real follow-through before pullback,
        # not just a wick through OR that immediately reverses.
        if self.config.get("require_clean_breakout_bar", False):
            clean_bar_found = False
            # Check bars AFTER breakout (the "handle" before retest)
            handle_bars = df_day.iloc[breakout_idx + 1:]  # Skip breakout bar itself
            for _, row in handle_bars.iterrows():
                if direction == "long" and row["low"] > or_high:
                    # Entire bar above OR high = real follow-through in handle
                    clean_bar_found = True
                    break
                if direction == "short" and row["high"] < or_low:
                    # Entire bar below OR low = real follow-through in handle
                    clean_bar_found = True
                    break
            if not clean_bar_found:
                return self.not_detected(
                    f"No clean handle: need bar completely {'above OR high' if direction == 'long' else 'below OR low'} before retest"
                )

        # Retest zone
        zone_size = or_range * self.config["retest_zone_or_pct"]
        # Stop buffer scales with OR range (wider stops on volatile days)
        stop_buffer = or_range * self.config["stop_buffer_or_pct"]
        if direction == "long":
            zone_low = or_high - zone_size
            zone_high = or_high + zone_size
            entry_price = or_high
            stop_price = or_high - stop_buffer
        else:
            zone_low = or_low - zone_size
            zone_high = or_low + zone_size
            entry_price = or_low
            stop_price = or_low + stop_buffer

        # Retest must be the most recent bar
        # Use prev_bar for confirmation (completed), entry_candle for current bar (only open is known)
        if len(df_day) < 2:
            return self.not_detected("Insufficient bars for retest")
        prev_bar = df_day.iloc[-2]  # Completed bar (for confirmation)
        entry_candle = df_day.iloc[-1]  # Current bar (only use open)
        last_idx = df_day.index[-1]
        breakout_time = df_day.iloc[breakout_idx]["ts_et"]
        if entry_candle["ts_et"] <= breakout_time:
            return self.not_detected("Waiting for retest")

        # Require at least 1 bar between breakout and confirmation bar for a proper retest
        # breakout_idx should be at least 2 bars before entry (i.e., before prev_bar)
        # This ensures there's room for an actual pullback, not just immediate entry after breakout
        if breakout_idx >= len(df_day) - 2:
            return self.not_detected(
                f"No retest: breakout too recent (need pullback between breakout and entry)"
            )

        # Check for actual pullback to retest zone AFTER breakout but BEFORE bounce bar
        # Bug fix: Previously the zone was calculated but never validated
        # For LONG: price must pull back INTO the retest zone (low touches zone)
        # For SHORT: price must pull back INTO the retest zone (high touches zone)
        # Only check bars between breakout and prev_bar (exclude both prev_bar and entry bar)
        post_breakout = df_day.iloc[breakout_idx + 1:-2]  # Exclude bounce bar and entry bar
        pullback_found = False
        pullback_start_idx = None  # Track when pullback to retest zone started

        for i, (idx, row) in enumerate(post_breakout.iterrows()):
            if direction == "long":
                # For longs: check if bar's low reached into the retest zone
                # Zone is centered around OR high: zone_low to zone_high
                if row["low"] <= zone_high:
                    pullback_found = True
                    pullback_start_idx = breakout_idx + 1 + i  # Absolute index in df_day
                    break
            else:
                # For shorts: check if bar's high reached into the retest zone
                # Zone is centered around OR low: zone_low to zone_high
                if row["high"] >= zone_low:
                    pullback_found = True
                    pullback_start_idx = breakout_idx + 1 + i  # Absolute index in df_day
                    break

        if not pullback_found:
            return self.not_detected(f"No retest: price never pulled back to zone (${zone_low:.2f}-${zone_high:.2f})")

        # Retest confirmation: after pullback, price must reclaim the OR level
        # Use prev_bar for confirmation (no lookahead bias)
        # For LONG: prev bar closed above OR high OR current bar opened above OR high
        # For SHORT: prev bar closed below OR low OR current bar opened below OR low
        if direction == "long":
            retest_confirmed = (prev_bar["close"] > or_high) or (entry_candle["open"] > or_high)
            if not retest_confirmed:
                return self.not_detected(
                    f"No retest: prev close {prev_bar['close']:.2f}, "
                    f"curr open {entry_candle['open']:.2f} <= OR high {or_high:.2f}"
                )
            # Prev bar should be bullish (bounce confirmation)
            if prev_bar["close"] <= prev_bar["open"]:
                return self.not_detected("No retest: confirmation bar not bullish")
        else:
            retest_confirmed = (prev_bar["close"] < or_low) or (entry_candle["open"] < or_low)
            if not retest_confirmed:
                return self.not_detected(
                    f"No retest: prev close {prev_bar['close']:.2f}, "
                    f"curr open {entry_candle['open']:.2f} >= OR low {or_low:.2f}"
                )
            # Prev bar should be bearish (bounce confirmation)
            if prev_bar["close"] >= prev_bar["open"]:
                return self.not_detected("No retest: confirmation bar not bearish")

        # Confirmation: strict (engulfing/pinbar) or basic reclaim based on mode
        confirm_mode = self.config.get("confirmation_mode", "basic")
        if confirm_mode == "strict":
            if direction == "long":
                is_confirmed = self._bullish_confirmation(prev_bar, entry_candle, or_high)
                if not is_confirmed:
                    return self.not_detected(
                        f"No confirmation: need bullish engulfing or pinbar at OR high "
                        f"(prev: O={prev_bar['open']:.2f} C={prev_bar['close']:.2f}, "
                        f"curr: O={entry_candle['open']:.2f} C={entry_candle['close']:.2f} H={entry_candle['high']:.2f} L={entry_candle['low']:.2f})"
                    )
            else:
                is_confirmed = self._bearish_confirmation(prev_bar, entry_candle, or_low)
                if not is_confirmed:
                    return self.not_detected(
                        f"No confirmation: need bearish engulfing or pinbar at OR low "
                        f"(prev: O={prev_bar['open']:.2f} C={prev_bar['close']:.2f}, "
                        f"curr: O={entry_candle['open']:.2f} C={entry_candle['close']:.2f} H={entry_candle['high']:.2f} L={entry_candle['low']:.2f})"
                    )
        else:
            # Basic confirmation: prior bar is already bullish/bearish and reclaimed OR;
            # no extra engulfing/pinbar check required.
            pass

        # MACD histogram hard gate (filter weak momentum entries)
        min_histogram = self.config.get("min_histogram_threshold", 0)
        if min_histogram > 0 and macd is not None and "histogram" in macd.columns:
            current_histogram = macd["histogram"].iloc[-1]
            if direction == "long":
                # For longs: histogram must be positive and above threshold
                if current_histogram < min_histogram:
                    return self.not_detected(
                        f"HARD GATE: MACD histogram {current_histogram:.4f} below threshold {min_histogram} for long"
                    )
            else:
                # For shorts: histogram must be negative and below -threshold
                if current_histogram > -min_histogram:
                    return self.not_detected(
                        f"HARD GATE: MACD histogram {current_histogram:.4f} above -{min_histogram} for short"
                    )

        # Pullback volume filter: avg RED bar volume during pullback < bounce bar volume
        # Healthy pullback = weak selling (low vol), strong bounce (high vol)
        # Only check bars from when price entered the retest zone, not all bars since breakout
        if self.config.get("require_healthy_pullback_volume", False) and "volume" in df_day.columns:
            # Pullback bars: from pullback_start (when price entered retest zone) to prev_bar
            # prev_bar is df_day.iloc[-2], so pullback is df_day.iloc[pullback_start_idx:-2]
            pullback_bars = df_day.iloc[pullback_start_idx:-2] if pullback_start_idx is not None else pd.DataFrame()
            bounce_volume = prev_bar["volume"] if "volume" in prev_bar.index else 0

            if len(pullback_bars) > 0 and bounce_volume > 0:
                # Get RED bars only (close < open = selling)
                red_bars = pullback_bars[pullback_bars["close"] < pullback_bars["open"]]

                if len(red_bars) > 0:
                    avg_red_volume = red_bars["volume"].mean()

                    if avg_red_volume >= bounce_volume:
                        return self.not_detected(
                            f"Pullback volume filter: avg RED bar vol {avg_red_volume:,.0f} >= "
                            f"bounce vol {bounce_volume:,.0f} (weak bounce, strong selling)"
                        )

        # Build result
        # Base confidence: 75% (displacement + retest)
        # +10% if FVG found (fair value gap confirms institutional buying/selling)
        # +5% if confirmed close beyond prior candle
        # +5% if trend aligned (disabled by default)
        confidence = 0.75
        if fvg_found:
            confidence += 0.10
        if confirmed:
            confidence += 0.05
        if self.config.get("trend_alignment", False):
            confidence += 0.05
        confidence = min(confidence, 0.95)

        return PatternResult(
            detected=True,
            pattern_name=self.__class__.__name__,
            confidence=confidence,
            entry_price=entry_price,
            stop_price=stop_price,
            pattern_start_idx=int(or_bars.index.min()),
            pattern_end_idx=int(last_idx),
            candle_count=len(df_day),
            exit_signals=None,
            details={
                "direction": direction,
                "or_high": or_high,
                "or_low": or_low,
                "or_range": or_range,
                "breakout_idx": int(df_day.index[breakout_idx]),
                "retest_idx": int(last_idx),
                "fvg": fvg_found,
                "confirmed": confirmed,
                "displacement": disp_threshold,
                "retest_zone_low": zone_low,
                "retest_zone_high": zone_high,
            },
        )

    def _bullish_confirmation(self, prev_bar: pd.Series, curr_bar: pd.Series, or_high: float) -> bool:
        """Bullish confirmation: engulfing or pinbar at OR high."""
        body = abs(curr_bar["close"] - curr_bar["open"])
        prev_body = abs(prev_bar["close"] - prev_bar["open"])
        body_ratio = self.config.get("confirm_body_ratio", 0.8)
        wick_ratio = self.config.get("confirm_wick_ratio", 2.0)

        engulfing = (
            curr_bar["close"] > curr_bar["open"]
            and body >= prev_body * body_ratio
            and curr_bar["close"] >= prev_bar["high"]
        )

        lower_wick = min(curr_bar["open"], curr_bar["close"]) - curr_bar["low"]
        upper_wick = curr_bar["high"] - max(curr_bar["open"], curr_bar["close"])
        pinbar = (
            body > 0
            and lower_wick >= wick_ratio * body
            and upper_wick <= wick_ratio * body
            and curr_bar["low"] <= or_high
            and curr_bar["close"] > curr_bar["open"]
        )

        return engulfing or pinbar

    def _bearish_confirmation(self, prev_bar: pd.Series, curr_bar: pd.Series, or_low: float) -> bool:
        """Bearish confirmation: engulfing or pinbar at OR low."""
        body = abs(curr_bar["close"] - curr_bar["open"])
        prev_body = abs(prev_bar["close"] - prev_bar["open"])
        body_ratio = self.config.get("confirm_body_ratio", 0.8)
        wick_ratio = self.config.get("confirm_wick_ratio", 2.0)

        engulfing = (
            curr_bar["close"] < curr_bar["open"]
            and body >= prev_body * body_ratio
            and curr_bar["close"] <= prev_bar["low"]
        )

        upper_wick = curr_bar["high"] - max(curr_bar["open"], curr_bar["close"])
        lower_wick = min(curr_bar["open"], curr_bar["close"]) - curr_bar["low"]
        pinbar = (
            body > 0
            and upper_wick >= wick_ratio * body
            and lower_wick <= wick_ratio * body
            and curr_bar["high"] >= or_low
            and curr_bar["close"] < curr_bar["open"]
        )

        return engulfing or pinbar

    def _trend_alignment_ok(self, df_day: pd.DataFrame, direction: str) -> bool:
        """Check 5-min EMA slope for trend alignment."""
        tf_minutes = self.config.get("trend_timeframe_minutes", 5)
        ema_period = self.config.get("trend_ema_period", 9)
        lookback = self.config.get("trend_lookback_bars", 3)

        df = df_day.copy()
        df = df.set_index("ts_et")
        close_5m = df["close"].groupby(df.index.floor(f"{tf_minutes}min")).last()

        if len(close_5m) < ema_period + lookback:
            return False

        ema = close_5m.ewm(span=ema_period, adjust=False).mean()
        slope = ema.iloc[-1] - ema.iloc[-(lookback + 1)]

        if direction == "long":
            return slope > 0
        return slope < 0

    def check_exit_signals(
        self,
        bars: pd.DataFrame,
        entry_idx: int,
        entry_price: float,
        stop_price: float,
        direction: str = "long",
        current_time: Optional[datetime] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> List[ExitSignal]:
        """
        Check for ORB-specific exit signals.

        Exits:
        - Stop hit
        - 90-min window expiry (11:00 AM ET)
        - Re-entry into opening range by X% of range
        - Opposite breakout beyond the other side of the range
        """
        signals = super().check_exit_signals(
            bars, entry_idx, entry_price, stop_price, direction,
            current_time=current_time, details=details
        )

        # Check 90-min ORB window exit (11:00 AM ET)
        # Exit 1 min early to allow order fill on next bar (backtest quirk)
        window_minutes = 90
        if self.exit_config:
            window_minutes = self.exit_config.get("window_exit_minutes", 90)
        exit_buffer_minutes = 1  # Exit 1 min before window expires

        if current_time is not None:
            et_tz = ZoneInfo("America/New_York")
            if current_time.tzinfo is None:
                current_time = current_time.replace(tzinfo=et_tz)
            else:
                current_time = current_time.astimezone(et_tz)

            market_open = current_time.replace(hour=9, minute=30, second=0, microsecond=0)
            minutes_since_open = (current_time - market_open).total_seconds() / 60

            if minutes_since_open >= (window_minutes - exit_buffer_minutes):
                signals.append(ExitSignal(
                    signal_type="window_exit",
                    triggered=True,
                    reason=f"90-min ORB window expired at {current_time.strftime('%H:%M')}",
                    price=bars.iloc[-1]["close"] if len(bars) > 0 else None,
                ))
                return signals  # Exit immediately on window expiry

        if "timestamp" not in bars.columns:
            return signals

        df = bars.copy().reset_index(drop=True)
        et_tz = ZoneInfo("America/New_York")

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["ts_et"] = df["timestamp"].dt.tz_localize(et_tz)
        else:
            df["ts_et"] = df["timestamp"].dt.tz_convert(et_tz)

        if entry_idx >= len(df) - 1:
            return signals

        entry_time = df.loc[entry_idx, "ts_et"]
        session_date = entry_time.date()
        or_start = datetime.combine(session_date, time(9, 30), tzinfo=et_tz)
        or_end = or_start + timedelta(minutes=self.config["opening_range_minutes"])

        or_bars = df[(df["ts_et"] >= or_start) & (df["ts_et"] < or_end)]
        if or_bars.empty:
            return signals

        or_high = float(or_bars["high"].max())
        or_low = float(or_bars["low"].min())
        or_range = or_high - or_low
        if or_range <= 0:
            return signals

        disp_pct = self.config["displacement_or_pct"]
        disp_cents = self.config["displacement_min_cents"] / 100
        disp_threshold = max(or_range * disp_pct, disp_cents)

        # Determine direction by entry relative to OR
        direction = "long" if entry_price >= or_high else "short"
        invalid_pct = self.config.get("retest_zone_or_pct", 0.20)
        invalid_level = or_range * invalid_pct

        post_entry = df.iloc[entry_idx + 1:].copy()

        for i, row in post_entry.iterrows():
            close = row["close"]

            if direction == "long":
                # Re-entry into OR by X% of range
                if close <= or_high - invalid_level:
                    signals.append(ExitSignal(
                        signal_type="orb_reentry",
                        triggered=True,
                        reason="ORB invalidation: re-entered opening range",
                        bar_idx=i,
                        price=close,
                    ))
                    break
                # Opposite breakout
                if close < or_low - disp_threshold:
                    signals.append(ExitSignal(
                        signal_type="orb_opposite_break",
                        triggered=True,
                        reason="ORB invalidation: broke below range",
                        bar_idx=i,
                        price=close,
                    ))
                    break
            else:
                if close >= or_low + invalid_level:
                    signals.append(ExitSignal(
                        signal_type="orb_reentry",
                        triggered=True,
                        reason="ORB invalidation: re-entered opening range",
                        bar_idx=i,
                        price=close,
                    ))
                    break
                if close > or_high + disp_threshold:
                    signals.append(ExitSignal(
                        signal_type="orb_opposite_break",
                        triggered=True,
                        reason="ORB invalidation: broke above range",
                        bar_idx=i,
                        price=close,
                    ))
                    break

        return signals
