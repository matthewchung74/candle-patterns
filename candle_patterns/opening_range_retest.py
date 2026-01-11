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

            # Retest zone (percent of OR range)
            "retest_zone_or_pct": 0.20,  # 20% of OR range

            # FVG requirement: "strict", "preferred", "off"
            "fvg_requirement": "preferred",

            # One-shot rule: only first valid retest
            "one_shot": True,

            # Stop buffer (widened from 10 to 15 for volatile ETFs)
            "stop_buffer_cents": 15,

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

            # Long breakout
            if row["close"] > or_high + disp_threshold and body_pct >= self.config["min_body_pct"]:
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
            if row["close"] < or_low - disp_threshold and body_pct >= self.config["min_body_pct"]:
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

        # Check for breakout invalidation: if price EVER broke opposite side, setup is invalid
        # For LONG breakout: invalidate if any close broke below OR low (anytime after OR formed)
        # For SHORT breakout: invalidate if any close broke above OR high (anytime after OR formed)
        # Bug fix: Check ALL bars after OR formation, not just after breakout candle
        # This catches cases like TQQQ Jan 2, 2026 where price broke OR low at 9:39-9:40
        # BEFORE breaking above OR high at 9:47
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
        if self.config.get("require_clean_breakout_bar", True):
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
        if direction == "long":
            zone_low = or_high - zone_size
            zone_high = or_high + zone_size
            entry_price = or_high
            stop_price = or_high - (self.config["stop_buffer_cents"] / 100)
        else:
            zone_low = or_low - zone_size
            zone_high = or_low + zone_size
            entry_price = or_low
            stop_price = or_low + (self.config["stop_buffer_cents"] / 100)

        # Retest must be the most recent bar (current bar)
        last_bar = df_day.iloc[-1]
        last_idx = df_day.index[-1]
        breakout_time = df_day.iloc[breakout_idx]["ts_et"]
        if last_bar["ts_et"] <= breakout_time:
            return self.not_detected("Waiting for retest")

        # Check for actual pullback to retest zone AFTER breakout
        # Bug fix: Previously the zone was calculated but never validated
        # For LONG: price must pull back INTO the retest zone (low touches zone)
        # For SHORT: price must pull back INTO the retest zone (high touches zone)
        post_breakout = df_day.iloc[breakout_idx + 1:]
        pullback_found = False

        for _, row in post_breakout.iterrows():
            if direction == "long":
                # For longs: check if bar's low reached into the retest zone
                # Zone is centered around OR high: zone_low to zone_high
                if row["low"] <= zone_high:
                    pullback_found = True
                    break
            else:
                # For shorts: check if bar's high reached into the retest zone
                # Zone is centered around OR low: zone_low to zone_high
                if row["high"] >= zone_low:
                    pullback_found = True
                    break

        if not pullback_found:
            return self.not_detected(f"No retest: price never pulled back to zone (${zone_low:.2f}-${zone_high:.2f})")

        # Retest confirmation: after pullback, price must reclaim the OR level
        # For LONG: bar's high must be above OR high AND bar must be bullish (green)
        # For SHORT: bar's low must be below OR low AND bar must be bearish (red)
        if direction == "long":
            if last_bar["high"] <= or_high:
                return self.not_detected("No retest: high not above OR high")
            if last_bar["close"] <= last_bar["open"]:
                return self.not_detected("No retest: entry bar not bullish")
        else:
            if last_bar["low"] >= or_low:
                return self.not_detected("No retest: low not below OR low")
            if last_bar["close"] >= last_bar["open"]:
                return self.not_detected("No retest: entry bar not bearish")

        # Confirmation quality filter: entry bar must show strong rejection
        if self.config.get("confirmation_filter", False):
            bar_range = last_bar["high"] - last_bar["low"]
            body = abs(last_bar["close"] - last_bar["open"])
            body_pct = (body / bar_range * 100) if bar_range > 0 else 0

            if direction == "long":
                lower_wick = min(last_bar["open"], last_bar["close"]) - last_bar["low"]
                upper_wick = last_bar["high"] - max(last_bar["open"], last_bar["close"])

                # Hammer-like: lower_wick >= 2x body AND upper_wick <= body
                hammer_ratio = self.config.get("confirm_hammer_wick_ratio", 2.0)
                is_hammer = (lower_wick >= hammer_ratio * body) and (upper_wick <= body)

                # Strong bullish: body >= 60% AND close > ORH + 0.1*OR range
                strong_body_pct = self.config.get("confirm_strong_body_pct", 60.0)
                close_buffer = self.config.get("confirm_strong_close_buffer", 0.10)
                is_strong = (body_pct >= strong_body_pct) and (last_bar["close"] > or_high + close_buffer * or_range)

                # Valid bullish: body >= 50% AND close > ORH (reclaimed level)
                is_valid_bullish = (body_pct >= 50.0) and (last_bar["close"] > or_high)

                if not (is_hammer or is_strong or is_valid_bullish):
                    return self.not_detected(
                        f"Weak confirmation: not hammer (lower_wick={lower_wick:.3f}, body={body:.3f}) "
                        f"and not strong (body_pct={body_pct:.1f}%, close={last_bar['close']:.2f})"
                    )
            else:
                upper_wick = last_bar["high"] - max(last_bar["open"], last_bar["close"])
                lower_wick = min(last_bar["open"], last_bar["close"]) - last_bar["low"]

                # Inverted hammer: upper_wick >= 2x body AND lower_wick <= body
                hammer_ratio = self.config.get("confirm_hammer_wick_ratio", 2.0)
                is_hammer = (upper_wick >= hammer_ratio * body) and (lower_wick <= body)

                # Strong bearish: body >= 60% AND close < ORL - 0.1*OR range
                strong_body_pct = self.config.get("confirm_strong_body_pct", 60.0)
                close_buffer = self.config.get("confirm_strong_close_buffer", 0.10)
                is_strong = (body_pct >= strong_body_pct) and (last_bar["close"] < or_low - close_buffer * or_range)

                # Valid bearish: body >= 50% AND close < ORL (reclaimed level)
                is_valid_bearish = (body_pct >= 50.0) and (last_bar["close"] < or_low)

                if not (is_hammer or is_strong or is_valid_bearish):
                    return self.not_detected(
                        f"Weak confirmation: not inv hammer (upper_wick={upper_wick:.3f}, body={body:.3f}) "
                        f"and not strong (body_pct={body_pct:.1f}%, close={last_bar['close']:.2f})"
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
