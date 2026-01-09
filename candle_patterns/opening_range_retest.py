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

            # Stop buffer
            "stop_buffer_cents": 10,

            # Trend alignment (5-min EMA slope)
            "trend_alignment": True,
            "trend_timeframe_minutes": 5,
            "trend_ema_period": 9,
            "trend_lookback_bars": 3,

            # Liquidity filter (first N minutes volume)
            "min_opening_volume": 0,
            "opening_volume_minutes": 15,

            # Handle quality scoring (replaces binary clean handle filter)
            # Scores conviction of breakout before retest - higher = better setup
            "handle_scoring_enabled": True,
            "handle_score_threshold": 2,  # Minimum score to take trade

            # Scoring weights:
            # +2 if breakout bar closes above ORH (strong acceptance)
            # +1 for each handle bar that closes above ORH
            # -1 for each handle bar that closes below ORH
            # +1 if retest holds within buffer and reclaims
            "score_breakout_close": 2,
            "score_handle_close_above": 1,
            "score_handle_close_below": -1,
            "score_retest_reclaim": 1,

            # Buffer for OR level (fixes micro-wick false rejects)
            # Uses: max(min_buffer_cents, buffer_pct * price, buffer_atr_mult * ATR)
            "or_buffer_min_cents": 1,      # Minimum 1 cent buffer
            "or_buffer_pct": 0.0005,       # 0.05% of price
            "or_buffer_atr_mult": 0.1,     # 0.1 * ATR(1min)

            # Body-based clean bar (alternative to full bar above)
            # If True: open >= ORH and close >= ORH (wick below allowed)
            # If False: entire bar (low) must be above ORH
            "use_body_above_rule": True,
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

        # Handle quality scoring (replaces binary clean handle filter)
        # Scores conviction level based on closes, not just wicks
        handle_score = 0
        handle_reason = ""
        if self.config.get("handle_scoring_enabled", True):
            # Calculate buffer to avoid micro-wick false rejects
            or_level = or_high if direction == "long" else or_low
            buffer = self._calculate_or_buffer(df_day, or_level)

            # Score the handle quality
            handle_score, handle_reason = self._score_handle_quality(
                df_day, breakout_idx, or_high, or_low, direction, buffer
            )

            threshold = self.config.get("handle_score_threshold", 2)
            if handle_score < threshold:
                return self.not_detected(
                    f"Handle score {handle_score} < {threshold} ({handle_reason})"
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
                "handle_score": handle_score,
                "handle_reason": handle_reason,
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

    def _calculate_or_buffer(self, df: pd.DataFrame, or_level: float) -> float:
        """
        Calculate buffer for OR level to avoid micro-wick false rejects.

        Buffer = max(min_cents, pct * price, atr_mult * ATR)
        """
        min_cents = self.config.get("or_buffer_min_cents", 1) / 100
        pct_buffer = or_level * self.config.get("or_buffer_pct", 0.0005)

        # Calculate 1-min ATR if we have enough bars
        atr_buffer = 0
        if len(df) >= 14:
            high_low = df["high"] - df["low"]
            atr = high_low.rolling(14).mean().iloc[-1]
            atr_buffer = atr * self.config.get("or_buffer_atr_mult", 0.1)

        return max(min_cents, pct_buffer, atr_buffer)

    def _score_handle_quality(
        self,
        df_day: pd.DataFrame,
        breakout_idx: int,
        or_high: float,
        or_low: float,
        direction: str,
        buffer: float,
    ) -> tuple[int, str]:
        """
        Score the quality of the handle (bars between breakout and retest).

        Scoring (for longs, reverse for shorts):
        +2 if breakout bar closes above ORH (strong acceptance)
        +1 for each handle bar with close above ORH
        -1 for each handle bar with close below ORH
        +1 if retest bar reclaims level after pullback

        Uses buffer to avoid micro-wick false penalties.
        Uses body-above rule if configured (allows wicks).

        Returns:
            (score, reason_string)
        """
        score = 0
        reasons = []

        breakout_bar = df_day.iloc[breakout_idx]
        handle_bars = df_day.iloc[breakout_idx + 1:-1] if len(df_day) > breakout_idx + 2 else pd.DataFrame()
        retest_bar = df_day.iloc[-1] if len(df_day) > breakout_idx + 1 else None

        use_body = self.config.get("use_body_above_rule", True)
        or_level = or_high if direction == "long" else or_low
        buffered_level = or_level + buffer if direction == "long" else or_level - buffer

        # Score breakout bar close
        if direction == "long":
            if breakout_bar["close"] >= or_level:
                score += self.config.get("score_breakout_close", 2)
                reasons.append(f"breakout close ${breakout_bar['close']:.2f} >= ORH")
        else:
            if breakout_bar["close"] <= or_level:
                score += self.config.get("score_breakout_close", 2)
                reasons.append(f"breakout close ${breakout_bar['close']:.2f} <= ORL")

        # Score handle bars
        closes_above = 0
        closes_below = 0

        for _, row in handle_bars.iterrows():
            if direction == "long":
                # Check if bar "holds" above level
                if use_body:
                    # Body above: open and close both above (with buffer tolerance)
                    bar_above = row["open"] >= (or_level - buffer) and row["close"] >= (or_level - buffer)
                else:
                    # Full bar above: low above level
                    bar_above = row["low"] >= (or_level - buffer)

                if row["close"] >= or_level:
                    closes_above += 1
                elif row["close"] < (or_level - buffer):
                    closes_below += 1
            else:
                # Short direction
                if use_body:
                    bar_below = row["open"] <= (or_level + buffer) and row["close"] <= (or_level + buffer)
                else:
                    bar_below = row["high"] <= (or_level + buffer)

                if row["close"] <= or_level:
                    closes_above += 1  # "above" means favorable for the direction
                elif row["close"] > (or_level + buffer):
                    closes_below += 1

        score += closes_above * self.config.get("score_handle_close_above", 1)
        score += closes_below * self.config.get("score_handle_close_below", -1)

        if closes_above > 0:
            reasons.append(f"{closes_above} handle bar(s) closed favorable")
        if closes_below > 0:
            reasons.append(f"{closes_below} handle bar(s) closed unfavorable")

        # Score retest reclaim
        if retest_bar is not None:
            if direction == "long":
                # Pulled back then reclaimed: low touched zone, close back above
                if retest_bar["close"] >= or_level and retest_bar["low"] < or_level:
                    score += self.config.get("score_retest_reclaim", 1)
                    reasons.append("retest reclaimed ORH")
            else:
                if retest_bar["close"] <= or_level and retest_bar["high"] > or_level:
                    score += self.config.get("score_retest_reclaim", 1)
                    reasons.append("retest reclaimed ORL")

        reason_str = "; ".join(reasons) if reasons else "no scoring factors"
        return score, reason_str

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
