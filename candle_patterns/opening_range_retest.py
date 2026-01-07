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
            if ts <= or_end or ts > window_end:
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

        retest_hit = (last_bar["low"] <= zone_high) and (last_bar["high"] >= zone_low)
        if not retest_hit:
            return self.not_detected("No retest in zone")

        # Build result
        confidence = 0.70
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
    ) -> List[ExitSignal]:
        """
        Check for ORB-specific exit signals.

        Exits:
        - Stop hit
        - Re-entry into opening range by X% of range
        - Opposite breakout beyond the other side of the range
        """
        signals = super().check_exit_signals(bars, entry_idx, entry_price, stop_price)

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
