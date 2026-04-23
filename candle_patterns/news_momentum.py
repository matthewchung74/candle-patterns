"""
NewsMomentum Pattern
====================

Long entry on the first full volume bar after a classified bullish catalyst
hits the wire. Designed to fire on fast-moving catalyst plays that don't
form a MicroPullback structure before the first leg is over.

Fires when ALL of:
  - metadata.catalyst_verdict is a bullish catalyst with confidence >= min_confidence
  - metadata.catalyst_verdict.category is in allowed_categories (whitelist)
  - metadata.news_article_time is within entry_window_start..entry_window_end ET
  - metadata.news_article_time is recent (within max_news_age_minutes of current time)
  - a news bar exists in the bars DataFrame containing the news time
  - the news bar has volume >= min_news_bar_volume
  - an entry bar (first bar with volume after the news bar) exists within max_entry_delay_bars
  - entry price >= min_price

Entry rules:
  - entry_price = close of entry bar + entry_buffer_cents/100 (pay up a
    tick so the limit actually fills against the next bar — momentum
    plays typically continue through the signal bar's close)
  - stop_price  = min(news_bar.low, entry_bar.low) * (1 - stop_buffer_pct/100),
    capped at max_stop_distance_pct (default 20%) below entry to prevent
    risk-sizing from rounding to 0 shares on extreme gap-open news bars
  - target_price = entry_price + target_r_multiple * (entry_price - stop_price)

The catalyst metadata is threaded in via self._current_metadata, which the
ibkr-scanner-core adapter layer sets before calling detect(). See
`patterns/adapters.py::CandlePatternsDetectorAdapter`.

This detector does NOT call DeepSeek — it reads an already-classified
verdict. It also does NOT look at bar structure for pattern shape; it
relies entirely on the news event + volume presence.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from candle_patterns.base import PatternDetector, PatternResult

ET = ZoneInfo("America/New_York")


class NewsMomentum(PatternDetector):
    """Bar-after-news long entry for classified bullish catalysts."""

    def default_config(self) -> Dict[str, Any]:
        return {
            # Catalyst gate
            "min_confidence": 0.80,
            "allowed_categories": [
                "earnings",
                "contract",
                "partnership",
                "buyback",
                "pivot",
                "product",
                "fda",
            ],
            # News timing
            # max_news_age_minutes bounds "how far past the news we'll still
            # fire." Measured from news_article_time to the entry bar's time,
            # NOT the latest bar — so replay on a full-session window doesn't
            # trivially reject everything.
            "max_news_age_minutes": 10,
            "entry_window_start": "04:00",
            "entry_window_end": "10:00",
            # Stop/target
            "stop_buffer_pct": 1.0,
            # Cap the natural structural stop at this % below entry. Prevents
            # risk-sizing from flooring to 0 shares on gap-open news bars
            # (AGPU 2026-04-22: news bar was first bar of day, stop at 53%
            # below entry → 0.92 shares → rounds to 0, trade disappears).
            # Below the cap, the natural stop is preserved (d62aefb's logic:
            # "wider stop → fewer shares, not more risk").
            "max_stop_distance_pct": 20.0,
            # Pay-up buffer above the entry bar's close. Prevents stale
            # limits from getting gapped past on the next bar — momentum
            # plays usually open above the signal bar's close. Set to 0 to
            # disable. 2 cents is enough for sub-$20 micro-caps.
            "entry_buffer_cents": 2.0,
            "target_r_multiple": 4.0,
            # Volume floors.
            # min_news_bar_volume = 0: the news bar is just a timestamp anchor,
            #   not a volume requirement. In pre-market on thin stocks the news
            #   tick often prints only a few hundred shares. Don't reject here.
            # min_entry_bar_volume = 5_000: the entry bar is where we actually
            #   fill. 5k shares at $5+ is enough for a paper-size position.
            "min_news_bar_volume": 0,
            "min_entry_bar_volume": 5_000,
            # Price floor. $2 is empirical — sub-$2 trades on this system
            # have a ~8% WR with net-negative P&L across all sessions to date.
            "min_price": 2.00,
            # How many bars after the news bar to look for a volume entry.
            # Bumped from 2 → 5 so thin pre-market stocks that take 3-4 min
            # to spin up have a chance to hit the volume floor.
            "max_entry_delay_bars": 5,
            # Max drop of latest close below entry_price (pct). Classification
            # can lag; a broken setup should not. Reject if the play has
            # already walked away from the signal level.
            "max_entry_drift_pct": 2.0,
        }

    def detect(
        self,
        bars: pd.DataFrame,
        vwap: Optional[pd.Series] = None,
        macd: Optional[pd.DataFrame] = None,
        prev_close: Optional[float] = None,
    ) -> PatternResult:
        """Detect NewsMomentum entry. Metadata is passed via self._current_metadata."""
        self.validate_bars(bars)
        if len(bars) < 2:
            return self._no("not enough bars")

        md = getattr(self, "_current_metadata", None) or {}
        verdict = md.get("catalyst_verdict")
        news_time = md.get("news_article_time")

        # Gate 1: catalyst metadata present
        if verdict is None:
            return self._no("no catalyst metadata")
        if news_time is None:
            return self._no("no news_article_time metadata")

        # Gate 2: catalyst is bullish + high confidence + whitelisted category
        if not getattr(verdict, "is_catalyst", False):
            return self._no("not a catalyst")
        if getattr(verdict, "direction", None) != "bullish":
            return self._no(f"direction={getattr(verdict, 'direction', 'None')}, need bullish")
        conf = getattr(verdict, "confidence", 0.0) or 0.0
        min_conf = self.config["min_confidence"]
        if conf < min_conf:
            return self._no(f"confidence {conf:.2f} < {min_conf}")
        category = getattr(verdict, "category", None)
        if category not in self.config["allowed_categories"]:
            return self._no(f"category '{category}' not in whitelist")

        # Gate 3: news time inside entry window (ET)
        news_et = _to_et(news_time)
        news_hhmm = news_et.strftime("%H:%M")
        if not (self.config["entry_window_start"] <= news_hhmm < self.config["entry_window_end"]):
            return self._no(f"news {news_hhmm} outside entry window")

        # Gate 4: news must not be in the future relative to the latest bar
        # (sanity check — if news hasn't arrived yet there's nothing to do)
        latest_bar_time = _extract_bar_time(bars, -1)
        if latest_bar_time is not None:
            latest_et = _to_et(latest_bar_time)
            if (latest_et - news_et).total_seconds() < 0:
                return self._no("news in the future relative to latest bar")

        # Gate 5: find the news bar (bar whose timestamp matches news minute)
        news_bar_idx = self._find_news_bar(bars, news_et)
        if news_bar_idx is None:
            return self._no("news bar not found in bars window")

        news_bar = bars.iloc[news_bar_idx]
        news_vol = int(news_bar.get("volume", 0) or 0)
        if news_vol < self.config["min_news_bar_volume"]:
            return self._no(f"news bar volume {news_vol:,} < {self.config['min_news_bar_volume']:,}")

        # Gate 6: find the entry bar (first volume bar after news bar, within delay cap)
        entry_bar_idx = None
        for offset in range(1, self.config["max_entry_delay_bars"] + 1):
            idx = news_bar_idx + offset
            if idx >= len(bars):
                break
            if int(bars.iloc[idx].get("volume", 0) or 0) >= self.config["min_entry_bar_volume"]:
                entry_bar_idx = idx
                break
        if entry_bar_idx is None:
            return self._no("no volume bar within entry delay window")

        # Gate 6b: entry-bar age relative to news. The age check is done
        # against the ENTRY BAR, not the latest bar, so replay on a full-
        # session window can still fire (it finds the entry bar within the
        # first few minutes after news and is satisfied).
        entry_bar_time = _extract_bar_time(bars, entry_bar_idx)
        if entry_bar_time is not None:
            entry_age_min = (_to_et(entry_bar_time) - news_et).total_seconds() / 60
            if entry_age_min > self.config["max_news_age_minutes"]:
                return self._no(
                    f"entry bar too late ({entry_age_min:.1f} min after news, "
                    f"max {self.config['max_news_age_minutes']})"
                )

        entry_bar = bars.iloc[entry_bar_idx]
        entry_close = float(entry_bar["close"])
        entry_buffer = self.config.get("entry_buffer_cents", 0) / 100.0
        entry_price = entry_close + entry_buffer

        # Gate 7: price floor
        if entry_price < self.config["min_price"]:
            return self._no(f"price ${entry_price:.2f} below min ${self.config['min_price']:.2f}")

        # Gate 8: entry staleness by price drift. The age check above is
        # entry-bar-vs-news; this one is latest-bar-vs-entry-price. Needed
        # because classification (DeepSeek) can lag the entry bar by several
        # minutes, during which price may have already collapsed.
        latest_close = float(bars.iloc[-1]["close"])
        max_drift_pct = self.config["max_entry_drift_pct"]
        drift_pct = (entry_price - latest_close) / entry_price * 100
        if drift_pct > max_drift_pct:
            return self._no(
                f"entry stale: latest close ${latest_close:.4f} drifted "
                f"{drift_pct:.1f}% below entry ${entry_price:.4f} (max {max_drift_pct:.1f}%)"
            )

        # Build stop + target
        raw_stop = min(float(news_bar["low"]), float(entry_bar["low"]))
        stop_price = raw_stop * (1 - self.config["stop_buffer_pct"] / 100)

        if stop_price >= entry_price:
            return self._no(f"stop {stop_price:.4f} >= entry {entry_price:.4f}")

        stop_distance_pct = (entry_price - stop_price) / entry_price * 100
        max_stop = self.config.get("max_stop_distance_pct")
        if max_stop is not None and stop_distance_pct > max_stop:
            stop_price = round(entry_price * (1 - max_stop / 100), 4)
            stop_distance_pct = max_stop

        risk = entry_price - stop_price
        target_price = entry_price + self.config["target_r_multiple"] * risk

        return PatternResult(
            detected=True,
            pattern_name="NewsMomentum",
            confidence=conf,
            entry_price=round(entry_price, 4),
            stop_price=round(stop_price, 4),
            stop_distance_cents=round((entry_price - stop_price) * 100, 2),
            target_price=round(target_price, 4),
            pattern_start_idx=news_bar_idx,
            pattern_end_idx=entry_bar_idx,
            candle_count=entry_bar_idx - news_bar_idx + 1,
            reason="NewsMomentum entry signal",
            details={
                "news_article_time": news_et.isoformat(),
                "news_bar_volume": news_vol,
                "entry_bar_volume": int(entry_bar.get("volume", 0) or 0),
                "stop_distance_pct": round(stop_distance_pct, 2),
                "category": category,
                "catalyst_confidence": conf,
                "catalyst_summary": getattr(verdict, "summary", "")[:200],
            },
        )

    def _no(self, reason: str) -> PatternResult:
        """Shortcut for undetected result with a reason string."""
        return PatternResult(
            detected=False,
            pattern_name="NewsMomentum",
            confidence=0.0,
            reason=reason,
        )

    def _find_news_bar(self, bars: pd.DataFrame, news_time_et: datetime) -> Optional[int]:
        """Return the index of the bar whose minute matches news_time_et.

        Assumes bars are 1-minute bars. The news bar is the bar with the same
        HH:MM ET timestamp as the news article. If bars are timestamped by
        open time (standard), the news bar is the one covering news_time_et.
        """
        news_minute = news_time_et.replace(second=0, microsecond=0)
        for i in range(len(bars)):
            bar_time = _extract_bar_time(bars, i)
            if bar_time is None:
                continue
            bar_et = _to_et(bar_time).replace(second=0, microsecond=0)
            if bar_et == news_minute:
                return i
        return None


def _extract_bar_time(bars: pd.DataFrame, i: int) -> Optional[datetime]:
    """Pull the timestamp for bar index i from either a DatetimeIndex or a
    'timestamp' column. Monitor builds bars with timestamp as a column +
    RangeIndex; direct unit tests build bars with a DatetimeIndex. This
    handles both."""
    if isinstance(bars.index, pd.DatetimeIndex):
        return _bar_time(bars.index[i])
    if "timestamp" in bars.columns:
        return _bar_time(bars.iloc[i]["timestamp"])
    # Fallback: try .name attribute
    return _bar_time(bars.iloc[i].name)


def _bar_time(ts: Any) -> Optional[datetime]:
    """Coerce various timestamp types to a timezone-aware datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    try:
        return pd.to_datetime(ts).to_pydatetime()
    except Exception:
        return None


def _to_et(ts: datetime) -> datetime:
    """Convert a datetime to America/New_York, assuming UTC if naive."""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(ET)
