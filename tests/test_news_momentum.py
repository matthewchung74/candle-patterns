"""
Tests for NewsMomentum pattern detector.

Run with: pytest tests/test_news_momentum.py -v

NewsMomentum fires on the first full volume bar after a classified
bullish catalyst. Unlike other detectors, it depends on metadata
(catalyst_verdict + news_article_time) passed via `_current_metadata`,
not on bar structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from candle_patterns import NewsMomentum

ET = ZoneInfo("America/New_York")


# ── Lightweight catalyst verdict stub ────────────────────────────────
# Mirrors the fields NewsMomentum reads from the real CatalystVerdict
# without pulling in the full Pydantic model from ibkr-scanner-core.


@dataclass
class _Verdict:
    is_catalyst: bool = True
    direction: str = "bullish"
    confidence: float = 0.85
    category: Optional[str] = "contract"
    summary: str = "Test catalyst"


# ── Bar fixture builder ──────────────────────────────────────────────


def _make_bars(
    rows: list[dict],
    start_et: datetime,
    interval_min: int = 1,
) -> pd.DataFrame:
    """Build a 1-min bars DataFrame with an ET DatetimeIndex."""
    timestamps = [start_et + timedelta(minutes=i * interval_min) for i in range(len(rows))]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, name="timestamp"))
    assert {"open", "high", "low", "close", "volume"}.issubset(df.columns)
    return df


def _canon_bars(news_minute: int = 5, symbol_price: float = 10.00) -> pd.DataFrame:
    """Canonical test bar window with a narrow-range news bar:
      bar 0-4: pre-news, no volume
      bar 5:   news bar — mostly green, small pullback (low = 98% of open)
      bar 6:   entry bar — volume, continues up modestly
      bar 7+:  post-entry bars

    The news bar's low is deliberately close to the open so the stop
    distance stays under the 8% max. This models a "clean" news bar, not
    the 87%-range RCT-style bar (which correctly fails the wide-stop gate).
    """
    start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
    rows = []
    for i in range(news_minute):
        rows.append({"open": symbol_price, "high": symbol_price, "low": symbol_price,
                     "close": symbol_price, "volume": 0})
    # News bar: open $10.00, low $9.80 (−2%), high $10.40, close $10.30
    rows.append({
        "open": symbol_price,
        "high": symbol_price * 1.04,
        "low": symbol_price * 0.98,
        "close": symbol_price * 1.03,
        "volume": 500_000,
    })
    # Entry bar: open $10.30, low $10.25, close $10.35
    rows.append({
        "open": symbol_price * 1.03,
        "high": symbol_price * 1.05,
        "low": symbol_price * 1.025,
        "close": symbol_price * 1.035,
        "volume": 200_000,
    })
    # Two more bars after
    rows.append({
        "open": symbol_price * 1.035,
        "high": symbol_price * 1.06,
        "low": symbol_price * 1.03,
        "close": symbol_price * 1.05,
        "volume": 180_000,
    })
    rows.append({
        "open": symbol_price * 1.05,
        "high": symbol_price * 1.08,
        "low": symbol_price * 1.045,
        "close": symbol_price * 1.065,
        "volume": 150_000,
    })
    return _make_bars(rows, start)


def _news_time(bars: pd.DataFrame, bar_idx: int) -> datetime:
    """Return the ET timestamp of the bar at `bar_idx`."""
    return bars.index[bar_idx].to_pydatetime().astimezone(ET)


class TestNewsMomentumHappyPath:
    """The happy-path cases where NewsMomentum should fire."""

    def setup_method(self):
        self.detector = NewsMomentum()
        self.bars = _canon_bars(news_minute=5, symbol_price=10.00)  # $10 stock → above min_price
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(self.bars, 5),
        }

    def test_happy_path_fires(self):
        r = self.detector.detect(self.bars)
        assert r.detected, f"should detect: {r.reason}"
        assert r.pattern_name == "NewsMomentum"

    def test_entry_is_close_plus_buffer(self):
        r = self.detector.detect(self.bars)
        assert r.detected
        # Entry bar is index 6, close is 10 * 1.035 = 10.35
        # Default entry_buffer_cents=2 → limit = 10.37
        assert r.entry_price == pytest.approx(10.37, abs=0.005)

    def test_entry_buffer_zero_uses_raw_close(self):
        detector = NewsMomentum(config={"entry_buffer_cents": 0})
        detector._current_metadata = self.detector._current_metadata
        r = detector.detect(self.bars)
        assert r.detected
        assert r.entry_price == pytest.approx(10.35, abs=0.005)

    def test_stop_below_news_bar_low(self):
        r = self.detector.detect(self.bars)
        assert r.detected
        # news_low = 10 * 0.98 = 9.80
        # entry_low = 10 * 1.025 = 10.25
        # min = 9.80; stop = 9.80 * 0.99 = 9.702
        assert r.stop_price == pytest.approx(9.702, rel=0.01)
        assert r.stop_price < r.entry_price

    def test_target_is_4r(self):
        r = self.detector.detect(self.bars)
        assert r.detected
        risk = r.entry_price - r.stop_price
        assert r.target_price == pytest.approx(r.entry_price + 4 * risk, rel=0.01)

    def test_confidence_equals_catalyst_confidence(self):
        r = self.detector.detect(self.bars)
        assert r.confidence == pytest.approx(0.85)

    def test_details_carry_category(self):
        r = self.detector.detect(self.bars)
        assert r.details is not None
        assert r.details["category"] == "contract"
        assert r.details["news_bar_volume"] == 500_000


class TestNewsMomentumMetadataGate:
    """Reject when catalyst metadata is missing, wrong direction, or too weak."""

    def setup_method(self):
        self.detector = NewsMomentum()
        self.bars = _canon_bars(news_minute=5, symbol_price=10.00)

    def _md(self, **overrides):
        md = {
            "catalyst_verdict": _Verdict(**overrides),
            "news_article_time": _news_time(self.bars, 5),
        }
        return md

    def test_no_metadata_at_all_rejects(self):
        self.detector._current_metadata = None
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "no catalyst metadata" in (r.reason or "")

    def test_missing_news_time_rejects(self):
        self.detector._current_metadata = {"catalyst_verdict": _Verdict()}
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "news_article_time" in (r.reason or "")

    def test_not_catalyst_rejects(self):
        self.detector._current_metadata = self._md(is_catalyst=False)
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "not a catalyst" in (r.reason or "")

    def test_bearish_direction_rejects(self):
        self.detector._current_metadata = self._md(direction="bearish", category="dilution")
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "bearish" in (r.reason or "")

    def test_neutral_direction_rejects(self):
        self.detector._current_metadata = self._md(direction="neutral")
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "neutral" in (r.reason or "")

    def test_low_confidence_rejects(self):
        self.detector._current_metadata = self._md(confidence=0.70)
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "confidence" in (r.reason or "")

    def test_category_not_in_whitelist_rejects(self):
        self.detector._current_metadata = self._md(category="fda")  # not in default whitelist
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "whitelist" in (r.reason or "")


class TestNewsMomentumTimingGate:
    """Reject when news falls outside entry window or is too old."""

    def setup_method(self):
        self.detector = NewsMomentum()

    def test_news_before_entry_window_rejects(self):
        # News at 03:55 ET — before default 04:00 start
        start = datetime(2026, 4, 14, 3, 50, tzinfo=ET)
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        rows.append({"open": 10.0, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 500_000})
        rows.append({"open": 11.5, "high": 12.2, "low": 11.4, "close": 12.0, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),  # 03:55
        }
        r = self.detector.detect(bars)
        assert not r.detected
        assert "outside entry window" in (r.reason or "")

    def test_news_after_entry_window_rejects(self):
        # News at 10:05 ET — after default 10:00 end
        start = datetime(2026, 4, 14, 10, 0, tzinfo=ET)
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        rows.append({"open": 10.0, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 500_000})
        rows.append({"open": 11.5, "high": 12.2, "low": 11.4, "close": 12.0, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),  # 10:05
        }
        r = self.detector.detect(bars)
        assert not r.detected
        assert "outside entry window" in (r.reason or "")

    def test_stale_news_rejects(self):
        """Entry bar arrives too late after the news — reject.

        The age check measures time from news_article_time to the ENTRY BAR,
        not the latest bar. So to test the stale-news rejection we need a
        window where the first volume bar is more than max_news_age_minutes
        after the news bar.
        """
        # News at bar 0 (08:00), but no volume for 15 bars, then volume spike.
        # Entry bar at 08:15 = 15 min after news → exceeds default 10 min cap.
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = []
        # News bar with zero volume (anchor only)
        rows.append({"open": 10.0, "high": 10.05, "low": 9.95, "close": 10.0, "volume": 0})
        # 14 more thin bars (no volume)
        for _ in range(14):
            rows.append({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0})
        # Finally a volume bar at index 15 (15 min after news)
        rows.append({"open": 10.0, "high": 10.1, "low": 9.98, "close": 10.05, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 0),  # bar 0 = 08:00
        }
        r = self.detector.detect(bars)
        assert not r.detected
        assert "too late" in (r.reason or "") or "entry delay window" in (r.reason or "")


class TestNewsMomentumBarGate:
    """Reject when the news bar or entry bar fails volume/structure requirements."""

    def setup_method(self):
        self.detector = NewsMomentum()

    def test_no_news_bar_in_range_rejects(self):
        """News time doesn't match any bar in the window."""
        bars = _canon_bars(news_minute=5, symbol_price=10.00)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": datetime(2026, 4, 14, 9, 0, tzinfo=ET),  # outside bars
        }
        r = self.detector.detect(bars)
        assert not r.detected
        # Any of these rejection reasons is acceptable — the important thing
        # is that we don't fire on a news event that doesn't map to a bar.
        reason = r.reason or ""
        assert (
            "news bar not found" in reason
            or "outside entry window" in reason
            or "news in the future" in reason
        )

    def test_thin_news_bar_passes_when_check_disabled(self):
        """Default min_news_bar_volume=0 — a thin news bar (e.g. 400 shares
        from an initial news tick on an ILLQ stock) does NOT reject. The
        entry bar's volume check handles thin stocks instead.
        """
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        # Tiny news bar (400 shares) — realistic for a pre-market news tick
        rows.append({"open": 10.0, "high": 10.2, "low": 9.95, "close": 10.1, "volume": 400})
        rows.append({"open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert r.detected, f"should detect despite thin news bar: {r.reason}"

    def test_explicit_news_bar_volume_floor_rejects(self):
        """When min_news_bar_volume is explicitly set > 0, it still gates."""
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        rows.append({"open": 10.0, "high": 10.2, "low": 9.95, "close": 10.1, "volume": 1_000})
        rows.append({"open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "volume": 200_000})
        bars = _make_bars(rows, start)
        detector = NewsMomentum(config={"min_news_bar_volume": 50_000})
        detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = detector.detect(bars)
        assert not r.detected
        assert "news bar volume" in (r.reason or "")

    def test_no_entry_bar_within_delay_rejects(self):
        """After news bar, no volume for N bars."""
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        rows.append({"open": 10.0, "high": 12.0, "low": 10.0, "close": 11.5, "volume": 500_000})
        # Next 3 bars all have zero volume
        rows.extend([{"open": 11.5, "high": 11.5, "low": 11.5, "close": 11.5, "volume": 0}] * 3)
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert not r.detected
        assert "entry delay window" in (r.reason or "")


class TestNewsMomentumPriceAndStopGates:
    """Reject on price floor or wide-news-bar stop safety."""

    def setup_method(self):
        self.detector = NewsMomentum()

    def test_price_below_floor_rejects(self):
        """Entry price below min_price ($0.50 default) → reject."""
        # $0.30 stock — after the 20% pop, still below $0.50
        bars = _canon_bars(news_minute=5, symbol_price=0.30)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert not r.detected
        assert "below min" in (r.reason or "")

    def test_wide_stop_caps_at_max_pct(self):
        """News bar with extreme range → stop capped at max_stop_pct, not rejected."""
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = [{"open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 0}] * 5
        # News bar has huge range: open $5, high $8, low $4.50, close $7
        rows.append({"open": 5.0, "high": 8.0, "low": 4.50, "close": 7.0, "volume": 500_000})
        # Entry bar: close $7.50
        rows.append({"open": 7.0, "high": 7.60, "low": 6.90, "close": 7.50, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        # Original stop would be 4.50 * 0.99 = 4.455 (40.6% from entry).
        # Now capped at 8%: stop = 7.52 * 0.92 = 6.9184
        assert r.detected
        assert r.stop_price == pytest.approx(7.52 * 0.92, abs=0.01)
        assert r.details["stop_distance_pct"] == 8.0

    def test_tight_max_stop_config_still_permits_narrow_stops(self):
        """When news bar is tight (2% range), stop fits comfortably."""
        bars = _canon_bars(news_minute=5, symbol_price=10.00)
        # Canon has ~20% range; tighten config to accept it
        self.detector.config["max_stop_pct_of_price"] = 25.0
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert r.detected


class TestNewsMomentumConfig:
    """Config loading and category whitelist defaults."""

    def test_default_config_has_whitelist(self):
        d = NewsMomentum()
        wl = d.config["allowed_categories"]
        assert "earnings" in wl
        assert "contract" in wl
        assert "partnership" in wl
        assert "buyback" in wl
        assert "fda" not in wl
        assert "dilution" not in wl
        assert "merger" not in wl

    def test_user_override_merges(self):
        d = NewsMomentum(config={"min_confidence": 0.90})
        assert d.config["min_confidence"] == 0.90
        # Other defaults preserved
        assert d.config["max_stop_pct_of_price"] == 8.0

    def test_max_entry_delay_bars_default(self):
        assert NewsMomentum().config["max_entry_delay_bars"] == 5

    def test_default_volume_floors(self):
        """Pre-market thin-stock-friendly defaults."""
        d = NewsMomentum()
        assert d.config["min_news_bar_volume"] == 0  # don't gate on news bar
        assert d.config["min_entry_bar_volume"] == 5_000  # loose for pre-market
