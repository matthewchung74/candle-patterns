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
        self.detector._current_metadata = self._md(category="offering")  # not in default whitelist
        r = self.detector.detect(self.bars)
        assert not r.detected
        assert "whitelist" in (r.reason or "")

    def test_category_product_accepted(self):
        """product is a bread-and-butter small-cap catalyst (ATNM, MMA)."""
        self.detector._current_metadata = self._md(category="product")
        r = self.detector.detect(self.bars)
        assert r.detected, f"product category should fire; got reason={r.reason!r}"

    def test_category_fda_accepted(self):
        """fda is a core biotech catalyst type (KYTX registrational trial)."""
        self.detector._current_metadata = self._md(category="fda")
        r = self.detector.detect(self.bars)
        assert r.detected, f"fda category should fire; got reason={r.reason!r}"


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
        """Entry price below min_price ($2.00 default) → reject."""
        # $1.50 stock — after the 20% pop, still below $2.00
        bars = _canon_bars(news_minute=5, symbol_price=1.50)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert not r.detected
        assert "below min" in (r.reason or "")

    def test_wide_stop_capped_not_rejected(self):
        """News bar with extreme range → stop CAPPED at max_stop_distance_pct,
        not rejected. Replaces the pre-cap behavior from commit d62aefb.
        (Cap was reintroduced 2026-04-22 after AGPU round-to-zero bug.)"""
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = [{"open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 0}] * 5
        rows.append({"open": 5.0, "high": 8.0, "low": 4.50, "close": 7.0, "volume": 500_000})
        rows.append({"open": 7.0, "high": 7.60, "low": 6.90, "close": 7.50, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert r.detected
        # With 20% cap: entry $7.52 → stop $6.016 → distance 20%
        assert r.details["stop_distance_pct"] == pytest.approx(20.0, abs=0.1)

    def test_wide_stop_capped_at_20pct(self):
        """When natural stop > 20% below entry, cap it at 20%.
        Today's AGPU (news bar is market-open bar, natural stop = 53%
        below entry) would have rounded shares to 0 without the cap."""
        start = datetime(2026, 4, 14, 8, 0, tzinfo=ET)
        rows = [{"open": 5.0, "high": 5.0, "low": 5.0, "close": 5.0, "volume": 0}] * 5
        # News bar: open $5, low $4.50, close $7 (big range, stop will be very wide)
        rows.append({"open": 5.0, "high": 8.0, "low": 4.50, "close": 7.0, "volume": 500_000})
        # Entry bar close $7.50, low $6.90
        rows.append({"open": 7.0, "high": 7.60, "low": 6.90, "close": 7.50, "volume": 200_000})
        bars = _make_bars(rows, start)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert r.detected
        # Entry = close + 2¢ = $7.52. Cap at 20% → stop = $7.52 * 0.80 = $6.016
        assert r.stop_price == pytest.approx(6.016, abs=0.01)
        assert r.details["stop_distance_pct"] == pytest.approx(20.0, abs=0.1)

    def test_narrow_stop_uses_natural_level(self):
        """Below the 20% cap, the natural stop is preserved (no cap applied)."""
        bars = _canon_bars(news_minute=5, symbol_price=10.00)  # narrow bar, stop ~3%
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert r.detected
        # Natural stop should be well under 20%
        assert r.details["stop_distance_pct"] < 10.0

    def test_tight_stop_still_works(self):
        """When news bar is tight (2% range), stop fits comfortably."""
        bars = _canon_bars(news_minute=5, symbol_price=10.00)
        self.detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = self.detector.detect(bars)
        assert r.detected


class TestNewsMomentumDriftGate:
    """Reject when price has drifted below the entry bar's close.

    Motivating case: ALBT 2026-04-21. Partnership news at 08:00:34, entry
    bar 08:01 close $0.56 (entry_price $0.58 with buffer). DeepSeek
    classification lagged; by the time detect() ran at 08:05, price had
    collapsed to ~$0.40. Without a drift check the stale $0.58 signal fires
    and the resulting long position is born below its own stop.
    """

    def test_albt_style_collapse_rejects(self):
        """Entry bar's close is the signal price. If subsequent bars fall
        more than max_entry_drift_pct below that close, the signal is stale.
        """
        # Bars replicated from ALBT 2026-04-21 08:00-08:05 ET, rounded.
        start = datetime(2026, 4, 21, 8, 0, tzinfo=ET)
        rows = [
            # News bar 08:00 (vol satisfies default min_news_bar_volume=0)
            {"open": 0.468, "high": 0.600, "low": 0.460, "close": 0.589, "volume": 76_916},
            # Entry bar 08:01 (first bar after news with vol >= 5k)
            {"open": 0.561, "high": 0.591, "low": 0.502, "close": 0.560, "volume": 603_338},
            # Continuation peak 08:02
            {"open": 0.560, "high": 0.610, "low": 0.545, "close": 0.599, "volume": 1_447_843},
            # Collapse bar 08:03
            {"open": 0.597, "high": 0.599, "low": 0.480, "close": 0.494, "volume": 1_306_543},
            # Collapse continuation 08:04
            {"open": 0.488, "high": 0.496, "low": 0.415, "close": 0.416, "volume": 1_094_043},
            # Latest bar 08:05 — price is now ~28% below entry_price
            {"open": 0.420, "high": 0.450, "low": 0.396, "close": 0.404, "volume": 357_785},
        ]
        bars = _make_bars(rows, start)
        # ALBT was a sub-$1 name and the default min_price floor would
        # reject it first — drop the floor here so we exercise the drift
        # gate (the behavior under test).
        detector = NewsMomentum(config={"min_price": 0.0})
        detector._current_metadata = {
            "catalyst_verdict": _Verdict(category="partnership"),
            "news_article_time": _news_time(bars, 0),
        }
        r = detector.detect(bars)
        assert not r.detected, (
            f"should reject stale signal after price collapse; got "
            f"entry={r.entry_price} stop={r.stop_price} reason={r.reason}"
        )
        assert "drift" in (r.reason or "").lower()

    def test_price_holding_still_fires(self):
        """Even if subsequent bars are choppy, if latest close stays within
        max_entry_drift_pct of entry_price the signal is still valid. This
        preserves late-DeepSeek entries when the setup hasn't broken.
        """
        bars = _canon_bars(news_minute=5, symbol_price=10.00)
        detector = NewsMomentum()
        detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = detector.detect(bars)
        assert r.detected, f"should fire when price holds: {r.reason}"

    def test_drift_check_uses_entry_price(self):
        """The drift threshold is measured against entry_price (signal level),
        not against the entry bar's low or the news bar's low."""
        start = datetime(2026, 4, 21, 8, 0, tzinfo=ET)
        # Entry bar close 10.00, buffer 0.02 → entry_price = 10.02.
        # Latest close 9.80 → drift = (10.02 - 9.80) / 10.02 ≈ 2.2% → rejects at 2%.
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        rows += [
            {"open": 10.0, "high": 10.1, "low": 9.95, "close": 10.05, "volume": 500_000},
            {"open": 10.05, "high": 10.10, "low": 9.98, "close": 10.00, "volume": 200_000},
            {"open": 10.00, "high": 10.00, "low": 9.78, "close": 9.80, "volume": 150_000},
        ]
        bars = _make_bars(rows, start)
        detector = NewsMomentum()
        detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = detector.detect(bars)
        assert not r.detected
        assert "drift" in (r.reason or "").lower()

    def test_looser_drift_pct_allows_larger_move(self):
        """User can relax max_entry_drift_pct via config."""
        start = datetime(2026, 4, 21, 8, 0, tzinfo=ET)
        rows = [{"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 0}] * 5
        rows += [
            {"open": 10.0, "high": 10.1, "low": 9.95, "close": 10.05, "volume": 500_000},
            {"open": 10.05, "high": 10.10, "low": 9.98, "close": 10.00, "volume": 200_000},
            {"open": 10.00, "high": 10.00, "low": 9.78, "close": 9.80, "volume": 150_000},
        ]
        bars = _make_bars(rows, start)
        detector = NewsMomentum(config={"max_entry_drift_pct": 5.0})
        detector._current_metadata = {
            "catalyst_verdict": _Verdict(),
            "news_article_time": _news_time(bars, 5),
        }
        r = detector.detect(bars)
        assert r.detected, f"2.2% drift should pass when cap is 5%: {r.reason}"


class TestNewsMomentumConfig:
    """Config loading and category whitelist defaults."""

    def test_default_config_has_whitelist(self):
        d = NewsMomentum()
        wl = d.config["allowed_categories"]
        assert "earnings" in wl
        assert "contract" in wl
        assert "partnership" in wl
        assert "buyback" in wl
        assert "pivot" in wl
        assert "product" in wl
        assert "fda" in wl
        assert "offering" not in wl
        assert "dilution" not in wl
        assert "merger" not in wl

    def test_user_override_merges(self):
        d = NewsMomentum(config={"min_confidence": 0.90})
        assert d.config["min_confidence"] == 0.90
        # Other defaults preserved
        assert d.config["stop_buffer_pct"] == 1.0

    def test_max_entry_delay_bars_default(self):
        assert NewsMomentum().config["max_entry_delay_bars"] == 5

    def test_default_volume_floors(self):
        """Pre-market thin-stock-friendly defaults."""
        d = NewsMomentum()
        assert d.config["min_news_bar_volume"] == 0  # don't gate on news bar
        assert d.config["min_entry_bar_volume"] == 5_000  # loose for pre-market

    def test_max_entry_drift_pct_default(self):
        """2% default catches collapsing plays without killing choppy ones."""
        assert NewsMomentum().config["max_entry_drift_pct"] == 2.0
