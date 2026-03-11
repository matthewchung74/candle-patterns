"""Tests for momentum deceleration check."""

import pandas as pd
import pytest

from candle_patterns.indicators.trend_confirmation import check_momentum_deceleration


def _make_5min_bars(closes, volumes=None):
    """Create a DataFrame mimicking 5-min OHLCV bars."""
    if volumes is None:
        volumes = [10000] * len(closes)
    data = {
        "open": [closes[0]] + closes[:-1],
        "high": [c * 1.005 for c in closes],
        "low": [c * 0.995 for c in closes],
        "close": closes,
        "volume": volumes,
    }
    return pd.DataFrame(data)


class TestMomentumDeceleration:
    def test_accelerating_with_volume_blocks(self):
        """Price accelerating + volume increasing -> BLOCK."""
        bars = _make_5min_bars(
            closes=[1.0, 1.05, 1.10, 1.20, 1.25, 1.32, 1.40, 1.45],
            volumes=[10000, 12000, 14000, 16000, 18000, 20000, 22000, 25000],
        )
        passed, reason = check_momentum_deceleration(bars)
        assert passed is False
        assert "accelerating" in reason.lower()

    def test_accelerating_but_volume_declining_allows(self):
        """Price accelerating but volume declining -> ALLOW (exhaustion)."""
        bars = _make_5min_bars(
            closes=[1.0, 1.05, 1.10, 1.20, 1.25, 1.32, 1.40, 1.45],
            volumes=[30000, 28000, 25000, 20000, 15000, 12000, 10000, 8000],
        )
        passed, reason = check_momentum_deceleration(bars)
        assert passed is True
        assert "exhaustion" in reason.lower()

    def test_decelerating_allows_short(self):
        """Price stalling (second half <2% ROC) -> ALLOW regardless of volume."""
        bars = _make_5min_bars(
            closes=[1.0, 1.05, 1.10, 1.20, 1.21, 1.215, 1.22, 1.22],
            volumes=[10000, 12000, 14000, 16000, 18000, 20000, 22000, 25000],
        )
        passed, reason = check_momentum_deceleration(bars)
        assert passed is True
        assert "decelerating" in reason.lower()

    def test_rolling_over_allows_short(self):
        """Stock rolling over (negative second half) -> ALLOW."""
        bars = _make_5min_bars(
            closes=[1.0, 1.05, 1.10, 1.20, 1.18, 1.15, 1.12, 1.10],
        )
        passed, _ = check_momentum_deceleration(bars)
        assert passed is True

    def test_insufficient_bars_blocks(self):
        """Not enough bars -> BLOCK (dangerous to short early in run)."""
        bars = _make_5min_bars(closes=[1.0, 1.05])
        passed, _ = check_momentum_deceleration(bars, lookback=8)
        assert passed is False

    def test_full_window_negative_passes(self):
        """Stock down overall across window -> ALLOW (no uptrend to block)."""
        bars = _make_5min_bars(
            closes=[1.30, 1.25, 1.20, 1.15, 1.18, 1.19, 1.20, 1.22],
        )
        passed, _ = check_momentum_deceleration(bars)
        assert passed is True

    def test_pullback_then_rip_blocks(self):
        """First half down, second half rips with volume -> BLOCK."""
        bars = _make_5min_bars(
            closes=[1.00, 0.95, 0.92, 0.90, 1.05, 1.15, 1.30, 1.45],
            volumes=[10000, 12000, 14000, 16000, 20000, 25000, 30000, 35000],
        )
        passed, reason = check_momentum_deceleration(bars)
        assert passed is False
        assert "accelerating" in reason.lower()

    def test_micro_move_not_blocked(self):
        """Tiny absolute moves shouldn't crash or produce absurd results."""
        bars = _make_5min_bars(
            closes=[2.00, 2.005, 2.01, 2.01, 2.015, 2.02, 2.025, 2.03],
        )
        passed, _ = check_momentum_deceleration(bars)
        assert isinstance(passed, bool)

    def test_none_bars_blocks(self):
        """None bars -> BLOCK."""
        passed, reason = check_momentum_deceleration(None)
        assert passed is False
        assert "insufficient" in reason.lower()
