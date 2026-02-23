"""Test percent-based stop buffers for BullFlag and VWAPBreak."""
import pytest
from candle_patterns import BullFlag, VWAPBreak


class TestBullFlagPercentStop:
    def test_cheap_stock_uses_min_floor(self):
        bf = BullFlag()
        flag_low = 2.00
        pct_buffer = flag_low * (bf.config["stop_buffer_pct"] / 100)
        min_buffer = bf.config["stop_buffer_min_cents"] / 100
        stop_buffer = max(pct_buffer, min_buffer)
        assert stop_buffer == 0.05  # Min floor wins

    def test_expensive_stock_uses_percent(self):
        bf = BullFlag()
        flag_low = 15.00
        pct_buffer = flag_low * (bf.config["stop_buffer_pct"] / 100)
        min_buffer = bf.config["stop_buffer_min_cents"] / 100
        stop_buffer = max(pct_buffer, min_buffer)
        assert stop_buffer == pytest.approx(0.075)

    def test_config_has_pct_and_min(self):
        bf = BullFlag()
        assert "stop_buffer_pct" in bf.config
        assert "stop_buffer_min_cents" in bf.config


class TestVWAPBreakPercentStop:
    def test_config_has_pct_and_min(self):
        vb = VWAPBreak()
        assert "stop_buffer_pct" in vb.config
        assert "stop_buffer_min_cents" in vb.config

    def test_expensive_stock_uses_percent(self):
        vb = VWAPBreak()
        vwap = 12.00
        pct_buffer = vwap * (vb.config["stop_buffer_pct"] / 100)
        min_buffer = vb.config["stop_buffer_min_cents"] / 100
        stop_buffer = max(pct_buffer, min_buffer)
        assert stop_buffer == pytest.approx(0.09)
