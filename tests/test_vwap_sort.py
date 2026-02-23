"""Test VWAP handles out-of-order bars."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from candle_patterns.indicators.vwap import calculate_vwap


class TestVWAPSort:
    def test_out_of_order_bars_produce_correct_vwap(self):
        """Bars arriving out of timestamp order should still produce correct VWAP."""
        base = datetime(2024, 1, 15, 10, 0)
        ordered_bars = pd.DataFrame({
            "timestamp": [base + timedelta(minutes=i) for i in range(5)],
            "high": [10.0, 10.1, 10.2, 10.1, 10.3],
            "low": [9.8, 9.9, 10.0, 9.9, 10.1],
            "close": [9.9, 10.0, 10.1, 10.0, 10.2],
            "volume": [1000, 1200, 800, 900, 1100],
        })
        vwap_ordered = calculate_vwap(ordered_bars)

        shuffled = ordered_bars.sample(frac=1, random_state=42).reset_index(drop=True)
        vwap_shuffled = calculate_vwap(shuffled)

        np.testing.assert_array_almost_equal(
            vwap_ordered.values,
            vwap_shuffled.values,
            decimal=6,
        )
