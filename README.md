# Candle Patterns

Momentum pattern detection library for day trading.

## Why This Library?

Tired of rewriting the same pattern detection logic every time I start a new trading project. This is a standalone, reusable library to stop that cycle.

## Overview

This library detects common momentum day trading patterns:

- **Micro Pullback** - Shallow retracement after a strong move
- **Bull Flag** - Consolidation pattern with declining volume
- **VWAP Break** - Price breaking above VWAP with volume
- **Opening Range Retest** - ORB breakout with displacement + retest
- **ABCD** - Harmonic pattern with Fibonacci retracements

## Installation

```bash
pip install candle-patterns
```

Or install from source:

```bash
git clone https://github.com/YOUR_USER/candle-patterns.git
cd candle-patterns
pip install -e .
```

## Quick Start

```python
import pandas as pd
from candle_patterns import MicroPullback, BullFlag, VWAPBreak, OpeningRangeRetest, ABCD

# Your OHLCV data (newest bar last)
bars = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...],
})

# Detect Micro Pullback
detector = MicroPullback()
result = detector.detect(bars)

if result.detected:
    print(f"Pattern: {result.pattern_name}")
    print(f"Entry: ${result.entry_price:.2f}")
    print(f"Stop: ${result.stop_price:.2f}")
    print(f"Confidence: {result.confidence:.0%}")
```

## Patterns

### Micro Pullback

A flexible retracement after a strong prior move, tuned for Ross's actual 1-3 candle pullbacks.

```
[GREEN][GREEN][GREEN][GREEN][red][red][GREEN→ENTRY]
     Prior Move (5%+)        Pullback   New High
```

**Configuration:**
```python
detector = MicroPullback({
    "min_prior_move_pct": 5.0,
    "min_green_candles_prior": 2,
    "max_pullback_pct": 20.0,
    # Two-tier candle limits based on pullback depth
    "shallow_pullback_threshold_pct": 12.0,  # <=12% is shallow, >12% is deep
    "max_pullback_candles_shallow": 12,      # Shallow pullbacks: more time
    "max_pullback_candles_deep": 7,          # Deep pullbacks: resolve quickly
    # Percent-based stop with minimum floor
    "stop_buffer_pct": 1.0,                  # 1% below pullback low
    "stop_buffer_min_cents": 3,              # Minimum 3 cents
})
```

### Bull Flag

Classic continuation pattern: strong pole (20%+ move) followed by consolidation with declining volume.

```
       ___
      /   \___  <- Flag (consolidation)
     /        \___
    /             \
   /               → BREAKOUT
  / <- Pole
```

**Configuration:**
```python
detector = BullFlag({
    "min_pole_move_pct": 20.0,
    "min_pullback_pct": 10.0,
    "max_pullback_pct": 25.0,
    "volume_declining": True,
})
```

### VWAP Break

Price breaking above VWAP with volume confirmation. The VWAP Hold variant is supported but disabled by default.

```python
from candle_patterns import VWAPBreak

detector = VWAPBreak()
result = detector.detect(bars, vwap=vwap_series)

if result.pattern_name == "VWAPHold":
    print("VWAP acted as support!")
```

To enable the VWAP Hold variant:

```python
detector = VWAPBreak({
    "vwap_hold_variant": {
        "enabled": True,
    },
})
```

### Opening Range Retest (ORB)

Opening Range Retest with displacement and retest entry within the first 90 minutes.

```python
detector = OpeningRangeRetest({
    "opening_range_minutes": 5,   # 9:30-9:35 ET
    "setup_window_minutes": 90,   # 9:30-11:00 ET
    "displacement_or_pct": 0.20,  # % of OR range
    "retest_zone_or_pct": 0.20,   # % of OR range
    "fvg_requirement": "preferred",
    "trend_alignment": True,      # 5-min EMA slope
})
```

### ABCD (Harmonic)

Harmonic pattern with Fibonacci retracements. Detects both bullish and bearish ABCD patterns.

```
Bullish ABCD:
    A - Swing low (start of impulse)
    B - Swing high (end of AB leg)
    C - Higher low (BC retracement of 38.2%-78.6% of AB)
    D - Projected target where CD ≈ AB

       B
      /\        D
     /  \      /
    /    \    /
   A      \  /
           C

Entry at D completion, stop below C
```

**Configuration:**
```python
detector = ABCD({
    "min_bc_retracement": 0.382,  # 38.2% min (Fibonacci)
    "max_bc_retracement": 0.786,  # 78.6% max (Fibonacci)
    "cd_ab_ratio_min": 0.75,      # CD must be at least 75% of AB
    "cd_ab_ratio_max": 1.25,      # CD must be at most 125% of AB
    "min_leg_pct": 1.0,           # Min 1% move for AB leg
    "swing_lookback": 3,          # Bars to confirm swing points
    "direction_filter": None,     # None = both, "long" or "short"
})

result = detector.detect(bars)
if result.detected:
    print(f"Direction: {result.details['direction']}")
    print(f"Projected D: ${result.details['projected_d']:.2f}")
    print(f"BC Retracement: {result.details['bc_retracement']:.1%}")
```

## Confirmations

### MACD (Auto-calculated)

MACD is automatically calculated when >= 35 bars are provided. No need to pass it manually.

- Uses standard (12, 26, 9) parameters
- `macd_positive` = True when histogram > 0
- `macd_slope_up` = True when MACD line is above its value 3 bars ago
- Adds confidence bonuses when positive

```python
result = detector.detect(bars)  # MACD calculated internally
print(result.macd_positive)  # True/False/None (None if < 35 bars)
print(result.macd_slope_up)  # True/False/None (None if < 35 bars)
```

### Volume Confirmation

- **Micro Pullback**: Pullback volume must be < surge volume
- **Bull Flag**: Volume must decline during flag consolidation
- **VWAP Break**: Volume spike (2x avg) on break

```python
print(result.volume_confirmation)  # True if volume confirms pattern
```

## Confidence Scoring

Confidence is a 0.0–1.0 score returned by each detector. It is advisory only; gating is handled by the consumer.

Standard bases/caps:
- **Micro Pullback / Bull Flag / VWAP Break / VWAP Hold / ABCD**: base 0.65, cap 0.90
- **Opening Range Retest**: base 0.70, cap 0.95

Boosts by pattern:

| Pattern | Boost | Notes |
|--------|-------|-------|
| Micro Pullback | volume_declining +0.10 | Pullback volume < surge volume |
| Micro Pullback | above_vwap +0.08 | Price above VWAP |
| Micro Pullback | macd_positive +0.08 | MACD histogram > 0 |
| Micro Pullback | macd_slope_up +0.04 | MACD line rising vs 3 bars ago |
| Micro Pullback | tight_pullback +0.06 | Pullback < 5% |
| Bull Flag | volume_declining +0.10 | Flag volume declining |
| Bull Flag | above_vwap +0.08 | Price above VWAP |
| Bull Flag | above_9ema +0.06 | Price above 9 EMA |
| Bull Flag | macd_positive +0.08 | MACD histogram > 0 |
| Bull Flag | macd_slope_up +0.04 | MACD line rising vs 3 bars ago |
| VWAP Break | volume_spike +0.10 | Break volume spike |
| VWAP Break | macd_positive +0.08 | MACD histogram > 0 |
| VWAP Break | macd_slope_up +0.04 | MACD line rising vs 3 bars ago |
| VWAP Hold (disabled) | macd_positive +0.08 | MACD histogram > 0 |
| VWAP Hold (disabled) | macd_slope_up +0.04 | MACD line rising vs 3 bars ago |
| Opening Range Retest | fvg_found +0.10 | Displacement/FVG present |
| Opening Range Retest | confirmed +0.05 | Breakout confirmation |
| Opening Range Retest | trend_alignment +0.05 | 5-min EMA slope aligned |
| ABCD | ideal_retracement | Higher confidence near 61.8% BC retracement |
| ABCD | cd_ab_match | Higher confidence when CD ≈ AB (perfect harmonic) |

## Trailing Stop

Lock in profits with a 2-bar low trailing stop that activates after reaching a profit threshold.

### How It Works

1. **Activation**: Trailing activates after:
   - Reaching +1R profit (configurable), OR
   - Taking a partial (scale-out)

2. **Calculation**: `Stop = min(low of last 2 bars) - buffer`

3. **Dynamic Buffer**: `buffer = max(spread × 2, ATR(14) × 0.1)`
   - Adapts to both liquidity (spread) and volatility (ATR)
   - Wide spreads → larger buffer (avoid getting stopped by spread)
   - High volatility → larger buffer (avoid noise stops)

4. **Safety**: Stop never moves against you (longs: never lowers, shorts: never raises)

### Usage

```python
from candle_patterns import MicroPullback

detector = MicroPullback()

# Track the trailed stop across calls to prevent "giving back" gains
trailed_stop = None

# After entry, on each new bar:
result = detector.calculate_trailing_stop(
    bars=updated_bars,
    entry_idx=5,
    entry_price=10.00,
    original_stop=9.50,
    current_spread=0.02,           # Current bid-ask spread
    partial_taken=False,            # Set True after taking partial
    direction="long",               # or "short"
    previous_trailed_stop=trailed_stop,  # Pass last trailed stop to prevent loosening
)

if result.active:
    trailed_stop = result.new_stop  # Save for next iteration
    print(f"Trailing active! New stop: ${result.new_stop:.2f}")
    print(f"Current R: {result.current_r_multiple:.1f}")
    print(f"High water mark: ${result.high_water_mark:.2f}")
```

**Important**: Pass `previous_trailed_stop` to prevent the stop from loosening after a peak. Without it, a pullback after a run-up could lower the trailing stop.

### Configuration

```python
trailing_config = {
    "trailing_bars": 2,           # Number of bars for low/high calculation
    "activation_r": 1.0,          # R-multiple to activate (default 1.0)
    "spread_multiplier": 2.0,     # Spread × 2 for buffer
    "atr_multiplier": 0.1,        # ATR × 0.1 for buffer
    "atr_period": 14,             # ATR lookback period
    "min_bars_after_entry": 2,    # Bars to wait before trailing
}

result = detector.calculate_trailing_stop(
    bars=bars,
    entry_idx=5,
    entry_price=10.00,
    original_stop=9.50,
    trailing_config=trailing_config,
)
```

### TrailingStopResult

| Field | Type | Description |
|-------|------|-------------|
| `active` | bool | Whether trailing stop is active |
| `new_stop` | float | New stop price (may equal original if not trailing yet) |
| `original_stop` | float | Original stop for reference |
| `high_water_mark` | float | Highest high since entry (longs) |
| `current_r_multiple` | float | Current profit in R multiples |
| `is_trailing` | bool | True if stop has moved from original |
| `reason` | str | Explanation of trailing status |

## Exit Signals

Monitor for trade invalidation after entry:

```python
# After entering a trade, check for exit signals on each new bar
signals = detector.check_exit_signals(
    bars=updated_bars,
    entry_idx=6,
    entry_price=5.43,
    stop_price=5.14
)

for signal in signals:
    if signal.triggered:
        print(f"EXIT: {signal.signal_type} - {signal.reason}")
```

| Signal | Trigger | Meaning |
|--------|---------|---------|
| `stop_hit` | Low <= stop price | Hard stop triggered |
| `macd_cross` | MACD crosses below signal | Momentum fading |
| `volume_decline` | Volume < 50% of entry | Buyers drying up |
| `jackknife` | New high then close below prior low | Trapped buyers, reversal |

## PatternResult

All detectors return a `PatternResult` with:

| Field | Type | Description |
|-------|------|-------------|
| `detected` | bool | Whether pattern was found |
| `pattern_name` | str | Name of the pattern |
| `confidence` | float | 0.0 to 1.0 confidence score |
| `entry_price` | float | Suggested entry price |
| `stop_price` | float | Suggested stop loss price |
| `stop_distance_cents` | float | Stop distance in cents |
| `above_vwap` | bool | Price above VWAP (if provided) |
| `macd_positive` | bool | MACD histogram positive (auto-calculated) |
| `macd_slope_up` | bool | MACD line slope positive (auto-calculated) |
| `volume_confirmation` | bool | Volume confirms pattern |
| `exit_signals` | list | Exit/invalidation signals |
| `reason` | str | Why pattern was/wasn't detected |
| `details` | dict | Pattern-specific metrics |

## Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Test fixtures in `tests/fixtures/` contain example bar data for each pattern.

## License

MIT License
