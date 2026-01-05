# Candle Patterns

Ross Cameron style momentum pattern detection for day trading.

## Overview

This library detects common momentum day trading patterns:

- **Micro Pullback** - Shallow retracement after a strong move
- **Bull Flag** - Consolidation pattern with declining volume
- **VWAP Break** - Price breaking above VWAP with volume

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
from candle_patterns import MicroPullback, BullFlag, VWAPBreak

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

A shallow retracement (1-2 candles, max 3%) after a strong prior move (3+ green candles, 5%+ gain).

```
[GREEN][GREEN][GREEN][GREEN][red][red][GREEN→ENTRY]
     Prior Move (5%+)        Pullback   New High
```

**Configuration:**
```python
detector = MicroPullback({
    "min_prior_move_pct": 5.0,
    "min_green_candles_prior": 3,
    "max_pullback_candles": 2,
    "max_pullback_pct": 3.0,
    "stop_loss_cents": 15,
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

Price breaking above VWAP with volume confirmation. Also detects VWAP Hold variant.

```python
from candle_patterns import VWAPBreak

detector = VWAPBreak()
result = detector.detect(bars, vwap=vwap_series)

if result.pattern_name == "VWAPHold":
    print("VWAP acted as support!")
```

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
| `macd_positive` | bool | MACD histogram positive (if provided) |
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
