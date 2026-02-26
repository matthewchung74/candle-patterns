# Candle Patterns

Momentum pattern detection library for day trading.

## Why This Library?

Tired of rewriting the same pattern detection logic every time I start a new trading project. This is a standalone, reusable library to stop that cycle.

## Overview

This library detects common momentum day trading patterns:

**Long patterns:**
- **Micro Pullback** - Shallow retracement after a strong move
- **Bull Flag** - Consolidation pattern with declining volume
- **VWAP Break** - Price breaking above VWAP with volume
- **Opening Range Retest** - ORB breakout with displacement + retest (disabled by default)
- **ABCD** - Harmonic pattern with Fibonacci retracements

**Short (reversal) patterns:**
- **Shooting Star** - Long upper wick rejection at highs
- **Bearish Engulfing** - Red candle engulfs prior green at highs
- **Evening Star** - 3-bar topping pattern (green, doji, red)
- **Volume Climax** - Extreme volume spike at highs with red reversal

All patterns run on 1-minute OHLCV bars.

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
from candle_patterns import MicroPullback, BullFlag, VWAPBreak, ABCD, ReversalPatternDetector

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

# Detect reversal patterns (short side)
reversal = ReversalPatternDetector()
result = reversal.detect(bars)
if result.detected:
    print(f"Reversal: {result.pattern_name}")  # ShootingStar, BearishEngulfing, EveningStar, VolumeClimax
    print(f"Direction: {result.details['direction']}")  # always "short"
```

## Patterns

### Micro Pullback

A shallow retracement after a 5-15% move. Moves >15% are routed to Bull Flag instead.

```
[GREEN][GREEN][GREEN][GREEN][red][red][GREEN→ENTRY]
     Prior Move (5-15%)      Pullback   New High
```

**Hard gates** (pattern rejected if not met):
- Price must be above VWAP (`require_above_vwap: True`)
- MACD histogram must be positive (`require_macd_positive: True`)

**Configuration:**
```python
detector = MicroPullback({
    "min_prior_move_pct": 5.0,       # Min 5% move before pullback
    "max_prior_move_pct": 15.0,      # Max 15% (larger moves -> Bull Flag)
    "min_green_candles_prior": 2,     # At least 2 green candles in prior move
    "max_pullback_pct": 12.0,        # Max 12% retracement
    "max_pullback_candles": 3,        # Max 3 candles in pullback (micro = tight)
    "require_above_vwap": True,       # HARD GATE: must be above VWAP
    "require_macd_positive": True,    # HARD GATE: MACD histogram > 0
    "stop_buffer_pct": 1.0,           # 1% below pullback low
    "stop_buffer_min_cents": 3,       # Minimum 3 cents buffer
    "min_rr_for_setup": 2.0,          # Minimum 2:1 R:R required
})
```

### Bull Flag

Classic continuation pattern: strong pole (15%+ move) followed by tight 1-3 candle consolidation with declining volume.

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
    "min_pole_move_pct": 15.0,   # Min 15% move in pole
    "min_pole_candles": 3,       # 3-10 candles in pole
    "max_pole_candles": 10,
    "min_flag_candles": 1,       # 1-3 candles in flag
    "max_flag_candles": 3,
    "min_pullback_pct": 10.0,    # 10-25% retracement of pole
    "max_pullback_pct": 25.0,
    "volume_declining": True,    # Volume must decrease in flag
    "stop_buffer_cents": 5,      # 5 cents below flag low
    "min_rr_for_setup": 2.0,     # Minimum 2:1 R:R required
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

**Configuration:**
```python
detector = VWAPBreak({
    "min_time_below_minutes": 5,     # Min 5 bars below VWAP before break
    "volume_spike_on_break": 2.0,    # 2x average volume on break bar
    "close_above_vwap": True,        # Bar must close above VWAP
    "stop_buffer_cents": 10,         # 10 cents below VWAP
    "min_rr_for_setup": 2.0,         # Minimum 2:1 R:R required
    "vwap_hold_variant": {
        "enabled": False,            # Disabled by default
    },
})
```

### Reversal Patterns (Short Side)

Detects bearish reversal patterns on extended stocks for short entry. Requires the stock to be up 20%+ from open.

Sub-patterns detected (in priority order):
1. **Evening Star** - 3-bar pattern: strong green, small doji, red closes below green midpoint
2. **Volume Climax** - Volume >3x 20-bar avg at HOD with red candle or topping tail
3. **Shooting Star** - Long upper wick (>2x body), body in lower third, at HOD
4. **Bearish Engulfing** - Red candle fully engulfs prior green candle at HOD

```python
from candle_patterns import ReversalPatternDetector

detector = ReversalPatternDetector()
result = detector.detect(bars, vwap=vwap_series, macd=macd_df)

if result.detected:
    print(f"Pattern: {result.pattern_name}")  # e.g. "VolumeClimax"
    print(f"Direction: {result.details['direction']}")  # always "short"
```

**Configuration:**
```python
detector = ReversalPatternDetector({
    # Extension requirements
    "min_extension_from_open_pct": 20.0,  # Stock must be up 20%+ from open
    "min_extension_from_low_pct": 25.0,   # 25%+ from intraday low

    # Volume climax
    "volume_climax_multiplier": 3.0,      # Volume > 3x 20-bar avg
    "volume_avg_period": 20,

    # Shooting star
    "min_upper_wick_ratio": 2.0,          # Upper wick >= 2x body
    "max_body_position_pct": 33.0,        # Body in lower third

    # Bearish engulfing
    "min_engulf_ratio": 1.0,              # Red body >= green body

    # Evening star
    "max_middle_body_pct": 30.0,          # Middle candle body < 30% of range

    # Risk
    "stop_buffer_pct": 2.0,              # Stop 2% above recent HOD
    "stop_buffer_min_cents": 5,
    "min_rr_for_setup": 2.0,
})
```

### Opening Range Retest (ORB)

Opening Range Retest with displacement and retest entry within the first 90 minutes.

**Important:** This pattern is **disabled by default**. You must pass `{"enabled": True}` to activate it.

```python
detector = OpeningRangeRetest({
    "enabled": True,                  # Must enable explicitly
    "opening_range_minutes": 5,       # 9:30-9:35 ET
    "setup_window_minutes": 90,       # 9:30-11:00 ET
    "displacement_or_pct": 0.20,      # % of OR range
    "retest_zone_or_pct": 0.20,       # % of OR range
    "fvg_requirement": "preferred",   # "strict", "preferred", or "off"
    "trend_alignment": True,          # 5-min EMA slope
    "confirmation_mode": "basic",     # "basic" or "strict" (engulfing/pinbar)
    "one_shot": True,                 # Only first valid retest
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

    B         D
   / \       /
  /   \   C /
 /     \ / \
A       v

Entry at D completion, stop below C

Bearish ABCD:
    A - Swing high (start of impulse)
    B - Swing low (end of AB leg)
    C - Lower high (BC retracement of 38.2%-78.6% of AB)
    D - Projected target where CD ≈ AB

A       ^
 \     / \
  \   /   C \
   \ /       \
    B         D

Entry at D completion, stop above C
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
- **Bull Flag**: Volume must decline during flag consolidation (10% leeway)
- **VWAP Break**: Volume spike (2x avg over all bars) on break

```python
print(result.volume_confirmation)  # True if volume confirms pattern
```

## Confidence Scoring

Confidence is a 0.0-1.0 score returned by each detector. It is advisory only; gating is handled by the consumer.

Standard bases/caps:
- **Micro Pullback / Bull Flag / VWAP Break / VWAP Hold**: base 0.65, cap 0.90
- **Opening Range Retest**: base 0.75, cap 0.95
- **ABCD**: formula-based (0.85 adjusted by retracement and CD/AB match), range 0.50-0.95
- **Reversal patterns**: base from pattern weight (normalized to ~0.65), cap 0.90

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
| ABCD | cd_ab_match | Higher confidence when CD ~ AB |
| Reversal | above_vwap +0.06 | Room to fall to VWAP |
| Reversal | macd_weakening +0.06 | MACD histogram declining |
| Volume Climax | volume_bonus +0.05 | Extra confidence for climax signal |

## All Patterns Enforce min_rr_for_setup

Every pattern (except ABCD which uses its own formula) requires a minimum risk/reward ratio of 2:1 by default. If the calculated entry/stop doesn't achieve this, the pattern is not returned as detected. This is configurable per-pattern via `min_rr_for_setup`.

## Stale Data Guard

Micro Pullback and Bull Flag silently reject patterns where the suggested entry price is >5% away from the current close (`max_entry_deviation_pct: 5.0`). This prevents acting on stale pattern calculations.

## Trailing Stop

Lock in profits with configurable trailing stop strategies that activate after reaching a profit threshold.

### Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `swing_low` | Trail to N-bar low minus buffer | Momentum trades |
| `atr` | Trail N x ATR from high water mark | Volatile stocks |

### How It Works

1. **Activation**: Trailing activates after:
   - Reaching +1R profit (configurable), OR
   - Taking a partial (scale-out)

2. **Calculation**:
   - `swing_low`: `Stop = min(low of last N bars) - buffer`
   - `atr`: `Stop = high_water_mark - (N x ATR)`

3. **Dynamic Buffer** (swing_low): `buffer = max(spread x 2, ATR(14) x 0.1)`
   - Adapts to both liquidity (spread) and volatility (ATR)

4. **Safety**: Stop never moves against you (longs: never lowers, shorts: never raises)

### Usage

```python
from candle_patterns import (
    calculate_trailing_stop,
    TrailingStopState,
    TrailingStopConfig,
)

# Initialize state once at entry
state = TrailingStopState.from_entry(
    entry_price=10.00,
    stop_price=9.50,
    direction="long",
    entry_idx=5,
)

# Configure strategy (defaults to swing_low)
config = TrailingStopConfig(
    strategy="swing_low",      # or "atr"
    activation_r=1.0,          # Activate at 1R profit
    current_spread=0.02,       # Current bid-ask spread
)

# On each new bar, calculate trailing stop
result = calculate_trailing_stop(bars, state, config)

if result.active:
    # Update state for next iteration (prevents stop from loosening)
    state.current_stop = result.new_stop
    state.high_water_mark = result.high_water_mark
    state.is_activated = True

    print(f"Trailing active! New stop: ${result.new_stop:.2f}")
    print(f"Current R: {result.current_r_multiple:.1f}")
    print(f"Strategy: {result.strategy_name}")
```

### Configuration Examples

```python
# Default (swing low, activate at 1R)
config = TrailingStopConfig()

# ATR trailing with 2.5x multiplier
config = TrailingStopConfig(
    strategy="atr",
    activation_r=1.0,
    params={"atr_multiplier": 2.5}
)

# Tight swing low (1-bar)
config = TrailingStopConfig(
    strategy="swing_low",
    activation_r=0.75,
    params={"trailing_bars": 1}
)

# Swing low with custom buffer
config = TrailingStopConfig(
    strategy="swing_low",
    current_spread=0.05,
    params={
        "trailing_bars": 2,
        "spread_multiplier": 2.0,
        "atr_multiplier": 0.1,
    }
)
```

### Strategy Parameters

**swing_low**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `trailing_bars` | 2 | Number of bars for low/high calculation |
| `spread_multiplier` | 2.0 | Spread x N for buffer |
| `atr_multiplier` | 0.1 | ATR x N for buffer |
| `atr_period` | 14 | ATR lookback period |

**atr**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `atr_multiplier` | 2.0 | Trail N x ATR from high water mark |
| `atr_period` | 14 | ATR lookback period |

### TrailingStopState

| Field | Type | Description |
|-------|------|-------------|
| `entry_price` | float | Price at which position was entered |
| `original_stop` | float | Original stop loss price |
| `current_stop` | float | Current trailing stop (update after each call) |
| `direction` | str | "long" or "short" |
| `high_water_mark` | float | Best price since entry (update after each call) |
| `is_activated` | bool | Whether trailing has activated (update after each call) |
| `risk_per_share` | float | Risk per share (entry - stop) |
| `partial_taken` | bool | Set True when partial taken to force activation |
| `entry_idx` | int | Bar index of entry |

### TrailingStopResult

| Field | Type | Description |
|-------|------|-------------|
| `active` | bool | Whether trailing stop is active |
| `new_stop` | float | New stop price (may equal original if not trailing yet) |
| `original_stop` | float | Original stop for reference |
| `high_water_mark` | float | Highest high since entry (longs) / lowest low (shorts) |
| `current_r_multiple` | float | Current profit in R multiples |
| `is_trailing` | property | True if stop has moved from original |
| `reason` | str | Explanation of trailing status |
| `strategy_name` | str | Name of strategy used |
| `just_activated` | bool | True if trailing just activated on this bar |
| `stop_moved` | bool | True if stop moved from previous level |

## Exit Signals

Monitor for trade invalidation after entry. All exit signals are direction-aware (work for both longs and shorts).

```python
# After entering a trade, check for exit signals on each new bar
signals = detector.check_exit_signals(
    bars=updated_bars,
    entry_idx=6,
    entry_price=5.43,
    stop_price=5.14,
    direction="long",     # or "short"
    vwap=vwap_series,     # optional, enables vwap_cross exit
)

for signal in signals:
    if signal.triggered:
        print(f"EXIT: {signal.signal_type} - {signal.reason}")
```

| Signal | Long Trigger | Short Trigger |
|--------|-------------|---------------|
| `stop_hit` | Low <= stop price | High >= stop price |
| `macd_cross` | MACD crosses below signal (N bars confirmed) | MACD crosses above signal (N bars confirmed) |
| `vwap_cross` | Price closes below VWAP (N bars confirmed) | Price closes above VWAP (N bars confirmed) |
| `volume_decline` | Volume < 50% of entry bar | Volume < 50% of entry bar |
| `jackknife` | New high then close below prior low | N/A |
| `bottoming_rejection` | N/A | New low then close above prior high |
| `topping_tail` | Long upper wick rejection while in profit | N/A |
| `bottoming_tail` | N/A | Long lower wick rejection while in profit |

MACD and VWAP exits use configurable confirmation bars (`macd_exit_confirmation_bars`, `vwap_exit_confirmation_bars`) to filter false signals. Default is 1 bar (immediate), but patterns can override (e.g. ORB uses 2 bars for MACD).

**ORB-specific exits** (in addition to the above):
- `orb_reentry` - Price re-enters the opening range
- `orb_opposite_break` - Price breaks the opposite side of the OR
- `window_exit` - Setup window expired

## Indicators Module

Standalone indicator calculations available via `candle_patterns.indicators`:

```python
from candle_patterns.indicators import (
    # ATR (Wilder's smoothing)
    calculate_atr, get_current_atr,
    # EMA
    calculate_ema, calculate_all_emas, price_above_ema, ema_slope,
    # VWAP (dual-mode: premarket from 4AM, regular resets at 9:30)
    calculate_vwap, calculate_premarket_vwap, calculate_regular_vwap,
    # MACD
    calculate_macd, macd_is_positive, macd_crossover, macd_histogram_slope,
    # RVOL (time-of-day relative volume)
    calculate_rvol_tod, calculate_cumulative_rvol, is_premarket, is_regular_hours,
    # Trend Confirmation (5-min)
    check_5min_trend_confirmation, check_candle_quality,
    is_green_candle, is_red_candle, is_doji, count_consecutive_dojis,
)
```

Note: `check_5min_trend_confirmation` expects 5-minute bars and checks if the higher timeframe trend supports entry. It is available but not currently wired into any pattern detector automatically.

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
| `pattern_start_idx` | int | Bar index where pattern starts |
| `pattern_end_idx` | int | Bar index where pattern ends |
| `candle_count` | int | Number of candles in pattern |
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
