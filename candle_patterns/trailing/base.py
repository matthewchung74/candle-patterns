"""
Trailing Stop Base Types
========================

Core dataclasses for trailing stop calculation.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any


@dataclass
class TrailingStopState:
    """
    Caller-managed state for trailing stop calculation.

    Pass this to calculate_trailing_stop() on each bar update.
    The function returns updated values but does NOT mutate this object.
    """
    entry_price: float
    original_stop: float
    current_stop: float
    direction: Literal["long", "short"]
    high_water_mark: float
    is_activated: bool = False
    risk_per_share: float = 0.0
    partial_taken: bool = False
    entry_idx: int = 0

    @classmethod
    def from_entry(
        cls,
        entry_price: float,
        stop_price: float,
        direction: Literal["long", "short"],
        entry_idx: int = 0,
    ) -> "TrailingStopState":
        """
        Factory method to create initial state from entry parameters.

        Args:
            entry_price: Price at which position was entered
            stop_price: Original stop loss price
            direction: "long" or "short"
            entry_idx: Index of entry bar in the bars DataFrame

        Returns:
            TrailingStopState initialized for tracking
        """
        risk = abs(entry_price - stop_price)
        return cls(
            entry_price=entry_price,
            original_stop=stop_price,
            current_stop=stop_price,
            direction=direction,
            high_water_mark=entry_price,
            is_activated=False,
            risk_per_share=risk,
            partial_taken=False,
            entry_idx=entry_idx,
        )


@dataclass
class TrailingStopConfig:
    """
    Configuration for trailing stop calculation.

    Attributes:
        strategy: Strategy name ("swing_low" or "atr")
        activation_r: R-multiple to activate trailing (default 1.0)
        activate_on_partial: Activate when partial taken (default True)
        min_bars_after_entry: Bars to wait before trailing (default 2)
        never_loosen_stop: Ratchet behavior - stop only tightens (default True)
        current_spread: Current bid-ask spread for buffer calc (default 0.01)
        params: Strategy-specific parameters
    """
    strategy: str = "swing_low"
    activation_r: float = 1.0
    activate_on_partial: bool = True
    min_bars_after_entry: int = 2
    never_loosen_stop: bool = True
    current_spread: float = 0.01
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrailingStopResult:
    """
    Result of trailing stop calculation.

    Attributes:
        active: Whether trailing stop is active
        new_stop: New stop price (may be same as original if not trailing yet)
        original_stop: Original stop for reference
        high_water_mark: Highest high since entry (longs) / lowest low (shorts)
        current_r_multiple: Current profit in R multiples
        reason: Why trailing is/isn't active
        strategy_name: Name of the strategy used
        just_activated: True if trailing just activated on this bar
        stop_moved: True if stop moved from previous level
    """
    active: bool
    new_stop: float
    original_stop: float
    high_water_mark: float
    current_r_multiple: float
    reason: str
    strategy_name: str = "swing_low"
    just_activated: bool = False
    stop_moved: bool = False

    @property
    def is_trailing(self) -> bool:
        """True if stop has moved from original."""
        return self.new_stop != self.original_stop
