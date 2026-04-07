"""
Cost policy enforcement.
Caps routing to models within a specified cost tier budget.
"""
from __future__ import annotations

from orchestrai.artifacts.schemas import CostTier
from orchestrai.providers.base import ModelCapability


def filter_by_cost(
    capabilities: list[ModelCapability],
    max_tier: CostTier,
) -> list[ModelCapability]:
    """Return only capabilities at or below the max cost tier."""
    allowed = _allowed_tiers(max_tier)
    return [c for c in capabilities if c.cost_tier in allowed]


def _allowed_tiers(max_tier: CostTier) -> set[CostTier]:
    order = [CostTier.CHEAP, CostTier.MEDIUM, CostTier.EXPENSIVE]
    idx = order.index(max_tier)
    return set(order[: idx + 1])
