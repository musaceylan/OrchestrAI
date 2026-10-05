"""
Privacy policy enforcement.
Ensures tasks with sensitive data only route to approved privacy tiers.
"""
from __future__ import annotations

from orchestrai.artifacts.schemas import PrivacyLevel
from orchestrai.policies.eligibility import PRIVACY_ORDER
from orchestrai.providers.base import ModelCapability


def filter_by_privacy(
    capabilities: list[ModelCapability],
    required_level: PrivacyLevel,
) -> list[ModelCapability]:
    """
    Return only capabilities that satisfy the required privacy level.

    SECRET       → only local/on-prem models
    CONFIDENTIAL → local + enterprise-only APIs
    INTERNAL     → any provider without external telemetry
    PUBLIC       → all providers allowed
    """
    if required_level == PrivacyLevel.PUBLIC:
        return capabilities

    allowed_levels = _allowed_levels(required_level)
    return [c for c in capabilities if c.privacy_level in allowed_levels]


def _allowed_levels(required: PrivacyLevel) -> set[PrivacyLevel]:
    return set(PRIVACY_ORDER[PRIVACY_ORDER.index(required):])
