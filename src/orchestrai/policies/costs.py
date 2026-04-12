"""
Per-model cost rates for budget tracking and enforcement.

Rates are approximate, based on publicly available pricing (Q1 2026).
Local / open-source models return 0.0 (not in the table).
"""
from __future__ import annotations

# (input_cost_per_token_usd, output_cost_per_token_usd)
_COST_MAP: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-opus-4-6":           (15.00 / 1_000_000, 75.00 / 1_000_000),
    "claude-sonnet-4-6":         ( 3.00 / 1_000_000, 15.00 / 1_000_000),
    "claude-haiku-4-5":          ( 0.80 / 1_000_000,  4.00 / 1_000_000),
    "claude-haiku-4-5-20251001": ( 0.80 / 1_000_000,  4.00 / 1_000_000),
    # OpenAI
    "gpt-4.1":                   ( 2.00 / 1_000_000,  8.00 / 1_000_000),
    "gpt-4o":                    ( 5.00 / 1_000_000, 20.00 / 1_000_000),
    "gpt-4o-mini":               ( 0.15 / 1_000_000,  0.60 / 1_000_000),
    "o3":                        (10.00 / 1_000_000, 40.00 / 1_000_000),
    "o4-mini":                   ( 1.10 / 1_000_000,  4.40 / 1_000_000),
    # Gemini
    "gemini-2.0-flash":          ( 0.10 / 1_000_000,  0.40 / 1_000_000),
    "gemini-2.0-flash-lite":     ( 0.075 / 1_000_000, 0.30 / 1_000_000),
    "gemini-2.0-pro":            ( 1.25 / 1_000_000,  5.00 / 1_000_000),
    "gemini-2.5-pro":            ( 1.25 / 1_000_000, 10.00 / 1_000_000),
}


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate cost in USD for a completion call.
    Returns 0.0 for unknown/local models.
    """
    rates = _COST_MAP.get(model_id)
    if rates is None:
        return 0.0
    return input_tokens * rates[0] + output_tokens * rates[1]
