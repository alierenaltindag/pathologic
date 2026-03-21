"""Utility helpers for search orchestration."""

from __future__ import annotations


def parse_model_pool(model_pool: str | None) -> list[str] | None:
    """Parse comma-separated model pool and normalize common aliases."""
    if model_pool is None:
        return None

    normalized_alias_map = {
        "xgnet": "xgboost",
        "xgb": "xgboost",
    }
    tokens = [item.strip().lower() for item in model_pool.split(",") if item.strip()]
    if not tokens:
        return None

    parsed: list[str] = []
    for token in tokens:
        alias = normalized_alias_map.get(token, token)
        if alias not in parsed:
            parsed.append(alias)
    return parsed
