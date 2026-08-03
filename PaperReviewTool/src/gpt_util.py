"""
Serialize OpenAI Responses API metadata (usage, token cache breakdown, response ids).

Used by :mod:`gpt_extractor` when logging each extraction run; safe to import elsewhere.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "token_details_to_dict",
    "cached_input_tokens_from_usage",
    "usage_from_response",
    "response_meta",
]


def token_details_to_dict(obj: Any) -> dict[str, Any]:
    """Normalize SDK token-detail objects (Pydantic models or plain objects) to a JSON-friendly dict."""
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    if isinstance(obj, dict):
        return dict(obj)
    out: dict[str, Any] = {}
    for key in (
        "cached_tokens",
        "text_tokens",
        "audio_tokens",
        "image_tokens",
        "reasoning_tokens",
    ):
        if hasattr(obj, key):
            val = getattr(obj, key)
            if val is not None:
                out[key] = val
    return out


def cached_input_tokens_from_usage(u: Any) -> int | None:
    """Read cached prompt/input token count from either Responses or legacy field names."""
    for details_attr in ("input_tokens_details", "prompt_tokens_details"):
        details = getattr(u, details_attr, None)
        if details is None:
            continue
        raw = getattr(details, "cached_tokens", None)
        if raw is None and isinstance(details, dict):
            raw = details.get("cached_tokens")
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
    return None


def usage_from_response(response: Any) -> dict[str, Any]:
    """Serialize token usage from a Responses API object, including prompt-cache breakdown when present."""
    u = getattr(response, "usage", None)
    if u is None:
        return {}

    out: dict[str, Any] = {}
    for name in ("input_tokens", "output_tokens", "total_tokens"):
        if hasattr(u, name):
            val = getattr(u, name)
            if val is not None:
                out[name] = val
    for name in ("prompt_tokens", "completion_tokens"):
        if hasattr(u, name):
            val = getattr(u, name)
            if val is not None:
                out.setdefault(name, val)

    if hasattr(u, "model_dump"):
        for key, val in u.model_dump().items():
            out.setdefault(key, val)

    for details_attr, out_key in (
        ("input_tokens_details", "input_tokens_details"),
        ("prompt_tokens_details", "prompt_tokens_details"),
    ):
        details = getattr(u, details_attr, None)
        if details is not None:
            out[out_key] = token_details_to_dict(details)

    cached = cached_input_tokens_from_usage(u)
    if cached is not None:
        out["cached_input_tokens"] = cached

    input_total = out.get("input_tokens")
    if input_total is None:
        input_total = out.get("prompt_tokens")
    if input_total is not None and cached is not None:
        try:
            out["non_cached_input_tokens"] = max(0, int(input_total) - int(cached))
        except (TypeError, ValueError):
            pass

    if hasattr(u, "output_tokens_details") and getattr(u, "output_tokens_details", None) is not None:
        out["output_tokens_details"] = token_details_to_dict(u.output_tokens_details)

    return out


def response_meta(response: Any) -> dict[str, Any]:
    """Identifiers and model name from the raw API response (when present)."""
    meta: dict[str, Any] = {}
    rid = getattr(response, "id", None)
    if rid is not None:
        meta["response_id"] = rid
    m = getattr(response, "model", None)
    if m is not None:
        meta["api_model"] = m
    status = getattr(response, "status", None)
    if status is not None:
        meta["status"] = status
    return meta
