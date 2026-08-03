"""
Utilities for combining Pydantic model instances (e.g. incremental extraction passes).
"""

from __future__ import annotations

import json
from typing import Any, List, Literal, Optional, Protocol, Type, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel


class SupportsLiteralNormalize(Protocol):
    """Matches :class:`normalizers.LiteralNormalizer` implementations."""

    def normalize(self, value: str, allowed: list[str]) -> str | None: ...

T = TypeVar("T", bound=BaseModel)


def merge_results(a: Optional[T], b: T) -> T:
    """
    Deep-merge two instances of the same Pydantic model type.

    - If ``a`` is None, returns a deep copy of ``b``.
    - Nested ``BaseModel`` fields are merged recursively.
    - Scalar / optional fields: prefer ``b`` when it is not None, else ``a``.
    - Lists: concatenated with order preserved; duplicate strings are removed
      (first occurrence wins). Other list elements are concatenated as-is.

    ``b`` is treated as the newer pass when both sides have a value.
    """
    if a is None:
        return b.model_copy(deep=True)
    if type(a) is not type(b):
        raise TypeError(
            f"merge_results expects the same model type; got {type(a).__name__} and {type(b).__name__}."
        )
    merged = _merge_values(a, b, type(a))
    if not isinstance(merged, BaseModel):
        return b.model_copy(deep=True)
    return merged  # type: ignore[return-value]


def _merge_values(a: Any, b: Any, hint: Any) -> Any:
    if a is None:
        return b
    if b is None:
        return a

    if isinstance(a, BaseModel) and isinstance(b, BaseModel) and type(a) is type(b):
        data: dict[str, Any] = {}
        for name in type(a).model_fields:
            va = getattr(a, name)
            vb = getattr(b, name)
            ann = type(a).model_fields[name].annotation
            data[name] = _merge_values(va, vb, ann)
        return type(a).model_validate(data)

    if isinstance(a, list) and isinstance(b, list):
        return _merge_lists(a, b, hint)

    return _merge_scalarish(a, b)


def _merge_scalarish(a: Any, b: Any) -> Any:
    if b is not None:
        return b
    return a


def _unwrap_optional(hint: Any) -> Any:
    origin = get_origin(hint)
    if origin is Union:
        args = [x for x in get_args(hint) if x is not type(None)]
        return args[0] if len(args) == 1 else hint
    return hint


def _merge_lists(a: list[Any], b: list[Any], hint: Any) -> list[Any]:
    if not a:
        return list(b)
    if not b:
        return list(a)

    elem_hint = _unwrap_optional(hint)
    inner_args = get_args(elem_hint)
    inner_origin = get_origin(elem_hint)
    # ``list[str]`` / ``List[str]``: inner_origin is list, element type is inner_args[0]
    if inner_origin is list and inner_args:
        elem_type = inner_args[0]
        elem_type = _unwrap_optional(elem_type)
        if isinstance(elem_type, type) and issubclass(elem_type, BaseModel):
            # Deduplicate Pydantic model lists by their serialized representation
            # so that repeated extraction passes don't produce duplicate entries.
            seen_dumps: set[str] = set()
            deduped_models: list[Any] = []
            for item in [*a, *b]:
                key = json.dumps(item.model_dump(mode="python"), sort_keys=True, default=str)
                if key not in seen_dumps:
                    seen_dumps.add(key)
                    deduped_models.append(item)
            return deduped_models

    # Primitive lists: concatenate and de-dup strings
    out = [*a, *b]
    if out and all(isinstance(x, str) for x in out):
        seen: set[str] = set()
        deduped: list[str] = []
        for x in out:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        return deduped
    return out


def get_field_description(
    path: str,
    schema: type[BaseModel],
) -> Optional[str]:
    """
    Extract field description from schema given a JSON pointer path.

    Args:
        path: JSON pointer path (e.g., "/results/quantitative_stats/0/value")
        schema: Pydantic model to extract field description from
    Returns:
        Field description if found, None otherwise
    """
    try:
        # Remove leading slash and split path
        parts = path.lstrip('/').split('/')
        field_info = None
        for i, part in enumerate(parts):
            # Skip array indices
            if part.isdigit():
                continue
            # Get model fields
            if not hasattr(schema, 'model_fields'):
                return None
            fields = schema.model_fields
            if part not in fields:
                return None
            field_info = fields[part]
            # If this is the last part, get description
            if i == len(parts) - 1:
                if hasattr(field_info, 'description') and field_info.description:
                    return field_info.description
                return None
            # Navigate to nested model
            field_type = field_info.annotation
            # Handle Optional/Union types
            origin = get_origin(field_type)
            if origin is Union:
                # Get the non-None type
                args = get_args(field_type)
                field_type = next((arg for arg in args if arg is not type(None)), None)
                if field_type is None:
                    return None
            # Handle List types
            if get_origin(field_type) in (list, List):
                args = get_args(field_type)
                if args:
                    field_type = args[0]
            # Check if it's a Pydantic model
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                schema = field_type
            else:
                return None
        return None
    except Exception:
        return None


def _json_pointer_parts(path: str) -> list[str]:
    raw = path.strip()
    if not raw.startswith("/"):
        raw = "/" + raw
    return [seg.replace("~1", "/").replace("~0", "~") for seg in raw[1:].split("/")]


def _literal_string_choices(ann: Any) -> tuple[str, ...] | None:
    """If *ann* is ``Literal[str, ...]`` (after optional unwrap), return the string choices."""
    ann = _unwrap_optional(ann)
    origin = get_origin(ann)
    if origin is not Literal:
        return None
    args = get_args(ann)
    out: list[str] = []
    for a in args:
        if isinstance(a, str):
            out.append(a)
        else:
            return None
    return tuple(out)


def _walk_to_nested_model(
    root: type[BaseModel],
    field_names: list[str],
) -> type[BaseModel] | None:
    """Walk *field_names* on nested ``BaseModel`` types starting at *root*."""
    cur: Any = root
    for name in field_names:
        if not hasattr(cur, "model_fields"):
            return None
        mf = cur.model_fields.get(name)
        if mf is None:
            return None
        ann = _unwrap_optional(mf.annotation)
        origin = get_origin(ann)
        if origin in (list, List):
            args = get_args(ann)
            ann = _unwrap_optional(args[0]) if args else type(None)
        if not (isinstance(ann, type) and issubclass(ann, BaseModel)):
            return None
        cur = ann
    return cur


def literal_allowed_values_for_path(
    schema: type[BaseModel],
    path: str,
) -> tuple[str, ...] | None:
    """
    Return allowed string labels if the JSON Pointer *path* targets a
    ``Literal[str, ...]`` field (scalar) or an element of ``List[Literal[str, ...]]``.

    Paths follow the same convention as patches (leading ``/``, ``~0`` / ``~1`` escapes).
    """
    parts = _json_pointer_parts(path)
    if not parts:
        return None
    if parts[-1].isdigit():
        if len(parts) < 2:
            return None
        list_field = parts[-2]
        prefix = parts[:-2]
        parent = _walk_to_nested_model(schema, prefix)
        if parent is None:
            return None
        mf = parent.model_fields.get(list_field)
        if mf is None:
            return None
        ann = _unwrap_optional(mf.annotation)
        if get_origin(ann) not in (list, List):
            return None
        args = get_args(ann)
        if not args:
            return None
        inner = _unwrap_optional(args[0])
        return _literal_string_choices(inner)
    fname = parts[-1]
    prefix = parts[:-1]
    parent = _walk_to_nested_model(schema, prefix)
    if parent is None:
        return None
    mf = parent.model_fields.get(fname)
    if mf is None:
        return None
    ann = _unwrap_optional(mf.annotation)
    lit = _literal_string_choices(ann)
    if lit is not None:
        return lit
    origin = get_origin(ann)
    if origin in (list, List):
        args = get_args(ann)
        if args:
            inner = _unwrap_optional(args[0])
            return _literal_string_choices(inner)
    return None


def normalize_literal_value_for_path(
    schema: type[BaseModel],
    path: str,
    value: Any,
    normalizer: SupportsLiteralNormalize | None,
    *,
    drop_unmatched: bool = True,
) -> Any:
    """
    If *path* resolves to string ``Literal`` (or list thereof) on *schema*, map *value*
    through *normalizer* (e.g. :class:`normalizers.FuzzyNormalizer`).

    Non-string *value*, unknown paths, or ``normalizer is None`` leave *value* unchanged.
    When normalization returns ``None``:
      - default behavior keeps the original sub-value,
      - with ``drop_unmatched=True`` the value is dropped (``None`` for scalar,
        omitted element for list values).
    """
    if normalizer is None or value is None:
        return value
    allowed = literal_allowed_values_for_path(schema, path)
    if allowed is None:
        return value
    allowed_list = list(allowed)
    if isinstance(value, str):
        out = normalizer.normalize(value, allowed_list)
        if out is not None:
            return out
        return None if drop_unmatched else value
    if isinstance(value, list):
        normalized: list[Any] = []
        for x in value:
            if isinstance(x, str):
                o = normalizer.normalize(x, allowed_list)
                if o is not None:
                    normalized.append(o)
                elif not drop_unmatched:
                    normalized.append(x)
            else:
                normalized.append(x)
        return normalized
    return value
