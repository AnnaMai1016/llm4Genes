from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional, TypeVar, Union
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .pydantic_util import normalize_literal_value_for_path

T = TypeVar("T", bound=BaseModel)

# -----------------------------
# Patch for verification
# -----------------------------

JSONScalar = Union[str, int, float, bool]
JSONValue = Union[JSONScalar, List[JSONScalar]]

class Patch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["replace", "add", "append"]
    path: str = Field(
        ...,
        description=(
            "JSON Pointer path. For op='append', path must end at the list *field* "
            "(e.g. '/results/summary_bullets'), not an index. For 'replace'/'add' on a "
            "list element, include the index (e.g. '/results/quantitative_stats/0/value')."
        ),
    )
    old_value: Optional[JSONValue] = Field(
        default=None,
        description=(
            "Existing value (required for replace with non-null new_value, to guard "
            "against blind overwrites). Not used for append."
        ),
    )
    new_value: Optional[JSONValue] = Field(
        ...,
        description=(
            "For replace: new value or null to clear an optional field. "
            "For add: value to insert at key or list index (see JSON Pointer). "
            "For append: one item or a list of items to append to the list at path; must not be null."
        ),
    )
    evidence: str = Field(
        ...,
        description="Verbatim quote or near-verbatim paraphrase from source text.",
    )

    @model_validator(mode="after")
    def _patch_op_rules(self) -> "Patch":
        if self.op == "replace" and self.old_value is None and self.new_value is not None:
            raise ValueError(
                "Patch with op='replace' and a non-null new_value must supply old_value. "
                "Use new_value=null (with old_value set) to explicitly clear a field."
            )
        if self.op == "append" and self.new_value is None:
            raise ValueError(
                "Patch with op='append' must supply new_value (the item or list of items to append)."
            )
        return self


class PatchSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patches: List[Patch] = Field(default_factory=list)


class VerificationResult(BaseModel):
    """
    Result of verifying a single patch against source context.

    Attributes:
        is_valid: Whether the patch is valid given the evidence (judged by LLM).
        reasoning: Explanation of why the patch is or isn't valid (from LLM).
        context_quality: Tier derived from the fuzzy-match score **after** LLM
            verification — NOT assessed by the LLM itself.
            excellent: score >= 85, good: >= 65, fair: >= 45, poor: < 45.
    """
    is_valid: bool = Field(..., description="Whether the patch is valid given the evidence")
    reasoning: str = Field(..., description="Explanation of why the patch is or isn't valid")
    context_quality: Literal["excellent", "good", "fair", "poor"] = Field(
        ...,
        description=(
            "Fuzzy-match quality tier set by PaperExtractor after LLM verification "
            "(not produced by the LLM). "
            "excellent: score>=85, good: >=65, fair: >=45, poor: <45."
        ),
    )


class PatchVerification(BaseModel):
    """
    Verification details for a patch.

    Attributes:
        patch: The patch being verified.
        matched_contexts: Fuzzy-match hits from the source document (up to max_matches).
        verification: LLM verification result (None if LLM step was skipped).
        is_verified: Whether this patch passed all verification steps.
    """
    patch: Patch
    matched_contexts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of matched contexts with their scores (up to max_matches)"
    )
    verification: Optional[VerificationResult] = Field(
        default=None,
        description="LLM verification result",
    )
    is_verified: bool = Field(
        default=False,
        description="Whether this patch passed verification",
    )


class VerificationReport(BaseModel):
    """
    Complete verification report for a PatchSet.

    Attributes:
        total_patches: Total number of patches in the set.
        verified_patches: Number of patches that passed verification.
        rejected_patches: Number of patches that failed verification.
        patch_verifications: Per-patch detail.
    """
    total_patches: int
    verified_patches: int
    rejected_patches: int
    patch_verifications: List[PatchVerification]


class ExtractionFieldVerification(BaseModel):
    """Verification result for a single extracted field."""
    is_correct: bool = Field(..., description="Whether the extracted value is correct")
    reasoning: str = Field(..., description="Explanation of why it's correct or incorrect")
    suggested_correction: Optional[JSONValue] = Field(
        default=None,
        description="Suggested corrected value if incorrect, None if should be removed"
    )
    action: Literal["keep", "correct", "remove"] = Field(
        ...,
        description="Recommended action: keep (correct), correct (fix value), remove (delete field)"
    )


class FieldVerification(BaseModel):
    """Verification details for an extracted field."""
    path: str = Field(..., description="JSON pointer path to the field")
    current_value: JSONValue = Field(..., description="Currently extracted value")
    matched_contexts: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="List of matched contexts from source document"
    )
    verification: Optional[ExtractionFieldVerification] = Field(
        default=None,
        description="LLM verification result"
    )
    is_verified: bool = Field(default=False, description="Whether this field passed verification")


# -----------------------------
# Patch application
# -----------------------------

class PatchMismatchError(ValueError):
    """Raised when old_value does not match the current value at the patch path."""


def _pointer_segments(path: str) -> list[str]:
    """
    Parse a JSON Pointer (RFC 6901) into a list of path segments.

    Example: ``'/results/summary_bullets/0'`` → ``['results', 'summary_bullets', '0']``
    Unescapes ``~1`` → ``/`` and ``~0`` → ``~`` per spec.
    """
    if not path.startswith("/"):
        raise ValueError(f"JSON Pointer must start with '/': {path!r}")
    return [
        seg.replace("~1", "/").replace("~0", "~")
        for seg in path[1:].split("/")
    ]


def _get_node(data: Any, segments: list[str]) -> Any:
    """Traverse *data* (nested dict/list) following *segments*; return the node."""
    cur = data
    for seg in segments:
        if isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError) as exc:
                raise KeyError(f"List index error at segment {seg!r}: {exc}") from exc
        elif isinstance(cur, dict):
            if seg not in cur:
                raise KeyError(f"Key {seg!r} not found in dict")
            cur = cur[seg]
        else:
            raise KeyError(
                f"Cannot traverse into {type(cur).__name__} with segment {seg!r}"
            )
    return cur


def _apply_patch_to_dict(
    data: dict[str, Any],
    patch: Patch,
    *,
    strict_old_value: bool,
) -> None:
    """
    Apply *patch* to *data* (a plain dict produced by ``model_dump``) in-place.

    Args:
        data: Mutable dict representing the model.
        patch: The patch to apply.
        strict_old_value: If True and ``patch.old_value`` is not None, raise
            :exc:`PatchMismatchError` when the current value doesn't match.
            Ignored for ``op='append'`` (no old_value check).
    """
    segments = _pointer_segments(patch.path)
    if not segments:
        raise ValueError(f"Patch path resolves to empty segments: {patch.path!r}")

    parent = _get_node(data, segments[:-1])
    last = segments[-1]

    if patch.op == "append":
        if not isinstance(parent, dict):
            raise KeyError(
                f"append expects a dict parent for the final segment at {patch.path!r}"
            )
        cur = parent.get(last)
        if cur is None:
            cur = []
        if not isinstance(cur, list):
            raise ValueError(
                f"append requires a list at {patch.path!r}, got {type(cur).__name__!r}"
            )
        if patch.new_value is None:
            raise ValueError(f"append at {patch.path!r} requires non-null new_value")
        if isinstance(patch.new_value, list):
            parent[last] = [*cur, *patch.new_value]
        else:
            parent[last] = [*cur, patch.new_value]

    elif patch.op == "replace":
        # Validate old_value before overwriting.
        if strict_old_value and patch.old_value is not None:
            if isinstance(parent, list):
                try:
                    current = parent[int(last)]
                except (ValueError, IndexError) as exc:
                    raise KeyError(
                        f"List index {last!r} out of range at {patch.path!r}"
                    ) from exc
            elif isinstance(parent, dict):
                current = parent.get(last)
            else:
                raise KeyError(
                    f"Cannot read from {type(parent).__name__} at {patch.path!r}"
                )
            if current != patch.old_value:
                raise PatchMismatchError(
                    f"old_value mismatch at {patch.path!r}: "
                    f"expected {patch.old_value!r}, got {current!r}"
                )

        # Apply.
        if isinstance(parent, list):
            parent[int(last)] = patch.new_value
        elif isinstance(parent, dict):
            parent[last] = patch.new_value
        else:
            raise KeyError(f"Cannot replace in {type(parent).__name__} at {patch.path!r}")

    elif patch.op == "add":
        if isinstance(parent, list):
            if last == "-":
                parent.append(patch.new_value)
            else:
                try:
                    parent.insert(int(last), patch.new_value)
                except ValueError as exc:
                    raise KeyError(
                        f"Invalid list index {last!r} for add at {patch.path!r}"
                    ) from exc
        elif isinstance(parent, dict):
            parent[last] = patch.new_value
        else:
            raise KeyError(f"Cannot add into {type(parent).__name__} at {patch.path!r}")

    else:
        raise ValueError(f"Unknown patch op: {patch.op!r}")


def apply_patch(
    model: T,
    patch: Patch,
    *,
    strict_old_value: bool = True,
    literal_normalizer: Any | None = None,
) -> T:
    """
    Apply a single *patch* to *model* and return a new validated instance.

    The original *model* is **not mutated**.

    Args:
        model: Source Pydantic model instance.
        patch: Patch to apply.
        strict_old_value: When True (default), raise :exc:`PatchMismatchError`
            if ``patch.old_value`` doesn't match the current field value.
        literal_normalizer: Optional normalizer (e.g. ``FuzzyNormalizer``) used to
            snap ``new_value`` string(s) to schema ``Literal`` labels for the patch path.

    Returns:
        A new instance of the same model type with the patch applied.

    Raises:
        PatchMismatchError: If ``strict_old_value`` is True and old_value mismatches.
        KeyError: If the path cannot be traversed.
        ValidationError: If the patched dict fails schema validation.
    """
    schema = type(model)
    nv = normalize_literal_value_for_path(
        schema, patch.path, patch.new_value, literal_normalizer
    )
    effective = (
        patch if nv == patch.new_value else patch.model_copy(update={"new_value": nv})
    )
    data: dict[str, Any] = model.model_dump(mode="python")
    _apply_patch_to_dict(data, effective, strict_old_value=strict_old_value)
    return type(model).model_validate(data)


def apply_patches(
    model: T,
    patches: list[Patch],
    *,
    strict_old_value: bool = True,
    literal_normalizer: Any | None = None,
) -> T:
    """
    Apply a sequence of patches to *model* and return a new validated instance.

    Patches are applied **sequentially** to a single working dict, so each
    patch sees the result of all previous patches.  The original *model* is
    **not mutated**.

    Args:
        model: Source Pydantic model instance.
        patches: Ordered list of patches to apply.
        strict_old_value: Forwarded to each ``_apply_patch_to_dict`` call.
        literal_normalizer: Optional ``Literal`` label normalizer (see :func:`apply_patch`).

    Returns:
        A new instance of the same model type with all patches applied.
    """
    if not patches:
        return model.model_copy(deep=True)
    schema = type(model)
    data: dict[str, Any] = model.model_dump(mode="python")
    for patch in patches:
        nv = normalize_literal_value_for_path(
            schema, patch.path, patch.new_value, literal_normalizer
        )
        effective = (
            patch if nv == patch.new_value else patch.model_copy(update={"new_value": nv})
        )
        _apply_patch_to_dict(data, effective, strict_old_value=strict_old_value)
    return type(model).model_validate(data)
