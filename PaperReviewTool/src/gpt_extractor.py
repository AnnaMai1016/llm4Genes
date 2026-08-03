from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, TypeVar, Type

from pydantic import BaseModel
from openai import OpenAI

from .prompt import ExtractionPrompt, ValidationPrompt
from .prompt import build_user_prompt, llm_verify_user_prompt
from .pydantic_util import merge_results, get_field_description, normalize_literal_value_for_path
from .gpt_util import response_meta, usage_from_response
# from meta_stru import VERIFICATION_PREFIXES

from .patches import (
    Patch,
    PatchSet,
    PatchVerification,
    VerificationResult,
    PatchMismatchError,
    apply_patches,
)
from .sentence_match import extract_sentence_matches
from .normalizers import LiteralNormalizer

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Basenames for `save_logs_and_clear` (each file holds a JSON array, appended across saves).
_LOG_EXTRACTION_RUNS = "extraction_runs.json"
_LOG_EXTRACT_TURN_SNAPSHOTS = "extract_turn_snapshots.json"
_LOG_VERIFICATION_RUNS = "verification_runs.json"
_LOG_VERIFICATION_TURN_SNAPSHOTS = "verification_turn_snapshots.json"
_LOG_UNUSED_PATCHES = "unused_patch_set_patches.json"
_LOG_PATCH_VERIFICATION_RESULTS = "patch_verification_results.json"
_LOG_CURRENT_RESULT_SNAPSHOTS = "current_result_snapshots.json"
_LOG_DOCUMENT_TEXT_SNAPSHOTS = "document_text_snapshots.json"
_LOG_SAVE_SESSIONS = "save_sessions.json"
_LOG_UPDATE_RUNS = "update_runs.json"
_LOG_UPDATE_TURN_SNAPSHOTS = "update_turn_snapshots.json"
_LOG_PATCH_APPLICATION_RUNS = "patch_application_runs.json"

# Default basenames for :meth:`PaperExtractor.save_current_patch_artifacts` (single-file JSON snapshots).
_SNAPSHOT_CURRENT_RESULT = "current_result.json"
_SNAPSHOT_UNUSED_PATCHES = "unused_patch_set_patches.json"
_SNAPSHOT_PATCH_VERIFICATION_RESULTS = "patch_verification_results.json"


def _read_json_array(path: Path) -> list[Any]:
    if not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def _write_json_array(
    path: Path,
    items: list[Any],
    *,
    indent: Optional[int],
    json_default: Any = str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=indent, default=json_default)


def _append_json_array(
    path: Path,
    new_items: list[Any],
    *,
    indent: Optional[int],
    json_default: Any = str,
) -> None:
    merged = _read_json_array(path)
    merged.extend(new_items)
    _write_json_array(path, merged, indent=indent, json_default=json_default)


class PaperExtractor:
    def __init__(
        self,
        schema: Type[T],
        prompt: ExtractionPrompt,
        client: OpenAI,
        model: str = "gpt-5.4-mini",
        file_unique_id: Optional[str] = None,
        cache_time: str = "24h",
        log_dir: str | Path = './logs',
    ) -> None:
        # openai configuration
        self.client = client
        self.model = model
        self.file_unique_id = file_unique_id
        self.cache_time = cache_time
        self.log_dir = Path(log_dir).expanduser()

        # schema configuration
        self.schema = schema
        #: Domain-specific system prompt (role + grounding rules). See `prompt.ExtractionPrompt`.
        self.prompt = prompt

        # document state
        self.document_text: Optional[str] = None

        # result state
        self.current_result: Optional[T] = None
        self.unused_patch_set: PatchSet = PatchSet()
        #: Latest results from `verify_patches` (one `PatchVerification` per patch in `unused_patch_set`).
        self.patch_verification_results: list[PatchVerification] = []

        # extraction state
        self.extraction_runs: list[dict[str, Any]] = []
        self.extract_turn_snapshots: list[dict[str, Any]] = []
        self.verification_runs: list[dict[str, Any]] = []
        self.verification_turn_snapshots: list[dict[str, Any]] = []
        self.update_runs: list[dict[str, Any]] = []
        self.update_turn_snapshots: list[dict[str, Any]] = []
        self.patch_application_runs: list[dict[str, Any]] = []

        #: When set via :meth:`set_patch_artifacts_autosave`, each mutation of
        #: ``current_result``, ``unused_patch_set.patches``, or
        #: ``patch_verification_results`` overwrites the corresponding JSON file
        #: under this directory.
        if file_unique_id is not None:
            self._patch_artifacts_dir = self.log_dir / f'{file_unique_id}_temp_results'
        else:
            self._patch_artifacts_dir = self.log_dir / 'default_temp_results'
        self._patch_artifact_indent: Optional[int] = 2
        self._patch_artifact_result_basename: str = _SNAPSHOT_CURRENT_RESULT
        self._patch_artifact_unused_basename: str = _SNAPSHOT_UNUSED_PATCHES
        self._patch_artifact_verification_basename: str = _SNAPSHOT_PATCH_VERIFICATION_RESULTS

    def set_patch_artifacts_autosave(
        self,
        directory: str | Path | None = None,
        *,
        indent: Optional[int] = 2,
        result_basename: str = _SNAPSHOT_CURRENT_RESULT,
        unused_patches_basename: str = _SNAPSHOT_UNUSED_PATCHES,
        verification_basename: str = _SNAPSHOT_PATCH_VERIFICATION_RESULTS,
        write_initial: bool = True,
    ) -> None:
        """
        Enable or disable automatic overwrite of three local JSON files whenever
        the matching in-memory value changes.

        When enabling, optionally writes the current state immediately if
        ``write_initial`` is True (default).
        """
        if directory is not None:
            self._patch_artifacts_dir = Path(directory).expanduser()

        self._patch_artifact_indent = indent
        self._patch_artifact_result_basename = result_basename
        self._patch_artifact_unused_basename = unused_patches_basename
        self._patch_artifact_verification_basename = verification_basename
        if self._patch_artifacts_dir is not None and write_initial:
            # self.save_current_result_artifact(indent=indent)
            # self.save_unused_patch_set_artifact(indent=indent,)
            # self.save_patch_verification_results_artifact(indent=indent)
            self.save_current_patch_artifacts(indent=self._patch_artifact_indent)
            logger.info("set_patch_artifacts_autosave: wrote current_result, unused_patch_set, patch_verification_results")

    def set_document(self, document_text: str) -> None:
        self.document_text = document_text
        logger.info("set_document: document_text set")

    def extract_turn(
        self,
        NAME_FIELDS: list[list[str]] | None = None,
        verbosity: str = "medium",
    ) -> None:
        logger.info("Start extract_turn:")
        self._require_document()

        runs_before_turn = len(self.extraction_runs)
        if NAME_FIELDS is None or len(NAME_FIELDS) == 0:
            logger.info("- extract_turn: no focus area provided, extracting entire document")
            self.extract(verbosity=verbosity)
        else:
            for focus_area in NAME_FIELDS:
                prev = self.current_result
                logger.info(f"- extract_turn {len(self.extraction_runs)+1-runs_before_turn}/{len(NAME_FIELDS)}: extracting focus area: {focus_area}")
                tmp_result = self.extract(focus_area, verbosity=verbosity)
                self.current_result = merge_results(prev, tmp_result)
        if self.current_result is None:
            logger.error("extract_turn: extraction produced no result; check API response and document text.")
            raise RuntimeError("Extraction produced no result; check API response and document text.")

        api_calls_this_turn = len(self.extraction_runs) - runs_before_turn
        turn_index = len(self.extract_turn_snapshots) + 1
        merged_dump: dict[str, Any] = self.current_result.model_dump(
            mode="python",
            exclude_none=True,
        )
        self.extract_turn_snapshots.append(
            {
                "turn_index": turn_index,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "extract_turn",
                "verbosity": verbosity,
                "name_fields": NAME_FIELDS,
                "api_calls_this_turn": api_calls_this_turn,
                "merged_result": merged_dump,
            }
        )
        self.save_current_result(indent=self._patch_artifact_indent)
        logger.info("End extract_turn")

    def extract(
        self,
        focus_area: list[str] | None = None,
        verbosity: str = "medium",
    ) -> T:
        """Public single-pass extraction (same as internal `_extract`)."""
        self._require_document()
        return self._extract(focus_area=focus_area, verbosity=verbosity)

    def _extract(
        self,
        focus_area: list[str] | None = None,
        verbosity: str = "medium"
    ) -> None:
        response = self._response(
            focus_area,
            verbosity,
            verification=False
        )

        tmp_result = response.output_parsed
        if tmp_result is None:
            raise RuntimeError("API returned no parsed output; check text_format and response.")

        self.current_result = tmp_result

        run_index = len(self.extraction_runs) + 1
        parsed_dump: dict[str, Any] = tmp_result.model_dump(
            mode="python",
            exclude_none=True,
        )
        record: dict[str, Any] = {
            "run_index": run_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "extract",
            "requested_model": self.model,
            "verbosity": verbosity,
            "focus_area": focus_area,
            "prompt_cache_key": self.file_unique_id,
            "prompt_cache_retention": self.cache_time,
            "usage": usage_from_response(response),
            **response_meta(response),
            "parsed_result": parsed_dump,
        }
        self.extraction_runs.append(record)
        return self.current_result

    def _response(
        self,
        focus_area: list[str] | None = None,
        verbosity: str = "medium",
        verification: bool = False,
        updation: bool = False,
        previous_extraction: BaseModel | None = None,
    ) -> T:
        self._require_document()
        user_prompt = build_user_prompt(
            full_text=self.document_text,
            focus_area=focus_area,
            verification=verification,
            updation=updation,
            previous_extraction=previous_extraction,
        )

        response = self.client.responses.parse(
            model=self.model,
            prompt_cache_key=self.file_unique_id,
            prompt_cache_retention=self.cache_time,
            input=[
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": self.prompt.system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            text_format=self.schema,
            text={"verbosity": verbosity},
        )
        return response

    def verification(
        self,
        NAME_FIELDS: list[list[str]] | None = None,
        verbosity: str = "medium",
        clear_previous_verification: bool = True,
    ) -> None:
        """Ask the LLM to review the current extraction and return a minimal PatchSet.

        Unlike :meth:`update`, this method does **not** modify ``current_result``.
        Detected patches accumulate in ``unused_patch_set`` and can be inspected
        and validated by calling :meth:`verify_patches` afterwards.

        Use this when you want to audit the existing extraction without overwriting
        it — the caller decides which patches (if any) to apply.
        """
        logger.info("Start verification:")
        if clear_previous_verification:
            logger.info("- verification: clearing previous verification results")
            self.unused_patch_set = PatchSet()
            self.patch_verification_results.clear()

        self._require_document()
        self._require_response()

        runs_before_turn = len(self.verification_runs)

        if NAME_FIELDS is None or len(NAME_FIELDS) == 0:
            # If no focus area is provided, verify the entire extraction.
            logger.info("- verification: no focus area provided, verifying entire extraction")
            self._verification(verbosity=verbosity)
        else:
            for focus_area in NAME_FIELDS:
                # If focus area is provided, verify the specific focus area.
                logger.info(f"- verification {len(self.verification_runs)+1-runs_before_turn}/{len(NAME_FIELDS)}: verifying focus area: {focus_area}")
                self._verification(focus_area, verbosity=verbosity)
        if self.current_result is None:
            logger.error("verification: verification produced no result; check API response and document text.")
            raise RuntimeError("Verification produced no result; check API response and document text.")
        api_calls_this_turn = len(self.verification_runs) - runs_before_turn
        turn_index = len(self.verification_turn_snapshots) + 1
        unused_patch_set_dump: dict[str, Any] = self.unused_patch_set.model_dump(
            mode="python",
            exclude_none=True,
        )
        self.verification_turn_snapshots.append(
            {
                "turn_index": turn_index,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "verification_turn",
                "verbosity": verbosity,
                "name_fields": NAME_FIELDS,
                "api_calls_this_turn": api_calls_this_turn,
                "unused_patch_set": unused_patch_set_dump,
            }
        )
        self.save_unused_patch_set(indent=self._patch_artifact_indent)
        logger.info("End verification")

    def _verification(
        self, focus_area: list[str] | None = None,
        verbosity: str = "medium",
    ) -> None:
        response = self._response(
            focus_area,
            verbosity,
            verification=True,
            previous_extraction=self.current_result,
        )

        verification_result = response.output_parsed
        if verification_result is None:
            raise RuntimeError("API returned no parsed verification output.")

        verify_index = len(self.verification_runs) + 1
        verification_dump: dict[str, Any] = verification_result.model_dump(
            mode="python",
            exclude_none=True,
        )
        patch_list = getattr(getattr(verification_result, "patch_set", None), "patches", []) or []
        patch_count = len(patch_list)
        new_unused_patches: list[Patch] = []
        for patch in patch_list:
            path = getattr(patch, "path", None)
            if isinstance(path, str):
                if isinstance(patch, Patch):
                    new_unused_patches.append(patch)
                elif hasattr(patch, "model_dump"):
                    new_unused_patches.append(Patch.model_validate(patch.model_dump(mode="python")))
                elif isinstance(patch, dict):
                    new_unused_patches.append(Patch.model_validate(patch))
        
        # Accumulate patches across verification runs, deduplicating by (path, new_value).
        existing_keys = {
            (p.path, str(p.new_value)) for p in self.unused_patch_set.patches
        }
        deduped_new = [
            p for p in new_unused_patches
            if (p.path, str(p.new_value)) not in existing_keys
        ]
        self.unused_patch_set.patches.extend(deduped_new)
        # self._maybe_autosave_unused_patches()
        self.verification_runs.append(
            {
                "verify_index": verify_index,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "verification",
                "requested_model": self.model,
                "verbosity": verbosity,
                "focus_area": focus_area,
                "prompt_cache_key": self.file_unique_id,
                "prompt_cache_retention": self.cache_time,
                "usage": usage_from_response(response),
                **response_meta(response),
                "patch_count": patch_count,
                "unused_patch_count": len(new_unused_patches),
                "unused_patches": [p.model_dump(mode="python", exclude_none=True) for p in new_unused_patches],
                "unused_patches_total": len(self.unused_patch_set.patches),
                "verification_result": verification_dump,
            }
        )
        # return verification_result

    def update(
        self,
        NAME_FIELDS: list[list[str]] | None = None,
        verbosity: str = "medium",
    ) -> None:
        """Re-extract and merge results using prior extraction as context.

        Unlike :meth:`verification`, this method **overwrites** ``current_result``
        (via :func:`merge_results`) with the freshly extracted output.  The LLM
        sees the previous extraction and the original document and produces an
        updated extraction — it does not return a PatchSet.

        Use this when you want the model to produce a complete revised extraction
        rather than a minimal set of corrections.
        """
        logger.info("Start update:")
        
        self._require_document()
        self._require_response()

        runs_before_turn = len(self.update_runs)
        if NAME_FIELDS is None or len(NAME_FIELDS) == 0:
            logger.info("- update: no focus area provided, updating entire extraction")
            self._update(verbosity=verbosity)
        else:
            for focus_area in NAME_FIELDS:
                logger.info(f"- update {len(self.update_runs)+1-runs_before_turn}/{len(NAME_FIELDS)}: updating focus area: {focus_area}")
                # prev = self.current_result
                self._update(focus_area, verbosity=verbosity)
                # self.current_result = merge_results(prev, self.current_result)
        if self.current_result is None:
            logger.error("- update: update produced no result; check API response and document text.")
            raise RuntimeError("Update produced no result; check API response and document text.")
        api_calls_this_turn = len(self.update_runs) - runs_before_turn
        turn_index = len(self.update_turn_snapshots) + 1
        merged_dump: dict[str, Any] = self.current_result.model_dump(
            mode="python",
            exclude_none=True,
        )
        self.update_turn_snapshots.append(
            {
                "turn_index": turn_index,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "event": "update_turn",
                "verbosity": verbosity,
                "name_fields": NAME_FIELDS,
                "api_calls_this_turn": api_calls_this_turn,
                "merged_result": merged_dump,
            }
        )

        self.unused_patch_set = PatchSet()
        self.patch_verification_results.clear()
        self.save_current_patch_artifacts(indent=self._patch_artifact_indent)

        logger.info("End update")
        # return self.current_result

    def _update(
        self,
        focus_area: list[str] | None = None,
        verbosity: str = "medium",
    ) -> None:
        response = self._response(
            focus_area,
            verbosity,
            updation=True,
            previous_extraction=self.current_result,
        )

        tmp_result = response.output_parsed
        if tmp_result is None:
            raise RuntimeError("API returned no parsed output; check text_format and response.")

        self.current_result = tmp_result

        run_index = len(self.update_runs) + 1
        parsed_dump: dict[str, Any] = tmp_result.model_dump(
            mode="python",
            exclude_none=True,
        )
        record: dict[str, Any] = {
            "run_index": run_index,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": "update",
            "requested_model": self.model,
            "verbosity": verbosity,
            "focus_area": focus_area,
            "prompt_cache_key": self.file_unique_id,
            "prompt_cache_retention": self.cache_time,
            "usage": usage_from_response(response),
            **response_meta(response),
            "parsed_result": parsed_dump,
        }
        self.update_runs.append(record)

    def _require_response(self) -> None:
        if self.current_result is None:
            raise ValueError("Run extraction first; current_result is empty.")

    def _require_document(self) -> None:
        text = getattr(self, "document_text", None)
        if text is None or (isinstance(text, str) and not text.strip()):
            raise ValueError(
                "Document text is not set or is empty; call set_document(document_text) before extract."
            )

    def clear_extraction_runs(self) -> None:
        """Drop per-API-call records. Does not clear `extract_turn_snapshots` or `current_result`."""
        self.extraction_runs.clear()

    def clear_extract_turn_snapshots(self) -> None:
        """Drop per-`extract_turn` merged snapshots. Does not clear `extraction_runs` or `current_result`."""
        self.extract_turn_snapshots.clear()

    def clear_all_extraction_logs(self) -> None:
        """Clear extraction and verification logs. Does not clear `current_result`."""
        self.extraction_runs.clear()
        self.extract_turn_snapshots.clear()
        self.verification_runs.clear()
        self.verification_turn_snapshots.clear()
        self.update_runs.clear()
        self.update_turn_snapshots.clear()
        # self.unused_patch_set = PatchSet()
        # self.patch_verification_results.clear()
        self.patch_application_runs.clear()

    def clear_verification_runs(self) -> None:
        """Drop verification call records. Does not clear `current_result`."""
        self.verification_runs.clear()
        self.verification_turn_snapshots.clear()
        # self.unused_patch_set = PatchSet()
        # self.patch_verification_results.clear()
        self.patch_application_runs.clear()

    def verify_patches(
        self,
        min_match_score: float = 40.0,
        context_window_padding: int = 5,
        context_words: int = 50,
        max_matches: int = 3,
        verify_with_llm: bool = True,
        clear_unused_patch_set: bool = True,
        # *, 
        # remove_rejected_from_unused: bool = False,
    ) -> None:
        """
        Verify patches in `unused_patch_set` against the source document using fuzzy matching
        and optional LLM validation.

        Args:
            min_match_score: Minimum fuzzy match score (0-100) to consider evidence found (default: 40.0)
            context_window_padding: Extra words to include in search window for matching
            context_words: Number of words to show before/after match for context
            max_matches: Maximum number of fuzzy matches to provide to LLM (default: 3)
            verify_with_llm: Whether to use LLM for verification (if False, only uses fuzzy matching)
            remove_rejected_from_unused: If True, after verification remove from
                :attr:`unused_patch_set` every patch with ``is_verified=False``
                (matched by ``(path, str(new_value))``). Passed patches remain in
                ``unused_patch_set`` until :meth:`apply_verified_patches` or manual edits.

        """
        logger.info("Start verify_patches:")
        if len(self.unused_patch_set.patches) == 0:
            raise ValueError("No patches to verify. Run a verification turn first.")
        
        self._require_document()
        
        patch_verifications = []
        for patch in self.unused_patch_set.patches:
            verification = self._verify_single_patch(
                patch=patch,
                min_match_score=min_match_score,
                context_window_padding=context_window_padding,
                context_words=context_words,
                max_matches=max_matches,
                verify_with_llm=verify_with_llm
            )
            patch_verifications.append(verification)
        self.patch_verification_results = patch_verifications

        # if remove_rejected_from_unused:
        #     rejected_keys = {
        #         (pv.patch.path, str(pv.patch.new_value))
        #         for pv in patch_verifications
        #         if not pv.is_verified
        #     }
        #     before = len(self.unused_patch_set.patches)
        #     self.unused_patch_set.patches = [
        #         p for p in self.unused_patch_set.patches
        #         if (p.path, str(p.new_value)) not in rejected_keys
        #     ]
        #     removed = before - len(self.unused_patch_set.patches)
        #     if removed:
        #         logger.info(
        #             "verify_patches: removed %d rejected patch(es) from unused_patch_set.",
        #             removed,
        #         )
        if clear_unused_patch_set:
            self.unused_patch_set = PatchSet()
            self.save_unused_patch_set(indent=self._patch_artifact_indent)
        self.save_patch_verification_results(indent=self._patch_artifact_indent)
        logger.info("End verify_patches")

    def apply_verified_patches(
        self,
        *,
        strict_old_value: bool = True,
        # clear_applied: bool = True,
        literal_normalizer: LiteralNormalizer | None = None,
        clear_patch_verification_results: bool = True,
    ) -> None:
        """
        Apply all verified patches from :attr:`patch_verification_results` to
        :attr:`current_result`.

        Patches are applied **sequentially** in the order they appear in
        ``patch_verification_results``.  Only patches with ``is_verified=True``
        are applied; rejected patches are skipped.

        Args:
            strict_old_value: When True (default), each ``replace`` patch's
                ``old_value`` is compared to the current field value before
                writing.  A mismatch raises :exc:`~patches.PatchMismatchError`.
                ``add`` and ``append`` do not use this check (see
                :func:`patches.apply_patches`).  Set to False if the extraction
                changed since verification or you need to force-apply.
            literal_normalizer: If set (e.g. :class:`~normalizers.FuzzyNormalizer`),
                ``new_value`` is normalized for schema ``Literal``\ s at the patch
                path (``replace`` / ``add`` on a leaf, or ``append`` when the path
                is a ``List[Literal[...]]`` field).  See
                :func:`pydantic_util.normalize_literal_value_for_path`.
            clear_patch_verification_results: If True, clear
                :attr:`patch_verification_results` after this call completes
                successfully (including when there are no verified patches to apply).

        Returns:
            None

        Raises:
            ValueError: If no patch verification results are available.
            RuntimeError: If no extraction result exists yet.
            PatchMismatchError: If ``strict_old_value=True`` and a ``replace``
                patch's ``old_value`` doesn't match the current field value.
        """
        self._require_response()
        if not self.patch_verification_results:
            raise ValueError(
                "No patch verification results found. "
                "Run verify_patches() before calling apply_verified_patches()."
            )

        verified: list[Patch] = [
            pv.patch for pv in self.patch_verification_results if pv.is_verified
        ]
        skipped = len(self.patch_verification_results) - len(verified)

        if not verified:
            logger.info(
                "apply_verified_patches: no verified patches to apply "
                "(%d rejected/unverified).",
                skipped,
            )
            if clear_patch_verification_results:
                self.patch_verification_results.clear()
                self.save_patch_verification_results(indent=self._patch_artifact_indent)
            return

        schema = type(self.current_result)
        verified_effective: list[Patch] = []
        dropped_by_normalizer = 0
        for p in verified:
            nv = normalize_literal_value_for_path(
                schema,
                p.path,
                p.new_value,
                literal_normalizer,
                drop_unmatched=True,
            )
            # Skip patches that become empty after dropping unmatched Literal labels.
            if nv is None or (isinstance(nv, list) and len(nv) == 0):
                dropped_by_normalizer += 1
                logger.info(
                    "apply_verified_patches: dropped patch at %s because Literal normalization produced no valid value.",
                    p.path,
                )
                continue
            verified_effective.append(
                p if nv == p.new_value else p.model_copy(update={"new_value": nv})
            )

        if not verified_effective:
            logger.info(
                "apply_verified_patches: no patches left after Literal normalization (%d dropped).",
                dropped_by_normalizer,
            )
            if clear_patch_verification_results:
                self.patch_verification_results.clear()
                self.save_patch_verification_results(indent=self._patch_artifact_indent)
            return 

        try:
            updated = apply_patches(
                self.current_result,  # type: ignore[arg-type]
                verified_effective,
                strict_old_value=strict_old_value,
            )
        except PatchMismatchError:
            logger.error(
                "apply_verified_patches: old_value mismatch — no patches were applied. "
                "Re-run verify_patches() or set strict_old_value=False.",
                exc_info=True,
            )
            raise
        except Exception:
            logger.error(
                "apply_verified_patches: unexpected error — no patches were applied.",
                exc_info=True,
            )
            raise

        self.current_result = updated
        self.save_current_result(indent=self._patch_artifact_indent)

        run_record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "patches_applied": len(verified_effective),
            "patches_skipped": skipped,
            "patches_dropped_by_literal_normalizer": dropped_by_normalizer,
            "strict_old_value": strict_old_value,
            "literal_normalizer": (
                type(literal_normalizer).__name__ if literal_normalizer is not None else None
            ),
            "applied_patches": [
                p.model_dump(mode="python", exclude_none=True) for p in verified_effective
            ],
        }
        self.patch_application_runs.append(run_record)
        logger.info(
            "apply_verified_patches: applied %d patch(es), skipped %d, dropped %d by literal normalizer.",
            len(verified_effective),
            skipped,
            dropped_by_normalizer,
        )

        # if clear_applied:
        #     applied_keys = {(p.path, str(p.new_value)) for p in verified}
        #     self.unused_patch_set.patches = [
        #         p for p in self.unused_patch_set.patches
        #         if (p.path, str(p.new_value)) not in applied_keys
        #     ]
        #     self.save_unused_patch_set_artifact(indent=self._patch_artifact_indent)

        if clear_patch_verification_results:
            self.patch_verification_results.clear()
            self.save_patch_verification_results(indent=self._patch_artifact_indent)
        logger.info("End apply_verified_patches")

    def _verify_single_patch(
        self,
        patch: Patch,
        min_match_score: float,
        context_window_padding: int,
        context_words: int,
        max_matches: int,
        verify_with_llm: bool,
    ) -> PatchVerification:
        """Verify a single patch using fuzzy matching and optionally LLM"""

        # Step 1: Extract matching contexts from source document (get top N matches)
        try:
            matches = extract_sentence_matches(
                context=self.document_text,
                target_sentence=patch.evidence,
                window_size_padding=context_window_padding,
                score_thr=min_match_score,
                context_words=context_words,
                max_results=max_matches
            )
        except Exception as e:
            logger.warning("Fuzzy matching failed for patch path=%r: %s", patch.path, e, exc_info=True)
            return PatchVerification(
                patch=patch,
                matched_contexts=None,
                verification=None,
                is_verified=False
            )

        # If no matches found, patch fails verification
        if not matches:
            return PatchVerification(
                patch=patch,
                matched_contexts=None,
                verification=None,
                is_verified=False
            )
        
        matched_contexts = self._build_matched_contexts(matches)

        # Step 2: If LLM verification is disabled, only use fuzzy matching score
        if not verify_with_llm:
            # Simple heuristic: if best match score is high enough, consider it verified
            best_score = matches[0]['score']
            is_verified = best_score >= min_match_score
            return PatchVerification(
                patch=patch,
                matched_contexts=matched_contexts,
                verification=None,
                is_verified=is_verified
            )
        
        llm_verification = self._llm_verify_patch(
            patch=patch,
            matched_contexts=matched_contexts
        )
        return PatchVerification(
            patch=patch,
            matched_contexts=matched_contexts,
            verification=llm_verification,
            is_verified=llm_verification.is_valid if llm_verification else False
        )

    def _build_matched_contexts(
        self, matches: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build matched contexts with scores"""
        matched_contexts = []
        for match in matches:
            matched_contexts.append({
                "text": f"{match['prefix']}{match['highlight']}{match['suffix']}",
                "score": match['score'],
                "in_table": match.get('in_table', False),
                "in_equation": match.get('in_equation', False)
            })
        return matched_contexts

    def _llm_verify_patch(
        self,
        patch: Patch,
        matched_contexts: list[dict[str, Any]]
    ) -> VerificationResult:
        """Verify a patch using LLM"""

        # Determine context quality based on best match score
        # Note: This is informational - low scores don't automatically invalidate changes
        best_score = matched_contexts[0]['score'] if matched_contexts else 0
        if best_score >= 85:
            context_quality = "excellent"
        elif best_score >= 65:
            context_quality = "good"
        elif best_score >= 45:
            context_quality = "fair"
        else:
            context_quality = "poor"

        # Get field description from schema
        field_description = get_field_description(patch.path, self.schema)
        field_desc_section = ""
        if field_description:
            field_desc_section = f"FIELD DESCRIPTION (from schema):\n{field_description}\n"

        contexts_section = ""
        for i, ctx in enumerate(matched_contexts, 1):
            context_label = f"MATCHED CONTEXT #{i} (similarity score: {ctx['score']:.1f}% - for reference only)"
            contexts_section += f"{context_label}:\n---\n{ctx['text']}\n---\n"
        
        user_prompt = llm_verify_user_prompt(
            patch=patch,
            field_desc_section=field_desc_section,
            contexts_section=contexts_section
        )
        
        messages = [
            {"role": "system", "content": ValidationPrompt.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.responses.parse(
                model=self.model,
                input=messages,
                text_format=VerificationResult
            )

            verification = response.output_parsed
            if verification is None:
                return VerificationResult(
                    is_valid=False,
                    reasoning="API returned no parsed verification output.",
                    context_quality=context_quality,
                )
            verification.context_quality = context_quality
            return verification

        except Exception as e:
            # If LLM verification fails, return a conservative rejection
            return VerificationResult(
                is_valid=False,
                reasoning=f"LLM verification failed: {str(e)}",
                context_quality=context_quality,
            )

    def save_logs_and_clear(
        self,
        *,
        log_dir: str | Path = None,
        indent: Optional[int] = 2,
        include_document_text: bool = False,
        include_current_result: bool = True,
        clear_current_result: bool = False,
    ) -> None:
        """
        Persist logs and optional artifacts under ``log_dir`` using **separate JSON files**
        (each file is a JSON **array**). If a file already exists, new rows are **appended**
        (read → extend → write).

        Files written (only non-empty batches extend the corresponding file, except
        ``save_sessions.json`` which always records one summary row per call):

        - ``extraction_runs.json``
        - ``extract_turn_snapshots.json``
        - ``verification_runs.json``
        - ``verification_turn_snapshots.json``
        - ``update_runs.json``, ``update_turn_snapshots.json``
        - ``unused_patch_set_patches.json`` (flat list of patch dicts)
        - ``patch_verification_results.json``
        - ``current_result_snapshots.json`` (if ``include_current_result``) — each element
          ``{saved_at_utc, result}``
        - ``document_text_snapshots.json`` (if ``include_document_text``) —
          ``{saved_at_utc, document_text}``
        - ``save_sessions.json`` — one metadata row per save (counts, ids, schema name)

        Then clears in-memory logs via :meth:`clear_all_extraction_logs`. ``document_text``
        is not cleared. ``current_result`` is kept unless ``clear_current_result=True``.
        """
        if log_dir is None:
            log_dir = self._patch_artifacts_dir
        d = Path(log_dir).expanduser()
        d.mkdir(parents=True, exist_ok=True)
        saved_at = datetime.now(timezone.utc).isoformat()
        schema_name = getattr(self.schema, "__name__", str(self.schema))

        def append(fname: str, new_items: list[Any]) -> None:
            if not new_items:
                return
            _append_json_array(d / fname, new_items, indent=indent, json_default=str)

        # Collect all writes first; only clear in-memory state if all writes succeed.
        try:
            append(_LOG_EXTRACTION_RUNS, list(self.extraction_runs))
            append(_LOG_EXTRACT_TURN_SNAPSHOTS, list(self.extract_turn_snapshots))
            append(_LOG_VERIFICATION_RUNS, list(self.verification_runs))
            append(_LOG_VERIFICATION_TURN_SNAPSHOTS, list(self.verification_turn_snapshots))
            append(_LOG_UPDATE_RUNS, list(self.update_runs))
            append(_LOG_UPDATE_TURN_SNAPSHOTS, list(self.update_turn_snapshots))
            append(_LOG_PATCH_APPLICATION_RUNS, list(self.patch_application_runs))

            new_patches = [
                p.model_dump(mode="python", exclude_none=True)
                for p in self.unused_patch_set.patches
            ]
            append(_LOG_UNUSED_PATCHES, new_patches)

            pv = [
                p.model_dump(mode="python", exclude_none=True)
                for p in self.patch_verification_results
            ]
            append(_LOG_PATCH_VERIFICATION_RESULTS, pv)

            if include_current_result:
                result_dump = (
                    self.current_result.model_dump(mode="python", exclude_none=True)
                    if self.current_result is not None
                    else None
                )
                append(
                    _LOG_CURRENT_RESULT_SNAPSHOTS,
                    [{"saved_at_utc": saved_at, "result": result_dump}],
                )

            if include_document_text:
                append(
                    _LOG_DOCUMENT_TEXT_SNAPSHOTS,
                    [{"saved_at_utc": saved_at, "document_text": self.document_text}],
                )

            session_row: dict[str, Any] = {
                "saved_at_utc": saved_at,
                "model": self.model,
                "file_unique_id": self.file_unique_id,
                "cache_time": self.cache_time,
                "schema": schema_name,
                "counts": {
                    "extraction_runs": len(self.extraction_runs),
                    "extract_turn_snapshots": len(self.extract_turn_snapshots),
                    "verification_runs": len(self.verification_runs),
                    "verification_turn_snapshots": len(self.verification_turn_snapshots),
                    "update_runs": len(self.update_runs),
                    "update_turn_snapshots": len(self.update_turn_snapshots),
                    "unused_patches": len(self.unused_patch_set.patches),
                    "patch_verification_results": len(self.patch_verification_results),
                    "patch_application_runs": len(self.patch_application_runs),
                },
                "included_current_result": include_current_result,
                "included_document_text": include_document_text,
            }
            _append_json_array(
                d / _LOG_SAVE_SESSIONS,
                [session_row],
                indent=indent,
                json_default=str,
            )
        except Exception:
            # Leave in-memory state intact so the caller can retry or recover data.
            logger.error("save_logs_and_clear failed mid-write; in-memory state preserved.", exc_info=True)
            raise

        self.clear_all_extraction_logs()
        if clear_current_result:
            self.current_result = None


    @property
    def operation_log(self) -> list[dict[str, Any]]:
        """Alias for :attr:`extraction_runs` (backward-compatible name)."""
        return self.extraction_runs

    def save_current_result(
        self,
        *,
        indent: Optional[int] = 2,
        path: str | None = None,
    ) -> None:
        """
        Overwrite one JSON file with :attr:`current_result` (``{"saved_at_utc", "result"}``).

        """
        if path is None:
            path = self._patch_artifacts_dir / self._patch_artifact_result_basename
        else:
            path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "result": (
                self.current_result.model_dump(mode="python", exclude_none=False)
                if self.current_result is not None
                else None
            ),
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=indent, default=str)

    def save_unused_patch_set(
        self,
        *,
        indent: Optional[int] = 2,
    ) -> None:
        """
        Overwrite one JSON file with :attr:`unused_patch_set` patches
        (``{"saved_at_utc", "patches": [...]}``).

        """
        path = self._patch_artifacts_dir / self._patch_artifact_unused_basename
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "patches": [
                p.model_dump(mode="python", exclude_none=False)
                for p in self.unused_patch_set.patches
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=indent, default=str)
    
    def save_patch_verification_results(
        self,
        *,
        indent: Optional[int] = 2,
    ) -> None:
        """
        Overwrite one JSON file with :attr:`patch_verification_results`
        (``{"saved_at_utc", "patch_verification_results": [...]}``).

        """
        path = self._patch_artifacts_dir / self._patch_artifact_verification_basename
        path.parent.mkdir(parents=True, exist_ok=True)
        body: dict[str, Any] = {
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "patch_verification_results": [
                pv.model_dump(mode="python", exclude_none=False)
                for pv in self.patch_verification_results
            ],
        }
        with path.open("w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=indent, default=str)

    def save_current_patch_artifacts(
        self,
        *,
        indent: Optional[int] = 2,
    ) -> None:
        self.save_current_result(indent=indent)
        self.save_unused_patch_set(indent=indent)
        self.save_patch_verification_results(indent=indent)

    def get_stats(self) -> dict[str, Any]:
        return {
            "extraction_runs": len(self.extraction_runs),
            "extract_turn_snapshots": len(self.extract_turn_snapshots),
            "verification_runs": len(self.verification_runs),
            "verification_turn_snapshots": len(self.verification_turn_snapshots),
            "update_runs": len(self.update_runs),
            "update_turn_snapshots": len(self.update_turn_snapshots),
            "unused_patches": len(self.unused_patch_set.patches),
            "patch_verification_results": len(self.patch_verification_results),
            "patch_application_runs": len(self.patch_application_runs),
            "Number of unused patches": len(self.unused_patch_set.patches),
            "Number of patch verification results": len(self.patch_verification_results),
        }       