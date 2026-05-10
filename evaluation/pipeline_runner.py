"""
Wrappers around individual pipeline stages for evaluation.

The main orchestrator (extraction_orchestrator.py) runs all five stages
atomically. Evaluation needs partial configurations (DI-only, DI+LLM) and
per-stage snapshots. This module exposes those by calling stages individually.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).parent.parent  # docpipeline-release/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from registry.field_registry import shape_output
from registry.di_router import get_di_model
from core.di_mapper import map_di_result
from core.azure_di_client import analyze_document, DINotConfiguredError
from validators.semantic_validator import semantic_validate, ValidationResult
from validators.validators import validate_and_normalize
from validators.llm_assist import backfill_and_validate

logger = logging.getLogger(__name__)


def _call_di_with_retry(model_id: str, image_bytes: bytes, max_retries: int = 3) -> Any:
    """Call Azure DI with exponential backoff on transient errors."""
    delays = [2, 4, 8]
    for attempt in range(max_retries):
        try:
            return analyze_document(model_id, image_bytes)
        except DINotConfiguredError:
            raise
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            wait = delays[attempt]
            logger.warning("DI attempt %d failed (%s), retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)


class StageSnapshots:
    """Holds field dicts captured after each pipeline stage."""
    def __init__(self) -> None:
        self.after_stage_1: dict = {}
        self.after_stage_2: dict = {}
        self.after_stage_3: dict = {}
        self.after_stage_4: dict = {}
        self.after_stage_5: dict = {}
        self.raw_text: str = ""
        self.field_confidence: dict[str, float] = {}
        self.validation_result: ValidationResult | None = None
        self.error: str | None = None


def run_all_stages(image_bytes: bytes, doc_type: str) -> StageSnapshots:
    """
    Run all five stages and capture snapshots after each.
    Returns StageSnapshots with intermediate field dicts and confidence scores.
    """
    snap = StageSnapshots()
    di_extracted: dict = {}

    # Stage 1 — Azure DI
    try:
        model_id = get_di_model(doc_type)
        di_result = _call_di_with_retry(model_id, image_bytes)
        di_extracted, snap.raw_text, _ = map_di_result(model_id, doc_type, di_result)
    except DINotConfiguredError:
        logger.warning("DI not configured — proceeding with empty Stage 1 result")
    except Exception as exc:
        logger.error("Stage 1 DI failed: %s", exc)
        snap.error = f"stage1:{exc}"

    snap.after_stage_1 = shape_output(doc_type, di_extracted)

    # Stage 2 — LLM backfill
    try:
        backfilled = backfill_and_validate(doc_type, snap.raw_text, dict(di_extracted))
    except Exception as exc:
        logger.error("Stage 2 LLM failed: %s", exc)
        backfilled = dict(di_extracted)

    snap.after_stage_2 = shape_output(doc_type, backfilled)

    # Stage 3 — Semantic validation
    try:
        sem_fields, val_result = semantic_validate(doc_type, backfilled, snap.raw_text)
        snap.validation_result = val_result
        snap.field_confidence = val_result.field_confidence
    except Exception as exc:
        logger.error("Stage 3 semantic validation failed: %s", exc)
        sem_fields = backfilled
        snap.field_confidence = {}

    snap.after_stage_3 = shape_output(doc_type, sem_fields)

    # Stage 4 — Monetary normalization
    try:
        normalized = validate_and_normalize(doc_type, sem_fields)
    except Exception as exc:
        logger.error("Stage 4 normalization failed: %s", exc)
        normalized = sem_fields

    snap.after_stage_4 = shape_output(doc_type, normalized)

    # Stage 5 — Schema shaping
    snap.after_stage_5 = shape_output(doc_type, normalized)

    return snap


def run_di_only(image_bytes: bytes, doc_type: str) -> tuple[str, dict]:
    """Stage 1 only. Returns (raw_text, shaped_fields)."""
    try:
        model_id = get_di_model(doc_type)
        di_result = _call_di_with_retry(model_id, image_bytes)
        di_extracted, raw_text, _ = map_di_result(model_id, doc_type, di_result)
    except DINotConfiguredError:
        logger.warning("DI not configured")
        return "", shape_output(doc_type, {})
    except Exception as exc:
        logger.error("DI-only failed: %s", exc)
        return "", shape_output(doc_type, {})

    return raw_text, shape_output(doc_type, di_extracted)


def run_di_llm(image_bytes: bytes, doc_type: str) -> tuple[str, dict]:
    """Stages 1–2 only. Returns (raw_text, shaped_fields)."""
    di_extracted: dict = {}
    raw_text = ""

    try:
        model_id = get_di_model(doc_type)
        di_result = _call_di_with_retry(model_id, image_bytes)
        di_extracted, raw_text, _ = map_di_result(model_id, doc_type, di_result)
    except DINotConfiguredError:
        logger.warning("DI not configured — LLM will extract from empty dict")
    except Exception as exc:
        logger.error("Stage 1 failed in DI+LLM run: %s", exc)

    try:
        backfilled = backfill_and_validate(doc_type, raw_text, dict(di_extracted))
    except Exception as exc:
        logger.error("Stage 2 LLM failed in DI+LLM run: %s", exc)
        backfilled = dict(di_extracted)

    return raw_text, shape_output(doc_type, backfilled)
