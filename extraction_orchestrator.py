"""
Five-stage document extraction pipeline.

Stages:
  1. Azure Document Intelligence — structured field extraction
  2. LLM backfill               — fill fields DI missed
  3. Semantic validation         — label-context checks, field disambiguation
  4. Monetary normalization      — normalize money strings, verify totals
  5. Schema shaping              — registry-keyed output with all fields present

Environment variables:
    AZURE_DI_ENDPOINT        Azure Document Intelligence endpoint URL
    AZURE_DI_KEY             Azure Document Intelligence API key
    AZURE_OPENAI_ENDPOINT    Azure OpenAI endpoint URL
    AZURE_OPENAI_KEY         Azure OpenAI API key
    AZURE_OPENAI_DEPLOYMENT  Deployment name (default: gpt-4o)
    AZURE_OPENAI_API_VERSION API version (default: 2024-12-01-preview)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).parent  # docpipeline-release/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from registry.field_registry import shape_output
from registry.di_router import get_di_model
from core.di_mapper import map_di_result
from core.azure_di_client import analyze_document, DINotConfiguredError
from validators.semantic_validator import semantic_validate
from validators.validators import validate_and_normalize
from validators.llm_assist import backfill_and_validate

logger = logging.getLogger(__name__)


def extract_fields_from_image(
    image_bytes: bytes,
    document_type: str,
) -> tuple[str, dict, dict]:
    """
    Full DI + LLM pipeline for a page image.

    Returns:
        (raw_text, extracted_fields, field_bboxes)
        raw_text         — document text extracted by DI
        extracted_fields — complete registry-shaped dict
        field_bboxes     — normalized polygon coords per field from DI
    """
    model_id = get_di_model(document_type)
    raw_text: str    = ""
    di_extracted: dict = {}
    di_bboxes: dict  = {}

    # Stage 1: Azure Document Intelligence
    try:
        di_result = analyze_document(model_id, image_bytes)
        di_extracted, raw_text, di_bboxes = map_di_result(model_id, document_type, di_result)
        logger.info("DI stage: model=%s extracted %d field(s)", model_id,
                    sum(1 for v in di_extracted.values() if v is not None))
    except DINotConfiguredError:
        logger.warning(
            "Azure DI not configured — proceeding with LLM-only extraction "
            "(set AZURE_DI_ENDPOINT + AZURE_DI_KEY to enable DI)"
        )
    except Exception as exc:
        logger.error("Azure DI failed (%s) — continuing with LLM backfill only", exc)

    # Stage 2: LLM backfill for missing fields
    backfilled = backfill_and_validate(document_type, raw_text, di_extracted)

    # Stage 3: Semantic validation
    sem_fields, sem_result = semantic_validate(document_type, backfilled, raw_text)
    if sem_result.validation_warnings:
        logger.info("Semantic validation [%s]: %d warning(s)",
                    document_type, len(sem_result.validation_warnings))

    # Stage 4: Monetary normalization
    validated = validate_and_normalize(document_type, sem_fields)

    # Stage 5: Schema shaping
    final = shape_output(document_type, validated)
    logger.info("Orchestrator: %d/%d field(s) populated [doc_type=%s]",
                sum(1 for v in final.values() if v), len(final), document_type)

    return raw_text, final, di_bboxes
