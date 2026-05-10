"""
Azure Document Intelligence API client.

Wraps azure-ai-formrecognizer so the rest of the pipeline never imports it directly.
Raises DINotConfiguredError when credentials are absent — the orchestrator treats this
as a graceful no-op and routes straight to LLM backfill.

Required environment variables:
    AZURE_DI_ENDPOINT  — e.g. https://my-resource.cognitiveservices.azure.com/
    AZURE_DI_KEY       — API key (32-char hex string)
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


class DINotConfiguredError(Exception):
    """Raised when AZURE_DI_ENDPOINT or AZURE_DI_KEY env vars are not set."""


def analyze_document(model_id: str, image_bytes: bytes):
    """
    Submit image bytes to Azure Document Intelligence.

    Returns an AnalyzeResult object from azure-ai-formrecognizer.
    Raises:
        DINotConfiguredError — credentials missing
        Exception            — any SDK / network error (caller should catch)
    """
    endpoint = os.environ.get("AZURE_DI_ENDPOINT", "").strip()
    key = os.environ.get("AZURE_DI_KEY", "").strip()
    if not endpoint or not key:
        raise DINotConfiguredError(
            "AZURE_DI_ENDPOINT and AZURE_DI_KEY must be set to use Azure Document Intelligence"
        )

    # Lazy import so the package is only required when DI is actually configured.
    from azure.ai.formrecognizer import DocumentAnalysisClient
    from azure.core.credentials import AzureKeyCredential

    logger.info("Azure DI: model=%s, payload=%d bytes", model_id, len(image_bytes))
    client = DocumentAnalysisClient(endpoint, AzureKeyCredential(key))
    poller = client.begin_analyze_document(model_id, image_bytes)
    result = poller.result()

    doc_count = len(result.documents) if result.documents else 0
    logger.info("Azure DI: complete — %d document(s) detected", doc_count)
    return result
