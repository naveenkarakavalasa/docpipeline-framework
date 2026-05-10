"""
LLM-assisted field extraction and backfill.

Calls Azure OpenAI to fill fields that Azure DI missed and to validate suspicious values.
Only invoked when AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY are set; skips silently otherwise.

Required environment variables:
    AZURE_OPENAI_ENDPOINT    — e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_KEY         — API key
    AZURE_OPENAI_DEPLOYMENT  — deployment name (e.g. gpt-4o, gpt-4o-mini)
    AZURE_OPENAI_API_VERSION — API version (default: 2024-12-01-preview)

Responsibilities:
  - Fill any registry field that is None / missing from the DI pass
  - Correct obviously wrong values (e.g. total < subtotal, date = "0000-00-00")
  - Normalize formats (ISO dates, plain numeric money strings)

NOT responsible for:
  - Replacing DI for fields it already extracted correctly
  - Running math / summing totals (validators.py handles that)
"""
from __future__ import annotations

import json
import logging
import os
import re

from registry.field_registry import get_type_def

logger = logging.getLogger(__name__)

# Optional per-doc-type context injected at the top of the extraction prompt.
_DOC_TYPE_CONTEXT: dict[str, str] = {
    "receipt": (
        "You are extracting structured information from a point-of-sale receipt. "
        "The receipt may come from grocery stores, supermarkets, retail stores, "
        "restaurants, cafes, pharmacies, or other POS systems. "
        "Use 'merchant_name', 'merchant_address', and 'merchant_phone' for the "
        "business name and contact details regardless of receipt subtype."
    ),
    "packing_slip": (
        "You are extracting structured information from a shipment confirmation or "
        "packing slip document (e.g. Amazon order confirmation, carrier label). "
        "CRITICAL date rules: only populate ship_date if the document explicitly "
        "labels it 'Ship Date', 'Shipped On', or 'Date Shipped'. Only populate "
        "delivery_date if labeled 'Delivery Date', 'Delivered On', or 'Expected "
        "Delivery'. Do NOT infer these from order dates or surrounding context. "
        "For order_date, extract from text like 'Your order of January 12, 2001'. "
        "For tracking_number, only extract clean values matching standard carrier "
        "formats: UPS=1Z+16 alphanumeric chars, FedEx=12/15/20 digits, "
        "USPS=20-22 digits. Reject OCR noise containing slashes, 'std-', or "
        "mixed fragments — leave blank if unsure. "
        "For the financial summary, extract: subtotal, shipping_cost "
        "(from 'Shipping & Handling'), discount (negative, from 'Promotional "
        "Certificate'), total_amount (from 'Order Total'), amount_paid and "
        "payment_method (from 'Paid via Visa/Mastercard/etc'), balance_due."
    ),
}

# Fields considered "important" — always passed to LLM if missing
_ALWAYS_CHECK: frozenset[str] = frozenset({
    "total", "total_amount", "subtotal", "tax_amount",
    "transaction_date", "bill_date", "po_date", "quote_date", "service_date",
    "merchant_name", "vendor_name", "restaurant_name", "provider_name",
    "service_provider_name",
    "po_number", "quote_number", "invoice_number", "service_order_number",
    "items", "expense_items",
})


def _missing_fields(extracted: dict, doc_type_id: str) -> list[str]:
    """Return registry fields that are absent or empty."""
    defn = get_type_def(doc_type_id)
    if not defn:
        return []

    missing: list[str] = []
    for field in defn["fields"]:
        if not extracted.get(field):
            missing.append(field)

    items_key = defn.get("items_key")
    if items_key and not extracted.get(items_key):
        missing.append(items_key)

    return missing


def _schema_hint(doc_type_id: str) -> str:
    defn = get_type_def(doc_type_id)
    if not defn:
        return ""
    lines = [f"Top-level fields: {', '.join(defn['fields'])}"]
    if defn["items_key"] and defn["items_fields"]:
        lines.append(f"  {defn['items_key']} (list of objects, each with): "
                     f"{', '.join(defn['items_fields'])}")
    return "\n".join(lines)


def _parse_json_from_response(text: str) -> dict:
    """Extract JSON from a response that may contain markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    m = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if m:
        text = m.group(1).strip()
    return json.loads(text)


def backfill_and_validate(
    doc_type_id: str,
    raw_text: str,
    extracted: dict,
    job_id: str | None = None,
    page_number: int | None = None,
) -> dict:
    """
    Call the LLM to fill missing/suspicious fields and return the merged result.

    - Only runs when AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY are set.
    - LLM output only fills None slots — does NOT overwrite existing DI values.
    - On any LLM error the original `extracted` dict is returned unchanged.
    """
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    api_key = os.environ.get("AZURE_OPENAI_KEY", "").strip()
    if not endpoint or not api_key:
        logger.warning(
            "LLM assist SKIPPED — AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_KEY not set. "
            "Missing fields will NOT be backfilled. Set these env vars to enable LLM extraction."
        )
        return extracted

    if not raw_text.strip():
        logger.warning("LLM assist SKIPPED — no document text available (DI returned empty content)")
        return extracted

    missing = _missing_fields(extracted, doc_type_id)
    if not missing:
        logger.info("LLM assist: all important fields present — skipping")
        return extracted

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
    schema = _schema_hint(doc_type_id)
    current_json = json.dumps(
        {k: v for k, v in extracted.items() if v is not None},
        indent=2,
    )

    context_line = _DOC_TYPE_CONTEXT.get(doc_type_id, "")
    prompt = f"""{context_line + chr(10) + chr(10) if context_line else ""}You are a document field extraction assistant.

Document type: {doc_type_id}

Schema:
{schema}

Already extracted fields (DO NOT change these unless clearly wrong):
{current_json}

Fields to fill (missing or empty): {", ".join(missing)}

Document text:
{raw_text[:8000]}

Instructions:
- Return ONLY a valid JSON object.
- Include only the fields listed in "Fields to fill".
- For monetary values: plain numeric string without symbols, e.g. "12.50".
- For dates: ISO 8601 format YYYY-MM-DD.
- For times: HH:MM or HH:MM:SS.
- For items / expense_items: a JSON array of objects using the sub-keys from the schema.
- Use null for any field you cannot find.
- Do not add fields outside the missing list.
- Include a top-level "_reasons" object mapping each filled field name to a short
  machine-readable reason string explaining the assignment (e.g. "found after label
  'Invoice Number:'", "inferred from subtotal minus tax"). Use null reasons for null
  values. Example: {{"_reasons": {{"merchant_name": "top-of-receipt header line"}}}}"""

    try:
        from openai import AzureOpenAI

        client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        logger.info("LLM assist: calling deployment=%s for %d missing field(s)", deployment, len(missing))
        response = client.chat.completions.create(
            model=deployment,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        response_text = response.choices[0].message.content
        llm_fields = _parse_json_from_response(response_text)

        # Extract per-field reasons before merge so they never enter the field dict
        reasons = llm_fields.pop("_reasons", {})
        if reasons and isinstance(reasons, dict):
            logger.debug("LLM field reasons [%s]: %s", doc_type_id, reasons)

        # Merge: LLM fills gaps only — never overwrites existing non-null DI values
        merged = dict(extracted)
        filled = 0
        for k, v in llm_fields.items():
            if v is not None and (k not in merged or merged[k] is None):
                merged[k] = v
                filled += 1

        logger.info("LLM assist: filled %d field(s)", filled)
        return merged

    except Exception as exc:
        logger.warning("LLM assist failed (%s) — using DI result as-is", exc)
        return extracted
