"""
Maps canonical document type IDs to Azure Document Intelligence prebuilt model IDs.

Routing rationale:
  prebuilt-receipt  — receipt (unified POS type; covers grocery, restaurant, retail, cafe)
  prebuilt-invoice  — structured commercial documents (PO, utility bill, vendor quote)
  prebuilt-layout   — everything else (packing slip, expense report, catalog)
                      Layout returns text + tables with no semantic field mapping;
                      the LLM backfill stage handles field extraction for these types.
"""
from __future__ import annotations

_DI_MODEL_MAP: dict[str, str] = {
    # Current canonical
    "receipt":                   "prebuilt-receipt",
    "receipt_warranty_service":  "prebuilt-receipt",
    "bill_utility":              "prebuilt-invoice",
    "purchase_order":            "prebuilt-invoice",
    "quote_vendor":              "prebuilt-invoice",
    "packing_slip":              "prebuilt-layout",
    "report_expense":            "prebuilt-layout",
    "catalog_product_list":      "prebuilt-layout",
    # Legacy IDs (kept so old in-flight jobs routed here still work)
    "receipt_grocery":           "prebuilt-receipt",
    "receipt_restaurant":        "prebuilt-receipt",
    "receipt_retail":            "prebuilt-receipt",
}


def get_di_model(doc_type_id: str) -> str:
    """Return the Azure DI model ID for the given canonical document type."""
    return _DI_MODEL_MAP.get(doc_type_id, "prebuilt-layout")
