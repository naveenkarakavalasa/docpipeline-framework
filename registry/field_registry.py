"""
Single source of truth for all supported document types.

Used by:
- GET /api/document-types  (endpoint response)
- document_processor.py   (output shaping / default keys)
- frontend dropdown + field rendering order
"""

from __future__ import annotations
from typing import TypedDict


class ItemFieldDef(TypedDict):
    key: str


class DocTypeDef(TypedDict):
    id: str
    label: str
    fields: list[str]
    items_key: str | None        # "items" | "expense_items" | None
    items_fields: list[str]      # sub-keys inside each item row


REGISTRY: list[DocTypeDef] = [
    {
        "id": "receipt",
        "label": "Receipt",
        "fields": [
            "merchant_name", "merchant_address", "merchant_phone",
            "receipt_number", "transaction_date", "transaction_time",
            "subtotal", "tax_amount", "tip", "discount", "total",
            "payment_method", "store_number", "notes",
        ],
        "items_key": "items",
        "items_fields": ["description", "quantity", "unit_price", "line_total"],
    },
    {
        "id": "bill_utility",
        "label": "Utility Bill",
        "fields": [
            "provider_name", "provider_address", "provider_phone", "parent_company",
            "customer_name", "service_address", "customer_address",
            "account_number", "bill_number", "billing_period_start",
            "billing_period_end", "bill_date", "due_date",
            "service_type", "rate_plan", "usage_value", "usage_unit", "usage_estimated",
            "previous_balance", "payments_received", "adjustments",
            "delivery_charges", "supply_charges",
            "subtotal", "tax_amount", "total_amount", "balance_due",
            "currency", "notes",
        ],
        "items_key": None,
        "items_fields": [],
    },
    {
        "id": "purchase_order",
        "label": "Purchase Order",
        "fields": [
            "po_number", "po_date", "vendor_name", "vendor_address",
            "vendor_phone", "vendor_email", "buyer_name", "buyer_address",
            "buyer_email", "ship_to", "bill_to", "delivery_date",
            "payment_terms", "currency",
            "subtotal", "tax_amount", "shipping", "discount", "total", "notes",
        ],
        "items_key": "items",
        "items_fields": ["sku", "description", "quantity", "unit_price", "line_total"],
    },
    {
        "id": "packing_slip",
        "label": "Shipping / Packing Slip",
        "fields": [
            "packing_slip_number", "order_number", "order_date", "po_number",
            "ship_date", "delivery_date",
            "ship_to_name", "ship_to_address",
            "bill_to_name", "bill_to_address",
            "shipper_name", "shipper_address",
            "carrier_name", "tracking_number", "shipment_id",
            "shipping_method", "weight", "total_packages",
            "subtotal", "shipping_cost", "discount",
            "total_amount", "payment_method", "amount_paid", "balance_due",
            "notes",
        ],
        "items_key": "items",
        "items_fields": ["sku", "description", "quantity", "unit_of_measure"],
    },
    {
        "id": "quote_vendor",
        "label": "Quote / Estimate",
        "fields": [
            "quote_number", "quote_date", "valid_until",
            "vendor_name", "vendor_address", "vendor_phone", "vendor_email",
            "buyer_name", "buyer_address", "buyer_email",
            "currency", "subtotal", "tax_amount", "discount", "shipping",
            "total", "payment_terms", "notes",
        ],
        "items_key": "items",
        "items_fields": ["sku", "description", "quantity", "unit_price", "line_total"],
    },
    {
        "id": "receipt_warranty_service",
        "label": "Service Invoice",
        "fields": [
            "service_order_number", "invoice_number", "service_date",
            "service_provider_name", "service_provider_address", "service_provider_phone",
            "product_name", "product_model", "serial_number", "warranty_period",
            "service_description", "labor_cost", "parts_cost",
            "subtotal", "tax_amount", "total_amount",
            "payment_method", "currency", "notes",
        ],
        "items_key": "items",
        "items_fields": ["part_number", "description", "quantity", "unit_price", "line_total"],
    },
    {
        "id": "catalog_product_list",
        "label": "Product Catalog / Price List",
        "fields": [
            "catalog_name", "vendor_name", "catalog_version",
            "effective_date", "expiry_date", "currency", "notes",
        ],
        "items_key": "items",
        "items_fields": ["sku", "product_name", "category", "description",
                         "unit_of_measure", "price", "discount"],
    },
    {
        "id": "report_expense",
        "label": "Expense Report",
        "fields": [
            "report_id", "report_name", "report_period_start", "report_period_end",
            "submission_date", "employee_name", "department", "cost_center",
            "project_code", "total_amount", "currency", "approval_status",
            "approved_by", "notes",
        ],
        "items_key": "expense_items",
        "items_fields": [
            "expense_date", "merchant_name", "category", "description",
            "amount", "tax_amount", "payment_method", "receipt_reference",
        ],
    },
]

# ── Lookup helpers ─────────────────────────────────────────────────────────────

_BY_ID: dict[str, DocTypeDef] = {d["id"]: d for d in REGISTRY}

# Legacy document_type values -> canonical id
# Covers both old free-text aliases (pre-canonical-ID era) and
# old canonical IDs that have been merged into the unified receipt type.
LEGACY_MAP: dict[str, str] = {
    # Old free-text aliases
    "grocery receipt":    "receipt",
    "invoice/bill":       "receipt",
    "invoice":            "receipt",
    "receipt":            "receipt",
    "generic":            "receipt",
    "other":              "receipt",
    # Old canonical IDs merged into receipt
    "receipt_grocery":    "receipt",
    "receipt_restaurant": "receipt",
    "receipt_retail":     "receipt",
    # Old label aliases for renamed types
    "vendor quote / estimate": "quote_vendor",
    "warranty / service receipt": "receipt_warranty_service",
}

CANONICAL_IDS: frozenset[str] = frozenset(_BY_ID)


def get_type_def(doc_type_id: str) -> DocTypeDef | None:
    """Look up a type definition, falling back through LEGACY_MAP for old stored IDs."""
    if doc_type_id in _BY_ID:
        return _BY_ID[doc_type_id]
    # Old stored IDs (e.g. receipt_grocery from pre-merge jobs) resolve to canonical
    canonical = LEGACY_MAP.get(doc_type_id)
    return _BY_ID.get(canonical) if canonical else None


def normalize_doc_type(raw: str) -> tuple[str, bool]:
    """
    Return (canonical_id, is_legacy).
    Raises ValueError if unknown and not in legacy map.
    """
    raw = raw.strip().lower()
    if raw in CANONICAL_IDS:
        return raw, False
    if raw in LEGACY_MAP:
        return LEGACY_MAP[raw], True
    raise ValueError(f"Unknown document_type: {raw!r}")


def blank_fields(doc_type_id: str) -> dict:
    """Return a dict with every field key set to None / empty list."""
    defn = _BY_ID.get(doc_type_id)
    if not defn:
        # Resolve legacy IDs so shape_output still works for old job re-processing
        canonical = LEGACY_MAP.get(doc_type_id)
        defn = _BY_ID.get(canonical) if canonical else None
    if not defn:
        return {}
    out: dict = {k: None for k in defn["fields"]}
    if defn["items_key"]:
        out[defn["items_key"]] = []
    return out


def shape_output(doc_type_id: str, extracted: dict) -> dict:
    """
    Merge extracted values into a blank template so the output always
    contains every key defined in the registry (missing keys -> None / []).
    """
    base = blank_fields(doc_type_id)
    # Resolve the effective type def (handles legacy IDs)
    defn = get_type_def(doc_type_id)
    for k, v in extracted.items():
        if k in base or (defn and k == defn.get("items_key")):
            base[k] = v
    return base
