"""
Ground truth helpers for the enterprise corpus.

Provides:
  create_ground_truth_template(doc_type, doc_id) -> dict
  load_enterprise_ground_truth(gt_path) -> dict
  load_quote_ground_truth(gt_record) -> dict   (converts quote_dataset format)
  load_expense_ground_truth(gt_path) -> dict
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent  # docpipeline-release/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from registry.field_registry import get_type_def


def create_ground_truth_template(doc_type: str, doc_id: str) -> dict:
    """
    Generate a blank ground-truth template for a given doc_type and doc_id.
    Fields are None; items array is an empty list (or None if type has no items).
    """
    defn = get_type_def(doc_type)
    if not defn:
        raise ValueError(f"Unknown doc_type: {doc_type!r}")

    fields: dict = {k: None for k in defn["fields"]}
    template: dict = {
        "doc_id": doc_id,
        "doc_type": doc_type,
        "fields": fields,
        "annotator_notes": "",
    }
    if defn["items_key"]:
        template["items"] = []

    return template


def load_enterprise_ground_truth(gt_path: str | Path) -> dict:
    """
    Load a hand-annotated ground truth JSON and return a flat registry-keyed dict.
    Merges template["fields"] with template["items"] into a single flat dict.
    """
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    result = dict(gt.get("fields", {}))
    items_key = _items_key_for_type(gt.get("doc_type", "receipt"))
    if items_key and "items" in gt:
        result[items_key] = gt["items"]

    return result


def _items_key_for_type(doc_type: str) -> str | None:
    defn = get_type_def(doc_type)
    return defn["items_key"] if defn else None


# ── Quote dataset ground truth conversion ─────────────────────────────────────
# The existing quote_dataset/ground_truth.json uses field names that differ
# from the registry. This mapping converts them.

_QUOTE_GT_FIELD_MAP = {
    "vendor":       "vendor_name",
    "customer":     "buyer_name",
    "issue_date":   "quote_date",
    "tax":          "tax_amount",
    "actual_total": "total",
}

_QUOTE_ITEM_FIELD_MAP = {
    "name":       "description",
    "qty":        "quantity",
    "unit_price": "unit_price",
    "total":      "line_total",
    # "type" and "unit" have no direct registry equivalent — dropped
}


def load_quote_ground_truth(gt_record: dict) -> dict:
    """
    Convert one record from quote_dataset/ground_truth.json to registry schema.

    gt_record keys: file, quote_number, vendor, customer, issue_date, valid_until,
                    items, subtotal, discount, discount_pct, tax, actual_total,
                    displayed_total, error_injected, error_detail
    """
    result: dict = {}

    for gt_key, reg_key in _QUOTE_GT_FIELD_MAP.items():
        val = gt_record.get(gt_key)
        if val is not None:
            result[reg_key] = _normalize_money_or_str(reg_key, val)

    # Fields with identical names
    for k in ("quote_number", "valid_until", "subtotal", "discount"):
        val = gt_record.get(k)
        if val is not None:
            result[k] = _normalize_money_or_str(k, val)

    # Items
    items = []
    for item in gt_record.get("items", []):
        row = {}
        for gt_k, reg_k in _QUOTE_ITEM_FIELD_MAP.items():
            v = item.get(gt_k)
            if v is not None:
                row[reg_k] = _normalize_money_or_str(reg_k, v)
        items.append(row)
    if items:
        result["items"] = items

    return result


_MONETARY_REGISTRY_KEYS = frozenset({
    "subtotal", "tax_amount", "discount", "total", "amount_paid",
    "unit_price", "line_total",
})


def _normalize_money_or_str(key: str, val) -> str:
    if key in _MONETARY_REGISTRY_KEYS and val is not None:
        try:
            return f"{float(val):.2f}"
        except (ValueError, TypeError):
            pass
    return str(val) if val is not None else val
