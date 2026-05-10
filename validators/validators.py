"""
Post-extraction monetary normalization.

Runs after semantic_validator → validate_and_normalize in the pipeline:
  1. Normalize all monetary strings to plain "12.50" format
  2. Log a warning when total ≠ subtotal + tax (tolerance: ±$1.00)
  3. Strip stray currency symbols / commas from numeric fields

Semantic correctness (label-context checks, field disambiguation, provider
normalization) is handled upstream by core.semantic_validator before this runs.
Does NOT modify non-numeric values — those are left for the frontend.
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_MONEY_KEYS: frozenset[str] = frozenset({
    "subtotal", "tax_amount", "total", "total_amount", "discount", "tip",
    "service_charge", "labor_cost", "parts_cost", "shipping",
    "previous_balance", "payments_received", "adjustments", "balance_due",
    # bill_utility charge components
    "delivery_charges", "supply_charges",
    # packing_slip financial summary
    "shipping_cost", "amount_paid",
})

_ITEM_MONEY_KEYS: frozenset[str] = frozenset({
    "unit_price", "line_total", "price", "amount", "tax_amount",
})


def _parse_number(s: str) -> float | None:
    """
    Parse a monetary string that may use . or , as thousands or decimal separators.

    Handles:
      "60.000"     → 60000.0  (Indonesian: dot = thousands when exactly 3 trailing digits)
      "1.000.000"  → 1000000.0 (multiple dots = all thousands)
      "100,000"    → 100000.0 (comma = thousands when exactly 3 trailing digits)
      "1,234.56"   → 1234.56  (US: comma = thousands, dot = decimal)
      "1.234,56"   → 1234.56  (European: dot = thousands, comma = decimal)
      "12.50"      → 12.5     (plain decimal)
      "-12.50"     → -12.5    (negative)
    """
    s = re.sub(r"[£$€¥₩\s]", "", s).strip()
    if not s:
        return None

    negative = s.startswith("-")
    if negative:
        s = s[1:]

    dot_count = s.count(".")
    comma_count = s.count(",")

    if dot_count == 0 and comma_count == 0:
        pass  # plain integer — parse as-is
    elif dot_count > 1 and comma_count == 0:
        # "1.000.000" — all dots are thousands separators
        s = s.replace(".", "")
    elif comma_count > 1 and dot_count == 0:
        # "1,000,000" — all commas are thousands separators
        s = s.replace(",", "")
    elif dot_count > 0 and comma_count > 0:
        # Both present — whichever comes last is the decimal separator
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")   # "1.234,56" → "1234.56"
        else:
            s = s.replace(",", "")                     # "1,234.56" → "1234.56"
    elif dot_count == 1:
        after = s.rsplit(".", 1)[1]
        if len(after) == 3:
            s = s.replace(".", "")   # "60.000" → "60000"
        # else: "12.50" — dot is decimal separator, leave as-is
    elif comma_count == 1:
        after = s.rsplit(",", 1)[1]
        if len(after) == 3:
            s = s.replace(",", "")   # "100,000" → "100000"
        else:
            s = s.replace(",", ".")  # "12,50" → "12.50"

    try:
        result = float(s)
        return -result if negative else result
    except ValueError:
        return None


def _to_float(v) -> float | None:
    if v is None:
        return None
    return _parse_number(str(v).strip())


def _normalize_money(v) -> str | None:
    f = _to_float(v)
    return f"{f:.2f}" if f is not None else None


def validate_and_normalize(doc_type_id: str, extracted: dict) -> dict:
    """
    Normalize monetary fields and validate total consistency.
    Returns a new dict (does not mutate the input).
    """
    out = dict(extracted)

    # Normalize top-level money fields
    for key in _MONEY_KEYS:
        if key in out and out[key] is not None:
            normalized = _normalize_money(out[key])
            if normalized is None:
                logger.warning("Could not normalize monetary field %s=%r", key, out[key])
            else:
                out[key] = normalized

    # Normalize item-level money fields
    for items_key in ("items", "expense_items"):
        items = out.get(items_key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            for k in _ITEM_MONEY_KEYS:
                if k in item and item[k] is not None:
                    item[k] = _normalize_money(item[k])

    # Sanity check: total ≈ subtotal + tax (tolerance ±$1.00)
    total = _to_float(out.get("total") or out.get("total_amount"))
    subtotal = _to_float(out.get("subtotal"))
    tax = _to_float(out.get("tax_amount"))

    if total is not None and subtotal is not None and tax is not None:
        expected = subtotal + tax
        if abs(total - expected) > 1.00:
            logger.warning(
                "Total mismatch [%s]: total=%.2f subtotal=%.2f tax=%.2f (expected %.2f)",
                doc_type_id, total, subtotal, tax, expected,
            )

    return out
