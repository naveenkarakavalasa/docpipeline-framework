"""
Field-level metric computation for evaluation.

Correctness rules:
- Monetary fields: match within ±0.10 after normalizing to float
- Date fields: exact string match after ISO normalization
- merchant_name / vendor_name: fuzzy match with threshold 0.85 (difflib)
- All other strings: exact match after lowercase + strip
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field


_MONETARY_FIELDS = frozenset({
    "subtotal", "tax_amount", "tip", "discount", "total", "total_amount",
    "discount_amount", "amount_paid", "labor_cost", "parts_cost",
    "delivery_charges", "supply_charges", "previous_balance",
    "payments_received", "adjustments", "balance_due", "shipping_cost",
    "shipping",
})

_DATE_FIELDS = frozenset({
    "transaction_date", "bill_date", "due_date", "po_date", "quote_date",
    "valid_until", "ship_date", "delivery_date", "order_date",
    "service_date", "report_period_start", "report_period_end",
    "submission_date", "effective_date", "expiry_date",
    "billing_period_start", "billing_period_end",
})

_FUZZY_FIELDS = frozenset({
    "merchant_name", "vendor_name", "provider_name", "buyer_name",
    "customer_name", "ship_to_name", "shipper_name",
})

_PHONE_FIELDS = frozenset({
    "merchant_phone", "customer_phone", "phone",
})

_ADDRESS_FIELDS = frozenset({
    "merchant_address", "service_address", "customer_address",
    "provider_address", "ship_to_address", "billing_address",
})

_FUZZY_THRESHOLD = 0.85
_ADDRESS_FUZZY_THRESHOLD = 0.70
_MONETARY_TOLERANCE = 0.10


def _to_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).replace(",", "").replace("$", "").replace("£", "").replace("€", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _normalize_str(v) -> str:
    return str(v).lower().strip() if v is not None else ""


def _normalize_phone(v) -> str:
    """Strip non-digits; drop leading country code 1 from 11-digit North American numbers."""
    digits = re.sub(r"\D", "", str(v))
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    return digits


def _normalize_address(v) -> str:
    """Collapse newlines to comma-space, lowercase, reduce whitespace."""
    s = re.sub(r"\s*\n\s*", ", ", str(v).strip())
    s = re.sub(r",\s*,+", ", ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.lower().strip()


def _normalize_iso_date(v: str) -> str:
    """Best-effort ISO 8601 normalization — returns original if unparseable."""
    import datetime
    s = str(v).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%B %d, %Y", "%b %d, %Y",
                "%d %B %Y", "%d %b %Y", "%Y/%m/%d"):
        try:
            return datetime.datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return s


def _fields_match(field_name: str, extracted, ground_truth) -> bool:
    if extracted is None or extracted == "" or extracted == []:
        return False
    if ground_truth is None or ground_truth == "" or ground_truth == []:
        return False

    if field_name in _MONETARY_FIELDS:
        a = _to_float(extracted)
        b = _to_float(ground_truth)
        if a is None or b is None:
            return _normalize_str(extracted) == _normalize_str(ground_truth)
        return abs(a - b) <= _MONETARY_TOLERANCE

    if field_name in _DATE_FIELDS:
        return _normalize_iso_date(str(extracted)) == _normalize_iso_date(str(ground_truth))

    if field_name in _FUZZY_FIELDS:
        ratio = difflib.SequenceMatcher(
            None,
            _normalize_str(extracted),
            _normalize_str(ground_truth),
        ).ratio()
        return ratio >= _FUZZY_THRESHOLD

    if field_name in _PHONE_FIELDS:
        return _normalize_phone(extracted) == _normalize_phone(ground_truth)

    if field_name in _ADDRESS_FIELDS:
        ratio = difflib.SequenceMatcher(
            None,
            _normalize_address(extracted),
            _normalize_address(ground_truth),
        ).ratio()
        return ratio >= _ADDRESS_FUZZY_THRESHOLD

    return _normalize_str(extracted) == _normalize_str(ground_truth)


def _value_in_raw_text(value, raw_text: str) -> bool:
    """Check if value (or a close substring) appears anywhere in raw OCR text."""
    if value is None or raw_text == "":
        return False
    s = str(value).strip()
    if len(s) < 2:
        return False
    return s.lower() in raw_text.lower()


@dataclass
class FieldMetrics:
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    mis_assignment_rate: float | None = None
    hallucination_rate: float | None = None
    blank_rate: float | None = None
    # Internal tallies
    n_gt_fields: int = 0
    n_extracted_populated: int = 0
    n_correct: int = 0
    n_misassigned: int = 0
    n_hallucinated: int = 0
    n_blank: int = 0


def compute_field_metrics(
    extracted: dict,
    ground_truth: dict,
    raw_text: str = "",
    skip_items: bool = True,
) -> FieldMetrics:
    """
    Compute field-level P/R/F1 and error rates.

    extracted      — shaped dict from pipeline
    ground_truth   — annotated dict (same schema)
    raw_text       — OCR text from pipeline for hallucination detection
    skip_items     — if True, skip the items/expense_items array (evaluated separately)
    """
    m = FieldMetrics()

    gt_keys = {k for k, v in ground_truth.items()
               if v is not None and v != "" and v != []
               and not (skip_items and k in ("items", "expense_items"))}
    ex_keys = {k for k, v in extracted.items()
               if v is not None and v != "" and v != []
               and not (skip_items and k in ("items", "expense_items"))}

    m.n_gt_fields = len(gt_keys)
    m.n_extracted_populated = len(ex_keys)

    for k in ex_keys:
        ex_val = extracted[k]
        gt_val = ground_truth.get(k)

        if gt_val is None or gt_val == "" or gt_val == []:
            # Extracted a value not in ground truth — check hallucination
            if not _value_in_raw_text(ex_val, raw_text):
                m.n_hallucinated += 1
            # Could also be a mis-assignment — don't double-count
        else:
            if _fields_match(k, ex_val, gt_val):
                m.n_correct += 1
            elif _value_in_raw_text(ex_val, raw_text):
                # Value is in the document but wrong field key
                m.n_misassigned += 1
            else:
                m.n_hallucinated += 1

    for k in gt_keys:
        if extracted.get(k) is None or extracted.get(k) == "":
            m.n_blank += 1

    m.precision = m.n_correct / m.n_extracted_populated if m.n_extracted_populated > 0 else None
    m.recall = m.n_correct / m.n_gt_fields if m.n_gt_fields > 0 else None
    if m.precision is not None and m.recall is not None and (m.precision + m.recall) > 0:
        m.f1 = 2 * m.precision * m.recall / (m.precision + m.recall)
    else:
        m.f1 = None

    m.mis_assignment_rate = m.n_misassigned / m.n_extracted_populated if m.n_extracted_populated > 0 else None
    m.hallucination_rate = m.n_hallucinated / m.n_extracted_populated if m.n_extracted_populated > 0 else None
    m.blank_rate = m.n_blank / m.n_gt_fields if m.n_gt_fields > 0 else None

    return m


def average_metrics(metrics_list: list[FieldMetrics]) -> FieldMetrics:
    """Average a list of per-document FieldMetrics into one aggregate."""
    if not metrics_list:
        return FieldMetrics()

    def _avg(vals):
        valid = [v for v in vals if v is not None]
        return sum(valid) / len(valid) if valid else None

    agg = FieldMetrics()
    agg.precision = _avg([m.precision for m in metrics_list])
    agg.recall = _avg([m.recall for m in metrics_list])
    agg.f1 = _avg([m.f1 for m in metrics_list])
    agg.mis_assignment_rate = _avg([m.mis_assignment_rate for m in metrics_list])
    agg.hallucination_rate = _avg([m.hallucination_rate for m in metrics_list])
    agg.blank_rate = _avg([m.blank_rate for m in metrics_list])
    return agg


def metrics_to_dict(m: FieldMetrics) -> dict:
    def _r(v):
        return round(v, 4) if v is not None else None

    return {
        "precision": _r(m.precision),
        "recall": _r(m.recall),
        "f1": _r(m.f1),
        "mis_assignment_rate": _r(m.mis_assignment_rate),
        "hallucination_rate": _r(m.hallucination_rate),
        "blank_rate": _r(m.blank_rate),
    }


def population_rate(fields: dict) -> float:
    """Fraction of non-null, non-empty top-level fields (excluding items arrays)."""
    top_level = {k: v for k, v in fields.items()
                 if k not in ("items", "expense_items")}
    if not top_level:
        return 0.0
    populated = sum(1 for v in top_level.values() if v is not None and v != "")
    return populated / len(top_level)


def compute_confidence_calibration(
    per_doc_results: list[dict],
) -> dict:
    """
    Compute precision within confidence score bins across all evaluated documents.

    Each entry in per_doc_results must have:
        "field_confidence": dict[str, float]
        "extracted": dict
        "ground_truth": dict

    Returns dict keyed by bin label.
    """
    bins = {
        "0.9_1.0": {"correct": 0, "total": 0},
        "0.7_0.9": {"correct": 0, "total": 0},
        "0.5_0.7": {"correct": 0, "total": 0},
        "0.0_0.5": {"correct": 0, "total": 0},
    }

    def _bin_for(conf: float) -> str:
        if conf >= 0.9:
            return "0.9_1.0"
        if conf >= 0.7:
            return "0.7_0.9"
        if conf >= 0.5:
            return "0.5_0.7"
        return "0.0_0.5"

    for doc in per_doc_results:
        extracted = doc["extracted"]
        gt = doc["ground_truth"]
        confidence = doc.get("field_confidence", {})

        for field_name, ex_val in extracted.items():
            if field_name in ("items", "expense_items") or ex_val is None or ex_val == "":
                continue
            conf = confidence.get(field_name, 1.0)  # implicit 1.0 for un-validated fields
            b = _bin_for(conf)
            bins[b]["total"] += 1
            gt_val = gt.get(field_name)
            if gt_val is not None and _fields_match(field_name, ex_val, gt_val):
                bins[b]["correct"] += 1

    result = {}
    for label, counts in bins.items():
        prec = counts["correct"] / counts["total"] if counts["total"] > 0 else None
        result[label] = {
            "n_fields": counts["total"],
            "precision": round(prec, 4) if prec is not None else None,
        }
    return result
