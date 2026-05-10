"""
Semantic field validator — document-type-aware cross-checking.

Core principles:
  1. Prefer blank over wrong.
  2. Only assign a field when label context clearly supports it.
  3. If two candidate values exist, choose the one with the strongest label match.
  4. Return warnings for ambiguous mappings instead of forcing a value.
  5. Preserve both raw extracted value and normalized value.

Input to semantic_validate():
  - doc_type_id:       str
  - extracted_fields:  dict  (from LLM backfill)
  - raw_text:          str   (full OCR text of the page)
  - source_snippets:   dict[str, str] | None  (field -> nearby OCR text, future)
  - candidate_values:  dict[str, list] | None (field -> list of DI alternatives, future)

Output: (normalized_fields, ValidationResult)
  - normalized_fields      — corrected field dict (pass to validate_and_normalize next)
  - ValidationResult       — rich diagnostics
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FieldValidation:
    raw_value: Any
    normalized_value: Any
    confidence: float                                     # 0.0 – 1.0
    status: Literal["valid", "warning", "invalid", "blank"]
    reason: str | None = None


@dataclass
class ValidationResult:
    normalized_fields: dict
    field_confidence: dict[str, float]
    validation_status: Literal["valid", "warning", "invalid"]
    validation_warnings: list[str]
    validation_errors: list[str]
    raw_to_normalized_mapping: dict[str, FieldValidation]


# ── Label-context scoring ────────────────────────────────────────────────────

def _label_precedes(text: str, value: str, label_patterns: list[str], window: int = 60) -> bool:
    """
    Return True only when a label *directly precedes* the value on the same line
    (within `window` non-newline chars). Stricter than _label_score — used for
    cross-field mis-assignment detection where proximity in either direction is
    too permissive.
    """
    if not value or not text:
        return False
    v = re.escape(str(value).strip())
    for pat in label_patterns:
        if re.search(rf"(?i){pat}[^\n]{{0,{window}}}{v}", text):
            return True
    return False


def _label_score(text: str, value: str, label_patterns: list[str], window: int = 200) -> float:
    """
    Return confidence (0.0–1.0) that `value` appears near one of `label_patterns`.
    1.0 = label precedes value within window chars.
    0.8 = label follows value within window chars.
    0.0 = no label context found.
    """
    if not value or not text:
        return 0.0
    v = re.escape(str(value).strip())
    for pat in label_patterns:
        if re.search(rf"(?i){pat}[\s\S]{{0,{window}}}{v}", text):
            return 1.0
        if re.search(rf"(?i){v}[\s\S]{{0,{window}}}{pat}", text):
            return 0.8
    return 0.0


def _value_in_text(text: str, value: str) -> bool:
    if not value or not text:
        return False
    return bool(re.search(re.escape(str(value).strip()), text, re.IGNORECASE))


# ── Monetary helper ──────────────────────────────────────────────────────────

def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        s = str(v).strip()
        negative = s.startswith("-")
        cleaned = re.sub(r"[^\d.]", "", s)
        if not cleaned:
            return None
        result = float(cleaned)
        return -result if negative else result
    except ValueError:
        return None


# ── Provider name normalization ──────────────────────────────────────────────

# (regex pattern, canonical name, parent company)
_PROVIDER_NORMALIZATIONS: list[tuple[str, str, str]] = [
    (r"delmarva\s+power",              "Delmarva Power",   "Exelon"),
    (r"peco\s+energy|peco\b",          "PECO Energy",      "Exelon"),
    (r"\bbge\b|baltimore\s+gas",       "BGE",              "Exelon"),
    (r"\bpepco\b",                     "Pepco",            "Exelon"),
    (r"dominion\s+energy",             "Dominion Energy",  ""),
    (r"duke\s+energy",                 "Duke Energy",      ""),
    (r"con\s*ed|consolidated\s+edison","Con Edison",       ""),
    (r"national\s+grid",               "National Grid",    ""),
    (r"xcel\s+energy",                 "Xcel Energy",      ""),
    (r"pg&?e|pacific\s+gas",           "PG&E",             ""),
    (r"florida\s+power|fpl\b",         "FPL",              "NextEra Energy"),
    (r"centerpoint\s+energy",          "CenterPoint Energy",""),
    (r"atmos\s+energy",                "Atmos Energy",     ""),
    (r"eversource",                    "Eversource",       ""),
    (r"ameren\b",                      "Ameren",           ""),
    (r"pseg|public\s+service\s+electric","PSE&G",          "PSEG"),
]


def _normalize_provider_name(raw: str) -> tuple[str, str]:
    """Return (canonical_name, parent_company). E.g. 'delmarva power EXELON' -> ('Delmarva Power', 'Exelon')."""
    if not raw:
        return raw, ""
    for pattern, canonical, parent in _PROVIDER_NORMALIZATIONS:
        if re.search(pattern, raw, re.IGNORECASE):
            return canonical, parent
    return raw.strip(), ""


# ── Base validator ────────────────────────────────────────────────────────────

class BaseDocValidator:
    doc_type_id: str = ""

    def __init__(
        self,
        extracted: dict,
        raw_text: str,
        source_snippets: dict | None = None,
        candidate_values: dict | None = None,
    ) -> None:
        self.extracted = dict(extracted)
        self.raw_text = raw_text or ""
        self.source_snippets = source_snippets or {}
        self.candidates = candidate_values or {}
        self.out: dict = dict(extracted)
        self.fv: dict[str, FieldValidation] = {}
        self.warnings: list[str] = []
        self.errors: list[str] = []

    # ── Field mutation helpers ────────────────────────────────────────────────

    def _set(
        self,
        field_name: str,
        value: Any,
        confidence: float,
        status: Literal["valid", "warning", "invalid", "blank"],
        reason: str | None = None,
    ) -> None:
        raw = self.extracted.get(field_name)
        self.out[field_name] = value
        self.fv[field_name] = FieldValidation(
            raw_value=raw,
            normalized_value=value,
            confidence=confidence,
            status=status,
            reason=reason,
        )

    def _blank(self, field_name: str, reason: str) -> None:
        """Clear a field and record why — prefer blank over wrong."""
        self._set(field_name, None, 0.0, "blank", reason)
        self.warnings.append(f"[{field_name}] blanked: {reason}")

    def _warn(self, field_name: str, reason: str) -> None:
        self.warnings.append(f"[{field_name}] {reason}")
        fv = self.fv.get(field_name)
        if fv and fv.status == "valid":
            fv.status = "warning"
            fv.reason = (fv.reason + "; " + reason) if fv.reason else reason

    def _error(self, field_name: str, reason: str) -> None:
        self.errors.append(f"[{field_name}] {reason}")
        fv = self.fv.get(field_name)
        if fv:
            fv.status = "invalid"

    # ── Finalize ─────────────────────────────────────────────────────────────

    def validate(self) -> ValidationResult:
        self._run()
        overall: Literal["valid", "warning", "invalid"] = "valid"
        if self.errors:
            overall = "invalid"
        elif self.warnings:
            overall = "warning"
        return ValidationResult(
            normalized_fields=self.out,
            field_confidence={k: v.confidence for k, v in self.fv.items()},
            validation_status=overall,
            validation_warnings=self.warnings,
            validation_errors=self.errors,
            raw_to_normalized_mapping=self.fv,
        )

    def _run(self) -> None:
        """Override in subclasses to add document-specific checks."""
        pass


# ── Receipt validator ─────────────────────────────────────────────────────────

# Trailing store-number patterns: "Taco Bell 040061", "McDonald's #1234"
_STORE_NUM_RE     = re.compile(r"^(.+?)\s+(?:store\s*#?\s*|#\s*)(\d{4,8})\s*$", re.IGNORECASE)
_TRAILING_NUM_RE  = re.compile(r"^(.+?)\s+(\d{4,8})\s*$")

# Receipt number label patterns tried in priority order.
# TC# (transaction confirmation) listed before TR# (transaction register) so that
# Walmart-style receipts with both prefer the longer confirmation number.
_RECEIPT_NUM_PATTERNS: list[re.Pattern] = [
    re.compile(r"tc#[:\s]+([A-Z0-9][A-Z0-9 \t\-]{1,44})",                          re.IGNORECASE),
    re.compile(r"receipt\s*(?:number|no\.?|#)[:\s]+([A-Z0-9][A-Z0-9 \t\-]{1,44})", re.IGNORECASE),
    re.compile(r"ticket\s*#?[:\s]+([A-Z0-9][A-Z0-9 \t\-]{1,44})",                  re.IGNORECASE),
    re.compile(r"check\s*#?[:\s]+([A-Z0-9][A-Z0-9 \t\-]{1,44})",                   re.IGNORECASE),
    re.compile(r"order\s*#?[:\s]+([A-Z0-9][A-Z0-9 \t\-]{1,44})",                   re.IGNORECASE),
    re.compile(r"tr#[:\s]+([A-Z0-9][A-Z0-9 \t\-]{1,44})",                          re.IGNORECASE),
]


class ReceiptValidator(BaseDocValidator):
    doc_type_id = "receipt"

    def _run(self) -> None:
        self._check_merchant_name()
        self._check_item_sum()
        self._check_receipt_number()
        # Delaware receipts legitimately have zero tax — never flag tax=0 as invalid

    def _check_merchant_name(self) -> None:
        name = self.out.get("merchant_name")
        if not name:
            return
        name_str = str(name).strip()

        # Try explicit "Store #NNNNN" pattern first, then bare trailing digits
        m = _STORE_NUM_RE.match(name_str) or _TRAILING_NUM_RE.match(name_str)
        if not m:
            return

        clean_name = m.group(1).strip()
        store_num  = m.group(2).strip()

        # Sanity: store codes are 4–6 digits; don't split phone/zip suffixes
        if len(store_num) > 6:
            return

        self._set(
            "merchant_name", clean_name, 0.9, "valid",
            f"store code {store_num!r} split from merchant name",
        )
        # Populate store_number (field_registry.py now includes it for receipt)
        self.out["store_number"] = store_num
        self.fv["store_number"] = FieldValidation(
            raw_value=self.extracted.get("store_number"),
            normalized_value=store_num,
            confidence=0.9,
            status="valid",
            reason="extracted from trailing code in merchant name",
        )

    def _check_item_sum(self) -> None:
        items    = self.out.get("items")
        subtotal = _to_float(self.out.get("subtotal"))
        if not items or subtotal is None:
            return
        calc = sum(
            _to_float(i.get("line_total")) or 0.0
            for i in items if isinstance(i, dict)
        )
        if abs(calc - subtotal) > 0.05:
            self._warn(
                "subtotal",
                f"item line sum {calc:.2f} differs from subtotal {subtotal:.2f}",
            )

    def _extract_receipt_number_from_text(self) -> str | None:
        """Try each label pattern in priority order; return first match that passes length check."""
        for pat in _RECEIPT_NUM_PATTERNS:
            m = pat.search(self.raw_text)
            if m:
                raw = m.group(1).strip()
                stripped = re.sub(r"[\s\-]", "", raw)
                if 3 <= len(stripped) <= 50:
                    return raw
        return None

    def _check_receipt_number(self) -> None:
        val = self.out.get("receipt_number")
        text_num = self._extract_receipt_number_from_text()

        if not text_num:
            return  # no label found — leave LLM value unchanged

        if not val:
            self._set(
                "receipt_number", text_num, 0.9, "valid",
                "recovered from receipt number label in raw OCR text",
            )
            return

        val_str = str(val).strip()
        if re.sub(r"[\s\-]", "", text_num) != re.sub(r"[\s\-]", "", val_str):
            self.warnings.append(
                f"[receipt_number] LLM value {val_str!r} overridden by "
                f"label-anchored value {text_num!r}",
            )
        # Always use the text-anchored value when a label is found — deterministic
        self._set(
            "receipt_number", text_num, 0.95, "valid",
            f"set from label-anchored scan (LLM had {val_str!r})",
        )


# ── Utility Bill validator ────────────────────────────────────────────────────

_UTIL_LABELS: dict[str, list[str]] = {
    "account_number": [
        r"account\s*(?:number|no\.?|#|num)",
        r"acct\.?\s*(?:no\.?|#|num)",
        r"your\s+account",
    ],
    "bill_number": [
        r"invoice\s*(?:number|no\.?|#)",
        r"bill\s*(?:number|no\.?|#)",
        r"(?:payment\s+coupon|remittance|return\s+(?:this\s+)?(?:stub|portion))",
        r"document\s*(?:number|no\.?|#)",
    ],
    "service_address": [
        r"service\s+(?:address|location|site)",
        r"premises\s+(?:address)?",
        r"delivery\s+(?:point|address)",
        r"service\s+for",
    ],
    "customer_address": [
        r"(?:billing|mailing)\s+address",
        r"bill\s+to",
        r"customer\s+address",
    ],
    "provider_address": [
        r"(?:remit|mail\s+(?:to|payment)|pay\s+to)",
        r"return\s+(?:this\s+)?(?:stub|portion)\s+with\s+(?:your\s+)?payment",
        r"send\s+payment",
        r"payment\s+(?:address|coupon)",
    ],
    "delivery_charges": [
        r"delivery\s+(?:charge|charges|service)",
        r"distribution\s+charge",
    ],
    "supply_charges": [
        r"supply\s+(?:charge|charges|service)",
        r"generation\s+charge",
        r"commodity\s+charge",
    ],
}

_ESTIMATED_RE = re.compile(
    r"(?:usage\s+)?estimated|estimate\s+(?:read|usage|amount)",
    re.IGNORECASE,
)

# PO Box pattern — flags addresses that belong to remittance, not service location
_PO_BOX_RE = re.compile(r"\bp\.?\s*o\.?\s+box\b|\bpo\s+box\b", re.IGNORECASE)


class UtilityBillValidator(BaseDocValidator):
    doc_type_id = "bill_utility"

    def _run(self) -> None:
        self._check_account_number()
        self._check_bill_number()
        self._check_addresses()
        self._check_usage_estimated()
        self._check_subtotal()
        self._check_charge_components()
        self._check_arithmetic()
        self._normalize_provider()

    # ── Account number helpers ────────────────────────────────────────────────

    _ACCT_NUM_RE = re.compile(
        r"(?i)account\s*(?:number|no\.?|#|num)[:\s]+([0-9][\d\s\-]{4,25}[0-9])"
    )

    def _extract_account_number_from_text(self) -> str | None:
        """
        Scan raw_text for an 'Account Number: XXXXX' pattern and return the
        normalized digits-only value.  Handles spaced formats like '5504 0071 056'.
        Returns None if not found or result is too short.
        """
        m = self._ACCT_NUM_RE.search(self.raw_text)
        if not m:
            return None
        raw = m.group(1).strip()
        normalized = re.sub(r"[\s\-]", "", raw)
        # Sanity: account numbers are typically 6–20 digits
        return normalized if 6 <= len(normalized) <= 20 else None

    def _check_account_number(self) -> None:
        val = self.out.get("account_number")

        # Always try to read the account number directly from labeled OCR text.
        # This serves as both a correction source and an independent verification.
        text_acct = self._extract_account_number_from_text()

        if not val:
            # DI/LLM missed it entirely — recover from raw text
            if text_acct:
                self._set("account_number", text_acct, 0.9, "valid",
                          "recovered from 'Account Number' label in raw OCR text")
            return

        val_str = str(val)
        # Normalize the extracted value for comparison (strip spaces/dashes)
        val_normalized = re.sub(r"[\s\-]", "", val_str)

        # If raw-text extraction found a different (better) value, prefer it
        if text_acct and text_acct != val_normalized:
            self._set("account_number", text_acct, 0.95, "valid",
                      f"replaced {val_str!r} with value found directly after "
                      "'Account Number' label in raw text")
            return

        # If the same value is in bill_number, the DI/LLM mis-mapped the invoice
        # number — blank and try to recover the real account number
        bill_str = re.sub(r"[\s\-]", "", str(self.out.get("bill_number") or ""))
        if val_normalized == bill_str and bill_str:
            if text_acct and text_acct != bill_str:
                self._set("account_number", text_acct, 0.9, "valid",
                          f"replaced duplicate-of-bill_number value with account "
                          "number found after 'Account Number' label in raw text")
            else:
                self._blank(
                    "account_number",
                    "value matches bill_number — invoice number mis-assigned to "
                    "account_number; no alternative found in raw text",
                )
            return

        # Check whether a bill/invoice label directly precedes this value (same line)
        if _label_precedes(self.raw_text, val_str, _UTIL_LABELS["bill_number"]):
            if text_acct and text_acct != val_normalized:
                self._set("account_number", text_acct, 0.9, "valid",
                          f"replaced value that followed invoice label with account "
                          "number found after 'Account Number' label in raw text")
            else:
                self._blank(
                    "account_number",
                    f"value {val_str!r} directly follows invoice/bill number label; "
                    "no alternative account number found in raw text",
                )
            return

        # Value looks legitimate — verify against account number label context
        acct_score = _label_score(self.raw_text, val_str, _UTIL_LABELS["account_number"])
        if acct_score >= 0.8:
            self._set("account_number", val, acct_score, "valid",
                      "found near account number label")
        elif acct_score > 0.0:
            self._set("account_number", val, acct_score, "warning",
                      "weak label context for account_number")
        else:
            self._set("account_number", val, 0.5, "warning",
                      "no 'Account Number' label found near this value — may be mis-assigned")
            self._warn("account_number",
                       "could not verify label context — cross-check against source document")

    def _check_bill_number(self) -> None:
        val = self.out.get("bill_number")
        if not val:
            return
        score = _label_score(self.raw_text, str(val), _UTIL_LABELS["bill_number"])
        if score >= 0.8:
            self._set("bill_number", val, score, "valid",
                      "found near invoice/bill number label or payment coupon section")
        elif score > 0.0:
            self._set("bill_number", val, 0.6, "warning",
                      "weak label context for bill_number")
        else:
            self._set("bill_number", val, 0.4, "warning",
                      "bill_number not found near expected labels — verify against Invoice/Bill Number")
            self._warn("bill_number",
                       "label context unconfirmed — check Invoice Number or Bill Number label")

    def _check_addresses(self) -> None:
        """Ensure service_address, customer_address, provider_address are not confused."""
        service_addr  = self.out.get("service_address")
        provider_addr = self.out.get("provider_address")

        # A PO Box in service_address is almost certainly the provider/remittance address
        if service_addr and _PO_BOX_RE.search(str(service_addr)):
            self._blank(
                "service_address",
                "contains PO Box — likely provider/remittance address, not service location",
            )
            if not provider_addr:
                self._set("provider_address", service_addr, 0.7, "warning",
                          "moved from service_address; appears to be remittance/provider address")
                self._warn("provider_address",
                           "value was in service_address — verify against original document")
            return  # re-read after potential swap
            # (provider_addr may have changed; re-read not needed since we already handled it)

        # If service_address is present, check its label context
        if service_addr:
            score = _label_score(self.raw_text, str(service_addr),
                                 _UTIL_LABELS["service_address"])
            if score < 0.5 and _value_in_text(self.raw_text, str(service_addr)):
                self._set("service_address", service_addr, 0.5, "warning",
                          "value found in text but label context is weak for service_address")
                self._warn("service_address",
                           "could be billing or provider address — check document layout")

        # If provider_address is a street address (not PO Box), warn if not near payment section
        if provider_addr and not _PO_BOX_RE.search(str(provider_addr)):
            score = _label_score(self.raw_text, str(provider_addr),
                                 _UTIL_LABELS["provider_address"])
            if score < 0.5:
                self._warn("provider_address",
                           "provider_address is a street address but not near a remittance/payment label")

    def _check_usage_estimated(self) -> None:
        """usage_estimated must be a boolean flag, never a usage quantity."""
        val = self.out.get("usage_estimated")

        if val is None:
            # Auto-detect from document text
            if _ESTIMATED_RE.search(self.raw_text):
                self._set("usage_estimated", True, 0.8, "valid",
                          "detected 'estimated' keyword in bill text")
            return

        # If it looks like a numeric usage value, it was mis-mapped
        if isinstance(val, (int, float)):
            self._blank("usage_estimated",
                        f"value {val!r} is numeric — looks like a usage quantity, not a boolean flag")
            return
        if isinstance(val, str) and re.match(r"^\d+\.?\d*\s*\w*$", val.strip()):
            self._blank("usage_estimated",
                        f"value {val!r} looks like a usage quantity (numeric with optional unit), not a flag")
            return

        # Normalize to Python bool
        truthy = str(val).lower() in ("true", "yes", "1", "estimated", "est")
        self._set("usage_estimated", truthy, 0.9, "valid",
                  f"normalized {val!r} to boolean {truthy}")

    def _check_subtotal(self) -> None:
        """Do not populate subtotal unless the document explicitly labels it."""
        if self.out.get("subtotal") is None:
            return
        if not re.search(r"(?i)\bsubtotal\b", self.raw_text):
            self._blank(
                "subtotal",
                "subtotal label not found in document text — cleared to avoid fabrication",
            )

    def _check_charge_components(self) -> None:
        """Validate delivery_charges and supply_charges label context."""
        for key in ("delivery_charges", "supply_charges"):
            val = self.out.get(key)
            if val is None:
                continue
            score = _label_score(self.raw_text, str(val), _UTIL_LABELS[key])
            if score >= 0.5:
                self._set(key, val, score, "valid", f"{key} found near expected label")
            else:
                self._set(key, val, 0.4, "warning",
                          f"weak label context for {key} — could not confirm from OCR text")

        # Guard: delivery + supply should not exceed total
        d     = _to_float(self.out.get("delivery_charges"))
        s     = _to_float(self.out.get("supply_charges"))
        total = _to_float(self.out.get("total_amount"))
        if d is not None and s is not None and total is not None:
            component_sum = d + s
            if component_sum > total + 0.02:
                self._warn(
                    "delivery_charges",
                    f"delivery ({d:.2f}) + supply ({s:.2f}) = {component_sum:.2f} "
                    f"exceeds total_amount ({total:.2f})",
                )

    def _check_arithmetic(self) -> None:
        """
        Validate: previous_balance - payments_received + adjustments + current_charges ≈ balance_due.
        Also validates delivery + supply ≈ total_amount when both components are present.
        """
        prev    = _to_float(self.out.get("previous_balance"))
        paid    = _to_float(self.out.get("payments_received"))
        adj     = _to_float(self.out.get("adjustments")) or 0.0
        balance = _to_float(self.out.get("balance_due"))
        total   = _to_float(self.out.get("total_amount"))

        if prev is not None and paid is not None and total is not None and balance is not None:
            expected = prev - paid + adj + total
            if abs(expected - balance) > 1.00:
                self._warn(
                    "balance_due",
                    f"arithmetic check: {prev:.2f} - {paid:.2f} + {adj:.2f} + {total:.2f} "
                    f"= {expected:.2f} but balance_due = {balance:.2f} "
                    f"(diff {abs(expected - balance):.2f})",
                )

    def _normalize_provider(self) -> None:
        raw = self.out.get("provider_name")
        if not raw:
            return
        canonical, parent = _normalize_provider_name(str(raw))
        if canonical != str(raw).strip():
            self._set("provider_name", canonical, 0.9, "valid",
                      f"normalized from {raw!r} to canonical name")

        if parent:
            existing = self.out.get("parent_company")
            # Always normalize case even when already set (e.g. "EXELON" → "Exelon")
            if not existing or str(existing).upper() == parent.upper():
                self.out["parent_company"] = parent
                self.fv["parent_company"] = FieldValidation(
                    raw_value=existing,
                    normalized_value=parent,
                    confidence=0.9,
                    status="valid",
                    reason=f"inferred from provider name {canonical!r}",
                )


# ── Service Invoice validator ─────────────────────────────────────────────────

_SVC_INV_LABELS: dict[str, list[str]] = {
    "invoice_number": [
        r"invoice\s*(?:number|no\.?|#)",
        r"inv\.?\s*(?:no\.?|#)",
    ],
    "service_order_number": [
        r"(?:service|work|repair)\s*(?:order|ticket)\s*(?:number|no\.?|#)?",
        r"work\s*order",
        r"wo\s*#",
        r"job\s*(?:number|no\.?|#)",
    ],
}


class ServiceInvoiceValidator(BaseDocValidator):
    doc_type_id = "receipt_warranty_service"

    def _run(self) -> None:
        self._check_invoice_vs_order()
        self._check_labor_parts()
        self._merge_multiline_descriptions()

    def _check_invoice_vs_order(self) -> None:
        inv = self.out.get("invoice_number")
        svc = self.out.get("service_order_number")

        if not inv or not svc or str(inv) != str(svc):
            return  # Values differ — no disambiguation needed

        # Same value in both fields: check which label appears in the document
        has_inv_label = bool(
            re.search(r"(?i)invoice\s*(?:number|no\.?|#)", self.raw_text)
        )
        has_svc_label = bool(
            re.search(
                r"(?i)(?:service|work|repair)\s*(?:order|ticket)\s*(?:number|no\.?|#)?|"
                r"work\s*order|wo\s*#|job\s*(?:no\.?|#)",
                self.raw_text,
            )
        )

        if has_inv_label and not has_svc_label:
            self._set("service_order_number", None, 0.8, "valid",
                      "both fields had same value; only 'Invoice Number' label found — "
                      "service_order_number cleared to avoid duplication")
        elif has_svc_label and not has_inv_label:
            self._set("invoice_number", None, 0.8, "valid",
                      "both fields had same value; only 'Service Order' label found — "
                      "invoice_number cleared to avoid duplication")
        else:
            # Both labels present, or neither — keep both but warn
            self._warn(
                "invoice_number",
                "invoice_number equals service_order_number — may be the same reference number",
            )

    def _check_labor_parts(self) -> None:
        """labor_cost and parts_cost are optional; only warn if their sum conflicts with subtotal."""
        labor    = _to_float(self.out.get("labor_cost"))
        parts    = _to_float(self.out.get("parts_cost"))
        subtotal = _to_float(self.out.get("subtotal"))
        if labor is not None and parts is not None and subtotal is not None:
            calc = labor + parts
            if abs(calc - subtotal) > 1.00:
                self._warn(
                    "subtotal",
                    f"labor_cost ({labor:.2f}) + parts_cost ({parts:.2f}) = {calc:.2f} "
                    f"differs from subtotal {subtotal:.2f}",
                )

    def _merge_multiline_descriptions(self) -> None:
        """Merge continuation lines: an item row with no amount but a description."""
        items = self.out.get("items")
        if not items or not isinstance(items, list):
            return

        merged: list[dict] = []
        i = 0
        while i < len(items):
            item = dict(items[i])
            while (
                i + 1 < len(items)
                and not items[i + 1].get("line_total")
                and not items[i + 1].get("unit_price")
                and items[i + 1].get("description")
            ):
                i += 1
                cont = items[i]["description"]
                item["description"] = (item.get("description") or "") + " " + cont
            merged.append(item)
            i += 1

        if merged != items:
            self.out["items"] = merged
            self.fv["items"] = FieldValidation(
                raw_value=items,
                normalized_value=merged,
                confidence=0.8,
                status="valid",
                reason="merged continuation lines in item descriptions",
            )


# ── Purchase Order validator ──────────────────────────────────────────────────

class PurchaseOrderValidator(BaseDocValidator):
    doc_type_id = "purchase_order"

    def _run(self) -> None:
        self._check_item_sum()

    def _check_item_sum(self) -> None:
        items    = self.out.get("items")
        subtotal = _to_float(self.out.get("subtotal"))
        if not items or subtotal is None:
            return
        calc = sum(
            _to_float(i.get("line_total")) or 0.0
            for i in items if isinstance(i, dict)
        )
        if abs(calc - subtotal) > 0.50:
            self._warn(
                "subtotal",
                f"item line total sum ({calc:.2f}) differs from subtotal ({subtotal:.2f})",
            )


# ── Quote / Estimate validator ────────────────────────────────────────────────

class QuoteEstimateValidator(BaseDocValidator):
    doc_type_id = "quote_vendor"

    def _run(self) -> None:
        self._check_validity_date()

    def _check_validity_date(self) -> None:
        valid_until = self.out.get("valid_until")
        quote_date  = self.out.get("quote_date")
        if not valid_until or not quote_date:
            return
        try:
            vd = date.fromisoformat(str(valid_until))
            qd = date.fromisoformat(str(quote_date))
            if vd < qd:
                self._warn(
                    "valid_until",
                    f"valid_until ({valid_until}) is before quote_date ({quote_date})",
                )
        except (ValueError, TypeError):
            pass


# ── Packing Slip / Shipment validator ────────────────────────────────────────

# ── Date label patterns ───────────────────────────────────────────────────────
_SHIP_DATE_LABELS: list[str] = [
    r"ship\s*date",
    r"shipped?\s+on",
    r"date\s+shipped",
    r"date\s+of\s+shipment",
]
_DELIVERY_DATE_LABELS: list[str] = [
    r"delivery\s+date",
    r"delivered\s+on",
    r"expected\s+delivery",
    r"est(?:imated)?\s+delivery",
    r"deliver\s+by",
]

# ── Order date extraction ─────────────────────────────────────────────────────
_ORDER_DATE_RES: list[re.Pattern] = [
    re.compile(r"(?i)your\s+order\s+(?:of|placed|on|from)?\s+([A-Za-z]+\s+\d{1,2},?\s*\d{4})"),
    re.compile(r"(?i)order\s+(?:date|placed)[:\s]+([A-Za-z]+\s+\d{1,2},?\s*\d{4})"),
    re.compile(r"(?i)order\s+(?:date|placed)[:\s]+(\d{4}-\d{2}-\d{2})"),
    re.compile(r"(?i)order\s+(?:date|placed)[:\s]+(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})"),
]

_MONTH_NAMES: dict[str, int] = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "oct": 10, "nov": 11, "dec": 12,
}


def _parse_named_date(text: str) -> str | None:
    """Parse 'Month DD[,] YYYY' → ISO 8601, or return None."""
    m = re.search(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s*(\d{4})\b", text)
    if not m:
        return None
    month_num = _MONTH_NAMES.get(m.group(1).lower())
    if not month_num:
        return None
    try:
        return date(int(m.group(3)), month_num, int(m.group(2))).isoformat()
    except ValueError:
        return None


def _parse_numeric_date(text: str) -> str | None:
    """Parse MM/DD/YY or MM-DD-YYYY → ISO 8601, or return None."""
    m = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", text)
    if not m:
        return None
    try:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        return date(year, month, day).isoformat()
    except ValueError:
        return None


# ── Tracking number validation ────────────────────────────────────────────────

_CARRIER_PATTERNS: dict[str, list[re.Pattern]] = {
    "ups":   [
        re.compile(r"^1Z[A-Z0-9]{16}$", re.IGNORECASE),
        re.compile(r"^\d{18}$"),
        re.compile(r"^\d{12}$"),
    ],
    "usps":  [
        re.compile(r"^\d{20,22}$"),
        re.compile(r"^9\d{15,21}$"),
    ],
    "fedex": [
        re.compile(r"^\d{12}$"),
        re.compile(r"^\d{15}$"),
        re.compile(r"^\d{20}$"),
    ],
    "dhl":   [
        re.compile(r"^\d{10,11}$"),
        re.compile(r"^[A-Z]{2}\d{9}[A-Z]{2}$", re.IGNORECASE),
    ],
}

# Generic: alphanumeric 8–30 chars, no path-like separators
_GENERIC_TRACKING_RE = re.compile(r"^[A-Z0-9]{8,30}$", re.IGNORECASE)


def _noisy_tracking_reason(val: str) -> str | None:
    """Return a rejection reason if the tracking number looks like OCR noise, else None."""
    if val.count("/") >= 2:
        return "contains multiple forward slashes"
    if re.search(r"(?i)\bstd[-_]|\bavbh\b", val):
        return "contains internal OCR artifact tokens (std-, avbh)"
    if re.search(r"[a-z]{3,}\d+[-/][a-z]+[-/]\d+", val, re.IGNORECASE):
        return "mixed alphanumeric fragments with separators — likely OCR noise"
    if len(val) > 40:
        return f"length {len(val)} exceeds expected maximum for a tracking number"
    return None


# ── Financial summary extraction ──────────────────────────────────────────────

# (label_regex, field_name, force_negative)
_SHIPMENT_FINANCIAL_LABELS: list[tuple[re.Pattern, str, bool]] = [
    (re.compile(r"(?i)\bsub\s*total\b"),                                        "subtotal",      False),
    (re.compile(r"(?i)\bshipping\s*(?:&\s*handling|and\s*handling|cost|fee)?\b"), "shipping_cost", False),
    (re.compile(r"(?i)\bpromotional?\s*(?:certificate|discount|savings|code)?\b"), "discount",     True),
    (re.compile(r"(?i)\border\s+total\b|\btotal\s+amount\b|\bgrand\s+total\b"),  "total_amount",  False),
    (re.compile(r"(?i)\bbalance\s+due\b"),                                       "balance_due",   False),
]

_PAID_VIA_RE   = re.compile(
    r"(?i)paid\s+(?:via|by|with|using)\s+([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+\$?([\d,]+\.?\d*)"
)
_AMOUNT_EOL_RE = re.compile(r"(-?\s*\$?\s*)([\d,]+\.?\d*)\s*$")


class PackingSlipValidator(BaseDocValidator):
    doc_type_id = "packing_slip"

    def _run(self) -> None:
        self._check_ship_date()
        self._check_delivery_date()
        self._check_order_date()
        self._check_tracking_number()
        self._check_shipper()
        self._check_ship_to()
        self._check_bill_to()
        self._check_party_assignments()   # must run after the above three
        self._extract_financial_summary()
        self._check_financial_math()

    # ── Date guards ───────────────────────────────────────────────────────────

    def _check_ship_date(self) -> None:
        if not self.out.get("ship_date"):
            return
        if not any(re.search(rf"(?i){p}", self.raw_text) for p in _SHIP_DATE_LABELS):
            self._blank(
                "ship_date",
                "no explicit ship date label found (e.g. 'Ship Date', 'Shipped On') — "
                "inferred dates not allowed; prefer blank over wrong",
            )

    def _check_delivery_date(self) -> None:
        if not self.out.get("delivery_date"):
            return
        if not any(re.search(rf"(?i){p}", self.raw_text) for p in _DELIVERY_DATE_LABELS):
            self._blank(
                "delivery_date",
                "no explicit delivery date label found (e.g. 'Delivery Date', "
                "'Expected Delivery') — inferred dates not allowed",
            )

    # ── Order date ────────────────────────────────────────────────────────────

    def _check_order_date(self) -> None:
        if self.out.get("order_date"):
            return  # already populated by DI/LLM — keep it
        iso = self._find_order_date()
        if iso:
            self.out["order_date"] = iso
            self.fv["order_date"] = FieldValidation(
                raw_value=None, normalized_value=iso,
                confidence=0.9, status="valid",
                reason="extracted from order date text pattern in document",
            )

    def _find_order_date(self) -> str | None:
        for pat in _ORDER_DATE_RES:
            m = pat.search(self.raw_text)
            if not m:
                continue
            raw = m.group(1).strip()
            # Try ISO, numeric, named-month in that order
            for parser in (
                lambda s: date.fromisoformat(s).isoformat(),
                _parse_numeric_date,
                _parse_named_date,
            ):
                try:
                    result = parser(raw)
                    if result:
                        return result
                except (ValueError, TypeError):
                    pass
        return None

    # ── Tracking number ───────────────────────────────────────────────────────

    def _check_tracking_number(self) -> None:
        val = self.out.get("tracking_number")
        if not val:
            return
        val_str = str(val).strip()

        # Noise check — reject OCR garbage immediately
        noise_reason = _noisy_tracking_reason(val_str)
        if noise_reason:
            self._blank("tracking_number",
                        f"rejected — noisy OCR candidate: {noise_reason}")
            return

        # Carrier-specific pattern validation
        carrier = str(self.out.get("carrier_name") or "").lower()
        carrier_key = next((k for k in _CARRIER_PATTERNS if k in carrier), None)

        if carrier_key:
            if any(p.match(val_str) for p in _CARRIER_PATTERNS[carrier_key]):
                self._set("tracking_number", val_str, 0.95, "valid",
                          f"validated against {carrier_key.upper()} tracking number pattern")
            else:
                self._blank(
                    "tracking_number",
                    f"rejected — value {val_str!r} does not match "
                    f"{carrier_key.upper()} tracking number format",
                )
        elif _GENERIC_TRACKING_RE.match(val_str):
            self._set("tracking_number", val_str, 0.7, "warning",
                      "generic alphanumeric format — carrier pattern not verified")
            self._warn("tracking_number",
                       "tracking_number accepted as generic format; "
                       "set carrier_name to enable carrier-specific validation")
        else:
            self._blank("tracking_number",
                        "rejected — does not match any known carrier tracking format")

    # ── Ship-to / Bill-to address extraction ─────────────────────────────────

    _SHIP_TO_LABEL_RE = re.compile(
        r"(?i)(?:ship(?:ped)?\s+to|deliver(?:y)?\s+to|recipient|consignee)"
    )
    _SHIP_FROM_LABEL_RE = re.compile(
        r"(?i)(?:ship(?:ped)?\s+from|sender|return\s+(?:to\s+)?address)"
    )
    _BILL_TO_LABEL_RE = re.compile(
        r"(?i)(?:bill(?:ed)?\s+to|invoice\s+to|sold\s+to|customer)"
    )

    def _check_shipper(self) -> None:
        """
        Validate shipper_name / shipper_address against label context, and attempt
        to extract them from raw text when DI/LLM left them blank.
        """
        self._validate_or_extract_address(
            name_field="shipper_name",
            addr_field="shipper_address",
            label_re=self._SHIP_FROM_LABEL_RE,
            label_desc="'Ship From' / 'Sender'",
        )

    def _check_ship_to(self) -> None:
        """
        Validate ship_to_name / ship_to_address against label context, and attempt
        to extract them from raw text when DI/LLM left them blank.
        """
        self._validate_or_extract_address(
            name_field="ship_to_name",
            addr_field="ship_to_address",
            label_re=self._SHIP_TO_LABEL_RE,
            label_desc="'Ship To' / 'Deliver To'",
        )

    def _check_bill_to(self) -> None:
        """
        Validate bill_to_name / bill_to_address against label context, and attempt
        to extract them from raw text when DI/LLM left them blank.
        """
        self._validate_or_extract_address(
            name_field="bill_to_name",
            addr_field="bill_to_address",
            label_re=self._BILL_TO_LABEL_RE,
            label_desc="'Bill To' / 'Sold To'",
        )

    def _extract_section_first_name(self, label_re: re.Pattern) -> str | None:
        """
        Return the first substantive name line immediately following a section
        label (e.g. the line after "SHIP FROM:" or "SHIP TO:").
        Skips FBA reference codes and generic prefix words.
        """
        m = label_re.search(self.raw_text)
        if not m:
            return None
        after = self.raw_text[m.end():]
        newline_pos = after.find("\n")
        if newline_pos == -1:
            rest = after.strip(" \t:,")
            tail = ""
        else:
            rest = after[:newline_pos].strip(" \t:,")
            tail = after[newline_pos + 1:]

        _SKIP_RE = re.compile(r"(?i)^(?:ship|bill|from|to|order|date|fba\s*:)\b")
        if rest and len(rest) >= 3 and not _SKIP_RE.match(rest):
            return rest

        for ln in tail.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            if re.match(r"(?i)^fba\s*:", stripped):
                continue  # FBA reference codes like "FBA: dnest+sta012"
            if len(stripped) >= 3 and not _SKIP_RE.match(stripped):
                return stripped
        return None

    def _check_party_assignments(self) -> None:
        """
        When both SHIP FROM and SHIP TO sections are present in the document,
        cross-check that shipper_name came from the SHIP FROM section and
        ship_to_name from the SHIP TO section.  Correct names (and addresses)
        if they appear to have been swapped by DI / LLM.
        """
        has_from = bool(self._SHIP_FROM_LABEL_RE.search(self.raw_text))
        has_to   = bool(self._SHIP_TO_LABEL_RE.search(self.raw_text))
        if not has_from or not has_to:
            return  # single-section document — no cross-check possible

        from_name = self._extract_section_first_name(self._SHIP_FROM_LABEL_RE)
        to_name   = self._extract_section_first_name(self._SHIP_TO_LABEL_RE)
        if not from_name or not to_name:
            return

        cur_shipper = self.out.get("shipper_name")
        cur_ship_to = self.out.get("ship_to_name")

        if not cur_shipper and not cur_ship_to:
            return  # nothing assigned yet — extraction methods already handle this

        def _matches(a: str | None, b: str) -> bool:
            if not a:
                return False
            return a.lower().strip() in b.lower() or b.lower() in a.lower().strip()

        shipper_in_to_section   = _matches(cur_shipper,  to_name)
        ship_to_in_from_section = _matches(cur_ship_to, from_name)
        shipper_in_from_section = _matches(cur_shipper,  from_name)
        ship_to_in_to_section   = _matches(cur_ship_to,  to_name)

        # Swap detected: one or both names appear in the wrong section,
        # and neither looks correctly placed.
        swapped = (
            (shipper_in_to_section or ship_to_in_from_section)
            and not (shipper_in_from_section or ship_to_in_to_section)
        )
        if not swapped:
            return

        cur_shipper_addr = self.out.get("shipper_address")
        cur_ship_to_addr = self.out.get("ship_to_address")

        self._set(
            "shipper_name",
            cur_ship_to or from_name,
            0.9, "valid",
            f"corrected swap — SHIP FROM section contains {from_name!r}",
        )
        self._set(
            "ship_to_name",
            cur_shipper or to_name,
            0.9, "valid",
            f"corrected swap — SHIP TO section contains {to_name!r}",
        )
        if cur_shipper_addr or cur_ship_to_addr:
            self._set("shipper_address", cur_ship_to_addr, 0.9, "valid",
                      "address swapped to match corrected shipper_name")
            self._set("ship_to_address", cur_shipper_addr, 0.9, "valid",
                      "address swapped to match corrected ship_to_name")
        self.warnings.append(
            "[shipper_name/ship_to_name] names were swapped — corrected from "
            "SHIP FROM / SHIP TO section labels in raw text"
        )

    def _validate_or_extract_address(
        self,
        name_field: str,
        addr_field: str,
        label_re: re.Pattern,
        label_desc: str,
    ) -> None:
        """
        If the field is already populated, verify label context exists and warn
        if it doesn't.  If blank, scan the raw text for the label and extract the
        lines that follow it as name + address.
        """
        existing_name = self.out.get(name_field)
        existing_addr = self.out.get(addr_field)

        label_match = label_re.search(self.raw_text)

        if existing_name or existing_addr:
            if not label_match:
                self._warn(name_field,
                           f"value present but no {label_desc} label found in raw text")
            return

        # Nothing extracted yet — try to pull from raw text
        if not label_match:
            return

        # Skip to the start of the line *after* the label line, then take up to 4
        # non-empty lines.  This handles both "Ship To: Name" (inline) and
        # "Ship To:\nName\nAddress" (multi-line) layouts.
        label_end = label_match.end()
        after = self.raw_text[label_end:]

        # If the remainder of the label's own line has content (inline layout),
        # treat that as the first candidate line; otherwise skip the blank label line.
        newline_pos = after.find("\n")
        if newline_pos == -1:
            rest_of_label_line = after.strip()
            tail = ""
        else:
            rest_of_label_line = after[:newline_pos].strip(" \t:,")
            tail = after[newline_pos + 1:]

        # Section-terminator: a line that looks like another label (ends with ":" or
        # matches known packing-slip headers).  Stop collecting before these.
        _SECTION_HEADER_RE = re.compile(
            r"(?i)^(?:ship(?:ped)?\s+to|bill(?:ed)?\s+to|sold\s+to|deliver(?:y)?\s+to|"
            r"from|return\s+to|order|carrier|tracking|subtotal|total|balance|paid).{0,20}:?\s*$"
        )

        _FBA_REF_RE = re.compile(r"(?i)^fba\s*:")

        candidate_lines: list[str] = []
        if rest_of_label_line and not _FBA_REF_RE.match(rest_of_label_line):
            candidate_lines.append(rest_of_label_line)
        for ln in tail.splitlines():
            stripped = ln.strip()
            if not stripped:
                continue
            if _SECTION_HEADER_RE.match(stripped):
                break  # new section starts here
            if _FBA_REF_RE.match(stripped):
                continue  # skip FBA reference codes like "FBA: dnest+sta012"
            candidate_lines.append(stripped)
            if len(candidate_lines) >= 4:
                break
        lines = candidate_lines

        if not lines:
            return

        # First line → name, remaining lines → address (joined with ", ")
        name = lines[0]
        addr = ", ".join(lines[1:]) if len(lines) > 1 else None

        # Sanity: skip if the "name" looks like another label or a short keyword
        if len(name) < 3 or re.match(r"(?i)^(?:ship|bill|from|to|order|date)\b", name):
            return

        self.out[name_field] = name
        self.fv[name_field] = FieldValidation(
            raw_value=None, normalized_value=name,
            confidence=0.75, status="valid",
            reason=f"extracted from text block following {label_desc} label",
        )
        if addr:
            self.out[addr_field] = addr
            self.fv[addr_field] = FieldValidation(
                raw_value=None, normalized_value=addr,
                confidence=0.75, status="valid",
                reason=f"extracted from text block following {label_desc} label",
            )

    # ── Financial summary ─────────────────────────────────────────────────────

    def _extract_financial_summary(self) -> None:
        lines = self.raw_text.splitlines()

        for label_re, field_name, force_negative in _SHIPMENT_FINANCIAL_LABELS:
            if self.out.get(field_name) is not None:
                continue  # already populated — don't overwrite
            for line in lines:
                if label_re.search(line):
                    amount = self._parse_line_amount(line, force_negative)
                    if amount is not None:
                        self.out[field_name] = amount
                        self.fv[field_name] = FieldValidation(
                            raw_value=None, normalized_value=amount,
                            confidence=0.9, status="valid",
                            reason=f"extracted from financial summary: {line.strip()!r}",
                        )
                        break

        # "Paid via Visa 499.99" → payment_method + amount_paid
        m = _PAID_VIA_RE.search(self.raw_text)
        if m:
            method = m.group(1).strip().upper()
            try:
                paid = f"{float(m.group(2).replace(',', '')):.2f}"
            except ValueError:
                paid = None
            if not self.out.get("payment_method"):
                self.out["payment_method"] = method
                self.fv["payment_method"] = FieldValidation(
                    raw_value=None, normalized_value=method, confidence=0.9,
                    status="valid",
                    reason=f"parsed from 'Paid via {m.group(1)}' pattern",
                )
            if not self.out.get("amount_paid") and paid:
                self.out["amount_paid"] = paid
                self.fv["amount_paid"] = FieldValidation(
                    raw_value=None, normalized_value=paid, confidence=0.9,
                    status="valid",
                    reason=f"parsed from 'Paid via {m.group(1)} {m.group(2)}' pattern",
                )

    def _parse_line_amount(self, line: str, force_negative: bool = False) -> str | None:
        m = _AMOUNT_EOL_RE.search(line)
        if not m:
            return None
        sign_text = m.group(1).strip()
        try:
            amount = float(m.group(2).replace(",", ""))
            if "-" in sign_text or (force_negative and amount > 0):
                amount = -abs(amount)
            return f"{amount:.2f}"
        except ValueError:
            return None

    # ── Financial arithmetic ──────────────────────────────────────────────────

    def _check_financial_math(self) -> None:
        subtotal = _to_float(self.out.get("subtotal"))
        shipping = _to_float(self.out.get("shipping_cost"))
        discount = _to_float(self.out.get("discount"))
        total    = _to_float(self.out.get("total_amount"))
        paid     = _to_float(self.out.get("amount_paid"))
        balance  = _to_float(self.out.get("balance_due"))

        if subtotal is not None and total is not None:
            expected = subtotal + (shipping or 0.0) + (discount or 0.0)
            if abs(expected - total) > 0.05:
                self._warn(
                    "total_amount",
                    f"financial math: subtotal({subtotal:.2f}) + "
                    f"shipping({shipping or 0:.2f}) + discount({discount or 0:.2f}) "
                    f"= {expected:.2f} but total_amount = {total:.2f}",
                )

        if paid is not None and balance is not None and total is not None:
            if abs(paid + balance - total) > 0.05:
                self._warn(
                    "amount_paid",
                    f"amount_paid({paid:.2f}) + balance_due({balance:.2f}) "
                    f"= {paid + balance:.2f} ≠ total_amount({total:.2f})",
                )


# ── Dispatcher ────────────────────────────────────────────────────────────────

_VALIDATORS: dict[str, type[BaseDocValidator]] = {
    "receipt":                   ReceiptValidator,
    "bill_utility":              UtilityBillValidator,
    "receipt_warranty_service":  ServiceInvoiceValidator,
    "purchase_order":            PurchaseOrderValidator,
    "quote_vendor":              QuoteEstimateValidator,
    "packing_slip":              PackingSlipValidator,
}


def semantic_validate(
    doc_type_id: str,
    extracted_fields: dict,
    raw_text: str = "",
    source_snippets: dict | None = None,
    candidate_values: dict | None = None,
) -> tuple[dict, ValidationResult]:
    """
    Run document-type-aware semantic validation.

    Returns:
        (normalized_fields, ValidationResult)
        normalized_fields — corrected field dict; pass to validate_and_normalize() next.
        ValidationResult  — full diagnostics (confidence, warnings, errors, field mapping).
    """
    cls = _VALIDATORS.get(doc_type_id, BaseDocValidator)
    validator = cls(
        extracted=extracted_fields,
        raw_text=raw_text,
        source_snippets=source_snippets,
        candidate_values=candidate_values,
    )
    result = validator.validate()

    if result.validation_warnings:
        logger.info(
            "Semantic validation [%s]: %d warning(s): %s",
            doc_type_id,
            len(result.validation_warnings),
            "; ".join(result.validation_warnings[:5]),
        )
    if result.validation_errors:
        logger.warning(
            "Semantic validation [%s]: %d error(s): %s",
            doc_type_id,
            len(result.validation_errors),
            "; ".join(result.validation_errors),
        )

    return result.normalized_fields, result
