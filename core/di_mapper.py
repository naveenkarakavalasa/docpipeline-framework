"""
Maps Azure Document Intelligence AnalyzeResult objects to canonical field dicts.

Each prebuilt model returns a different field schema:
  prebuilt-receipt  → MerchantName, TransactionDate, Total, TotalTax, Items …
  prebuilt-invoice  → VendorName, InvoiceDate, InvoiceTotal, TotalTax, Items …
  prebuilt-layout   → no semantic fields; raw text only (LLM handles extraction)

Returns (extracted: dict, raw_text: str, bboxes: dict).
bboxes maps canonical field names → normalized polygon [[x,y], …] (0-1 coords)
or None when no spatial data is available for that field.
Only non-None values are included in the extracted dict — missing fields are
filled by the LLM backfill stage.
"""
from __future__ import annotations

import logging

from validators.validators import _parse_number as _locale_parse

logger = logging.getLogger(__name__)


# ── Spatial helpers ────────────────────────────────────────────────────────────

def _bbox(field, pages: list) -> list[list[float]] | None:
    """
    Extract normalized polygon coordinates [[x,y], …] from a DocumentField.
    Coordinates are normalized to 0-1 by dividing by page width/height.
    Returns None if spatial data is missing or cannot be normalized.
    """
    if field is None:
        return None
    brs = getattr(field, "bounding_regions", None)
    if not brs:
        return None
    br = brs[0]  # Use first bounding region (first page the field appears on)
    page_num = getattr(br, "page_number", 1)
    page = next((p for p in pages if getattr(p, "page_number", None) == page_num), None)
    if page is None and pages:
        page = pages[0]
    pw = getattr(page, "width",  None) if page else None
    ph = getattr(page, "height", None) if page else None
    if not pw or not ph:
        return None
    polygon = getattr(br, "polygon", None)
    if not polygon:
        return None
    try:
        return [[round(p.x / pw, 5), round(p.y / ph, 5)] for p in polygon]
    except Exception:
        return None


# ── Field value extraction helpers ────────────────────────────────────────────

def _val(field) -> str | list | None:
    """
    Extract a Python value from a DocumentField.

    Returns:
      str      for string / date / time / number / currency fields
      list     for list fields (caller handles items)
      None     if field is None or value cannot be extracted
    """
    if field is None:
        return None

    vt = getattr(field, "value_type", None)
    v = getattr(field, "value", None)

    if vt == "currency":
        # Parse raw content first — DI may misinterpret locale-specific separators
        # (e.g. Indonesian "60.000" means 60,000 but DI SDK returns amount=60.0).
        import re
        content = getattr(field, "content", "") or ""
        amount = v.amount if (v is not None and hasattr(v, "amount")) else None
        logger.debug("DI currency field: content=%r amount=%r", content, amount)
        content_clean = re.sub(r"[£$€¥₩]", "", content)
        content_clean = re.sub(r"\bRp\.?\s*", "", content_clean, flags=re.IGNORECASE).strip()
        parsed_from_content = _locale_parse(content_clean)
        if parsed_from_content is not None:
            return f"{parsed_from_content:.2f}"
        if amount is not None:
            return f"{amount:.2f}"
        return None

    if vt in ("date", "time"):
        if v is not None and hasattr(v, "isoformat"):
            return v.isoformat()
        return getattr(field, "content", None)

    if vt == "number":
        return str(v) if v is not None else getattr(field, "content", None)

    if vt == "list":
        return v  # list of DocumentField — caller iterates

    if vt == "address":
        # AddressValue is not JSON-serializable — use raw content text
        return getattr(field, "content", None) or (str(v) if v is not None else None)

    # string / phoneNumber / selectionMark / unknown
    raw = v if v is not None else getattr(field, "content", None)
    # Safety: convert any remaining SDK objects to string
    if raw is not None and not isinstance(raw, (str, int, float, bool)):
        raw = str(raw)
    return raw


def _item_fields(item_field) -> dict | None:
    """Extract a single line-item from a list DocumentField entry."""
    if item_field is None:
        return None
    if getattr(item_field, "value_type", None) == "dictionary":
        return getattr(item_field, "value", None)  # dict[str, DocumentField]
    return None


# ── prebuilt-receipt mapper ────────────────────────────────────────────────────

def _map_receipt(doc_type_id: str, fields: dict, pages: list) -> tuple[dict, dict]:
    extracted: dict = {}
    bboxes: dict    = {}

    merchant_f = fields.get("MerchantName")
    addr_f     = fields.get("MerchantAddress")
    phone_f    = fields.get("MerchantPhoneNumber")

    merchant_name  = _val(merchant_f)
    merchant_addr  = _val(addr_f)
    merchant_phone = _val(phone_f)

    name_key, addr_key, phone_key = "merchant_name", "merchant_address", "merchant_phone"

    if merchant_name:  extracted[name_key]  = merchant_name
    if merchant_addr:  extracted[addr_key]  = merchant_addr
    if merchant_phone: extracted[phone_key] = merchant_phone
    bboxes[name_key]  = _bbox(merchant_f, pages)
    bboxes[addr_key]  = _bbox(addr_f, pages)
    bboxes[phone_key] = _bbox(phone_f, pages)

    for canonical, di_key in [
        ("transaction_date",  "TransactionDate"),
        ("transaction_time",  "TransactionTime"),
        ("subtotal",          "Subtotal"),
        ("tax_amount",        "TotalTax"),
        ("total",             "Total"),
        ("payment_method",    "PaymentType"),
    ]:
        f = fields.get(di_key)
        val = _val(f)
        if val is not None:
            extracted[canonical] = val
        bboxes[canonical] = _bbox(f, pages)

    # Line items
    items_raw = _val(fields.get("Items"))
    if isinstance(items_raw, list):
        items = []
        for entry in items_raw:
            ifd = _item_fields(entry)
            if not ifd:
                continue
            item: dict = {
                "description": _val(ifd.get("Description")),
                "quantity":    _val(ifd.get("Quantity")),
                "unit_price":  _val(ifd.get("Price")),
                "line_total":  _val(ifd.get("TotalPrice")),
                "sku":         None,
            }
            items.append({k: v for k, v in item.items() if v is not None})
        if items:
            extracted["items"] = items

    return {k: v for k, v in extracted.items() if v is not None}, bboxes


# ── prebuilt-invoice mapper ────────────────────────────────────────────────────

def _map_invoice(doc_type_id: str, fields: dict, pages: list) -> tuple[dict, dict]:
    extracted: dict = {}
    bboxes: dict    = {}

    field_map = {
        "VendorName":              fields.get("VendorName"),
        "VendorAddress":           fields.get("VendorAddress"),
        "VendorAddressRecipient":  fields.get("VendorAddressRecipient"),
        "CustomerName":            fields.get("CustomerName"),
        "CustomerAddress":         fields.get("CustomerAddress"),
        "CustomerEmail":           fields.get("CustomerEmail"),
        "InvoiceId":               fields.get("InvoiceId"),
        "InvoiceDate":             fields.get("InvoiceDate"),
        "DueDate":                 fields.get("DueDate"),
        "SubTotal":                fields.get("SubTotal"),
        "TotalTax":                fields.get("TotalTax"),
        "InvoiceTotal":            fields.get("InvoiceTotal"),
        "PaymentTerm":             fields.get("PaymentTerm"),
        "ShippingHandling":        fields.get("ShippingHandling"),
    }

    def v(key): return _val(field_map[key])
    def b(key): return _bbox(field_map[key], pages)

    def set_f(canonical, di_key):
        val = v(di_key)
        if val is not None:
            extracted[canonical] = val
        bboxes[canonical] = b(di_key)

    if doc_type_id == "bill_utility":
        set_f("provider_name",    "VendorName")
        set_f("provider_address", "VendorAddress")
        set_f("account_number",   "InvoiceId")
        set_f("bill_date",        "InvoiceDate")
        set_f("due_date",         "DueDate")
        set_f("subtotal",         "SubTotal")
        set_f("tax_amount",       "TotalTax")
        set_f("total_amount",     "InvoiceTotal")

    elif doc_type_id == "purchase_order":
        set_f("vendor_name",    "VendorName")
        set_f("vendor_address", "VendorAddress")
        set_f("buyer_name",     "CustomerName")
        set_f("buyer_address",  "CustomerAddress")
        set_f("buyer_email",    "CustomerEmail")
        set_f("po_number",      "InvoiceId")
        set_f("po_date",        "InvoiceDate")
        set_f("delivery_date",  "DueDate")
        set_f("payment_terms",  "PaymentTerm")
        set_f("subtotal",       "SubTotal")
        set_f("tax_amount",     "TotalTax")
        set_f("total",          "InvoiceTotal")
        set_f("shipping",       "ShippingHandling")

    elif doc_type_id == "quote_vendor":
        set_f("vendor_name",    "VendorName")
        set_f("vendor_address", "VendorAddress")
        set_f("buyer_name",     "CustomerName")
        set_f("buyer_address",  "CustomerAddress")
        set_f("buyer_email",    "CustomerEmail")
        set_f("quote_number",   "InvoiceId")
        set_f("quote_date",     "InvoiceDate")
        set_f("valid_until",    "DueDate")
        set_f("payment_terms",  "PaymentTerm")
        set_f("subtotal",       "SubTotal")
        set_f("tax_amount",     "TotalTax")
        set_f("total",          "InvoiceTotal")

    # Line items (PO + Quote only)
    if doc_type_id in ("purchase_order", "quote_vendor"):
        items_raw = _val(fields.get("Items"))
        if isinstance(items_raw, list):
            items = []
            for entry in items_raw:
                ifd = _item_fields(entry)
                if not ifd:
                    continue
                item = {
                    "description": _val(ifd.get("Description")),
                    "quantity":    _val(ifd.get("Quantity")),
                    "unit_price":  _val(ifd.get("UnitPrice")),
                    "line_total":  _val(ifd.get("Amount")),
                    "sku":         _val(ifd.get("ProductCode")),
                }
                items.append({k: v for k, v in item.items() if v is not None})
            if items:
                extracted["items"] = items

    return {k: v for k, v in extracted.items() if v is not None}, bboxes


# ── Public mapper ──────────────────────────────────────────────────────────────

def map_di_result(model_id: str, doc_type_id: str, di_result) -> tuple[dict, str, dict]:
    """
    Map a DI AnalyzeResult to canonical extracted fields.

    Returns:
        (extracted, raw_text, bboxes)
        extracted  — non-null field values keyed by canonical field name
        raw_text   — full document text from DI (di_result.content)
        bboxes     — normalized polygon coords per canonical field name;
                     value is None when DI has no spatial data for that field
    """
    raw_text: str = di_result.content or ""
    pages: list   = list(di_result.pages) if di_result.pages else []

    if not di_result.documents:
        logger.info("DI mapper: no document entities (model=%s) — text only", model_id)
        return {}, raw_text, {}

    doc = di_result.documents[0]
    fields: dict = doc.fields or {}
    logger.info("DI mapper: %d field(s) in DI response, doc_type=%s", len(fields), doc_type_id)

    if model_id == "prebuilt-receipt":
        extracted, bboxes = _map_receipt(doc_type_id, fields, pages)
    elif model_id == "prebuilt-invoice":
        extracted, bboxes = _map_invoice(doc_type_id, fields, pages)
    else:
        # prebuilt-layout: no semantic fields, LLM does all extraction
        extracted = {}
        bboxes    = {}

    has_bboxes = sum(1 for v in bboxes.values() if v is not None)
    logger.info("DI mapper: mapped %d field(s), %d with bounding boxes", len(extracted), has_bboxes)
    return extracted, raw_text, bboxes
