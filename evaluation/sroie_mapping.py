"""
SROIE dataset → receipt field registry mapping.

SROIE (Scanned Receipts OCR and Information Extraction, ICDAR 2019) annotates
4 fields per receipt stored in gt_parse JSON:
  company  → merchant_name
  date     → transaction_date  (normalized to YYYY-MM-DD)
  address  → merchant_address
  total    → total             (normalized to "1234.56" string)

All other registry fields are left as None (not annotated by SROIE).
"""
from __future__ import annotations

import json
import re
from pathlib import Path


def _normalize_date(s: str) -> str | None:
    """
    Normalize Malaysian receipt date strings to YYYY-MM-DD.

    Formats seen in SROIE (all DD-first, Malaysian convention):
      22/03/2018  →  2018-03-22
      26-05-2018  →  2018-05-26
      17/05/17    →  2017-05-17
      06/07/16    →  2016-07-06
    """
    s = s.strip()
    m = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", s)
    if not m:
        return s  # return as-is if unrecognized
    day, month, year = m.group(1), m.group(2), m.group(3)
    if len(year) == 2:
        year = "20" + year
    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except ValueError:
        return s


def _normalize_total(s: str) -> str | None:
    """Strip currency prefix (RM, $, etc.) and normalize to '1234.56' string."""
    s = re.sub(r"[£$€¥₩]", "", s)
    s = re.sub(r"\bRM\.?\s*", "", s, flags=re.IGNORECASE).strip()
    try:
        return f"{float(s):.2f}"
    except ValueError:
        return s if s else None


def extract_sroie_ground_truth(json_path: str | Path) -> dict:
    """
    Read a SROIE gt_parse JSON and return a flat dict matching the receipt registry schema.

    Only the 4 SROIE-annotated fields are populated; all others are None.
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    gt_parse = data.get("gt_parse", data)  # some files have gt_parse wrapper, some don't

    company = gt_parse.get("company") or None
    date    = gt_parse.get("date")    or None
    address = gt_parse.get("address") or None
    total   = gt_parse.get("total")   or None

    return {
        "merchant_name":     company.strip() if company else None,
        "merchant_address":  address.strip() if address else None,
        "merchant_phone":    None,
        "receipt_number":    None,
        "transaction_date":  _normalize_date(date) if date else None,
        "transaction_time":  None,
        "subtotal":          None,
        "tax_amount":        None,
        "tip":               None,
        "discount":          None,
        "total":             _normalize_total(total) if total else None,
        "payment_method":    None,
        "store_number":      None,
        "notes":             None,
        "items":             None,
    }


def list_sroie_documents(sroie_dir: str | Path) -> list[tuple[Path, Path]]:
    """
    Return sorted list of (image_path, json_path) pairs from evaluation/sroie/.
    Expects image/ and json/ subdirectories with matching filenames.
    """
    sroie_dir = Path(sroie_dir)
    image_dir = sroie_dir / "image"
    json_dir  = sroie_dir / "json"

    pairs = []
    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        json_path = json_dir / (img_path.stem + ".json")
        if json_path.exists():
            pairs.append((img_path, json_path))

    return pairs
