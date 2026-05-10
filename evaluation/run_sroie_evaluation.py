#!/usr/bin/env python3
"""
Full SROIE evaluation — all 626 documents, 3 configurations.

Configurations:
  1. DI-only      — Azure Document Intelligence, no LLM
  2. DI+LLM       — DI + Claude LLM backfill
  3. Full pipeline — all 5 stages (DI → LLM → semantic → normalize → shape)

Writes results to evaluation/results/sroie_results.json.

Usage:
    cd <repo-root>
    python evaluation/run_sroie_evaluation.py

Prerequisites:
    - SROIE data at evaluation/sroie/image/ and evaluation/sroie/json/
      (run download: podbilabs/sroie-donut train+validation splits)
    - AZURE_DI_ENDPOINT + AZURE_DI_KEY
    - AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_KEY (for LLM stages)
"""
from __future__ import annotations

import difflib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_EVAL_DIR = Path(__file__).parent
_ROOT     = _EVAL_DIR.parent  # docpipeline-release/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import metrics as _metrics_module
from metrics import (
    FieldMetrics, average_metrics, compute_field_metrics,
    metrics_to_dict, population_rate,
    _normalize_str,
)
from pipeline_runner import run_all_stages, run_di_llm, run_di_only
from sroie_mapping import extract_sroie_ground_truth, list_sroie_documents

# SROIE address annotations are truncated substrings of the printed address — the
# dataset captures only the first line or a partial string, while the pipeline OCRs
# the full multi-line address block.  Exact match therefore always fails even when
# the extracted value is correct.  0.60 fuzzy threshold accepts partial-address
# matches without collapsing to random coincidences.
_SROIE_ADDRESS_FUZZY_THRESHOLD = 0.60  # merchant_address only; all other thresholds unchanged


def _compute_sroie_metrics(extracted: dict, gt: dict, raw_text: str = "") -> FieldMetrics:
    """compute_field_metrics with merchant_address fuzzy threshold patched to 0.60."""
    orig = _metrics_module._fields_match

    # Closure captures `orig` so the fallback always calls the real original,
    # not the patched version (which would cause infinite recursion).
    def _sroie_match(field_name: str, ex, gt_val) -> bool:
        if field_name == "merchant_address":
            if not ex or not gt_val:
                return False
            ratio = difflib.SequenceMatcher(
                None, _normalize_str(ex), _normalize_str(gt_val),
            ).ratio()
            return ratio >= _SROIE_ADDRESS_FUZZY_THRESHOLD
        return orig(field_name, ex, gt_val)

    _metrics_module._fields_match = _sroie_match
    try:
        return compute_field_metrics(extracted, gt, raw_text)
    finally:
        _metrics_module._fields_match = orig

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SROIE_DIR    = _EVAL_DIR / "sroie"
RESULTS_PATH = _EVAL_DIR / "results" / "sroie_results.json"
DOC_TYPE     = "receipt"
SROIE_FIELDS = ("merchant_name", "transaction_date", "merchant_address", "total")


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def run_evaluation() -> None:
    pairs = list_sroie_documents(SROIE_DIR)
    if not pairs:
        print(f"ERROR: No SROIE documents found at {SROIE_DIR}")
        sys.exit(1)

    print(f"SROIE Evaluation — {len(pairs)} documents")
    print(f"Results → {RESULTS_PATH}")

    di_only_metrics:  list[FieldMetrics] = []
    di_llm_metrics:   list[FieldMetrics] = []
    full_metrics:     list[FieldMetrics] = []
    per_doc_results:  list[dict] = []
    failed = 0

    for idx, (img_path, json_path) in enumerate(pairs, 1):
        doc_entry: dict = {"filename": img_path.name, "errors": []}

        try:
            gt = extract_sroie_ground_truth(json_path)
        except Exception as exc:
            print(f"[{idx}/{len(pairs)}] {img_path.name} — GT parse FAILED: {exc}")
            failed += 1
            continue

        try:
            image_bytes = img_path.read_bytes()
        except Exception as exc:
            print(f"[{idx}/{len(pairs)}] {img_path.name} — image read FAILED: {exc}")
            failed += 1
            continue

        # DI-only
        try:
            raw_text, fields = run_di_only(image_bytes, DOC_TYPE)
            m = _compute_sroie_metrics(fields, gt, raw_text)
            di_only_metrics.append(m)
            doc_entry["di_only"] = metrics_to_dict(m)
        except Exception as exc:
            doc_entry["di_only"] = {"error": str(exc)}
            print(f"  [{idx}] di_only FAILED: {exc}")
            failed += 1

        # DI + LLM
        try:
            raw_text, fields = run_di_llm(image_bytes, DOC_TYPE)
            m = _compute_sroie_metrics(fields, gt, raw_text)
            di_llm_metrics.append(m)
            doc_entry["di_llm"] = metrics_to_dict(m)
        except Exception as exc:
            doc_entry["di_llm"] = {"error": str(exc)}
            print(f"  [{idx}] di_llm FAILED: {exc}")
            failed += 1

        # Full pipeline
        try:
            snap = run_all_stages(image_bytes, DOC_TYPE)
            m = _compute_sroie_metrics(snap.after_stage_5, gt, snap.raw_text)
            full_metrics.append(m)
            doc_entry["full_pipeline"] = metrics_to_dict(m)
        except Exception as exc:
            doc_entry["full_pipeline"] = {"error": str(exc)}
            print(f"  [{idx}] full_pipeline FAILED: {exc}")
            failed += 1

        per_doc_results.append(doc_entry)

        if idx % 50 == 0 or idx == len(pairs):
            agg_di_interim   = average_metrics(di_only_metrics)
            agg_llm_interim  = average_metrics(di_llm_metrics)
            agg_full_interim = average_metrics(full_metrics)
            print(f"  ── {idx}/{len(pairs)} processed, {failed} failures ──")
            print(f"     di_only:       F1={_fmt(agg_di_interim.f1)}  P={_fmt(agg_di_interim.precision)}  R={_fmt(agg_di_interim.recall)}")
            print(f"     di_llm:        F1={_fmt(agg_llm_interim.f1)}  P={_fmt(agg_llm_interim.precision)}  R={_fmt(agg_llm_interim.recall)}")
            print(f"     full_pipeline: F1={_fmt(agg_full_interim.f1)}  P={_fmt(agg_full_interim.precision)}  R={_fmt(agg_full_interim.recall)}")
            # Write intermediate results so they're readable before completion
            interim = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "n_documents_processed": idx,
                "n_total": len(pairs),
                "n_failed": failed,
                "configurations": {
                    "di_only":       metrics_to_dict(agg_di_interim),
                    "di_llm":        metrics_to_dict(agg_llm_interim),
                    "full_pipeline": metrics_to_dict(agg_full_interim),
                },
                "per_document": per_doc_results,
            }
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, "w") as f:
                json.dump(interim, f, indent=2)

    agg_di   = average_metrics(di_only_metrics)
    agg_llm  = average_metrics(di_llm_metrics)
    agg_full = average_metrics(full_metrics)

    print(f"\n{'═' * 60}")
    print(f"SROIE EVALUATION — {len(pairs) - failed}/{len(pairs)} succeeded")
    header = f"{'Config':<16} {'P':>6} {'R':>6} {'F1':>6} {'Halluc':>8} {'Blank':>8}"
    print(header)
    print("─" * len(header))
    for label, m in [("DI-only", agg_di), ("DI+LLM", agg_llm), ("Full pipeline", agg_full)]:
        print(
            f"{label:<16} {_fmt(m.precision):>6} {_fmt(m.recall):>6} {_fmt(m.f1):>6} "
            f"{_fmt(m.hallucination_rate, pct=True):>8} {_fmt(m.blank_rate, pct=True):>8}"
        )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_documents": len(pairs),
        "n_failed": failed,
        "configurations": {
            "di_only":       metrics_to_dict(agg_di),
            "di_llm":        metrics_to_dict(agg_llm),
            "full_pipeline": metrics_to_dict(agg_full),
        },
        "per_document": per_doc_results,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    run_evaluation()
