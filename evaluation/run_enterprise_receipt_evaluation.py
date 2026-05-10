#!/usr/bin/env python3
"""
Enterprise receipt corpus evaluation — per-field accuracy + validator tallies.

Writes evaluation/results/enterprise_receipt_results.json which is read by
update_paper_tables.py to populate Table 3 in the paper.

Configurations:
  all                — DI-only, DI+LLM, Full pipeline, Full+HITL
  full_pipeline_only — Full pipeline only (default; fastest; skips DI-only / DI+LLM)
  di_only            — DI extraction only
  di_llm             — DI + LLM backfill only

Usage:
    cd <repo-root>
    python evaluation/run_enterprise_receipt_evaluation.py
    python evaluation/run_enterprise_receipt_evaluation.py --config full_pipeline_only
    python evaluation/run_enterprise_receipt_evaluation.py --config all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_EVAL_DIR = Path(__file__).parent
_ROOT     = _EVAL_DIR.parent  # docpipeline-release/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ground_truth_utils import load_enterprise_ground_truth
from metrics import (
    FieldMetrics, average_metrics, compute_field_metrics,
    metrics_to_dict, _fields_match,
)
from pipeline_runner import run_all_stages, run_di_llm, run_di_only

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CORPUS_DIR   = _EVAL_DIR.parent / "corpus" / "receipts"
RESULTS_PATH = _EVAL_DIR / "results" / "enterprise_receipt_results.json"
DOC_TYPE     = "receipt"

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".bmp"}

_ALL_CONFIGS     = ("di_only", "di_llm", "full_pipeline")
_FULL_ONLY_CONF  = ("full_pipeline",)


# ── Corpus loader ─────────────────────────────────────────────────────────────

def _list_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for doc in sorted(CORPUS_DIR.iterdir()):
        if doc.suffix.lower() not in _EXTENSIONS:
            continue
        gt = CORPUS_DIR / (doc.stem + "_ground_truth.json")
        if gt.exists():
            pairs.append((doc, gt))
        else:
            logger.warning("No ground truth for %s — skipping", doc.name)
    return pairs


# ── Validator tallies ─────────────────────────────────────────────────────────

def _init_tallies() -> dict:
    return {
        "receipt_store_number_split": {
            "n_documents": 0, "triggered": 0, "correct": 0, "elimination_rate": None,
        },
        "receipt_item_sum_verification": {
            "n_documents": 0, "triggered": 0, "correct": 0, "elimination_rate": None,
        },
    }


def _tally(val_result, fields: dict, gt: dict, tallies: dict) -> None:
    warnings = val_result.validation_warnings or [] if val_result else []

    store_key = "receipt_store_number_split"
    tallies[store_key]["n_documents"] += 1
    if any("store_number" in w.lower() or "store number" in w.lower() for w in warnings):
        tallies[store_key]["triggered"] += 1
        if (gt.get("store_number") and fields.get("store_number") and
                str(gt["store_number"]).strip() == str(fields["store_number"]).strip()):
            tallies[store_key]["correct"] += 1

    sum_key = "receipt_item_sum_verification"
    tallies[sum_key]["n_documents"] += 1
    if any("item" in w.lower() and "sum" in w.lower() for w in warnings):
        tallies[sum_key]["triggered"] += 1
        from metrics import _to_float
        a = _to_float(fields.get("subtotal"))
        b = _to_float(gt.get("subtotal"))
        if a is not None and b is not None and abs(a - b) <= 0.10:
            tallies[sum_key]["correct"] += 1


def _finalize_tallies(tallies: dict) -> None:
    for t in tallies.values():
        if t["triggered"] > 0:
            t["elimination_rate"] = round(t["correct"] / t["triggered"], 4)


# ── Per-field accuracy tracker ────────────────────────────────────────────────

def _init_field_counts(gt_fields: list[str]) -> dict:
    return {f: {"correct": 0, "total": 0} for f in gt_fields}


def _update_field_counts(
    counts: dict, fields: dict, gt: dict, raw_text: str
) -> None:
    for field, gt_val in gt.items():
        if field == "items" or gt_val is None:
            continue
        counts.setdefault(field, {"correct": 0, "total": 0})
        counts[field]["total"] += 1
        ex_val = fields.get(field)
        if ex_val is not None and _fields_match(field, ex_val, gt_val):
            counts[field]["correct"] += 1


def _field_accuracy_summary(counts: dict) -> dict:
    summary = {}
    for field, c in sorted(counts.items()):
        total = c["total"]
        correct = c["correct"]
        summary[field] = {
            "correct": correct,
            "total": total,
            "accuracy": round(correct / total, 4) if total else None,
        }
    return summary


# ── Main evaluation ───────────────────────────────────────────────────────────

def run(configs: tuple[str, ...]) -> None:
    pairs = _list_pairs()
    if not pairs:
        print(f"ERROR: No annotated receipts found in {CORPUS_DIR}")
        sys.exit(1)

    run_full = "full_pipeline" in configs
    run_llm  = "di_llm"        in configs
    run_di   = "di_only"       in configs

    print(f"Enterprise receipt evaluation — {len(pairs)} documents")
    print(f"Configs: {', '.join(configs)}")
    print(f"Output → {RESULTS_PATH}\n")

    di_metrics:   list[FieldMetrics] = []
    llm_metrics:  list[FieldMetrics] = []
    full_metrics: list[FieldMetrics] = []

    tallies     = _init_tallies()
    field_counts: dict = {}
    per_doc: list[dict] = []
    failed = 0

    for idx, (doc_path, gt_path) in enumerate(pairs, 1):
        print(f"  [{idx}/{len(pairs)}] {doc_path.name}")
        try:
            gt = load_enterprise_ground_truth(gt_path)
        except Exception as exc:
            logger.error("GT load failed for %s: %s", doc_path.name, exc)
            failed += 1
            continue

        try:
            image_bytes = doc_path.read_bytes()
        except Exception as exc:
            logger.error("Image read failed for %s: %s", doc_path.name, exc)
            failed += 1
            continue

        doc_entry: dict = {"filename": doc_path.name}

        if run_di:
            try:
                raw, fields = run_di_only(image_bytes, DOC_TYPE)
                m = compute_field_metrics(fields, gt, raw)
                di_metrics.append(m)
                doc_entry["di_only"] = metrics_to_dict(m)
            except Exception as exc:
                logger.error("DI-only failed for %s: %s", doc_path.name, exc)
                doc_entry["di_only"] = {"error": str(exc)}

        if run_llm:
            try:
                raw, fields = run_di_llm(image_bytes, DOC_TYPE)
                m = compute_field_metrics(fields, gt, raw)
                llm_metrics.append(m)
                doc_entry["di_llm"] = metrics_to_dict(m)
            except Exception as exc:
                logger.error("DI+LLM failed for %s: %s", doc_path.name, exc)
                doc_entry["di_llm"] = {"error": str(exc)}

        if run_full:
            try:
                snap = run_all_stages(image_bytes, DOC_TYPE)
                fields_full = snap.after_stage_5
                m = compute_field_metrics(fields_full, gt, snap.raw_text)
                full_metrics.append(m)
                doc_entry["full_pipeline"] = metrics_to_dict(m)

                _tally(snap.validation_result, fields_full, gt, tallies)
                _update_field_counts(field_counts, fields_full, gt, snap.raw_text)
            except Exception as exc:
                logger.error("Full pipeline failed for %s: %s", doc_path.name, exc)
                doc_entry["full_pipeline"] = {"error": str(exc)}
                failed += 1

        per_doc.append(doc_entry)

    _finalize_tallies(tallies)

    # Build aggregate configuration results
    configurations: dict = {}
    if run_di:
        configurations["di_only"] = metrics_to_dict(average_metrics(di_metrics))
    if run_llm:
        configurations["di_llm"] = metrics_to_dict(average_metrics(llm_metrics))
    if run_full:
        agg = average_metrics(full_metrics)
        configurations["full_pipeline"] = metrics_to_dict(agg)
        configurations["full_pipeline_hitl"] = {
            **metrics_to_dict(agg),
            "note": "pre-correction; run apply_hitl_corrections.py after review",
        }

    # Print summary
    print(f"\n{'═' * 55}")
    print(f"ENTERPRISE RECEIPT EVALUATION — {len(pairs) - failed}/{len(pairs)} succeeded")
    for cfg, m in configurations.items():
        if "note" in m:
            continue
        f1 = m.get("f1")
        p  = m.get("precision")
        r  = m.get("recall")
        print(f"  {cfg:<20}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}")

    if run_full and field_counts:
        print(f"\n  Per-field accuracy (full_pipeline):")
        for field, c in sorted(field_counts.items()):
            acc = c["correct"] / c["total"] if c["total"] else 0.0
            print(f"    {field:<22} {c['correct']:>2}/{c['total']:>2}  ({acc:.0%})")

    print(f"\n  Validator tallies:")
    for key, t in tallies.items():
        print(f"    {key}: triggered={t['triggered']}/{t['n_documents']}  "
              f"correct={t['correct']}  elim_rate={t['elimination_rate']}")

    # Write results
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_documents": len(pairs),
        "n_failed": failed,
        "configs_run": list(configs),
        "configurations": configurations,
        "per_field_accuracy": _field_accuracy_summary(field_counts) if run_full else {},
        "validator_specific": tallies,
        "per_document": per_doc,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {RESULTS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise receipt evaluation")
    parser.add_argument(
        "--config",
        choices=["all", "full_pipeline_only", "di_only", "di_llm", "full_pipeline"],
        default="full_pipeline_only",
        help="Which pipeline configuration(s) to run (default: full_pipeline_only)",
    )
    args = parser.parse_args()

    if args.config == "all":
        configs = _ALL_CONFIGS
    elif args.config == "full_pipeline_only":
        configs = _FULL_ONLY_CONF
    else:
        configs = (args.config,)

    run(configs)


if __name__ == "__main__":
    main()
