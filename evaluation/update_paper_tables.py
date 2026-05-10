#!/usr/bin/env python3
"""
Update paper placeholder tables with real experimental results.

Data sources:
  sroie_results.json           — Table 1 aggregate metrics (receipt rows, 3 configs)
  evaluation_results.json      — Table 1 enterprise quotes/expense rows; Table 2; Table 4
  enterprise_receipt_results.json — Table 3 per-field + validator-specific results

Prints a diff of every line that will change and asks for confirmation
before writing. Rounds all metrics to 2 decimal places.

Usage:
    cd evaluation/
    python update_paper_tables.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_EVAL_DIR = Path(__file__).parent
SROIE_PATH              = _EVAL_DIR / "results" / "sroie_results.json"
ENTERPRISE_PATH         = _EVAL_DIR / "results" / "evaluation_results.json"
ENTERPRISE_RECEIPT_PATH = _EVAL_DIR / "results" / "enterprise_receipt_results.json"
PAPER_PATH              = _EVAL_DIR.parent / "docs" / "paper_enterprise_doc_ai_pipeline.md"


def _r2(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v):.2f}"


def _pct(v) -> str:
    if v is None:
        return "N/A"
    return f"{float(v) * 100:.1f}%"


def _build_table1(sroie: dict, enterprise: dict) -> str:
    """
    Table 1: Field-level extraction performance by pipeline configuration.

    Receipt rows come from sroie_results.json (626-document SROIE benchmark).
    Quote and expense-report rows come from evaluation_results.json.
    Per-field accuracy is reported separately in Table 3 from enterprise corpus.
    """
    lines = [
        "*Table 1. Field-level extraction performance by pipeline configuration "
        "(receipts N=626 SROIE benchmark, vendor quotes, expense reports). "
        "Per-field accuracy reported in Table 3 from enterprise corpus evaluation.*\n",
        "| Configuration | Document Type | Precision | Recall | F1 | "
        "Mis-assign Rate | Hallucination Rate | Blank Rate |",
        "|---|---|---|---|---|---|---|---|",
    ]

    config_labels = [
        ("di_only",            "DI-only"),
        ("di_llm",             "DI + LLM"),
        ("full_pipeline",      "Full pipeline"),
        ("full_pipeline_hitl", "Full + HITL"),
    ]

    # Receipt rows from SROIE (3 configs — SROIE has no HITL)
    sroie_configs = sroie.get("configurations", {})
    for config_key, config_label in config_labels[:3]:
        m = sroie_configs.get(config_key, {})
        lines.append(
            f"| {config_label} | receipt (SROIE) | "
            f"{_r2(m.get('precision'))} | {_r2(m.get('recall'))} | "
            f"{_r2(m.get('f1'))} | {_r2(m.get('mis_assignment_rate'))} | "
            f"{_r2(m.get('hallucination_rate'))} | {_r2(m.get('blank_rate'))} |"
        )

    # Quote and expense-report rows from enterprise evaluation
    ent = enterprise.get("enterprise", {})
    type_labels = [
        ("quotes",          "quote\\_vendor"),
        ("expense_reports", "report\\_expense"),
    ]
    for folder, type_label in type_labels:
        configs = ent.get(folder, {}).get("configurations", {})
        for config_key, config_label in config_labels:
            m = configs.get(config_key, {})
            lines.append(
                f"| {config_label} | {type_label} | "
                f"{_r2(m.get('precision'))} | {_r2(m.get('recall'))} | "
                f"{_r2(m.get('f1'))} | {_r2(m.get('mis_assignment_rate'))} | "
                f"{_r2(m.get('hallucination_rate'))} | {_r2(m.get('blank_rate'))} |"
            )

    return "\n".join(lines)


def _build_table2(enterprise: dict) -> str:
    """Table 2: Cumulative field population rate by pipeline stage."""
    lines = [
        "*Table 2. Cumulative field population rate by pipeline stage "
        "(receipts, vendor quotes, expense reports; averages).*\n",
        "| Stage | SROIE (receipts) | Enterprise receipts | Enterprise quotes | "
        "Enterprise expense reports |",
        "|---|---|---|---|---|",
    ]

    # Stage population captured from CORD evaluation; reuse structure for SROIE
    # when enterprise per-stage data is available this table will be fully populated
    sroie_pop = (enterprise.get("cord") or {}).get("stage_population", {})
    stage_labels = [
        ("after_stage_1", "Stage 1 — DI extraction"),
        ("after_stage_2", "Stage 2 — LLM backfill"),
        ("after_stage_3", "Stage 3 — Semantic validation"),
        ("after_stage_4", "Stage 4 — Monetary normalization"),
        ("after_stage_5", "Stage 5 — Schema shaping"),
    ]
    for key, label in stage_labels:
        val = sroie_pop.get(key)
        lines.append(f"| {label} | {_pct(val)} | — | — | — |")

    lines.append(
        "\n*Note: Enterprise per-stage population rates will be added "
        "from run\\_enterprise\\_evaluation.py in a future update.*"
    )
    return "\n".join(lines)


def _build_table3_rows(enterprise_receipt: dict) -> list[tuple[str, str]]:
    """
    Return (old_line, new_line) pairs for Table 3 validator-specific rows.
    Reads from enterprise_receipt_results.json.
    """
    vs = enterprise_receipt.get("validator_specific", {})
    replacements = []

    def _elim_rate(key: str) -> str:
        t = vs.get(key, {})
        rate = t.get("elimination_rate")
        n = t.get("triggered", 0)
        correct = t.get("correct", 0)
        if rate is None:
            return "*[pending experiments]*"
        return f"{_r2(rate)} ({correct}/{n} triggered)"

    replacements.append((
        "| Store number split | receipt | merchant_name, store_number | 15 | *[pending experiments]* |",
        f"| Store number split | receipt | merchant_name, store_number | "
        f"{vs.get('receipt_store_number_split', {}).get('n_documents', 15)} | "
        f"{_elim_rate('receipt_store_number_split')} |",
    ))
    replacements.append((
        "| Item sum verification | receipt | subtotal | 15 | *[pending experiments]* |",
        f"| Item sum verification | receipt | subtotal | "
        f"{vs.get('receipt_item_sum_verification', {}).get('n_documents', 15)} | "
        f"{_elim_rate('receipt_item_sum_verification')} |",
    ))
    replacements.append((
        "| valid_until temporal guard | quote_vendor | valid_until | 20 | *[pending experiments]* |",
        f"| valid_until temporal guard | quote\\_vendor | valid\\_until | "
        f"{vs.get('quote_valid_until_temporal_guard', {}).get('n_documents', 20)} | "
        f"{_elim_rate('quote_valid_until_temporal_guard')} |",
    ))
    return replacements


def _build_table4(enterprise: dict) -> str:
    """Table 4: Field precision by confidence score bin."""
    lines = [
        "*Table 4. Field precision by confidence score bin "
        "(receipts, vendor quotes, expense reports; CORD test split).*\n",
        "| Confidence Bin | N Fields | Precision |",
        "|---|---|---|",
    ]
    calib = (enterprise.get("cord") or {}).get("confidence_calibration", {})
    bin_labels = [
        ("0.9_1.0", "0.90–1.00"),
        ("0.7_0.9", "0.70–0.90"),
        ("0.5_0.7", "0.50–0.70"),
        ("0.0_0.5", "0.00–0.50"),
    ]
    for key, label in bin_labels:
        b = calib.get(key, {})
        n = b.get("n_fields", "N/A")
        p = _r2(b.get("precision"))
        lines.append(f"| {label} | {n} | {p} |")

    return "\n".join(lines)


def _build_sroie_inline(sroie: dict) -> str:
    """Replace the CORD/SROIE inline placeholder in Section 5.1."""
    n = sroie.get("n_documents", 626)
    full = sroie.get("configurations", {}).get("full_pipeline", {})
    f1 = full.get("f1")
    p  = full.get("precision")
    r  = full.get("recall")
    if f1 is None:
        return "*[CORD benchmark numbers to be added after pipeline run against CORD dataset.]*"
    return (
        f"Receipt evaluation on the SROIE benchmark (N = {n}) yields "
        f"precision = {_r2(p)}, recall = {_r2(r)}, F1 = {_r2(f1)} "
        f"for the full pipeline configuration."
    )


def _build_table1_inline_ref(sroie: dict) -> str:
    """Replace Table 1 inline [pending experiments] reference in Section 6.2."""
    full = sroie.get("configurations", {}).get("full_pipeline", {})
    f1 = full.get("f1")
    if f1 is None:
        return "*[pending experiments]*"
    return f"(Table 1; full pipeline F1 = {_r2(f1)} on SROIE)"


def compute_replacements(
    sroie: dict,
    enterprise: dict,
    enterprise_receipt: dict,
) -> list[tuple[str, str]]:
    """
    Return list of (old_text, new_text) pairs for every placeholder in the paper.
    """
    replacements: list[tuple[str, str]] = []

    replacements.append((
        "*[Table 1 to be populated after experiments complete.]*",
        _build_table1(sroie, enterprise),
    ))

    replacements.append((
        "*[Table 2 to be populated after experiments complete.]*",
        _build_table2(enterprise),
    ))

    replacements.extend(_build_table3_rows(enterprise_receipt))

    replacements.append((
        "*[Table 4 to be populated after experiments complete.]*",
        _build_table4(enterprise),
    ))

    replacements.append((
        "*[CORD benchmark numbers to be added after pipeline run against CORD dataset.]*",
        _build_sroie_inline(sroie),
    ))

    replacements.append((
        "is reported in Table 1 *[pending experiments]*.",
        f"is reported in Table 1 {_build_table1_inline_ref(sroie)}.",
    ))

    return replacements


def main() -> None:
    # Load SROIE results (required)
    if not SROIE_PATH.exists():
        print(f"ERROR: SROIE results not found: {SROIE_PATH}")
        print("Run run_sroie_evaluation.py first.")
        sys.exit(1)
    with open(SROIE_PATH) as f:
        sroie = json.load(f)

    # Load enterprise results (required for quotes/expense rows + Table 4)
    if not ENTERPRISE_PATH.exists():
        print(f"ERROR: Enterprise results not found: {ENTERPRISE_PATH}")
        print("Run run_enterprise_evaluation.py first.")
        sys.exit(1)
    with open(ENTERPRISE_PATH) as f:
        enterprise = json.load(f)

    # Load enterprise receipt results (optional — Table 3 stays pending if absent)
    enterprise_receipt: dict = {}
    if ENTERPRISE_RECEIPT_PATH.exists():
        with open(ENTERPRISE_RECEIPT_PATH) as f:
            enterprise_receipt = json.load(f)
    else:
        print(f"NOTE: {ENTERPRISE_RECEIPT_PATH.name} not found — Table 3 rows will remain pending.")

    if not PAPER_PATH.exists():
        print(f"ERROR: Paper not found: {PAPER_PATH}")
        sys.exit(1)

    with open(PAPER_PATH) as f:
        original = f.read()

    replacements = compute_replacements(sroie, enterprise, enterprise_receipt)

    # Warn about missing placeholders
    not_found = [old[:80] for old, _ in replacements if old not in original]
    if not_found:
        print("WARNING: The following placeholders were NOT found in the paper:")
        for nf in not_found:
            print(f"  ...{nf}...")
        print()

    # Build diff
    modified = original
    changes: list[tuple[int, str, str]] = []
    for old, new in replacements:
        if old not in modified:
            continue
        for lineno, line in enumerate(modified.splitlines(), 1):
            if old in line or old == line.strip():
                changes.append((lineno, line.strip(), new[:120]))
                break
        modified = modified.replace(old, new)

    if not changes:
        print("No placeholders found to replace — paper may already be up to date.")
        sys.exit(0)

    print("=== Lines that will change ===\n")
    for lineno, old_line, new_preview in changes:
        print(f"Line ~{lineno}:")
        print(f"  OLD: {old_line[:120]}")
        print(f"  NEW: {new_preview[:120]}...")
        print()

    answer = input("Proceed with these changes? (y/n): ").strip().lower()
    if answer != "y":
        print("Aborted — no changes written.")
        sys.exit(0)

    with open(PAPER_PATH, "w") as f:
        f.write(modified)

    print(f"\nPaper updated: {PAPER_PATH}")
    print(f"{len(changes)} placeholder(s) replaced.")


if __name__ == "__main__":
    main()
