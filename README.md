# DocPipeline Framework

DocPipeline Framework is a research implementation of a production-oriented enterprise document extraction architecture combining Azure Document Intelligence, LLM-assisted field recovery, semantic validation, arithmetic verification, and confidence-aware human review escalation. The framework prioritizes reliability over benchmark-only accuracy through document-type-aware validators and the principle of “prefer blank over wrong.

## Paper

"Designing Reliable Enterprise Document AI Pipelines: A Production
Framework for Scalable Document Extraction, Validation, and
Human-in-the-Loop Governance"

*Citation will be added upon publication.*

## Architecture 
<img width="1795" height="1780" alt="Figure1 Architecture" src="https://github.com/user-attachments/assets/67c96aea-b56d-4a5b-9d29-c70b33a2b173" />


## What This Repository Contains
validators/
    semantic_validator.py       # Stage 3: per-type validator classes
    validators.py               # Stage 4: monetary normalization
    llm_assist.py               # Stage 2: LLM backfill and prompts
    test_semantic_validator.py  # Unit tests
registry/
    field_registry.py           # Field schemas for 8 document types
    di_router.py                # Azure DI model routing table
core/
    azure_di_client.py          # Azure DI API wrapper
    di_mapper.py                # Maps DI AnalyzeResult to canonical fields
evaluation/
    run_sroie_evaluation.py              # SROIE benchmark (626 receipts)
    run_enterprise_receipt_evaluation.py # Enterprise corpus (13 receipts)
    apply_hitl_corrections.py            # Apply human review corrections
    update_paper_tables.py               # Update paper from results JSON
    ground_truth_utils.py                # Ground truth helpers
    metrics.py                           # Field-level metric computation
    pipeline_runner.py                   # Pipeline stage wrappers for eval
    sroie_mapping.py                     # SROIE dataset ground truth mapping
corpus/receipts/              # 13 annotated real-world receipts
    *.jpg / *.jpeg / *.png / *.pdf    # Receipt images
    *_ground_truth.json               # Field-level annotations
extraction_orchestrator.py    # Five-stage pipeline entry point

## Corpus

13 real-world receipt documents spanning:
- Grocery and supermarket (Target, Walmart x2, Costco x2, ALDI, Food Lion,
  Newark Farmers Market)
- Restaurant (Taco Bell)
- Department store (Marshalls)
- Service (USPS postal receipts x2)
- Handwritten payment receipt (daycare)

Each document includes a `_ground_truth.json` with field-level
annotations across 14 registry fields. License: CC BY 4.0.
Full dataset also archived at: https://doi.org/10.5281/zenodo.20113972

Additional document-type implementations (utility bills, quotes, packing slips, purchase orders, expense reports, service invoices, product catalogs) are included in the framework architecture. Public annotated corpora are being expanded incrementally and will be released in future versions.


## Requirements

```bash
pip install azure-ai-documentintelligence openai \
            python-dotenv pymupdf
```

## Azure Credentials Required

The full pipeline requires Azure credentials in a `.env` file:

AZURE_DI_ENDPOINT=...
AZURE_DI_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_KEY=...
AZURE_OPENAI_DEPLOYMENT=gpt-4o

The validator classes (`semantic_validator.py`, `validators.py`)
and field registry run without any credentials and can be
used independently.

## Running the Evaluation

### SROIE benchmark (626 receipts, 4 fields):
```bash
# Download SROIE first: https://github.com/zzzDavid/ICDAR-2019-SROIE
# Place in evaluation/sroie/test/
cd evaluation
python run_sroie_evaluation.py
```

### Enterprise corpus (13 receipts, 14 fields):
```bash
cd evaluation
python run_enterprise_receipt_evaluation.py
```

## Key Design Principles

**Prefer blank over wrong:** When a field value cannot be confirmed
against label context in the raw OCR text, the framework clears it
rather than retaining a suspect value. A blank field triggers human
review; a wrong value corrupts downstream systems silently.

**Label-context confidence scoring:** Every extracted field is scored
against expected label patterns in the raw OCR output. Fields below
a configurable threshold are flagged for human review.

**Graceful degradation:** Each pipeline stage fails independently.
A network timeout or API error causes that stage to be skipped with
the best-effort result passed forward, rather than aborting the job.

## License

Code: MIT License
Corpus: Creative Commons Attribution 4.0 (CC BY 4.0)

## Contact

Naveen Karakavalasa
https://nkspace.dev
