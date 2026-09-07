# Papertrail — Fact Knowledge Layer

An LLM-powered document intelligence system that extracts structured facts from PDFs and automatically identifies corroborations, contradictions, and reconcilable discrepancies across documents.

---

## Setup and Run Instructions

> **Python 3.11 or higher is required.**

1. **Clone the repository**

   ```bash
   git clone https://github.com/VijayPant375/Papertrail.git
   cd Papertrail
   ```

2. **Create your environment file**

   ```bash
   cp .env.example .env
   ```

3. **Add your Gemini API key**

   Open `.env` and set:

   ```
   GEMINI_API_KEY=your_key_here
   ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Start the server**

   ```bash
   python main.py
   ```

6. **Open the app**

   Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

---

## Video Demo

[Demo video link — to be added]

---

## Approach

**Two-pass LLM pipeline.** Rather than asking the model to extract facts and compare them in a single prompt, the system separates these into two distinct passes. The first pass — extraction — focuses the model on a single page chunk and asks it to produce structured, evidence-backed facts. The second pass — comparison — takes a pre-filtered pair of facts from different documents and asks only whether they corroborate, contradict, or reconcile. Splitting these passes keeps each prompt narrow and auditable, dramatically reduces hallucination, and makes failures easier to isolate and debug.

**Dynamic schema — no hardcoded fact types.** The extraction prompt instructs the LLM to infer the appropriate `fact_type` from context rather than selecting from a fixed list. An economic survey produces facts of type `gdp_growth_rate` and `fiscal_deficit`; a corporate annual report produces `revenue`, `ebitda_margin`, and `headcount`. New document categories produce new fact types automatically, without any code changes. This generalises the system to arbitrary PDF domains.

**Embedding-based candidate selection.** Comparing every fact from a new document against every fact in the corpus would require O(n²) LLM calls — prohibitively expensive. Instead, each extracted fact is embedded using `gemini-embedding-002` with `SEMANTIC_SIMILARITY` task type, and the embedding is stored as a `float32` byte blob in SQLite. When a new document is processed, cosine similarity is computed in-memory across all stored embeddings, and only pairs exceeding a threshold of 0.75 are forwarded to the LLM comparison pass. This reduces expensive model calls by orders of magnitude on any non-trivial corpus. The storage layer is SQLite with the `sqlite-vec` extension — zero additional services, fully inspectable with any SQLite browser, and trivially portable.

**Evidence-first design.** Every fact row stores a verbatim `evidence_quote` from the source text and the `evidence_page` number it was found on. Nothing is orphaned from its source. When the comparison pass identifies a relationship, it records not just the relationship type and confidence but an `explanation` field containing the model's reasoning. The result is a fully auditable chain from claim → quote → page → document, across every relationship stored.

---

## The Four Cases

### 1. Corroboration

India's GDP growth rate for FY2025 is reported across multiple documents in the macroeconomy dataset — the Economic Survey, the RBI Annual Report, and the IMF Article IV Consultation. The system extracts a `gdp_growth_rate` fact from each document independently, embeds them, detects high cosine similarity between the three fact vectors, and classifies their relationship as **corroborates** after the LLM comparison pass confirms the figures align. See demo for the specific values extracted.

### 2. Contradiction

The IMF and RBI publish GDP growth forecasts for FY2026 that differ from each other. The system extracts a `gdp_growth_forecast` fact from each document, identifies the pair as a candidate via embedding similarity, and classifies the relationship as **contradicts** — flagging the discrepancy in the explanation field with the two differing figures and their respective sources. See demo for the specific values extracted.

### 3. Contextual Reconciliation

Delhivery's revenue figure appears in two documents with different unit conventions: the Annual Report states it in ₹ million while the Q4 Earnings Release states it in ₹ Crore. The raw numbers therefore appear different. The system detects the high semantic similarity between the two `revenue` facts, passes them to the LLM comparison pass, and correctly classifies the relationship as **reconcilable** — the explanation notes the unit discrepancy and confirms the underlying figures are equivalent once the conversion is applied.

### 4. Extraction Failure

Tables with merged cells or multi-line numeric values — common in financial PDFs — sometimes cause the extracted fact's `value` field to be split across lines or truncated. The system handles this gracefully: such facts are assigned a low `confidence` score and their `evidence_quote` is flagged as partial in the extraction output. They are stored rather than discarded, so they appear in the output for human review, but the low confidence score allows downstream consumers to filter or deprioritise them.

---

## Limitations and Next Steps

**Current limitations:**

- **Synchronous processing** — PDF extraction and all LLM calls run in the same thread as the upload request. Large documents block the endpoint for the duration of processing. An async task queue (Celery or ARQ) with a polling endpoint would fix this.
- **Embedding token limit** — `gemini-embedding-002` accepts up to 2048 tokens. Very long evidence quotes are truncated before embedding, which can reduce the accuracy of similarity matching for verbose passages.
- **Similarity threshold hard-cuts** — Candidate pair selection only considers facts with cosine similarity ≥ 0.75. Facts that express the same claim using very different language or vocabulary may fall below this threshold and never reach the comparison pass.
- **No intra-document deduplication** — If the same fact appears on multiple pages of a single document (e.g. a headline figure repeated in an executive summary and a detailed table), it is extracted and stored multiple times.

**Next steps:**

- Async processing with background workers and a progress-polling API so large uploads don't time out.
- Fact deduplication within a document using embedding similarity before storage.
- A graph visualisation of the relationship network — nodes are facts, edges are relationship types — rendered in the frontend.
- Support for scanned PDFs via an OCR pre-processing step (e.g. Tesseract or a vision model) before text extraction.

---

## Additional Notes

Both starter datasets — Delhivery corporate documents and India macroeconomy reports — are included in the repository under `sample_output/` as pre-run JSON exports. Evaluators can inspect extracted facts and detected relationships without running the pipeline. The JSON files contain the full fact rows including `evidence_quote`, `evidence_page`, `confidence`, and all relationship records with `explanation` fields.

The system does not rely on any document-specific rules, hardcoded entity names, or fixed schemas. Every fact type, subject, and predicate is inferred by the LLM from the document content. Uploading an entirely different class of document — a legal contract, a clinical trial report, a municipal budget — will produce a coherent, structured fact graph with no configuration changes.
