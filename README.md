<div align="center">

# 🔎 Papertrail

### Cross-Document Fact Extraction & Relationship Detection System

**Upload PDFs → Extract grounded facts → Connect evidence across documents → Find what agrees, conflicts, or only appears to conflict**

</div>
---

## 🎥 Video Demo

> **Demo video — coming soon**

The final demo will show:

* A PDF being processed through the pipeline
* Extracted facts with source evidence and page references
* A corroborated fact across documents
* A genuine contradiction / reasoning failure discovered during testing
* An apparent contradiction reconciled through context and units
* An extraction failure and how the system handles or documents it

---

## ✨ What is Papertrail?

Important facts rarely live in isolation.

The same claim can appear across multiple reports, be expressed using different wording, use different units, or genuinely conflict with another source. **Papertrail** is a prototype Fact Knowledge Layer designed to turn scattered PDF information into structured, evidence-backed facts and identify meaningful relationships between them.

For every extracted fact, Papertrail keeps track of:

* 📄 **Where it came from**
* 📖 **The source evidence**
* 📍 **The page number**
* 🏷️ **The type of fact**
* 🧩 **Its contextual information**
* 🔗 **Its relationship to facts in other documents**

The goal is not simply to extract information or visualize a graph. The focus is on the full chain:

> **Document → Fact → Evidence → Candidate Match → Relationship → Explanation**

---

## 🚀 Features

### Core

* 📄 **PDF Fact Extraction** — Extracts meaningful numerical and semantic facts from PDF documents
* 🧠 **LLM-Powered Understanding** — Uses Gemini to convert unstructured document content into structured facts
* 🔍 **Evidence-First Facts** — Every fact is linked to a source quote and page number
* 🏷️ **Dynamic Fact Types** — Fact types are inferred from document content instead of using a fixed schema
* 🔗 **Cross-Document Comparison** — Finds related facts across different documents
* 🤝 **Corroboration Detection** — Identifies facts that support the same claim
* ⚠️ **Contradiction Detection** — Designed to identify conflicting claims from different sources
* 🔄 **Contextual Reconciliation** — Handles apparent conflicts caused by units, periods, scope, or context
* 💾 **Persistent Knowledge Layer** — Stores documents, facts, relationships, and processing information in SQLite

### Engineering Decisions

* ⚡ Candidate selection prevents unnecessary comparison of every fact against every other fact
* 📦 SQLite keeps the prototype self-contained with no external database service
* 🔎 Every relationship is designed to remain traceable back to source evidence
* 🧩 Facts and relationships are stored separately so new documents can be added incrementally

---

## 🏗️ Architecture

```text
Papertrail/
│
├── main.py                       FastAPI application
│
├── pipeline/
│   ├── extractor.py              PDF text and table extraction
│   ├── fact_finder.py            LLM-powered fact extraction pipeline
│   ├── comparator.py             Cross-document fact comparison
│   └── prompts.py                Extraction and comparison prompts
│
├── db/
│   ├── schema.py                 SQLite schema and initialization
│   └── queries.py                Database access layer
│
├── ui/
│   └── index.html                Single-page frontend
│
├── sample_pdfs/                  Example PDF dataset
│
├── sample_output/                Space for exported sample results
│
├── facts.db                      Local knowledge-layer database
│
├── requirements.txt
└── .env.example
```

### 🔄 Processing Flow

```text
                    ┌─────────────────┐
                    │   Upload PDF    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ PDF Extraction  │
                    │ pdfplumber      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Page Chunks +   │
                    │ Extracted Tables│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Gemini Fact     │
                    │ Extraction      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Structured Facts│
                    │ + Evidence      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Candidate       │
                    │ Selection       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Cross-Document  │
                    │ Comparison      │
                    └────────┬────────┘
                             │
                             ▼
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     Corroborates       Contradicts        Reconcilable
```

---

## 🛠️ Tech Stack

| Layer                  | Technology            |
| ---------------------- | --------------------- |
| Backend                | FastAPI               |
| Language               | Python                |
| PDF Processing         | pdfplumber            |
| AI / LLM               | Google Gemini API     |
| AI SDK                 | google-genai          |
| Database               | SQLite                |
| Vector Support         | sqlite-vec            |
| Frontend               | HTML, CSS, JavaScript |
| Environment Management | python-dotenv         |

---

## 🧠 Approach

### 1. Two-Stage Fact Pipeline

Papertrail separates **fact extraction** from **fact comparison**.

Instead of asking a single LLM prompt to read an entire corpus and simultaneously discover relationships, the pipeline breaks the problem into smaller stages.

**Stage 1 — Extraction**

Each PDF is processed page by page. Extracted text and tables are passed to the LLM, which produces structured facts grounded in the document.

Each fact contains fields such as:

```text
subject
predicate
value
value_normalized
fact_type
context
evidence_quote
evidence_page
confidence
```

**Stage 2 — Comparison**

Facts from different documents are selected as potential candidates and passed to a separate comparison step.

The comparison prompt focuses on one question:

> Do these two facts corroborate each other, contradict each other, or can their difference be reconciled through context?

This separation makes the pipeline easier to inspect and debug. Extraction failures can be distinguished from comparison failures instead of being hidden inside one large model call.

---

### 2. Dynamic Fact Representation

The system does not use document-specific extraction rules.

There is no hardcoded list such as:

```text
revenue
GDP
EBITDA
headcount
inflation
```

Instead, the extraction pipeline asks the model to infer the appropriate `fact_type`, subject, predicate, value, and context from the document itself.

For example:

```text
Corporate Report
    ↓
revenue
EBITDA
operating_margin

Economic Report
    ↓
GDP_growth
inflation
fiscal_deficit
```

This makes the schema flexible enough to process different categories of documents without requiring a new database design for every domain.

The extracted facts are stored using a general structure rather than a document-specific schema.

---

### 3. Evidence-First Design

A fact without provenance is difficult to trust.

Every extracted fact stores:

* 📖 `evidence_quote`
* 📍 `evidence_page`
* 📄 Source document information
* 🎯 Extraction confidence

Relationships between facts also store:

* 🔗 Relationship type
* 🧠 Explanation
* 🎯 Comparison confidence

This creates an auditable path back to the original source:

```text
Relationship
      ↓
Fact A ─────── Fact B
  ↓                ↓
Evidence         Evidence
  ↓                ↓
Page + PDF       Page + PDF
```

The UI and API are therefore intended to expose not just a conclusion, but the evidence behind it.

---

### 4. Candidate Selection Before LLM Comparison

A naive implementation would compare every fact against every other fact.

For `n` facts, that quickly becomes expensive:

```text
All-pairs comparison → O(n²)
```

Papertrail therefore uses a candidate-selection stage before expensive LLM comparison.

The intended architecture uses semantic embeddings and cosine similarity to identify potentially related facts, allowing unrelated facts to be filtered out before the reasoning stage.

This is an important trade-off:

> **Spend cheaper computation on narrowing the search space before spending expensive LLM calls on reasoning.**

The implementation stores embeddings alongside facts as vector data in SQLite.

---

### 5. Incremental Knowledge Layer

Documents are stored independently from their extracted facts and relationships.

The database contains separate layers for:

```text
documents
    ↓
facts
    ↓
relationships

+ processing_log
```

This means the architecture is designed so that new PDFs can be added to the knowledge layer without rebuilding the entire corpus from scratch.

---

## 📊 Current Dataset

The current pre-populated database contains facts extracted from four documents:

### 📦 Delhivery

* Delhivery Annual Report FY24
* Delhivery Q4 FY24 Earnings Presentation

### 🇮🇳 India Macroeconomy

* RBI Annual Report 2024–25
* IMF India Article IV Consultation 2025

Across these documents, the current database contains **1,800+ extracted facts**.

The dataset was intentionally useful for testing the system across different document styles and domains:

```text
Corporate Financial Reports
            +
Macroeconomic Reports
            ↓
Cross-Domain Fact Extraction
```

---

# 🔍 The Four Required Cases

The assignment requires examples of:

1. A corroborated fact
2. A contradiction
3. An apparent contradiction explained by context
4. An extraction or reasoning failure

Papertrail was evaluated against all four cases.

---

## 1️⃣ Corroboration — ✅ Working

A clear corroboration was successfully identified across the Delhivery documents.

Examples in the dataset include matching financial information expressed across:

* Delhivery Annual Report FY24
* Delhivery Q4 FY24 Earnings Presentation

For example, the system identifies corresponding financial facts such as:

* **Revenue: ₹8,142 Cr**
* **EBITDA: ₹127 Cr**

The facts originate from separate documents but describe the same underlying financial information.

The relationship is stored as:

```text
corroborates
```

Each fact remains linked to its own source evidence and page.

### Why this matters

The documents do not need to contain an identical sentence for the system to recognize that they are supporting the same claim. The comparison process operates on structured facts and their context rather than requiring exact string matches.

---

## 2️⃣ Contradiction — ⚠️ Data Found, Comparison Blocked by API Quota

A likely contradiction was successfully located in the extracted data:

> **The RBI and IMF documents contain differing GDP growth forecasts.**

The relevant facts were extracted from the documents and are present in the knowledge layer.

However, the final LLM comparison required to classify the pair as:

```text
contradicts
```

could not complete because the Gemini API quota was exhausted during the comparison phase.

This is intentionally documented as a real limitation rather than presented as a successful result.

### What happened

The pipeline depends on Gemini for:

* Fact extraction
* Embedding generation
* Final relationship reasoning

During testing, repeated quota limits produced `429` responses.

As a result:

```text
Relevant facts extracted
        ↓
Contradiction candidate identified
        ↓
Final LLM comparison
        ↓
❌ API quota exhausted
```

### Why this is included

The assignment explicitly asks for honest handling of failures and reasoning limitations.

Rather than inventing a contradiction classification, Papertrail preserves the extracted evidence and documents that the reasoning stage could not complete under the available API quota.

A production version should use retry logic, rate-limit-aware queues, fallback models, or local embedding/comparison infrastructure.

---

## 3️⃣ Contextual Reconciliation — ✅ Working

A successful reconciliation case was identified in Delhivery financial data.

The same underlying revenue information appears using different numerical representations:

```text
₹814
vs
₹8,142
```

Taken literally, these values appear inconsistent.

However, the discrepancy is explained by **unit context**.

The comparison process identifies that the figures are expressed using different conventions, allowing the apparent contradiction to be classified as:

```text
reconcilable
```

### Why this matters

Not every numerical mismatch is a contradiction.

Facts can differ because of:

* 📅 Different reporting periods
* 🌍 Different scopes
* 📏 Different units
* 📊 Different levels of aggregation

Papertrail explicitly treats contextual reconciliation as a separate relationship category instead of forcing every mismatch into `contradicts`.

---

## 4️⃣ Extraction / Reasoning Failure — ⚠️ Empty Embeddings

During extraction, Gemini API quota exhaustion also affected embedding generation.

Some extracted facts therefore ended up with **empty embedding data**.

This created an important failure mode:

```text
Fact Extraction
      ↓
Embedding Generation
      ↓
❌ Gemini quota exhausted
      ↓
Empty embedding
```

Instead of treating the entire pipeline as invalid, this exposed a weakness in the dependency chain.

The comparison pipeline contains fallback logic for cases where embeddings cannot be used, but final relationship classification still depends on an LLM call. Once the API quota was exhausted, that final stage could also fail.

### What this taught

The current architecture has too much dependence on a single external AI service.

A more resilient system would separate these concerns:

```text
Extraction Model
      +
Local Embedding Model
      +
Independent Comparison / Reasoning Model
```

That would allow one service failure to degrade only part of the pipeline instead of blocking multiple stages.

This was a useful engineering failure discovered during testing and directly informed the proposed next steps.

---

## ⚙️ Local Setup

### Prerequisites

* Python **3.11+**
* Git
* A Gemini API key

---

### 1. Clone the Repository

```bash
git clone https://github.com/VijayPant375/Papertrail.git
cd Papertrail
```

---

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

**Windows**

```bash
.venv\Scripts\activate
```

**macOS / Linux**

```bash
source .venv/bin/activate
```

---

### 3. Create Your Environment File

```bash
cp .env.example .env
```

On Windows, if `cp` is unavailable:

```bash
copy .env.example .env
```

---

### 4. Add Your Gemini API Key

Open `.env` and add:

```env
GEMINI_API_KEY=your_api_key_here
```

> ⚠️ Never commit your `.env` file or API key to GitHub.

---

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 6. Start the Application

```bash
python main.py
```

---

### 7. Open Papertrail

Navigate to:

```text
http://localhost:8000
```

---

## 📡 API Overview

| Method | Endpoint                                 | Description                       |
| ------ | ---------------------------------------- | --------------------------------- |
| `POST` | `/upload`                                | Upload and process a PDF          |
| `POST` | `/compare`                               | Trigger cross-document comparison |
| `GET`  | `/documents`                             | List processed documents          |
| `GET`  | `/documents/{document_id}`               | Inspect a document and its facts  |
| `GET`  | `/documents/{document_id}/relationships` | View document relationships       |
| `GET`  | `/relationships`                         | View all detected relationships   |
| `GET`  | `/health`                                | Check application health          |

---

## 🔐 Configuration

Example environment configuration:

```env
GEMINI_API_KEY=your_api_key_here
DB_PATH=facts.db
```

Credentials are intentionally excluded from the repository through `.gitignore`.

---

# ⚠️ Limitations

Papertrail is a prototype built to explore the Fact Knowledge Layer problem rather than a production-ready document intelligence platform.

### 🤖 External API Dependency

The biggest limitation discovered during development was dependency on Gemini API availability and quota.

The same external service is involved in multiple stages:

* Fact extraction
* Embedding generation
* Relationship reasoning

When quota limits are reached, parts of the pipeline cannot complete.

---

### ⏳ Synchronous Processing

PDF extraction and AI processing currently occur as part of the request workflow.

Large documents may therefore take a significant amount of time to process.

A production architecture should move processing into background workers with:

```text
Upload
  ↓
Job Queue
  ↓
Background Worker
  ↓
Processing Status API
```

---

### 🧮 Candidate Selection Depends on Embeddings

The intended semantic candidate-selection system works best when valid embeddings are available.

If embedding generation fails, similarity-based matching becomes less reliable and the fallback path becomes more important.

A future version should use a local embedding model so candidate selection does not depend on the same API used for extraction and reasoning.

---

### 📄 Scanned PDFs

The current pipeline uses `pdfplumber` and therefore works best with PDFs containing extractable text.

Image-only or scanned PDFs require an additional OCR stage.

---

### 🔁 Duplicate Facts

The same fact can appear multiple times within a document, especially in annual reports where headline figures are repeated across:

* Executive summaries
* Financial statements
* Tables
* Notes

The current prototype does not yet perform robust intra-document fact deduplication.

---

### 🧠 LLM Uncertainty

LLM-generated extraction and comparison are probabilistic.

Confidence scores help expose uncertainty, but they should not be treated as a guarantee of factual correctness.

For high-stakes use cases, relationships should support:

* Human review
* Rule-based validation
* Numerical consistency checks
* Independent verification

---

# 🚧 Next Steps

### 🔌 Decouple the AI Pipeline

Use independent components for:

* Fact extraction
* Embeddings
* Relationship reasoning

This would prevent one provider's rate limits from affecting the entire system.

---

### 🧠 Local Embeddings

Replace API-dependent embeddings with a local model.

Benefits:

* No embedding quota exhaustion
* Faster repeated experimentation
* Lower API cost
* More reliable candidate selection

---

### ⚡ Asynchronous Processing

Move document processing into background workers.

This would allow:

* Large PDFs
* Multiple simultaneous uploads
* Progress indicators
* Retry handling
* Rate-limit-aware scheduling

---

### 🔁 Fact Deduplication

Detect repeated facts within the same document before storing or comparing them.

---

### 🧮 Hybrid Relationship Detection

Combine LLM reasoning with deterministic checks.

For example:

```text
Numeric values
      ↓
Unit normalization
      ↓
Period comparison
      ↓
Scope comparison
      ↓
LLM explanation
```

This could make common financial and numerical reconciliation cases cheaper and more reliable.

---

### 📷 OCR Support

Add OCR for scanned or image-based PDFs.

---

### 🕸️ Relationship Visualization

A graph view could eventually show:

```text
Fact Nodes
    +
Document Nodes
    +
Relationship Edges
```

However, the graph would remain a visualization layer on top of the core system—not a replacement for evidence-backed fact discovery and comparison.

---

# 📝 AI Tools Used

Papertrail uses the following AI technology as part of the implementation:

* **Google Gemini API** — Structured fact extraction and relationship reasoning
* **Gemini embeddings** — Intended semantic candidate selection

AI was used as a component of the pipeline rather than as a replacement for the system architecture.

The surrounding engineering work includes:

* PDF extraction
* Page-level chunking
* Structured storage
* Evidence tracking
* Candidate selection
* Relationship persistence
* API design
* Failure handling and debugging

---

# 💡 Additional Notes

### 🧪 Built as an Engineering Prototype

The goal of Papertrail was to explore the central challenge:

> **How can facts scattered across independent documents be discovered, grounded, compared, and explained?**

The project deliberately prioritizes:

* Understandable architecture
* Evidence and provenance
* Flexible fact representation
* Explicit relationship types
* Honest failure documentation

over production-scale infrastructure.

---

### 📚 No Document-Specific Rules

The pipeline does not depend on:

* Hardcoded filenames
* Hardcoded entity names
* Fixed document schemas
* Document-specific extraction rules

The same general pipeline can be applied to new PDFs through the upload interface.

Different documents may produce different fact types based on their content.

---

### 💾 Pre-Populated Database

The repository includes a pre-populated local knowledge layer containing results from the current test dataset.

This allows the extracted facts and existing relationships to be inspected without requiring the evaluator to repeat the full API-intensive processing run.

This is particularly important because the prototype currently depends on an external Gemini API with usage limits.

---

### 🎯 Known Relationship Results

At the time of submission, the populated database contains successfully stored:

* **Corroboration relationships**
* **Reconcilable relationships**

The contradiction candidate and embedding failure are documented honestly as outcomes discovered during testing when the external API quota prevented the final comparison stage from completing.

---

### 🔮 What I Would Build With More Time

The highest-priority improvement would be making the pipeline resilient to external AI service limits.

My next version would focus on:

```text
Local Embeddings
        ↓
Reliable Candidate Retrieval
        ↓
Queued LLM Reasoning
        ↓
Retry + Rate Limit Handling
        ↓
Deterministic Numeric Validation
```

That would preserve the core idea of Papertrail while making it substantially more reliable for larger document collections.

---

## ❤️ Built for the Superjoin Engineering Intern Assignment

**Papertrail** is an exploration of how a useful Fact Knowledge Layer can be built from unstructured documents.

The project is intentionally honest about what worked, what failed, and what those failures revealed about the architecture.

> **Facts are only useful when you can trace them back to evidence — and relationships are only useful when you can explain why they exist.**

Built by **Vijay** 🚀
