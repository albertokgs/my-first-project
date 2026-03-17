# DocForge — Planning Context from Prior Chat

## Overview

I've been planning a personal PDF management system called **DocForge** (working name — open to suggestions). It has three modules sharing core infrastructure. This document summarizes architectural decisions and analysis from a prior planning conversation.

---

## Three-Module Architecture

### Module 1: `pdf-nav` — Technical/Academic PDF Library
- **Scope:** 5,000+ technical PDFs — books, papers, articles, references.
- **Sensitivity:** None. Can use cloud APIs (Claude API, etc.) freely.
- **Purpose:** Navigation, search, metadata extraction, semantic search, author/theme taxonomy, duplicate detection.
- **Key features:**
  - File renaming to standardized format: `"LastName1, LastName2 - Title Case Title.pdf"`
  - Duplicate and near-duplicate detection (hash-based + embedding cosine similarity)
  - Version tracking (different editions/versions of same work)
  - Language detection and grouping (Portuguese/English pairs)
  - Author disambiguation (handling initials, name variants — flagged as medium-high difficulty)
  - Multi-level theme taxonomy (e.g., `math/physics`, `applied_math/gravity`, `FFT/Hawking_radiation`)
  - Semantic search using embeddings
  - Document summarization stored for search augmentation
  - Star/rating system: 1 star (good), 3 stars (exceptional), -1 star (dislike), suppress-from-search flag
  - Statistics: top authors, theme distributions, embedding quality visualization
  - Cross-reference graph (later phase): works referencing other works, "learn more" threading
  - Metadata scraping: authors, dates, version/edition, and more
- **Future integration points:** Will connect to a physical book inventory system, a news/article hyperlink organizer, and a reference/podcast/video link organizer (all separate projects in planning).

### Module 2: `pdf-vault` — Personal Document Organizer (Shared with Wife)
- **Scope:** Bank statements, credit card statements (5-6 cards), health documents (exams, invoices, notas fiscais), large purchase receipts, insurance docs, etc.
- **Sensitivity:** HIGH. Must be fully local/offline. No cloud API calls for document content.
- **Purpose:** Classification, filing into structured folders, data extraction → tabular exports.
- **Key features:**
  - Inbox watcher: drop a PDF → system classifies, renames, files it
  - Rule-based classification first (bank/credit card statements have predictable layouts), LLM fallback via Ollama for ambiguous cases
  - Folder structure: `archive/{category}/{subcategory}/{year}/` (e.g., `credit_cards/nubank/2025/`)
  - Per-person tagging (household members)
  - Data extraction: amounts, dates, vendors, categories
  - Validation/review queue: system proposes classification + extracted data, user confirms
  - Excel/CSV export for spending visualization
  - Ingestion of older Excel spreadsheets over time to build historical data
  - Shared access with wife (read/browse/search on classified documents)

### Module 3: `pdf-ledger` — Tax & Financial Planning (Private)
- **Scope:** Same source documents as Module 2, but with private analytical layer.
- **Sensitivity:** HIGHEST. Private to me only. Fully local.
- **Purpose:** Spending pattern analysis, tax planning, financial modeling.
- **Key features:**
  - Spending habit analysis over time
  - Tax optimization (Brazilian tax context — notas fiscais, deductions)
  - Financial dashboards and reports
  - Private tables and analysis not visible in Module 2
  - Reads from shared database but has independent access-scoped views and additional analytical tables

### Module Relationship
- Modules 2 and 3 share the same ingestion pipeline, database, and PDF extraction core.
- Module 3 adds analytical tables and logic on top of Module 2's data, with access scoping.
- Module 1 is conceptually independent but shares core infrastructure (PDF extraction, embedding engine, database layer).
- The split between 2 and 3 is maintained primarily for access control (shared vs. private), not just privacy from the internet.

---

## Architectural Decisions (Agreed or Under Discussion)

### Compute & Privacy Model
- **Module 1:** Can use Claude API (Haiku for metadata/summarization — cheap, effective). Embeddings generated locally regardless.
- **Modules 2 & 3:** Fully local. Rule-based classification + Ollama (Llama 3.1 70B or Qwen 2.5 72B on DGX Spark) for LLM tasks. No document content leaves the machine.
- **Hybrid approach** is the consensus: local for sensitive, API for non-sensitive.

### Database
- Under discussion: **SQLite** (simpler, portable, good start) vs. **PostgreSQL** (pgvector for embeddings, learning goal).
- Recommendation from prior chat: start SQLite with clear migration path to Postgres, OR go straight to Postgres if learning it is a priority.

### Embeddings & Vector Search
- Embedding models under consideration: `all-MiniLM-L6-v2` (fast), `e5-large-v2` or `bge-large` (better quality).
- Vector store options: ChromaDB (simple, file-based), pgvector (if Postgres), FAISS (raw performance).
- All embedding generation runs locally.

### GUI / Interfaces
- **Phase 1:** CLI (Click or Typer)
- **Phase 2:** GUI — Streamlit or Gradio (web-based, easy) or Textual (TUI in terminal)
- **Future:** Web interface (FastAPI) for remote access
- TUI noted as a later-phase addition.

### Excel / Spreadsheet Strategy
- **Do NOT use Excel/Numbers as system of record.** Database is the source of truth.
- Generate Excel/CSV exports on demand using `openpyxl`.
- Excel for Mac has improved but historically problematic. Recommendation: use Excel exports viewable on either Mac or Windows, with Windows as fallback for heavy editing.
- Existing Numbers files can be batch-converted to xlsx/csv using `numbers-parser` Python library.

### Inbox / Ingestion Pipeline
- `watchdog` Python library for filesystem event monitoring on the inbox directory.
- Should accumulate work into a review queue — user processes in batches.
- Goal: near-zero-shot reliability. Drop file → process → review → done.

### OCR
- Options: Tesseract (standard), EasyOCR (better multilingual — important for Portuguese), Surya (newer, high quality).
- Multilingual support is important (Portuguese + English documents).

### PDF Extraction Stack
- `pdfplumber` and/or `PyMuPDF` for text extraction.
- OCR fallback for scanned documents.
- GROBID for academic paper metadata extraction (Module 1).

---

## Proposed Phasing

- **Phase 0 — Foundation:** Project scaffolding, DB schema, core PDF extraction, basic CLI.
- **Phase 1 — Personal Vault (Module 2):** Rule-based classifier, inbox watcher, filing, review queue, Excel export.
- **Phase 2 — Technical Library (Module 1):** GROBID + LLM metadata extraction, embeddings, semantic search, author extraction, renaming.
- **Phase 3 — Enrichment:** Theme taxonomy, duplicate detection, statistics, star/rating, GUI/TUI.
- **Phase 4 — Financial (Module 3):** Spending analysis, tax planning, private dashboards.
- **Phase 5 — Future:** Web interface, cross-reference graph, integration with book inventory and hyperlink organizer projects.

---

## Development Environment
- Primary: macOS (MacBook)
- Secondary: Windows laptop (Dell)
- Local compute: DGX Spark (for Ollama / local LLM inference)
- IDE: VS Code
- Language: Python
- Development companion: Claude Code for implementation
- Planning: Claude browser (this chat)

---

## Open Questions for This Chat
1. Final database choice: SQLite vs. Postgres from the start?
2. Embedding model selection — benchmark before committing?
3. GUI framework preference?
4. Exact folder taxonomy for Module 2 (what document categories exist)?
5. How to handle the Module 2/3 access scoping in practice (separate DB views? separate apps? auth layer?)
6. Project naming — DocForge is working name. Alternatives discussed: `paperhub`, `archivum`, `papergraph`.
