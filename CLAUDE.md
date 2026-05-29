# Buddhist RAG Q&A System - Project Summary

## Project Overview

Local RAG (Retrieval-Augmented Generation) knowledge base system for Buddhist teachings Q&A.
- **Embedding Model**: BGE-M3 (via sentence-transformers)
- **Vector Search**: FAISS
- **API Framework**: FastAPI + Uvicorn
- **Tokenization**: Jieba (Chinese word segmentation)
- **LLM**: OpenAI-compatible API (optional)

---

## Completed Features

### Core RAG Pipeline
- [x] BGE-M3 embedding model local inference
- [x] FAISS vector index build & search
- [x] Hybrid search (vector + BM25 keyword)
- [x] FAQ fast-path matching
- [x] Query rewriting & synonym expansion
- [x] Route-based question classification (doctrine/practice/scripture/concept/sect/history/ritual/basic)
- [x] Cross-encoder rerank
- [x] Answer formatting (structured output with bullet points)
- [x] Media attachment router (image/video suggestions)
- [x] Clarification engine (disambiguation for ambiguous questions)
- [x] Keyword extraction (Jieba + TF-IDF)
- [x] Rate limiting, caching, middleware

### Knowledge Base
- [x] Multi-file & sub-directory knowledge organization
- [x] 10 content types supported: hierarchical commentary (科判), chapter-based (品), ritual (仪轨), Q&A, lecture (演讲), methods/steps, koans (公案), annotated texts (注疏), verse collections (偈颂), letters (书信)
- [x] Buddhist main knowledge base (`knowledge/buddhism/main.txt` — 221 lines)
- [x] FAQ pairs (`faq.txt` — 25 Q&A pairs)
- [x] Term aliases (`alias.txt` — 30 synonym groups)
- [x] 入行论 (Bodhicaryavatara) 10 chapters + FAQ

### Admin Dashboard (admin_page.html)
- [x] Phase 1A: HTML framework + CSS + Tab navigation
- [x] Phase 1B: Login auth + apiFetch wrapper
- [x] Phase 1C: Dashboard tab — stats cards + product list
- [x] Phase 2A: Index management tab
- [x] Phase 2B: Logs tab — Q&A / miss / error logs
- [x] Phase 3A: Knowledge tab — product select + file list
- [x] Phase 3B: File edit modal — view/save/delete
- [x] Phase 3C: Create/delete product + file upload
- [x] Phase 4A: Synonym management — list + add + review + delete
- [x] Phase 4B: Synonym batch operations + import/export
- [x] Phase 4C: Synonym expansion test
- [x] Phase 5A: Keyword tab — active terms + synonym override CRUD
- [x] Phase 5B: Disambiguation rules CRUD
- [x] Phase 5C: LLM smart expansion + history
- [x] Phase 6A: Knowledge import tab — text import + LLM preview
- [x] Phase 6B: URL crawl import
- [x] Phase 6C: Refine + commit workflow
- [x] Phase 7A: System tab — LLM model config
- [x] Phase 7B: Service management — Embedding/LLM start/stop panel

### Content Import Pipeline
- [x] File upload (.doc/.docx) with preview/edit
- [x] URL auto-import (one-click URL-to-knowledge)
- [x] Batch URL pattern import for series content
- [x] Auto-crawl: discover all series links and bulk import
- [x] Crawl v2: background jobs, SSE progress, resume support
- [x] Per-link progress table in crawl UI
- [x] Stop/restart/retry controls for crawl jobs
- [x] Lecture type for full transcript import
- [x] Timeout & retry for LLM API calls during crawl

### Bug Fixes & Hardening (94 total commits)
- [x] P0: Auth test mismatch fix + path traversal defense
- [x] P1: Persist LLM start state + cache management UI + service tests
- [x] P2: Debug/server config UI + knowledge & synonym tests
- [x] P3: Remove dead routes + mobile.html error handling
- [x] 30+ critical bug fixes across routing, concurrency, thread safety, scoring, SSRF, error handling, memory, encoding

### Modified Files (Key)
| File | Lines | Description |
|------|-------|-------------|
| `api_server.py` | ~161K | FastAPI server, all admin endpoints |
| `admin_page.html` | ~174K | Single-page admin dashboard |
| `rag_answer.py` | ~67K | Main RAG Q&A pipeline |
| `search_utils.py` | ~63K | Text processing, hybrid search |
| `build_faiss.py` | ~32K | Index builder for all content types |
| `rag_runtime_config.py` | ~31K | Central config, routes, thresholds |
| `import_knowledge.py` | ~35K | Knowledge import with LLM refinement |
| `query_rewrite.py` | ~26K | Query expansion/rewriting |
| `mobile.html` | ~30K | Mobile chat frontend |
| `crawl_v2.py` | ~20K | Background crawl jobs with SSE |
| `keyword_extractor.py` | ~21K | Keyword extraction engine |
| `clarification_engine.py` | ~20K | Disambiguation logic |
| `keyword_store.py` | ~15K | Keyword persistence |
| `llm_client.py` | ~14K | LLM service client |
| `test_api.py` | ~24K | API endpoint tests |

---

## Pending Tasks

### Phase: Testing (P0-P2)
- [ ] Fix `test_admin_query_param_auth` — tests query param auth but backend only supports Bearer header
- [ ] Add tests for service management endpoints (5 endpoints, 0 coverage)
- [ ] Add tests for knowledge management endpoints (6 endpoints, 0 coverage)
- [ ] Add tests for import workflow endpoints (3 endpoints, 0 coverage)
- [ ] Add tests for synonym operation endpoints (7 endpoints, 0 coverage)
- [ ] Add tests for other untested endpoints (`rebuild_shared`, `upload`, `upload_zip`, `keywords/llm-expand`)

### Phase: Frontend Completeness (P1-P2)
- [ ] System tab: add cache management card (backend `/admin/cache` + `/admin/cache/clear` exist)
- [ ] System tab: add debug toggle (backend `/admin/debug` exists)
- [ ] System tab: add server config UI (backend `/admin/config/server` exists)
- [ ] mobile.html: add error toast for `loadProducts()` silent failure
- [ ] mobile.html: add fetch timeout mechanism

### Phase: Orphan API Cleanup (P3)
- [ ] Evaluate whether to keep or remove orphan endpoints:
  - `/admin/debug` POST
  - `/admin/config` GET/POST
  - `/admin/config/model` GET/POST
  - `/admin/config/server` GET/POST
  - `/admin/config/nginx` POST
  - `/admin/cache` GET
  - `/admin/cache/clear` POST
  - `/admin/import_knowledge_file` POST
  - `/admin/synonyms/learned` GET/PUT

### Phase: Knowledge Base Gaps
- [x] **Buddhism & Daily Life** — work stress, relationships, diet, wealth, mental health (added `life` route + knowledge + FAQ)
- [x] **Comparative Religion** — Buddhism vs Taoism, Christianity, Confucianism (added in main.txt section 12)
- [x] **Misunderstandings/Myths** — "消极避世" / "迷信" / "烧香求保佑" / "神通感应" (added section 11)
- [x] **Specific Practice Guidance** — beginner book recommendations, home altar setup (added section 12)
- [x] **Vegetarianism/Diet** — detailed treatment of Buddhist dietary rules (added in life section)
- [x] **Dreams/Supernatural** — dreams, spiritual experiences, proper attitude (added in section 11)
- [ ] **Modern Buddhist Figures** — more on contemporary teachers and organizations

### Phase: RAG Quality
- [ ] Optimize chunking strategy (from original PROGRESS.md)
- [ ] Improve retrieval accuracy
- [ ] Add reranker improvements
- [ ] Video knowledge base support
- [ ] Production deployment

---

## Technical Decisions & Notes

### Architecture
- **Single-file admin dashboard**: `admin_page.html` is a monolithic SPA (~174K). All tabs, modals, and logic in one file.
- **No frontend framework**: Pure vanilla HTML/CSS/JS for zero build dependencies.
- **In-process embedding model**: BGE-M3 loaded once in memory, shared across requests. No external embedding service needed.
- **Hybrid search**: Vector similarity (FAISS) + BM25 keyword scoring, combined with configurable weights.

### Configuration
- All runtime config in `rag_runtime_config.py` with environment variable overrides.
- Question routing uses keyword matching against `QUESTION_ROUTES` dict (8 routes: doctrine, practice, scripture, concept, sect, history, ritual, basic).
- Thresholds tunable via admin API at runtime.

### Known Issues
- `start_llm_service()` does not persist state to disk, while `stop_llm_service()` does — restart defaults LLM to off.
- `/admin/service/embedding/start` may timeout if model loading is slow (no async).
- LLM test endpoint creates new OpenAI client each time instead of reusing cached one.
- 26 admin endpoints have zero test coverage.
- Windows: Chinese characters in `print()` can cause `UnicodeEncodeError` (partially fixed).

### Knowledge Base Format
- UTF-8 `.txt` files only
- Structural markers (科判 headings, Q/A tags, verse markers) must be on separate lines
- No mixed formatting or Markdown inside knowledge files
- One work/topic per file
- Format spec documented in `docs/知识库格式规范.md`

### Question Routes
| Route | Coverage | Examples |
|-------|----------|---------|
| `doctrine` | Four Noble Truths, Eightfold Path, Twelve Links, Three Marks | "四圣谛是什么" |
| `practice` | Meditation, chanting, Pure Land, precepts, daily practice | "如何开始禅修" |
| `scripture` | Heart Sutra, Diamond Sutra, Lotus Sutra, Bodhicaryavatara | "心经讲什么" |
| `concept` | Emptiness, Nirvana, Karma, Samsara, Bodhicitta | "什么是空性" |
| `sect` | Zen, Pure Land, Tiantai, Tibetan, Theravada | "佛教有哪些宗派" |
| `history` | Chinese Buddhism history, key figures, Eight Schools | "玄奘的贡献" |
| `ritual` | Ceremonies, festivals, etiquette, offerings | "盂兰盆节是什么" |
| `basic` | General intro, Buddha's life, getting started | "什么是佛教" |
| `life` | Buddhism & modern life, stress, relationships, diet, myths, comparisons, beginner guidance | "佛教怎么看待婚姻" |

### Regression Test Coverage
- 15 test cases in `regression_cases.json` covering all 8 existing routes
- `run_regression.py` for automated quality checks
- `test_accuracy.py`, `test_api.py`, `test_core.py` for unit/integration tests
