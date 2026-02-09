# LitScribe Tests

## Overview

All tests are standalone scripts that can be run directly with `python tests/<test_file>.py`.
No pytest or external test runner required.

## Test Files

### Core Infrastructure

| File | Module | Tests | Phase |
|------|--------|-------|-------|
| `test_cache_manual.py` | Cache DB / Paper Cache / Search Cache | Cache database CRUD, paper caching, search caching, async operations | 8 |
| `test_discovery_cache.py` | Discovery Agent + Cache | State creation, cache-enabled search, discovery agent integration | 8 |
| `test_critical_reading_cache.py` | Critical Reading Agent + Cache | PDF caching, parse caching, cached_tools parameter in agents | 8 |
| `test_checkpointing.py` | LangGraph Checkpointing | Checkpoint imports, SQLite saver, graph compilation, state fields, `run_literature_review` signature, **ablation flags** | 8.6 + 9.5 |
| `test_exporters.py` | BibTeX / Citation / Pandoc | BibTeX export, citation formatting, Markdown generation | 8 |
| `test_graphrag.py` | GraphRAG Pipeline | State types, entity normalization, graph building, community detection, supervisor routing, workflow routing, **tracker params**, **retry logic, threshold clustering** | 7.5 + 9.5 + 10 |

### Phase 9.5 - Evaluation & Instrumentation

| File | Module | Tests | Count |
|------|--------|-------|-------|
| `test_token_tracker.py` | `TokenTracker` | Init, record, multi-agent, multi-model, cost estimation (Opus vs Sonnet), fuzzy model matching, None/missing usage, to_dict, CLI formatting, elapsed time | 13 |
| `test_citation_grounding.py` | `citation_grounding` | Citation extraction (simple, et al., two authors, multi-citation, scattered), grounding (all matched, ungrounded, no papers, no citations), author format handling, deduplication, string authors | 15 |
| `test_evaluator.py` | `ReviewEvaluator` | Import, search quality, anti-keywords, theme coverage (full/partial), domain purity (clean/contaminated), grounding evaluation, content quality, self-review extraction, efficiency, failure detection (search/theme/hallucination), overall score, report formatting | 18 |

## Running Tests

### Run a single test file
```bash
python tests/test_token_tracker.py
python tests/test_citation_grounding.py
python tests/test_evaluator.py
```

### Run all tests
```bash
for f in tests/test_*.py; do echo "=== $f ==="; python "$f"; echo; done
```

### Run all Phase 9.5 tests only
```bash
python tests/test_token_tracker.py && \
python tests/test_citation_grounding.py && \
python tests/test_evaluator.py
```

## Test Results Summary

As of Phase 10 completion:

| File | Status | Notes |
|------|--------|-------|
| test_cache_manual.py | All Pass | No Phase 9.5 impact |
| test_discovery_cache.py | All Pass | No Phase 9.5 impact |
| test_critical_reading_cache.py | All Pass | No Phase 9.5 impact |
| test_checkpointing.py | **6/6 Pass** | New Test 6 added for ablation flags |
| test_exporters.py | All Pass | No Phase 9.5 impact |
| test_graphrag.py | **10/10 Pass** | Tests 9-10 added for Phase 10 (retry logic, threshold clustering) |
| test_token_tracker.py | **13/13 Pass** | New file |
| test_citation_grounding.py | **15/15 Pass** | New file |
| test_evaluator.py | **18/18 Pass** | New file |

## Architecture Notes

- Tests use `sys.path.insert(0, ...)` to add `src/` to Python path
- Each test file has a `main()` function that runs all tests and reports results
- Tests return exit code 0 on success, 1 on failure
- No external mocking library needed - tests use real module imports and pure function testing
- LLM-dependent tests (e.g. `call_llm` integration) are not included as they require API keys and incur costs
