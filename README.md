# SGC-ISynKGR: Semantic Graph-Calibrated Industrial Schema Mapping

This repository is a clean research fork of the original ISynKGR benchmark framework. It keeps the original benchmark structure, but the main contribution is a new model:

```text
semantic_graph_calibrated
```

The goal is to improve industrial cross-standard schema/path mapping by combining semantic label normalization, graph/context matching, compatibility checks, retrieval support, and calibrated confidence-based no-match rejection.

## Main contribution

The new model adds a **Semantic Graph-Calibrated Ranker** for industrial semantic interoperability.

Instead of relying only on fixed heuristic ranking, the model estimates a calibrated mapping confidence from multiple evidence channels:

```text
source node
+ target candidate
+ semantic label tokens
+ graph/context tokens
+ datatype compatibility
+ unit compatibility
+ rule support
+ retrieval support
+ signal-family match
+ ambiguity penalty
→ calibrated confidence
→ accepted mapping or no_match
```

This is designed to reduce unsafe false-positive mappings, especially when target candidates are ambiguous or semantically close but incorrect.

## Repository structure

```text
benchmark/      Benchmark runner, scenarios, metrics, orchestration
isynkgr/        Translation core, adapters, retrieval, rules, ranking
isynkgr/pipeline/adaptive_candidate_ranker.py
                Includes the semantic_graph_calibrated model implementation
docs/           Architecture, benchmark, and model documentation
tests/          Unit and benchmark contract tests
```

## Important scenario

Run the new model directly:

```bash
PYTHONPATH=. python -m benchmark.run \
  --scenario semantic_graph_calibrated \
  --out results/semantic_graph_calibrated
```

Run the full benchmark workflow:

```bash
PYTHONPATH=. python -m benchmark.orchestrate
```

Or with Docker:

```bash
docker compose up --build full-run
```

## Canonical scenario set

```text
full_framework
rule_based_only
llm_only
rag_only
embedding_similarity
semantic_graph_calibrated
ablation_no_rules
ablation_no_retrieval
ablation_no_llm
```

See `docs/SCENARIO_MATRIX.md` for the scenario definitions.

## Why keep this as a separate repository?

Keeping this model in a new repository makes the contribution cleaner:

1. The original ISynKGR repository remains a baseline benchmark.
2. This repository becomes the research contribution repository.
3. Paper experiments can compare original ISynKGR against SGC-ISynKGR directly.
4. Future changes to the new method do not pollute the original codebase.

## Suggested paper/repository title

```text
SGC-ISynKGR: A Semantic Graph-Calibrated Model for Cross-Standard Industrial Knowledge Mapping
```

## Installation

```bash
python -m venv .venv
. .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows PowerShell
pip install -e ".[dev]"
```

## Tests

Run the core tests:

```bash
PYTHONPATH=. pytest tests/test_semantic_graph_calibrated.py tests/test_hybrid_framework_upgrade.py tests/test_scenario_separation.py
```

Run all tests:

```bash
PYTHONPATH=. pytest
```

## Notes

- LLM-based scenarios require a reachable Ollama endpoint.
- The new semantic graph-calibrated model is dependency-free and can run without an external LLM.
- Benchmark outputs are generated under `results/` and `artifacts/`; these folders are intentionally ignored by Git.
