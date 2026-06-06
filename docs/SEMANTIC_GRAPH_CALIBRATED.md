# Semantic Graph Calibrated Ranker

`semantic_graph_calibrated` is an improved ISynKGR model for industrial schema/path mapping. It keeps the repository dependency-free but replaces the previous fixed additive score with a calibrated semantic graph confidence model.

## Main idea

For every source node and target candidate, the model builds normalized semantic features:

- label-token overlap with industrial aliases such as `temp -> temperature`, `rpm -> speed`, and `status -> state`;
- graph/context overlap from parent paths, asset/equipment identifiers, process-line hints, and candidate path context;
- datatype and unit compatibility;
- retrieval score and rule support;
- duplicate-candidate ambiguity penalty;
- signal-family agreement for temperature, pressure, flow, speed, state, and vibration.

These features are passed through a logistic confidence head. The final selector accepts a mapping only if both confidence and top-1/top-2 margin are high enough. Otherwise it emits `no_match`, which should reduce false-positive mappings in noisy and realistic benchmark tiers.

## How to run

```bash
PYTHONPATH=. python -m benchmark.run --scenario semantic_graph_calibrated --out results/semantic_graph_calibrated
```

Or include it in the full orchestrated benchmark via `benchmark/benchmark_full.json`.

## Why this should be stronger than the existing adaptive ranker

The existing adaptive candidate ranker combines lexical, rule, datatype, unit, parent, and retrieval evidence with fixed weights. The new model adds semantic alias normalization, richer node-context features, confidence calibration, and explicit ambiguity/no-match rejection. This is especially useful where two candidates share similar labels but differ by asset, parent context, unit, or signal role.
