# Model Contribution: Semantic Graph-Calibrated Ranker

## Problem

Industrial cross-standard schema mapping often fails when labels are similar but semantically different, or when several target candidates are plausible. A high recall system can therefore create unsafe false-positive mappings.

## Proposed model

The `semantic_graph_calibrated` scenario introduces a calibrated ranker that combines:

- semantic label normalization,
- graph/context overlap,
- parent-path evidence,
- unit compatibility,
- datatype compatibility,
- rule support,
- retrieval support,
- signal-family matching,
- ambiguity penalty,
- confidence-margin rejection.

## Core intuition

The model should only emit a mapping when there is enough multi-source evidence. Otherwise, it should return `no_match`.

```text
score = f(label_semantics,
          graph_context,
          unit_compatibility,
          datatype_compatibility,
          rule_evidence,
          retrieval_evidence,
          ambiguity_penalty)
```

The final score is converted into a calibrated confidence and filtered by threshold and margin.

## Expected improvement over baseline

Compared with a fixed-weight candidate ranker, this model is expected to improve:

- robustness to noisy source labels,
- rejection of duplicate/ambiguous target candidates,
- no-match accuracy,
- interpretability of ranking decisions,
- safer industrial interoperability behavior.
