# New Repository Setup

Use this file when creating the clean GitHub repository for SGC-ISynKGR.

## 1. Create a new GitHub repository

Suggested repository name:

```text
SGC-ISynKGR
```

Suggested description:

```text
Semantic Graph-Calibrated Industrial Schema Mapping and Benchmark Framework
```

## 2. Initialize Git locally

From the repository root:

```bash
git init
git add .
git commit -m "Initial clean release of SGC-ISynKGR"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/SGC-ISynKGR.git
git push -u origin main
```

## 3. Keep the original repository separate

Do not push this code into the old ISynKGR repository if the goal is a clean research contribution. The old repository should remain the baseline; this repository should contain the new semantic graph-calibrated model.

## 4. Suggested comparison in paper

Use the original repository as the baseline and compare:

```text
ISynKGR baseline
vs.
SGC-ISynKGR semantic_graph_calibrated
```

Recommended metrics:

```text
Exact Match
Precision
Recall
F1
False Positive Rate
No-Match Accuracy
Ambiguity Rejection Rate
Confidence Calibration Error
```
