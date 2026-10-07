# SciCode scanners

Inspect Scout scanners run through Hawk. `hawk.yaml` selects a shuffled pilot of 12 main-problem transcripts from `scicode-verified-default--tu1flh8tik8bh6l2`, with GLM-5.3 at max reasoning as judge.

## Run

Publish this repository where the Hawk runner can install it. The package URL in `hawk.yaml` is currently `https://github.com/xantheocracy/scicode-scanners.git@main`. Update it if you publish elsewhere, and pin a commit SHA for reproducibility. Configure your Hawk deployment and authentication before running:

```bash
hawk scan run hawk.yaml
```

This launches paid inference. GPT source transcripts are excluded because they expose no CoT. Each remaining main-problem transcript can produce several failed-subproblem results. Review the pilot, then remove `transcripts.filter.limit` to scan the full run. This repository does not provide a separate local runner or transcript database.

## Files

- `hawk.yaml` specifies the run, scanner, judge model, and pilot size.
- `scicode_scanners/verified_memory.py` extracts failed subproblems and runs the classifier.
- `scicode_scanners/memory.py` defines the classification criteria and structured answer.
- `scicode_scanners/data/audit_context.json` contains the audit evidence supplied to the judge.
- `scicode_scanners/logs.py` supports reading older Zstandard-compressed Inspect logs.
- `pyproject.toml` packages and registers the scanner for Hawk.

## Detection criterion

Each judge call assesses the current subproblem using its prompt, solution, exposed CoT, grading errors, and audit context. Previous solution code already present in that prompt remains visible. Earlier responses and CoTs are not added. A positive requires the current response to show recall, adoption, and a resulting error. A failure inherited solely from earlier code is negative for this case.

The result value is boolean. Only `supported_memory_induced_failure` is positive (`true`). Possible cases, recognition alone, ordinary errors, inherited failures without established memory origin, insufficient evidence, empty submissions, and all-passing transcripts are negative (`false`). The full assessment is retained in result metadata. Scanner version 2 corrects the previous structured result value; existing scan outputs do not change automatically.

A supported finding requires evidence of a specific recalled original behavior, adoption in submitted code, conflict with a correction, and a connection to the failure. Benchmark recognition alone receives a separate label. An original formula appearing in code without evidence of recall could be an independently generated mistake. The scanner identifies inherited failures and missing evidence separately. It does not execute model code. Findings require review and do not establish training-data provenance.

Cases include the model-visible prompt, response, exposed reasoning, grading errors, and original-to-verified audit evidence. GPT is excluded because its reasoning is unavailable. Analyst context is labelled separately from model-visible content. Messages are preserved without truncation.

All-passing transcripts return `no_failed_subproblems` without a judge call. Empty token-limit submissions return `no_submission_token_limit` without a judge call. Across the three included source models there are 145 failed model-subproblem attempts, including 24 empty token-limit submissions. This leaves 121 judge calls for a full scan.

## Audit context and validation

The bundled audit context covers the 101 distinct failed subproblems in the configured run. It includes historical findings for corrections omitted from the retained defect taxonomy. It contains audit evidence, not model transcripts. The scanner raises on an uncovered failed step; extend this context before targeting other runs or datasets.

The Hawk config schema and extraction across all four logs were checked, reproducing 195 failures and 24 empty submissions. The empty-submission branch and an inference-free Scout dry run were also checked. No paid scan was launched during setup. Semantic detection accuracy remains unvalidated.
