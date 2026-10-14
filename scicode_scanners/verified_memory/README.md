# Verified memory scanner

Inspect Scout scanners run through Hawk. `hawk.yaml` selects all non-GPT main-problem transcripts from `scicode-verified-default--tu1flh8tik8bh6l2`, with GLM-5.3 at max reasoning as judge.

## Run

Publish this repository where the Hawk runner can install it. The package URL in `hawk.yaml` is currently `https://github.com/xantheocracy/scicode-scanners.git@main`. Update it if you publish elsewhere, and pin a commit SHA for reproducibility. Configure your Hawk deployment and authentication before running:

```bash
hawk scan run scicode_scanners/verified_memory/hawk.yaml
```

This launches paid inference. GPT source transcripts are excluded because they expose no CoT. Each remaining main-problem transcript can produce several failed-subproblem results. For a shuffled pilot, add `limit: 12` and `shuffle: 42` under `transcripts.filter` before launching. This repository does not provide a separate local runner or transcript database.

## Files

- `hawk.yaml` specifies the run, scanner, judge model, and transcript filter.
- `scanner.py` extracts failed subproblems and runs the classifier.
- `memory.py` defines the classification criteria and structured answer.
- `context.py` assembles audit context for any bundled subproblem at runtime.
- `data/defects.json` contains the complete retained audit log; `data/historical_evidence.json` preserves optional historical review and target patches by problem.
- `data/sources.json` records data provenance and checksums.
- `../logs.py` supports reading older Zstandard-compressed Inspect logs.
- `../../pyproject.toml` packages and registers the scanner for Hawk.

## Detection criterion

Each judge call assesses the current subproblem using its prompt, solution, exposed CoT, grading errors, and audit context. Previous solution code already present in that prompt remains visible. Earlier responses and CoTs are not added. A positive requires the current response to show recall, adoption, and a resulting error. A failure inherited solely from earlier code is negative for this case.

The result value is boolean. Only `supported_memory_induced_failure` is positive (`true`). Possible cases, recognition alone, ordinary errors, inherited failures without established memory origin, insufficient evidence, empty submissions, and all-passing transcripts are negative (`false`). The full assessment is retained in result metadata. Scanner version 2 corrects the previous structured result value; existing scan outputs do not change automatically.

A supported finding requires evidence of a specific recalled original behavior, adoption in submitted code, conflict with a correction, and a connection to the failure. Benchmark recognition alone receives a separate label. An original formula appearing in code without evidence of recall could be an independently generated mistake. The scanner identifies inherited failures and missing evidence separately. It does not execute model code. Findings require review and do not establish training-data provenance.

Cases include the model-visible prompt, response, exposed reasoning, grading errors, and original-to-verified audit evidence. GPT is excluded because its reasoning is unavailable. Analyst context is labelled separately from model-visible content. Messages are preserved without truncation.

All-passing transcripts return `no_failed_subproblems` without a judge call. Empty token-limit submissions return `no_submission_token_limit` without a judge call. Across the three included source models there are 145 failed model-subproblem attempts, including 24 empty token-limit submissions. This leaves 121 judge calls for a full scan.

## Audit context and validation

Scanner version 7 builds context programmatically for all 291 steps in the bundled 65-problem original test set (including the three provided steps). It combines the current and preceding original step text/tests with the whole problem's audit records, plus historical findings and corrected target patches when available. Future original steps and original reference code are excluded. It contains audit evidence, not model transcripts. A different pattern of failures needs no new per-step context entries. An unknown problem or step outside the bundled dataset still raises an explicit error; this is not an automatic dataset-version updater. No data is downloaded at scan time.

The former failure-specific `data/audit_context.json` has been replaced with complete source audit records and deduplicated historical evidence. Historical evidence is preserved without inventing conclusions for previously unreviewed problems. Missing audit records or historical evidence do not establish that no fix occurred. Existing scan results are unchanged; rerun to use the new context assembly.

The Hawk config schema and extraction across all four logs were checked, reproducing 195 failures and 24 empty submissions. The empty-submission branch and an inference-free Scout dry run were also checked. No paid scan was launched during setup. Semantic detection accuracy remains unvalidated.

## Provider-default GLM rerun, judged by Opus 5.5

`hawk-glm-opus55.yaml` targets only GLM-5.3 transcripts from
`scicode-verified-provider-88bhokyeuxsftey2`. `hawk-glm-opus55-pilot.yaml`
selects eight shuffled main-problem transcripts (seed 42).
Both use Opus 5.5 via OpenRouter/Anthropic, medium reasoning, an 8,192-output-token
limit per judge call, and five concurrent transcripts.

Scanner version 7 automatically covers all bundled steps, including the nine additional failures in this run. Detection criteria are unchanged.

Commit and push the updated scanner and context before launching: Hawk installs
the Git package, not this working directory. The configs currently use `@main`;
replace it with the published commit SHA to freeze the scanner version.

```bash
hawk scan run scicode_scanners/verified_memory/hawk-glm-opus55-pilot.yaml
hawk scan run scicode_scanners/verified_memory/hawk-glm-opus55.yaml
```

The cached log has 64 main-problem transcripts, 49 failed subproblems, and two
empty token-limit submissions: 47 paid judge calls. The inference-free preflight
and cost assumptions are saved in `../analysis/memory_scan/` relative to this
repository's parent workspace. No paid scan was launched while preparing these configs.
