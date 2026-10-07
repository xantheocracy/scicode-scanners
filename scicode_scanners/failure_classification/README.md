# Failure classification

Ask “why did this subproblem fail?” for the default SciCode and SciCode-Verified harnesses. One Scout result is returned per scored failed step, with any supported causal findings and their originating steps.

## Categories and output

`underspecified` means the submitted behavior follows a valid interpretation of the prompt but the test imposes an unstated requirement. `wrongly_specified` means an explicit prompt requirement conflicts with the test. `model_error` means a model mistake causally contributes. `other` describes an established cause outside those definitions.

The result `value` is a list of unique supported categories. `answer` is `resolved`, `partially_resolved`, or `unresolved`. Full assessments live in `metadata.assessment`; the source identity is in `metadata.source`, and judge calls, token usage, retrievals, diagnostic scripts and outputs are in `metadata.investigation`. Unresolved assessments have an empty category list. All-passing transcripts return an accounting result without a judge call.

Each cause records `origin_type`, `origin_steps`, `dependency_path`, evidence and `causal_contribution`. A stable `cause_id` permits deduplication across affected steps within the same transcript. Matching is conservative: category, origin and normalized `defect_key` must agree. Different model attempts remain separate. Inspect the evidence before treating findings or deduplication as validated.

## Evidence and tools

The initial packet contains current requirements, the current response/code, tests, original grading evidence and typed HDF5 target previews. Earlier transcript context and exposed reasoning are retrieved on demand. The tools are `retrieve_step`, `full_transcript`, `inspect_target`, `helper_source`, `python`, `rerun`, and the structured `answer` tool.

No reference solutions or audit findings are exposed. Reference-code fields present in source metadata are removed recursively. Author-provided steps are available because they were visible to the evaluated model. HDF5 targets are the grader's expectations, not a guarantee of scientific correctness.

Python runs in fresh working directories in a Kubernetes sandbox on Hawk, using the source harness's image, helper modules, target decoding and interpreter(s). Sandbox images are pinned by digest. The sandbox cannot access the network or a Kubernetes API token. Each experiment starts with canonical artifacts so modifications from earlier diagnostics cannot affect later results. Source submissions and results remain intact.

Verified's default grading accepts a pass in either the 2024 or 2025 interpreter. Actual source settings override defaults, including scientific background and environment selection. Diagnostic timeouts are recorded separately from original grading failures.

## Run through Hawk

Source eval sets:

| Configuration prefix | Eval set |
| --- | --- |
| `hawk-scicode-` | `scicode-default-max-6nsnsp7trfmhbofk` |
| `hawk-scicode_verified-` | `scicode-verified-default--tu1flh8tik8bh6l2` |

Set the deployment and authenticate:

```bash
export HAWK_API_URL="https://api.hawk.hawk.generalitylabs.ai/"
hawk login
```

Run these from the repository root after publishing the scanner package. The package Git references must point to the implementation commit; pin a commit SHA before substantive runs.

```bash
hawk scan run scicode_scanners/failure_classification/hawk-scicode-smoke.yaml
hawk scan run scicode_scanners/failure_classification/hawk-scicode_verified-smoke.yaml

hawk scan run scicode_scanners/failure_classification/hawk-scicode-pilot.yaml
hawk scan run scicode_scanners/failure_classification/hawk-scicode_verified-pilot.yaml
```

Smoke configurations select a known failed transcript and exercise extraction, HDF5 artifacts and sandbox interpreters without inference. Pilot configurations select two shuffled main-problem transcripts each; each main problem may contain several failed steps. They launch paid GLM-5.3 inference. Review these before using the corresponding `-full.yaml` configurations.

Source logs are read from their Hawk URI, with full attachments resolved. Target files are downloaded with pinned checksums and only relevant problem groups are copied into sandboxes. Custom target files must match the expected full-file checksum.

## Budget and resume

The default judge is GLM-5.3 with `high` reasoning, the middle supported level. The allowance is **8,192 generated tokens per failed subproblem**, including reasoning and tool arguments across calls, with **1,536 reserved for finalization**. Investigation calls are capped at 4,096 tokens. The judge is told its total and remaining allowance before each call; the harness enforces it. Inputs are accounted separately. There is no global dollar cap.

Defaults also allow six investigation tool calls, 30 seconds per Python diagnostic and 120 seconds total diagnostic execution. All limits are configurable scanner arguments. No automatic model retries are enabled. Missing usage is charged at the request's full allowance. Budget exhaustion produces an unresolved assessment rather than another unbudgeted call.

Checkpoint records are stored under the Hawk scan results URI, discovered from the runner's infrastructure configuration. An explicit `checkpoint_uri` can override it. A request's full potential charge is saved before sending; interruptions conservatively retain that charge. Completed assessments are reused on resume. A restarted sandbox has fresh files; saved diagnostic evidence remains available to the judge.

```bash
hawk scan resume SCAN_RUN_ID
```

## Export

Download the Scout result directory from Hawk, then export one CSV row per affected-step/cause pair. Unresolved steps retain a row with no cause.

```bash
python -m scicode_scanners.failure_classification.export DOWNLOADED_SCAN_DIRECTORY causes.csv
```

The command reports affected-step and distinct-cause totals. Results retain source model, transcript, epoch, affected step, origin and category for filtering.

## Verification

```bash
python -m pytest -q
```

Tests cover extraction differences, retry handling, reference-field removal, cumulative code, origin validation, deduplication, both HDF5 decoders, token accounting and budget awareness, finalization, checkpoints and diagnostic timeouts. These tests use fixtures and mocked model calls; they do not run evaluations locally.

Extraction was also checked against all eight source logs: 260 SciCode transcripts (573 failed steps) and 256 Verified transcripts (195 failed steps). Reconstructed SciCode grading matched all 1,119 recorded programs. All 183 distinct failed SciCode target groups and 101 Verified groups decoded successfully. The six configurations validate against Hawk 3.6.0. Semantic classification quality requires reviewing the Hawk pilot.

The detailed design is in [PLAN.md](PLAN.md); vendored helper provenance is in [vendor/README.md](vendor/README.md).
