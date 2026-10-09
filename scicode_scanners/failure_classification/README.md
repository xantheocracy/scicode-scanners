# Failure classification

Ask “why did this subproblem fail?” for the default SciCode and SciCode-Verified harnesses. One Scout result is returned per scored failed step, with any supported causal findings and their originating steps.

## Categories and output

`underspecified` means the submitted behavior follows a valid interpretation of the prompt but the test imposes an unstated requirement. `wrongly_specified` means an explicit prompt requirement conflicts with the test. `model_error` means a model mistake causally contributes. `other` describes an established cause outside those definitions.

The result `value` is a list of unique supported categories. `answer` is `resolved`, `partially_resolved`, or `unresolved`. Full assessments live in `metadata.assessment`; the source identity is in `metadata.source`, and judge calls, token usage, read-only evidence retrievals are in `metadata.investigation`. Unresolved assessments have an empty category list. All-passing transcripts return an accounting result without a judge call.

Each cause records `origin_type`, `origin_steps`, `dependency_path`, evidence and `causal_contribution`. A stable `cause_id` permits deduplication across affected steps within the same transcript. Matching is conservative: category, origin and normalized `defect_key` must agree. Different model attempts remain separate. Inspect the evidence before treating findings or deduplication as validated.

## Evidence and tools

The initial packet contains the current submitted solution, all preceding solutions (including author-provided code), their subproblem descriptions and interfaces, dependency imports, and the current tests. HDF5 targets are represented only by type, shape/dtype for arrays and sparse matrices, or container size; values are retrieved on demand. The packet excludes raw responses, generation prompts, system instructions, grading output, and run settings.

The tools are `current_evidence` (packet pagination), `retrieve_step` (exact step evidence and model messages), `full_transcript` (event reading and literal search), `inspect_target` (target values, nested paths, flattened pagination, optional two-dimensional row/column slices, and `metadata_only=True` to inspect shape/dtype without values), and the structured `answer` tool. Original grading observations and exposed reasoning remain available through transcript retrieval. No earlier judge findings are exposed.

Scanner version 4 changes the context and available tools. Earlier outputs and checkpoints are not reused by this version; scans must be reprocessed to use the new evidence packet.

Evidence retrieval is bounded to 12,000 characters per response. Large payloads return a JSON-text excerpt and `next_char_offset`; repeat the same tool arguments with that value as `char_offset` to continue. Oversized initial evidence is paginated too, with the rest available through `current_evidence`. Omitted text is explicitly marked incomplete. A conservative 100,000-byte serialized-history ceiling stops investigation with an unresolved `input_context_limit` assessment before sending an oversized request.

No reference solutions or audit findings are exposed. Reference-code fields present in source metadata are removed recursively. Author-provided steps are available because they were visible to the evaluated model. HDF5 targets are the grader's expectations, not a guarantee of scientific correctness.

The scanner is inspection-only. It does not provision a sandbox, execute submitted or model-generated code, or rerun tests. The judge distinguishes original recorded observations from deductions and untested hypotheses. Numerical behavior, upstream inheritance, or proposed fixes that require execution to verify must remain uncertain; material gaps are recorded in assessment limitations.

Verified's default grading accepts a pass in either the 2024 or 2025 interpreter. The scanner preserves the recorded results and actual source settings, including scientific background and environment selection.

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

Smoke configurations select a known failed transcript and exercise extraction and HDF5 artifacts without inference. Pilot configurations select two shuffled main-problem transcripts each; each main problem may contain several failed steps. They launch paid GLM-5.3 inference. Review these before using the corresponding `-full.yaml` configurations.

Native Hawk scan jobs can run this scanner without sandbox permissions. Source logs are read from their Hawk URI with full attachments resolved. Target files are downloaded with pinned checksums and decoded with the exact source harness helpers. Custom target files must match the expected full-file checksum.

Opus 5.5-only configurations use an exact source-model filter and pin the scanner package to an implementation commit:

```bash
hawk scan run scicode_scanners/failure_classification/hawk-scicode-opus55.yaml
hawk scan run scicode_scanners/failure_classification/hawk-scicode_verified-opus55.yaml
```

These select 65 SciCode transcripts (106 failed scored steps) and 64 Verified transcripts (48 failed scored steps). The source model is Opus 5.5; the classification judge remains GLM-5.3. All-passing transcripts make no judge calls. These configurations use the same inspection-only scanner.

## Budget and resume

The default judge is GLM-5.3 with `high` reasoning, the middle supported level. The allowance is **8,192 generated tokens per failed subproblem**, including reasoning and tool arguments across calls, with **1,536 reserved for finalization**. Investigation calls are capped at 4,096 tokens. The judge is told its total and remaining allowance before each call; the harness enforces it. Inputs are accounted separately. There is no global dollar cap.

Defaults also allow six read-only investigation tool calls. All limits are configurable scanner arguments. Model calls are also capped at ten by default to bound empty or malformed responses. No automatic model retries are enabled. Missing usage is charged at the request's full allowance. Budget exhaustion produces an unresolved assessment rather than another unbudgeted call.

Checkpoint records are stored under the Hawk scan results URI, discovered from the runner's infrastructure configuration. An explicit `checkpoint_uri` can override it. A request's full potential charge is saved before sending; interruptions conservatively retain that charge. Completed assessments are reused on resume. Inspection-only checkpoints use scanner version 3 and a mode identifier, so they cannot reuse older sandbox investigations.

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

Tests cover extraction differences, retry handling, reference-field removal, cumulative code, origin validation, deduplication, both HDF5 decoders, token accounting and budget awareness, finalization, checkpoints, read-only tool restrictions, and both harnesses through the full scanner path. These tests use fixtures and mocked model calls; they do not run evaluations locally.

Extraction was also checked against all eight source logs: 260 SciCode transcripts (573 failed steps) and 256 Verified transcripts (195 failed steps). Reconstructed SciCode grading matched all 1,119 recorded programs. All 183 distinct failed SciCode target groups and 101 Verified groups decoded successfully. The eight configurations validate against Hawk 3.6.0. Semantic classification quality requires reviewing the Hawk pilot.

The detailed design is in [PLAN.md](PLAN.md); vendored helper provenance is in [vendor/README.md](vendor/README.md).
