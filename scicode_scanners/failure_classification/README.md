# Failure classification

Ask “why did this subproblem fail?” for the default SciCode and SciCode-Verified harnesses. One Scout result is returned per scored failed step, with any supported causal findings and their originating steps.

## Categories and output

`underspecified` means the submitted behavior follows a valid interpretation of the prompt but the test imposes an unstated requirement. `wrongly_specified` means an explicit prompt requirement conflicts with the test. `model_error` means a model mistake causally contributes. `operational_error` means an evidenced operational interruption or transient source-eval failure, such as a generation limit truncating the submission or a provider/tool/infrastructure error preventing execution. A syntax error or timeout alone is insufficient: ordinary coding mistakes and inefficient algorithms remain `model_error`. The scanner exhausting its own budget remains an unresolved assessment with a limitation. `other` describes an established cause outside those definitions.

The result `value` is a boolean: positive (`True`) when any supported cause is `underspecified`, `wrongly_specified`, `operational_error`, or `other`; negative (`False`) for model-error-only assessments or no supported causes. Mixed causes are positive if any non-model cause is established. Unique supported categories are retained in `metadata.categories`. `answer` is `resolved`, `partially_resolved`, or `unresolved`. Full assessments live in `metadata.assessment`; the source identity is in `metadata.source`, and judge calls, token usage, read-only evidence retrievals are in `metadata.investigation`. Unresolved assessments are negative with an empty category list; this indicates no established positive cause, not proof of model error. All-passing transcripts return an accounting result without a judge call.

Each cause records `origin_type`, `origin_steps`, `dependency_path`, evidence and `causal_contribution`. A stable `cause_id` permits deduplication across affected steps within the same transcript. Matching is conservative: category, origin and normalized `defect_key` must agree. Different model attempts remain separate. Inspect the evidence before treating findings or deduplication as validated.

## Evidence and tools

The initial packet contains the current submitted solution with its description, interface and tests; dependency imports; an index of earlier steps (including author-provided code) with a one-line summary, signature, defined names, code size, `code_status` (`ok`, `empty` or `does_not_parse`) and which of their names the current code uses, and any abnormal `stop_reason`; the current step's `stop_reason`; and `recorded_issues`, the transcript's error, sample-limit, and unassigned or failed generation events (usually empty). Earlier code is fetched with `read_code` or `retrieve_step(section="code")`; in sandbox mode the full cumulative program is also available to `python` as `cumulative_code`. The packet is sent whole in the first message, without a size cap. HDF5 targets are represented only by type, shape/dtype for arrays and sparse matrices, or container size; values are examined in the sandbox with `python`, and are unavailable in inspection-only mode. The packet excludes raw responses, generation prompts, system instructions, grading output, and run settings.

The tools are `retrieve_step` (selected grading, submitted code, reasoning, tests, or explicit raw prompt/response), `read_code` (one function or class definition by name, from the current or an earlier step), `python` (a diagnostic in the grading interpreter with the current step's `targets`, `load_targets(step_id)` for other steps, and the cumulative submitted code available), `rerun` (exact original grading, an isolated test, or labelled replacement of current/earlier step code), and the structured `answer` tool. Each execution receives an `X:<n>` reference that causes may cite. Original grading observations and exposed reasoning remain available through `retrieve_step`. No earlier judge findings are exposed.

Scanner version 6 removes default transcript code duplication and finalizes under context pressure; version 5 added enforced answer-tool finalization and a larger protected answer budget; version 4 changed the context and available tools. Earlier outputs and checkpoints are not reused by this version; scans must be reprocessed to use the new evidence packet.

Evidence retrieval is bounded to 12,000 characters per response. Large payloads return a JSON-text excerpt and `next_char_offset`; repeat the same tool arguments with that value as `char_offset` to continue. Omitted text is explicitly marked incomplete. Context pressure at 75,000 serialized-history bytes triggers answer-only finalization. Retrieved evidence that would push history beyond 85,000 bytes is withheld with an explicit missing-evidence notice, preserving space for submission. A 100,000-byte hard ceiling remains for exceptional oversized model responses.

No reference solutions or audit findings are exposed. Reference-code fields present in source metadata are removed recursively. Author-provided steps are available because they were visible to the evaluated model. HDF5 targets are the grader's expectations, not a guarantee of scientific correctness.

By default, each failed step gets a fresh Kubernetes sandbox from the pinned grading image (`sandbox.IMAGES`), provisioned by the scanner through `inspect-k8s-sandbox`: gVisor, no network, 2 CPU / 4 GiB, single-threaded BLAS. Only the transcript's target groups and the vendored grading helpers are copied in. Every experiment runs in a new working directory, so diagnostic modifications never carry over. Diagnostics are limited to `python_timeout` (30 s) per run and `diagnostic_seconds` (120 s) per failed step; a diagnostic timeout does not reproduce the source's 1,800-second grading limit. Full stdout/stderr are kept in `metadata.investigation.tools`; the judge sees the head of stdout and the tail of stderr, 5,000 characters each.

`execution: false` restores inspection-only classification with no sandbox. The judge is then told execution is unavailable and must leave hypotheses that need execution uncertain.

Verified's default grading accepts a pass in either the 2024 or 2025 interpreter; `rerun` preserves that short-circuit, and `python` can target either environment. The scanner preserves the recorded results and actual source settings, including scientific background and environment selection.

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

hawk scan run scicode_scanners/failure_classification/hawk-scicode-full.yaml
hawk scan run scicode_scanners/failure_classification/hawk-scicode_verified-full.yaml
```

Smoke configurations select a known failed transcript and run classification with the same budgets as full runs. Set `dry_run: true` to check extraction and HDF5 artifacts without inference or sandboxes, or `smoke_only: true` to also provision a sandbox and check each grading interpreter, still without inference (`hawk-*-sandbox-smoke.yaml`). Full configurations scan all transcripts in the corresponding eval set without a transcript cap. Both smoke and full runs use paid GLM-5.3 inference; review the smoke results before a full run.

Sandbox runs need `inspect-k8s-sandbox` in the config's `packages` and scan-runner permissions for Kubernetes sandboxes. Inspection-only runs (`execution: false`) need neither. Source logs are read from their Hawk URI with full attachments resolved. Target files are downloaded with pinned checksums and decoded with the exact source harness helpers. Custom target files must match the expected full-file checksum.


## Budget and resume

The default judge is GLM-5.3 with `high` reasoning, the middle supported level. The allowance is **8,192 generated tokens per failed subproblem**, including reasoning and tool arguments across calls, with **1,536 reserved for finalization**. Investigation calls are capped at 4,096 tokens. The judge is told its total and remaining allowance before each call; the harness enforces it. Inputs are accounted separately. There is no global dollar cap.

Defaults also allow six read-only investigation tool calls. All limits are configurable scanner arguments. Model calls are also capped at ten by default to bound empty or malformed responses. No automatic model retries are enabled. Missing usage is charged at the request's full allowance. Budget exhaustion produces an unresolved assessment rather than another unbudgeted call.

Checkpoint records are stored under the Hawk scan results URI, discovered from the runner's infrastructure configuration. An explicit `checkpoint_uri` can override it. A request's full potential charge is saved before sending; interruptions conservatively retain that charge. Completed assessments are reused on resume. Checkpoint identity includes the scanner version, investigation mode and sandbox image, so sandbox and inspection-only investigations never reuse each other. Consumed diagnostic seconds and experiment numbering persist across resume.

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

Tests cover extraction differences, retry handling, reference-field removal, cumulative code, origin validation, deduplication, both HDF5 decoders, token accounting and budget awareness, finalization, checkpoints, read-only tool restrictions, diagnostic time and output limits, Verified's either-environment rerun, fresh experiment directories, `smoke_only`, and both harnesses through the full scanner path in both modes. These tests use fixtures, mocked model calls and fake sandboxes; they do not run evaluations locally.

Extraction was also checked against all eight source logs: 260 SciCode transcripts (573 failed steps) and 256 Verified transcripts (195 failed steps). Reconstructed SciCode grading matched all 1,119 recorded programs. All 183 distinct failed SciCode target groups and 101 Verified groups decoded successfully. Semantic classification quality requires reviewing smoke results before a full run.

The detailed design is in [PLAN.md](PLAN.md); vendored helper provenance is in [vendor/README.md](vendor/README.md).

### Generation budgets

Smoke and full configurations use 32,768 generated tokens per failed
subproblem, including reasoning, with 8,192 tokens protected for finalization.
Investigation calls are capped at 4,096 tokens and 12 retrieval calls. Finalization
disables retrieval, requires the answer tool, lowers reasoning effort to `low`,
and can use the entire remaining token budget rather than the investigation
per-call cap. Four additional model calls are available for finalization and
validation corrections, within the total token budget. Empty responses also
count toward the model-call limit, which triggers finalization before exhaustion.

Publish the implementation and update each YAML's scanner package commit before
running it remotely; local budget edits alone do not update the remote code.

Transcript retrieval does not reload model-event inputs by default. This matters
for SciCode's conversation history and Verified's prompts containing preceding
solutions. `retrieve_step` defaults to grading observations without executed programs.
Reasoning can be requested separately without loading response text. Exact raw
prompts and responses remain available explicitly for investigating harness
behavior; these may duplicate code already present in the initial packet.

Scanner version 7 adds `operational_error`. Grading retrieval includes the source
generation stop reason, event reference, and usage when available, without
loading code or reasoning. Existing classifications are not relabeled; reprocess
with the updated scanner to apply this category.

Scanner version 8 changes Scout positivity from category-list truthiness to an
explicit boolean. Model-error-only results are negative; all other established
cause categories are positive. Existing scan outputs retain their old values and
need reprocessing or deterministic remapping to adopt this counting convention.

Scanner version 9 restores sandboxed `python` and `rerun` experiments as the
default (`execution: false` keeps inspection-only behavior) and restores
`smoke_only`. Result metadata records `investigation_mode` and `sandbox_image`.

Scanner version 10 makes answer validation tolerant of punctuation after
references (`X:1: stdout ...`) and reports every validation problem across all
causes in one rejection. Rejected submissions are replaced in the judge's
history by a stub, so retries do not accumulate copies of large assessments; the
full submission remains in `metadata.investigation.calls`. Schema descriptions
state the step-ID and reference formats. Target downloads are serialized across
Scout worker processes with a file lock.

Scanner version 11 replaces the packet's full copies of every earlier step's code
with an index plus the earlier definitions the current code uses directly. On
the Verified smoke transcript this cuts packets from 52–93KB to 8–25KB, so they
fit in the first message without paging.

Scanner version 12 tightens the `underspecified` definition: requirements a
competent expert would assume by default, such as the field's standard
conventions, count as specified, and the judge must name the competing reading
the submission followed. The earlier-code copies in
the packet (`called_earlier_code`) are removed; the index's `used_by_current`
names what to fetch with `read_code`. The packet is no longer capped, so
`current_evidence` is removed. `full_transcript` and
`read_message` are removed. The packet now carries
stop reasons and `recorded_issues` (error, sample-limit and unassigned
generation events), the only transcript events not reachable through
`retrieve_step` that bear on a cause. On the Opus full-run transcripts, packets
for the 154 failed steps are 2.7–13.7KB (median about 6KB). `inspect_target` is removed: target values are examined in
the sandbox, where `python` provides `targets` for the current step and
`load_targets(step_id)` for any other. The largest targets (problem 63, up to
160MB per step) were decoded in the scanner process on every call and could
only be previewed 64 values at a time.
