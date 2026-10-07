# SciCode defect investigation: implementation plan

## Confirmed decisions (supersede defaults below)

- Support original SciCode and SciCode-Verified from the first version through separate adapters and `implementation` task arguments. Run and report them separately, keeping prompts, tests, targets, decoder, comparator, and grading environment paired. Use the existing Inspect harness for each, pinned by revision. Preserve Verified's multiple-environment acceptance semantics and record per-environment results. Sample IDs include implementation and problem ID.
- False acceptance means a plausible incorrect solution that a model could reasonably produce given the standard SciCode prompt. Require a rationale identifying an ordinary misunderstanding, algorithmic error, or numerical mistake. Exclude hardcoded expected outputs, test-input lookup tables, reading grader state, comparator/assertion monkeypatching, early exits that bypass tests, and branches designed solely around known tests. The investigator can inspect tests and targets, but the witness must be plausible without that privileged information. Specification-justified constants and special cases remain admissible. Reject cheating witnesses rather than report them as discoveries.
- During dataset setup, shard HDF5 to only the groups relevant to the sample's main problem. Stage the complete shard in the sandbox and provide a helper such as `load_targets(step_id)` returning exact decoded Python objects using that implementation's decoder. Include a manifest, access instructions, and optional clearly marked typed previews in the prompt. Full arrays need not be serialized into model context. Validate group coverage and test counts before running.
- Require strong evidence for a demonstrated finding: complete admissible candidate source, canonical pristine grading replay, and independent semantic support. False acceptance needs a valid counterexample or required-property violation; false rejection needs positive evidence that the implementation satisfies the specification. Include separate `semantic_confidence` and `grader_confidence` ratings (`low`, `medium`, `high`), each with an evidence-based rationale. Confidence is not a calibrated probability and cannot replace a witness. Keep unsupported claims suspected or inconclusive; repeat stochastic experiments where necessary.
- Selected starting investigator: GLM 5.3 through OpenRouter, using `openrouter/z-ai/glm-5.3`, matching the existing Hawk provider configuration. Keep model selection configurable. Start with high reasoning effort and verify provider support during smoke setup. Preserve the 8k generated-token smoke and 100k generated-token pilot limits, including reasoning tokens in generation accounting. Use actual smoke usage to estimate pilot costs; do not launch a run as part of plan preparation.
- Validate harness parity, sharding, report generation, and witness admissibility for both implementations. Prepare a smoke entry for each and a pilot covering both. Tests must include cheating candidates that are rejected and a plausible ordinary incorrect implementation accepted by weak tests.
- Smoke sizing: one selected subproblem per implementation, retaining the full main-problem context and prerequisite code. Add an `investigate_steps` parameter to restrict investigation/report coverage without changing canonical grading semantics. Cap each sample at 8,000 generated tokens, 20 model calls, and 10 minutes. Run the implementations separately when useful; both together have a 16,000-token generation ceiling. This checks the pipeline, not exhaustive discovery. The pilot remains 100,000 generated tokens per main problem, 150 model calls, and 60 minutes, adjusted after observing usage. Use explicit cumulative generation limits in the solver and per-call caps. Do not use Inspect's combined input/output sample token limit as the output-only allowance; verify provider accounting before launch. Input prompts and repeated tool/history input are billed separately and can make total billed tokens substantially larger than generation caps. Reasoning tokens consume the generation allowance where exposed/accounted by the provider. Reserve approximately 1,000 smoke tokens and 5,000 pilot tokens for the final report. Limits are ceilings, not requested minimum spend.

Implementation status: the task, both adapters, sandbox definitions, structured reporting, replay scorer, export, and smoke/pilot configurations are implemented. See README.md for current commands and validation. Paid real-model/Hawk smoke runs remain a separate rollout step.

## Objective and scope

Implement `scicode_scanners/defect_investigation` as a registered Inspect AI task, run with Hawk eval-set configuration. It consumes benchmark problems directly, not model transcripts, and does not create an Inspect Scout scan.

For every investigated subproblem, answer separately:

1. Could a submission satisfy the problem specification but receive an incorrect grade? (False rejection.)
2. Could a submission violate the problem specification but receive a correct grade? (False acceptance.)

The HDF5 values describe what the grader expects; they are not automatically scientific truth. A finding requires both an argument about specification correctness and evidence of grader behavior. Failure to find a witness is not proof that no defect exists.

Initial scope: original SciCode and SciCode-Verified through separate adapters, default cumulative grading, released evaluation split, scientific background included. Make split, problem IDs, background inclusion, and harness revision explicit parameters; do not mix implementations in one result.

## 1. Package and task registration

Add this layout:

```text
defect_investigation/
  __init__.py
  task.py                 # @task scicode_defect_investigation
  dataset.py              # pinned source records -> Inspect Samples
  prompts.py              # evidence packet and investigation instructions
  agent.py                # investigation scaffold
  harness.py              # exact submission assembly and grading
  tools.py                # candidate grading and evidence recording
  schema.py               # validated report and witness models
  scorer.py               # report validation and witness replay
  export.py               # .eval logs -> JSONL and Markdown reports
  sandbox/                # local Docker and Hawk Kubernetes definitions
  hawk-smoke.yaml
  hawk-pilot.yaml
  README.md
```

Import the task in the existing `_registry.py` and document that the package now contains both scanners and evals. Keep existing scanner behavior intact. The package already declares the Inspect entry point and `inspect-ai>=0.3.263`; verify the installed/Hawk version before selecting agent APIs and pin the tested environment for runs.

## 2. Dataset and evidence packet

Use one Inspect sample per main problem, with stable problem IDs. Ask for findings for every constituent subproblem and explicitly identify coverage. This preserves cumulative dependencies and avoids repeatedly sending the same main-problem context. Log subproblem IDs in each finding; allow problem filtering for pilots.

Build samples from pinned benchmark JSONL rather than existing evaluation logs. Record source revision, JSONL/HDF5 hashes, harness revision, prompt settings, image digest, and schema/prompt versions in metadata. Exclude reference implementations and previous audit findings; retain author-provided code for steps the benchmark supplies.

Provide the initial prompt with:

- Full main-problem and subproblem prompts, interfaces, dependencies, scientific background, and supplied code, in benchmark order.
- Exact test-case source for every step, with stable test IDs.
- Instructions for programmatic access to complete expected outputs through the selected harness's HDF5 decoder, preserving dtype, shape, nesting, complex numbers, sparse representations, and nonfinite values.
- Comparator and grading semantics, including cumulative code assembly, execution order, imports, timeouts, and grade aggregation.

Also stage `problem.json`, `prompt.txt`, `tests/`, helper source, and a lossless problem-specific `targets.h5` shard in the sandbox. Reuse the existing failure-classification target checksums, sharding, decoding, and typed serialization where suitable, moving neutral helpers into a shared module if necessary. Avoid making this task depend on Scout transcript adapters.

Measure packet size before launch and never truncate specification or test source silently. Complete expected values are available programmatically from a problem-specific HDF5 shard, without serializing large arrays into the initial prompt.

## 3. Sandbox and grading fidelity

Use Inspect's normal task/sample sandbox lifecycle, with Docker for local development and `inspect-k8s-sandbox` on Hawk. Do not reuse the scanner's manually managed sandbox lifecycle. Pin the grading image by digest and preserve its numerical-library versions, working directory, and thread settings.

Give the agent a writable investigation workspace with shell/Python and editing tools. It can create implementations, probe expected outputs, and write additional tests. Arbitrary candidate code executes only in containers; sandboxes have no credentials, host mounts, or internet requirement.

Maintain a separate grader service inaccessible to the investigation tools. Candidate grading copies only submitted source into a fresh execution context backed by pristine tests, helpers, and targets. This prevents workspace edits from fabricating evidence. Restore state between candidates and between scorer replays.

Create a harness adapter independent of `CaseSet`/transcripts. It must preserve the chosen benchmark implementation's extraction/import handling, cumulative step assembly, supplied steps, target indexing, test execution order, process boundaries, and pass/fail aggregation. Pin whether the target is the upstream harness or the Inspect implementation; default to the existing original-SciCode Inspect harness used in this audit and name its revision explicitly. Do not combine upstream tests with modified helper behavior without recording that choice.

Distinguish canonical full-suite grading from diagnostic isolated tests. Record stdout, stderr, exit status, timeout, executed program hash, environment, and test/step IDs. Infrastructure failures and shorter diagnostic timeouts do not establish a benchmark false rejection.

## 4. Investigation agent

Start with one configurable Inspect ReAct agent per sample. No delegation is needed for the first version. Give it the complete evidence packet and explicit instructions to investigate both directions, rather than solve the benchmark and stop.

Expose standard workspace tools plus a `grade_candidate` tool accepting a step-to-code mapping and requested grading scope. Return a durable evidence ID and bounded observations; save complete programs and outputs in Inspect's sample store/log artifacts. Add an independent-check tool only if it materially simplifies recording experiments; otherwise use workspace Python.

Suggested investigation sequence:

1. Read the specification and tests, map dependencies, and identify plausible mismatches.
2. Construct a correct implementation or independently justified calculation to test false rejection.
3. Construct a plausible incorrect implementation a model could produce from the standard prompt; show a valid input or required property it violates.
4. Minimize witnesses and rerun them against pristine canonical grading.
5. Submit a structured report covering both questions for every step, including unresolved questions.

Prompt categories: wrong targets, overly strict tolerances, unstated representation/order assumptions, nondeterminism, missing assertions, comparator loopholes, missing input coverage, and cumulative dependency effects. Categories guide search; they are not predefined answers.

Reject cheating witnesses. Require a plausibility rationale connecting the candidate to an ordinary misunderstanding or implementation error a model could make from the standard prompt. Automated screens catch obvious interference; human review is required to establish plausibility and rule out hardcoding.

Require correctness reasoning grounded in the prompt, mathematical derivations, independent implementations, or valid counterexamples. Do not accept agreement with HDF5 as the correctness argument. For dependent steps, identify whether the defect originates in the current step, earlier code, or supplied code, and avoid blaming the current grader for an incorrect prerequisite.

## 5. Report schema and validation

Each sample report includes problem ID, coverage, summary, and per-step results for both questions. Each direction has status `demonstrated`, `suspected`, `not_found`, or `inconclusive`. Infrastructure and budget exhaustion are recorded independently.

Each finding includes:

- Affected and originating steps; defect category; precise claim and relevant prompt/test references.
- Complete candidate source and prerequisite implementations, plus submission-access assumptions.
- Canonical grading evidence IDs and observed grade.
- Correctness argument for false rejection, or a valid counterexample/property violation for false acceptance.
- Independent diagnostic source/output, confidence, and limitations.
- Submission plausibility rationale, explicit ambiguity/assumptions, and separate semantic/grader confidence ratings with rationales.

Validate the report with Pydantic and check all evidence references. Use a structured submission tool or final JSON, with at most one bounded format-repair step. Keep malformed output separate from an absence of findings. Persist candidates as they are produced so budget exhaustion still leaves useful artifacts.

Implement an Inspect scorer that validates reports and independently replays claimed witnesses in pristine grading contexts. Replay confirms grader behavior, not semantic correctness. Mark findings as replay-confirmed but semantically unreviewed until reviewed; optionally add a separately configurable reviewer model later. A specification ambiguity should retain its assumptions rather than be promoted automatically to a proven defect.

Do not invent ground-truth defect labels or report discovery rates as investigator accuracy. Export counts/rates of reported and replay-confirmed findings, coverage, unresolved outcomes, costs, and defect categories, separately for each direction. Deduplicate a shared root cause affecting several steps. Human review determines final accepted findings.

## 6. Hawk configuration and limits

Use Hawk `tasks:` entries referencing `scicode_defect_investigation`; no `transcripts:` or `scanners:` section. Pin the package commit and image digest. Configure the investigator model through Hawk, not source code.

Proposed starting budgets, adjustable after the smoke run:

- Smoke: one selected subproblem per implementation with full problem context, one epoch, 8k generated tokens, 20 model calls, 10 minutes per sample.
- Pilot: 5–10 varied problems, one epoch, 100k tokens, 150 model calls, 60 minutes per sample.
- Diagnostic execution: 120 seconds per call initially, with an explicit larger timeout for canonical replay when required by the selected harness.
- Bound sample/sandbox concurrency and resource consumption; set a model-specific cost cap before launch.

Reserve enough budget to investigate both directions and submit the report. Record partial coverage on timeout. Include sandbox resources and network policy in sandbox definitions, separate from Hawk runner resources. Leave checkpointing disabled initially; enable only after verifying support for the chosen agent and sandbox on a short run.

Prepare reproducible local and Hawk commands in the README after checking CLI compatibility. Implementation does not launch paid evaluations; a subsequent requested smoke run validates the actual remote environment.

## 7. Validation and rollout

1. Verify task import/discovery, packaging of sandbox/vendor assets, stable sample IDs, source hashes, target decoding, and prompt completeness using local fixtures.
2. Test harness parity against the pinned existing grader: cumulative dependencies, supplied steps, helper imports, target ordering, canonical aggregation, exceptions, and timeouts.
3. Use synthetic fixtures with a known wrong target, an unchecked wrong implementation, a sound case with no demonstrated defect, and an ambiguous specification. Test malformed reports and missing evidence separately.
4. Confirm that modifying workspace tests/helpers/targets cannot alter canonical grading, and that candidates cannot contaminate subsequent replays.
5. Run one end-to-end local sample when container/model access is available; inspect the `.eval` log, candidate artifacts, both-direction report, and exported results.
6. Run one Hawk smoke sample, then a small diverse pilot. Review finding quality and budget balance before generating a full-run configuration.

Completion criteria: a discoverable Inspect task runs in a sandbox, gives the investigator the complete specified evidence, produces separately reasoned answers to both questions, preserves replayable witnesses, and exports reports with clear coverage and validation status.

## Implementation order

First build the pinned dataset/evidence packet and exact harness adapter. Next add sandbox definitions and candidate grading. Then add the investigation prompt, agent, report schema, scorer, and exports. Finally add focused validation, documentation, and smoke/pilot Hawk configurations.

## References

- Inspect agent scaffolds: https://inspect.aisi.org.uk/agents.html
- Inspect ReAct agent: https://inspect.aisi.org.uk/react-agent.html
- Inspect sandbox lifecycle: https://inspect.aisi.org.uk/sandboxing.html
- Hawk eval-set schema: https://hawk.metr.org/user-guide/eval-set-config-reference/
