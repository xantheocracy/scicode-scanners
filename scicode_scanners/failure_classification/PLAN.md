# Failure classification scanner — review draft

## Objective

For every scored, failed subproblem, answer: “Why did this subproblem fail?” Return one assessment per failed subproblem, containing one or more supported causal findings. Usually there should be one. Keep unresolved assessments separate from the `other` category.

This is a classification scanner, not a boolean detection scanner. Passing and author-provided steps remain available as context but do not receive failure classifications. Preserve all-passing transcripts in run accounting without invoking the judge.

## Supported implementations

- Original SciCode: `inspect_evals`, branch `branch`, default non-scaling harness. Locally inspected revision: `51f92017ccbdc63bd768ed6aba36942d43243b25`.
- SciCode-Verified: `scicode-verified`, branch `inspect-wrapper`, default harness. Locally inspected revision: `3fd9ef2713244d58f6cfeab5a5d92c10e2b907de`.

Run all scans through Hawk, including smoke checks and pilots. Provide explicit `implementation=scicode|scicode_verified` options and separate Hawk run configurations.

Use these source eval sets:

| Implementation | Source eval set |
| --- | --- |
| SciCode | `scicode-default-max-6nsnsp7trfmhbofk` |
| SciCode-Verified | `scicode-verified-default--tu1flh8tik8bh6l2` |

Select a small pilot from each of these sets before scanning them in full. Pin code revisions, dataset checksums, target checksums, and sandbox image digests for actual scans. Prefer recorded run settings when reconstructing a transcript; defaults must not override settings used in the source evaluation.

Use separate adapters that produce a common failure-case representation. Preserve these differences:

| Detail | SciCode | SciCode-Verified |
| --- | --- | --- |
| Model context | Growing conversation, including previous responses | Fresh request containing previous extracted code |
| Scientific background default | Disabled | Enabled |
| Targets | Original `test_data.h5` | Corrected `test_data_cleaned.h5` |
| Grading | Single default sandbox interpreter | Pass in either selected environment; default selects 2024 and 2025 |
| Evidence in logs | Scores, scorer explanations, nested scorer/sandbox events | Per-step scores and per-environment output; stored extracted code |

Match each adapter's extraction, supplied steps, helper imports, cumulative program construction, target decoding, test ordering, and timeout behavior. Recover intermediate requests from model events, resolving pooled attachments when needed. Do not infer correspondence solely by counting model calls: handle retries, supplied steps, and explicit step identifiers.

## Classification rules

- `underspecified`: the relevant solution behavior is valid under the model-visible specification, but tests demand an unstated requirement. Identify the missing requirement and explain why the implementation is a valid interpretation.
- `wrongly_specified`: the relevant solution follows an explicit requirement that directly conflicts with the tests. Quote both sides of the contradiction.
- `model_error`: the solution contains a mistake that causally contributes to failure. Identify the incorrect behavior and the violated requirement or reasoning error.
- `other`: a supported cause outside those definitions, such as a defective target, comparator bug, or execution environment problem. Describe the actual cause.

Apply categories to individual causal findings, not indiscriminately to the whole solution. If a solution has an unrelated mistake, that does not erase a supported prompt/test mismatch. Multiple findings require multiple demonstrated causal contributions; an incidental defect is insufficient. Distinguish independent causes from interacting causes.

Treat HDF5 contents as the grader's expected outputs, not as proof of scientific correctness. They are not human ground-truth labels for the failure categories. An assertion failure alone establishes neither model error nor a specification defect.

## Initial evidence packet

Supply the current subproblem description, function interface, exact applicable system instructions and scientific background, raw response and extracted submitted code, test source, original grading results, and decoded HDF5 targets linked to test indices. Include necessary dependency declarations and harness/environment details.

Represent small targets completely. For large arrays, provide dtype, shape, a clearly labelled preview, and a tool-accessible lossless artifact. Preserve nested structures, complex values, sparse arrays, ordering, and numerical precision using the implementation's target loader.

Earlier responses, reasoning traces, descriptions, and cumulative code are available on demand rather than appended to the initial judge prompt. Separate current requirements from earlier material embedded in cumulative prompts, keeping the exact original request retrievable. If the judge needs earlier context to establish a cause, it must retrieve it.

## Investigation tools

Implement a custom Scout scanner with a bounded judge tool loop; the installed Scout `llm_scanner` interface does not expose a general investigation-tool argument.

- Retrieve a selected subproblem's prompt, response, reasoning, code, tests, and results.
- Retrieve or search the full transcript, with stable message/event references and pagination. Include system messages, supplied code, grader events, and any exposed reasoning; report unavailable content explicitly.
- Inspect complete targets and test helper source.
- Execute Python diagnostics in an isolated sandbox matching the appropriate grading environment.
- Rerun the exact original cumulative program, or an explicitly labelled diagnostic variant. Allow selected-test runs and replacement of one earlier implementation to test inheritance hypotheses.

Execute candidate code in sandboxes with no network and resource limits. Preserve tool calls, scripts, interpreter versions, stdout, stderr, timeouts, and modifications. Never overwrite the original submission or original grading evidence. Diagnostic test isolation may change shared state; record that distinction. For Verified, preserve the pass-in-either-environment rule and allow diagnostics in both environments.

Python and transcript tools require a smoke check on Hawk, including sandbox provisioning, target artifact access, and full-transcript retrieval. Configure judge model, reasoning, token/tool limits, per-call execution timeout, and total investigation budget. Budget exhaustion yields an explicit incomplete or unresolved assessment, not an invented cause.

## Causal origins and deduplication

Each finding records the affected step and the step(s) where the cause originated. Origins may be current, earlier, author-provided, external to a step, or unknown. Record a dependency path where supported; a passing earlier step can still contain an error exposed only by a later test.

Investigate inherited failures using earlier evidence and, when useful, minimal counterfactual experiments. Keep the originating category: an inherited specification defect is still a specification defect. Do not infer causality solely from code reuse or a previous failed score.

Assign stable cause IDs within each main-problem transcript using origin and the specific defect. Process failed steps in order, then reconcile matching causes across their assessments; origin alone is not a unique cause identifier. Preserve separate affected-step records and their evidence. Deduplicate by implementation, source transcript/epoch, and cause ID; do not automatically merge distinct model attempts or benchmarks.

## Output

One Scout result per failed step, labelled with its step ID. Store a list of unique supported categories as the classification value and retain the full structured assessment in metadata.

Assessment fields:

- Source identity: implementation, revisions/checksums, transcript, model, epoch, main problem, affected step, and original score.
- Status: resolved, partially resolved, or unresolved, with explicit completeness and limitations.
- Causes: cause ID, category, mechanism, origin step IDs/type, dependency path, evidence references, and demonstrated causal contribution.
- Evidence: prompt/test quotations, code locations, target references, diagnostic experiments, and alternative explanations considered.
- Investigation provenance: tools/context accessed, execution settings, judge model/configuration, and budget usage.

Unresolved assessments may have no supported categories. Partially resolved assessments retain established causes while explaining remaining uncertainty. Scanner implementation errors are run errors, not `other` findings.

## Implementation and validation sequence

1. Finalize category boundaries, output schema, and unresolved handling. Inspect representative default-harness logs from both implementations.
2. Build adapters and reproducible target access. Verify extracted submissions, scoring evidence, and reconstructed grading scripts against source logs. Missing artifacts must be explicit.
3. Build sandbox and retrieval tools; verify exact reruns before relying on counterfactual experiments.
4. Implement the judge, causal-origin tracking, and cause reconciliation. Do not provide reference solutions or historical audit findings to the judge, including through tools. Author-provided steps remain available because they were part of the evaluated harness.
5. Test extraction and output semantics with fixtures from both harnesses: supplied steps, retries, missing reasoning, empty submissions, timeouts, large targets, inherited failures, multiple causes, and unresolved cases.
6. Review a small pilot spanning both implementations and the four categories, with local and inherited causes. Use independent human labels for validation; report category-level precision/recall, origin agreement, unresolved frequency, and deduplication errors. Avoid tuning and evaluating on the same examples.
7. Add separate run examples, resume/caching support, and an export containing one row per affected-step/cause pair. Report both affected-step totals and distinct-cause totals. Scale after reviewing pilot quality and cost.

## Judge and token budget

Use `openrouter/z-ai/glm-5.3` with configurable reasoning effort. GLM-5.3 documents `low`, `high`, and `max`, rather than a medium setting. Choose `high`, the middle supported level, as the initial default. [Supported reasoning levels](https://openrouter.ai/z-ai/glm-5.3) checked 2026-10-07. Confirm provider support during the deployment smoke check; do not silently downgrade reasoning.

Set `generated_token_budget=8192` per failed subproblem, shared across all judge calls for that investigation. This includes reasoning tokens, tool-call arguments, schema-repair attempts, and the final assessment. Input tokens and tool-result text are measured separately; they do not consume this generation allowance. This fixed configurable allowance replaces the global $5 ceiling and monetary reservation scheme. Report actual token usage and estimated cost for iteration.

The harness maintains remaining allowance from provider completion usage, including reasoning exactly once. Set each request's generation limit to at most the remaining allowance, normally capped at 4096 tokens per investigation call. Reserve 1536 of the 8192 tokens for finalization. When the investigation allowance is nearly exhausted, disable further investigation tools and request the final structured assessment using the remaining reserve. Verify that the provider's output limit and usage accounting include reasoning; missing usage must not permit an unbounded loop. Persist usage across resume and charge retries against the same allowance.

Tell the judge its total allowance, remaining allowance, and finalization reserve in its initial instructions and update the remaining allowance before every subsequent call. Explain that reasoning, tool requests, and the final answer share the allowance. Explicitly instruct it to prioritize decisive experiments, finish before exhaustion, and report unresolved causes when evidence is insufficient. The allowance is enforced by the harness as well as communicated to the model.

Start with at most six investigation tool rounds per failed step, 30 seconds per Python diagnostic, and 120 seconds total diagnostic execution; exact grading reruns that need longer must be explicitly distinguished. Keep these limits configurable and tell the judge its remaining tool/time limits. Cache deterministic retrieval and experiments. Use deterministic cause reconciliation initially so no additional unbudgeted judge calls are introduced.

Every failed step gets an output record. If the judge exhausts its allowance before producing valid output, preserve its evidence and return an explicit incomplete/unresolved assessment with `budget_exhausted`. Do not invent a cause or start an unbudgeted recovery call. Record the configured allowance, consumed tokens, reasoning setting, and termination reason in result metadata.

## Pilot review

Review whether the initial token/tool/time limits give enough diagnostic coverage; adjust the configurable per-subproblem limits after the Hawk pilot.

Planning only: no scanner implementation or paid inference has been launched.
