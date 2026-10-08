# SciCode defect investigation

An **Inspect eval**, run with Hawk eval-set, that investigates original SciCode and SciCode-Verified directly. It does not scan existing model transcripts and does not use Scout.

Each main problem is a sample. For every selected subproblem, the agent asks whether a correct solution could fail grading and whether a plausible incorrect solution could pass. It has the full specification, tests, helper source, and programmatic access to a lossless HDF5 shard containing only this problem's targets.

The starting investigator is `openrouter/z-ai/glm-5.3`, with high reasoning effort. Model selection stays in the run configuration.

## Evidence and interpretation

The agent constructs cumulative candidate implementations, runs canonical grading, and records independent semantic checks. Submitted witnesses must be plausible from the standard prompt: no hardcoded test answers, grader file access, comparator manipulation, or process-exit bypasses. A conservative syntax screen rejects obvious violations; it cannot prove absence of hardcoding or establish plausibility. Those remain explicit review requirements.

Reports cover both directions for every selected step using `demonstrated`, `suspected`, `not_found`, or `inconclusive`, with separate semantic and grader confidence ratings and rationales. `not_found` does not prove the benchmark sound. Demonstrated false acceptance requires a successful executable counterexample check plus a semantic argument. False rejection requires canonical failure and positive independent correctness reasoning. Timeout and infrastructure failures cannot establish a demonstrated finding.

The scorer replays demonstrated candidates in a separate grading service. It confirms **grading behavior**, not semantic truth or submission plausibility. Reports retain both as unreviewed until a person reviews them. Discovery counts are not investigator accuracy.

## Setup

From the `scicode-scanners` project directory, install using the project's normal environment workflow, for example:

```bash
uv pip install -e '.[hawk,test]'
```

Docker is required locally; Hawk uses `inspect-k8s-sandbox`. Both sandbox services use image digests pinned in `sandbox/`. Network access is disabled for sandbox code. The evaluation runner downloads and checksum-verifies the released HDF5 file once. Original SciCode uses `xantheocracy/scicode-mirror` at revision `2f903a64a1c4eb62c7c649b0caf7ecb2c0217590`, preserving the upstream SHA-256 and avoiding Google Drive download quotas; `targets_path` can instead name a local, checksum-matching release. Generated problem shards and helper files are cached under `~/.cache/scicode_defect_investigation`, configurable with `cache_dir`.

No reference solutions or previous audit findings are sent to the agent. Benchmark-provided prerequisite code is retained. Original includes 65 evaluation problems; the pinned Verified release contains 64. Original supports its development split via `include_dev=true`; Verified's release has no development split.

## Small local smoke

The following uses real inference and requires provider credentials. Implementation tests below require none.

```bash
inspect eval scicode_scanners/scicode_defect_investigation \
  --model openrouter/z-ai/glm-5.3 \
  -T implementation=scicode \
  -T 'problem_ids=["5"]' -T 'investigate_steps=["5.1"]' \
  -T generated_token_budget=8000 -T finalization_reserve=1000 \
  -T per_call_tokens=3000 -T model_call_limit=20 -T tool_call_limit=16 \
  -T time_limit=600 --max-samples 1
```

Repeat with `implementation=scicode_verified` for Verified. The agent retains the whole problem context but investigates only step 5.1. The workspace's original interpreter is `python`; Verified uses `/opt/scicode-2025/bin/python` (or `2024`). Target access:

```python
from target_access import load_targets

expected = load_targets("5.1")
```

Candidate and semantic-check tools accept a mapping from step IDs to complete source for every generated prerequisite through the selected step. Supplied steps are inserted automatically. Workspace tools can edit diagnostic files; the canonical grader always stages pristine host-authoritative resources into a new working directory and Python process.

## Hawk

`hawk-smoke.yaml` configures **two samples total**, one selected step from each implementation. `hawk-pilot.yaml` configures five main problems per implementation (ten samples total). Both disable automatic retries and use Hawk-managed concurrency. Do not set top-level `max_samples` or `max_tasks`: Hawk supplies these internally, so extra config keys produce duplicate keyword errors. Each task has its own isolated default and grader services.

Both configurations inherit Hawk's deployment-pinned Inspect and Kubernetes sandbox dependencies. Do not add an `inspect-k8s-sandbox` requirement: Hawk already requires its own Git source, and another source conflicts during installation. The task packages a Helm chart that preserves disabled service-account token mounts. Do not add a recent `inspect-ai==...` override: Hawk's package-age cutoff can exclude newly published releases. The project requires Inspect >=0.3.263; local validation used 0.3.277.

Before launch, push the implementation and replace `@main` in the package URL with that commit SHA. The supplied configs cannot fetch an unpushed local implementation. Check `hawk version` and use the CLI matching your deployment. For Hawk 3:

```bash
hawk eval-set run scicode_scanners/defect_investigation/hawk-smoke.yaml
```

This launches paid inference. No run is launched by importing the task or executing its tests. Provider secrets, if required by your deployment, are passed by name at launch rather than embedded in YAML.

| Configuration | Generated tokens per sample | Final-report reserve | Model calls | Time per sample |
| --- | ---: | ---: | ---: | ---: |
| Smoke | 8,000 | 3,000 | 20 | 10 minutes |
| Pilot | 100,000 | 5,000 | 150 | 60 minutes |

Finalization uses low reasoning effort to leave room for the structured report. The solver limits each model call to the remaining allowance and charges provider `output_tokens`, which includes reasoning for the OpenRouter/OpenAI-compatible route. It does not add reasoning tokens again. Missing usage conservatively charges the full call cap; reported usage above the cap fails rather than permitting budget drift. Input/history/tool-observation tokens are additional billed usage. Budget ceilings do not request minimum spend. Model billing/token accounting must still be verified on the first real GLM smoke; the automated tests use scripted outputs.

Diagnostics have a 120-second timeout. Canonical grading and scorer replay use the selected harness's 1,800-second timeout, subject to the overall sample time limit. Checkpointing is not enabled. A time or budget limit can leave no report; partial evidence remains in the log.

## Export and review

```bash
python -m scicode_scanners.defect_investigation.export path/to/run.eval results/
```

`reports.jsonl` contains complete candidates, executed programs, checks, stdout/stderr, confidence, provenance, replay results, and usage. `reports.md` provides a readable findings overview. Stable evidence IDs are `X:N`; specification/test references are `P:STEP` and `T:STEP:INDEX` (one-based).

The sample store contains `defect_report`, `defect_evidence`, `defect_validation`, token counts, and termination reason. Partial and invalid reports remain distinct from absence of a defect. Shared root causes use the same `defect_key` and are counted once per direction within a problem.

## Validation

```bash
python -m pytest tests/test_defect_investigation.py -q
RUN_DEFECT_DOCKER_TESTS=1 python -m pytest tests/test_defect_investigation_docker.py -q
```

The Docker tests use scripted agent outputs and synthetic HDF5 fixtures, exercising real Inspect sandbox setup, candidate grading, independent checks, scorer replay, workspace isolation, and log export for both implementations. They spend no inference tokens. They do not replace the first real-data/model/Hawk smoke.

Bundled benchmark assets and harness snapshots are hashed in `asset_manifest.json`; provenance is included in each sample. See `vendor/README.md` for source revisions and licenses.

Inspect architecture references: [custom solvers](https://inspect.aisi.org.uk/solvers.html), [sandboxes](https://inspect.aisi.org.uk/sandboxing.html), and [Hawk eval-set configuration](https://hawk.metr.org/user-guide/eval-set-config-reference/).
