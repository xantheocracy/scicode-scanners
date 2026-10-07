"""Inspect task for the released SciCode-Verified benchmark.

Run from this checkout with ``inspect eval scicode_inspect.py --model ...``.
"""

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageUser
from inspect_ai.scorer import Metric, SampleScore, Score, Scorer, Target, metric, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import download, sandbox

from eval_clean.run_deepseek_eval import build_prompt, extract_code

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "scicode_verified"
VENDOR = ROOT / "eval_clean" / "vendor"
PROVIDED_STEPS = {"13.6", "62.1", "76.3"}
GRADING_ENVS = {
    "2024": "/opt/scicode-2024/bin/python",
    "2025": "/opt/scicode-2025/bin/python",
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
TARGETS_REVISION = "eea11a866be6860725258702b39ef8651ed26abd"
TARGETS_SHA256 = "8fb6e575b7b6dda5e48b04dea338fc6af4fe185774b8f19221c96945df9b4142"


def verify_file(path: Path, expected_md5: str) -> None:
    """Check a released artifact against the dataset manifest."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Download the SciCode-Verified data release (see README)."
        )
    with path.open("rb") as file:
        actual = hashlib.file_digest(file, "md5").hexdigest()
    if actual != expected_md5:
        raise ValueError(f"Manifest checksum mismatch for {path}: {actual}")


def get_dataset(
    targets_path: str | None = None,
    *,
    require_targets: bool = True,
    download_targets: bool = False,
) -> MemoryDataset:
    """Load one sample per main problem and verify the release artifacts."""
    manifest = json.loads((DATA_DIR / "manifest.json").read_text())
    problems_path = DATA_DIR / "problems_test.jsonl"
    verify_file(problems_path, manifest["problems_test_jsonl_md5"])
    targets = (
        Path(targets_path).expanduser().resolve()
        if targets_path is not None
        else DATA_DIR / "test_data_cleaned.h5"
    )
    if require_targets:
        if download_targets and targets_path is None and not targets.is_file():
            targets = download(
                "https://huggingface.co/datasets/shhu2001/SciCode-Verified/resolve/"
                f"{TARGETS_REVISION}/test_data_cleaned.h5",
                TARGETS_SHA256,
                Path.home() / ".cache" / "scicode_verified" / "test_data_cleaned.h5",
                timeout=60,
            )
        verify_file(targets, manifest["h5_md5"])
    records = [json.loads(line) for line in problems_path.read_text().splitlines()]
    if [str(record["problem_id"]) for record in records] != manifest["problem_order"]:
        raise ValueError("Problem order does not match the release manifest.")
    samples = [
        Sample(
            id=str(record["problem_id"]),
            input=str(record["problem_id"]),
            metadata=record,
            files={"test_data_cleaned.h5": str(targets)} if require_targets else {},
        )
        for record in records
    ]
    if require_targets:
        prepare_target_shards(samples, targets, manifest["h5_md5"])
    return MemoryDataset(samples, name=f"SciCode-Verified-{manifest['version']}")


def prepare_target_shards(
    samples: list[Sample], targets: Path, source_md5: str
) -> None:
    """Copy only a problem's scored HDF5 groups into its sandbox artifact."""
    import h5py

    cache = Path.home() / ".cache" / "scicode_verified" / f"shards_v1_{source_md5}"
    pending = []
    for sample in samples:
        step_ids = list(
            dict.fromkeys(
                str(step["step_number"])
                for step in (sample.metadata or {})["sub_steps"]
                if step["step_number"] not in PROVIDED_STEPS
            )
        )
        key = hashlib.sha256(json.dumps(step_ids).encode()).hexdigest()
        shard = cache / f"{key}.h5"
        sample.files = {"test_data_cleaned.h5": str(shard)}
        if not shard.is_file() or shard.stat().st_size == 0:
            pending.append((shard, step_ids))
    if not pending:
        return
    cache.mkdir(parents=True, exist_ok=True)
    with h5py.File(targets, "r") as source:
        for shard, step_ids in pending:
            with NamedTemporaryFile(dir=cache, suffix=".partial", delete=False) as tmp:
                temporary = Path(tmp.name)
            try:
                with h5py.File(temporary, "w") as output:
                    for step_id in step_ids:
                        if step_id not in source:
                            raise ValueError(
                                f"Released targets missing step {step_id}."
                            )
                        source.copy(step_id, output)
                os.replace(temporary, shard)
            finally:
                temporary.unlink(missing_ok=True)


@solver
def solve_scicode(provide_scientific_background: bool = True) -> Solver:
    """Use the canonical single-turn cumulative prompt for each generated step."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        codes: list[str] = []
        # Store cumulative code independently of the transcript for later re-scoring.
        state.store.set("scicode_codes", codes)
        for index, step in enumerate(state.metadata["sub_steps"]):
            if step["step_number"] in PROVIDED_STEPS:
                reference = (
                    VENDOR / "eval_data" / f"{step['step_number']}.txt"
                ).read_text()
                codes.append(extract_code(reference))
            else:
                prompt = build_prompt(
                    state.metadata, index, codes, provide_scientific_background
                )
                # Each request contains the full previous code, without previous replies.
                state.messages = [ChatMessageUser(content=prompt)]
                state = await generate(state)
                codes.append(extract_code(state.output.completion))
            state.store.set("scicode_codes", list(codes))
        return state

    return solve


@metric
def problem_accuracy() -> Metric:
    """Fraction of main problems with every scored step correct."""

    def compute(scores: list[SampleScore]) -> float:
        if not scores:
            return float("nan")
        values = [cast(dict[str, float], score.score.value) for score in scores]
        return sum(
            all(value == 1 for value in steps.values()) for steps in values
        ) / len(values)

    return compute


@metric
def subproblem_accuracy() -> Metric:
    """Fraction of scored steps correct, weighted by the number of steps."""

    def compute(scores: list[SampleScore]) -> float:
        values = [
            value
            for score in scores
            for value in cast(dict[str, float], score.score.value).values()
        ]
        return sum(values) / len(values) if values else float("nan")

    return compute


def grading_script(problem: dict[str, Any], index: int, codes: list[str]) -> str:
    """Construct the canonical cumulative program and this step's tests."""
    step = problem["sub_steps"][index]
    sections = [
        problem["required_dependencies"],
        *codes[: index + 1],
        "from scicode.parse.parse import process_hdf5_to_tuple",
        f"targets = process_hdf5_to_tuple({step['step_number']!r}, "
        f"{len(step['test_cases'])}, 'test_data_cleaned.h5')",
    ]
    for test_index, test_case in enumerate(step["test_cases"]):
        sections.extend([f"target = targets[{test_index}]", test_case])
    return "\n\n".join(sections)


@scorer(metrics=[problem_accuracy(), subproblem_accuracy()])
def verify_scicode(
    timeout: int = 1800, grading_environments: str = "both", full_envs: bool = False
) -> Scorer:
    """Accept a step if its tests pass in any selected sandbox interpreter."""
    if grading_environments not in {"both", *GRADING_ENVS}:
        raise ValueError("grading_environments must be 'both', '2024', or '2025'.")
    selected = (
        GRADING_ENVS
        if grading_environments == "both"
        else {grading_environments: GRADING_ENVS[grading_environments]}
    )

    async def score(state: TaskState, target: Target) -> Score:
        codes = state.store.get("scicode_codes")
        if codes is None or len(codes) != len(state.metadata["sub_steps"]):
            raise ValueError("Missing completed SciCode step code in the sample store.")
        values: dict[str, float] = {}
        details: dict[str, Any] = {}
        for index, step in enumerate(state.metadata["sub_steps"]):
            step_id = step["step_number"]
            if step_id in PROVIDED_STEPS:
                continue
            code = grading_script(state.metadata, index, codes)
            per_env = {}
            for label, python in selected.items():
                # Check benchmark resources separately so instrument errors raise.
                check = await sandbox().exec(
                    [
                        python,
                        "-c",
                        "from scicode.compare.cmp import cmp_tuple_or_list, are_dicts_close; "
                        "from scicode.parse.parse import process_hdf5_to_tuple; "
                        f"process_hdf5_to_tuple({step_id!r}, {len(step['test_cases'])}, "
                        "'test_data_cleaned.h5')",
                    ],
                    env=THREAD_ENV,
                    timeout=60,
                )
                if not check.success:
                    raise RuntimeError(
                        f"Grading setup failed in {label}: {check.stderr}"
                    )
                try:
                    result = await sandbox().exec(
                        [python, "-c", code],
                        env=THREAD_ENV,
                        timeout=timeout,
                        timeout_retry=False,
                    )
                    per_env[label] = {
                        "passed": result.success,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                except TimeoutError:
                    per_env[label] = {"passed": False, "stderr": "Execution timed out."}
                if per_env[label]["passed"] and not full_envs:
                    break
            values[step_id] = float(
                any(result["passed"] for result in per_env.values())
            )
            details[step_id] = per_env
        return Score(value=values, metadata={"per_environment": details})

    return score


@task
def scicode_verified(
    provide_scientific_background: bool = True,
    timeout: int = 1800,
    grading_environments: str = "both",
    full_envs: bool = False,
    targets_path: str | None = None,
    generate_only: bool = False,
    download_targets: bool = False,
) -> Task:
    """Run SciCode-Verified v2 using Inspect's models, logging, and sandboxes.

    Args:
        provide_scientific_background: Include expert-written background in prompts.
        timeout: Maximum grading time in seconds per step and environment.
        grading_environments: Grade in 'both', '2024', or '2025' environments.
        full_envs: Run all selected environments even after a passing result.
        targets_path: Override the local corrected HDF5 path; its hash is verified.
        generate_only: Generate without requiring targets or starting Docker.
        download_targets: Download pinned targets into the local cache if absent.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive.")
    return Task(
        dataset=get_dataset(
            targets_path,
            require_targets=not generate_only,
            download_targets=download_targets,
        ),
        solver=solve_scicode(provide_scientific_background),
        scorer=None
        if generate_only
        else verify_scicode(timeout, grading_environments, full_envs),
        sandbox=None
        if generate_only
        else ("docker", str(ROOT / "eval_clean" / "inspect" / "compose.yaml")),
        version=1,
        metadata={
            "dataset_version": "v2",
            "grading_environments": grading_environments,
        },
    )
