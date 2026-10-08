"""Registered Inspect eval for prospective benchmark defect investigation."""

import hashlib
from pathlib import Path

from inspect_ai import Task, task

from .agent import investigate
from .dataset import get_dataset
from .harness import REVISIONS, ROOT, interpreters
from .schema import Implementation, Limits
from .scorer import validate_findings


def _sandbox_spec(sandbox_type: str, config: str) -> tuple:
    if sandbox_type == "k8s":
        from k8s_sandbox import K8sSandboxEnvironmentConfig

        return (
            sandbox_type,
            K8sSandboxEnvironmentConfig(
                chart=str(ROOT / "sandbox" / "chart"),
                values=Path(config),
                restarted_container_behavior="raise",
            ),
        )
    return sandbox_type, config


@task
def scicode_defect_investigation(
    implementation: Implementation = "scicode",
    problem_ids: list[str] | None = None,
    investigate_steps: list[str] | None = None,
    include_dev: bool = False,
    provide_scientific_background: bool = True,
    targets_path: str | None = None,
    cache_dir: str | None = None,
    sandbox_type: str = "docker",
    sandbox_config: str | None = None,
    grading_environments: str = "both",
    generated_token_budget: int | None = None,
    base_token_budget: int = 10_000,
    tokens_per_subproblem: int = 20_000,
    finalization_reserve: int | None = None,
    per_call_tokens: int = 8_000,
    tool_call_limit: int | None = None,
    model_call_limit: int | None = None,
    diagnostic_timeout: int = 120,
    grading_timeout: int = 1800,
    reasoning_effort: str = "high",
    time_limit: int | None = None,
    max_prompt_chars: int = 500_000,
) -> Task:
    interpreters(implementation, grading_environments)
    if sandbox_type not in {"docker", "k8s"}:
        raise ValueError(
            "sandbox_type must be docker or k8s; host execution is unsupported."
        )
    if base_token_budget < 0 or tokens_per_subproblem < 1:
        raise ValueError(
            "Budget base must be nonnegative and per-subproblem allowance positive."
        )
    config = sandbox_config or str(
        ROOT
        / "sandbox"
        / f"{implementation}.{'yaml' if sandbox_type == 'k8s' else 'compose.yaml'}"
    )
    dataset = get_dataset(
        implementation,
        problem_ids,
        investigate_steps,
        include_dev,
        provide_scientific_background,
        targets_path,
        cache_dir,
        max_prompt_chars,
    )
    for sample in dataset:
        count = len(sample.metadata["selected_steps"])
        budget = (
            generated_token_budget
            if generated_token_budget is not None
            else base_token_budget + tokens_per_subproblem * count
        )
        limits = Limits(
            generated_tokens=budget,
            finalization_reserve=(
                finalization_reserve
                if finalization_reserve is not None
                else max(5000, budget // 10)
            ),
            per_call_tokens=per_call_tokens,
            tool_calls=tool_call_limit
            if tool_call_limit is not None
            else 10 + 20 * count,
            model_calls=model_call_limit
            if model_call_limit is not None
            else 20 + 20 * count,
            diagnostic_timeout=diagnostic_timeout,
            grading_timeout=grading_timeout,
            reasoning_effort=reasoning_effort,
        )
        sample.metadata.update(
            {
                "sandbox_config": config,
                "sandbox_definition": Path(config).read_text(),
                "sandbox_config_sha256": hashlib.sha256(
                    Path(config).read_bytes()
                ).hexdigest(),
                "grading_environments": grading_environments,
                "generation_limits": limits.model_dump(),
                "prompt_version": 1,
                "budget_policy": {
                    "mode": "fixed"
                    if generated_token_budget is not None
                    else "per_subproblem",
                    "selected_subproblems": count,
                    "base_tokens": base_token_budget,
                    "tokens_per_subproblem": tokens_per_subproblem,
                },
            }
        )
    fallback_limits = Limits()
    effective_time_limit = (
        time_limit
        if time_limit is not None
        else max(
            max(1800, 900 + 300 * len(sample.metadata["selected_steps"]))
            for sample in dataset
        )
    )
    return Task(
        dataset=dataset,
        solver=investigate(fallback_limits, grading_environments),
        scorer=validate_findings(fallback_limits, grading_environments),
        sandbox=_sandbox_spec(sandbox_type, config),
        time_limit=effective_time_limit,
        version=1,
        metadata={
            "implementation": implementation,
            "harness_revision": REVISIONS[implementation],
            "measurement": "prospective defect discovery; semantic review required",
        },
    )
