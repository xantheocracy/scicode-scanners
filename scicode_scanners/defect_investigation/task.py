"""Registered Inspect eval for prospective benchmark defect investigation."""

import hashlib
from pathlib import Path

from inspect_ai import Task, task

from .agent import investigate
from .dataset import get_dataset
from .harness import REVISIONS, ROOT, interpreters
from .schema import Implementation, Limits
from .scorer import validate_findings


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
    generated_token_budget: int = 100_000,
    finalization_reserve: int = 5_000,
    per_call_tokens: int = 8_000,
    tool_call_limit: int = 140,
    model_call_limit: int = 150,
    diagnostic_timeout: int = 120,
    grading_timeout: int = 1800,
    reasoning_effort: str = "high",
    time_limit: int = 3600,
    max_prompt_chars: int = 500_000,
) -> Task:
    interpreters(implementation, grading_environments)
    if sandbox_type not in {"docker", "k8s"}:
        raise ValueError(
            "sandbox_type must be docker or k8s; host execution is unsupported."
        )
    limits = Limits(
        generated_tokens=generated_token_budget,
        finalization_reserve=finalization_reserve,
        per_call_tokens=per_call_tokens,
        tool_calls=tool_call_limit,
        model_calls=model_call_limit,
        diagnostic_timeout=diagnostic_timeout,
        grading_timeout=grading_timeout,
        reasoning_effort=reasoning_effort,
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
            }
        )
    return Task(
        dataset=dataset,
        solver=investigate(limits, grading_environments),
        scorer=validate_findings(limits, grading_environments),
        sandbox=(sandbox_type, config),
        time_limit=time_limit,
        version=1,
        metadata={
            "implementation": implementation,
            "harness_revision": REVISIONS[implementation],
            "measurement": "prospective defect discovery; semantic review required",
        },
    )
