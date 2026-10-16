"""Generic causal failure classification for both SciCode default harnesses."""

import hashlib
import json
import re
from contextlib import nullcontext

from inspect_ai.model import get_model
from inspect_scout import Reference, Result, Scanner, Transcript, scanner

from .adapters import REVISIONS, Implementation, load_cases
from .assets import (
    TARGETS,
    decode_targets,
    shard_targets,
    target_file,
    target_metadata,
)
from .checkpoints import CheckpointStore, hawk_results_uri
from .judge import investigate
from .sandbox import IMAGES, interpreters, investigation_sandbox
from .schema import Limits
from .tools import Investigation

VERSION = 11

POSITIVE_CATEGORIES = {
    "underspecified",
    "wrongly_specified",
    "operational_error",
    "other",
}


def is_positive_failure(causes):
    """Flag established non-model causes; unknown or model-only causes are non-hits."""
    return any(cause["category"] in POSITIVE_CATEGORIES for cause in causes)


def reconcile(assessment, implementation, transcript_id):
    rows = []
    for cause in assessment.causes:
        detail = cause.model_dump()
        # Conservative deterministic matching: origin/category/specific defect must all agree.
        key = [
            implementation,
            transcript_id,
            "submitted"
            if cause.origin_type in {"current", "earlier"}
            else cause.origin_type,
            sorted(cause.origin_steps),
            cause.category,
            " ".join(cause.defect_key.casefold().split()),
        ]
        detail["cause_id"] = hashlib.sha256(json.dumps(key).encode()).hexdigest()[:20]
        rows.append(detail)
    return rows


async def smoke_check(env, cases, sid):
    """Confirm each grading interpreter can import its stack and decode targets."""
    implementation = cases.implementation
    module = "process_data" if implementation == "scicode" else "scicode.parse.parse"
    filename = TARGETS[implementation]["filename"]
    target_args = f"{sid!r}, {len(cases.steps[sid]['record']['test_cases'])}" + (
        f", {filename!r}" if implementation == "scicode_verified" else ""
    )
    checks = {}
    for label, python in interpreters(cases).items():
        check = await env.exec(
            [
                python,
                "-c",
                (
                    "import sys, numpy, scipy, h5py; "
                    f"from {module} import process_hdf5_to_tuple; "
                    f"assert len(process_hdf5_to_tuple({target_args})) > 0; "
                    "print(sys.version); print(numpy.__version__, scipy.__version__, h5py.__version__)"
                ),
            ],
            timeout=30,
            timeout_retry=False,
        )
        if not check.success:
            raise RuntimeError(f"Hawk sandbox smoke check failed: {check.stderr}")
        checks[label] = check.stdout
    return checks


@scanner(messages="all", events="all", version=VERSION)
def failure_classification(
    implementation: Implementation,
    generated_token_budget: int = 32768,
    finalization_reserve: int = 8192,
    per_call_tokens: int = 4096,
    tool_rounds: int = 12,
    python_timeout: int = 30,
    diagnostic_seconds: int = 120,
    reasoning_effort: str = "high",
    targets_path: str | None = None,
    execution: bool = True,
    sandbox_image: str | None = None,
    runtime_class: str = "gvisor",
    checkpoint_uri: str | None = None,
    dry_run: bool = False,
    smoke_only: bool = False,
) -> Scanner[Transcript]:
    """Classify each failed step.

    dry_run performs extraction/target checks without inference or sandboxes; smoke_only also
    provisions a sandbox and checks its interpreters. execution=False is inspection-only.
    """
    if implementation not in {"scicode", "scicode_verified"}:
        raise ValueError("implementation must be scicode or scicode_verified.")
    if smoke_only and not execution:
        raise ValueError("smoke_only checks the sandbox; it requires execution.")
    limits = Limits(
        generated_token_budget=generated_token_budget,
        finalization_reserve=finalization_reserve,
        per_call_tokens=per_call_tokens,
        tool_rounds=tool_rounds,
        python_timeout=python_timeout,
        diagnostic_seconds=diagnostic_seconds,
        reasoning_effort=reasoning_effort,
    )
    mode = "sandbox" if execution else "inspection_only"
    image = (sandbox_image or IMAGES[implementation]) if execution else None

    async def scan(t: Transcript) -> list[Result]:
        cases = await load_cases(t, implementation)
        failed = [
            s
            for s in cases.steps
            if not cases.steps[s]["provided"] and cases.scores[s] == 0
        ]
        if not failed:
            return [
                Result(
                    label="no_failed_subproblems",
                    value=False,
                    answer="no_failed_subproblems",
                    explanation="All scored steps passed; no judge calls made.",
                )
            ]
        target = await target_file(implementation, targets_path)
        packets = {}
        for sid in failed:
            packet = cases.packet(sid)
            packet["expected_targets"] = [
                {
                    "reference": f"T:{sid}:{i + 1}:target",
                    "metadata": target_metadata(value),
                }
                for i, value in enumerate(
                    decode_targets(
                        target,
                        implementation,
                        sid,
                        len(cases.steps[sid]["record"]["test_cases"]),
                    )
                )
            ]
            packets[sid] = packet
        if dry_run:
            return [
                Result(
                    label=sid,
                    value=False,
                    answer="extraction_checked",
                    metadata={
                        "evidence_packet": packets[sid],
                        "code_sha256": hashlib.sha256(
                            cases.grading_script(sid).encode()
                        ).hexdigest(),
                    },
                    explanation="Extraction and exact target decoding checked; no judge call.",
                )
                for sid in failed
            ]
        shard = (
            shard_targets(
                target, [s for s in cases.steps if not cases.steps[s]["provided"]]
            )
            if execution
            else None
        )
        results, previous = [], []
        for sid in failed:
            checkpoint = CheckpointStore(
                checkpoint_uri or hawk_results_uri(),
                {
                    "scanner_version": VERSION,
                    "implementation": implementation,
                    "transcript": t.transcript_id,
                    "source_uri": t.source_uri,
                    "step": sid,
                    "limits": limits.model_dump(),
                    "investigation_mode": mode,
                    "image": image,
                    "model": get_model().name,
                    "targets": TARGETS[implementation]["sha256"],
                },
            )
            # A fresh sandbox per failed step; experiments never see earlier steps' diagnostics.
            async with (
                investigation_sandbox(cases, shard, image, runtime_class)
                if execution
                else nullcontext()
            ) as env:
                if smoke_only:
                    return [
                        Result(
                            label="hawk_smoke",
                            value=False,
                            answer="smoke_passed",
                            metadata={
                                "implementation": implementation,
                                "image": image,
                                "checks": await smoke_check(env, cases, sid),
                                "failed_steps": len(failed),
                            },
                            explanation="Source extraction, target artifacts, and Hawk sandbox interpreters checked without model inference.",
                        )
                    ]
                assessment, provenance = await investigate(
                    get_model(),
                    Investigation(cases, sid, target, env, limits),
                    packets[sid],
                    limits,
                    previous,
                    checkpoint,
                )
            causes = reconcile(assessment, implementation, t.transcript_id)
            previous.extend(
                {
                    k: c[k]
                    for k in (
                        "cause_id",
                        "category",
                        "origin_type",
                        "origin_steps",
                        "defect_key",
                        "mechanism",
                    )
                }
                for c in causes
            )
            source = {
                "implementation": implementation,
                "source_uri": t.source_uri,
                "transcript_id": t.transcript_id,
                "epoch": cases.epoch,
                "source_model": t.model,
                "main_problem": cases.problem["problem_id"],
                "affected_step": sid,
                "original_score": cases.scores[sid],
                "adapter_revision": REVISIONS[implementation],
                "source_revision": cases.source_revision,
                "targets_sha256": TARGETS[implementation]["sha256"],
                "sandbox_image": image,
                "investigation_mode": mode,
                "settings": cases.settings,
                "judge_model": str(get_model().name),
            }
            result = assessment.model_dump()
            result["causes"] = causes
            results.append(
                Result(
                    label=sid,
                    value=is_positive_failure(causes),
                    answer=assessment.status,
                    explanation=assessment.explanation,
                    references=[
                        Reference(
                            type="message" if kind == "M" else "event",
                            id=identity,
                            cite=f"{kind}:{identity}",
                        )
                        for kind, identity in sorted(
                            {
                                match
                                for c in causes
                                for e in c["evidence"]
                                for match in re.findall(r"\b([ME]):([A-Za-z0-9_-]+)", e)
                            }
                        )
                    ],
                    metadata={
                        "categories": sorted({c["category"] for c in causes}),
                        "source": source,
                        "investigation_mode": mode,
                        "assessment": result,
                        "investigation": provenance,
                    },
                )
            )
        return results

    return scan
