"""Generic causal failure classification for both SciCode default harnesses."""

import hashlib
import json
import re

from inspect_ai.model import get_model
from inspect_scout import Reference, Result, Scanner, Transcript, scanner

from .adapters import REVISIONS, Implementation, load_cases
from .assets import TARGETS, decode_targets, target_file, typed
from .checkpoints import CheckpointStore, hawk_results_uri
from .judge import investigate
from .schema import Limits
from .tools import Investigation


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


@scanner(messages="all", events="all", version=3)
def failure_classification(
    implementation: Implementation,
    generated_token_budget: int = 8192,
    finalization_reserve: int = 1536,
    per_call_tokens: int = 4096,
    tool_rounds: int = 6,
    reasoning_effort: str = "high",
    targets_path: str | None = None,
    checkpoint_uri: str | None = None,
    dry_run: bool = False,
) -> Scanner[Transcript]:
    """Classify each failed step; dry_run performs extraction/target checks without inference."""
    if implementation not in {"scicode", "scicode_verified"}:
        raise ValueError("implementation must be scicode or scicode_verified.")
    limits = Limits(
        generated_token_budget=generated_token_budget,
        finalization_reserve=finalization_reserve,
        per_call_tokens=per_call_tokens,
        tool_rounds=tool_rounds,
        reasoning_effort=reasoning_effort,
    )

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
                    value=[],
                    answer="no_failed_subproblems",
                    explanation="All scored steps passed; no judge calls made.",
                )
            ]
        target = await target_file(implementation, targets_path)
        packets = {}
        for sid in failed:
            packet = cases.packet(sid)
            packet["expected_targets"] = [
                {"reference": f"T:{sid}:{i + 1}:target", "value": typed(value)}
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
                    value=[],
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
        results, previous = [], []
        for sid in failed:
            investigation = Investigation(cases, sid, target)
            checkpoint = CheckpointStore(
                checkpoint_uri or hawk_results_uri(),
                {
                    "scanner_version": 3,
                    "implementation": implementation,
                    "transcript": t.transcript_id,
                    "source_uri": t.source_uri,
                    "step": sid,
                    "limits": limits.model_dump(),
                    "investigation_mode": "inspection_only",
                    "model": get_model().name,
                    "targets": TARGETS[implementation]["sha256"],
                },
            )
            assessment, provenance = await investigate(
                get_model(),
                investigation,
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
                "investigation_mode": "inspection_only",
                "settings": cases.settings,
                "judge_model": str(get_model().name),
            }
            result = assessment.model_dump()
            result["causes"] = causes
            results.append(
                Result(
                    label=sid,
                    value=sorted({c["category"] for c in causes}),
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
                        "source": source,
                        "investigation_mode": "inspection_only",
                        "assessment": result,
                        "investigation": provenance,
                    },
                )
            )
        return results

    return scan
