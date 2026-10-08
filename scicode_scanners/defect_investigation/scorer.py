"""Report validation and independent canonical replay, without a semantic truth oracle."""

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState

from .schema import Limits, Report
from .tools import Investigation, validate_report


@scorer(metrics={"report_valid": [mean()], "replay_confirmed": [mean()]})
def validate_findings(limits: Limits, grading_environments: str = "both") -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        raw = state.store.get("defect_report")
        if raw is None:
            return Score(
                value={"report_valid": 0, "replay_confirmed": 0},
                explanation="No valid final report. Partial experiments are retained in the sample store.",
                metadata={
                    "termination": state.store.get("defect_termination", "interrupted"),
                    "validation_status": "missing_report",
                },
            )
        report = Report.model_validate(raw)
        evidence = state.store.get("defect_evidence", [])
        validate_report(report, state.metadata, evidence)
        refs = {item["id"]: item for item in evidence}
        investigation = Investigation(
            state,
            Limits.model_validate(state.metadata["generation_limits"])
            if "generation_limits" in state.metadata
            else limits,
            grading_environments,
        )
        replays = {}
        details = []
        for finding in report.findings:
            validation = {
                "step_id": finding.step_id,
                "direction": finding.direction,
                "defect_key": finding.defect_key,
                "status": finding.status,
                "semantic_review": "unreviewed",
                "plausibility_review": "unreviewed",
            }
            if finding.status == "demonstrated":
                original = refs[finding.candidate_evidence]
                if original["id"] not in replays:
                    replays[original["id"]] = await investigation.canonical_grade(
                        original["step_id"], original["codes"]
                    )
                replay = replays[original["id"]]
                expected = finding.direction == "false_acceptance"
                confirmed = (
                    replay["kind"] == "canonical_grade"
                    and replay["passed"] is expected
                    and not replay.get("instrument_error")
                )
                confirmed = confirmed and not any(
                    r.get("timed_out")
                    for r in replay.get("per_environment", {}).values()
                )
                validation.update(
                    {
                        "replay_confirmed": confirmed,
                        "replay_evidence": replay["id"],
                        "candidate_evidence": original["id"],
                    }
                )
            details.append(validation)
        confirmed_keys = {
            (item["direction"], item["defect_key"])
            for item in details
            if item.get("replay_confirmed")
        }
        values = {"report_valid": 1, "replay_confirmed": len(confirmed_keys)}
        for direction in ("false_rejection", "false_acceptance"):
            values[f"{direction}_replayed"] = len(
                {key for d, key in confirmed_keys if d == direction}
            )
        state.store.set("defect_validation", details)
        return Score(
            value=values,
            answer=report.summary,
            explanation="Replay checks grading behavior. Semantic correctness and submission plausibility require review; these are discovery counts, not accuracy.",
            metadata={
                "validation_status": "validated",
                "findings": details,
                "coverage": state.metadata["selected_steps"],
                "uncovered_steps": report.uncovered_steps,
            },
        )

    return score
