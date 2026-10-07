"""Export reports plus complete replayable experiment provenance from Inspect logs."""

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def export(log_path: str, output_dir: str) -> None:
    log = read_eval_log(log_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    markdown = [
        "# SciCode defect investigations",
        "",
        "Grading replay does not establish semantic correctness or submission plausibility.",
        "",
    ]
    for sample in log.samples or []:
        store = sample.store or {}
        report = store.get("defect_report")
        row = {
            "sample_id": sample.id,
            "epoch": sample.epoch,
            "report": report,
            "evidence": store.get("defect_evidence", []),
            "validation": store.get("defect_validation", []),
            "termination": store.get("defect_termination", "interrupted"),
            "generated_tokens": store.get("defect_generated_tokens", 0),
            "metadata": sample.metadata,
            "error": sample.error.model_dump(mode="json") if sample.error else None,
            "model_usage": sample.model_usage,
            "scores": sample.scores,
        }
        rows.append(row)
        markdown.extend(
            [
                f"## {sample.id} (epoch {sample.epoch})",
                "",
                report["summary"]
                if report
                else "No final report; consult retained evidence.",
                "",
            ]
        )
        for finding in (report or {}).get("findings", []):
            markdown.extend(
                [
                    f"### {finding['step_id']}: {finding['direction']} — {finding['status']}",
                    "",
                    finding["claim"],
                    "",
                    finding["semantic_argument"],
                    "",
                    f"Semantic confidence: {finding['semantic_confidence']['rating']}. {finding['semantic_confidence']['rationale']}",
                    "",
                    f"Grader confidence: {finding['grader_confidence']['rating']}. {finding['grader_confidence']['rationale']}",
                    "",
                    f"Candidate evidence: {finding['candidate_evidence']}. Semantic and plausibility review: pending.",
                    "",
                ]
            )
            refs = {item["id"]: item for item in row["evidence"]}
            candidate = refs.get(finding["candidate_evidence"])
            if candidate:
                for step_id, code in candidate["codes"].items():
                    markdown.extend(
                        [f"Candidate step {step_id}:", "", "```python", code, "```", ""]
                    )
                markdown.extend(
                    [
                        "Canonical grading observations:",
                        "",
                        "```json",
                        json.dumps(candidate["per_environment"], indent=2),
                        "```",
                        "",
                    ]
                )
            for reference in finding["semantic_evidence"]:
                check = refs.get(reference)
                if check:
                    markdown.extend(
                        [
                            f"Independent check {reference}:",
                            "",
                            "```python",
                            check.get("check", ""),
                            "```",
                            "",
                            "```json",
                            json.dumps(check["per_environment"], indent=2),
                            "```",
                            "",
                        ]
                    )
    (destination / "reports.jsonl").write_text(
        "".join(
            json.dumps(
                row, default=lambda x: x.model_dump(mode="json"), ensure_ascii=False
            )
            + "\n"
            for row in rows
        )
    )
    (destination / "reports.md").write_text("\n".join(markdown))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    export(args.log, args.output_dir)


if __name__ == "__main__":
    main()
