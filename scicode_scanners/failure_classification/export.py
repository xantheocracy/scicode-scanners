"""Export a downloaded Hawk Scout result directory to affected-step/cause CSV."""

import argparse
import csv
import json

from inspect_scout import scan_results_df


def cause_rows(metadata):
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict) or "assessment" not in metadata:
        return
    source = metadata.get("source", {})
    assessment = metadata.get("assessment", {})
    for cause in assessment.get("causes") or [None]:
        yield {
            **{
                key: source.get(key)
                for key in (
                    "implementation",
                    "transcript_id",
                    "source_model",
                    "epoch",
                    "main_problem",
                    "affected_step",
                )
            },
            "status": assessment.get("status"),
            "cause_id": cause.get("cause_id") if cause else None,
            "category": cause.get("category") if cause else None,
            "origin_type": cause.get("origin_type") if cause else None,
            "origin_steps": json.dumps(cause.get("origin_steps", []))
            if cause
            else "[]",
            "mechanism": cause.get("mechanism") if cause else None,
            "explanation": assessment.get("explanation"),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scan_directory")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    results = scan_results_df(args.scan_directory)
    frame = results.scanners["failure_classification"]
    rows = []
    for metadata in frame["metadata"]:
        if metadata is not None:
            rows.extend(cause_rows(metadata))
    if not rows:
        raise SystemExit("No failure assessments found.")
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    affected = {
        (r["implementation"], r["transcript_id"], r["affected_step"]) for r in rows
    }
    causes = {
        (r["implementation"], r["transcript_id"], r["cause_id"])
        for r in rows
        if r["cause_id"]
    }
    print(f"{len(affected)} affected steps; {len(causes)} distinct supported causes.")


if __name__ == "__main__":
    main()
