"""Build review context for any bundled SciCode step, independent of run failures."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path

DATA = Path(__file__).parent / "data"
ORIGINALS = Path(__file__).parents[1] / "defect_investigation/data/scicode.jsonl"


@lru_cache(maxsize=1)
def _sources():
    originals = {
        str(problem["problem_id"]): problem
        for problem in map(json.loads, ORIGINALS.read_text().splitlines())
    }
    defects = json.loads((DATA / "defects.json").read_text())["defects"]
    by_problem = {}
    for defect in defects:
        by_problem.setdefault(str(defect["problem_id"]), []).append(defect)
    historical = json.loads((DATA / "historical_evidence.json").read_text())
    return originals, by_problem, historical


def audit_context(step: str) -> dict:
    """Include original steps up to this one, audit records, and optional history.

    The current verified prompt, response, and grading errors are supplied by
    the caller. Historical review is optional; missing history does not imply
    that the task was unchanged or that a submission used benchmark memory.
    """
    problem, index = step.split(".")
    originals, defects, historical = _sources()
    if problem not in originals:
        raise ValueError(f"Original SciCode problem {problem} is not in the bundled dataset.")
    steps = originals[problem]["sub_steps"]
    if not any(item["step_number"] == step for item in steps):
        raise ValueError(f"Original SciCode step {step} is not in the bundled dataset.")
    context = {
        "current_subproblem": step,
        "audit_records": defects.get(problem, []),
        "original_step_text_and_tests": [
            {key: value for key, value in item.items() if key != "ground_truth_code"}
            for item in steps
            if int(item["step_number"].split(".")[1]) <= int(index)
        ],
        "historical_audit_evidence": historical.get(problem, {}).get("historical_audit_evidence", {}),
        "corrected_target_patches": historical.get(problem, {}).get("corrected_target_patches", {}),
        "note": (
            "Analyst context, not model-visible. Audit records cover the whole problem; "
            "use only records relevant to the current or preceding steps. Original text/tests "
            "include only the current and preceding steps. Historical review and target patches "
            "are optional and may be unavailable. The audit taxonomy is incomplete: missing "
            "records do not establish that no correction occurred. This context alone is not "
            "evidence of benchmark recall or its causal role in a failure."
        ),
    }
    return deepcopy(context)
