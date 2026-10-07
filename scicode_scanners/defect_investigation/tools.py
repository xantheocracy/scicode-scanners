"""Recorded sandbox experiments and pristine candidate grading."""

import hashlib
import json
import shlex
from pathlib import Path
from uuid import uuid4

from inspect_ai.tool import ToolDef
from inspect_ai.util import sandbox

from .dataset import check_hash
from .harness import (
    TARGET_NAMES,
    THREAD_ENV,
    admissibility_issues,
    cumulative_codes,
    grading_script,
    helper_files,
    interpreters,
)
from .schema import Limits, Report


def bounded(record: dict) -> str:
    preview = dict(record)
    if "per_environment" in record:
        preview["per_environment"] = {
            label: {
                k: (
                    v[:12000] + "\n[truncated in observation; full value in log]"
                    if isinstance(v, str) and len(v) > 12000
                    else v
                )
                for k, v in result.items()
            }
            for label, result in record["per_environment"].items()
        }
    for key in ("program", "codes", "source", "check"):
        preview.pop(key, None)
    if "stdout" in preview:
        preview["stdout"] = preview["stdout"][:12000]
        preview["stderr"] = preview["stderr"][:12000]
    return json.dumps(preview, ensure_ascii=False)


def validate_report(report: Report, metadata: dict, evidence: list[dict]) -> None:
    if (
        report.implementation != metadata["implementation"]
        or report.problem_id != metadata["problem"]["problem_id"]
    ):
        raise ValueError("Report identity does not match sample.")
    expected = {
        (sid, direction)
        for sid in metadata["selected_steps"]
        for direction in ("false_rejection", "false_acceptance")
    }
    actual = [(f.step_id, f.direction) for f in report.findings]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError(
            "Report must cover each selected step and direction exactly once."
        )
    order = [s["step_number"] for s in metadata["problem"]["sub_steps"]]
    if set(report.uncovered_steps) - set(metadata["selected_steps"]):
        raise ValueError("Unknown uncovered step.")
    refs = {r["id"]: r for r in evidence}
    spec_refs = {f"P:{s['step_number']}" for s in metadata["problem"]["sub_steps"]}
    spec_refs.update(
        f"T:{s['step_number']}:{i + 1}"
        for s in metadata["problem"]["sub_steps"]
        for i in range(len(s["test_cases"]))
    )
    for finding in report.findings:
        if (
            finding.step_id in report.uncovered_steps
            and finding.status != "inconclusive"
        ):
            raise ValueError("Uncovered steps must have inconclusive outcomes.")
        if any(
            s not in order or order.index(s) > order.index(finding.step_id)
            for s in finding.origin_steps
        ):
            raise ValueError("Origin must be the current or a preceding step.")
        if set(finding.prompt_test_references) - spec_refs:
            raise ValueError("Unknown prompt/test evidence reference.")
        candidate = refs.get(finding.candidate_evidence)
        if finding.candidate_evidence and (
            not candidate
            or candidate["kind"] != "canonical_grade"
            or candidate["step_id"] != finding.step_id
        ):
            raise ValueError(
                "Candidate evidence must reference this step's canonical grading."
            )
        for ref in finding.semantic_evidence:
            item = refs.get(ref)
            if (
                not item
                or item["kind"] != "semantic_check"
                or item["step_id"] != finding.step_id
            ):
                raise ValueError("Unknown semantic evidence for this step.")
            if candidate and item["codes"] != candidate["codes"]:
                raise ValueError(
                    "Semantic checks must use exactly the graded candidate."
                )
        if finding.status == "demonstrated":
            expected_pass = finding.direction == "false_acceptance"
            if candidate["passed"] is not expected_pass or candidate.get(
                "instrument_error"
            ):
                raise ValueError(
                    "Canonical grading does not establish claimed behavior."
                )
            if any(r.get("timed_out") for r in candidate["per_environment"].values()):
                raise ValueError(
                    "Timeout witnesses need manual review; use suspected/inconclusive."
                )
            if candidate["admissibility_issues"]:
                raise ValueError("Candidate violates the admissibility screen.")
            if finding.direction == "false_acceptance" and not any(
                refs[ref]["passed"] and not refs[ref].get("instrument_error")
                for ref in finding.semantic_evidence
            ):
                raise ValueError(
                    "Counterexample must execute successfully and print decisive evidence."
                )


class Investigation:
    def __init__(self, state, limits: Limits, grading_environments: str):
        self.state, self.limits = state, limits
        self.metadata = state.metadata
        self.implementation = state.metadata["implementation"]
        self.problem = state.metadata["problem"]
        self.environments = grading_environments
        self.evidence = state.store.get("defect_evidence", [])

    def record(self, item: dict) -> dict:
        item = {"id": f"X:{len(self.evidence) + 1}", **item}
        self.evidence.append(item)
        self.state.store.set("defect_evidence", self.evidence)
        return item

    @property
    def definitions(self):
        return [
            ToolDef(self.workspace, name="workspace", parallel=False),
            ToolDef(self.grade_candidate, name="grade_candidate", parallel=False),
            ToolDef(self.semantic_check, name="semantic_check", parallel=False),
        ]

    async def workspace(self, command: str) -> str:
        """Execute a shell command in the investigation workspace, never the grader.

        Args:
            command: Shell command for editing code or running Python diagnostics.
        """
        try:
            result = await sandbox().exec(
                ["sh", "-c", command],
                env=THREAD_ENV,
                timeout=self.limits.diagnostic_timeout,
                timeout_retry=False,
            )
            item = {
                "kind": "workspace",
                "command": command,
                "passed": result.success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
        except TimeoutError:
            item = {
                "kind": "workspace",
                "command": command,
                "passed": False,
                "timed_out": True,
                "stdout": "",
                "stderr": "Diagnostic timeout",
            }
        return bounded(self.record(item))

    async def grade_candidate(self, step_id: str, codes: dict[str, str]) -> str:
        """Grade a complete cumulative candidate against unchanged benchmark tests.

        Args:
            step_id: Scored subproblem ID; must be among selected investigation steps.
            codes: Mapping of current and all preceding generated steps to complete Python source.
                Supplied benchmark steps are inserted automatically. No cheating or imports.
        """
        return bounded(await self.canonical_grade(step_id, codes))

    async def canonical_grade(self, step_id: str, codes: dict[str, str]) -> dict:
        if step_id not in self.metadata["selected_steps"]:
            raise ValueError("Grade only selected investigation steps.")
        program = grading_script(self.problem, codes, step_id, self.implementation)
        issues = admissibility_issues(codes, self.implementation)
        if issues:
            return self.record(
                {
                    "kind": "inadmissible_candidate",
                    "step_id": step_id,
                    "codes": codes,
                    "admissibility_issues": issues,
                    "passed": False,
                }
            )
        return await self.run_pristine(
            program, step_id, codes, "canonical_grade", self.limits.grading_timeout
        )

    async def semantic_check(
        self, step_id: str, codes: dict[str, str], check: str
    ) -> str:
        """Execute an independent semantic diagnostic with the exact cumulative candidate.

        Args:
            step_id: Selected step being investigated.
            codes: Same complete step-to-source mapping used for canonical grading.
            check: Python that independently establishes correctness or a valid counterexample.
                For false acceptance, assert the wrong result differs from independently justified
                truth, and print the input, actual output, expected output and reasoning. A successful
                diagnostic establishes only execution; semantic validity still requires review.
        """
        if step_id not in self.metadata["selected_steps"]:
            raise ValueError("Select an investigated step.")
        program = "\n\n".join(
            [
                self.problem["required_dependencies"],
                *cumulative_codes(self.problem, codes, step_id, self.implementation),
                check,
            ]
        )
        item = await self.run_pristine(
            program, step_id, codes, "semantic_check", self.limits.diagnostic_timeout
        )
        item["check"] = check
        self.state.store.set("defect_evidence", self.evidence)
        return bounded(item)

    async def run_pristine(
        self, program: str, step_id: str, codes: dict[str, str], kind: str, timeout: int
    ) -> dict:
        env = sandbox("grader")
        shard = Path(self.metadata["targets_path"])
        check_hash(shard, self.metadata["shard_sha256"])
        results = {}
        step = next(s for s in self.problem["sub_steps"] if s["step_number"] == step_id)
        # Every interpreter gets a new directory with host-authoritative resource bytes.
        for label, python in interpreters(
            self.implementation, self.environments
        ).items():
            directory = f"/tmp/defect-{uuid4().hex}"
            made = await env.exec(["mkdir", "-p", directory], timeout=30)
            if not made.success:
                raise RuntimeError("Could not initialize pristine grader directory.")
            try:
                files = {
                    TARGET_NAMES[self.implementation]: shard.read_bytes(),
                    **helper_files(self.implementation),
                    "candidate.py": program.encode(),
                }
                for name, content in files.items():
                    parent = str(Path(directory, name).parent)
                    made = await env.exec(["mkdir", "-p", parent], timeout=30)
                    if not made.success:
                        raise RuntimeError("Could not initialize grader helpers.")
                    await env.write_file(f"{directory}/{name}", content)
                module = (
                    "process_data"
                    if self.implementation == "scicode"
                    else "scicode.parse.parse"
                )
                decode = (
                    f"process_hdf5_to_tuple({step_id!r}, {len(step['test_cases'])})"
                    if self.implementation == "scicode"
                    else (
                        f"process_hdf5_to_tuple({step_id!r}, {len(step['test_cases'])}, 'test_data_cleaned.h5')"
                    )
                )
                preflight = f"from {module} import process_hdf5_to_tuple; {decode}"
                env_vars = {
                    **THREAD_ENV,
                    "PYTHONPATH": directory,
                    "PYTHONNOUSERSITE": "1",
                }
                prefix = f"cd {shlex.quote(directory)} && exec {shlex.quote(python)} "
                check = await env.exec(
                    ["sh", "-c", prefix + "-c " + shlex.quote(preflight)],
                    env=env_vars,
                    timeout=60,
                    timeout_retry=False,
                )
                if not check.success:
                    results[label] = {
                        "passed": False,
                        "instrument_error": True,
                        "stdout": check.stdout,
                        "stderr": check.stderr,
                    }
                    continue
                try:
                    result = await env.exec(
                        ["sh", "-c", prefix + "-c " + shlex.quote(program)],
                        env=env_vars,
                        timeout=timeout,
                        timeout_retry=False,
                    )
                    results[label] = {
                        "passed": result.success,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                except TimeoutError:
                    results[label] = {
                        "passed": False,
                        "timed_out": True,
                        "stdout": "",
                        "stderr": "Execution timed out",
                    }
            finally:
                await env.exec(["rm", "-rf", directory], timeout=30)
        return self.record(
            {
                "kind": kind,
                "step_id": step_id,
                "codes": dict(codes),
                "program": program,
                "program_sha256": hashlib.sha256(program.encode()).hexdigest(),
                "timeout": timeout,
                "per_environment": results,
                "passed": any(r["passed"] for r in results.values()),
                "instrument_error": any(
                    r.get("instrument_error") for r in results.values()
                ),
                "admissibility_issues": admissibility_issues(
                    codes, self.implementation
                ),
            }
        )
