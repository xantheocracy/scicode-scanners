"""On-demand evidence and reproducible Python experiments."""

import json
import time
from typing import Any

from inspect_ai.tool import ToolDef

from .adapters import CaseSet, redact
from .assets import ROOT, decode_targets, typed
from .sandbox import THREAD_ENV, interpreters
from .schema import Limits


class Investigation:
    def __init__(self, cases: CaseSet, sid: str, targets_path, sandbox, limits: Limits):
        self.cases, self.sid, self.targets_path, self.sandbox, self.limits = (
            cases,
            sid,
            targets_path,
            sandbox,
            limits,
        )
        self.trace: list[dict[str, Any]] = []
        self.seconds = 0.0
        self.experiments = 0
        self.previous_causes = []
        self.functions = {
            name: getattr(self, name)
            for name in (
                "retrieve_step",
                "full_transcript",
                "inspect_target",
                "helper_source",
                "python",
                "rerun",
                "prior_findings",
            )
        }

    @property
    def definitions(self):
        return [
            ToolDef(function, name=name, parallel=False)
            for name, function in self.functions.items()
        ]

    async def prior_findings(self) -> str:
        """Retrieve earlier same-scan cause summaries to help reuse a defect key.

        These are judge-generated hypotheses, not ground truth or historical audit findings.
        Verify inherited causes using original evidence before attributing them here.
        """
        return json.dumps(self.previous_causes, ensure_ascii=False)

    async def retrieve_step(self, step_id: str) -> str:
        """Retrieve a step's exact prompt, response including exposed reasoning, tests and original results.

        Args:
            step_id: Subproblem identifier, e.g. 13.2. Passing and supplied steps are available too.
        """
        s = self.cases.steps[step_id]
        event = s.get("event")
        evidence = {
            "step": step_id,
            "provided": s["provided"],
            "description": s["record"],
            "code": s["code"],
            "score": self.cases.scores.get(step_id),
            "original_grading": self.cases.grading.get(step_id),
            "response_ref": s["response_ref"],
        }
        if event:
            evidence["messages"] = [
                {"reference": f"M:{m.id}", **m.model_dump(mode="json")}
                for m in [*event.input, event.output.choices[0].message]
            ]
        else:
            evidence["reasoning"] = "Unavailable: this is author-provided code."
        return json.dumps(redact(evidence), ensure_ascii=False)

    async def full_transcript(
        self, offset: int = 0, limit: int = 4, query: str = ""
    ) -> str:
        """Read/search all transcript events, including previous steps, reasoning and grader events.

        Args:
            offset: Zero-based offset into events matching the optional query.
            limit: Number of events, at most 10. Continue using next_offset.
            query: Optional literal case-insensitive search string.
        """
        if offset < 0 or not 1 <= limit <= 10:
            raise ValueError("Use offset >= 0 and 1 <= limit <= 10.")
        events = [
            {"reference": f"E:{e.uuid}", **redact(e.model_dump(mode="json"))}
            for e in self.cases.events
        ]
        if query:
            events = [
                e
                for e in events
                if query.casefold() in json.dumps(e, ensure_ascii=False).casefold()
            ]
        end = min(offset + limit, len(events))
        return json.dumps(
            {
                "events": events[offset:end],
                "total": len(events),
                "next_offset": end if end < len(events) else None,
            },
            ensure_ascii=False,
        )

    async def inspect_target(
        self,
        step_id: str,
        test_index: int,
        path: list[str] | None = None,
        offset: int = 0,
        limit: int = 64,
    ) -> str:
        """Inspect expected HDF5 output with typed previews and pagination, without changing it.

        Args:
            step_id: Step whose expected output to inspect.
            test_index: One-based test index.
            path: Nested dictionary keys or list/tuple indices, expressed as strings.
            offset: Start index of an array/list/dictionary preview.
            limit: Maximum preview entries, from 1 to 256.
        """
        if offset < 0 or not 1 <= limit <= 256:
            raise ValueError("Invalid preview range.")
        tests = self.cases.steps[step_id]["record"]["test_cases"]
        if not 1 <= test_index <= len(tests):
            raise ValueError("Invalid one-based test index.")
        targets = decode_targets(
            self.targets_path, self.cases.implementation, step_id, len(tests)
        )
        value = targets[test_index - 1]
        for key in path or []:
            value = (
                value[key]
                if isinstance(value, dict) and key in value
                else value[int(key)]
            )
        return json.dumps(
            {
                "reference": f"T:{step_id}:{test_index}:target",
                "value": typed(value, offset, limit),
            },
            ensure_ascii=False,
        )

    async def helper_source(self, name: str) -> str:
        """Read the exact target-decoding or comparator helper source used by this harness.

        Args:
            name: Either targets or comparator.
        """
        if name not in {"targets", "comparator"}:
            raise ValueError("name must be targets or comparator.")
        if self.cases.implementation == "scicode":
            p = (
                ROOT
                / "vendor"
                / ("process_data.py" if name == "targets" else "test_util.py")
            )
        else:
            p = (
                ROOT
                / "vendor/verified/scicode"
                / ("parse/parse.py" if name == "targets" else "compare/cmp.py")
            )
        return p.read_text()

    async def python(self, code: str, environment: str = "") -> str:
        """Run diagnostic Python with targets and cumulative_code available, in the grading sandbox.

        Args:
            code: Python diagnostic. targets is the current step's decoded expected-output list;
                cumulative_code is a string of the submitted cumulative program without tests.
                Use exec(cumulative_code) to load its functions. Each call is a fresh process.
            environment: default for SciCode; 2024 or 2025 for Verified. Empty selects the first.
        """
        sid = self.sid
        module = (
            "process_data"
            if self.cases.implementation == "scicode"
            else "scicode.parse.parse"
        )
        filename = (
            "test_data.h5"
            if self.cases.implementation == "scicode"
            else "test_data_cleaned.h5"
        )
        order = list(self.cases.steps)
        cumulative = "\n\n".join(
            [
                self.cases.problem["required_dependencies"],
                *(
                    ["from test_util import are_dicts_close, cmp_tuple_or_list"]
                    if self.cases.implementation == "scicode"
                    else []
                ),
                *[self.cases.steps[s]["code"] for s in order[: order.index(sid) + 1]],
            ]
        )
        target_args = (
            f"{sid!r}, {len(self.cases.steps[sid]['record']['test_cases'])}"
            + (
                f", {filename!r}"
                if self.cases.implementation == "scicode_verified"
                else ""
            )
        )
        prefix = (
            f"from {module} import process_hdf5_to_tuple\n"
            f"targets = process_hdf5_to_tuple({target_args})\n"
            f"cumulative_code = {cumulative!r}\n"
        )
        return json.dumps(
            await self.execute(prefix + code, environment), ensure_ascii=False
        )

    async def rerun(
        self,
        step_id: str,
        replacements: dict[str, str] | None = None,
        test_index: int | None = None,
        environment: str = "",
    ) -> str:
        """Rerun original grading or a labelled diagnostic replacement/isolated-test experiment.

        Args:
            step_id: Step to grade, including a preceding or passing step.
            replacements: Optional mapping of step IDs to replacement Python implementations.
            test_index: Optional one-based isolated test. Isolation can change shared state.
            environment: Empty runs all selected grading environments; otherwise selects one.
        """
        diagnostic = bool(replacements) or test_index is not None
        program = self.cases.grading_script(step_id, replacements, test_index)
        if not diagnostic and self.cases.implementation == "scicode":
            program = (self.cases.grading.get(step_id) or {}).get(
                "executed_program"
            ) or program
        envs = [environment] if environment else list(interpreters(self.cases))
        results = {}
        for label in envs:
            result = await self.execute(program, label)
            results[label] = result
            # Match Verified's short-circuit semantics, unless source requested full environments.
            if result.get("passed") and not self.cases.settings.get("full_envs", False):
                break
        return json.dumps(
            {
                "kind": "diagnostic_variant"
                if diagnostic
                else "original_program_rerun",
                "step": step_id,
                "isolated_test": test_index,
                "replacement_steps": list(replacements or {}),
                "per_environment": results,
                "passed": any(r.get("passed") for r in results.values()),
                "note": "A diagnostic timeout uses scanner limits, not the original 1800-second grading limit.",
            },
            ensure_ascii=False,
        )

    async def execute(self, program, environment):
        if self.sandbox is None:
            raise RuntimeError("No Hawk sandbox provisioned.")
        choices = interpreters(self.cases)
        label = environment or next(iter(choices))
        if label not in choices:
            raise ValueError(f"Environment must be one of {list(choices)}.")
        remaining = self.limits.diagnostic_seconds - self.seconds
        if remaining < 1:
            return {"error": "diagnostic_time_exhausted"}
        timeout = min(self.limits.python_timeout, int(remaining))
        self.experiments += 1
        reference = f"X:{self.experiments}"
        start = time.monotonic()
        try:
            result = await self.sandbox.exec(
                [choices[label], "-c", program],
                env=THREAD_ENV,
                timeout=timeout,
                timeout_retry=False,
            )
            output = {
                "reference": reference,
                "environment": label,
                "passed": result.success,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timeout_seconds": timeout,
            }
        except TimeoutError:
            output = {
                "reference": reference,
                "environment": label,
                "passed": False,
                "timed_out": True,
                "timeout_seconds": timeout,
            }
        finally:
            self.seconds += time.monotonic() - start
        self.trace.append(
            {"reference": reference, "program": program, "result": output}
        )
        # Preserve full output in provenance while bounding judge context.
        return {
            k: (
                v[:16000] + "\n[truncated; full output retained in result metadata]"
                if isinstance(v, str) and len(v) > 16000
                else v
            )
            for k, v in output.items()
        }
