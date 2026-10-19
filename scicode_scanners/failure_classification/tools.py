"""On-demand evidence retrieval and, when a sandbox is provisioned, Python experiments."""

import json
import time
from typing import Any, Literal

from inspect_ai.tool import ToolDef
from inspect_ai.util import OutputLimitExceededError

from .adapters import CaseSet, code_outline, redact
from .sandbox import THREAD_ENV, interpreters, sandbox_errors
from .schema import Limits

PAGE_CHARS = 12000
OUTPUT_CHARS = 5000


def evidence_page(value, char_offset=0, page_chars=PAGE_CHARS):
    """Bound serialized evidence; subsequent pages retain access to the full payload."""
    if char_offset < 0:
        raise ValueError("char_offset must be nonnegative.")
    text = json.dumps(redact(value), ensure_ascii=False)
    if len(text) <= page_chars and char_offset == 0:
        return text
    end = min(char_offset + page_chars, len(text))
    return json.dumps(
        {
            "format": "json_text_excerpt",
            "total_chars": len(text),
            "char_offset": char_offset,
            "excerpt": text[char_offset:end],
            "next_char_offset": end if end < len(text) else None,
            "note": "Incomplete evidence. Repeat the same tool arguments with next_char_offset as char_offset to continue; omitted text is not evidence of absence.",
        },
        ensure_ascii=False,
    )


class Investigation:
    def __init__(self, cases: CaseSet, sid: str, sandbox=None, limits=None):
        self.cases, self.sid = cases, sid
        self.sandbox, self.limits = sandbox, limits or Limits()
        self.trace: list[dict[str, Any]] = []
        self.seconds = 0.0
        self.experiments = 0
        self.previous_causes = []
        self.functions = {
            name: getattr(self, name)
            for name in (
                "retrieve_step",
                "read_code",
                *(("python", "rerun") if sandbox is not None else ()),
            )
        }

    @property
    def mode(self):
        return "inspection_only" if self.sandbox is None else "sandbox"

    @property
    def definitions(self):
        return [
            ToolDef(function, name=name, parallel=False)
            for name, function in self.functions.items()
        ]

    @staticmethod
    def reasoning(message):
        return (
            [
                block.model_dump(mode="json")
                for block in message.content
                if getattr(block, "type", None) == "reasoning"
            ]
            if isinstance(message.content, list)
            else []
        )

    async def retrieve_step(
        self,
        step_id: str,
        char_offset: int = 0,
        section: Literal[
            "grading", "code", "reasoning", "tests", "prompt", "response"
        ] = "grading",
    ) -> str:
        """Read a selected step section; default grading excludes solutions and model inputs.

        Args:
            step_id: Subproblem identifier, as listed in earlier_steps.
            char_offset: Character offset for pagination; follow next_char_offset.
            section: grading for recorded observations, code for the step's extracted submitted
                code, reasoning for exposed reasoning only, tests for test source, prompt for
                exact model inputs, or response for raw output. Prompt and response explicitly
                reload code; request only when necessary.
        """
        step = self.cases.steps[step_id]
        event = step.get("event")
        evidence = {"step": step_id, "response_reference": step["response_ref"]}
        if section == "code":
            evidence["provided"] = step["provided"]
            evidence["code_status"] = code_outline(step["code"])[0]
            evidence["submitted_code"] = step["code"]
        elif section == "grading":
            evidence["score"] = self.cases.scores.get(step_id)
            if event:
                evidence["generation"] = {
                    "event_reference": f"E:{event.uuid}",
                    "stop_reason": step.get("stop_reason"),
                }
                usage = getattr(event.output, "usage", None)
                if usage is not None:
                    evidence["generation"]["usage"] = usage.model_dump(mode="json")
            evidence["original_grading"] = {
                k: v
                for k, v in (self.cases.grading.get(step_id) or {}).items()
                if k != "executed_program"
            }
        elif section == "tests":
            evidence["tests"] = [
                {"reference": f"T:{step_id}:{i + 1}", "source": test}
                for i, test in enumerate(step["record"]["test_cases"])
            ]
        elif section == "reasoning":
            evidence["reasoning"] = (
                self.reasoning(event.output.choices[0].message) if event else []
            )
            evidence["note"] = (
                "Only exposed reasoning is returned; an empty list means unavailable."
            )
        elif section in {"prompt", "response"}:
            messages = (
                event.input
                if section == "prompt" and event
                else ([event.output.choices[0].message] if event else [])
            )
            evidence["messages"] = [
                {"reference": f"M:{m.id}", **m.model_dump(mode="json")}
                for m in messages
            ]
        else:
            raise ValueError("Unknown step section.")
        return evidence_page(evidence, char_offset)

    async def read_code(self, name: str, char_offset: int = 0) -> str:
        """Read the submitted definition of one function or class, from the current or an earlier step.

        Args:
            name: Top-level function or class name, as listed in earlier_steps defines.
            char_offset: Character offset for pagination; follow next_char_offset.
        """
        order = list(self.cases.steps)
        after = order[order.index(self.sid) + 1] if self.sid != order[-1] else None
        found = self.cases.definitions(before=after).get(name)
        if found is None:
            raise ValueError(
                f"No parsed definition of {name!r} up to step {self.sid}; use retrieve_step(section='code') for steps whose code does not parse."
            )
        sid, source = found
        return evidence_page({"step": sid, "name": name, "source": source}, char_offset)

    async def python(self, code: str, environment: str = "") -> str:
        """Run diagnostic Python with targets and cumulative_code available, in the grading sandbox.

        Args:
            code: Python diagnostic. targets is the current step's decoded expected-output list,
                one entry per test; load_targets(step_id) returns another step's. Summarize large
                targets (shape, norms, differences) rather than printing them. cumulative_code is
                a string of the submitted cumulative program without tests. Use
                exec(cumulative_code) to load its functions. Each call is a fresh process.
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
        test_counts = {
            s: len(step["record"]["test_cases"])
            for s, step in self.cases.steps.items()
            if not step["provided"]
        }
        extra = (
            f", {filename!r}" if self.cases.implementation == "scicode_verified" else ""
        )
        prefix = (
            f"from {module} import process_hdf5_to_tuple\n"
            f"targets = process_hdf5_to_tuple({target_args})\n"
            f"def load_targets(step_id):\n"
            f"    return process_hdf5_to_tuple(step_id, {test_counts!r}[step_id]{extra})\n"
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
        infrastructure = sandbox_errors()
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
        except OutputLimitExceededError as ex:
            output = {
                "reference": reference,
                "environment": label,
                "passed": False,
                "output_limit_exceeded": ex.limit_str,
                "stdout": ex.truncated_output or "",
                "timeout_seconds": timeout,
            }
        except infrastructure as ex:
            # Experiments rewrite their files into a fresh directory, so later ones
            # work on a replacement pod; only this experiment's result is lost.
            output = {
                "reference": reference,
                "environment": label,
                "passed": None,
                "sandbox_failure": f"{type(ex).__name__}: {ex}"[:1000],
                "note": "The sandbox pod was replaced or restarted during this experiment, "
                "possibly because it used too much memory (4 GiB limit). This is not "
                "evidence about the submission. Later experiments run on a fresh pod.",
                "timeout_seconds": timeout,
            }
        finally:
            self.seconds += time.monotonic() - start
        self.trace.append(
            {"reference": reference, "program": program, "result": output}
        )
        # Full output stays in provenance; keep stdout's head and stderr's traceback tail.
        bounded = dict(output)
        for key, keep in (("stdout", "head"), ("stderr", "tail")):
            text = bounded.get(key)
            if isinstance(text, str) and len(text) > OUTPUT_CHARS:
                omitted = f"[{len(text) - OUTPUT_CHARS} chars omitted; full output retained in result metadata]"
                bounded[key] = (
                    f"{text[:OUTPUT_CHARS]}\n{omitted}"
                    if keep == "head"
                    else f"{omitted}\n{text[-OUTPUT_CHARS:]}"
                )
        return bounded
