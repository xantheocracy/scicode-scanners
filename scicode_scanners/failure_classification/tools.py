"""On-demand evidence retrieval and, when a sandbox is provisioned, Python experiments."""

import json
import time
from typing import Any, Literal

from inspect_ai.tool import ToolDef
from inspect_ai.util import OutputLimitExceededError

from .adapters import CaseSet, redact
from .assets import decode_targets, target_metadata, typed
from .sandbox import THREAD_ENV, interpreters, sandbox_errors
from .schema import Limits

PAGE_CHARS = 12000
OUTPUT_CHARS = 5000


def evidence_page(value, char_offset=0):
    """Bound serialized evidence; subsequent pages retain access to the full payload."""
    if char_offset < 0:
        raise ValueError("char_offset must be nonnegative.")
    text = json.dumps(redact(value), ensure_ascii=False)
    if len(text) <= PAGE_CHARS and char_offset == 0:
        return text
    end = min(char_offset + PAGE_CHARS, len(text))
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
    def __init__(
        self, cases: CaseSet, sid: str, targets_path, sandbox=None, limits=None
    ):
        self.cases, self.sid, self.targets_path = cases, sid, targets_path
        self.sandbox, self.limits = sandbox, limits or Limits()
        self.trace: list[dict[str, Any]] = []
        self.seconds = 0.0
        self.experiments = 0
        self.previous_causes = []
        self.packet = {}
        self.functions = {
            name: getattr(self, name)
            for name in (
                "current_evidence",
                "retrieve_step",
                "full_transcript",
                "read_message",
                "inspect_target",
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

    async def current_evidence(self, char_offset: int = 0) -> str:
        """Read the initial evidence packet with character pagination if it was clipped.

        Args:
            char_offset: Character position within the selected JSON payload; follow next_char_offset.
        """
        return evidence_page({"case": self.packet}, char_offset)

    def message_index(self):
        messages = {}
        for event in self.cases.events:
            if getattr(event, "event", None) == "model":
                for message in [
                    *event.input,
                    *[c.message for c in event.output.choices],
                ]:
                    if message.id:
                        messages[message.id] = message
        return messages

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
            "grading", "reasoning", "tests", "prompt", "response"
        ] = "grading",
    ) -> str:
        """Read a selected step section; default grading excludes solutions and model inputs.

        Args:
            step_id: Subproblem identifier. Current and previous solutions are in current_evidence.
            char_offset: Character offset for pagination; follow next_char_offset.
            section: grading for recorded observations, reasoning for exposed reasoning only,
                tests for test source, prompt for exact model inputs, or response for raw output.
                Prompt and response explicitly reload code; request only when necessary.
        """
        step = self.cases.steps[step_id]
        event = step.get("event")
        evidence = {"step": step_id, "response_reference": step["response_ref"]}
        if section == "grading":
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

    async def read_message(
        self,
        message_id: str,
        section: Literal["text", "reasoning"] = "text",
        char_offset: int = 0,
    ) -> str:
        """Read one selected original message, rather than an entire model-event input.

        Args:
            message_id: Message ID from full_transcript, optionally prefixed with M:.
            section: text for original text (may reload code), reasoning for exposed reasoning only.
            char_offset: Character offset for pagination; follow next_char_offset.
        """
        message_id = message_id.removeprefix("M:")
        message = self.message_index()[message_id]
        if section not in {"text", "reasoning"}:
            raise ValueError("Unknown message section.")
        return evidence_page(
            {
                "reference": f"M:{message_id}",
                "role": message.role,
                section: message.text if section == "text" else self.reasoning(message),
            },
            char_offset,
        )

    async def full_transcript(
        self, offset: int = 0, limit: int = 4, query: str = "", char_offset: int = 0
    ) -> str:
        """List/search transcript events without reloading model prompts or submitted code.

        Args:
            char_offset: Character offset for pagination; follow next_char_offset.
            offset: Zero-based offset into events matching the optional query.
            limit: Number of events, at most 10. Continue using next_offset.
            query: Literal case-insensitive search in original events. Matches return summaries;
                use read_message or retrieve_step to read selected evidence.
        """
        if offset < 0 or not 1 <= limit <= 10:
            raise ValueError("Use offset >= 0 and 1 <= limit <= 10.")
        events = []
        for event in self.cases.events:
            raw = redact(event.model_dump(mode="json"))
            if (
                query
                and query.casefold()
                not in json.dumps(raw, ensure_ascii=False).casefold()
            ):
                continue
            reference = f"E:{event.uuid}"
            if getattr(event, "event", None) == "model":
                messages = [*event.input, *[c.message for c in event.output.choices]]
                summary = {
                    "reference": reference,
                    "event": "model",
                    "steps": [
                        sid
                        for sid, step in self.cases.steps.items()
                        if step.get("event") is event
                    ],
                    "messages": [
                        {
                            "reference": f"M:{m.id}",
                            "role": m.role,
                            "reasoning_available": bool(self.reasoning(m)),
                        }
                        for m in messages
                    ],
                    "note": "Message contents omitted; use read_message for selected content.",
                }
            else:
                # Sandbox programs and score answers repeat the cumulative submitted code.
                summary = {
                    "reference": reference,
                    **{
                        k: v
                        for k, v in raw.items()
                        if k not in {"cmd", "answer", "input", "output"}
                    },
                }
                if "output" in raw:
                    summary["output"] = raw["output"]
                if isinstance(summary.get("score"), dict):
                    summary["score"] = {
                        k: v for k, v in summary["score"].items() if k != "answer"
                    }
            events.append(summary)
        end = min(offset + limit, len(events))
        return evidence_page(
            {
                "events": events[offset:end],
                "total": len(events),
                "next_offset": end if end < len(events) else None,
            },
            char_offset,
        )

    async def inspect_target(
        self,
        step_id: str,
        test_index: int,
        path: list[str] | None = None,
        offset: int = 0,
        limit: int = 64,
        char_offset: int = 0,
        metadata_only: bool = False,
        row_start: int | None = None,
        row_stop: int | None = None,
        column_start: int | None = None,
        column_stop: int | None = None,
    ) -> str:
        """Inspect expected HDF5 output with typed previews and pagination, without changing it.

        Args:
            char_offset: Character position within the selected JSON payload; follow next_char_offset.
            step_id: Step whose expected output to inspect.
            test_index: One-based test index.
            metadata_only: Return only type, shape/dtype or container size, without values.
            row_start: Optional first matrix row (inclusive).
            row_stop: Optional final matrix row (exclusive).
            column_start: Optional first matrix column (inclusive).
            column_stop: Optional final matrix column (exclusive).
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
        if metadata_only:
            if any(
                bound is not None
                for bound in (row_start, row_stop, column_start, column_stop)
            ):
                raise ValueError("Use metadata_only without matrix slice bounds.")
            return evidence_page(
                {
                    "reference": f"T:{step_id}:{test_index}:target",
                    "metadata": target_metadata(value),
                },
                char_offset,
            )
        selection = None
        bounds = (row_start, row_stop, column_start, column_stop)
        if any(bound is not None for bound in bounds):
            if getattr(value, "ndim", None) != 2:
                raise ValueError("Matrix slicing requires a two-dimensional target.")
            if any(bound is not None and bound < 0 for bound in bounds):
                raise ValueError("Matrix slice bounds must be nonnegative.")
            rows = slice(row_start, row_stop)
            columns = slice(column_start, column_stop)
            selection = {
                "rows": [row_start, row_stop],
                "columns": [column_start, column_stop],
            }
            value = value[rows, columns]
            # A sparse slice is converted only after bounding the selected matrix.
            if hasattr(value, "toarray") and value.shape[0] * value.shape[1] <= 256:
                value = value.toarray()
        return evidence_page(
            {
                "reference": f"T:{step_id}:{test_index}:target",
                "selection": selection,
                "value": typed(value, offset, limit),
            },
            char_offset,
        )

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
