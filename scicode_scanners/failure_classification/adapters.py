"""Recover default-harness submissions without exposing reference answers."""

import asyncio
import re
import shlex
from dataclasses import dataclass, field
from typing import Any, Literal

from inspect_ai.log import read_eval_log, read_eval_log_sample
from inspect_scout import Transcript

from ..logs import enable_zstd_zip

Implementation = Literal["scicode", "scicode_verified"]
PROVIDED = {"13.6", "62.1", "76.3"}
REVISIONS = {
    "scicode": "51f92017ccbdc63bd768ed6aba36942d43243b25",
    "scicode_verified": "3fd9ef2713244d58f6cfeab5a5d92c10e2b907de",
}
FORBIDDEN = {
    "ground_truth_code",
    "general_solution",
    "reference_solution",
    "audit_context",
    "audit_findings",
}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items() if k not in FORBIDDEN}
    if isinstance(value, list):
        return [redact(v) for v in value]
    return value


def extract_code(text: str, implementation: Implementation) -> str:
    if implementation == "scicode":
        match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        return (match.group(1) if match else text).strip()
    if "```" in text:
        text = (
            text.split("```python", 1)[1].split("```", 1)[0]
            if "```python" in text
            else text.split("```", 2)[1]
        )
    return re.sub(
        r"^\s*(import .*|from .*\s+import\s+.*)", "", text, flags=re.MULTILINE
    )


@dataclass
class CaseSet:
    implementation: Implementation
    transcript: Transcript
    problem: dict[str, Any]
    scores: dict[str, float]
    settings: dict[str, Any]
    steps: dict[str, dict[str, Any]]
    events: list[Any]
    grading: dict[str, Any]
    source_revision: str | None = None
    epoch: int | None = None
    files: dict[str, str] = field(default_factory=dict)

    def validate_cause(self, cause, affected: str, experiment_refs=()):
        order = list(self.steps)
        origins = cause.origin_steps
        if any(
            s not in self.steps or order.index(s) > order.index(affected)
            for s in origins
        ):
            raise ValueError(
                "Origin must be an existing current or preceding subproblem."
            )
        if cause.origin_type == "current" and origins != [affected]:
            raise ValueError("Current origin must name the affected step.")
        if cause.origin_type == "earlier" and (not origins or affected in origins):
            raise ValueError("Earlier origin must name preceding steps.")
        if cause.origin_type == "author_provided" and (
            not origins or any(not self.steps[s]["provided"] for s in origins)
        ):
            raise ValueError("Author-provided origins must refer to supplied code.")
        if cause.origin_type in {"external", "unknown"} and origins:
            raise ValueError("External/unknown origins use an empty step list.")
        if any(
            s not in self.steps or order.index(s) > order.index(affected)
            for s in cause.dependency_path
        ):
            raise ValueError("Invalid dependency path.")
        known = set(experiment_refs)
        for event in self.events:
            known.add(f"E:{event.uuid}")
            if event.event == "model":
                for message in [
                    *event.input,
                    *[c.message for c in event.output.choices],
                ]:
                    if message.id:
                        known.add(f"M:{message.id}")
        for sid, step in self.steps.items():
            for i in range(len(step["record"]["test_cases"])):
                known.update({f"T:{sid}:{i + 1}", f"T:{sid}:{i + 1}:target"})
        cited = {
            reference.rstrip(".")
            for evidence in cause.evidence
            for reference in re.findall(r"\b[METX]:[A-Za-z0-9_.:-]+", evidence)
        }
        if not cited or cited - known:
            raise ValueError(
                f"Cite existing evidence references; unknown references: {sorted(cited - known)}"
            )

    def packet(self, step_id: str) -> dict[str, Any]:
        order = list(self.steps)

        def solution(sid):
            step = self.steps[sid]
            return {
                "step": sid,
                "provided": step["provided"],
                "description": step["record"]["step_description_prompt"],
                "interface": {
                    k: step["record"].get(k) for k in ("function_header", "return_line")
                },
                "submitted_code": step["code"],
                "response_reference": step["response_ref"],
            }

        return {
            "implementation": self.implementation,
            "affected_step": step_id,
            "current_solution": solution(step_id),
            "previous_solutions": [
                solution(sid) for sid in order[: order.index(step_id)]
            ],
            "dependencies": self.problem["required_dependencies"],
            "tests": [
                {"reference": f"T:{step_id}:{i + 1}", "source": test}
                for i, test in enumerate(self.steps[step_id]["record"]["test_cases"])
            ],
        }

    def grading_script(
        self,
        step_id: str,
        replacements: dict[str, str] | None = None,
        test_index: int | None = None,
    ) -> str:
        replacements = replacements or {}
        order = list(self.steps)
        if any(
            s not in order or order.index(s) > order.index(step_id)
            for s in replacements
        ):
            raise ValueError("Replacements must name current or preceding steps.")
        record = self.steps[step_id]["record"]
        tests = record["test_cases"]
        if test_index is not None and not 1 <= test_index <= len(tests):
            raise ValueError("test_index is one-based and out of range.")
        parts = [self.problem["required_dependencies"]]
        if self.implementation == "scicode":
            parts.append("from test_util import are_dicts_close, cmp_tuple_or_list")
        parts.extend(
            replacements.get(s, self.steps[s]["code"])
            for s in order[: order.index(step_id) + 1]
        )
        module = (
            "process_data"
            if self.implementation == "scicode"
            else "scicode.parse.parse"
        )
        filename = (
            "test_data.h5"
            if self.implementation == "scicode"
            else "test_data_cleaned.h5"
        )
        if self.implementation == "scicode":
            parts.append("# Test cases:")
        target_args = f"{step_id!r}, {len(tests)}" + (
            f", {filename!r}" if self.implementation == "scicode_verified" else ""
        )
        parts.extend(
            [
                f"from {module} import process_hdf5_to_tuple",
                f"targets = process_hdf5_to_tuple({target_args})",
            ]
        )
        for i, test in enumerate(tests):
            if test_index is None or test_index == i + 1:
                parts.extend([f"target = targets[{i}]", test])
        return ("\n" if self.implementation == "scicode" else "\n\n").join(parts)


async def load_cases(t: Transcript, implementation: Implementation) -> CaseSet:
    """Source logs are authoritative for settings, supplied code, and attachments."""
    if not t.source_uri:
        raise ValueError("A source Inspect log URI is required for complete evidence.")
    enable_zstd_zip()
    sample, header = await asyncio.gather(
        asyncio.to_thread(
            read_eval_log_sample,
            t.source_uri,
            uuid=t.transcript_id,
            resolve_attachments=True,
        ),
        asyncio.to_thread(read_eval_log, t.source_uri, header_only=True),
    )
    return cases_from_sample(
        t,
        implementation,
        sample,
        header.eval.task_args,
        getattr(header.eval, "revision", None),
    )


def cases_from_sample(t, implementation, sample, settings, revision=None) -> CaseSet:
    if settings.get("scaling"):
        raise ValueError("This scanner supports the default non-scaling harness only.")
    problem = redact(sample.metadata)
    raw_steps = problem["sub_steps"]
    expected_scorer = "verify" if implementation == "scicode" else "verify_scicode"
    if expected_scorer not in sample.scores:
        raise ValueError(
            f"Expected {expected_scorer} for implementation={implementation}."
        )
    score = sample.scores[expected_scorer]
    values = score.value
    if not isinstance(values, dict):
        raise TypeError("Missing per-subproblem scores.")
    messages: dict[str, Any] = {}
    grading: dict[str, Any] = dict((score.metadata or {}).get("per_environment", {}))
    for event in sample.events:
        if event.event == "sandbox" and event.action == "exec" and event.cmd:
            command = shlex.split(event.cmd)
            if "-c" in command:
                program = command[command.index("-c") + 1]
                match = re.search(
                    r"targets\s*=\s*process_hdf5_to_tuple\(['\"]([\d.]+)['\"]", program
                )
                if match and match[1] in values and implementation == "scicode":
                    grading[match[1]] = {
                        "exit_code": event.result,
                        "output": event.output,
                        "timeout": (event.options or {}).get("timeout"),
                        "executed_program": program,
                        "event_ref": f"E:{event.uuid}",
                    }
        if (
            event.event == "score"
            and isinstance(event.score.value, dict)
            and len(event.score.value) == 1
        ):
            sid = next(iter(event.score.value))
            if sid in values:
                grading[sid] = {
                    "explanation": event.score.explanation,
                    "executed_program": event.score.answer,
                    "event_ref": f"E:{event.uuid}",
                }
        if event.event != "model" or not event.output.choices:
            continue
        users = [m for m in event.input if m.role == "user"]
        if not users:
            continue
        prompt = users[-1].text
        # Verified prompts include earlier descriptions; the last matching step is current.
        matches = [
            s for s in raw_steps if s["step_description_prompt"].strip() in prompt
        ]
        if not matches:
            raise ValueError(
                f"Cannot associate model event {event.uuid} with a subproblem."
            )
        current = matches[-1]
        sid = current["step_number"]
        # Last successful response wins for a retried step. Empty final responses are retained.
        messages[sid] = event
    from pathlib import Path

    steps = {}
    stored = (
        sample.store.get("scicode_codes")
        if implementation == "scicode_verified"
        else None
    )
    if stored is not None and len(stored) != len(raw_steps):
        raise ValueError("Stored Verified code does not match the number of steps.")
    for i, record in enumerate(raw_steps):
        sid = record["step_number"]
        provided = record.get("provided_code") is not None or (
            implementation == "scicode_verified" and sid in PROVIDED
        )
        event = messages.get(sid)
        if provided:
            code = record.get("provided_code")
            if code is None:
                code = extract_code(
                    (
                        Path(__file__).parent / "vendor" / "provided" / f"{sid}.txt"
                    ).read_text(),
                    implementation,
                )
            steps[sid] = {
                "record": record,
                "provided": True,
                "code": code,
                "response": code,
                "current_prompt": record["step_description_prompt"],
                "system": [],
                "response_ref": f"provided:{sid}",
                "reasoning_available": False,
            }
            continue
        if event is None or sid not in values:
            raise ValueError(f"Missing model response or score for {sid}.")
        msg = event.output.choices[0].message
        user = [m for m in event.input if m.role == "user"][-1].text
        if implementation == "scicode_verified":
            # Keep the instructions and current section, exclude earlier code/descriptions.
            marker = "NEXT STEP - "
            if marker not in user:
                raise ValueError(f"Unrecognised Verified prompt layout for {sid}.")
            prefix = re.split(
                r"PREVIOUS STEPS DESCRIPTION:|PROBLEM STEPS AND FUNCTION CODE:",
                user,
                maxsplit=1,
            )[0]
            current_prompt = prefix + marker + user.rsplit(marker, 1)[1]
        else:
            current_prompt = user
        code = extract_code(msg.text, implementation)
        if stored is not None and code != stored[i]:
            raise ValueError(
                f"Reconstructed code differs from stored submission at {sid}."
            )
        steps[sid] = {
            "record": record,
            "provided": False,
            "code": code,
            "response": msg.text,
            "current_prompt": current_prompt,
            "system": [m.text for m in event.input if m.role == "system"],
            "response_ref": f"M:{msg.id}",
            "reasoning_available": any(
                getattr(c, "type", "") == "reasoning" for c in msg.content
            )
            if isinstance(msg.content, list)
            else False,
            "event": event,
            "stop_reason": event.output.choices[0].stop_reason,
        }
    revision_value = (
        revision.model_dump(mode="json")
        if hasattr(revision, "model_dump")
        else revision
    )
    return CaseSet(
        implementation,
        t,
        problem,
        values,
        settings,
        steps,
        sample.events,
        grading,
        revision_value,
        sample.epoch,
        sample.files or {},
    )
