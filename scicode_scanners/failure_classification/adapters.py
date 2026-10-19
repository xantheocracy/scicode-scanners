"""Recover default-harness submissions without exposing reference answers."""

import ast
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
# Event fields that locate rather than describe an event.
EVENT_BOOKKEEPING = {
    "uuid",
    "span_id",
    "timestamp",
    "working_start",
    "pending",
    "metadata",
}


def code_outline(code: str) -> tuple[str, dict[str, str], set[str]]:
    """Return parse status, top-level definitions with source, and referenced names."""
    if not code.strip():
        return "empty", {}, set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "does_not_parse", {}, set()
    definitions = {
        node.name: ast.get_source_segment(code, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    # Name loads cover calls and functions passed as values, e.g. solver(f, x).
    referenced = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return "ok", definitions, referenced


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
        """Raise one ValueError listing every problem, so a single correction suffices."""
        order = list(self.steps)
        origins = cause.origin_steps
        problems = []

        def invalid_steps(steps):
            return [
                s
                for s in steps
                if s not in self.steps or order.index(s) > order.index(affected)
            ]

        if bad := invalid_steps(origins):
            problems.append(
                f"origin_steps must be existing current or preceding step IDs; invalid: {bad}"
            )
        if cause.origin_type == "current" and origins != [affected]:
            problems.append(
                f"origin_type current requires origin_steps == [{affected!r}]."
            )
        if cause.origin_type == "earlier" and (not origins or affected in origins):
            problems.append(
                f"origin_type earlier requires preceding step IDs only, excluding {affected!r}."
            )
        if cause.origin_type == "author_provided" and (
            not origins
            or any(s in self.steps and not self.steps[s]["provided"] for s in origins)
        ):
            problems.append(
                "origin_type author_provided requires origin_steps naming supplied-code steps."
            )
        if cause.origin_type in {"external", "unknown"} and origins:
            problems.append(
                "origin_type external/unknown requires an empty origin_steps list."
            )
        if bad := invalid_steps(cause.dependency_path):
            problems.append(
                f"dependency_path must contain only existing current or preceding step IDs (no descriptions); invalid: {bad}"
            )
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
        # References are often followed by prose punctuation, e.g. "X:1: stdout shows ...".
        cited = {
            reference.rstrip(".:,;")
            for evidence in cause.evidence
            for reference in re.findall(r"\b[METX]:[A-Za-z0-9_.:-]+", evidence)
        }
        if not cited:
            problems.append("evidence must cite at least one M/E/T/X reference.")
        elif cited - known:
            problems.append(
                "evidence cites unknown references (use exact M:<id>, E:<id>, T:<step>:<test>[:target] "
                f"with one test index, or X:<n>): {sorted(cited - known)}"
            )
        if problems:
            raise ValueError(" ".join(problems))

    def definitions(self, before: str | None = None) -> dict[str, tuple[str, str]]:
        """Latest top-level definition of each name, as (step, source), up to a step."""
        order = list(self.steps)
        steps = order[: order.index(before)] if before else order
        found = {}
        for sid in steps:
            for name, source in code_outline(self.steps[sid]["code"])[1].items():
                found[name] = (sid, source)
        return found

    def recorded_issues(self) -> list[dict[str, Any]]:
        """Error, limit and unassigned generation events; usually empty."""
        assigned = {
            id(step["event"])
            for step in self.steps.values()
            if step.get("event") is not None
        }
        issues = []
        for event in self.events:
            kind = getattr(event, "event", None)
            reference = f"E:{event.uuid}"
            if kind == "model" and id(event) not in assigned:
                choices = getattr(event.output, "choices", None) or []
                issues.append(
                    {
                        "reference": reference,
                        "event": "model",
                        "stop_reason": choices[0].stop_reason if choices else None,
                        "error": getattr(event, "error", None),
                    }
                )
            elif kind in {"error", "sample_limit"}:
                raw = redact(event.model_dump(mode="json"))
                if isinstance(raw.get("error"), dict):
                    # Duplicates the traceback with terminal colour codes.
                    raw["error"].pop("traceback_ansi", None)
                issues.append(
                    {
                        "reference": reference,
                        **{k: v for k, v in raw.items() if k not in EVENT_BOOKKEEPING},
                    }
                )
        return issues

    def packet(self, step_id: str) -> dict[str, Any]:
        order = list(self.steps)
        step = self.steps[step_id]
        status, _, referenced = code_outline(step["code"])
        earlier = self.definitions(before=step_id)
        used = {n for n in referenced if n in earlier}

        def index_entry(sid):
            earlier_step = self.steps[sid]
            code_status, defs, _ = code_outline(earlier_step["code"])
            description = earlier_step["record"]["step_description_prompt"].strip()
            # Headers embed full docstrings; the signature line suffices for the index.
            header = (earlier_step["record"].get("function_header") or "").strip()
            stop_reason = earlier_step.get("stop_reason")
            return {
                "step": sid,
                "provided": earlier_step["provided"],
                "summary": description.split("\n", 1)[0][:300],
                "signature": header.split("\n", 1)[0],
                "defines": list(defs),
                "used_by_current": sorted(n for n in defs if n in used),
                "code_chars": len(earlier_step["code"]),
                "code_status": code_status,
                "response_reference": earlier_step["response_ref"],
                # Only abnormal stops (e.g. max_tokens) are informative.
                **(
                    {"stop_reason": stop_reason}
                    if stop_reason not in {None, "stop"}
                    else {}
                ),
            }

        return {
            "implementation": self.implementation,
            "affected_step": step_id,
            "current_solution": {
                "step": step_id,
                "provided": step["provided"],
                "description": step["record"]["step_description_prompt"],
                "interface": {
                    k: step["record"].get(k) for k in ("function_header", "return_line")
                },
                "submitted_code": step["code"],
                "code_status": status,
                "response_reference": step["response_ref"],
                "stop_reason": step.get("stop_reason"),
            },
            "earlier_steps": [
                index_entry(sid) for sid in order[: order.index(step_id)]
            ],
            "recorded_issues": self.recorded_issues(),
            "dependencies": self.problem["required_dependencies"],
            "tests": [
                {"reference": f"T:{step_id}:{i + 1}", "source": test}
                for i, test in enumerate(step["record"]["test_cases"])
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
