"""On-demand, read-only evidence retrieval for inspection-only classification."""

import json
from typing import Any

from inspect_ai.tool import ToolDef

from .adapters import CaseSet, redact
from .assets import ROOT, decode_targets, typed

PAGE_CHARS = 12000


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
    def __init__(self, cases: CaseSet, sid: str, targets_path):
        self.cases, self.sid, self.targets_path = cases, sid, targets_path
        self.trace: list[dict[str, Any]] = []
        self.previous_causes = []
        self.packet = {}
        self.functions = {
            name: getattr(self, name)
            for name in (
                "current_evidence",
                "retrieve_step",
                "full_transcript",
                "inspect_target",
                "helper_source",
                "prior_findings",
            )
        }

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
        return evidence_page(self.packet, char_offset)

    async def prior_findings(self, char_offset: int = 0) -> str:
        """Retrieve earlier same-scan cause summaries to help reuse a defect key.

        These are judge-generated hypotheses, not ground truth or historical audit findings.
        Verify inherited causes using original evidence before attributing them here.
        """
        return evidence_page(self.previous_causes, char_offset)

    async def retrieve_step(self, step_id: str, char_offset: int = 0) -> str:
        """Retrieve a step's exact prompt, response including exposed reasoning, tests and original results.

        Args:
            char_offset: Character position within the selected JSON payload; follow next_char_offset.
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
        return evidence_page(evidence, char_offset)

    async def full_transcript(
        self, offset: int = 0, limit: int = 4, query: str = "", char_offset: int = 0
    ) -> str:
        """Read/search all transcript events, including previous steps, reasoning and grader events.

        Args:
            char_offset: Character position within the selected JSON payload; follow next_char_offset.
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
    ) -> str:
        """Inspect expected HDF5 output with typed previews and pagination, without changing it.

        Args:
            char_offset: Character position within the selected JSON payload; follow next_char_offset.
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
        return evidence_page(
            {
                "reference": f"T:{step_id}:{test_index}:target",
                "value": typed(value, offset, limit),
            },
            char_offset,
        )

    async def helper_source(self, name: str, char_offset: int = 0) -> str:
        """Read the exact target-decoding or comparator helper source used by this harness.

        Args:
            char_offset: Character position within the selected JSON payload; follow next_char_offset.
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
        return evidence_page({"name": name, "source": p.read_text()}, char_offset)
