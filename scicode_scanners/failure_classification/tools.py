"""On-demand, read-only evidence retrieval for inspection-only classification."""

import json
from typing import Any

from inspect_ai.tool import ToolDef

from .adapters import CaseSet, redact
from .assets import ROOT, decode_targets, typed


class Investigation:
    def __init__(self, cases: CaseSet, sid: str, targets_path):
        self.cases, self.sid, self.targets_path = cases, sid, targets_path
        self.trace: list[dict[str, Any]] = []
        self.previous_causes = []
        self.functions = {
            name: getattr(self, name)
            for name in (
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
