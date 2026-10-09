"""On-demand, read-only evidence retrieval for inspection-only classification."""

import json
from typing import Any, Literal

from inspect_ai.tool import ToolDef

from .adapters import CaseSet, redact
from .assets import decode_targets, target_metadata, typed

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
                "read_message",
                "inspect_target",
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
