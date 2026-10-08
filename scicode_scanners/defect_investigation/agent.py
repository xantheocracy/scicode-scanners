"""Budget-aware ReAct loop with durable evidence and structured submission."""

import json

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolDef

from .prompts import INSTRUCTIONS
from .schema import Limits, Report
from .tools import Investigation, validate_report


@solver
def investigate(limits: Limits, grading_environments: str = "both") -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        investigation = Investigation(state, limits, grading_environments)
        state.messages.insert(0, ChatMessageSystem(content=INSTRUCTIONS))
        used, tool_count, repairs = 0, 0, 0
        state.store.set("defect_evidence", [])
        state.store.set("defect_termination", "budget_exhausted")

        async def submit_report(report: Report) -> str:
            """Submit the final report covering both directions for every selected step.

            Args:
                report: Structured findings with witness references, semantic arguments and confidence.
            """
            validate_report(report, state.metadata, investigation.evidence)
            state.store.set("defect_report", report.model_dump(mode="json"))
            return "Report accepted; independent replay and semantic review follow."

        functions = {
            "workspace": investigation.workspace,
            "grade_candidate": investigation.grade_candidate,
            "semantic_check": investigation.semantic_check,
            "submit_report": submit_report,
        }
        for turn in range(limits.model_calls):
            remaining = limits.generated_tokens - used
            finalizing = (
                remaining <= limits.finalization_reserve
                or tool_count >= limits.tool_calls
                or turn >= limits.model_calls - 2
            )
            cap = min(
                limits.per_call_tokens,
                remaining if finalizing else remaining - limits.finalization_reserve,
            )
            # Leave room to correct a rejected structured report.
            if finalizing and repairs == 0 and remaining > 1000:
                cap = min(cap, remaining - 1000)
            if cap < 1:
                break
            state.messages.append(
                ChatMessageUser(
                    content=json.dumps(
                        {
                            "remaining_generated_tokens": remaining,
                            "finalization_reserve": limits.finalization_reserve,
                            "remaining_tool_calls": max(
                                0, limits.tool_calls - tool_count
                            ),
                            "instruction": "Submit a compact report now; only submit_report is available. Use null candidate_evidence without canonical_grade evidence, and suspected/inconclusive for unverified claims. Do not repeat programs or diagnostics."
                            if finalizing
                            else "Investigate both directions or submit the report.",
                        }
                    )
                )
            )
            definitions = [ToolDef(submit_report, name="submit_report", parallel=False)]
            if not finalizing:
                definitions.extend(investigation.definitions)
            output = await get_model().generate(
                state.messages,
                tools=definitions,
                config=GenerateConfig(
                    max_tokens=cap,
                    max_retries=0,
                    parallel_tool_calls=False,
                    extra_body={
                        "reasoning": {
                            "effort": "low" if finalizing else limits.reasoning_effort
                        }
                    },
                ),
            )
            charged = output.usage.output_tokens if output.usage is not None else cap
            if charged < 0 or charged > cap:
                raise RuntimeError(
                    "Provider generation accounting violates the per-call cap."
                )
            used += max(
                1, charged
            )  # Output usage includes reasoning; never count it twice.
            state.store.set("defect_generated_tokens", used)
            state.store.set("defect_model_calls", turn + 1)
            state.messages.append(output.message)
            state.output = output
            calls = output.message.tool_calls or []
            if not calls:
                state.messages.append(
                    ChatMessageUser(
                        content="Use submit_report to finish; prose is not a report."
                    )
                )
                continue
            # Do not run possibly incomplete tool arguments after an output cutoff.
            if output.stop_reason == "max_tokens":
                for call in calls:
                    state.messages.append(
                        ChatMessageTool(
                            tool_call_id=call.id,
                            function=call.function,
                            content="Output was truncated; resubmit a complete request.",
                        )
                    )
                continue
            for call in calls:
                try:
                    if call.function not in functions:
                        raise ValueError("Unknown tool.")
                    if call.function != "submit_report" and (
                        finalizing or tool_count >= limits.tool_calls
                    ):
                        raise ValueError("Tool allowance exhausted; submit the report.")
                    if call.function == "submit_report":
                        content = await submit_report(
                            Report.model_validate(call.arguments["report"])
                        )
                    else:
                        tool_count += 1
                        content = await functions[call.function](**call.arguments)
                except (ValueError, KeyError, TypeError) as ex:
                    content = f"Invalid request: {ex}"
                    if call.function == "submit_report":
                        repairs += 1
                state.messages.append(
                    ChatMessageTool(
                        tool_call_id=call.id, function=call.function, content=content
                    )
                )
            state.store.set("defect_tool_calls", tool_count)
            if state.store.get("defect_report"):
                state.store.set("defect_termination", "submitted")
                break
            if repairs >= 2:
                state.store.set("defect_termination", "invalid_report")
                break
        state.completed = True
        return state

    return solve
