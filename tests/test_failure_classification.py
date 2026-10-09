import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall
from inspect_scout import Transcript

from scicode_scanners.failure_classification.adapters import (
    cases_from_sample,
    extract_code,
    redact,
)
from scicode_scanners.failure_classification.assets import (
    decode_targets,
    shard_targets,
    typed,
)
from scicode_scanners.failure_classification.checkpoints import CheckpointStore
from scicode_scanners.failure_classification.judge import investigate
from scicode_scanners.failure_classification.scanner import reconcile
from scicode_scanners.failure_classification.schema import Assessment, Cause, Limits
from scicode_scanners.failure_classification.tools import Investigation


def event(sid, description, response, *, verified=False):
    prompt = description
    if verified:
        prompt = f"PROBLEM DESCRIPTION:\nInstructions\nPREVIOUS STEPS DESCRIPTION:\nsecret earlier code\nNEXT STEP - PROBLEM DESCRIPTION AND FUNCTION HEADER:\n{description}"
    msg = ChatMessageAssistant(id=f"m-{sid}", content=response)
    return SimpleNamespace(
        event="model",
        uuid=f"e-{sid}",
        input=[ChatMessageUser(content=prompt)],
        output=SimpleNamespace(
            choices=[SimpleNamespace(message=msg, stop_reason="stop")]
        ),
    )


def fixture_cases(verified=False):
    impl = "scicode_verified" if verified else "scicode"
    descriptions = ["Define f.", "Define g using f."]
    records = [
        {
            "step_number": f"1.{i + 1}",
            "step_description_prompt": d,
            "function_header": f"def {'fg'[i]}():",
            "return_line": "return value",
            "test_cases": [f"assert {'fg'[i]}() == target"],
            "ground_truth_code": "FORBIDDEN_REFERENCE",
        }
        for i, d in enumerate(descriptions)
    ]
    events = [
        event(
            "1.1",
            descriptions[0],
            "```python\ndef f(): return 0\n```",
            verified=verified,
        ),
        event(
            "1.2",
            descriptions[1],
            "```python\ndef g(): return f()\n```",
            verified=verified,
        ),
    ]
    sample = SimpleNamespace(
        metadata={
            "problem_id": "1",
            "required_dependencies": "",
            "sub_steps": records,
            "general_solution": "FORBIDDEN_GENERAL",
        },
        store={},
        events=events,
        scores={
            "verify_scicode" if verified else "verify": SimpleNamespace(
                value={"1.1": 0, "1.2": 0}, metadata={}, explanation=None
            )
        },
        epoch=1,
        files={},
    )
    t = Transcript(transcript_id="transcript", source_uri="fixture")
    return cases_from_sample(t, impl, sample, {"provide_scientific_background": False})


def assessment():
    return Assessment(
        status="resolved",
        causes=[
            Cause(
                category="model_error",
                mechanism="f returns zero instead of one",
                origin_type="earlier",
                origin_steps=["1.1"],
                dependency_path=["1.1", "1.2"],
                evidence=["M:m-1.1", "T:1.2:1"],
                causal_contribution="g calls f, so the wrong return is inherited",
                defect_key="f returns zero",
            )
        ],
        explanation="Inherited wrong return.",
        limitations=[],
        alternatives_considered=[],
    )


def test_reference_fields_removed_recursively():
    cases = fixture_cases(True)
    packet = json.dumps(cases.packet("1.2"))
    assert "FORBIDDEN" not in json.dumps(cases.problem)
    assert "secret earlier code" not in packet
    assert "NEXT STEP" not in packet
    assert (
        cases.packet("1.2")["previous_solutions"][0]["submitted_code"].strip()
        == "def f(): return 0"
    )
    assert redact({"x": [{"ground_truth_code": "secret", "ok": 1}]}) == {
        "x": [{"ok": 1}]
    }


def test_extraction_matches_harness_differences():
    text = "```python\nimport numpy as np\ndef f(): return 1\n```"
    assert "import numpy" in extract_code(text, "scicode")
    assert "import numpy" not in extract_code(text, "scicode_verified")
    assert extract_code("```\ndef f(): return 1\n```", "scicode").startswith("```")
    assert (
        extract_code("```\ndef f(): return 1\n```", "scicode_verified")
        .strip()
        .startswith("def f")
    )


def test_retry_response_wins_without_count_alignment():
    c = fixture_cases()
    # Construct a sample with an additional response for the first step.
    events = [
        c.steps["1.1"]["event"],
        event("1.1", "Define f.", "```python\ndef f(): return 2\n```"),
        c.steps["1.2"]["event"],
    ]
    sample = SimpleNamespace(
        metadata=c.problem,
        store={},
        events=events,
        scores={"verify": SimpleNamespace(value=c.scores, metadata={})},
        epoch=1,
        files={},
    )
    recovered = cases_from_sample(c.transcript, "scicode", sample, {})
    assert recovered.steps["1.1"]["code"] == "def f(): return 2"


def test_cumulative_grading_and_minimal_replacement():
    c = fixture_cases()
    original = c.grading_script("1.2")
    variant = c.grading_script("1.2", {"1.1": "def f(): return 1"})
    assert "def f(): return 0" in original and "def g(): return f()" in original
    assert "def f(): return 1" in variant and "def g(): return f()" in variant
    assert "from test_util" in original
    assert "from scicode.parse.parse" in fixture_cases(True).grading_script("1.2")
    with pytest.raises(ValueError):
        c.grading_script("1.1", {"1.2": "invalid future patch"})


def test_origins_and_conservative_deduplication():
    c = fixture_cases()
    cause = assessment().causes[0]
    c.validate_cause(cause, "1.2")
    with pytest.raises(ValueError):
        c.validate_cause(cause.model_copy(update={"origin_steps": ["1.2"]}), "1.2")
    first = reconcile(assessment(), "scicode", "t")
    assert (
        first[0]["cause_id"] == reconcile(assessment(), "scicode", "t")[0]["cause_id"]
    )
    assert (
        first[0]["cause_id"]
        != reconcile(assessment(), "scicode", "different_attempt")[0]["cause_id"]
    )


def test_unresolved_is_not_other():
    a = Assessment(
        status="unresolved",
        causes=[],
        explanation="Missing evidence.",
        limitations=["missing results"],
        alternatives_considered=[],
    )
    assert reconcile(a, "scicode", "t") == []
    with pytest.raises(ValueError):
        Assessment(
            status="resolved",
            causes=[],
            explanation="No cause.",
            limitations=[],
            alternatives_considered=[],
        )


@pytest.mark.parametrize("impl", ["scicode", "scicode_verified"])
def test_exact_decoders_and_target_shards(tmp_path, impl):
    path = tmp_path / "targets.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("1.1/test1/var1", data=np.array([1 + 2j, 3 + 4j]))
        f.create_dataset("1.2/test1/var1", data=9)
    targets = decode_targets(path, impl, "1.1", 1)
    np.testing.assert_array_equal(targets[0], [1 + 2j, 3 + 4j])
    shard = tmp_path / "shard.h5"
    shard.write_bytes(shard_targets(path, ["1.1"]))
    with h5py.File(shard) as f:
        assert list(f) == ["1.1"]
    preview = typed(np.arange(100), offset=32, limit=8)
    assert preview["values"] == list(range(32, 40)) and preview["truncated"]


class MockModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.requests = []
        self.tool_choices = []

    async def generate(self, history, tools, config, tool_choice=None):
        self.tool_choices.append(tool_choice)
        self.requests.append((list(history), tools, config))
        return self.outputs.pop(0)


def output(call, tokens, reasoning=0):
    result = ModelOutput.from_message(
        ChatMessageAssistant(content="", tool_calls=[call])
    )
    result.usage = ModelUsage(
        input_tokens=100,
        output_tokens=tokens,
        total_tokens=100 + tokens,
        reasoning_tokens=reasoning,
    )
    return result


@pytest.mark.asyncio
async def test_budget_counts_reasoning_once_and_informs_judge():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    model = MockModel(
        [
            output(
                ToolCall(
                    id="call",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                1200,
                1000,
            )
        ]
    )
    a, provenance = await investigate(model, inv, c.packet("1.2"), Limits())
    assert a.status == "resolved" and provenance["generated_tokens"] == 1200
    request, _, config = model.requests[0]
    assert config.max_tokens == 4096 and config.max_retries == 0
    assert config.extra_body["reasoning"]["effort"] == "high"
    update = json.loads(request[-1].text)["budget_update"]
    assert (
        update["remaining_generated_tokens"] == 32768
        and update["finalization_reserve"] == 8192
    )


@pytest.mark.asyncio
async def test_finalization_disables_tools_and_respects_remaining_budget():
    limits = Limits(
        generated_token_budget=2048,
        finalization_reserve=1024,
        per_call_tokens=1024,
        tool_rounds=1,
    )
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    model = MockModel(
        [
            output(
                ToolCall(
                    id="bad", function="retrieve_step", arguments={"step_id": "bad"}
                ),
                1000,
            ),
            output(
                ToolCall(
                    id="answer",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                600,
            ),
        ]
    )
    _, provenance = await investigate(model, inv, c.packet("1.2"), limits)
    assert provenance["generated_tokens"] == 1600
    _, tools, config = model.requests[1]
    assert [d.name for d in tools] == ["answer"] and config.max_tokens == 1048


@pytest.mark.asyncio
async def test_checkpoint_reserves_charge_before_request_and_reuses_answer(tmp_path):
    store = CheckpointStore(str(tmp_path), {"case": "1.2"})
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)

    class CheckedModel(MockModel):
        async def generate(self, history, tools, config, tool_choice=None):
            saved = await store.read()
            assert saved["used"] == config.max_tokens
            return await super().generate(history, tools, config)

    model = CheckedModel(
        [
            output(
                ToolCall(
                    id="answer",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                1200,
            )
        ]
    )
    a, provenance = await investigate(
        model, inv, c.packet("1.2"), Limits(), checkpoint=store
    )
    no_calls = MockModel([])
    cached, cached_provenance = await investigate(
        no_calls, inv, c.packet("1.2"), Limits(), checkpoint=store
    )
    assert cached == a and cached_provenance == provenance and not no_calls.requests


@pytest.mark.asyncio
async def test_interrupted_call_consumes_reservation_on_resume(tmp_path):
    store = CheckpointStore(str(tmp_path), {"case": "interrupted"})
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)

    class InterruptedModel:
        async def generate(self, *args, **kwargs):
            raise RuntimeError("Connection lost after sending request")

    with pytest.raises(RuntimeError):
        await investigate(
            InterruptedModel(), inv, c.packet("1.2"), Limits(), checkpoint=store
        )
    assert (await store.read())["used"] == 4096
    model = MockModel(
        [
            output(
                ToolCall(
                    id="answer",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                1000,
            )
        ]
    )
    _, provenance = await investigate(
        model, inv, c.packet("1.2"), Limits(), checkpoint=store
    )
    assert provenance["generated_tokens"] == 5096


@pytest.mark.asyncio
async def test_empty_provider_responses_cannot_create_unbounded_input_spend():
    c = fixture_cases()
    limits = Limits()
    inv = Investigation(c, "1.2", None)
    empty = ModelOutput.from_message(ChatMessageAssistant(content=""))
    empty.usage = ModelUsage(input_tokens=100, output_tokens=0, total_tokens=100)
    model = MockModel([empty] * (limits.tool_rounds + 4))
    a, provenance = await investigate(model, inv, c.packet("1.2"), limits)
    assert a.status == "unresolved" and provenance["termination"] == "model_call_limit"
    assert len(model.requests) == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [False, True])
async def test_inspection_only_scanner_preserves_causes_without_sandbox(
    monkeypatch, verified
):
    import builtins
    import importlib

    scanner_module = importlib.import_module(
        "scicode_scanners.failure_classification.scanner"
    )
    c = fixture_cases(verified)
    first = assessment()
    first.causes[0].origin_type = "current"
    first.causes[0].dependency_path = ["1.1"]
    model = MockModel(
        [
            output(
                ToolCall(
                    id="first",
                    function="answer",
                    arguments={"assessment": first.model_dump()},
                ),
                800,
            ),
            output(
                ToolCall(
                    id="second",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                800,
            ),
        ]
    )

    async def cases(*args):
        return c

    async def targets(*args):
        return "fixture.h5"

    original_import = builtins.__import__

    def forbid_sandbox(name, *args, **kwargs):
        if name.startswith("k8s_sandbox"):
            raise AssertionError("Inspection must not import a sandbox provider")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", forbid_sandbox)
    monkeypatch.setattr(scanner_module, "load_cases", cases)
    monkeypatch.setattr(scanner_module, "target_file", targets)
    monkeypatch.setattr(scanner_module, "decode_targets", lambda *args: [1])
    monkeypatch.setattr(scanner_module, "get_model", lambda: model)
    monkeypatch.setattr(scanner_module, "hawk_results_uri", lambda: None)
    model.name = "mock-judge"
    results = await scanner_module.failure_classification(c.implementation)(
        c.transcript
    )
    assert [r.label for r in results] == ["1.1", "1.2"]
    causes = [r.metadata["assessment"]["causes"][0] for r in results]
    assert causes[0]["cause_id"] == causes[1]["cause_id"]
    for result in results:
        assert result.metadata["investigation_mode"] == "inspection_only"
        assert (
            result.metadata["investigation"]["investigation_mode"] == "inspection_only"
        )
    for history, tools, _ in model.requests:
        assert {t.name for t in tools} == {
            "answer",
            "current_evidence",
            "retrieve_step",
            "full_transcript",
            "read_message",
            "inspect_target",
        }
        assert "Never claim to have executed code" in history[0].text
        update = json.loads(history[-1].text)["budget_update"]
        assert update["investigation_mode"] == "inspection_only"
        assert "remaining_python_seconds" not in update


@pytest.mark.asyncio
async def test_retrieval_keeps_previous_findings_out_of_default_context():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    prior = [{"defect_key": "previous hypothesis"}]
    model = MockModel(
        [
            output(
                ToolCall(
                    id="answer",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                500,
            )
        ]
    )
    await investigate(model, inv, c.packet("1.2"), Limits(), previous_causes=prior)
    assert "previous hypothesis" not in json.dumps(
        [m.model_dump(mode="json") for m in model.requests[0][0]]
    )
    assert "prior_findings" not in inv.functions


@pytest.mark.asyncio
async def test_large_step_and_transcript_retrieval_are_paginated():
    from scicode_scanners.failure_classification.tools import PAGE_CHARS

    c = fixture_cases()
    huge = "reasoning and previous code " * 60000
    c.steps["1.1"]["event"].input = [ChatMessageUser(content=huge)]
    c.events = [
        SimpleNamespace(
            uuid="large",
            model_dump=lambda **kwargs: {"event": "sandbox", "output": huge},
        )
    ]
    inv = Investigation(c, "1.2", None)
    for tool, args in (
        (inv.retrieve_step, {"step_id": "1.1", "section": "prompt"}),
        (inv.full_transcript, {}),
    ):
        page = json.loads(await tool(**args))
        assert page["total_chars"] > 1000000
        assert len(page["excerpt"]) == PAGE_CHARS
        following = json.loads(await tool(**args, char_offset=page["next_char_offset"]))
        assert following["char_offset"] == PAGE_CHARS
        assert len(following["excerpt"]) <= PAGE_CHARS
        final = json.loads(await tool(**args, char_offset=page["total_chars"] - 100))
        assert final["next_char_offset"] is None and len(final["excerpt"]) == 100


@pytest.mark.asyncio
async def test_large_initial_evidence_is_bounded_and_remains_retrievable():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    packet = {**c.packet("1.2"), "large": "a" * 1000000}
    model = MockModel(
        [
            output(
                ToolCall(
                    id="answer",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                500,
            )
        ]
    )
    await investigate(model, inv, packet, Limits())
    initial = json.loads(model.requests[0][0][1].text)
    assert initial["next_char_offset"] == 12000
    assert len(initial["excerpt"]) == 12000
    remaining = json.loads(await inv.current_evidence(char_offset=12000))
    assert remaining["char_offset"] == 12000 and remaining["total_chars"] > 1000000
    serialized = json.dumps({"case": packet}, ensure_ascii=False, default=str)
    assert initial["excerpt"] == serialized[:12000]
    assert remaining["excerpt"] == serialized[12000:24000]


@pytest.mark.asyncio
async def test_context_limit_prevents_oversized_followup_requests():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    giant = ModelOutput.from_message(ChatMessageAssistant(content="x" * 100001))
    giant.usage = ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11)
    model = MockModel([giant])
    result, provenance = await investigate(model, inv, c.packet("1.2"), Limits())
    assert len(model.requests) == 1
    assert result.status == "unresolved"
    assert provenance["termination"] == "input_context_limit"


def test_initial_packet_contains_only_scoped_evidence():
    c = fixture_cases()
    c.steps["1.2"]["response"] = "UNNEEDED_RESPONSE"
    c.steps["1.2"]["current_prompt"] = "UNNEEDED_PROMPT"
    c.grading["1.2"] = {"output": "UNNEEDED_GRADING"}
    packet = c.packet("1.2")
    assert set(packet) == {
        "implementation",
        "affected_step",
        "current_solution",
        "previous_solutions",
        "dependencies",
        "tests",
    }
    assert "UNNEEDED" not in json.dumps(packet)
    assert [s["step"] for s in packet["previous_solutions"]] == ["1.1"]
    assert not c.packet("1.1")["previous_solutions"]


def test_large_target_metadata_has_no_values():
    from scicode_scanners.failure_classification.assets import target_metadata

    metadata = target_metadata(np.zeros((1000, 1000)))
    assert metadata == {"type": "ndarray", "shape": [1000, 1000], "dtype": "float64"}
    assert len(json.dumps(metadata)) < 100


@pytest.mark.asyncio
async def test_matrix_target_slice(monkeypatch):
    import scicode_scanners.failure_classification.tools as tools_module

    matrix = np.arange(100).reshape(10, 10)
    monkeypatch.setattr(tools_module, "decode_targets", lambda *args: [matrix])
    inv = Investigation(fixture_cases(), "1.2", "fixture.h5")
    result = json.loads(
        await inv.inspect_target(
            "1.2", 1, row_start=2, row_stop=4, column_start=3, column_stop=5
        )
    )
    assert result["value"]["shape"] == [2, 2]
    assert result["value"]["values"] == [23, 24, 33, 34]
    assert result["selection"] == {"rows": [2, 4], "columns": [3, 5]}
    with pytest.raises(ValueError, match="nonnegative"):
        await inv.inspect_target("1.2", 1, row_start=-1)
    monkeypatch.setattr(tools_module, "decode_targets", lambda *args: [np.arange(10)])
    with pytest.raises(ValueError, match="two-dimensional"):
        await inv.inspect_target("1.2", 1, row_start=0)


@pytest.mark.asyncio
async def test_target_metadata_only_supports_nested_matrices(monkeypatch):
    import scicode_scanners.failure_classification.tools as tools_module

    matrix = np.zeros((1000, 2000))
    monkeypatch.setattr(
        tools_module, "decode_targets", lambda *args: [{"matrix": matrix}]
    )
    inv = Investigation(fixture_cases(), "1.2", "fixture.h5")
    result = json.loads(
        await inv.inspect_target("1.2", 1, path=["matrix"], metadata_only=True)
    )
    assert result["metadata"] == {
        "type": "ndarray",
        "shape": [1000, 2000],
        "dtype": "float64",
    }
    assert "value" not in result
    with pytest.raises(ValueError, match="without matrix slice bounds"):
        await inv.inspect_target("1.2", 1, metadata_only=True, row_start=0)


@pytest.mark.asyncio
async def test_production_budget_preserves_full_answer_reserve():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    limits = Limits()
    model = MockModel(
        [
            *[
                output(
                    ToolCall(
                        id=f"read-{i}",
                        function="retrieve_step",
                        arguments={"step_id": "missing"},
                    ),
                    4096,
                )
                for i in range(6)
            ],
            output(
                ToolCall(
                    id="final",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                7000,
            ),
        ]
    )
    result, provenance = await investigate(model, inv, c.packet("1.2"), limits)
    assert result.status == "resolved"
    assert provenance["generated_tokens"] == 6 * 4096 + 7000
    _, tools, config = model.requests[-1]
    assert [tool.name for tool in tools] == ["answer"]
    assert config.max_tokens == 8192
    assert config.extra_body["reasoning"]["effort"] == "low"
    assert model.tool_choices[-1] == "any"


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [False, True])
async def test_selective_retrieval_does_not_duplicate_solutions(verified):
    c = fixture_cases(verified)
    for event in c.events:
        event.model_dump = lambda **kwargs: {
            "event": "model",
            "input": "DUPLICATED_PREVIOUS_CODE",
            "output": "DUPLICATED_RESPONSE",
        }
    c.grading["1.2"] = {
        "output": "AssertionError",
        "executed_program": "DUPLICATED_PROGRAM",
    }
    inv = Investigation(c, "1.2", None)
    default = await inv.retrieve_step("1.2")
    assert "AssertionError" in default
    assert "DUPLICATED" not in default and "def g" not in default
    transcript = json.loads(await inv.full_transcript(query="DUPLICATED"))
    assert transcript["total"] == 2
    assert "DUPLICATED" not in json.dumps(transcript)
    assert "def f" not in json.dumps(transcript)
    assert transcript["events"][0]["messages"][-1]["reference"] == "M:m-1.1"
    selected = json.loads(await inv.read_message("M:m-1.1"))
    assert "def f" in selected["text"]
    assert "def g" not in selected["text"]
    assert "messages" not in json.loads(
        await inv.retrieve_step("1.1", section="reasoning")
    )
    assert "messages" in json.loads(await inv.retrieve_step("1.1", section="prompt"))


@pytest.mark.asyncio
async def test_context_pressure_triggers_answer_before_hard_limit():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)
    large = ModelOutput.from_message(ChatMessageAssistant(content="x" * 75000))
    large.usage = ModelUsage(input_tokens=10, output_tokens=100, total_tokens=110)
    model = MockModel(
        [
            large,
            output(
                ToolCall(
                    id="final",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                500,
            ),
        ]
    )
    result, provenance = await investigate(model, inv, c.packet("1.2"), Limits())
    assert result.status == "resolved"
    assert provenance["termination"] == "answered"
    assert [tool.name for tool in model.requests[-1][1]] == ["answer"]
    assert model.tool_choices[-1] == "any"


@pytest.mark.asyncio
async def test_reasoning_retrieval_excludes_response_code():
    from inspect_ai.model import ContentReasoning, ContentText

    c = fixture_cases(True)
    c.steps["1.1"]["event"].output.choices[0].message.content = [
        ContentReasoning(reasoning="A specific reasoning observation."),
        ContentText(text="DUPLICATE_SOLUTION_CODE"),
    ]
    inv = Investigation(c, "1.2", None)
    for result in (
        await inv.retrieve_step("1.1", section="reasoning"),
        await inv.read_message("m-1.1", section="reasoning"),
    ):
        assert "A specific reasoning observation." in result
        assert "DUPLICATE_SOLUTION_CODE" not in result


@pytest.mark.asyncio
async def test_oversized_retrieval_is_withheld_and_answer_still_submitted():
    c = fixture_cases()
    inv = Investigation(c, "1.2", None)

    async def oversized(**kwargs):
        return "x" * 90000

    inv.functions["retrieve_step"] = oversized
    model = MockModel(
        [
            output(
                ToolCall(
                    id="read", function="retrieve_step", arguments={"step_id": "1.1"}
                ),
                100,
            ),
            output(
                ToolCall(
                    id="final",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                500,
            ),
        ]
    )
    result, provenance = await investigate(model, inv, c.packet("1.2"), Limits())
    assert result.status == "resolved"
    assert "Evidence withheld" in provenance["tools"][0]["result"]
    assert model.tool_choices[-1] == "any"
    assert (
        len(
            json.dumps(
                [m.model_dump(mode="json") for m in model.requests[-1][0]]
            ).encode()
        )
        < 100000
    )


@pytest.mark.asyncio
async def test_operational_error_submission_uses_recorded_generation_limit():
    c = fixture_cases()
    c.steps["1.2"]["stop_reason"] = "max_tokens"
    inv = Investigation(c, "1.2", None)
    evidence = json.loads(await inv.retrieve_step("1.2"))
    assert evidence["generation"] == {
        "event_reference": "E:e-1.2",
        "stop_reason": "max_tokens",
    }
    assert "code" not in evidence
    a = assessment()
    cause = a.causes[0]
    cause.category = "operational_error"
    cause.origin_type = "current"
    cause.origin_steps = ["1.2"]
    cause.dependency_path = ["1.2"]
    cause.mechanism = "Recorded generation cutoff left submitted code incomplete."
    cause.evidence = ["E:e-1.2 recorded max_tokens stop"]
    cause.defect_key = "submission truncated by generation limit"
    model = MockModel(
        [
            output(
                ToolCall(
                    id="answer",
                    function="answer",
                    arguments={"assessment": a.model_dump()},
                ),
                500,
            )
        ]
    )
    result, provenance = await investigate(model, inv, c.packet("1.2"), Limits())
    assert result.causes[0].category == "operational_error"
    assert provenance["termination"] == "answered"
    assert "operational_error" in model.requests[0][0][0].text
