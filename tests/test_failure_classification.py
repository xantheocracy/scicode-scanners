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
    inv = Investigation(c, "1.2")
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
    inv = Investigation(c, "1.2")
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
    inv = Investigation(c, "1.2")

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
    inv = Investigation(c, "1.2")

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
    inv = Investigation(c, "1.2")
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
    results = await scanner_module.failure_classification(
        c.implementation, execution=False
    )(c.transcript)
    assert [r.label for r in results] == ["1.1", "1.2"]
    causes = [r.metadata["assessment"]["causes"][0] for r in results]
    assert causes[0]["cause_id"] == causes[1]["cause_id"]
    for result in results:
        assert result.value is False
        assert result.metadata["categories"] == ["model_error"]
        assert result.metadata["investigation_mode"] == "inspection_only"
        assert (
            result.metadata["investigation"]["investigation_mode"] == "inspection_only"
        )
    for history, tools, _ in model.requests:
        assert {t.name for t in tools} == {
            "answer",
            "retrieve_step",
            "read_code",
        }
        assert "Never claim to have executed code" in history[0].text
        update = json.loads(history[-1].text)["budget_update"]
        assert update["investigation_mode"] == "inspection_only"
        assert "remaining_python_seconds" not in update


@pytest.mark.asyncio
async def test_retrieval_keeps_previous_findings_out_of_default_context():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
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
    inv = Investigation(c, "1.2")
    for tool, args in ((inv.retrieve_step, {"step_id": "1.1", "section": "prompt"}),):
        page = json.loads(await tool(**args))
        assert page["total_chars"] > 1000000
        assert len(page["excerpt"]) == PAGE_CHARS
        following = json.loads(await tool(**args, char_offset=page["next_char_offset"]))
        assert following["char_offset"] == PAGE_CHARS
        assert len(following["excerpt"]) <= PAGE_CHARS
        final = json.loads(await tool(**args, char_offset=page["total_chars"] - 100))
        assert final["next_char_offset"] is None and len(final["excerpt"]) == 100


@pytest.mark.asyncio
async def test_initial_packet_is_sent_whole():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
    packet = {**c.packet("1.2"), "large": "a" * 60000}
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
    assert json.loads(model.requests[0][0][1].text) == {"case": packet}


@pytest.mark.asyncio
async def test_context_limit_prevents_oversized_followup_requests():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
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
        "earlier_steps",
        "recorded_issues",
        "dependencies",
        "tests",
    }
    assert "UNNEEDED" not in json.dumps(packet)
    assert [s["step"] for s in packet["earlier_steps"]] == ["1.1"]
    assert not c.packet("1.1")["earlier_steps"]


def test_large_target_metadata_has_no_values():
    from scicode_scanners.failure_classification.assets import target_metadata

    metadata = target_metadata(np.zeros((1000, 1000)))
    assert metadata == {"type": "ndarray", "shape": [1000, 1000], "dtype": "float64"}
    assert len(json.dumps(metadata)) < 100


@pytest.mark.asyncio
async def test_production_budget_preserves_full_answer_reserve():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
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
    inv = Investigation(c, "1.2")
    default = await inv.retrieve_step("1.2")
    assert "AssertionError" in default
    assert "DUPLICATED" not in default and "def g" not in default
    assert "messages" not in json.loads(
        await inv.retrieve_step("1.1", section="reasoning")
    )
    assert "messages" in json.loads(await inv.retrieve_step("1.1", section="prompt"))


@pytest.mark.asyncio
async def test_context_pressure_triggers_answer_before_hard_limit():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
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
    inv = Investigation(c, "1.2")
    result = await inv.retrieve_step("1.1", section="reasoning")
    assert "A specific reasoning observation." in result
    assert "DUPLICATE_SOLUTION_CODE" not in result


@pytest.mark.asyncio
async def test_oversized_retrieval_is_withheld_and_answer_still_submitted():
    c = fixture_cases()
    inv = Investigation(c, "1.2")

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
    inv = Investigation(c, "1.2")
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


@pytest.mark.parametrize(
    ("categories", "expected"),
    [
        (["model_error"], False),
        (["underspecified"], True),
        (["wrongly_specified"], True),
        (["operational_error"], True),
        (["other"], True),
        (["model_error", "underspecified"], True),
        ([], False),
    ],
)
def test_scout_positive_mapping(categories, expected):
    from scicode_scanners.failure_classification.scanner import is_positive_failure

    assert (
        is_positive_failure([{"category": category} for category in categories])
        is expected
    )


class RecordingSandbox:
    def __init__(self, passed=lambda cmd: True, stdout="", stderr=""):
        self.passed, self.stdout, self.stderr = passed, stdout, stderr
        self.commands = []

    async def exec(self, cmd, **kwargs):
        assert kwargs["timeout_retry"] is False
        self.commands.append(cmd)
        ok = self.passed(cmd)
        return SimpleNamespace(
            success=ok,
            returncode=0 if ok else 1,
            stdout=self.stdout,
            stderr=self.stderr,
        )


@pytest.mark.asyncio
async def test_python_timeout_is_not_recorded_as_original_failure():
    class TimeoutSandbox:
        async def exec(self, *args, **kwargs):
            assert kwargs["timeout_retry"] is False
            raise TimeoutError

    c = fixture_cases(True)
    inv = Investigation(c, "1.2", TimeoutSandbox(), Limits())
    result = json.loads(await inv.rerun("1.2", environment="2024"))
    assert result["per_environment"]["2024"]["timed_out"]
    assert "diagnostic" in result["note"]
    assert c.scores == {"1.1": 0, "1.2": 0}


@pytest.mark.asyncio
async def test_verified_accepts_second_environment_pass():
    c = fixture_cases(True)
    sandbox = RecordingSandbox(passed=lambda cmd: "2025" in cmd[0])
    inv = Investigation(c, "1.2", sandbox, Limits())
    result = json.loads(await inv.rerun("1.2"))
    assert result["passed"]
    assert not result["per_environment"]["2024"]["passed"]
    assert result["per_environment"]["2025"]["passed"]


@pytest.mark.asyncio
async def test_diagnostics_restore_canonical_artifacts_in_fresh_directories():
    from scicode_scanners.failure_classification.sandbox import FreshSandbox

    class RecordingEnvironment:
        def __init__(self):
            self.writes = []
            self.directories = []

        async def write_file(self, path, content):
            self.writes.append((path, content))

        async def exec(self, cmd, **kwargs):
            if "cwd" in kwargs:
                self.directories.append(kwargs["cwd"])
            return SimpleNamespace(success=True, returncode=0, stdout="", stderr="")

    env = RecordingEnvironment()
    sandbox = FreshSandbox(env, {"test_data.h5": b"canonical"})
    await sandbox.exec(["python", "-c", "modify targets"], timeout=30)
    await sandbox.exec(["python", "-c", "inspect targets"], timeout=30)
    assert len(set(env.directories)) == 2
    assert [content for _, content in env.writes] == [b"canonical", b"canonical"]


@pytest.mark.asyncio
async def test_diagnostic_time_allowance_is_shared_and_enforced():
    c = fixture_cases()
    inv = Investigation(c, "1.2", RecordingSandbox(), Limits(diagnostic_seconds=1))
    inv.seconds = 0.5
    result = json.loads(await inv.python("print(1)"))
    assert result == {"error": "diagnostic_time_exhausted"}
    assert inv.experiments == 0


@pytest.mark.asyncio
async def test_experiment_output_is_bounded_but_retained_in_trace():
    c = fixture_cases()
    sandbox = RecordingSandbox(stdout="a" * 20000, stderr="b" * 19990 + "Traceback")
    inv = Investigation(c, "1.2", sandbox, Limits())
    result = json.loads(await inv.python("print('x')"))
    assert result["reference"] == "X:1"
    assert len(result["stdout"]) < 6000 and result["stdout"].startswith("a")
    assert result["stderr"].endswith("Traceback")
    assert len(inv.trace[0]["result"]["stdout"]) == 20000
    # The python tool loads cumulative submitted code and the current step's targets.
    program = sandbox.commands[0][2]
    assert "process_hdf5_to_tuple('1.2', 1)" in program
    assert "def g(): return f()" in program


@pytest.mark.asyncio
async def test_output_limit_is_reported_as_experiment_result():
    from inspect_ai.util import OutputLimitExceededError

    class NoisySandbox:
        async def exec(self, *args, **kwargs):
            raise OutputLimitExceededError("10 MiB", "partial")

    inv = Investigation(fixture_cases(), "1.2", NoisySandbox(), Limits())
    result = json.loads(await inv.python("print('x' * 10**9)"))
    assert result["output_limit_exceeded"] == "10 MiB"
    assert result["stdout"] == "partial"


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [False, True])
async def test_sandbox_scanner_offers_experiments_and_accepts_x_references(
    monkeypatch, verified
):
    import importlib
    from contextlib import asynccontextmanager

    scanner_module = importlib.import_module(
        "scicode_scanners.failure_classification.scanner"
    )
    c = fixture_cases(verified)
    sandbox = RecordingSandbox(passed=lambda cmd: False)
    provisioned = []

    @asynccontextmanager
    async def fake_sandbox(cases, shard, image, runtime_class):
        provisioned.append((shard, image, runtime_class))
        yield sandbox

    first = assessment()
    first.causes[0].origin_type = "current"
    first.causes[0].dependency_path = ["1.1"]
    first.causes[0].evidence = ["X:1", "T:1.1:1"]
    model = MockModel(
        [
            output(
                ToolCall(
                    id="run",
                    function="rerun",
                    arguments={"step_id": "1.1"},
                ),
                300,
            ),
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

    monkeypatch.setattr(scanner_module, "load_cases", cases)
    monkeypatch.setattr(scanner_module, "target_file", targets)
    monkeypatch.setattr(scanner_module, "decode_targets", lambda *args: [1])
    monkeypatch.setattr(scanner_module, "shard_targets", lambda *args: b"shard")
    monkeypatch.setattr(scanner_module, "investigation_sandbox", fake_sandbox)
    monkeypatch.setattr(scanner_module, "get_model", lambda: model)
    monkeypatch.setattr(scanner_module, "hawk_results_uri", lambda: None)
    model.name = "mock-judge"
    results = await scanner_module.failure_classification(c.implementation)(
        c.transcript
    )
    assert [r.label for r in results] == ["1.1", "1.2"]
    # One fresh sandbox per failed step, using the pinned image for the implementation.
    assert len(provisioned) == 2
    assert provisioned[0][1] == scanner_module.IMAGES[c.implementation]
    assert sandbox.commands
    for result in results:
        assert result.metadata["investigation_mode"] == "sandbox"
        assert result.metadata["source"]["sandbox_image"] == provisioned[0][1]
    trace = results[0].metadata["investigation"]["tools"]
    assert trace[0]["reference"] == "X:1"
    history, tools, _ = model.requests[0]
    assert {"python", "rerun"} <= {t.name for t in tools}
    assert "X:<experiment>" in history[0].text
    assert "Never claim to have executed code" not in history[0].text
    update = json.loads(history[-1].text)["budget_update"]
    assert update["investigation_mode"] == "sandbox"
    assert update["remaining_python_seconds"] == 120


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [False, True])
async def test_smoke_only_checks_each_interpreter_without_inference(
    monkeypatch, verified
):
    import importlib
    from contextlib import asynccontextmanager

    scanner_module = importlib.import_module(
        "scicode_scanners.failure_classification.scanner"
    )
    c = fixture_cases(verified)
    sandbox = RecordingSandbox(stdout="3.11\n")

    @asynccontextmanager
    async def fake_sandbox(*args):
        yield sandbox

    async def cases(*args):
        return c

    async def targets(*args):
        return "fixture.h5"

    def no_model():
        raise AssertionError("smoke_only must not call the judge")

    monkeypatch.setattr(scanner_module, "load_cases", cases)
    monkeypatch.setattr(scanner_module, "target_file", targets)
    monkeypatch.setattr(scanner_module, "decode_targets", lambda *args: [1])
    monkeypatch.setattr(scanner_module, "shard_targets", lambda *args: b"shard")
    monkeypatch.setattr(scanner_module, "investigation_sandbox", fake_sandbox)
    monkeypatch.setattr(
        scanner_module, "get_model", lambda: SimpleNamespace(name="unused")
    )
    monkeypatch.setattr(scanner_module, "investigate", no_model)
    monkeypatch.setattr(scanner_module, "hawk_results_uri", lambda: None)
    [result] = await scanner_module.failure_classification(
        c.implementation, smoke_only=True
    )(c.transcript)
    assert result.answer == "smoke_passed"
    assert set(result.metadata["checks"]) == (
        {"2024", "2025"} if verified else {"default"}
    )


def test_smoke_only_requires_execution():
    from scicode_scanners.failure_classification.scanner import (
        failure_classification,
    )

    with pytest.raises(ValueError, match="requires execution"):
        failure_classification("scicode", execution=False, smoke_only=True)


def test_concurrent_target_downloads_do_not_share_partial_file(tmp_path, monkeypatch):
    import hashlib
    import os
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    from scicode_scanners.failure_classification import assets

    content = b"targets"
    sha = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(
        assets.TARGETS,
        "scicode_verified",
        {**assets.TARGETS["scicode_verified"], "sha256": sha},
    )
    downloads = []
    start = threading.Barrier(2)

    def fixed_partial_download(url, expected, dest):
        # Mirrors inspect_ai.util.download: shared `.partial` path, then os.replace.
        if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            return dest
        downloads.append(url)
        tmp = dest.with_suffix(dest.suffix + ".partial")
        tmp.write_bytes(content)
        time.sleep(0.2)
        os.replace(tmp, dest)
        return dest

    monkeypatch.setattr(assets, "download", fixed_partial_download)

    def worker():
        start.wait()
        return assets.fetch_targets("scicode_verified", tmp_path)

    with ThreadPoolExecutor(2) as pool:
        paths = [f.result() for f in [pool.submit(worker) for _ in range(2)]]
    assert paths[0] == paths[1]
    assert paths[0].read_bytes() == content
    assert len(downloads) == 1


def test_references_followed_by_punctuation_are_accepted():
    c = fixture_cases()
    cause = assessment().causes[0]
    cause.evidence = [
        "X:1: stdout shows f() == 0",
        "M:m-1.1: returns zero;",
        "see T:1.2:1, which expects one",
    ]
    c.validate_cause(cause, "1.2", ["X:1"])


def test_validation_reports_every_problem_in_one_rejection():
    c = fixture_cases()
    cause = assessment().causes[0]
    cause.origin_type = "external"
    cause.dependency_path = ["1.1 f (passed, but returns zero)", "1.2"]
    cause.evidence = ["T:1.2", "X:9"]
    with pytest.raises(ValueError) as ex:
        c.validate_cause(cause, "1.2", ["X:1"])
    message = str(ex.value)
    assert "external/unknown requires an empty origin_steps" in message
    assert "dependency_path must contain only" in message
    assert "1.1 f (passed, but returns zero)" in message
    assert "'T:1.2'" in message and "'X:9'" in message


@pytest.mark.asyncio
async def test_rejected_answer_is_stubbed_in_history_but_kept_in_calls():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
    bad = assessment()
    bad.causes.append(bad.causes[0].model_copy())
    bad.causes[0].evidence = ["M:unknown-message"]
    bad.causes[1].dependency_path = ["1.1 description"]
    model = MockModel(
        [
            output(
                ToolCall(
                    id="bad",
                    function="answer",
                    arguments={"assessment": bad.model_dump()},
                ),
                900,
            ),
            output(
                ToolCall(
                    id="good",
                    function="answer",
                    arguments={"assessment": assessment().model_dump()},
                ),
                900,
            ),
        ]
    )
    result, provenance = await investigate(model, inv, c.packet("1.2"), Limits())
    assert result.status == "resolved"
    retry_history = model.requests[1][0]
    rejected = [m for m in retry_history if getattr(m, "tool_calls", None)][
        -1
    ].tool_calls[0]
    assert rejected.arguments == {
        "assessment": "[rejected submission omitted; see error]"
    }
    error = retry_history[-2].text
    assert "causes[0]" in error and "causes[1]" in error
    assert "f returns zero" in json.dumps(provenance["calls"][0]["message"])


@pytest.mark.asyncio
async def test_replaced_sandbox_pod_is_an_experiment_result_not_a_scan_error(
    monkeypatch,
):
    from scicode_scanners.failure_classification import tools

    class PodReplaced(Exception):
        pass

    class FlakySandbox(RecordingSandbox):
        async def exec(self, cmd, **kwargs):
            if not self.commands:
                self.commands.append(cmd)
                raise PodReplaced("Pod 'agent-env-x-default-0' has been replaced")
            return await super().exec(cmd, **kwargs)

    monkeypatch.setattr(tools, "sandbox_errors", lambda: (PodReplaced,))
    inv = Investigation(fixture_cases(), "1.2", FlakySandbox(), Limits())
    lost = json.loads(await inv.python("print('heavy')"))
    assert lost["reference"] == "X:1"
    assert lost["passed"] is None
    assert "has been replaced" in lost["sandbox_failure"]
    assert "not evidence about the submission" in lost["note"]
    after = json.loads(await inv.python("print('light')"))
    assert after["reference"] == "X:2" and after["passed"] is True


def test_sandbox_requests_its_full_memory_limit():
    from scicode_scanners.failure_classification.sandbox import helm_values

    resources = helm_values("scicode_verified")["services"]["default"]["resources"]
    assert resources["requests"]["memory"] == resources["limits"]["memory"]


def test_packet_indexes_earlier_steps_without_their_code():
    c = fixture_cases()
    packet = c.packet("1.2")
    [entry] = packet["earlier_steps"]
    assert entry["step"] == "1.1"
    assert entry["defines"] == ["f"]
    assert entry["used_by_current"] == ["f"]
    assert entry["code_status"] == "ok"
    assert "submitted_code" not in entry
    assert "return 0" not in json.dumps(packet["earlier_steps"])
    assert packet["current_solution"]["code_status"] == "ok"


def test_empty_and_unparsable_code_are_flagged():
    from scicode_scanners.failure_classification.adapters import code_outline

    assert code_outline("")[0] == "empty"
    assert code_outline("def f(:\n  pass")[0] == "does_not_parse"
    c = fixture_cases()
    c.steps["1.1"]["code"] = "def f(:"
    packet = c.packet("1.2")
    assert packet["earlier_steps"][0]["code_status"] == "does_not_parse"


@pytest.mark.asyncio
async def test_code_retrieval_by_step_and_by_name():
    c = fixture_cases()
    inv = Investigation(c, "1.1")
    by_step = json.loads(await inv.retrieve_step("1.1", section="code"))
    assert by_step["submitted_code"].strip() == "def f(): return 0"
    assert by_step["code_status"] == "ok"
    by_name = json.loads(await inv.read_code("f"))
    assert by_name == {"step": "1.1", "name": "f", "source": "def f(): return 0"}
    # Later steps are not part of this investigation's evidence.
    with pytest.raises(ValueError, match="No parsed definition"):
        await inv.read_code("g")
    later = Investigation(c, "1.2")
    assert json.loads(await later.read_code("g"))["step"] == "1.2"


@pytest.mark.asyncio
async def test_slim_packet_is_sent_whole_in_first_message():
    c = fixture_cases()
    inv = Investigation(c, "1.2")
    packet = {**c.packet("1.2"), "filler": "a" * 30000}
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
    first = json.loads(model.requests[0][0][1].text)
    assert "next_char_offset" not in first
    assert first["case"]["filler"] == "a" * 30000


def test_packet_records_stop_reasons_and_issues():
    c = fixture_cases()
    c.steps["1.1"]["stop_reason"] = "max_tokens"
    limit = SimpleNamespace(
        event="sample_limit",
        uuid="limit",
        model_dump=lambda **kwargs: {
            "event": "sample_limit",
            "uuid": "limit",
            "timestamp": "t",
            "type": "time",
            "limit": 60,
        },
    )
    retry = SimpleNamespace(
        event="model",
        uuid="retry",
        error="provider timeout",
        output=SimpleNamespace(choices=[]),
    )
    c.events = [*c.events, limit, retry]
    packet = c.packet("1.2")
    assert packet["current_solution"]["stop_reason"] == "stop"
    assert packet["earlier_steps"][0]["stop_reason"] == "max_tokens"
    assert packet["recorded_issues"] == [
        {"reference": "E:limit", "event": "sample_limit", "type": "time", "limit": 60},
        {
            "reference": "E:retry",
            "event": "model",
            "stop_reason": None,
            "error": "provider timeout",
        },
    ]


def test_packet_omits_normal_stops_and_has_no_issues_by_default():
    packet = fixture_cases().packet("1.2")
    assert "stop_reason" not in packet["earlier_steps"][0]
    assert packet["recorded_issues"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize("verified", [False, True])
async def test_python_can_load_any_steps_targets(verified):
    sandbox = RecordingSandbox()
    inv = Investigation(fixture_cases(verified), "1.2", sandbox, Limits())
    await inv.python("pass")
    program = sandbox.commands[0][2]
    # Replace the harness decoder import with a stub that records its arguments.
    preamble = program.split("\n", 1)[1]
    namespace = {"process_hdf5_to_tuple": lambda *args: args}
    exec(preamble, namespace)
    extra = ("test_data_cleaned.h5",) if verified else ()
    assert namespace["targets"] == ("1.2", 1, *extra)
    assert namespace["load_targets"]("1.1") == ("1.1", 1, *extra)
