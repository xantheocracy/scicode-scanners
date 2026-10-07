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
from scicode_scanners.failure_classification.sandbox import FreshSandbox
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
    assert "NEXT STEP" in packet
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

    async def generate(self, history, tools, config):
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
    inv = Investigation(c, "1.2", None, None, Limits())
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
        update["remaining_generated_tokens"] == 8192
        and update["finalization_reserve"] == 1536
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
    inv = Investigation(c, "1.2", None, None, limits)
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
    assert [d.name for d in tools] == ["answer"] and config.max_tokens == 1024


@pytest.mark.asyncio
async def test_checkpoint_reserves_charge_before_request_and_reuses_answer(tmp_path):
    store = CheckpointStore(str(tmp_path), {"case": "1.2"})
    c = fixture_cases()
    inv = Investigation(c, "1.2", None, None, Limits())

    class CheckedModel(MockModel):
        async def generate(self, history, tools, config):
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
    inv = Investigation(c, "1.2", None, None, Limits())

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
async def test_python_timeout_is_not_recorded_as_original_failure():
    class TimeoutSandbox:
        async def exec(self, *args, **kwargs):
            assert kwargs["timeout_retry"] is False
            raise TimeoutError

    c = fixture_cases(True)
    inv = Investigation(c, "1.2", None, TimeoutSandbox(), Limits())
    result = json.loads(await inv.rerun("1.2", environment="2024"))
    assert result["per_environment"]["2024"]["timed_out"]
    assert "diagnostic" in result["note"]
    assert c.scores == {"1.1": 0, "1.2": 0}


@pytest.mark.asyncio
async def test_empty_provider_responses_cannot_create_unbounded_input_spend():
    c = fixture_cases()
    limits = Limits()
    inv = Investigation(c, "1.2", None, None, limits)
    empty = ModelOutput.from_message(ChatMessageAssistant(content=""))
    empty.usage = ModelUsage(input_tokens=100, output_tokens=0, total_tokens=100)
    model = MockModel([empty] * (limits.tool_rounds + 4))
    a, provenance = await investigate(model, inv, c.packet("1.2"), limits)
    assert a.status == "unresolved" and provenance["termination"] == "model_call_limit"
    assert len(model.requests) == 10


@pytest.mark.asyncio
async def test_verified_accepts_second_environment_pass():
    class DualSandbox:
        async def exec(self, cmd, **kwargs):
            passed = "2025" in cmd[0]
            return SimpleNamespace(
                success=passed, returncode=0 if passed else 1, stdout="", stderr=""
            )

    c = fixture_cases(True)
    inv = Investigation(c, "1.2", None, DualSandbox(), Limits())
    result = json.loads(await inv.rerun("1.2"))
    assert result["passed"]
    assert not result["per_environment"]["2024"]["passed"]
    assert result["per_environment"]["2025"]["passed"]


@pytest.mark.asyncio
async def test_diagnostics_restore_canonical_artifacts_in_fresh_directories():
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
