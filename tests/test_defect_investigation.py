import ast
import json
from types import SimpleNamespace

import h5py
import pytest
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageUser,
    ModelOutput,
    ModelUsage,
)
from inspect_ai.tool import ToolCall
from inspect_ai.util import Store

from scicode_scanners.defect_investigation import agent, dataset
from scicode_scanners.defect_investigation.dataset import (
    decode_shard,
    file_hash,
    get_dataset,
    records,
    shard_problem,
)
from scicode_scanners.defect_investigation.harness import (
    PROVIDED,
    VENDOR,
    admissibility_issues,
    cumulative_codes,
    grading_script,
    interpreters,
)
from scicode_scanners.defect_investigation.prompts import evidence_packet
from scicode_scanners.defect_investigation.schema import Limits, Report
from scicode_scanners.defect_investigation.tools import validate_report


def problem():
    return {
        "problem_id": "1",
        "required_dependencies": "import numpy as np",
        "sub_steps": [
            {
                "step_number": "1.1",
                "step_description_prompt": "Return twice x.",
                "function_header": "def f(x):",
                "return_line": "return y",
                "step_background": "Double means multiply by two.",
                "test_cases": ["assert f(2) == target"],
            },
            {
                "step_number": "1.2",
                "step_description_prompt": "Call f then add one.",
                "function_header": "def g(x):",
                "return_line": "return y",
                "step_background": "Use the prior function.",
                "test_cases": ["assert g(2) == target"],
            },
        ],
    }


def targets(path, first=4):
    with h5py.File(path, "w") as f:
        f.create_dataset("1.1/test1/var1", data=first)
        f.create_dataset("1.2/test1/var1", data=5)
        f.create_dataset("99.1/test1/var1", data=999)
    return path


def metadata(impl="scicode"):
    return {"implementation": impl, "problem": problem(), "selected_steps": ["1.1"]}


def finding(direction="false_acceptance", status="demonstrated"):
    return {
        "step_id": "1.1",
        "direction": direction,
        "status": status,
        "claim": "A square implementation passes the only doubling test.",
        "defect_key": "doubling-coverage",
        "category": "missing_input_coverage",
        "origin_steps": ["1.1"],
        "prompt_test_references": ["P:1.1", "T:1.1:1"],
        "candidate_evidence": "X:1" if status == "demonstrated" else None,
        "semantic_evidence": ["X:2"] if status == "demonstrated" else [],
        "semantic_argument": "At x=3, the specification requires 6 but the candidate returns 9.",
        "plausibility_rationale": "A model could confuse squaring with doubling.",
        "semantic_confidence": {
            "rating": "high",
            "rationale": "Exact integer counterexample.",
        },
        "grader_confidence": {
            "rating": "high",
            "rationale": "Canonical test replay passed.",
        },
    }


def report():
    return Report(
        implementation="scicode",
        problem_id="1",
        summary="Coverage gap.",
        findings=[finding(), finding("false_rejection", "not_found")],
    )


def evidence():
    codes = {"1.1": "def f(x):\n    return x*x"}
    return [
        {
            "id": "X:1",
            "kind": "canonical_grade",
            "step_id": "1.1",
            "codes": codes,
            "passed": True,
            "admissibility_issues": [],
            "per_environment": {"default": {"passed": True}},
        },
        {
            "id": "X:2",
            "kind": "semantic_check",
            "step_id": "1.1",
            "codes": codes,
            "passed": True,
        },
    ]


@pytest.mark.parametrize("impl,count", [("scicode", 65), ("scicode_verified", 64)])
def test_pinned_dataset_and_complete_packets(impl, count):
    ps = records(impl)
    assert len(ps) == count
    raw_records = {
        str(p["problem_id"]): p
        for p in [
            json.loads(line)
            for line in (dataset.ROOT / "data" / f"{impl}.jsonl")
            .read_text()
            .splitlines()
        ]
    }
    for p in ps:
        packet = json.loads(evidence_packet(p, impl, [], True))
        assert len(packet["tests"]) == sum(len(s["test_cases"]) for s in p["sub_steps"])
        serialized = json.dumps(p)
        assert (
            "ground_truth_code" not in serialized
            and "general_solution" not in serialized
        )
        for key in (
            "problem_description_main",
            "problem_background_main",
            "problem_io",
        ):
            assert p[key] == raw_records[p["problem_id"]][key]


@pytest.mark.parametrize("implementation", ["scicode", "scicode_verified"])
def test_canonical_harness_snapshot_parity(implementation):
    """Compare every step to the pinned source, including supplied prerequisites."""
    filename = (
        "original_scorer.py.txt"
        if implementation == "scicode"
        else "verified_task.py.txt"
    )
    tree = ast.parse((VENDOR / filename).read_text())
    name = "compose_code" if implementation == "scicode" else "grading_script"
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"Any": object}
    exec(  # noqa: S102 — execute only a checksum-pinned, trusted harness function.
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            filename,
            "exec",
        ),
        namespace,
    )
    for p in records(implementation):
        codes = {
            s["step_number"]: f"# submission for {s['step_number']}\npass"
            for s in p["sub_steps"]
            if s["step_number"] not in PROVIDED
        }
        for index, step in enumerate(p["sub_steps"]):
            sid = step["step_number"]
            if sid in PROVIDED:
                continue
            selected = {
                key: value
                for key, value in codes.items()
                if int(key.split(".")[1]) <= index + 1
            }
            cumulative = cumulative_codes(p, selected, sid, implementation)
            if implementation == "scicode":
                namespace.update(
                    subproblem=step,
                    state=SimpleNamespace(metadata=p),
                    get_solution_code=lambda state, cumulative=cumulative: cumulative,
                    subproblem_str_to_int=lambda value: int(value.split(".")[1]),
                )
                expected = namespace[name]()
            else:
                expected = namespace[name](p, index, cumulative)
            assert grading_script(p, selected, sid, implementation) == expected


@pytest.mark.parametrize("impl", ["scicode", "scicode_verified"])
def test_lossless_shard_and_missing_groups(tmp_path, impl):
    source = targets(tmp_path / "source.h5")
    shard = tmp_path / "shard.h5"
    digest = shard_problem(source, problem(), shard, impl)
    assert digest == file_hash(shard)
    with h5py.File(shard) as f:
        assert set(f) == {"1.1", "1.2"}
    assert decode_shard(shard, impl, "1.1", 1)[0] == 4
    with h5py.File(source, "a") as f:
        del f["1.2"]
    with pytest.raises(ValueError, match="Missing targets"):
        shard_problem(source, problem(), shard, impl)
    assert file_hash(shard) == digest  # failed rebuild cannot replace a valid shard


@pytest.mark.parametrize("impl", ["scicode", "scicode_verified"])
def test_dataset_selection_and_files(tmp_path, monkeypatch, impl):
    source = targets(tmp_path / "source.h5")
    monkeypatch.setattr(dataset, "records", lambda *_: [problem()])
    monkeypatch.setitem(dataset.TARGET_HASHES, impl, file_hash(source))
    samples = get_dataset(
        impl,
        ["1"],
        ["1.2"],
        targets_path=str(source),
        cache_dir=str(tmp_path / "cache"),
    )
    sample = samples[0]
    assert sample.id == f"{impl}:1"
    assert sample.metadata["selected_steps"] == ["1.2"]
    assert "tests/1.1_1.py" in sample.files and "target_access.py" in sample.files
    assert json.loads(sample.input)["target_manifest"] == {"1.1": 1, "1.2": 1}
    with pytest.raises(ValueError, match="investigate_steps"):
        get_dataset(
            impl, ["1"], ["1.3"], targets_path=str(source), cache_dir=str(tmp_path)
        )
    with pytest.raises(ValueError, match="max_prompt_chars"):
        get_dataset(
            impl,
            ["1"],
            targets_path=str(source),
            cache_dir=str(tmp_path),
            max_prompt_chars=1,
        )


def test_cumulative_assembly_and_environment_semantics():
    codes = {"1.1": "def f(x):\n return 2*x", "1.2": "def g(x):\n return f(x)+1"}
    for impl in ("scicode", "scicode_verified"):
        script = grading_script(problem(), codes, "1.2", impl)
        assert script.index("def f") < script.index("def g") < script.index("target =")
        assert "assert g(2)" in script and "assert f(2)" not in script
        with pytest.raises(ValueError, match="prerequisite"):
            grading_script(problem(), {"1.2": codes["1.2"]}, "1.2", impl)
    assert set(interpreters("scicode_verified")) == {"2024", "2025"}
    assert set(interpreters("scicode_verified", "2024")) == {"2024"}


@pytest.mark.parametrize(
    "source",
    [
        "def f(x): return targets[0]",
        "def f(x): return open('test_data.h5').read()",
        "def f(x): exit(0)",
        "cmp_tuple_or_list = lambda a,b: True",
    ],
)
def test_cheating_screen(source):
    assert admissibility_issues({"1.1": source}, "scicode")


def test_target_named_parameters_are_not_privileged_access():
    assert not admissibility_issues(
        {"1.1": "def f(targets):\n target = targets[0]*2\n return target"}, "scicode"
    )


@pytest.mark.parametrize("run", ["smoke", "pilot"])
def test_hawk_configs_inherit_deployment_inspect_pin(run):
    path = dataset.ROOT / f"hawk-{run}.yaml"
    config = json.loads(
        "\n".join(
            line for line in path.read_text().splitlines() if not line.startswith("#")
        )
    )
    assert not any(
        spec.startswith(("inspect-ai", "inspect_ai")) for spec in config["packages"]
    )
    assert {item["args"]["implementation"] for item in config["tasks"][0]["items"]} == {
        "scicode",
        "scicode_verified",
    }


def test_report_evidence_validation():
    validate_report(report(), metadata(), evidence())
    bad = evidence()
    bad[1]["codes"] = {"1.1": "def f(x): return 2*x"}
    with pytest.raises(ValueError, match="exactly"):
        validate_report(report(), metadata(), bad)
    bad = evidence()
    bad[0]["passed"] = False
    with pytest.raises(ValueError, match="behavior"):
        validate_report(report(), metadata(), bad)
    bad = evidence()
    bad[0]["per_environment"]["default"]["timed_out"] = True
    with pytest.raises(ValueError, match="Timeout"):
        validate_report(report(), metadata(), bad)
    bad_report = report().model_copy(deep=True)
    bad_report.findings.pop()
    with pytest.raises(ValueError, match="exactly once"):
        validate_report(bad_report, metadata(), evidence())


@pytest.mark.asyncio
async def test_agent_generated_budget_and_partial_evidence(monkeypatch):
    caps = []

    async def generate(messages, tools, config):
        caps.append(config.max_tokens)
        return ModelOutput(
            model="mock",
            choices=[
                {
                    "message": ChatMessageAssistant(content="thinking"),
                    "stop_reason": "max_tokens",
                }
            ],
            usage=ModelUsage(
                input_tokens=1000,
                output_tokens=config.max_tokens,
                total_tokens=1000 + config.max_tokens,
            ),
        )

    monkeypatch.setattr(agent, "get_model", lambda: SimpleNamespace(generate=generate))
    state = SimpleNamespace(
        metadata=metadata(),
        store=Store(),
        messages=[ChatMessageUser(content="Investigate")],
    )
    limits = Limits(
        generated_tokens=2000,
        finalization_reserve=500,
        per_call_tokens=800,
        model_calls=10,
    )
    await agent.investigate(limits)(state, None)
    assert sum(caps) == 2000
    assert caps[-1] == 500
    assert state.store.get("defect_report") is None
    assert state.store.get("defect_termination") == "budget_exhausted"


@pytest.mark.asyncio
async def test_agent_structured_submission(monkeypatch):
    r = report().model_dump()
    r["findings"] = [
        finding("false_acceptance", "not_found"),
        finding("false_rejection", "inconclusive"),
    ]

    async def generate(messages, tools, config):
        return ModelOutput(
            model="mock",
            choices=[
                {
                    "message": ChatMessageAssistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="answer",
                                function="submit_report",
                                arguments={"report": r},
                            )
                        ],
                    ),
                    "stop_reason": "tool_calls",
                }
            ],
            usage=ModelUsage(input_tokens=10, output_tokens=200, total_tokens=210),
        )

    monkeypatch.setattr(agent, "get_model", lambda: SimpleNamespace(generate=generate))
    state = SimpleNamespace(
        metadata=metadata(),
        store=Store(),
        messages=[ChatMessageUser(content="Investigate")],
    )
    await agent.investigate(Limits())(state, None)
    assert state.store.get("defect_report")["problem_id"] == "1"
    assert state.store.get("defect_termination") == "submitted"


def test_k8s_uses_packaged_chart():
    from pathlib import Path

    pytest.importorskip("k8s_sandbox")
    from scicode_scanners.defect_investigation.harness import ROOT
    from scicode_scanners.defect_investigation.task import _sandbox_spec

    values = str(ROOT / "sandbox" / "scicode.yaml")
    kind, config = _sandbox_spec("k8s", values)
    assert kind == "k8s"
    assert Path(config.chart) == ROOT / "sandbox" / "chart"
    assert config.values == Path(values)
    assert config.restarted_container_behavior == "raise"
    assert _sandbox_spec("docker", "compose.yaml") == ("docker", "compose.yaml")
