"""No inference spend: scripted agent turns through real Inspect/Docker services.

Run with RUN_DEFECT_DOCKER_TESTS=1 python -m pytest tests/test_defect_investigation_docker.py.
"""

import json
import os
from types import SimpleNamespace

import pytest
from inspect_ai import Task, eval_async
from inspect_ai.model import ChatMessageAssistant, ModelOutput, ModelUsage
from inspect_ai.tool import ToolCall
from test_defect_investigation import finding, problem, targets

from scicode_scanners.defect_investigation import agent, dataset
from scicode_scanners.defect_investigation.dataset import file_hash, get_dataset
from scicode_scanners.defect_investigation.export import export
from scicode_scanners.defect_investigation.harness import ROOT, TARGET_NAMES
from scicode_scanners.defect_investigation.schema import Limits
from scicode_scanners.defect_investigation.scorer import validate_findings

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_DEFECT_DOCKER_TESTS") != "1", reason="Opt-in Docker integration"
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "implementation,direction",
    [
        ("scicode", "false_acceptance"),
        ("scicode_verified", "false_acceptance"),
        ("scicode", "false_rejection"),
    ],
)
async def test_scripted_eval_replay_and_workspace_isolation(
    tmp_path, monkeypatch, implementation, direction
):
    source = targets(
        tmp_path / "source.h5", first=5 if direction == "false_rejection" else 4
    )
    monkeypatch.setattr(dataset, "records", lambda *_: [problem()])
    monkeypatch.setitem(dataset.TARGET_HASHES, implementation, file_hash(source))
    samples = get_dataset(
        implementation,
        ["1"],
        ["1.1"],
        targets_path=str(source),
        cache_dir=str(tmp_path / "cache"),
    )
    codes = {
        "1.1": "def f(x):\n    return 2*x"
        if direction == "false_rejection"
        else "def f(x):\n    return x*x"
    }
    semantic = (
        "assert f(3) == 6; print('f(3)=6 as required')"
        if direction == "false_rejection"
        else "assert f(3) != 6; print('x=3; actual=', f(3), '; expected=6 from twice x')"
    )
    report = {
        "implementation": implementation,
        "problem_id": "1",
        "summary": "Synthetic witness, replayed.",
        "findings": [
            finding(direction),
            finding(
                "false_rejection"
                if direction == "false_acceptance"
                else "false_acceptance",
                "not_found",
            ),
        ],
    }
    report["findings"][0].update(candidate_evidence="X:2", semantic_evidence=["X:3"])
    if direction == "false_rejection":
        report["findings"][0].update(
            claim="Target requires five for twice two.",
            semantic_argument="Twice two is four; candidate implements 2*x exactly.",
        )
    python = "python" if implementation == "scicode" else "/opt/scicode-2025/bin/python"
    poison = f"{python} -c \"import h5py; f=h5py.File('{TARGET_NAMES[implementation]}', 'a'); f['1.1/test1/var1'][...]=123; f.close()\""
    calls = [
        ("workspace", {"command": poison}),
        ("grade_candidate", {"step_id": "1.1", "codes": codes}),
        ("semantic_check", {"step_id": "1.1", "codes": codes, "check": semantic}),
        ("submit_report", {"report": report}),
    ]
    turn = 0

    async def generate(messages, tools, config):
        nonlocal turn
        name, arguments = calls[turn]
        turn += 1
        return ModelOutput(
            model="mock",
            choices=[
                {
                    "message": ChatMessageAssistant(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=f"call-{turn}", function=name, arguments=arguments
                            )
                        ],
                    ),
                    "stop_reason": "tool_calls",
                }
            ],
            usage=ModelUsage(input_tokens=10, output_tokens=200, total_tokens=210),
        )

    monkeypatch.setattr(agent, "get_model", lambda: SimpleNamespace(generate=generate))
    limits = Limits(
        generated_tokens=8000, finalization_reserve=1000, grading_timeout=30
    )
    task = Task(
        dataset=samples,
        solver=agent.investigate(limits),
        scorer=validate_findings(limits),
        sandbox=("docker", str(ROOT / "sandbox" / f"{implementation}.compose.yaml")),
        name=f"defect-fixture-{implementation}-{direction}",
    )
    logs = await eval_async(
        task, model="mockllm/model", log_dir=str(tmp_path / "logs"), max_samples=1
    )
    log = logs[0]
    assert log.status == "success", log.error
    sample = log.samples[0]
    assert sample.store["defect_termination"] == "submitted"
    assert sample.store["defect_evidence"][0]["passed"]  # poisoning workspace succeeded
    assert sample.scores["validate_findings"].value["report_valid"] == 1
    assert sample.scores["validate_findings"].value["replay_confirmed"] == 1
    validation = sample.store["defect_validation"][0]
    assert validation["semantic_review"] == "unreviewed"
    assert validation["replay_confirmed"]
    if implementation == "scicode_verified":
        assert set(sample.store["defect_evidence"][1]["per_environment"]) == {
            "2024",
            "2025",
        }
    export(log.location, str(tmp_path / "export"))
    rows = [
        json.loads(line)
        for line in (tmp_path / "export/reports.jsonl").read_text().splitlines()
    ]
    assert len(rows[0]["evidence"]) == 4
    assert "def f(x)" in rows[0]["evidence"][1]["program"]
    assert (tmp_path / "export/reports.md").is_file()
