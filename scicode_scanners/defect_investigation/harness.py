"""Pinned cumulative harness assembly; candidate code never runs on the host."""

import ast
import re
from pathlib import Path
from typing import Any

from .schema import Implementation

ROOT = Path(__file__).parent
VENDOR = ROOT / "vendor"
PROVIDED = {"13.6", "62.1", "76.3"}
TARGET_NAMES = {"scicode": "test_data.h5", "scicode_verified": "test_data_cleaned.h5"}
REVISIONS = {
    "scicode": "e02028192f8bbf3677c6a672eaf48cf20e7a6d53",
    "scicode_verified": "3fd9ef2713244d58f6cfeab5a5d92c10e2b907de",
}
THREAD_ENV = {
    key: "1"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


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


def supplied_code(step_id: str, implementation: Implementation) -> str:
    return extract_code((VENDOR / f"{step_id}.txt").read_text(), implementation)


def cumulative_codes(
    problem: dict[str, Any],
    codes: dict[str, str],
    step_id: str,
    implementation: Implementation,
) -> list[str]:
    order = [str(s["step_number"]) for s in problem["sub_steps"]]
    if step_id not in order or step_id in PROVIDED:
        raise ValueError("Select an existing, scored step.")
    required = order[: order.index(step_id) + 1]
    if set(codes) - set(required):
        raise ValueError(
            "Candidate code must name only the current and preceding steps."
        )
    if set(codes) & PROVIDED:
        raise ValueError(
            "Author-provided steps are inserted automatically; do not replace them."
        )
    missing = [sid for sid in required if sid not in PROVIDED and sid not in codes]
    if missing:
        raise ValueError(f"Supply cumulative prerequisite code: {missing}")
    return [
        supplied_code(sid, implementation)
        if sid in PROVIDED
        else extract_code(codes[sid], implementation)
        for sid in required
    ]


def grading_script(
    problem: dict[str, Any],
    codes: dict[str, str],
    step_id: str,
    implementation: Implementation,
) -> str:
    sections = [problem["required_dependencies"]]
    if implementation == "scicode":
        sections.append("from test_util import are_dicts_close, cmp_tuple_or_list")
    sections.extend(cumulative_codes(problem, codes, step_id, implementation))
    if implementation == "scicode":
        sections.append("# Test cases:")
    step = next(s for s in problem["sub_steps"] if str(s["step_number"]) == step_id)
    module = "process_data" if implementation == "scicode" else "scicode.parse.parse"
    sections.extend(
        [
            f"from {module} import process_hdf5_to_tuple",
            f"targets = process_hdf5_to_tuple({step_id!r}, {len(step['test_cases'])})"
            if implementation == "scicode"
            else f"targets = process_hdf5_to_tuple({step_id!r}, {len(step['test_cases'])}, 'test_data_cleaned.h5')",
        ]
    )
    for index, test in enumerate(step["test_cases"]):
        sections.extend([f"target = targets[{index}]", test])
    return ("\n" if implementation == "scicode" else "\n\n").join(sections)


def interpreters(
    implementation: Implementation, environments: str = "both"
) -> dict[str, str]:
    if environments not in {"both", "2024", "2025"}:
        raise ValueError("grading_environments must be both, 2024, or 2025.")
    if implementation == "scicode":
        return {"default": "python"}
    return {
        year: f"/opt/scicode-{year}/bin/python"
        for year in ("2024", "2025")
        if environments in {"both", year}
    }


def helper_files(implementation: Implementation) -> dict[str, bytes]:
    if implementation == "scicode":
        return {
            name: (VENDOR / name).read_bytes()
            for name in ("process_data.py", "test_util.py")
        }
    return {
        str(p.relative_to(VENDOR)): p.read_bytes()
        for p in (VENDOR / "scicode").rglob("*.py")
    }


def admissibility_issues(
    codes: dict[str, str], implementation: Implementation
) -> list[str]:
    """Conservative syntactic screen; plausibility/hardcoding still need semantic review."""
    issues = []
    forbidden_names = {
        "target",
        "targets",
        "load_targets",
        "process_hdf5_to_tuple",
        "open",
        "__import__",
        "eval",
        "exec",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "os",
        "sys",
        "subprocess",
        "builtins",
        "exit",
        "quit",
        "are_dicts_close",
        "cmp_tuple_or_list",
    }
    for sid, source in codes.items():
        try:
            tree = ast.parse(extract_code(source, implementation))
        except SyntaxError as ex:
            issues.append(f"{sid}: invalid Python: {ex.msg}")
            continue
        bound_names = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        } | {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                issues.append(f"{sid}: candidate imports violate the standard prompt")
            # Ordinary scientific parameters/local variables can be called target(s).
            # Unbound access to the grader's injected target globals is inadmissible.
            if (
                isinstance(node, ast.Name)
                and node.id in forbidden_names
                and (node.id not in {"target", "targets"} or node.id not in bound_names)
            ):
                issues.append(f"{sid}: privileged/interference name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                issues.append(f"{sid}: reflective access {node.attr}")
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and (".h5" in node.value or "test_data" in node.value)
            ):
                issues.append(f"{sid}: grader asset reference")
    return sorted(set(issues))
