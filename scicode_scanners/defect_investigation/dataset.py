"""Pinned problem definitions and lossless per-problem HDF5 artifacts."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.util import download

from .harness import PROVIDED, REVISIONS, ROOT, TARGET_NAMES, VENDOR, helper_files
from .prompts import evidence_packet
from .schema import Implementation

PROBLEM_HASHES = {
    "scicode": "38797fef78f434720be6d053b4f3a86839d6f8ea5fb9115450677cd3a6edf81d",
    "scicode_verified": "427771cb8bceb5058e8b510af0ee2c8210827a1491e0cdf8e6db56d4ed1440ba",
    "scicode_dev": "e02e0170641999b32dd108a546e616a84697db3d2aea4a7cc046706574dd7682",
}
TARGET_HASHES = {
    "scicode": "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890",
    "scicode_verified": "8fb6e575b7b6dda5e48b04dea338fc6af4fe185774b8f19221c96945df9b4142",
}
TARGET_URLS = {
    "scicode": (
        "https://huggingface.co/datasets/xantheocracy/scicode-mirror/resolve/"
        "2f903a64a1c4eb62c7c649b0caf7ecb2c0217590/test_data.h5"
    ),
    "scicode_verified": (
        "https://huggingface.co/datasets/shhu2001/SciCode-Verified/resolve/"
        "eea11a866be6860725258702b39ef8651ed26abd/test_data_cleaned.h5"
    ),
}


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_hash(path: Path, expected: str) -> None:
    if file_hash(path) != expected:
        raise ValueError(f"Asset checksum mismatch: {path}")


def verify_bundled_assets() -> dict:
    manifest = json.loads((ROOT / "asset_manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        check_hash(ROOT / relative, digest)
    return manifest


def records(implementation: Implementation, include_dev: bool = False) -> list[dict]:
    if implementation not in TARGET_NAMES:
        raise ValueError("Unknown implementation.")
    if include_dev and implementation == "scicode_verified":
        raise ValueError(
            "The pinned Verified release only supplies the evaluation split."
        )
    result = []
    for key in [implementation] + (["scicode_dev"] if include_dev else []):
        path = ROOT / "data" / f"{key}.jsonl"
        check_hash(path, PROBLEM_HASHES[key])
        for line in path.read_text().splitlines():
            raw = json.loads(line)
            # Whitelist the specification: never expose reference implementations or audit fields.
            record = {
                k: raw[k] for k in ("problem_id", "required_dependencies", "sub_steps")
            }
            for k in (
                "problem_description",
                "problem_description_prompt",
                "problem_background",
                "problem_name",
                "problem_description_main",
                "problem_background_main",
                "problem_io",
                "general_tests",
            ):
                if k in raw:
                    record[k] = raw[k]
            record["problem_id"] = str(record["problem_id"])
            record["sub_steps"] = [
                {
                    k: s[k]
                    for k in (
                        "step_number",
                        "step_description_prompt",
                        "function_header",
                        "return_line",
                        "step_background",
                        "test_cases",
                    )
                    if k in s
                }
                for s in raw["sub_steps"]
            ]
            for step in record["sub_steps"]:
                step["step_number"] = str(step["step_number"])
                if implementation == "scicode":
                    step["test_cases"] = [
                        test.replace(
                            "from scicode.compare.cmp import cmp_tuple_or_list", ""
                        ).replace("from scicode.compare.cmp import are_dicts_close", "")
                        for test in step["test_cases"]
                    ]
            result.append(record)
    ids = [p["problem_id"] for p in result]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate problem IDs.")
    return result


def targets_file(
    implementation: Implementation, override: str | None, cache: Path
) -> Path:
    path = (
        Path(override).expanduser().resolve()
        if override
        else cache / TARGET_NAMES[implementation]
    )
    if not override:
        download(TARGET_URLS[implementation], TARGET_HASHES[implementation], path)
    check_hash(path, TARGET_HASHES[implementation])
    return path


def decode_shard(path: Path, implementation: Implementation, step_id: str, count: int):
    helper = VENDOR / (
        "process_data.py" if implementation == "scicode" else "scicode/parse/parse.py"
    )
    spec = importlib.util.spec_from_file_location("_defect_target_decoder", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if implementation == "scicode":
        module.H5PY_FILE = str(path)
        return module.process_hdf5_to_tuple(step_id, count)
    return module.process_hdf5_to_tuple(step_id, count, str(path))


def shard_problem(
    source: Path, problem: dict, destination: Path, implementation: Implementation
) -> str:
    """Atomically rebuild a shard from validated source; preserve only relevant groups."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=destination.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        with h5py.File(source, "r") as src, h5py.File(temporary, "w") as dst:
            for step in problem["sub_steps"]:
                sid = step["step_number"]
                if sid in PROVIDED:
                    continue
                if sid not in src:
                    raise ValueError(f"Missing targets: {sid}")
                for i in range(len(step["test_cases"])):
                    if f"{sid}/test{i + 1}" not in src:
                        raise ValueError(f"Missing test targets: {sid}/test{i + 1}")
                src.copy(sid, dst)
        for step in problem["sub_steps"]:
            if step["step_number"] not in PROVIDED:
                decoded = decode_shard(
                    temporary,
                    implementation,
                    step["step_number"],
                    len(step["test_cases"]),
                )
                if len(decoded) != len(step["test_cases"]):
                    raise ValueError("Decoded target count does not match tests.")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return file_hash(destination)


def loader_source(implementation: Implementation) -> str:
    module = "process_data" if implementation == "scicode" else "scicode.parse.parse"
    call = (
        "process_hdf5_to_tuple(step_id, len(step['test_cases']))"
        if implementation == "scicode"
        else (
            "process_hdf5_to_tuple(step_id, len(step['test_cases']), 'test_data_cleaned.h5')"
        )
    )
    return (
        f"import json\nfrom {module} import process_hdf5_to_tuple\n"
        "def load_targets(step_id):\n"
        "    with open('problem.json') as f: problem = json.load(f)\n"
        "    step = next(s for s in problem['sub_steps'] if s['step_number'] == step_id)\n"
        f"    return {call}\n"
    )


def get_dataset(
    implementation: Implementation,
    problem_ids: list[str] | None = None,
    investigate_steps: list[str] | None = None,
    include_dev: bool = False,
    provide_scientific_background: bool = True,
    targets_path: str | None = None,
    cache_dir: str | None = None,
    max_prompt_chars: int = 500_000,
) -> MemoryDataset:
    provenance = verify_bundled_assets()
    problems = records(implementation, include_dev)
    if problem_ids is not None:
        missing = set(problem_ids) - {p["problem_id"] for p in problems}
        if missing:
            raise ValueError(f"Unknown problem IDs: {sorted(missing)}")
        problems = [p for p in problems if p["problem_id"] in problem_ids]
    if not problems:
        raise ValueError("No problems selected.")
    known_steps = {
        s["step_number"]
        for p in problems
        for s in p["sub_steps"]
        if s["step_number"] not in PROVIDED
    }
    if investigate_steps is not None:
        if not investigate_steps or set(investigate_steps) - known_steps:
            raise ValueError(
                "investigate_steps must select existing scored steps in selected problems."
            )
        problems = [
            p
            for p in problems
            if any(s["step_number"] in investigate_steps for s in p["sub_steps"])
        ]
    cache = (
        Path(cache_dir).expanduser()
        if cache_dir
        else Path.home() / ".cache/scicode_defect_investigation"
    )
    cache.mkdir(parents=True, exist_ok=True)
    targets = targets_file(implementation, targets_path, cache)
    samples = []
    for problem in problems:
        pid = problem["problem_id"]
        selected = [
            s["step_number"]
            for s in problem["sub_steps"]
            if s["step_number"] not in PROVIDED
            and (investigate_steps is None or s["step_number"] in investigate_steps)
        ]
        directory = cache / implementation / TARGET_HASHES[implementation] / pid
        directory.mkdir(parents=True, exist_ok=True)
        shard = directory / TARGET_NAMES[implementation]
        shard_hash = shard_problem(targets, problem, shard, implementation)
        packet = evidence_packet(
            problem, implementation, selected, provide_scientific_background
        )
        if len(packet) > max_prompt_chars:
            raise ValueError(
                f"Problem {pid} exceeds max_prompt_chars; no prompt was truncated."
            )
        files = {TARGET_NAMES[implementation]: str(shard)}
        contents = {
            "problem.json": json.dumps(problem),
            "prompt.txt": packet,
            "target_access.py": loader_source(implementation),
        }
        contents.update(
            {
                name: source.decode()
                for name, source in helper_files(implementation).items()
            }
        )
        for step in problem["sub_steps"]:
            for i, test in enumerate(step["test_cases"]):
                contents[f"tests/{step['step_number']}_{i + 1}.py"] = test
        for name, content in contents.items():
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            files[name] = str(path)
        metadata = {
            "problem": problem,
            "implementation": implementation,
            "selected_steps": selected,
            "targets_path": str(shard),
            "targets_sha256": TARGET_HASHES[implementation],
            "targets_source_url": TARGET_URLS[implementation],
            "shard_sha256": shard_hash,
            "problems_sha256": PROBLEM_HASHES[implementation],
            "harness_revision": REVISIONS[implementation],
            "schema_version": 1,
            "provide_scientific_background": provide_scientific_background,
            "asset_manifest": provenance,
        }
        samples.append(
            Sample(
                id=f"{implementation}:{pid}",
                input=packet,
                metadata=metadata,
                files=files,
            )
        )
    return MemoryDataset(samples, name=f"{implementation}-defect-investigation")
