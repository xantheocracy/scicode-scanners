"""Pinned HDF5 assets, exact harness decoders, and target metadata."""

import asyncio
import fcntl
import hashlib
import importlib.util
import io
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from inspect_ai.util import download, gdrive_download
from scipy import sparse

from .adapters import Implementation

ROOT = Path(__file__).parent
TARGETS = {
    "scicode": {
        "filename": "test_data.h5",
        "sha256": "48b0272a88b17dbd29777c217e1b4fb2b019b92e11cc2add847409db9541b890",
        "gdrive": "17G_k65N_6yFFZ2O-jQH00Lh6iaw3z-AW",
    },
    "scicode_verified": {
        "filename": "test_data_cleaned.h5",
        "sha256": "8fb6e575b7b6dda5e48b04dea338fc6af4fe185774b8f19221c96945df9b4142",
        "url": "https://huggingface.co/datasets/shhu2001/SciCode-Verified/resolve/eea11a866be6860725258702b39ef8651ed26abd/test_data_cleaned.h5",
    },
}
_lock = asyncio.Lock()


def verify_targets(path: Path, implementation: Implementation) -> Path:
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    if digest != TARGETS[implementation]["sha256"]:
        raise ValueError(f"Target checksum mismatch for {implementation}.")
    return path


def fetch_targets(implementation: Implementation, cache: Path) -> Path:
    """Download and verify targets, serialized across Scout worker processes."""
    cache.mkdir(parents=True, exist_ok=True)
    spec = TARGETS[implementation]
    path = cache / spec["filename"]
    # Downloads share a fixed `.partial` path, so concurrent processes must not overlap.
    with open(path.with_suffix(path.suffix + ".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if implementation == "scicode":
            gdrive_download(spec["gdrive"], spec["sha256"], path)
        else:
            download(spec["url"], spec["sha256"], path)
        return verify_targets(path, implementation)


async def target_file(
    implementation: Implementation, override: str | None = None
) -> Path:
    async with _lock:
        if override:
            return await asyncio.to_thread(
                verify_targets, Path(override), implementation
            )
        cache = Path.home() / ".cache" / "scicode_failure_classification"
        return await asyncio.to_thread(fetch_targets, implementation, cache)


def shard_targets(path: Path, step_ids: list[str]) -> bytes:
    stream = io.BytesIO()
    with h5py.File(path, "r") as src, h5py.File(stream, "w") as dest:
        for sid in step_ids:
            if sid not in src:
                raise ValueError(f"Missing HDF5 target group {sid}.")
            src.copy(sid, dest)
    return stream.getvalue()


def decode_targets(path: Path, implementation: Implementation, sid: str, count: int):
    # Load independently rather than importing a potentially conflicting installed scicode package.
    file = (
        ROOT
        / "vendor"
        / (
            "process_data.py"
            if implementation == "scicode"
            else "verified/scicode/parse/parse.py"
        )
    )
    spec = importlib.util.spec_from_file_location(
        f"_failure_targets_{implementation}", file
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if implementation == "scicode":
        module.H5PY_FILE = str(path)
        return module.process_hdf5_to_tuple(sid, count)
    return module.process_hdf5_to_tuple(sid, count, str(path))


def target_metadata(value: Any) -> Any:
    """Describe targets without copying potentially large values into context."""
    if sparse.issparse(value) or isinstance(value, np.ndarray):
        result = {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if sparse.issparse(value):
            result["nnz"] = value.nnz
        return result
    if isinstance(value, (tuple, list, dict)):
        return {"type": type(value).__name__, "size": len(value)}
    return {"type": type(value).__name__}
