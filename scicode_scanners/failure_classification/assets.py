"""Pinned HDF5 assets, exact harness decoders, and lossless typed previews."""

import asyncio
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


async def target_file(
    implementation: Implementation, override: str | None = None
) -> Path:
    async with _lock:
        if override:
            return await asyncio.to_thread(
                verify_targets, Path(override), implementation
            )
        cache = Path.home() / ".cache" / "scicode_failure_classification"
        cache.mkdir(parents=True, exist_ok=True)
        spec = TARGETS[implementation]
        path = cache / spec["filename"]
        if implementation == "scicode":
            await asyncio.to_thread(
                gdrive_download, spec["gdrive"], spec["sha256"], path
            )
        else:
            await asyncio.to_thread(download, spec["url"], spec["sha256"], path)
        return await asyncio.to_thread(verify_targets, path, implementation)


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


def typed(value: Any, offset: int = 0, limit: int = 64) -> Any:
    """Bound previews without losing dtype/shape or hiding their incompleteness."""
    if sparse.issparse(value):
        return {
            "type": type(value).__name__,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nnz": value.nnz,
            "data": typed(value.data, offset, limit),
            "note": "Full sparse object is available as targets in Python.",
        }
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "offset": offset,
            "size": value.size,
            "values": [typed(v) for v in flat[offset : offset + limit]],
            "truncated": offset > 0 or offset + limit < value.size,
        }
    if isinstance(value, np.generic):
        return typed(value.item())
    if isinstance(value, complex):
        return {"type": "complex", "real": value.real, "imag": value.imag}
    if isinstance(value, float) and not np.isfinite(value):
        return {"type": "float", "value": str(value)}
    if isinstance(value, (tuple, list)):
        return {
            "type": type(value).__name__,
            "size": len(value),
            "offset": offset,
            "items": [typed(v, limit=limit) for v in value[offset : offset + limit]],
            "truncated": offset > 0 or offset + limit < len(value),
        }
    if isinstance(value, dict):
        entries = list(value.items())
        return {
            "type": "dict",
            "size": len(entries),
            "offset": offset,
            "entries": [
                [typed(k), typed(v, limit=limit)]
                for k, v in entries[offset : offset + limit]
            ],
            "truncated": offset > 0 or offset + limit < len(entries),
        }
    if isinstance(value, bytes):
        return {"type": "bytes", "hex": value.hex()}
    return value
