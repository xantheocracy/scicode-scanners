"""Conservative per-step budget checkpoints under the Hawk scan's results URI."""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

import fsspec
import yaml


def hawk_results_uri():
    # Hawk invokes its scan runner with USER_CONFIG_FILE and INFRA_CONFIG_FILE.
    for argument in sys.argv[1:]:
        path = Path(argument)
        if path.suffix not in {".yaml", ".yml", ".json"} or not path.is_file():
            continue
        config = yaml.safe_load(path.read_text())
        if (
            isinstance(config, dict)
            and config.get("job_id")
            and config.get("results_dir")
        ):
            return config["results_dir"].rstrip("/") + "/failure_checkpoints"
    return None


class CheckpointStore:
    def __init__(self, base_uri, identity):
        fingerprint = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()
        self.uri = f"{base_uri.rstrip('/')}/{fingerprint}.json" if base_uri else None

    async def read(self):
        if not self.uri:
            return None

        def load():
            try:
                with fsspec.open(self.uri, "r") as f:
                    return json.load(f)
            except FileNotFoundError:
                return None

        return await asyncio.to_thread(load)

    async def write(self, value):
        if not self.uri:
            return

        def save():
            fs, path = fsspec.core.url_to_fs(self.uri)
            if (
                fs.protocol in {"file", "local"}
                or isinstance(fs.protocol, tuple)
                and "file" in fs.protocol
            ):
                destination = Path(path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(".partial")
                temporary.write_text(json.dumps(value, ensure_ascii=False))
                temporary.replace(destination)
            else:
                with fs.open(path, "w") as f:
                    json.dump(value, f, ensure_ascii=False)

        await asyncio.to_thread(save)
