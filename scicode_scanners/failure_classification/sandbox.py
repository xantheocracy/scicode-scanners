"""Explicit Kubernetes sandbox lifecycle for Hawk Scout scan jobs."""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from .adapters import CaseSet
from .assets import ROOT, TARGETS

IMAGES = {
    "scicode": "ghcr.io/xantheocracy/scicode-sandbox:v1@sha256:16dfd49d3ca8b006097bb6c8583003c237eb422779a76183914c18bc027418d6",
    "scicode_verified": "ghcr.io/xantheocracy/scicode-sandbox:verified-v1@sha256:bfe90450667f23be60d1c05978281ff29efaf1f3979a875f8362d0595784a981",
}
THREAD_ENV = {
    name: "1"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


class FreshSandbox:
    """New working directory per experiment prevents diagnostic modifications carrying over."""

    def __init__(self, environment, files):
        self.environment, self.files = environment, files

    async def exec(self, cmd, **kwargs):
        directory = f"/tmp/scicode-diagnostic-{uuid.uuid4().hex}"
        result = await self.environment.exec(
            ["mkdir", "-p", directory], timeout=30, timeout_retry=False
        )
        if not result.success:
            raise RuntimeError(f"Cannot create diagnostic directory: {result.stderr}")
        for name, content in self.files.items():
            await self.environment.write_file(f"{directory}/{name}", content)
        try:
            return await self.environment.exec(cmd, cwd=directory, **kwargs)
        finally:
            # Only remove the fresh directory allocated by this wrapper inside the sandbox.
            await self.environment.exec(
                ["rm", "-rf", directory], timeout=30, timeout_retry=False
            )


def helm_values(implementation, image=None, runtime_class="gvisor"):
    return {
        "allowDomains": [],
        "allowEntities": [],
        "allowCIDR": [],
        "automountServiceAccountToken": False,
        "services": {
            "default": {
                "image": image or IMAGES[implementation],
                "command": ["tail", "-f", "/dev/null"],
                "runtimeClassName": runtime_class,
                "nodeSelector": {"kubernetes.io/arch": "amd64"},
                "resources": {
                    "requests": {"cpu": "500m", "memory": "1Gi"},
                    "limits": {"cpu": "2", "memory": "4Gi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
            }
        },
        "labels": {
            "app.kubernetes.io/component": "sandbox",
            "app.kubernetes.io/part-of": "inspect-ai",
        },
    }


@asynccontextmanager
async def investigation_sandbox(
    cases: CaseSet,
    shard: bytes,
    image: str | None = None,
    runtime_class: str = "gvisor",
):
    if not os.environ.get("KUBERNETES_SERVICE_HOST"):
        raise RuntimeError("Python investigations run on Hawk Kubernetes runners only.")
    from k8s_sandbox import K8sSandboxEnvironment, K8sSandboxEnvironmentConfig

    task = "scicode-failure-classification"
    with TemporaryDirectory(prefix="failure-sandbox-") as directory:
        config_file = Path(directory) / "values.yaml"
        # JSON is valid YAML; avoid an additional YAML dependency in the scanner.
        config_file.write_text(
            json.dumps(helm_values(cases.implementation, image, runtime_class))
        )
        config = K8sSandboxEnvironmentConfig(
            values=config_file, restarted_container_behavior="raise"
        )
        await K8sSandboxEnvironment.task_init(task, config)
        envs = await K8sSandboxEnvironment.sample_init(
            task, config, {"transcript": cases.transcript.transcript_id}
        )
        try:
            env = envs["default"]
            filename = TARGETS[cases.implementation]["filename"]
            files = {filename: shard}
            if cases.implementation == "scicode":
                files.update(
                    {
                        name: (ROOT / "vendor" / name).read_bytes()
                        for name in ("process_data.py", "test_util.py")
                    }
                )
            else:
                files.update(
                    {
                        str(p.relative_to(ROOT / "vendor" / "verified")): p.read_bytes()
                        for p in (ROOT / "vendor" / "verified").rglob("*.py")
                    }
                )
            yield FreshSandbox(env, files)
        finally:
            # Scout has no task teardown hook; clean each release even on cancellation.
            await asyncio.shield(
                K8sSandboxEnvironment.sample_cleanup(task, config, envs, False)
            )


def interpreters(cases: CaseSet):
    if cases.implementation == "scicode":
        return {"default": "python"}
    selected = cases.settings.get("grading_environments", "both")
    if selected not in {"both", "2024", "2025"}:
        raise ValueError(f"Unknown grading environment: {selected}")
    return {
        key: f"/opt/scicode-{key}/bin/python"
        for key in ("2024", "2025")
        if selected == "both" or key == selected
    }
