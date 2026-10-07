# SciCode scanners

Inspect Scout scanners for SciCode experiments, run through Hawk. Each scanner has its own subdirectory under `scicode_scanners/`, containing its implementation, supporting data, run configuration, and documentation.

## Scanners

- [Verified memory](scicode_scanners/verified_memory/README.md): detects failed subproblems caused by recalling original SciCode behavior.

## Layout

```text
scicode_scanners/
  verified_memory/
    __init__.py
    scanner.py
    memory.py
    data/audit_context.json
    hawk.yaml
    README.md
  _registry.py
  logs.py
pyproject.toml
```

`pyproject.toml` defines shared dependencies and the Hawk package entry point. `_registry.py` imports each registered scanner; `logs.py` provides shared log-reading utilities.

To add a scanner, create another subdirectory under `scicode_scanners/`, export its scanner function from its `__init__.py`, and import it in `_registry.py`. Keep its supporting files and Hawk configuration in that directory.

Run the existing scanner from the repository root:

```bash
hawk scan run scicode_scanners/verified_memory/hawk.yaml
```

This launches paid inference. See the scanner's README for setup and detection criteria.
