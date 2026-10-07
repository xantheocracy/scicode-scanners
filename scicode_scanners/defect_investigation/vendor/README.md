# Vendored benchmark resources

Original SciCode prompts, target decoder, comparator, and scorer snapshot come from `inspect_evals` revision `e02028192f8bbf3677c6a672eaf48cf20e7a6d53`. Original evaluation problems come from `scicode-bench/SciCode` revision `69a8cfc829fe8788a426ce8b5de6292366dce7ef` (`problems_all.jsonl`).

SciCode-Verified problems, helpers, supplied code, prompt templates, and task snapshot come from `xantheocracy/scicode-verified` revision `3fd9ef2713244d58f6cfeab5a5d92c10e2b907de`. Verified HDF5 comes from the dataset release at `eea11a866be6860725258702b39ef8651ed26abd`.

The development JSONL is the pristine original SciCode development split vendored in that Verified checkout. Source files and data are checksum-verified through `../asset_manifest.json`. Original and Verified licenses/attributions are retained in source headers and `LICENSE` (Apache 2.0).

`original_scorer.py.txt` and `verified_task.py.txt` are inert source snapshots for harness-parity tests. They are never imported, executed as tasks, or provided to the investigation agent. Candidate-dependent previous code is represented explicitly in the specification packet rather than fabricated as a benchmark reference solution.
