# SIGMo Python Examples

This folder contains runnable examples showing how to use the Python interface built on top of SIGMo.

The examples are organized by complexity:

- `basic_usage.py`: minimal high-level usage with `sigmo.match()`;
- `advanced_pipeline.py`: direct `PipelineContext` usage for step-by-step execution;
- `run_pipeline.py`: full command-line kernel-level pipeline execution;
- `visualization_usage.py`: molecule, CSR graph and match-pair visualization examples;
- `outputs/`: folder used to store generated PNG/CSV/JSON outputs.

All commands should be executed from the **repository root**, not from inside the `examples/` folder.

```bash
cd /path/to/sigmo-python
```

---

## Environment Setup

Before running the examples, build the native extension and install the package in editable mode.

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh

./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
```

After the editable install, examples can be executed directly with `python` without setting `PYTHONPATH=python` manually.

Verify the package import:

```bash
python - <<'PY'
import sigmo
print(sigmo.__file__)
print("SIGMo import OK")
PY
```

---

## `basic_usage.py`

This is the simplest example and the recommended starting point.

It demonstrates:

- importing `sigmo`;
- running a single query-target match with `sigmo.match()`;
- printing `summary()` and `explain()`;
- optionally converting the result to a pandas DataFrame if pandas is installed.

Run:

```bash
python examples/basic_usage.py
```

The example matches:

```text
query:  CC
target: CCC
```

Expected behavior:

```text
SIGMo search summary
Status: OK
Matches found: 1
```

This example is useful to verify that the package, native extension and SYCL device selection are working correctly.

---

## `advanced_pipeline.py`

This example shows how to use the lower-level `PipelineContext` API directly.

It demonstrates the kernel-level pipeline:

```text
allocate()
generate_signatures()
filter_candidates()
refine()
join()
build_match_result()
```

Run:

```bash
python examples/advanced_pipeline.py
```

This example loads a small subset of the benchmark dataset and keeps the run lightweight.

It is useful for users who want to inspect the individual SIGMo stages instead of using the high-level `sigmo.match()` or `sigmo.search()` APIs.

---

## `run_pipeline.py`

This is the most complete command-line example.

It exposes the SIGMo pipeline at kernel level and prints detailed logs about:

- loaded query/data graphs;
- selected SYCL device;
- allocation of native structures;
- signature generation;
- candidate filtering;
- optional refinement;
- final join/isomorphism;
- match preview;
- `MatchResult.summary()`;
- `MatchResult.explain()`.

Run a small safe example:

```bash
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 0 \
  --device auto
```

Run with forced refinement:

```bash
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 6 \
  --force-refine \
  --device auto
```

`--force-refine` is an advanced option. It forces refinement even when small graphs are present. This may be useful for debugging or benchmarking, but it can be unstable on some datasets if the native backend does not handle micro-molecules safely.

Export CSV and JSON:

```bash
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 0 \
  --csv examples/outputs/matches.csv \
  --json examples/outputs/summary.json
```

For large result sets, CSV export is written in streaming mode and JSON output is reduced to a lightweight summary to avoid materializing millions of matches in memory.

Main options:

```text
--base-dir PATH
    Directory containing query/data files.

--query-file FILE
    Query SMARTS/SMILES file.

--data-file FILE
    Database SMARTS/SMILES file.

--input-format auto|smiles|smarts
    Input format used by sigmo.load_molecules().

--device auto|gpu|cpu|cuda|cuda:gpu
    SYCL device selector.

--iterations N
    Maximum number of refinement iterations.

--find-first / --no-find-first
    Stop at the first match per pair when supported.

--query-limit N
    Maximum number of query graphs to load. Use -1 for no limit.

--data-limit N
    Maximum number of data graphs to load. Use -1 for no limit.

--csv PATH
    Optional CSV path for exporting matches.

--json PATH
    Optional JSON path for exporting the result or summary.

--force-refine
    Force refinement even when small graphs are present.

--max-print-matches N
    Maximum number of matches printed to the terminal. Use 0 to disable.
```

---

## `visualization_usage.py`

This example demonstrates the optional visualization utilities provided by `sigmo.visualize`.

It generates:

- a single molecule drawing;
- a query-target pair drawing;
- a small internal SIGMo CSR graph;
- multiple real molecules from the benchmark dataset;
- real query-target match pairs found by SIGMo.

Run:

```bash
python examples/visualization_usage.py
```

Generated PNG files are saved in:

```text
examples/outputs/
```

Typical generated files include:

```text
molecule.png
match_pair.png
csr_graph_small.png
dataset_molecule_1.png
csr_graph_dataset_sample_1.png
real_dataset_match_pair_1.png
```

Important note: match-pair highlighting is computed with RDKit and is used only for visualization. Current SIGMo results are pair-level results: SIGMo reports that query graph `i` matches data graph `j`, but it does not expose an atom-level mapping for drawing.

---

## Output Files

The `examples/outputs/` folder is used for generated files.

Possible generated outputs include:

```text
*.png   visualization outputs
*.csv   match exports
*.json  result summaries or lightweight metadata
```

Generated files should normally not be committed.

The repository should keep only:

```text
examples/outputs/.gitkeep
```

Recommended `.gitignore` rules:

```gitignore
examples/outputs/*
!examples/outputs/.gitkeep
```

---

## Notes on RDKit

RDKit is used by the examples and package for:

- parsing SMILES/SMARTS inputs;
- converting molecules into CSR-compatible graph dictionaries;
- drawing molecule images;
- highlighting query substructures in visualization examples.

The examples do not perform RDKit validation or RDKit-vs-SIGMo comparison. This is intentional: SIGMo currently works on a simplified CSR representation, while RDKit uses full SMARTS semantics. Direct comparison can be misleading unless designed as a separate controlled experiment.

---

## Recommended Example Workflow

After installation, run the examples in this order:

```bash
python examples/basic_usage.py
python examples/advanced_pipeline.py
python examples/run_pipeline.py --query-limit 5 --data-limit 20 --iterations 0
python examples/visualization_usage.py
```

Then run the tests:

```bash
pytest tests -vv
```

---

## Current Status

```text
basic_usage.py: OK
advanced_pipeline.py: OK
run_pipeline.py: OK
visualization_usage.py: OK
```
