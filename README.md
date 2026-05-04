# SIGMo Python Interface

Python interface for **SIGMo**, a high-performance subgraph isomorphism library based on C++/SYCL kernels.

This repository provides a Python layer on top of the native SIGMo binding. The goal is to make SIGMo usable not only from low-level C++/SYCL code, but also from Python workflows used by chemists, researchers and technical users working with molecular datasets.

The interface is designed around three usage levels:

1. **High-level API** for simple molecule matching.
2. **Batch/search API** for query datasets against large data datasets.
3. **Kernel-level API** for advanced users who want to execute and inspect each SIGMo kernel step-by-step.

The current implementation supports molecule loading from SMARTS/SMILES, conversion to SIGMo-compatible CSR graphs, execution of the full matching pipeline, explainable results, test coverage, visualization utilities and streaming output for very large result sets.

---

## Table of Contents

- [Project Goals](#project-goals)
- [Main Features](#main-features)
- [Repository Structure](#repository-structure)
- [Architecture Overview](#architecture-overview)
- [Installation and Environment](#installation-and-environment)
- [Quick Start](#quick-start)
- [High-Level API](#high-level-api)
- [Batch Search API](#batch-search-api)
- [Object-Oriented API](#object-oriented-api)
- [Kernel-Level Pipeline](#kernel-level-pipeline)
- [Command-Line Kernel Pipeline Example](#command-line-kernel-pipeline-example)
- [Input Format](#input-format)
- [Bond Label Policy](#bond-label-policy)
- [Output Format](#output-format)
- [Large Result Handling](#large-result-handling)
- [Visualization Utilities](#visualization-utilities)
- [Testing](#testing)
- [Experimental Results](#experimental-results)
- [Known Limitations](#known-limitations)
- [Development Notes](#development-notes)
- [Current Status](#current-status)

---

## Project Goals

The original SIGMo library exposes high-performance kernels for subgraph isomorphism. This Python interface aims to provide a more accessible and research-oriented frontend.

The main goals are:

- expose SIGMo functionality through a Pythonic API;
- support molecular input through SMARTS and SMILES;
- hide low-level SYCL details from non-HPC users;
- preserve access to individual kernels for advanced users;
- provide explainable outputs through summaries and execution traces;
- support molecule, CSR graph and match-pair visualization;
- handle very large outputs without exhausting memory;
- provide examples and tests for reproducible usage.

---

## Main Features

Current features include:

- high-level `sigmo.match()` function for single query-target matching;
- batch `sigmo.search()` function for query datasets against data datasets;
- object-oriented `SIGMoMatcher` API;
- advanced `PipelineContext` API for kernel-level execution;
- SMARTS/SMILES parsing using RDKit;
- conversion from molecular strings to SIGMo-compatible CSR graphs;
- automatic SYCL device selection through `dpctl`;
- support for GPU or CPU execution;
- kernel step tracking;
- `MatchResult` object with:
  - `summary()`;
  - `explain()`;
  - `to_csv()`;
  - `to_json()`;
- optional DataFrame conversion through pandas;
- visualization utilities for:
  - individual molecules;
  - query-target pairs;
  - internal SIGMo CSR graphs;
  - real match pairs from benchmark datasets;
- large-result mode for millions of matches;
- CSV streaming export;
- lightweight JSON summary export;
- automated pytest test suite.

---

## Repository Structure

Typical structure:

```text
sigmo-python/
├── python/
│   └── sigmo/
│       ├── __init__.py
│       ├── config.py
│       ├── graph.py
│       ├── matcher.py
│       ├── pipeline.py
│       ├── result.py
│       ├── utils.py
│       └── visualize.py
│
├── examples/
│   ├── basic_usage.py
│   ├── advanced_pipeline.py
│   ├── run_pipeline.py
│   ├── visualization_usage.py
│   └── outputs/
│       └── .gitkeep
│
├── tests/
│   ├── conftest.py
│   ├── helpers.py
│   ├── test_graph.py
│   ├── test_low_level_kernels.py
│   ├── test_matcher.py
│   ├── test_pipeline.py
│   └── test_visualize.py
│
├── benchmarks/
│   └── datasets/
│       ├── query.smarts
│       └── data.smarts
│
├── external/
│   └── sigmo/
│
├── scripts/
│   ├── build.sh
│   └── dev_env.sh
│
├── docs/
├── pyproject.toml
├── setup.py
├── .gitignore
└── README.md
```

The `benchmarks/datasets/*.smarts` files may be ignored by Git if they are large or externally distributed.

The `examples/outputs/` directory is intended for generated PNG, CSV and JSON outputs. Its contents should normally be ignored by Git, except for `.gitkeep`.

---

## Architecture Overview

The Python interface is organized as a layered system.

```text
User
 |
 |  sigmo.match()
 |  sigmo.search()
 |  SIGMoMatcher.run()
 v
matcher.py
 |
 |  load_molecules()
 v
graph.py
 |
 |  SMARTS/SMILES parsing
 |  RDKit molecule conversion
 |  CSR graph generation
 v
PipelineContext
 |
 |  allocate()
 |  generate_signatures()
 |  filter_candidates()
 |  refine()
 |  join()
 v
_core C++/SYCL binding
 |
 |  generate_csr_signatures
 |  refine_csr_signatures
 |  filter_candidates
 |  refine_candidates
 |  join_candidates
 v
raw join result
 |
 v
build_match_result()
 |
 v
MatchResult
 |
 |  summary()
 |  explain()
 |  to_csv()
 |  to_json()
 v
User-readable result / CSV / JSON
```

The native C++/SYCL binding remains thin. The Python layer is responsible for:

- input handling;
- graph conversion;
- pipeline orchestration;
- output formatting;
- large-result management;
- visualization helpers;
- user-facing API design.

RDKit is used for molecule parsing and visualization support. It is not used as a public validation or benchmarking layer in the current interface.

---

## Installation and Environment

The project expects an environment where oneAPI/SYCL, `dpctl`, RDKit and the native SIGMo dependencies are available.

A typical development setup is:

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh
```

Build the native extension from the repository root:

```bash
./scripts/build.sh
```

Then install the Python package in editable mode:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

After this step, the package can be imported without setting `PYTHONPATH` manually.

Verify the import:

```bash
python - <<'PY'
import sigmo
print(sigmo.__file__)
print("SIGMo import OK")
PY
```

Run tests:

```bash
pytest tests -vv
```

Main runtime dependencies include:

- Python;
- RDKit;
- dpctl;
- the compiled SIGMo Python extension;
- pytest for testing.

Optional dependencies may include:

- pandas, if using DataFrame-based workflows;
- networkx and matplotlib, if using CSR graph visualization.

---

## Quick Start

Run a simple molecule match:

```python
import sigmo

result = sigmo.match(
    query="CC",
    target="CCC",
    input_format="smiles",
    iterations=0,
    find_first=True,
    device="auto",
)

print(result.summary())
print(result.explain())
```

Example output:

```text
SIGMo search summary
--------------------
Status: OK
Device: NVIDIA GeForce RTX 3060 Laptop GPU
Queries: 1
Database graphs: 1
Matches found: 1
Refinement iterations: 0/0
Kernel steps:
  - generate_query_signatures
  - generate_data_signatures
  - filter_candidates
  - join_candidates
```

The corresponding example script is:

```bash
python examples/basic_usage.py
```

---

## High-Level API

The simplest entry point is `sigmo.match()`.

```python
import sigmo

result = sigmo.match(
    query="c1ccccc1",
    target="CCOC(=O)c1ccccc1",
    input_format="smiles",
    iterations=0,
    find_first=True,
    device="auto",
)

print(result.summary())
print(result.explain())
```

This function:

1. parses the query and target;
2. converts them to CSR graphs;
3. runs the SIGMo pipeline;
4. returns a `MatchResult`.

This API is intended for users who do not need to manage queues, signatures, candidates or kernels manually.

---

## Batch Search API

For matching many queries against a database:

```python
import sigmo

result = sigmo.search(
    queries="benchmarks/datasets/query.smarts",
    database="benchmarks/datasets/data.smarts",
    input_format="auto",
    iterations=6,
    find_first=True,
    device="auto",
)

print(result.summary())
```

The `queries` and `database` arguments may be:

- file paths;
- lists of SMILES/SMARTS strings;
- RDKit `Mol` objects;
- already constructed CSR graph dictionaries.

---

## Object-Oriented API

For more configurable usage:

```python
import sigmo

matcher = sigmo.SIGMoMatcher(
    device="auto",
    iterations=6,
    find_first=True,
    input_format="auto",
)

result = matcher.run(
    queries="benchmarks/datasets/query.smarts",
    database="benchmarks/datasets/data.smarts",
)

print(result.summary())
```

The object stores:

- loaded query graphs;
- loaded data graphs;
- last execution context;
- last result.

This is useful when building larger Python workflows around SIGMo.

---

## Kernel-Level Pipeline

Advanced users can access the pipeline step-by-step through `PipelineContext`.

```python
import sigmo

query_graphs = sigmo.load_molecules(
    "benchmarks/datasets/query.smarts",
    input_format="auto",
)

data_graphs = sigmo.load_molecules(
    "benchmarks/datasets/data.smarts",
    input_format="auto",
)

ctx = sigmo.PipelineContext(
    query_graphs=query_graphs,
    data_graphs=data_graphs,
    device="auto",
)

ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(iterations=6)
ctx.join(find_first=True)
```

This level exposes the actual SIGMo pipeline stages:

```text
allocate
generate signatures
filter candidates
refine signatures
refine candidates
join candidates
```

It is useful for debugging, benchmarking and research experiments.

The corresponding example script is:

```bash
python examples/advanced_pipeline.py
```

---

## Command-Line Kernel Pipeline Example

The main advanced example is:

```bash
python examples/run_pipeline.py
```

Example with full dataset, 6 refinement iterations and streaming output:

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0 \
  --csv examples/outputs/matches_full_refine.csv \
  --json examples/outputs/matches_full_refine_summary.json
```

Main options:

```text
--query-limit N
    Limit the number of query graphs. Use -1 for no limit.

--data-limit N
    Limit the number of data graphs. Use -1 for no limit.

--iterations N
    Number of refinement iterations.

--force-refine
    Force refinement even if small graphs are detected.

--max-print-matches N
    Maximum number of matches printed to terminal. Use 0 to disable match preview.

--csv PATH
    Export matches to CSV.

--json PATH
    Export a JSON result or lightweight summary.

--device auto|gpu|cpu|cuda|cuda:gpu
    Select the SYCL device.
```

By default, the command-line example uses conservative settings suitable for small test runs.

---

## Input Format

The interface supports SMARTS and SMILES strings through RDKit.

Example file format:

```text
C1CCCCC1 Cyclohexane
CC[NH+]CC
[NX3][NX2]=[*] Query_1
```

If a molecule name is present after the SMARTS/SMILES string, it is preserved.

If no name is present, an automatic identifier is generated.

Each molecule is converted to a CSR graph with fields such as:

```python
{
    "row_offsets": [...],
    "column_indices": [...],
    "node_labels": [...],
    "edge_labels": [...],
    "num_nodes": ...,
    "name": ...,
    "input": ...,
    "input_format": ...
}
```

---

## Bond Label Policy

The current conversion uses:

```python
int(bond.GetBondTypeAsDouble())
```

This means:

```text
single bond    -> 1
double bond    -> 2
triple bond    -> 3
aromatic bond  -> int(1.5) = 1
```

This policy follows the original prototype behavior and avoids passing unsupported bond labels to the backend.

An earlier explicit aromatic label such as `12` caused native crashes in some cases, so the current version prioritizes backend compatibility and stability.

This makes the CSR representation less chemically expressive than RDKit's full SMARTS semantics, but keeps the SIGMo backend stable.

Future versions may add a configurable aromatic policy once backend support is validated.

---

## Output Format

Most APIs return a `MatchResult`.

A `MatchResult` contains:

- total number of matches;
- query count;
- data graph count;
- device name;
- requested and executed refinement iterations;
- kernel steps;
- warnings;
- errors;
- raw result metadata.

Example:

```python
print(result.summary())
```

```python
print(result.explain())
```

For small and medium outputs:

```python
result.to_csv("matches.csv")
result.to_json("matches.json")
```

For optional pandas workflows:

```python
df = result.to_dataframe()
```

---

## Large Result Handling

Very large molecular searches may produce millions of matches.

Materializing every match as a Python object or exporting a complete JSON list can exhaust memory. To avoid this, `examples/run_pipeline.py` supports a large-result mode.

In large-result mode:

- the full match list is not materialized inside `MatchResult`;
- the terminal output shows only a preview;
- CSV is written in streaming mode;
- JSON contains only summary and kernel statistics.

Recommended command:

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0 \
  --csv examples/outputs/matches_full_refine.csv \
  --json examples/outputs/matches_full_refine_summary.json
```

The CSV output contains rows such as:

```csv
query_index,query_name,query_input,data_index,data_name,data_input
602,query:603,[CX3](=[OX1])...,2977,data:2978,O=C1N2N(...)
```

The JSON summary contains metadata, warnings and kernel timings, but not all matches.

---

## Visualization Utilities

The package includes optional visualization helpers in `sigmo.visualize`.

They support:

- drawing a single molecule;
- drawing a query-target pair;
- highlighting a query inside a target molecule using RDKit;
- converting a SIGMo CSR graph to NetworkX;
- drawing an internal CSR graph for debugging;
- generating real match-pair examples from the benchmark dataset.

Example:

```python
from sigmo.visualize import draw_molecule, draw_match_pair, draw_graph

draw_molecule(
    "CCO",
    input_format="smiles",
    output_path="examples/outputs/molecule.png",
)

draw_match_pair(
    "CC",
    "CCC",
    query_format="smiles",
    target_format="smiles",
    output_path="examples/outputs/match_pair.png",
    highlight=True,
)
```

Run the full visualization example:

```bash
python examples/visualization_usage.py
```

This script generates PNG files in:

```text
examples/outputs/
```

Generated images should not normally be committed.

Important note: match-pair highlighting is RDKit-based and is used only for visualization. Current SIGMo results are pair-level results, meaning that SIGMo reports that query graph `i` matches data graph `j`, but it does not expose an atom-level mapping for drawing.

---

## Testing

Run the full test suite:

```bash
pytest tests -vv
```

The tests cover:

- graph conversion;
- SMARTS/SMILES loading;
- aromaticity stability;
- low-level kernels;
- high-level API;
- batch API;
- object-oriented API;
- `PipelineContext`;
- CSV/JSON export;
- visualization utilities.

Test files:

```text
tests/test_graph.py
tests/test_low_level_kernels.py
tests/test_matcher.py
tests/test_pipeline.py
tests/test_visualize.py
```

---

## Experimental Results

The pipeline has been tested on a full molecular dataset composed of:

```text
619 query graphs
114,901 data graphs
```

### Full dataset without refinement

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 0 \
  --max-print-matches 0 \
  --csv examples/outputs/matches_full_no_refine.csv \
  --json examples/outputs/matches_full_no_refine_summary.json
```

Observed results:

```text
Candidates after initial filter: 3,610,045,526
Final matches: 17,839,985
Join time: approximately 45.8 seconds
```

### Full dataset with 6 refinement iterations

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0 \
  --csv examples/outputs/matches_full_refine.csv \
  --json examples/outputs/matches_full_refine_summary.json
```

Observed results:

```text
Candidates after initial filter: 3,610,045,526
Candidates after 6 refinement iterations: 389,794,092
Final matches: 15,898,409
Join time: approximately 13.4 seconds
```

Summary table:

| Scenario | Candidates after filter | Candidates after refinement | Final matches | Join time |
|---|---:|---:|---:|---:|
| Full dataset, no refinement | 3,610,045,526 | - | 17,839,985 | ~45.8s |
| Full dataset, 6 refinements | 3,610,045,526 | 389,794,092 | 15,898,409 | ~13.4s |

The refinement stage reduced the candidate space by approximately 89.2% and significantly reduced the final join time.

---

## Known Limitations

Current limitations:

- aromatic bonds are currently collapsed to bond label `1` for backend compatibility;
- the CSR representation is structural and does not preserve the full SMARTS semantics used by RDKit;
- JSON export for millions of matches is intentionally avoided;
- large outputs should be exported using CSV streaming;
- `--force-refine` may execute refinement on small graphs and should be considered an advanced option;
- visualization highlighting is RDKit-based and does not represent a SIGMo atom-level mapping;
- the exact behavior depends on the available SYCL backend and device.

---

## Development Notes

Recommended workflow during development:

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh

./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
```

Run the test suite:

```bash
pytest tests -vv
```

Run a simple example:

```bash
python examples/basic_usage.py
```

Run visualization example:

```bash
python examples/visualization_usage.py
```

Run kernel-level pipeline on a small subset:

```bash
python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

Run full dataset pipeline:

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0 \
  --csv examples/outputs/matches_full_refine.csv \
  --json examples/outputs/matches_full_refine_summary.json
```

Generated outputs should not normally be committed.

The repository should track only:

```text
examples/outputs/.gitkeep
```

and ignore generated files through:

```gitignore
examples/outputs/*
!examples/outputs/.gitkeep
```

Editable install notes:

- if only Python files are modified, reinstalling is not required;
- if C++/pybind11/CMake files are modified, rerun `./scripts/build.sh`;
- if a new environment is created, rerun `python -m pip install -e . --no-build-isolation --no-deps`.

---

## Current Status

The current Python interface provides:

```text
High-level API: OK
Batch API: OK
Object-oriented API: OK
Kernel-level API: OK
Visualization utilities: OK
Automated tests: OK
Full dataset without refinement: OK
Full dataset with refinement: OK
Streaming CSV export: OK
JSON summary export: OK
Editable package install: OK
```

## Additional Documentation

More detailed documentation is available in the `docs/` folder:

- [`docs/api.md`](docs/api.md): Python API reference.
- [`docs/architecture.md`](docs/architecture.md): internal architecture and pipeline flow.
- [`docs/build.md`](docs/build.md): build and environment setup.
- [`docs/benchmarks.md`](docs/benchmarks.md): benchmark commands and experimental results.
