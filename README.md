# SIGMo Python Interface

Python interface for **SIGMo**, a high-performance subgraph isomorphism library based on C++/SYCL kernels.

This repository provides a Python layer on top of the native SIGMo binding. The goal is to make SIGMo usable not only from low-level C++/SYCL code, but also from Python workflows used by chemists, researchers and technical users working with molecular datasets.

The interface is designed around three usage levels:

1. **High-level API** for simple molecule matching.
2. **Batch/search API** for query datasets against large data datasets.
3. **Kernel-level API** for advanced users who want to execute and inspect each SIGMo kernel step-by-step.

The current implementation supports molecule loading from SMARTS/SMILES, conversion to SIGMo-compatible CSR graphs, execution of the full matching pipeline, optional RDKit validation, explainable results, test coverage and streaming output for very large result sets.

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
- [RDKit Validation](#rdkit-validation)
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
- validate results against RDKit when needed;
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
  - validation metadata;
- optional validation against RDKit `HasSubstructMatch`;
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
│       └── validation.py
│
├── examples/
│   ├── basic_usage.py
│   ├── validation_usage.py
|   ├── advanced_pipeline.py
│   ├── run_pipeline.py
│   └── outputs/
│       └── .gitkeep
│
├── tests/
│   ├── conftest.py
│   ├── test_graph.py
│   ├── test_low_level_kernels.py
│   ├── test_matcher.py
│   ├── test_pipeline.py
│   ├── test_validation.py
│   └── test_validation_against_rdkit.py
│
├── benchmarks/
│   └── datasets/
│       ├── query.smarts
│       └── data.smarts
│
├── pyproject.toml
├── .gitignore
└── README.md
```

The `benchmarks/datasets/*.smarts` files may be ignored by Git if they are large or externally distributed.

The `examples/outputs/` directory is intended for generated CSV and JSON outputs. Its contents should normally be ignored by Git, except for `.gitkeep`.

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
 |  validation
 v
User-readable result / CSV / JSON / validation report
```

The native C++/SYCL binding remains thin. The Python layer is responsible for:

- input handling;
- graph conversion;
- pipeline orchestration;
- output formatting;
- validation;
- large-result management;
- user-facing API design.

---

## Installation and Environment

The project expects an environment where the native SIGMo extension and its dependencies are already built and importable.

Typical development usage from the repository root:

```bash
PYTHONPATH=python python examples/basic_usage.py
```

For tests:

```bash
PYTHONPATH=python pytest tests -vv
```

Main runtime dependencies include:

- Python;
- RDKit;
- dpctl;
- the compiled SIGMo Python extension;
- pytest for testing.

Optional dependencies may include:

- pandas, if using DataFrame-based workflows;
- additional visualization libraries, if molecule visualization is added later.

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
    validate_with_rdkit=False,
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

---

## Command-Line Kernel Pipeline Example

The main advanced example is:

```bash
PYTHONPATH=python python examples/run_pipeline.py
```

Example with full dataset, 6 refinement iterations and streaming output:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
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
    Export a JSON summary.

--device auto|gpu|cpu
    Select the SYCL device.
```

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
- raw result metadata;
- optional validation metadata.

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

---

## Large Result Handling

Very large molecular searches may produce millions of matches.

For example, a full dataset run produced more than 15 million matches.

Materializing every match as a Python object or exporting a complete JSON list can exhaust memory. To avoid this, `examples/run_pipeline.py` supports a large-result mode.

In large-result mode:

- the full match list is not materialized inside `MatchResult`;
- the terminal output shows only a preview;
- CSV is written in streaming mode;
- JSON contains only summary and kernel statistics.

Recommended command:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
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

## RDKit Validation

The interface supports optional validation against RDKit.

Example:

```python
import sigmo

result = sigmo.match(
    query="c1ccccc1",
    target="CCOC(=O)c1ccccc1",
    input_format="smiles",
    iterations=0,
    validate_with_rdkit=True,
)

print(result.summary())
print(result.validation)
```

Validation compares SIGMo results with:

```python
target_mol.HasSubstructMatch(query_mol)
```

The validation output includes:

```text
enabled
method
checked_pairs
agreements
disagreements
passed
```

Validation is intended for small and controlled datasets. For very large outputs, complete pairwise RDKit validation may be too expensive and should be performed on subsets or samples.

---

## Testing

Run the full test suite:

```bash
PYTHONPATH=python pytest tests -vv
```

Current status:

```text
31 passed
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
- RDKit validation.

Test files:

```text
tests/test_graph.py
tests/test_low_level_kernels.py
tests/test_matcher.py
tests/test_pipeline.py
tests/test_validation.py
tests/test_validation_against_rdkit.py
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
PYTHONPATH=python python examples/run_pipeline.py \
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
PYTHONPATH=python python examples/run_pipeline.py \
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
- complete RDKit validation is only practical on small datasets or sampled subsets;
- JSON export for millions of matches is intentionally avoided;
- large outputs should be exported using CSV streaming;
- `--force-refine` may execute refinement on small graphs and should be considered an advanced option;
- the exact behavior depends on the available SYCL backend and device.

---

## Development Notes

Recommended workflow during development:

```bash
PYTHONPATH=python pytest tests -vv
```

Run a simple example:

```bash
PYTHONPATH=python python examples/basic_usage.py
```

Run validation example:

```bash
PYTHONPATH=python python examples/validation_usage.py
```

Run kernel-level pipeline on a small subset:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

Run full dataset pipeline:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
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

---

## Current Status

The current Python interface provides:

```text
High-level API: OK
Batch API: OK
Object-oriented API: OK
Kernel-level API: OK
RDKit validation: OK
Automated tests: 31/31 passing
Full dataset without refinement: OK
Full dataset with refinement: OK
Streaming CSV export: OK
JSON summary export: OK
```

## Additional Documentation

More detailed documentation is available in the `docs/` folder:

- [`docs/api.md`](docs/api.md): Python API reference.
- [`docs/architecture.md`](docs/architecture.md): internal architecture and pipeline flow.
- [`docs/build.md`](docs/build.md): build and environment setup.
- [`docs/benchmarks.md`](docs/benchmarks.md): benchmark commands and experimental results.