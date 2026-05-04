# SIGMo Python Package

This folder contains the Python interface built on top of the native SIGMo C++/SYCL binding.

The goal of this package is to provide a clean, Pythonic and research-oriented API for running SIGMo from Python, while keeping access to the low-level kernels for advanced users.

The package is designed around three main usage levels:

1. **High-level API** for users who want simple molecule matching.
2. **Configurable API** for users who want reusable matcher objects.
3. **Kernel-level API** for users who want to execute and inspect the SIGMo pipeline step by step.

---

## Package Structure

```text
python/sigmo/
├── __init__.py
├── config.py
├── graph.py
├── matcher.py
├── pipeline.py
├── result.py
├── utils.py
└── visualize.py
```

The native C++/SYCL extension is exposed through the compiled `_core` module and re-exported where needed by the Python interface.

---

## Architecture Overview

The Python package acts as a user-facing layer over the native SIGMo backend.

```text
User code
   |
   | sigmo.match()
   | sigmo.search()
   | SIGMoMatcher.run()
   v
matcher.py
   |
   | load_molecules()
   v
graph.py
   |
   | SMARTS / SMILES parsing
   | RDKit molecule conversion
   | CSR graph construction
   v
pipeline.py
   |
   | PipelineContext
   | allocate()
   | generate_signatures()
   | filter_candidates()
   | refine()
   | join()
   v
_core C++/SYCL binding
   |
   | generate_csr_signatures
   | refine_csr_signatures
   | filter_candidates
   | refine_candidates
   | join_candidates
   v
result.py
   |
   | MatchResult
   | summary()
   | explain()
   | to_csv()
   | to_json()
   v
User-readable output
```

The C++ backend performs the computational work.  
The Python package handles:

- molecule loading;
- graph conversion;
- pipeline orchestration;
- device selection;
- result formatting;
- explainability;
- export utilities;
- visualization helpers.

RDKit is used for molecule parsing and visualization support. It is not used as a public validation or benchmarking layer in the current package.

---

## Public API

The main public API is exposed from `sigmo.__init__.py`.

Typical user-facing functions and classes are:

```python
sigmo.match(...)
sigmo.search(...)
sigmo.run_isomorphism(...)
sigmo.SIGMoMatcher
sigmo.PipelineContext
sigmo.load_molecules(...)
sigmo.MatchResult
```

Low-level kernel bindings are still available for advanced users:

```python
sigmo.generate_csr_signatures(...)
sigmo.refine_csr_signatures(...)
sigmo.filter_candidates(...)
sigmo.refine_candidates(...)
sigmo.join_candidates(...)

sigmo.Signature
sigmo.Candidates
sigmo.GMCR
```

This allows the package to support both simple usage and advanced kernel-level experimentation.

---

## Module Overview

### `__init__.py`

This file defines the public package interface.

It exposes the main high-level API:

```python
from sigmo import match, search, SIGMoMatcher, PipelineContext
```

and also re-exports the low-level SIGMo kernel wrappers and native objects:

```python
Signature
Candidates
GMCR
generate_csr_signatures
refine_csr_signatures
filter_candidates
refine_candidates
join_candidates
```

This file is the entry point for users who simply run:

```python
import sigmo
```

Responsibilities:

- expose the high-level Python API;
- expose the kernel-level API;
- keep imports centralized;
- import `dpctl` before the native extension to reduce SYCL/Unified Runtime loading conflicts;
- make the package easier to use from examples and notebooks.

---

### `config.py`

This module handles SYCL device and queue selection through `dpctl`.

Main functions include:

```python
get_sycl_queue(...)
get_default_queue()
describe_queue(...)
```

Typical behavior:

```text
device="auto"  -> try CUDA GPU, Level Zero GPU, OpenCL GPU, generic GPU, then CPU
device="gpu"   -> try available GPU backends
device="cuda"  -> try CUDA GPU selectors
device="cpu"   -> try available CPU backends
```

The environment variable `SIGMO_SYCL_DEVICE` can be used to override device selection.

Example:

```python
from sigmo.config import get_sycl_queue

q = get_sycl_queue("auto")
print(q.sycl_device.name)
```

Responsibilities:

- select the execution device;
- create `dpctl.SyclQueue` objects;
- provide fallback logic;
- keep device management outside the core pipeline code.

---

### `graph.py`

This module handles molecule and graph input.

Its main responsibility is converting user-friendly molecular inputs into SIGMo-compatible CSR graph dictionaries.

Supported inputs include:

- SMILES strings;
- SMARTS strings;
- files containing SMARTS/SMILES;
- Python lists of molecular strings;
- RDKit `Mol` objects;
- NetworkX graphs;
- already constructed CSR graph dictionaries.

Important functions include:

```python
make_csr_graph(...)
toy_two_node_graph()
chemical_string_to_csr(...)
smarts_to_csr_from_string(...)
smarts_to_csr(...)
load_molecules(...)
rdkit_mol_to_csr(...)
to_networkx(...)
from_networkx(...)
```

The CSR graph format expected by SIGMo is represented as a Python dictionary:

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

The additional metadata fields such as `name`, `input` and `input_format` are used by the Python interface to produce readable results.

#### Bond label policy

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

This policy follows the original prototype behavior and avoids passing unsupported aromatic labels to the backend.

An earlier explicit aromatic label caused native crashes in some cases, so the current version prioritizes compatibility and stability.

This makes the CSR representation less chemically expressive than RDKit's full SMARTS semantics, but keeps the SIGMo backend stable.

Responsibilities:

- parse molecular inputs with RDKit;
- build CSR graph dictionaries;
- preserve molecule names and original input strings;
- support both high-level and batch workflows;
- avoid exposing low-level CSR construction to ordinary users.

---

### `matcher.py`

This module provides the main user-facing API.

Main functions and classes:

```python
match(...)
search(...)
run_isomorphism(...)
SIGMoMatcher
```

#### `match()`

High-level entry point for matching a single query against a single target.

Example:

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
```

Flow:

```text
query string + target string
        |
        v
load_molecules()
        |
        v
run_isomorphism()
        |
        v
PipelineContext.run()
        |
        v
MatchResult
```

#### `search()`

Batch entry point for matching multiple queries against a database.

Example:

```python
result = sigmo.search(
    queries="benchmarks/datasets/query.smarts",
    database="benchmarks/datasets/data.smarts",
    iterations=6,
    device="auto",
)
```

#### `run_isomorphism()`

Intermediate API for users who already have CSR graphs.

Example:

```python
result = sigmo.run_isomorphism(
    q_graphs,
    d_graphs,
    iterations=6,
    find_first=True,
)
```

#### `SIGMoMatcher`

Object-oriented API for configurable workflows.

Example:

```python
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
```

Responsibilities:

- expose simple matching APIs;
- load inputs through `graph.py`;
- create and run `PipelineContext`;
- return `MatchResult` objects.

---

### `pipeline.py`

This module contains the kernel-level orchestration logic.

The central class is:

```python
PipelineContext
```

`PipelineContext` manages:

- query graphs;
- data graphs;
- SYCL queue;
- SIGMo `Signature`;
- SIGMo `Candidates`;
- SIGMo `GMCR`;
- kernel execution steps;
- warnings;
- errors;
- raw join results.

Main methods:

```python
allocate()
generate_signatures()
filter_candidates()
refine()
join()
run()
```

---

## PipelineContext Flow

The full execution flow is:

```text
PipelineContext.run()
   |
   | allocate()
   |   -> create Signature
   |   -> create Candidates
   |   -> create GMCR
   |
   | generate_signatures()
   |   -> generate_csr_signatures(query)
   |   -> generate_csr_signatures(data)
   |
   | filter_candidates()
   |   -> filter_candidates(...)
   |   -> save last_candidates_count
   |
   | refine()
   |   -> refine_csr_signatures(query)
   |   -> refine_csr_signatures(data)
   |   -> refine_candidates(...)
   |   -> repeat for N iterations
   |
   | join()
   |   -> join_candidates(...)
   |
   v
build_match_result()
```

The refinement logic is intentionally explicit and deterministic.

For each iteration:

```text
1. refine query signatures
2. refine data signatures
3. refine candidates
4. read candidates_count from returned stats
```

The implementation keeps the same native `Signature`, `Candidates` and `GMCR` objects alive across all pipeline steps. This preserves the stateful execution model expected by the native SIGMo backend.

Candidate counts and match statistics are read through safe statistics returned by the binding wrapper, rather than by reconstructing pipeline state.

Responsibilities:

- orchestrate native kernels safely;
- preserve the stateful SIGMo pipeline;
- record timing and kernel statistics;
- support both safe and forced refinement modes;
- return a structured `MatchResult`.

---

### Safe refinement policy

`PipelineContext.run()` supports a safety option:

```python
disable_refine_for_small_graphs=True
```

When enabled, refinement can be disabled automatically if very small graphs are detected.

This protects high-level users from possible backend instability on micro-molecules.

Advanced users can force refinement through lower-level calls or command-line options in `examples/run_pipeline.py`.

---

### Force refinement mode

Advanced workflows can still execute all kernels, including refinement, even when small graphs are present.

This is useful for benchmarking and debugging.

In command-line examples, this is exposed through:

```bash
--force-refine
```

---

### `result.py`

This module defines the structured result model.

Main classes and functions:

```python
KernelStep
Match
MatchResult
build_match_result(...)
```

`MatchResult` provides:

```python
result.summary()
result.explain()
result.to_records()
result.to_dataframe()
result.to_csv(...)
result.to_json(...)
```

The result object stores:

- total matches;
- query count;
- data count;
- device name;
- requested refinement iterations;
- executed refinement iterations;
- kernel steps;
- warnings;
- errors;
- raw result metadata.

Example:

```python
print(result.summary())
print(result.explain())
```

Responsibilities:

- transform raw C++/SYCL output into Python-friendly objects;
- provide readable summaries;
- explain the executed pipeline;
- support CSV and JSON export.

---

### `utils.py`

This module contains utility functions and compatibility helpers.

It is currently used for result formatting and legacy-style helpers.

As the package evolved toward `MatchResult`, most user-facing formatting moved into `result.py`.

Responsibilities:

- keep small helper functions;
- preserve compatibility with earlier interface experiments;
- avoid cluttering core modules.

---

### `visualize.py`

This module provides optional visualization utilities based on RDKit and, when available, NetworkX/Matplotlib.

Main functions include:

```python
mol_from_input(...)
draw_molecule(...)
draw_match_pair(...)
to_networkx(...)
draw_graph(...)
```

Supported visualization tasks include:

- drawing individual molecules;
- drawing query-target molecule pairs;
- highlighting query substructures inside target molecules using RDKit;
- converting SIGMo CSR graphs to NetworkX graphs;
- drawing internal SIGMo CSR graphs for debugging.

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

Important note: match-pair highlighting is RDKit-based and is used only for visualization. Current SIGMo results are pair-level results, meaning that SIGMo reports that query graph `i` matches data graph `j`, but it does not expose an atom-level mapping for drawing.

Responsibilities:

- provide molecule rendering helpers;
- provide CSR graph debug visualization;
- keep visualization separate from the core native pipeline;
- support examples and documentation without changing the backend.

---

## Native Binding Boundary

The Python package is built on top of the native `_core` binding.

The `_core` layer exposes the actual C++/SYCL objects and kernels, such as:

```python
Signature
Candidates
GMCR
generate_csr_signatures
refine_csr_signatures
filter_candidates
refine_candidates
join_candidates
```

The Python interface does not reimplement the SIGMo algorithm.

Instead, it handles:

```text
input conversion
pipeline orchestration
result formatting
export
visualization helpers
error/warning reporting
```

The computational work remains inside the native backend.

---

## Typical Internal Flow

### High-level single match

```text
sigmo.match()
   |
   v
matcher.match()
   |
   v
graph.load_molecules()
   |
   v
matcher.run_isomorphism()
   |
   v
PipelineContext.run()
   |
   v
build_match_result()
   |
   v
MatchResult
```

### Batch search

```text
sigmo.search()
   |
   v
load query file
load data file
   |
   v
convert each molecule to CSR
   |
   v
PipelineContext.run()
   |
   v
MatchResult
```

### Advanced pipeline

```text
query_graphs = sigmo.load_molecules(...)
data_graphs = sigmo.load_molecules(...)

ctx = sigmo.PipelineContext(...)
ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(...)
ctx.join(...)

result = build_match_result(...)
```

### Visualization flow

```text
SMILES / SMARTS / CSR graph
   |
   v
sigmo.visualize
   |
   | draw_molecule()
   | draw_match_pair()
   | draw_graph()
   v
PNG image / NetworkX graph
```

---

## Error and Warning Strategy

The package distinguishes between:

- Python-level errors;
- backend/runtime warnings;
- explainability warnings;
- large-result warnings.

Examples:

```text
Refinement disabled because small graphs were detected.
Force refine enabled on small graphs.
Large result detected; matches are not materialized in MatchResult.
```

Warnings are stored in:

```python
result.warnings
```

Errors are stored in:

```python
result.errors
```

This makes the result object self-descriptive and suitable for downstream reporting.

---

## Large Result Strategy

For normal result sizes, `MatchResult` can materialize all matches and export them directly.

For very large result sets, such as millions of matches, materializing every match as a Python object can exhaust memory.

The large-result strategy is implemented mainly in `examples/run_pipeline.py`, but it relies on the structured output model provided by `result.py`.

The large-result behavior is:

```text
- preserve total match count;
- avoid materializing every match in MatchResult;
- write matches to CSV in streaming mode;
- write only summary/statistics to JSON;
- show only a limited terminal preview.
```

This makes full dataset execution practical even when the number of matches is very large.

---

## Development Guidelines

When adding new functionality:

1. Keep the native binding thin.
2. Keep chemistry input handling in `graph.py`.
3. Keep user-facing APIs in `matcher.py`.
4. Keep pipeline execution in `pipeline.py`.
5. Keep result formatting in `result.py`.
6. Keep visualization features in `visualize.py`.
7. Add tests for every new public behavior.

---

## Testing This Package

From the repository root, after building the native extension and installing the package in editable mode:

```bash
pytest tests -vv
```

Recommended setup:

```bash
./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
pytest tests -vv
```

Important test coverage includes:

- graph conversion;
- aromatic bond stability;
- low-level kernel calls;
- high-level matching;
- batch search;
- `SIGMoMatcher`;
- `PipelineContext`;
- CSV/JSON export;
- visualization utilities.

---

## Current Status

```text
High-level API: implemented
Batch API: implemented
Object-oriented API: implemented
Kernel-level API: implemented
Result summary/explain: implemented
CSV/JSON export: implemented
Large-result support: implemented in examples/run_pipeline.py
Visualization API: implemented
Tests: passing
```
