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
├── validation.py
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
- validation;
- export utilities.

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
- make the package easier to use from examples and notebooks.

---

### `config.py`

This module handles SYCL device and queue selection through `dpctl`.

Main functions include:

```python
get_sycl_queue(...)
get_default_queue()
```

Typical behavior:

```text
device="auto"  -> try GPU first, fallback to CPU
device="gpu"   -> force GPU selection
device="cpu"   -> force CPU selection
```

This allows user-facing APIs to accept simple device strings instead of requiring users to manually create SYCL queues.

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
- already constructed CSR graph dictionaries.

Important functions include:

```python
make_csr_graph(...)
toy_two_node_graph()
smarts_to_csr_from_string(...)
smarts_to_csr(...)
load_molecules(...)
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
- optionally trigger RDKit validation;
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

The implementation avoids unsafe direct reads from the internal `Candidates` object and relies on the statistics returned by the binding wrapper.

This behavior is important because the candidate buffer is stored on the device and should be accessed carefully.

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
- raw result metadata;
- optional validation metadata.

Example:

```python
print(result.summary())
print(result.explain())
```

Responsibilities:

- transform raw C++/SYCL output into Python-friendly objects;
- provide readable summaries;
- explain the executed pipeline;
- support CSV and JSON export;
- store validation metadata.

---

### `validation.py`

This module provides optional validation against RDKit.

Main function:

```python
validate_result_with_rdkit(...)
```

It compares SIGMo results with RDKit:

```python
target_mol.HasSubstructMatch(query_mol)
```

The validation result is stored in:

```python
result.validation
```

Example structure:

```python
{
    "enabled": True,
    "method": "RDKit HasSubstructMatch",
    "checked_pairs": ...,
    "agreements": ...,
    "disagreements": ...,
    "skipped": ...,
    "passed": ...
}
```

Responsibilities:

- validate SIGMo matches against a known chemistry toolkit;
- report agreements and disagreements;
- support correctness checks on small or controlled datasets.

Complete validation over very large datasets can be expensive and should be performed on subsets or samples.

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

This module is currently empty.

It is intentionally kept as a placeholder for future visualization features.

Planned possible responsibilities include:

- converting SIGMo CSR graphs to NetworkX graphs;
- drawing query and data molecules;
- visualizing molecule graphs;
- displaying query-target match pairs;
- integrating RDKit molecule drawing;
- optionally highlighting matched substructures if atom-level mappings become available.

At the moment, no public visualization API is implemented.

The file is kept in the package to make the future extension point explicit.

Possible future API examples:

```python
sigmo.visualize.draw_molecule(...)
sigmo.visualize.draw_graph(...)
sigmo.visualize.draw_match(...)
```

Current status:

```text
visualize.py exists but is empty.
No production code should depend on it yet.
```

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
validation
export
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
RDKit validation skipped for very large result.
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
6. Keep validation logic in `validation.py`.
7. Keep visualization features in `visualize.py`.
8. Add tests for every new public behavior.

---

## Testing This Package

From the repository root:

```bash
PYTHONPATH=python pytest tests -vv
```

Current expected status:

```text
31 passed
```

Important test coverage includes:

- graph conversion;
- aromatic bond stability;
- low-level kernel calls;
- high-level matching;
- batch search;
- `SIGMoMatcher`;
- `PipelineContext`;
- RDKit validation;
- CSV/JSON export.

---

## Current Status

```text
High-level API: implemented
Batch API: implemented
Object-oriented API: implemented
Kernel-level API: implemented
RDKit validation: implemented
Result summary/explain: implemented
CSV/JSON export: implemented
Large-result support: implemented in examples/run_pipeline.py
Visualization API: placeholder only
Tests: 31/31 passing
```

The package is currently suitable as a stable research-oriented Python interface over the SIGMo backend.