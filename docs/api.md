# SIGMo Python API Reference

This document describes the public Python API exposed by the SIGMo Python interface.

The API is organized into multiple abstraction levels:

1. **High-level API**: simple functions such as `sigmo.match()` and `sigmo.search()`.
2. **Object-oriented API**: reusable matcher object through `SIGMoMatcher`.
3. **Pipeline API**: explicit kernel-level orchestration through `PipelineContext`.
4. **Low-level kernel API**: direct access to native SIGMo C++/SYCL bindings.
5. **Visualization API**: optional helpers for drawing molecules, CSR graphs and query-target pairs.

Most users should start from the high-level API.  
Advanced users can use `PipelineContext` or the low-level kernels when they need more control.

---

## Importing the Package

During development, build the native extension and install the package in editable mode from the repository root:

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh

./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
```

Then the package can be imported normally:

```python
import sigmo
```

The package exposes the main API directly from the `sigmo` namespace:

```python
sigmo.match
sigmo.search
sigmo.run_isomorphism
sigmo.SIGMoMatcher
sigmo.PipelineContext
sigmo.load_molecules
sigmo.MatchResult
```

Low-level bindings are also exposed:

```python
sigmo.Signature
sigmo.Candidates
sigmo.GMCR

sigmo.generate_csr_signatures
sigmo.refine_csr_signatures
sigmo.filter_candidates
sigmo.refine_candidates
sigmo.join_candidates
```

Visualization helpers are available from the `sigmo.visualize` module:

```python
from sigmo.visualize import draw_molecule, draw_match_pair, draw_graph
```

---

## API Layers

The recommended API choice depends on the use case.

| User need | Recommended API |
|---|---|
| Match one query against one target | `sigmo.match()` |
| Search many queries against a dataset | `sigmo.search()` |
| Reuse matcher configuration | `sigmo.SIGMoMatcher` |
| Execute pipeline step by step | `sigmo.PipelineContext` |
| Call native kernels directly | Low-level kernel API |
| Draw molecules, CSR graphs or match pairs | `sigmo.visualize` |

---

# High-Level API

## `sigmo.match()`

High-level function for matching a single query molecule against a single target molecule.

### Typical signature

```python
sigmo.match(
    query,
    target,
    *,
    input_format="auto",
    iterations=1,
    find_first=True,
    device="auto",
    queue=None,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `query` | `str` | Query molecule or substructure, usually SMARTS or SMILES |
| `target` | `str` | Target molecule, usually SMILES or SMARTS |
| `input_format` | `str` | `"auto"`, `"smiles"` or `"smarts"` |
| `iterations` | `int` | Number of refinement iterations |
| `find_first` | `bool` | Stop at first match per pair when supported |
| `device` | `str` | `"auto"`, `"gpu"`, `"cpu"`, `"cuda"` or an explicit SYCL selector |
| `queue` | optional | Existing SYCL queue |

### Returns

```python
MatchResult
```

### Example

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

### Notes

`sigmo.match()` is the easiest entry point. It hides:

- molecule parsing;
- graph conversion;
- SYCL queue creation;
- SIGMo structure allocation;
- kernel orchestration.

RDKit is used for parsing molecular strings, but this function does not perform RDKit-based validation or benchmarking.

---

## `sigmo.search()`

High-level batch search function.

It matches multiple query graphs against multiple data graphs.

### Typical signature

```python
sigmo.search(
    queries,
    database,
    *,
    input_format="auto",
    iterations=1,
    find_first=True,
    device="auto",
    queue=None,
    strict=False,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `queries` | file path, list, graph list | Query molecules or query graph dataset |
| `database` | file path, list, graph list | Target/data molecules or graph dataset |
| `input_format` | `str` | `"auto"`, `"smiles"` or `"smarts"` |
| `iterations` | `int` | Number of refinement iterations |
| `find_first` | `bool` | Stop at first match per pair when supported |
| `device` | `str` | `"auto"`, `"gpu"`, `"cpu"` or device selector |
| `queue` | optional | Existing SYCL queue |
| `strict` | `bool` | Whether invalid inputs should raise errors |

### Returns

```python
MatchResult
```

### Example with files

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

### Example with lists

```python
import sigmo

result = sigmo.search(
    queries=["CC", "CO"],
    database=["CCC", "CCO"],
    input_format="smiles",
    iterations=0,
)

print(result.summary())
```

### Notes

`sigmo.search()` is the recommended API for batch workloads.

For very large datasets, consider using `examples/run_pipeline.py`, which includes large-result handling and CSV streaming.

---

## `sigmo.run_isomorphism()`

Intermediate API for users who already have SIGMo-compatible CSR graphs.

### Typical signature

```python
sigmo.run_isomorphism(
    q_graphs,
    d_graphs,
    *,
    queue=None,
    device="auto",
    find_first=True,
    iterations=1,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `q_graphs` | `list[dict]` | Query CSR graphs |
| `d_graphs` | `list[dict]` | Data CSR graphs |
| `queue` | optional | Existing SYCL queue |
| `device` | `str` | Device selection string |
| `find_first` | `bool` | Stop at first match per pair when supported |
| `iterations` | `int` | Number of refinement iterations |

### Returns

```python
MatchResult
```

### Example

```python
import sigmo

q_graphs = sigmo.load_molecules(["CC"], input_format="smiles")
d_graphs = sigmo.load_molecules(["CCC"], input_format="smiles")

result = sigmo.run_isomorphism(
    q_graphs,
    d_graphs,
    iterations=0,
    find_first=True,
)

print(result.summary())
```

### Notes

This function is useful when input conversion has already been performed separately.

---

# Object-Oriented API

## `sigmo.SIGMoMatcher`

Reusable matcher class.

This class stores configuration and can be reused across runs.

### Constructor

```python
matcher = sigmo.SIGMoMatcher(
    device="auto",
    queue=None,
    iterations=1,
    find_first=True,
    input_format="auto",
    strict=False,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `device` | `str` | `"auto"`, `"gpu"`, `"cpu"` or device selector |
| `queue` | optional | Existing SYCL queue |
| `iterations` | `int` | Default refinement iterations |
| `find_first` | `bool` | Default join behavior |
| `input_format` | `str` | Default input format |
| `strict` | `bool` | Whether invalid inputs should raise errors |

---

## `SIGMoMatcher.run()`

Runs the configured matcher.

### Typical signature

```python
matcher.run(
    queries=None,
    database=None,
    *,
    iterations=None,
    find_first=None,
)
```

### Example

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

### Reusing the matcher

```python
matcher = sigmo.SIGMoMatcher(iterations=0)

result1 = matcher.run(
    queries=["CC"],
    database=["CCC"],
)

result2 = matcher.run(
    queries=["CO"],
    database=["CCO"],
)
```

### Stored attributes

After execution, the matcher stores:

```python
matcher.query_graphs
matcher.data_graphs
matcher.last_context
matcher.last_result
```

### Notes

`SIGMoMatcher` is useful for applications where the same configuration is reused multiple times.

---

# Graph Loading API

## `sigmo.load_molecules()`

Loads molecules or graphs and converts them into SIGMo-compatible CSR graph dictionaries.

### Typical usage

```python
graphs = sigmo.load_molecules(
    "benchmarks/datasets/query.smarts",
    input_format="auto",
)
```

or:

```python
graphs = sigmo.load_molecules(
    ["CC", "CCC", "c1ccccc1"],
    input_format="smiles",
)
```

### Returns

```python
list[dict]
```

Each graph dictionary has fields such as:

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

### Supported input types

| Input | Description |
|---|---|
| `str` path | File containing SMARTS/SMILES |
| `str` molecule | Single SMARTS/SMILES string |
| `list[str]` | List of molecular strings |
| `list[dict]` | Already converted CSR graphs |
| `list[RDKit Mol]` | Existing RDKit molecule objects |

### Notes

Invalid molecules may be skipped unless strict behavior is enabled by the caller.

---

## CSR Graph Format

SIGMo expects graph data in CSR form.

Required fields:

| Field | Description |
|---|---|
| `row_offsets` | CSR row offsets |
| `column_indices` | CSR adjacency indices |
| `node_labels` | Atom/node labels |
| `edge_labels` | Bond/edge labels |
| `num_nodes` | Number of nodes |

Additional metadata fields:

| Field | Description |
|---|---|
| `name` | Molecule or generated graph name |
| `input` | Original SMARTS/SMILES string |
| `input_format` | Parsed input format |
| `original_index` | Original index in the input source, when available |

---

## Bond Label Policy

The current RDKit bond conversion uses:

```python
int(bond.GetBondTypeAsDouble())
```

Mapping:

| RDKit bond type | Label |
|---|---:|
| Single | `1` |
| Double | `2` |
| Triple | `3` |
| Aromatic | `1` |

Aromatic bonds are currently collapsed to label `1` for backend compatibility.

This behavior follows the original prototype and avoids unsupported labels that previously caused native crashes.

This means that the CSR representation is stable for the SIGMo backend, but it does not preserve the full chemical semantics of RDKit SMARTS.

---

# Pipeline API

## `sigmo.PipelineContext`

Stateful object for executing the SIGMo pipeline step by step.

### Constructor

```python
ctx = sigmo.PipelineContext(
    query_graphs=query_graphs,
    data_graphs=data_graphs,
    device="auto",
    queue=None,
)
```

### Main attributes

| Attribute | Description |
|---|---|
| `query_graphs` | Query CSR graphs |
| `data_graphs` | Data CSR graphs |
| `queue` | SYCL queue |
| `device_name` | Selected device name |
| `signature` | SIGMo `Signature` object |
| `candidates` | SIGMo `Candidates` object |
| `gmcr` | SIGMo `GMCR` object |
| `steps` | List of recorded kernel steps |
| `warnings` | Execution warnings |
| `errors` | Execution errors |
| `raw_join_result` | Raw join output |
| `executed_iterations` | Number of refinement iterations executed |
| `last_candidates_count` | Last known candidate count |

---

## `PipelineContext.allocate()`

Allocates native SIGMo structures:

```text
Signature
Candidates
GMCR
```

### Example

```python
ctx.allocate()
```

---

## `PipelineContext.generate_signatures()`

Runs signature generation for both query and data graphs.

Internally calls:

```python
generate_csr_signatures(..., "query")
generate_csr_signatures(..., "data")
```

### Example

```python
ctx.generate_signatures()
```

Recorded steps:

```text
generate_query_signatures
generate_data_signatures
```

---

## `PipelineContext.filter_candidates()`

Runs the initial candidate filtering kernel.

### Example

```python
filter_stats = ctx.filter_candidates()
print(filter_stats["candidates_count"])
```

Internally calls:

```python
filter_candidates(
    queue,
    query_graphs,
    data_graphs,
    signature,
    candidates,
)
```

The candidate count is stored in:

```python
ctx.last_candidates_count
```

---

## `PipelineContext.refine()`

Runs iterative refinement.

### Typical signature

```python
ctx.refine(
    iterations=6,
    start_view_size=1,
    stop_on_fixed_point=True,
)
```

or simply:

```python
ctx.refine(6)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `iterations` | `int` | Maximum number of refinement iterations |
| `start_view_size` | `int` | Initial view size |
| `stop_on_fixed_point` | `bool` | Stop if candidate count no longer changes |

### Internal flow

For each iteration:

```text
1. refine_csr_signatures(query)
2. refine_csr_signatures(data)
3. refine_candidates()
```

The candidate count is read from the statistics returned by `refine_candidates()`.

### Example

```python
ctx.refine(
    6,
    start_view_size=1,
    stop_on_fixed_point=True,
)
```

### Notes

The implementation avoids unsafe direct reads from the internal `Candidates` object and relies on the statistics returned by the kernel wrapper.

---

## `PipelineContext.join()`

Runs the final join/isomorphism phase.

### Typical signature

```python
ctx.join(find_first=True)
```

### Example

```python
raw_join = ctx.join(find_first=True)
print(raw_join["num_matches"])
```

Internally calls:

```python
join_candidates(
    queue,
    query_graphs,
    data_graphs,
    candidates,
    gmcr,
    find_first,
)
```

The result is stored in:

```python
ctx.raw_join_result
```

---

## `PipelineContext.run()`

Runs the full pipeline and returns a `MatchResult`.

### Typical signature

```python
ctx.run(
    iterations=1,
    find_first=True,
    disable_refine_for_small_graphs=True,
    min_refine_nodes=6,
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `iterations` | `int` | Requested refinement iterations |
| `find_first` | `bool` | Join behavior |
| `disable_refine_for_small_graphs` | `bool` | Safety option for micro-graphs |
| `min_refine_nodes` | `int` | Minimum graph size for safe refinement |

### Example

```python
result = ctx.run(
    iterations=6,
    find_first=True,
)

print(result.summary())
```

### Safe refinement policy

By default, refinement may be disabled if very small graphs are detected.

This prevents possible native instability for high-level users.

Advanced users can call `ctx.refine()` directly or use `examples/run_pipeline.py --force-refine`.

---

# Result API

## `sigmo.MatchResult`

Structured result object returned by high-level APIs.

### Main fields

| Field | Description |
|---|---|
| `total_matches` | Total number of matches |
| `matches` | List of materialized match records |
| `query_count` | Number of query graphs |
| `data_count` | Number of data graphs |
| `device` | Device used for execution |
| `requested_iterations` | Requested refinement iterations |
| `executed_iterations` | Actually executed refinement iterations |
| `steps` | Recorded kernel steps |
| `warnings` | Execution warnings |
| `errors` | Execution errors |
| `raw_result` | Raw backend result metadata |

---

## `MatchResult.summary()`

Returns a readable execution summary.

### Example

```python
print(result.summary())
```

Typical output includes:

```text
Status
Device
Queries
Database graphs
Matches found
Refinement iterations
Kernel steps
Warnings
Errors
```

---

## `MatchResult.explain()`

Returns a step-by-step explanation of the pipeline execution.

### Example

```python
print(result.explain())
```

This is useful for explainability and reporting.

---

## `MatchResult.to_records()`

Returns a list of match records.

### Example

```python
records = result.to_records()
```

Each record typically contains:

```text
query_index
query_name
query_input
data_index
data_name
data_input
```

### Notes

For very large result sets, avoid materializing all records in memory. Use streaming export from `examples/run_pipeline.py` instead.

---

## `MatchResult.to_dataframe()`

Returns a pandas DataFrame if pandas is installed.

### Example

```python
df = result.to_dataframe()
```

### Notes

This requires pandas.

For very large result sets, avoid this method because it materializes all matches in memory.

---

## `MatchResult.to_csv()`

Exports matches to CSV.

### Example

```python
result.to_csv("matches.csv")
```

### Notes

This is suitable for small and medium result sets.

For millions of matches, prefer the streaming CSV export implemented in `examples/run_pipeline.py`.

---

## `MatchResult.to_json()`

Exports the result to JSON.

### Example

```python
result.to_json("matches.json")
```

### Notes

This is suitable for small and medium result sets.

For large result sets, prefer JSON summary export from `examples/run_pipeline.py`.

---

# Device API

## `sigmo.config.get_sycl_queue()`

Creates a SYCL queue from a device selector.

### Example

```python
from sigmo.config import get_sycl_queue

q = get_sycl_queue("auto")
print(q.sycl_device.name)
```

Supported selectors include:

```text
auto
gpu
cpu
cuda
cuda:gpu
cuda:gpu:0
level_zero:gpu
opencl:gpu
opencl:cpu
```

---

## `sigmo.config.get_default_queue()`

Returns the default queue using the package default device selection policy.

### Example

```python
from sigmo.config import get_default_queue

q = get_default_queue()
```

This is used by tests and examples when no explicit queue is provided.

---

# Low-Level Kernel API

Advanced users can call the native kernel wrappers directly.

These functions are exposed from the `sigmo` namespace.

---

## `sigmo.generate_csr_signatures()`

Generates structural signatures for query or data graphs.

### Example

```python
sigmo.generate_csr_signatures(
    queue,
    graphs,
    signature,
    "data",
)
```

---

## `sigmo.refine_csr_signatures()`

Refines structural signatures.

### Example

```python
sigmo.refine_csr_signatures(
    queue,
    graphs,
    signature,
    "query",
    view_size,
)
```

---

## `sigmo.filter_candidates()`

Runs candidate filtering.

### Example

```python
sigmo.filter_candidates(
    queue,
    query_graphs,
    data_graphs,
    signature,
    candidates,
)
```

---

## `sigmo.refine_candidates()`

Runs candidate refinement.

### Example

```python
sigmo.refine_candidates(
    queue,
    query_graphs,
    data_graphs,
    signature,
    candidates,
)
```

---

## `sigmo.join_candidates()`

Runs the final join/isomorphism phase.

### Example

```python
sigmo.join_candidates(
    queue,
    query_graphs,
    data_graphs,
    candidates,
    gmcr,
    True,
)
```

---

## Native Objects

The package also exposes native SIGMo objects:

```python
sigmo.Signature
sigmo.Candidates
sigmo.GMCR
```

These are intended for advanced users and internal pipeline code.

Most users should not need to instantiate them directly.

---

# Visualization API

Visualization helpers are implemented in:

```python
sigmo.visualize
```

They are optional and are not required by the core SIGMo pipeline.

---

## `mol_from_input()`

Converts a SMILES/SMARTS string or RDKit Mol into an RDKit Mol.

### Example

```python
from sigmo.visualize import mol_from_input

mol = mol_from_input("CCO", input_format="smiles")
```

---

## `draw_molecule()`

Draws a single molecule and optionally saves it to disk.

### Example

```python
from sigmo.visualize import draw_molecule

draw_molecule(
    "CCO",
    input_format="smiles",
    output_path="examples/outputs/molecule.png",
    legend="Ethanol",
)
```

---

## `draw_match_pair()`

Draws a query-target pair.

If `highlight=True`, RDKit is used to highlight one query substructure occurrence inside the target molecule.

### Example

```python
from sigmo.visualize import draw_match_pair

draw_match_pair(
    "CC",
    "CCC",
    query_format="smiles",
    target_format="smiles",
    output_path="examples/outputs/match_pair.png",
    legends=("Query", "Target"),
    highlight=True,
)
```

### Important note

The highlight is RDKit-based and is used only for visualization.  
Current SIGMo results are pair-level results: SIGMo reports that query graph `i` matches data graph `j`, but it does not expose an atom-level mapping for drawing.

---

## `to_networkx()`

Converts a SIGMo CSR graph dictionary into a NetworkX graph.

### Example

```python
import sigmo
from sigmo.visualize import to_networkx

graphs = sigmo.load_molecules(["CCO"], input_format="smiles")
G = to_networkx(graphs[0])
```

---

## `draw_graph()`

Draws an internal SIGMo CSR graph using NetworkX and Matplotlib.

This is intended for debugging the CSR representation, not for chemically accurate molecule rendering.

### Example

```python
import sigmo
from sigmo.visualize import draw_graph

graphs = sigmo.load_molecules(["CCO"], input_format="smiles")

draw_graph(
    graphs[0],
    output_path="examples/outputs/csr_graph_small.png",
    layout="spring",
)
```

Run the full visualization example:

```bash
python examples/visualization_usage.py
```

---

# Recommended Usage Patterns

## Simple matching

```python
result = sigmo.match("CC", "CCC", input_format="smiles")
```

## Batch search

```python
result = sigmo.search(
    "benchmarks/datasets/query.smarts",
    "benchmarks/datasets/data.smarts",
)
```

## Reusable matcher

```python
matcher = sigmo.SIGMoMatcher(iterations=6)
result = matcher.run(queries, database)
```

## Step-by-step pipeline

```python
ctx = sigmo.PipelineContext(q_graphs, d_graphs)
ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(6)
ctx.join()
```

## Full dataset with streaming export

Use:

```bash
python examples/run_pipeline.py   --query-limit -1   --data-limit -1   --iterations 6   --force-refine   --max-print-matches 0   --csv examples/outputs/matches_full_refine.csv   --json examples/outputs/matches_full_refine_summary.json
```

---

# API Stability Notes

The currently stable public APIs are:

```python
sigmo.match
sigmo.search
sigmo.run_isomorphism
sigmo.SIGMoMatcher
sigmo.PipelineContext
sigmo.load_molecules
sigmo.MatchResult
```

The low-level kernel API is available but should be considered advanced.

The visualization API is available, but it should be considered an optional utility layer rather than part of the core matching pipeline.

---

# Current Status

```text
High-level API: available
Batch API: available
Object-oriented API: available
Pipeline API: available
Low-level kernel API: available
Result summary/explain: available
CSV/JSON export: available
Visualization API: available
Tests: passing
```
