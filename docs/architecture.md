# SIGMo Python Interface Architecture

This document describes the internal architecture of the Python interface built on top of the native SIGMo C++/SYCL backend.

The goal of this interface is to expose SIGMo through a Pythonic, modular and explainable API while preserving access to the original low-level kernels for advanced users.

---

## Design Goals

The Python interface was designed with the following goals:

- keep the original SIGMo backend unmodified;
- expose high-level functions for chemists and researchers;
- support batch molecular search on SMARTS/SMILES datasets;
- preserve kernel-level access for debugging and benchmarking;
- provide explainable outputs through summaries and execution traces;
- support optional validation against RDKit;
- handle very large result sets without exhausting memory;
- keep the binding thin and move usability logic to Python.

The resulting design separates the project into two conceptual layers:

```text
Native SIGMo backend
    C++/SYCL kernels, memory structures, device execution

Python interface
    input handling, pipeline orchestration, result formatting,
    validation, examples, tests, large-output management
```

---

## High-Level Architecture

At a high level, the interface follows this flow:

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
   | build_match_result()
   | MatchResult
   | summary()
   | explain()
   | to_csv()
   | to_json()
   v
User-readable output
```

The main idea is that Python does not reimplement the SIGMo algorithm.  
Instead, Python coordinates the backend kernels and provides a more usable interface around them.

---

## Main Package Layout

The Python package is located in:

```text
python/sigmo/
```

Current structure:

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

Each module has a specific responsibility.

| Module | Responsibility |
|---|---|
| `__init__.py` | Public API exports |
| `config.py` | SYCL device and queue selection |
| `graph.py` | SMARTS/SMILES loading and CSR graph conversion |
| `matcher.py` | High-level and object-oriented matching API |
| `pipeline.py` | Kernel-level pipeline orchestration |
| `result.py` | Structured result objects and output formatting |
| `validation.py` | Optional RDKit-based validation |
| `utils.py` | Utility and compatibility helpers |
| `visualize.py` | Placeholder for future visualization features |

---

## Public API Layers

The interface exposes three usage levels.

---

### Level 1: High-Level API

This level is intended for users who want a simple function call.

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

This hides all low-level details such as:

- SYCL queues;
- CSR graph construction;
- `Signature`;
- `Candidates`;
- `GMCR`;
- kernel calls.

The main functions are:

```python
sigmo.match(...)
sigmo.search(...)
```

---

### Level 2: Configurable Object-Oriented API

This level is intended for users who want reusable matcher objects.

Example:

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
```

The object-oriented API stores:

- matcher configuration;
- query graphs;
- data graphs;
- last execution context;
- last result.

This is useful for larger Python applications and notebooks.

---

### Level 3: Kernel-Level API

This level is intended for advanced users, debugging, benchmarking and research experiments.

Example:

```python
import sigmo
from sigmo.result import build_match_result

query_graphs = sigmo.load_molecules("benchmarks/datasets/query.smarts")
data_graphs = sigmo.load_molecules("benchmarks/datasets/data.smarts")

ctx = sigmo.PipelineContext(
    query_graphs=query_graphs,
    data_graphs=data_graphs,
    device="gpu",
)

ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(6, start_view_size=1, stop_on_fixed_point=True)
ctx.join(find_first=True)

result = build_match_result(
    ctx.raw_join_result,
    query_graphs,
    data_graphs,
    steps=ctx.steps,
    warnings=ctx.warnings,
    errors=ctx.errors,
    device=ctx.device_name,
    requested_iterations=6,
    executed_iterations=ctx.executed_iterations,
)
```

This API exposes the actual SIGMo execution phases.

---

## Input Flow

The input flow starts from user-provided molecular strings or files.

Supported input types include:

- SMILES strings;
- SMARTS strings;
- files containing SMARTS/SMILES;
- Python lists of molecular strings;
- already constructed CSR graph dictionaries.

The main input function is:

```python
sigmo.load_molecules(...)
```

Internally, it delegates to `graph.py`.

---

### Molecule Loading Flow

```text
Input file / string / list
   |
   v
load_molecules()
   |
   v
RDKit parsing
   |
   v
atom labels + bond labels
   |
   v
CSR graph dictionary
   |
   v
query_graphs / data_graphs
```

A CSR graph is represented as a Python dictionary:

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

The fields required by the backend are:

```text
row_offsets
column_indices
node_labels
edge_labels
num_nodes
```

The additional fields are used by the Python interface to produce readable outputs:

```text
name
input
input_format
original_index
```

---

## Bond Label Policy

Bond labels are generated using RDKit bond information.

The current policy is:

```python
int(bond.GetBondTypeAsDouble())
```

This maps bond types as follows:

```text
single bond    -> 1
double bond    -> 2
triple bond    -> 3
aromatic bond  -> int(1.5) = 1
```

This choice was made for backend compatibility.

An alternative explicit aromatic label was tested during development, but it caused native backend instability in some cases. The current policy follows the original prototype behavior and avoids passing unsupported labels to the C++/SYCL kernels.

Future versions may add a configurable aromatic policy if explicit aromatic support is validated in the backend.

---

## Device Selection Flow

Device and queue management is handled in `config.py`.

The user can request:

```python
device="auto"
device="gpu"
device="cpu"
```

The typical flow is:

```text
device="auto"
   |
   | try GPU
   | if unavailable, try CPU
   v
dpctl.SyclQueue
```

The selected queue is then passed to `PipelineContext` and used by all native kernel calls.

---

## PipelineContext Architecture

`PipelineContext` is the central stateful object in the Python interface.

It stores:

```text
query_graphs
data_graphs
queue
device_name
signature
candidates
gmcr
steps
warnings
errors
raw_join_result
executed_iterations
last_candidates_count
```

Its main role is to orchestrate the backend kernels in a reproducible order.

---

## Pipeline Execution Flow

The complete pipeline is:

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
   |   -> save candidates_count
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

---

## Allocation Phase

The allocation phase creates the native SIGMo data structures:

```text
Signature
Candidates
GMCR
```

These are created using the selected SYCL queue.

This stage prepares the memory structures required by later kernels.

---

## Signature Generation Phase

The signature generation phase calls:

```python
generate_csr_signatures(queue, query_graphs, signature, "query")
generate_csr_signatures(queue, data_graphs, signature, "data")
```

This computes initial structural signatures for both query and data graphs.

The execution is recorded as kernel steps:

```text
generate_query_signatures
generate_data_signatures
```

---

## Candidate Filtering Phase

The candidate filtering phase calls:

```python
filter_candidates(
    queue,
    query_graphs,
    data_graphs,
    signature,
    candidates,
)
```

This produces an initial candidate set between query and data graphs.

The returned statistics include a candidate count, which is stored as:

```python
self.last_candidates_count
```

This value is used later to track refinement progress.

---

## Refinement Phase

The refinement phase is explicit and deterministic.

For each iteration, the Python interface executes:

```text
1. refine_csr_signatures(query)
2. refine_csr_signatures(data)
3. refine_candidates()
```

In pseudocode:

```python
for i in range(iterations):
    view_size = start_view_size + i

    refine_csr_signatures(queue, query_graphs, signature, "query", view_size)
    queue.wait()

    refine_csr_signatures(queue, data_graphs, signature, "data", view_size)
    queue.wait()

    refine_candidates(queue, query_graphs, data_graphs, signature, candidates)
    queue.wait()
```

The important design decision is that the Python pipeline reads the candidate count from the statistics returned by `refine_candidates()`.

It avoids unsafe direct reads from the internal `Candidates` object.

This was introduced to keep the pipeline stable when candidate buffers live on the device.

---

## Safe Refinement Policy

By default, high-level pipeline execution may disable refinement for very small graphs.

This behavior is controlled by:

```python
disable_refine_for_small_graphs=True
min_refine_nodes=6
```

The reason is that some backend configurations may be unstable on micro-molecules.

In safe mode:

```text
if any graph has fewer than min_refine_nodes:
    skip refinement
    continue with filter + join
```

This behavior is intended for user-facing stability.

---

## Force Refinement

Advanced users can force refinement even when small graphs are detected.

This is exposed in the command-line example through:

```bash
--force-refine
```

In this case, all refinement kernels are executed.

This mode is useful for benchmarking and kernel-level testing, but it is considered advanced because it bypasses the safety policy.

---

## Join Phase

The join phase calls:

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

This is the final isomorphism/matching phase.

It returns a raw result containing fields such as:

```text
num_matches
execution_time
num_query_graph
num_data_graph
matches_dict
```

The `matches_dict` maps query indices to lists of data graph indices.

For large datasets, this dictionary may contain millions of matches.

---

## Result Construction

Raw backend output is converted into a structured `MatchResult`.

The conversion is handled by:

```python
build_match_result(...)
```

`MatchResult` stores:

```text
total_matches
matches
query_count
data_count
device
requested_iterations
executed_iterations
steps
warnings
errors
raw_result
validation
```

It provides:

```python
result.summary()
result.explain()
result.to_csv(...)
result.to_json(...)
```

---

## Explainability Model

Every kernel step is recorded as a `KernelStep`.

A step contains:

```text
name
duration
stats
```

This allows the result object to explain the execution:

```text
Loaded query and data graphs.
Converted inputs to CSR.
Generated query signatures.
Generated data signatures.
Filtered candidates.
Executed refinement steps.
Ran final join.
```

This is important because users can understand not only the final number of matches, but also how the result was produced.

---

## Validation Architecture

Optional validation is implemented in `validation.py`.

The validation flow is:

```text
MatchResult
   |
   v
validate_result_with_rdkit()
   |
   | for each query-data pair:
   |     parse query with RDKit
   |     parse target with RDKit
   |     run HasSubstructMatch
   |     compare with SIGMo result
   v
result.validation
```

The validation metadata includes:

```text
enabled
method
checked_pairs
agreements
disagreements
skipped
passed
```

Validation is useful for correctness checks on small or controlled datasets.

For very large datasets, complete validation is not recommended because it may be computationally expensive.

---

## Large Result Architecture

Large result handling is mainly implemented in `examples/run_pipeline.py`, but it relies on the structured result model defined in `result.py`.

The problem is that large molecular searches may generate millions of matches.

Materializing all matches as Python objects can exhaust RAM.

The large-result strategy is:

```text
if num_matches > threshold:
    do not materialize all matches in MatchResult
    preserve total_matches
    write CSV in streaming mode
    write JSON summary only
    print only a small preview
```

This makes full dataset execution practical.

---

## Large Result Flow

```text
raw_join_result
   |
   | num_matches > large_result_threshold
   v
large_result = True
   |
   | create MatchResult without full matches list
   | keep total_matches
   | keep kernel steps and warnings
   v
summary-only MatchResult
   |
   | optional CSV streaming
   | optional JSON summary
   v
outputs/
```

CSV streaming iterates over `matches_dict` without creating a huge intermediate list.

JSON summary intentionally avoids storing all matches and only contains metadata and statistics.

---

## Visualization Placeholder

The package contains:

```text
visualize.py
```

At the moment, this file is empty.

It is intentionally kept as a placeholder for future visualization features.

Possible future responsibilities include:

```text
CSR graph to NetworkX conversion
molecule drawing
query-target visualization
match visualization
RDKit-based drawing
substructure highlighting
```

No production code currently depends on `visualize.py`.

Future possible API:

```python
sigmo.visualize.draw_graph(...)
sigmo.visualize.draw_molecule(...)
sigmo.visualize.draw_match(...)
```

The file exists to make the planned extension point explicit.

---

## Error and Warning Strategy

The interface records warnings and errors inside result objects.

Warnings include situations such as:

```text
refinement disabled for stability
force refinement enabled on small graphs
large result detected
RDKit validation skipped
```

Errors are stored when a Python-level exception occurs during the pipeline.

This allows users to inspect execution metadata after the run:

```python
print(result.warnings)
print(result.errors)
```

Native segmentation faults cannot be caught by Python exceptions. For this reason, the interface includes safe defaults and explicit force modes.

---

## Full Dataset Execution

The architecture was tested on a full molecular dataset with:

```text
619 query graphs
114,901 data graphs
```

With 6 refinement iterations, the pipeline completed:

```text
generate signatures
filter candidates
refine query signatures
refine data signatures
refine candidates
join candidates
```

Observed candidate reduction:

```text
3,610,045,526 candidates after filter
389,794,092 candidates after 6 refinement iterations
```

This demonstrates that the Python orchestration supports full dataset execution and that refinement significantly reduces the candidate space.

---

## Design Principles

The current architecture follows these principles:

1. **Keep the backend thin.**  
   The C++/SYCL backend performs computation. Python does not reimplement the algorithm.

2. **Keep the API layered.**  
   Simple users should call `match()` or `search()`. Advanced users can use `PipelineContext`.

3. **Keep results explainable.**  
   Every run should expose summary, steps, warnings and errors.

4. **Keep large outputs safe.**  
   Millions of matches should not be materialized in memory unnecessarily.

5. **Keep validation optional.**  
   RDKit validation is useful but should not be forced on large workloads.

6. **Keep future extensions isolated.**  
   Visualization is reserved for `visualize.py`.
