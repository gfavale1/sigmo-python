# SIGMo Python Test Suite

This folder contains the automated test suite for the SIGMo Python interface.

The tests verify both the high-level Python API and the low-level native kernel bindings exposed through the Python package.

The current test suite checks:

- molecule loading and SMARTS/SMILES conversion;
- CSR graph construction;
- aromatic bond stability;
- low-level kernel calls;
- high-level `sigmo.match()` and `sigmo.search()`;
- object-oriented `SIGMoMatcher`;
- `PipelineContext` step-by-step execution;
- result export;
- visualization utilities.

Current expected status:

```text
25 passed
```

---

## Running the Tests

Before running the tests, build the native extension and install the package in editable mode from the repository root:

```bash
./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
```

Run all tests:

```bash
pytest tests -vv
```

Compact output:

```bash
pytest tests -q
```

Run a single test file:

```bash
pytest tests/test_matcher.py -vv
```

Run a single test:

```bash
pytest tests/test_matcher.py::test_match_high_level_positive -vv
```

If the package has not been installed in editable mode, the temporary fallback is:

```bash
SIGMO_USE_PYTHONPATH=1 source scripts/dev_env.sh
pytest tests -vv
```

However, the recommended workflow is to use the editable install.

---

## Test Folder Structure

```text
tests/
├── conftest.py
├── helpers.py
├── test_graph.py
├── test_low_level_kernels.py
├── test_matcher.py
├── test_pipeline.py
├── test_visualize.py
└── README.md
```

Optional development/debug scripts may temporarily exist during debugging, but they should not be considered part of the automatic test suite unless they follow the `test_*.py` naming convention and are intended to be run by `pytest`.

---

## `conftest.py`

This file defines shared pytest fixtures used across the test suite.

It provides reusable objects such as:

- a shared SYCL queue;
- simple CSR graphs;
- `Signature` objects;
- `Candidates` objects;
- temporary SMARTS files.

Main fixtures include:

```python
q
ethane_graph
simple_graphs
sig_simple
cand_simple
sample_smarts_file
invalid_smarts_file
```

The shared queue fixture uses the same device-selection logic as the package, through `get_default_queue()`.

---

## `helpers.py`

This file contains common helper assertions used by multiple test files.

The main helper is:

```python
assert_match_result(result)
```

This helper checks that a returned object behaves like a `MatchResult`, verifying that it exposes fields and methods such as:

```text
total_matches
matches
steps
warnings
summary()
explain()
```

Keeping helper assertions in `helpers.py` avoids importing directly from `conftest.py` and keeps pytest fixtures separate from reusable test utilities.

---

## `test_graph.py`

This file tests molecule loading and graph conversion.

It verifies that the Python interface correctly builds SIGMo-compatible CSR graph dictionaries.

Covered functionality:

- `make_csr_graph()`;
- `toy_two_node_graph()`;
- `smarts_to_csr_from_string()`;
- `smarts_to_csr()`;
- `load_molecules()`;
- automatic name generation;
- invalid SMARTS/SMILES handling;
- aromatic bond stability.

Important checks include:

```text
CSR graph contains row_offsets
CSR graph contains column_indices
CSR graph contains node_labels
CSR graph contains edge_labels
num_nodes matches node_labels length
invalid molecules are skipped without crashing
aromatic bonds do not introduce unsupported labels
```

The aromatic stability test is important because unsupported explicit aromatic labels previously caused native backend crashes. The current conversion maps RDKit aromatic bond order `1.5` to integer label `1`, keeping bond labels compatible with the backend.

---

## `test_low_level_kernels.py`

This file tests the low-level native kernel bindings exposed through Python.

It directly checks calls such as:

```python
sigmo.generate_csr_signatures(...)
sigmo.refine_csr_signatures(...)
sigmo.filter_candidates(...)
sigmo.join_candidates(...)
```

Covered functionality:

- signature generation for data graphs;
- signature refinement;
- candidate filtering;
- join/isomorphism kernel;
- empty graph list handling.

These tests are closer to the native C++/SYCL layer and are useful to verify that the binding can call the actual backend kernels without crashing.

They are intentionally lower-level than the user-facing API tests.

---

## `test_matcher.py`

This file tests the main high-level Python API.

Covered functionality:

- positive single match with `sigmo.match()`;
- negative single match with `sigmo.match()`;
- `sigmo.run_isomorphism()`;
- batch search with `sigmo.search()`;
- object-oriented execution with `SIGMoMatcher`;
- CSV/JSON export from `MatchResult`.

Example API tested:

```python
result = sigmo.match(
    query="CC",
    target="CCC",
    input_format="smiles",
    iterations=0,
    find_first=True,
    device="auto",
)
```

The tests verify that these APIs return a valid `MatchResult`, not a raw dictionary or `None`.

This file is important because it protects the public Python API from regressions.

---

## `test_pipeline.py`

This file tests `PipelineContext`, the advanced API for kernel-level execution.

Covered functionality:

- context creation;
- allocation of SIGMo structures;
- signature generation;
- candidate filtering;
- join execution;
- full `PipelineContext.run()` execution;
- step tracking.

It verifies that a user can manually run:

```python
ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.join(find_first=True)
```

and can also use:

```python
result = ctx.run(iterations=0, find_first=True)
```

The returned object must be a `MatchResult`.

---

## `test_visualize.py`

This file tests the optional visualization utilities.

Covered functionality:

- SMILES parsing through `mol_from_input()`;
- SMARTS parsing through `mol_from_input()`;
- drawing a single molecule with `draw_molecule()`;
- drawing a query-target pair with `draw_match_pair()`;
- converting a SIGMo CSR graph to NetworkX with `to_networkx()`.

The tests verify that the visualization helpers return valid image/graph objects and do not interfere with the core SIGMo pipeline.

NetworkX-specific tests use:

```python
pytest.importorskip("networkx")
```

so the test is skipped automatically if NetworkX is not installed.

---

## What the Tests Protect

The test suite protects against several classes of regressions.

### API regressions

For example:

```text
sigmo.match() returns None
sigmo.search() returns a raw dict
SIGMoMatcher.run() forgets to return last_result
PipelineContext.run() forgets to return MatchResult
```

These kinds of bugs happened during development and are now covered by tests.

---

### Graph conversion regressions

For example:

```text
SMILES/SMARTS conversion breaks
invalid molecules crash the loader
aromatic bonds are assigned unsupported labels
CSR graph fields are missing
```

---

### Kernel binding regressions

For example:

```text
generate_csr_signatures fails
filter_candidates crashes
join_candidates does not return num_matches
refine_csr_signatures stops returning valid stats
```

---

### Visualization regressions

For example:

```text
draw_molecule() stops returning an image
draw_match_pair() fails on valid query-target inputs
CSR graphs cannot be converted to NetworkX
visualization helpers import heavy optional dependencies too early
```

Visualization tests are intentionally lightweight and do not validate visual aesthetics pixel-by-pixel.

---

## Notes on Device Usage

The tests use the same device-selection logic as the package.

The shared queue fixture usually relies on:

```python
get_default_queue()
```

Depending on the local environment, this may select:

```text
GPU if available
CPU otherwise
```

If GPU initialization fails, check your SYCL, dpctl and driver setup.

The tests are designed to be small enough to run quickly on development machines.

---

## Notes on Large Datasets

The test suite intentionally does **not** run the full benchmark dataset.

Full dataset execution is handled by:

```text
examples/run_pipeline.py
```

This is because full dataset runs may involve:

- hundreds of query graphs;
- more than one hundred thousand data graphs;
- billions of candidates;
- millions of matches;
- large CSV outputs.

Such runs are useful for benchmarking and performance checks, but they are not appropriate for a regular unit test suite.

---

## Expected Development Workflow

Before committing changes, run:

```bash
pytest tests -vv
```

Expected result:

```text
25 passed
```

Then optionally run lightweight examples:

```bash
python examples/basic_usage.py
python examples/advanced_pipeline.py
python examples/run_pipeline.py --query-limit 5 --data-limit 20 --iterations 0
python examples/visualization_usage.py
```

For pipeline or performance checks, use:

```bash
python examples/run_pipeline.py   --query-limit 100   --data-limit 5000   --iterations 6   --force-refine   --max-print-matches 20
```

Generated outputs in `examples/outputs/` should not normally be committed.

---

## Adding New Tests

When adding new package features, add tests in the appropriate file:

| Feature type | Suggested test file |
|---|---|
| Graph loading / CSR conversion | `test_graph.py` |
| Low-level kernel wrappers | `test_low_level_kernels.py` |
| High-level API | `test_matcher.py` |
| Pipeline orchestration | `test_pipeline.py` |
| Visualization utilities | `test_visualize.py` |
| Export behavior | `test_matcher.py` or a dedicated export test |

Test names should describe the behavior being checked.

Example:

```python
def test_match_high_level_positive():
    ...
```

Prefer small deterministic molecules and controlled toy graphs.

Avoid adding full dataset tests to the regular pytest suite.

---

## Debugging Failed Tests

Useful commands:

```bash
pytest tests -vv
```

Show print output:

```bash
pytest tests -vv -s
```

Run only matcher tests:

```bash
pytest tests/test_matcher.py -vv
```

Run only pipeline tests:

```bash
pytest tests/test_pipeline.py -vv
```

Run only visualization tests:

```bash
pytest tests/test_visualize.py -vv
```

If a test returns `None` instead of `MatchResult`, check:

- `PipelineContext.run()`;
- `run_isomorphism()`;
- `SIGMoMatcher.run()`.

These methods must always return the result object.

If imports fail, verify that the package has been installed in editable mode:

```bash
python - <<'PY'
import sigmo
print(sigmo.__file__)
print("SIGMo import OK")
PY
```

---

## Current Status

```text
Graph conversion tests: passing
Low-level kernel tests: passing
High-level API tests: passing
PipelineContext tests: passing
Visualization tests: passing
Export tests: passing

Current expected result: 25 passed
```
