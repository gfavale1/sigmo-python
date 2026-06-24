# SIGMo Python Interface

Python interface for **SIGMo**, a C++/SYCL library for high-performance subgraph isomorphism.

This repository provides a thin Python layer around the native SIGMo backend. The C++/SYCL code performs the computation, while Python handles molecule loading, CSR conversion, pipeline orchestration, result formatting, visualization, examples and tests.

---

## What this repository contains

- A Python package in `python/sigmo/`.
- A native pybind11/SYCL extension exposed as `sigmo._core`.
- High-level APIs such as `sigmo.match()` and `sigmo.search()`.
- A step-by-step `PipelineContext` API for advanced users.
- Examples in `examples/`.
- Tests in `tests/`.
- Documentation in `docs/`.
- A SIGMo submodule in `external/sigmo`.

---

## Main features

- Load molecules from SMARTS/SMILES strings or files.
- Convert molecules into SIGMo-compatible CSR graphs.
- Select SYCL devices through `dpctl`.
- Run the native SIGMo pipeline from Python.
- Return structured and explainable `MatchResult` objects.
- Export results to CSV/JSON.
- Handle large outputs with CSV streaming.
- Visualize molecules, query-target pairs and internal CSR graphs.

RDKit is used for parsing and visualization. It is not used as a public validation layer, because SIGMo works on a simplified CSR representation while RDKit applies full SMARTS/SMILES semantics.

---

## Installation

Activate the Conda/SYCL environment:

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh
```

Build the native extension:

```bash
./scripts/build.sh
```

Install the package in editable mode:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

Verify the installation:

```bash
python - <<'PY'
import sigmo
print(sigmo.__file__)
print("SIGMo import OK")
PY
```

A minimal Conda environment is provided in `environment.yml`. The oneAPI/SYCL and `dpctl` setup may still depend on the target machine.

---

## API overview

### High-level matching

```python
result = sigmo.match("CC", "CCC", input_format="smiles")
```

### Batch search

```python
result = sigmo.search(
    queries="benchmarks/datasets/query.smarts",
    database="benchmarks/datasets/data.smarts",
    input_format="auto",
)
```

### Object-oriented API

```python
matcher = sigmo.SIGMoMatcher(device="auto", iterations=6)
result = matcher.run(queries, database)
```

### Step-by-step pipeline

```python
ctx = sigmo.PipelineContext(query_graphs, data_graphs, device="auto")
ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(6)
ctx.join(find_first=True)
```

---

## Notes on chemistry representation

RDKit is used to parse SMILES and SMARTS inputs and to expose their molecular graph structure. Before graphs are passed to the SIGMo backend, the Python interface converts them into a CSR representation aligned with the native SIGMo preprocessing workflow.

In particular, the conversion preserves:

the native SIGMo atom-label encoding;
the native bond-label convention;
the original adjacency ordering used by the native CSR construction.

Bond labels follow the SIGMo convention:

UNSPECIFIED -> 0
SINGLE      -> 1
DOUBLE      -> 2
TRIPLE      -> 3
AROMATIC    -> 4

Aromatic bonds are therefore kept distinct from single bonds. This is important because RDKit represents aromatic bonds with bond order 1.5; directly applying int(bond.GetBondTypeAsDouble()) would incorrectly map them to 1.

RDKit is used only for input parsing and graph construction; candidate filtering, refinement, and subgraph-isomorphism matching remain executed by the original SIGMo C++/SYCL backend.

---

## Large result handling

For runs producing millions of matches, `examples/run_pipeline.py` avoids materializing all matches in memory. It can:

- preserve the total match count;
- print only a preview;
- export full matches through streaming CSV;
- write a lightweight JSON summary.

Example:

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0 \
  --csv examples/outputs/matches.csv \
  --json examples/outputs/summary.json
```

---

## Documentation

More details are available in:

- [`docs/api.md`](docs/api.md)
- [`docs/architecture.md`](docs/architecture.md)
- [`docs/build.md`](docs/build.md)
- [`docs/benchmarks.md`](docs/benchmarks.md)

