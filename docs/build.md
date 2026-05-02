# Build and Environment Setup

This document describes how to set up the development environment for the SIGMo Python interface and how to verify that the native binding and Python API are working correctly.

The project is composed of two layers:

1. the native SIGMo C++/SYCL backend and Python binding;
2. the Python interface located under `python/sigmo/`.

The Python interface expects the native SIGMo extension to be already built and importable.

---

## Repository Layout

Relevant folders:

```text
sigmo-python/
├── python/
│   └── sigmo/
│       ├── __init__.py
│       ├── graph.py
│       ├── matcher.py
│       ├── pipeline.py
│       ├── result.py
│       ├── validation.py
│       └── ...
│
├── examples/
│   ├── basic_usage.py
│   ├── validation_usage.py
│   ├── advanced_pipeline.py
│   ├── run_pipeline.py
│   └── outputs/
│
├── tests/
├── docs/
├── benchmarks/
└── pyproject.toml
```

During development, the package is usually executed directly from the repository root using:

```bash
PYTHONPATH=python
```

---

## Requirements

The exact build requirements may depend on the native SIGMo backend configuration and the target device.

The current development setup assumes:

- Python environment with SIGMo dependencies installed;
- RDKit for molecule parsing and optional validation;
- dpctl for SYCL queue/device handling;
- pytest for testing;
- a working C++/SYCL toolchain for the native backend;
- a supported backend device such as GPU or CPU.

Typical Python-side dependencies include:

```text
rdkit
dpctl
pytest
```

Optional dependencies:

```text
pandas
```

`pandas` is only needed for DataFrame-based workflows. CSV export in the current interface can also work without pandas.

---

## Development Environment

Activate the Python environment used for development.

Example:

```bash
conda activate hpc_env
```

From the repository root:

```bash
cd /path/to/sigmo-python
```

Verify Python:

```bash
python --version
```

Verify that the package can be imported from source:

```bash
PYTHONPATH=python python -c "import sigmo; print('SIGMo import OK')"
```

If this fails, the native extension or the Python path is not correctly configured.

---

## Running From Source

The current development workflow runs the package directly from the local `python/` directory.

Use:

```bash
PYTHONPATH=python python examples/basic_usage.py
```

or:

```bash
PYTHONPATH=python pytest tests -vv
```

This avoids requiring installation into site-packages during development.

---

## Editable Installation

If the project packaging is configured through `pyproject.toml`, it may also be possible to install the project in editable mode:

```bash
pip install -e .
```

Then imports should work without manually setting `PYTHONPATH`:

```bash
python -c "import sigmo; print('SIGMo import OK')"
```

If editable installation is not currently configured or does not expose the native extension correctly, use the `PYTHONPATH=python` workflow.

---

## Native Extension Build

The Python interface depends on the native SIGMo binding.

Depending on the current repository configuration, the native extension may be built through CMake, Python packaging, or a custom build script.

A typical CMake-style native build may look like:

```bash
mkdir -p build
cd build
cmake ..
cmake --build . -j
```

However, the exact command may differ depending on:

- compiler;
- SYCL implementation;
- CUDA/ROCm/Level Zero backend;
- local SIGMo build configuration;
- pybind11 configuration;
- target device.

If using a SYCL compiler such as Intel `icpx`/`dpcpp`, make sure the compiler is visible:

```bash
which icpx
which dpcpp
```

If targeting NVIDIA GPUs through SYCL/CUDA, make sure the CUDA-capable backend and drivers are correctly installed.

After building, verify that the Python binding is importable:

```bash
PYTHONPATH=python python -c "import sigmo; print(sigmo)"
```

---

## Verifying Device Selection

The Python interface uses `dpctl` to create SYCL queues.

To check the selected default device:

```bash
PYTHONPATH=python python - <<'PY'
import sigmo
from sigmo.config import get_default_queue

q = get_default_queue()
print("Selected device:", q.sycl_device.name)
PY
```

To test a specific device:

```bash
PYTHONPATH=python python - <<'PY'
from sigmo.config import get_sycl_queue

q = get_sycl_queue("gpu")
print("GPU device:", q.sycl_device.name)
PY
```

or:

```bash
PYTHONPATH=python python - <<'PY'
from sigmo.config import get_sycl_queue

q = get_sycl_queue("cpu")
print("CPU device:", q.sycl_device.name)
PY
```

---

## Smoke Tests

After building or changing the environment, run a minimal import test:

```bash
PYTHONPATH=python python -c "import sigmo; print('import OK')"
```

Run the basic example:

```bash
PYTHONPATH=python python examples/basic_usage.py
```

Run the validation example:

```bash
PYTHONPATH=python python examples/validation_usage.py
```

Run the full test suite:

```bash
PYTHONPATH=python pytest tests -vv
```

Expected result:

```text
31 passed
```

---

## Test Suite

The test suite checks:

- graph conversion;
- SMARTS/SMILES loading;
- aromatic bond stability;
- low-level kernel bindings;
- high-level matching API;
- batch search API;
- `SIGMoMatcher`;
- `PipelineContext`;
- RDKit validation;
- CSV/JSON export.

Run all tests:

```bash
PYTHONPATH=python pytest tests -vv
```

Run compact tests:

```bash
PYTHONPATH=python pytest tests -q
```

Run a single test file:

```bash
PYTHONPATH=python pytest tests/test_matcher.py -vv
```

---

## Running Examples

### Basic usage

```bash
PYTHONPATH=python python examples/basic_usage.py
```

### RDKit validation

```bash
PYTHONPATH=python python examples/validation_usage.py
```

### Advanced pipeline

```bash
PYTHONPATH=python python examples/advanced_pipeline.py
```

### Kernel-level command-line pipeline

Small default run:

```bash
PYTHONPATH=python python examples/run_pipeline.py
```

Medium run with refinement:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

Full dataset run with refinement and export:

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

---

## Benchmark Dataset

The full pipeline example expects dataset files under:

```text
benchmarks/datasets/query.smarts
benchmarks/datasets/data.smarts
```

Check that they exist:

```bash
ls benchmarks/datasets/query.smarts
ls benchmarks/datasets/data.smarts
```

These files may be ignored by Git if they are large.

If they are not present, place them manually in `benchmarks/datasets/` before running the full pipeline example.

---

## Generated Outputs

Generated files should be written under:

```text
examples/outputs/
```

Create the folder if needed:

```bash
mkdir -p examples/outputs
```

The repository should track only:

```text
examples/outputs/.gitkeep
```

Generated CSV and JSON outputs should normally be ignored by Git.

Recommended `.gitignore` rules:

```gitignore
examples/outputs/*
!examples/outputs/.gitkeep
```

---

## Cleaning Build Artifacts

Common generated files and folders include:

```text
build/
bin/
lib/
*.o
*.a
*.so
*.pyd
*.out
.cache/
__pycache__/
.pytest_cache/
dist/
build_python/
*.egg-info/
```

To clean Python cache files:

```bash
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type d -name ".pytest_cache" -exec rm -rf {} +
```

To remove generated example outputs:

```bash
rm -f examples/outputs/*.csv
rm -f examples/outputs/*.json
```

Do not remove:

```text
examples/outputs/.gitkeep
```

unless you intentionally want to remove the placeholder.

---

## Common Issues

### `ModuleNotFoundError: No module named 'sigmo'`

Make sure you are running from the repository root and using:

```bash
PYTHONPATH=python
```

Example:

```bash
PYTHONPATH=python python examples/basic_usage.py
```

---

### Native extension import error

If `import sigmo` fails because the native extension is missing, the C++/SYCL binding may not be built or may not be located where Python expects it.

Check:

```bash
find . -name "*.so"
find . -name "*.pyd"
```

Then rebuild the native extension according to the current backend build configuration.

---

### No GPU device found

If GPU selection fails, verify that the device is visible to `dpctl`:

```bash
python - <<'PY'
import dpctl
print(dpctl.get_devices())
PY
```

If no GPU appears, check:

- GPU driver installation;
- SYCL backend installation;
- CUDA/Level Zero/ROCm support;
- environment variables required by the backend.

You can also try CPU execution:

```bash
PYTHONPATH=python python examples/basic_usage.py
```

or explicitly:

```python
device="cpu"
```

---

### Segmentation fault during native kernel execution

A segmentation fault usually comes from the native backend, not from Python exceptions.

Useful debugging steps:

1. reduce dataset size;
2. run with `--iterations 0`;
3. run with a small query/data limit;
4. test individual kernels through `PipelineContext`;
5. compare with the debug or advanced pipeline examples.

Example:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 0 \
  --max-print-matches 20
```

Then try refinement:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

---

### Process killed during large export

If the process is killed with:

```text
Killed
```

the operating system likely terminated it due to excessive memory usage.

For large outputs, avoid:

- printing all matches;
- building a complete JSON with all matches;
- materializing all matches in a pandas DataFrame.

Use:

```bash
--max-print-matches 0
```

and CSV streaming:

```bash
--csv examples/outputs/matches_full_refine.csv
```

The JSON output for large runs should be a summary:

```bash
--json examples/outputs/matches_full_refine_summary.json
```

---

## Recommended Development Workflow

A typical development cycle is:

```bash
conda activate hpc_env
cd /path/to/sigmo-python
```

Verify import:

```bash
PYTHONPATH=python python -c "import sigmo; print('import OK')"
```

Run tests:

```bash
PYTHONPATH=python pytest tests -vv
```

Run examples:

```bash
PYTHONPATH=python python examples/basic_usage.py
PYTHONPATH=python python examples/validation_usage.py
```

Run a small pipeline:

```bash
PYTHONPATH=python python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

Only after this, run the full dataset pipeline.

---

## Current Expected Status

The current expected status of the Python interface is:

```text
import sigmo: OK
basic_usage.py: OK
validation_usage.py: OK
pytest tests: 31 passed
full dataset without refinement: OK
full dataset with 6 refinements: OK
CSV streaming export: OK
JSON summary export: OK
```

