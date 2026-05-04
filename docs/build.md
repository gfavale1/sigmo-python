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
│       ├── config.py
│       ├── graph.py
│       ├── matcher.py
│       ├── pipeline.py
│       ├── result.py
│       ├── utils.py
│       ├── visualize.py
│       └── ...
│
├── examples/
│   ├── basic_usage.py
│   ├── advanced_pipeline.py
│   ├── run_pipeline.py
│   ├── visualization_usage.py
│   └── outputs/
│
├── tests/
├── docs/
├── benchmarks/
├── scripts/
│   ├── build.sh
│   └── dev_env.sh
├── setup.py
└── pyproject.toml
```

During development, the recommended workflow is:

```bash
./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
```

After the editable installation, examples and tests can be executed without setting `PYTHONPATH` manually.

---

## Requirements

The exact build requirements may depend on the native SIGMo backend configuration and the target device.

The current development setup assumes:

- Python environment with SIGMo dependencies installed;
- RDKit for molecule parsing and visualization support;
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
networkx
matplotlib
pillow
```

`pandas` is only needed for DataFrame-based workflows. CSV export in the current interface can also work without pandas.

`networkx` and `matplotlib` are only needed for CSR graph visualization through `sigmo.visualize.draw_graph()`.

---

## Development Environment

Activate the Python environment used for development.

Example:

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh
```

From the repository root:

```bash
cd /path/to/sigmo-python
```

Verify Python:

```bash
python --version
```

Build the native extension:

```bash
./scripts/build.sh
```

Install the package in editable mode:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

Verify that the package can be imported:

```bash
python -c "import sigmo; print('SIGMo import OK')"
```

To check which source tree is being imported:

```bash
python - <<'PY'
import sigmo
print(sigmo.__file__)
PY
```

The printed path should point to the local repository, for example:

```text
/path/to/sigmo-python/python/sigmo/__init__.py
```

---

## Running From Source

The recommended development workflow uses an editable installation.

After running:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

you can run examples directly:

```bash
python examples/basic_usage.py
```

or run tests:

```bash
pytest tests -vv
```

The editable installation points Python to the local `python/sigmo/` source tree. If Python files are modified, reinstalling is not required.

If C++/pybind11/CMake files are modified, rebuild the native extension:

```bash
./scripts/build.sh
```

A fallback source-only workflow is still possible through `scripts/dev_env.sh`:

```bash
SIGMO_USE_PYTHONPATH=1 source scripts/dev_env.sh
```

This fallback is useful for debugging, but editable installation is the preferred workflow.

---

## Editable Installation

The project can be installed in editable mode:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

The flags are intentional:

- `--no-build-isolation` avoids creating a temporary isolated build environment;
- `--no-deps` avoids reinstalling sensitive packages such as `dpctl`.

This is useful because the SYCL/dpctl environment can be version-sensitive.

After editable installation, imports should work without manually setting `PYTHONPATH`:

```bash
python -c "import sigmo; print('SIGMo import OK')"
```

You normally need to run the editable installation only once per environment.

Run it again only if:

- a new Conda environment is created;
- the package is uninstalled;
- packaging files such as `setup.py` or `pyproject.toml` are changed significantly.

---

## Native Extension Build

The Python interface depends on the native SIGMo binding.

The recommended build command is:

```bash
./scripts/build.sh
```

This script configures and builds the CMake project and places the compiled native module under:

```text
python/sigmo/_core*.so
```

The script supports optional environment variables:

```bash
BUILD_TYPE=Debug ./scripts/build.sh
JOBS=4 ./scripts/build.sh
BUILD_DIR=build-debug ./scripts/build.sh
```

It also accepts additional CMake arguments:

```bash
./scripts/build.sh -DCMAKE_CXX_COMPILER=icpx
```

If using a SYCL compiler such as Intel `icpx`/`dpcpp`, make sure the compiler is visible:

```bash
which icpx
which dpcpp
```

If targeting NVIDIA GPUs through SYCL/CUDA, make sure the CUDA-capable backend and drivers are correctly installed.

After building, verify that the Python binding is importable:

```bash
python -c "import sigmo; print(sigmo)"
```

---

## Verifying Device Selection

The Python interface uses `dpctl` to create SYCL queues.

To check the selected default device:

```bash
python - <<'PY'
import sigmo
from sigmo.config import get_default_queue

q = get_default_queue()
print("Selected device:", q.sycl_device.name)
PY
```

To test a specific device:

```bash
python - <<'PY'
from sigmo.config import get_sycl_queue

q = get_sycl_queue("gpu")
print("GPU device:", q.sycl_device.name)
PY
```

or:

```bash
python - <<'PY'
from sigmo.config import get_sycl_queue

q = get_sycl_queue("cpu")
print("CPU device:", q.sycl_device.name)
PY
```

You can also inspect devices directly through `dpctl`:

```bash
python - <<'PY'
import dpctl
print(dpctl.get_devices())
PY
```

---

## Smoke Tests

After building or changing the environment, run a minimal import test:

```bash
python -c "import sigmo; print('import OK')"
```

Run the basic example:

```bash
python examples/basic_usage.py
```

Run the visualization example:

```bash
python examples/visualization_usage.py
```

Run the full test suite:

```bash
pytest tests -vv
```

Expected result:

```text
all tests passing
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
- CSV/JSON export;
- visualization utilities.

Run all tests:

```bash
pytest tests -vv
```

Run compact tests:

```bash
pytest tests -q
```

Run a single test file:

```bash
pytest tests/test_matcher.py -vv
```

Run visualization tests:

```bash
pytest tests/test_visualize.py -vv
```

---

## Running Examples

### Basic usage

```bash
python examples/basic_usage.py
```

### Advanced pipeline

```bash
python examples/advanced_pipeline.py
```

### Visualization usage

```bash
python examples/visualization_usage.py
```

This generates PNG files under:

```text
examples/outputs/
```

### Kernel-level command-line pipeline

Small default run:

```bash
python examples/run_pipeline.py
```

Medium run with refinement:

```bash
python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

Full dataset run with refinement and export:

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

Generated PNG, CSV and JSON outputs should normally be ignored by Git.

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
rm -f examples/outputs/*.png
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

Make sure the package has been installed in editable mode from the repository root:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

Then verify:

```bash
python - <<'PY'
import sigmo
print(sigmo.__file__)
PY
```

If you intentionally want to run without editable installation, source the fallback development environment:

```bash
SIGMO_USE_PYTHONPATH=1 source scripts/dev_env.sh
```

---

### Native extension import error

If `import sigmo` fails because the native extension is missing, the C++/SYCL binding may not be built or may not be located where Python expects it.

Check:

```bash
find . -name "*.so"
find . -name "*.pyd"
```

Then rebuild the native extension:

```bash
./scripts/build.sh
```

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

You can also try CPU execution by passing:

```python
device="cpu"
```

or by running a standard example:

```bash
python examples/basic_usage.py
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
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 0 \
  --max-print-matches 20
```

Then try refinement:

```bash
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

---

### Clock skew detected during build

When working under `/mnt/c` in WSL, CMake or Make may report:

```text
Clock skew detected. Your build may be incomplete.
```

This is usually caused by timestamp differences between Windows and WSL file systems.

If the build still completes successfully, it is often harmless. If it becomes persistent, try:

```bash
find . -exec touch {} +
./scripts/build.sh
```

A more stable long-term solution is to work from a native Linux directory such as:

```text
~/projects/sigmo-python
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
source /opt/intel/oneapi/setvars.sh
cd /path/to/sigmo-python
```

Build the native extension:

```bash
./scripts/build.sh
```

Install the package in editable mode:

```bash
python -m pip install -e . --no-build-isolation --no-deps
```

Verify import:

```bash
python -c "import sigmo; print('import OK')"
```

Run tests:

```bash
pytest tests -vv
```

Run examples:

```bash
python examples/basic_usage.py
python examples/advanced_pipeline.py
python examples/visualization_usage.py
```

Run a small pipeline:

```bash
python examples/run_pipeline.py \
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
advanced_pipeline.py: OK
visualization_usage.py: OK
pytest tests: OK
full dataset without refinement: OK
full dataset with 6 refinements: OK
CSV streaming export: OK
JSON summary export: OK
editable package install: OK
```
