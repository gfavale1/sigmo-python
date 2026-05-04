# SIGMo Python Benchmarks

This document reports the benchmark-style runs performed with the SIGMo Python interface.

The purpose of this document is to document the current behavior of the Python interface on a real molecular dataset and to make the experiments reproducible.

The benchmarks focus on:

- full pipeline execution;
- candidate filtering;
- iterative refinement;
- final join/isomorphism;
- large-result handling;
- CSV streaming output;
- JSON summary output.

---

## Important Note

The numbers reported here are a snapshot of the current development environment, dataset and backend configuration.

They should be interpreted as reproducibility notes for this repository, not as final performance results.

Performance may vary depending on:

- GPU model;
- CPU model;
- available memory;
- SYCL backend;
- driver version;
- SIGMo backend version;
- dataset composition;
- refinement settings;
- `find_first` behavior;
- output/export configuration.

---

## Hardware and Environment

The benchmark runs documented here were executed on:

```text
GPU: NVIDIA GeForce RTX 3060 Laptop GPU
```

Before running the benchmark commands, build the native extension and install the package in editable mode from the repository root:

```bash
conda activate hpc_env
source /opt/intel/oneapi/setvars.sh

./scripts/build.sh
python -m pip install -e . --no-build-isolation --no-deps
```

Typical command prefix:

```bash
python examples/run_pipeline.py
```

---

## Dataset

The benchmark dataset is expected under:

```text
benchmarks/datasets/query.smarts
benchmarks/datasets/data.smarts
```

The full dataset used in these runs contains:

| Component | Count |
|---|---:|
| Query graphs | 619 |
| Data graphs | 114,901 |
| Query nodes | 3,417 |
| Data nodes | 2,745,872 |
| Query directed CSR edges | 6,062 |
| Data directed CSR edges | 5,890,028 |

The dataset files are not necessarily tracked by Git.  
If they are missing, place them manually under:

```text
benchmarks/datasets/
```

with the expected names:

```text
query.smarts
data.smarts
```

---

## Pipeline Stages

The benchmarked pipeline consists of the following stages:

```text
1. load query and data graphs
2. allocate SIGMo structures
3. generate query signatures
4. generate data signatures
5. filter candidates
6. optionally refine signatures and candidates
7. run final join/isomorphism
8. optionally export results
```

The native SIGMo kernels involved are:

```text
generate_csr_signatures
filter_candidates
refine_csr_signatures
refine_candidates
join_candidates
```

---

## Output Policy for Large Results

Full dataset runs may produce millions of matches.

For large results, `examples/run_pipeline.py` uses a memory-safe strategy:

```text
- do not print the full matches_dict;
- do not materialize all matches into MatchResult;
- preserve total match count;
- write CSV in streaming mode when requested;
- write JSON summary instead of full JSON when requested.
```

This avoids out-of-memory conditions when dealing with very large match sets.

---

# Reproducible Commands

All commands should be executed from the repository root.

---

## Full Dataset Without Refinement

This run executes:

```text
generate signatures
filter candidates
join candidates
```

and skips refinement.

### Command without export

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 0 \
  --max-print-matches 0
```

### Command with export

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 0 \
  --max-print-matches 0 \
  --csv examples/outputs/matches_full_no_refine.csv \
  --json examples/outputs/matches_full_no_refine_summary.json
```

### Observed Results

| Metric | Value |
|---|---:|
| Query graphs | 619 |
| Data graphs | 114,901 |
| Candidates after filter | 3,610,045,526 |
| Executed refinement iterations | 0 |
| Final matches | 17,839,985 |
| Join time | ~45.8s |

### Notes

Without refinement, the candidate space remains very large before the final join.

The join still completes on the tested environment, but it produces a very large number of final matches.

For large output export, CSV streaming is required.

---

## Full Dataset With 6 Refinement Iterations

This run executes the complete pipeline:

```text
generate signatures
filter candidates
refine query signatures
refine data signatures
refine candidates
join candidates
```

### Command without export

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0
```

### Command with export

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

### Observed Results

| Metric | Value |
|---|---:|
| Query graphs | 619 |
| Data graphs | 114,901 |
| Candidates after filter | 3,610,045,526 |
| Candidates after 6 refinement iterations | 389,794,092 |
| Executed refinement iterations | 6 |
| Final matches | 15,898,409 |
| Join time | ~13.4s |

### Candidate Count by Refinement Iteration

| Stage | Candidate count |
|---|---:|
| After initial filter | 3,610,045,526 |
| After refinement 1 | 1,374,971,529 |
| After refinement 2 | 762,236,035 |
| After refinement 3 | 528,117,577 |
| After refinement 4 | 426,987,766 |
| After refinement 5 | 398,719,768 |
| After refinement 6 | 389,794,092 |

### Notes

The refinement stage significantly reduces the candidate space before the final join.

The reduction is strongest during the first refinement iterations and becomes smaller in later iterations.

This suggests that future versions may benefit from an automatic stopping criterion based on candidate reduction.

---

# Comparison: No Refinement vs 6 Refinements

| Scenario | Candidates after filter | Candidates after refinement | Final matches | Join time |
|---|---:|---:|---:|---:|
| Full dataset, no refinement | 3,610,045,526 | - | 17,839,985 | ~45.8s |
| Full dataset, 6 refinements | 3,610,045,526 | 389,794,092 | 15,898,409 | ~13.4s |

---

## Candidate Reduction

With 6 refinement iterations:

```text
Initial candidates: 3,610,045,526
Final candidates:     389,794,092
```

Absolute reduction:

```text
3,610,045,526 - 389,794,092 = 3,220,251,434
```

Approximate percentage reduction:

```text
~89.2%
```

---

## Match Reduction

Final matches without refinement:

```text
17,839,985
```

Final matches with 6 refinement iterations:

```text
15,898,409
```

Absolute reduction:

```text
17,839,985 - 15,898,409 = 1,941,576
```

Approximate reduction:

```text
~10.9%
```

---

## Join Time Reduction

Observed join time without refinement:

```text
~45.8s
```

Observed join time with 6 refinement iterations:

```text
~13.4s
```

This indicates that the refinement stage reduces the amount of work left for the final join.

---

# Medium-Scale Refinement Run

A smaller run was also used to validate the pipeline before executing the full dataset.

### Command

```bash
python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

### Observed Results

| Metric | Value |
|---|---:|
| Query graphs | 100 |
| Data graphs | 5,000 |
| Query nodes | 581 |
| Data nodes | 118,514 |
| Candidates after filter | 27,809,881 |
| Candidates after 6 refinement iterations | 2,799,201 |
| Final matches | 105,712 |
| Executed refinement iterations | 6 |

### Candidate Count by Refinement Iteration

| Stage | Candidate count |
|---|---:|
| After initial filter | 27,809,881 |
| After refinement 1 | 10,210,943 |
| After refinement 2 | 5,701,942 |
| After refinement 3 | 3,735,305 |
| After refinement 4 | 2,979,379 |
| After refinement 5 | 2,845,939 |
| After refinement 6 | 2,799,201 |

### Notes

This medium-scale run was useful to verify that all refinement kernels were executed correctly before moving to the full dataset.

---

# Small Debug Run

A small run was used during development to verify the individual refinement kernels.

### Command

```bash
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

This run is useful for quick debugging because it exercises the full kernel flow while keeping execution time low.

---

# Interpreting the Results

The benchmark results show that:

1. The Python interface can execute the full SIGMo pipeline on a large dataset.
2. The refinement stage works on the full dataset when explicitly forced.
3. The candidate space is reduced significantly by refinement.
4. The final join becomes faster after refinement.
5. Very large result sets require streaming export instead of full in-memory materialization.

---

# Refinement Behavior

The refinement phase performs:

```text
refine query signatures
refine data signatures
refine candidates
```

for each iteration.

In the full dataset run, the candidate counts decreased as follows:

```text
3.61B -> 1.37B -> 762M -> 528M -> 427M -> 399M -> 390M
```

Most of the reduction happens in the first three iterations.

This suggests that future benchmarking could include:

```text
iterations = 1
iterations = 2
iterations = 3
iterations = 6
```

to determine the best trade-off between refinement cost and join-time reduction.

---

# Safe Mode and Force Mode

`examples/run_pipeline.py` detects small graphs before refinement.

Small graphs may be potentially unstable for some backend configurations.

By default, refinement can be skipped for safety.

To force the complete pipeline, use:

```bash
--force-refine
```

Example:

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 0
```

In the documented full dataset run, `--force-refine` was used successfully.

---

# Large Output Export

## CSV Streaming

When `--csv` is provided and the result is large, matches are written incrementally.

Example:

```bash
--csv examples/outputs/matches_full_refine.csv
```

The CSV contains:

```csv
query_index,query_name,query_input,data_index,data_name,data_input
```

This format is suitable for downstream analysis with:

- Python;
- pandas;
- R;
- command-line tools;
- databases.

For very large CSV files, loading the entire file into memory may not be practical. Use chunked reading when using pandas:

```python
import pandas as pd

for chunk in pd.read_csv("examples/outputs/matches_full_refine.csv", chunksize=1_000_000):
    print(chunk.shape)
```

---

## JSON Summary

When `--json` is provided and the result is large, the JSON file contains only metadata and kernel statistics.

Example:

```bash
--json examples/outputs/matches_full_refine_summary.json
```

The JSON summary includes:

```text
status
device
query_count
data_count
total_matches
requested_iterations
executed_iterations
warnings
errors
kernel_steps
```

It intentionally does not contain all matches.

---

# Recommended Benchmark Workflow

For a quick sanity check:

```bash
python examples/run_pipeline.py \
  --query-limit 5 \
  --data-limit 20 \
  --iterations 0 \
  --max-print-matches 20
```

For a medium refinement test:

```bash
python examples/run_pipeline.py \
  --query-limit 100 \
  --data-limit 5000 \
  --iterations 6 \
  --force-refine \
  --max-print-matches 20
```

For a full no-refinement baseline:

```bash
python examples/run_pipeline.py \
  --query-limit -1 \
  --data-limit -1 \
  --iterations 0 \
  --max-print-matches 0 \
  --json examples/outputs/matches_full_no_refine_summary.json
```

For a full refinement run:

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

# Suggested Future Benchmarks

Future benchmark work could include:

- running with different numbers of refinement iterations;
- comparing CPU and GPU execution;
- measuring total pipeline time, not only kernel timings;
- measuring memory usage;
- testing different dataset sizes;
- testing compact CSV output;
- testing `find_first=True` versus `find_first=False`;
- evaluating the effect of aromatic bond policies;
- testing on different GPUs.

Suggested iteration study:

| Iterations | Purpose |
|---:|---|
| 0 | Baseline without refinement |
| 1 | First refinement impact |
| 2 | Early refinement trade-off |
| 3 | Medium refinement |
| 6 | Current full refinement setting |

---

# Current Benchmark Status

```text
Full dataset without refinement: completed
Full dataset with 6 refinements: completed
CSV streaming export: completed
JSON summary export: completed
Large-result mode: working
Full refinement on RTX 3060 Laptop GPU: completed
```
