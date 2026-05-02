# SIGMo Python Examples

This folder contains runnable examples showing how to use the Python interface built on top of SIGMo.

The examples are organized by complexity:

- `basic_usage.py`: minimal high-level usage with `sigmo.match()`;
- `validation_usage.py`: example with optional RDKit validation;
- `advanced_pipeline.py`: direct `PipelineContext` usage for step-by-step execution;
- `run_pipeline.py`: full command-line kernel-level pipeline execution;
- `outputs/`: folder used to store generated CSV/JSON outputs.

All commands should be executed from the **repository root**, not from inside the `examples/` folder.

```bash
cd /path/to/sigmo-python