from pathlib import Path

import sigmo
from sigmo.result import build_match_result


QUERY_FILE = Path("benchmarks/datasets/query.smarts")
DATA_FILE = Path("benchmarks/datasets/data.smarts")


def main():
    """
    Run the SIGMo pipeline step by step on a small benchmark subset.

    This example is intended for users who want to inspect the intermediate
    pipeline stages instead of using the high-level sigmo.match() or
    sigmo.search() APIs.
    """
    if not QUERY_FILE.exists() or not DATA_FILE.exists():
        raise FileNotFoundError(
            "Benchmark datasets not found. Expected files:\n"
            f"  - {QUERY_FILE}\n"
            f"  - {DATA_FILE}"
        )

    query_graphs = sigmo.load_molecules(
        QUERY_FILE,
        input_format="smarts",
    )
    data_graphs = sigmo.load_molecules(
        DATA_FILE,
        input_format="smiles",
    )

    # Keep the example lightweight and fast.
    query_graphs = query_graphs[:5]
    data_graphs = data_graphs[:20]

    ctx = sigmo.PipelineContext(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        device="auto",
    )

    ctx.allocate()
    ctx.generate_signatures()
    ctx.filter_candidates()

    # Refinement can be unstable for very small graphs in the native backend.
    # Therefore this example runs it only when all graphs are large enough.
    min_nodes = min(
        [graph["num_nodes"] for graph in query_graphs + data_graphs],
        default=0,
    )
    requested_iterations = 3

    if min_nodes >= 6:
        ctx.refine(
            requested_iterations,
            start_view_size=1,
            stop_on_fixed_point=True,
        )
    else:
        ctx.warnings.append(
            "Refinement skipped in this example because at least one graph "
            "has fewer than 6 nodes."
        )
        requested_iterations = 0

    ctx.join(find_first=True)

    result = build_match_result(
        ctx.raw_join_result,
        query_graphs,
        data_graphs,
        steps=ctx.steps,
        warnings=ctx.warnings,
        errors=ctx.errors,
        device=ctx.device_name,
        requested_iterations=requested_iterations,
        executed_iterations=ctx.executed_iterations,
    )

    print(result.summary())
    print()
    print(result.explain())


if __name__ == "__main__":
    main()