import sigmo
from sigmo.result import build_match_result


def main():
    query_graphs = sigmo.load_molecules("benchmarks/datasets/query.smarts")
    data_graphs = sigmo.load_molecules("benchmarks/datasets/data.smarts")

    # Optional: keep the example lightweight.
    query_graphs = query_graphs[:5]
    data_graphs = data_graphs[:20]

    ctx = sigmo.PipelineContext(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        device="gpu",
    )

    ctx.allocate()
    ctx.generate_signatures()
    ctx.filter_candidates()

    ctx.refine(
        3,
        start_view_size=1,
        stop_on_fixed_point=True,
    )

    ctx.join(find_first=True)

    result = build_match_result(
        ctx.raw_join_result,
        query_graphs,
        data_graphs,
        steps=ctx.steps,
        warnings=ctx.warnings,
        errors=ctx.errors,
        device=ctx.device_name,
        requested_iterations=3,
        executed_iterations=ctx.executed_iterations,
    )

    print(result.summary())
    print()
    print(result.explain())


if __name__ == "__main__":
    main()