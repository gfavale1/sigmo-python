import sigmo

query_graphs = sigmo.load_molecules("benchmarks/datasets/query.smarts")
data_graphs = sigmo.load_molecules("benchmarks/datasets/data.smarts")

ctx = sigmo.PipelineContext(query_graphs, data_graphs, device="gpu")
ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(max_iterations=3)
ctx.join(find_first=True)

result = sigmo.result.build_match_result(  # oppure usa direttamente ctx.run() se non ti serve step-by-step
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
