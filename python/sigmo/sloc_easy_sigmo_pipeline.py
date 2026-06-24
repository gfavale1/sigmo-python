import sigmo
from sigmo.result import build_match_result

query_graphs = sigmo.load_molecules(
    "benchmarks/datasets/query.smarts",
    input_format="smarts",
)

data_graphs = sigmo.load_molecules(
    "benchmarks/datasets/data.smarts",
    input_format="smarts",
)

ctx = sigmo.PipelineContext(query_graphs, data_graphs, device="auto")

ctx.allocate()
ctx.generate_signatures()
ctx.filter_candidates()
ctx.refine(7, start_view_size=1, stop_on_fixed_point=False)

raw_join = ctx.join(find_first=True)

result = build_match_result(
    raw_join,
    query_graphs,
    data_graphs,
    steps=ctx.steps,
    warnings=ctx.warnings,
    errors=ctx.errors,
    device=ctx.device_name,
    requested_iterations=7,
    executed_iterations=ctx.executed_iterations,
)

print(result.summary())