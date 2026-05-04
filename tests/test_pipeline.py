import sigmo

from sigmo.graph import load_molecules


def assert_match_result(result):
    """
    Assert that an object behaves like a MatchResult.
    """
    assert hasattr(result, "total_matches")
    assert hasattr(result, "matches")
    assert hasattr(result, "steps")
    assert hasattr(result, "warnings")
    assert hasattr(result, "summary")
    assert hasattr(result, "explain")
    assert isinstance(result.summary(), str)
    assert isinstance(result.explain(), str)
    return result


def test_pipeline_context_step_by_step(q):
    query_graphs = load_molecules(["CC"], input_format="smiles")
    data_graphs = load_molecules(["CCC"], input_format="smiles")

    ctx = sigmo.PipelineContext(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        queue=q,
    )

    ctx.allocate()

    assert ctx.signature is not None
    assert ctx.candidates is not None
    assert ctx.gmcr is not None

    ctx.generate_signatures()

    assert any(step.name == "generate_query_signatures" for step in ctx.steps)
    assert any(step.name == "generate_data_signatures" for step in ctx.steps)

    filter_stats = ctx.filter_candidates()

    assert isinstance(filter_stats, dict)
    assert any(step.name == "filter_candidates" for step in ctx.steps)

    raw_join = ctx.join(find_first=True)

    assert isinstance(raw_join, dict)
    assert raw_join.get("num_matches", 0) >= 1
    assert any(step.name == "join_candidates" for step in ctx.steps)


def test_pipeline_context_run_returns_match_result(q):
    query_graphs = load_molecules(["CC"], input_format="smiles")
    data_graphs = load_molecules(["CCC"], input_format="smiles")

    ctx = sigmo.PipelineContext(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        queue=q,
    )

    result = ctx.run(
        iterations=0,
        find_first=True,
    )

    assert_match_result(result)
    assert result.total_matches >= 1
    assert any(step.name == "filter_candidates" for step in result.steps)
    assert any(step.name == "join_candidates" for step in result.steps)