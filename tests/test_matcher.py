import sigmo

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


def test_match_high_level_positive():
    result = sigmo.match(
        query="CC",
        target="CCC",
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    assert_match_result(result)
    assert result.total_matches >= 1
    assert result.query_count == 1
    assert result.data_count == 1
    assert "Matches found" in result.summary()


def test_match_high_level_negative():
    result = sigmo.match(
        query="CO",
        target="CCC",
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    assert_match_result(result)
    assert result.total_matches == 0


def test_run_isomorphism_returns_match_result(q, ethane_graph):
    result = sigmo.run_isomorphism(
        [ethane_graph],
        [ethane_graph],
        queue=q,
        iterations=0,
        find_first=True,
    )

    assert_match_result(result)
    assert result.total_matches >= 1


def test_search_batch_api():
    result = sigmo.search(
        queries=["CC", "CO"],
        database=["CCC", "CCO"],
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    assert_match_result(result)
    assert result.query_count == 2
    assert result.data_count == 2
    assert result.total_matches >= 1


def test_sigmo_matcher_object_api():
    matcher = sigmo.SIGMoMatcher(
        device="auto",
        iterations=0,
        find_first=True,
        input_format="smiles",
    )

    result = matcher.run(
        queries=["CC"],
        database=["CCC"],
    )

    assert_match_result(result)
    assert result.total_matches >= 1
    assert matcher.last_result is result
    assert matcher.last_context is not None


def test_result_export_methods(tmp_path):
    result = sigmo.match(
        query="CC",
        target="CCC",
        input_format="smiles",
        iterations=0,
        find_first=True,
        device="auto",
    )

    csv_path = tmp_path / "matches.csv"
    json_path = tmp_path / "matches.json"

    result.to_csv(csv_path)
    result.to_json(json_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.read_text(encoding="utf-8").strip() != ""
    assert json_path.read_text(encoding="utf-8").strip() != ""