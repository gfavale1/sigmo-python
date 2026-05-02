import pytest
import sigmo


def test_generate_csr_signatures_data(q, simple_graphs, sig_simple):
    result = sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")

    assert isinstance(result, dict)
    assert result.get("num_graphs") == len(simple_graphs)
    if "scope" in result:
        assert result["scope"] == "data"


def test_refine_csr_signatures_data(q, simple_graphs, sig_simple):
    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")
    result = sigmo.refine_csr_signatures(q, simple_graphs, sig_simple, "data", 1)

    assert isinstance(result, dict)
    if "view_size" in result:
        assert result["view_size"] == 1


def test_filter_candidates_exact_match_low_level(q, simple_graphs, sig_simple, cand_simple):
    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "query")
    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")
    q.wait()

    result = sigmo.filter_candidates(
        q,
        simple_graphs,
        simple_graphs,
        sig_simple,
        cand_simple,
    )
    q.wait()

    assert isinstance(result, dict)
    assert result.get("total_candidates", result.get("candidates_count", 1)) >= 1


def test_join_candidates_exact_match_low_level(q, simple_graphs, sig_simple, cand_simple):
    gmcr = sigmo.GMCR(q)

    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "query")
    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")
    sigmo.filter_candidates(q, simple_graphs, simple_graphs, sig_simple, cand_simple)
    q.wait()

    result = sigmo.join_candidates(
        q,
        simple_graphs,
        simple_graphs,
        cand_simple,
        gmcr,
        True,
    )
    q.wait()

    assert isinstance(result, dict)
    assert "num_matches" in result
    assert result["num_matches"] >= 1


def test_generate_csr_signatures_empty_does_not_crash(q):
    sig = sigmo.Signature(q, 0, 0)
    result = sigmo.generate_csr_signatures(q, [], sig, "data")
    assert result.get("num_graphs") == 0