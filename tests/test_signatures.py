import pytest
import sigmo


def test_generate_csr_signatures_data(dev, simple_graphs):
    result = sigmo.generate_csr_signatures(dev, simple_graphs, "data")

    assert isinstance(result, dict)
    assert result["num_graphs"] == 1
    assert result["total_nodes"] == 2
    assert result["total_edges"] == 2
    assert result["allocated_bytes"] > 0
    assert result["scope"] == "data"


def test_generate_csr_signatures_query(dev, simple_graphs):
    result = sigmo.generate_csr_signatures(dev, simple_graphs, "query")

    assert isinstance(result, dict)
    assert result["num_graphs"] == 1
    assert result["total_nodes"] == 2
    assert result["total_edges"] == 2
    assert result["allocated_bytes"] > 0
    assert result["scope"] == "query"


def test_refine_csr_signatures_data(dev, simple_graphs):
    result = sigmo.refine_csr_signatures(dev, simple_graphs, "data", 1)

    assert isinstance(result, dict)
    assert result["num_graphs"] == 1
    assert result["total_nodes"] == 2
    assert result["total_edges"] == 2
    assert result["allocated_bytes"] > 0
    assert result["scope"] == "data"
    assert result["view_size"] == 1


def test_refine_csr_signatures_query(dev, simple_graphs):
    result = sigmo.refine_csr_signatures(dev, simple_graphs, "query", 1)

    assert isinstance(result, dict)
    assert result["num_graphs"] == 1
    assert result["total_nodes"] == 2
    assert result["total_edges"] == 2
    assert result["allocated_bytes"] > 0
    assert result["scope"] == "query"
    assert result["view_size"] == 1


def test_generate_csr_signatures_empty(dev):
    result = sigmo.generate_csr_signatures(dev, [], "data")

    assert result["num_graphs"] == 0
    assert result["total_nodes"] == 0
    assert result["total_edges"] == 0
    assert result["allocated_bytes"] == 0


def test_refine_csr_signatures_invalid_scope(dev, simple_graphs):
    with pytest.raises(RuntimeError):
        sigmo.refine_csr_signatures(dev, simple_graphs, "ciao", 1)


def test_generate_csr_signatures_invalid_graph(dev, invalid_graphs):
    with pytest.raises(RuntimeError):
        sigmo.generate_csr_signatures(dev, invalid_graphs, "data")
