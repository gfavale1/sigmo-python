import pytest

from sigmo.graph import (
    load_molecules,
    make_csr_graph,
    smarts_to_csr,
    smarts_to_csr_from_string,
    toy_two_node_graph,
)


def assert_valid_csr_graph(graph):
    assert "row_offsets" in graph
    assert "column_indices" in graph
    assert "node_labels" in graph
    assert "edge_labels" in graph
    assert "num_nodes" in graph
    assert "name" in graph
    assert graph["num_nodes"] == len(graph["node_labels"])
    assert len(graph["row_offsets"]) == graph["num_nodes"] + 1
    assert len(graph["column_indices"]) == len(graph["edge_labels"])


def test_make_csr_graph_builds_standard_dict():
    graph = make_csr_graph(
        row_offsets=[0, 1, 2],
        column_indices=[1, 0],
        node_labels=[6, 6],
        edge_labels=[1, 1],
        name="ethane",
    )

    assert_valid_csr_graph(graph)
    assert graph["name"] == "ethane"
    assert graph["num_nodes"] == 2


def test_toy_two_node_graph_is_valid():
    graph = toy_two_node_graph()
    assert_valid_csr_graph(graph)
    assert graph["num_nodes"] == 2


def test_smarts_to_csr_from_string_smiles():
    graph = smarts_to_csr_from_string("CC")
    assert_valid_csr_graph(graph)
    assert graph["num_nodes"] == 2
    assert graph["node_labels"] == [6, 6]


def test_smarts_to_csr_from_string_aromatic_is_stable():
    """
    Regression test: i legami aromatici non devono usare label non supportate
    dal backend, perché in passato label esplicite come 12 potevano causare
    segmentation fault lato C++/SYCL.
    """
    graph = smarts_to_csr_from_string("c1ccccc1")
    assert_valid_csr_graph(graph)
    assert graph["num_nodes"] == 6
    assert set(graph["edge_labels"]).issubset({1, 2, 3})


def test_smarts_loading_with_names(sample_smarts_file):
    graphs = smarts_to_csr(sample_smarts_file)

    assert len(graphs) == 3
    assert "Cyclohexane" in graphs[0]["name"]
    assert graphs[1]["name"]
    assert graphs[1]["name"] != graphs[0]["name"]
    assert graphs[1]["name"] != graphs[2]["name"]
    if "input" in graphs[1]:
        assert graphs[1]["input"] == "CC[NH+]CC"
    assert "Query_1" in graphs[2]["name"]
    assert_valid_csr_graph(graphs[0])


def test_invalid_smarts_are_skipped_without_crash(invalid_smarts_file):
    graphs = smarts_to_csr(invalid_smarts_file)
    assert len(graphs) == 2
    assert all("NOT_A_MOLECULE" not in g.get("name", "") for g in graphs)


def test_load_molecules_from_list():
    graphs = load_molecules(["CC", "CCC"], input_format="smiles")
    assert len(graphs) == 2
    assert all(g["num_nodes"] > 0 for g in graphs)
    assert all("input" in g for g in graphs)