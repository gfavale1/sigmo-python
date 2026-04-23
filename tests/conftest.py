import pytest
import dpctl

from sigmo.graph import toy_two_node_graph, make_csr_graph


@pytest.fixture(scope="module")
def dev():
    return dpctl.SyclDevice("cuda:gpu")


@pytest.fixture
def simple_graphs():
    return [toy_two_node_graph()]


@pytest.fixture
def invalid_graphs():
    return [
        make_csr_graph(
            row_offsets=[0, 1],   # sbagliato: dovrebbe essere num_nodes + 1
            column_indices=[1, 0],
            node_labels=[0, 1],
            edge_labels=[0, 0],
            num_nodes=2,
        )
    ]