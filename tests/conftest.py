import pytest
import sigmo

from sigmo.config import get_default_queue
from sigmo.graph import make_csr_graph


@pytest.fixture(scope="session")
def q():
    """
    Shared SYCL queue used across tests.

    The fixture follows the same device selection logic as the package:
    use the best available device in auto mode. Session scope avoids
    recreating the queue for every test.
    """
    return get_default_queue()


@pytest.fixture
def ethane_graph():
    """
    Minimal CSR graph representing ethane: C-C.
    """
    return make_csr_graph(
        row_offsets=[0, 1, 2],
        column_indices=[1, 0],
        node_labels=[6, 6],
        edge_labels=[1, 1],
        num_nodes=2,
        name="ethane",
    )


@pytest.fixture
def simple_graphs(ethane_graph):
    """
    Small CSR graph list used by low-level binding tests.
    """
    return [ethane_graph]


@pytest.fixture
def sig_simple(q, simple_graphs):
    """
    Signature object sized for the test graphs.
    """
    n_nodes = sum(graph["num_nodes"] for graph in simple_graphs)
    return sigmo.Signature(q, n_nodes + 16, n_nodes + 16)


@pytest.fixture
def cand_simple(q, simple_graphs):
    """
    Candidates object sized for the test graphs.
    """
    n_nodes = sum(graph["num_nodes"] for graph in simple_graphs)
    return sigmo.Candidates(q, n_nodes + 16, n_nodes + 16)


@pytest.fixture(scope="session")
def sample_smarts_file(tmp_path_factory):
    """
    Temporary SMARTS file containing named and unnamed rows.
    """
    content = "\n".join(
        [
            "C1CCCCC1 Cyclohexane",
            "CC[NH+]CC",
            "[NX3][NX2]=[*] Query_1",
        ]
    )

    path = tmp_path_factory.mktemp("data") / "test.smarts"
    path.write_text(content, encoding="utf-8")
    return str(path)


@pytest.fixture(scope="session")
def invalid_smarts_file(tmp_path_factory):
    """
    Temporary SMARTS file containing one invalid row.
    """
    content = "\n".join(
        [
            "C1CCCCC1 cyclohexane",
            "NOT_A_MOLECULE invalid",
            "CCCC butane",
        ]
    )

    path = tmp_path_factory.mktemp("data") / "invalid.smarts"
    path.write_text(content, encoding="utf-8")
    return str(path)


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