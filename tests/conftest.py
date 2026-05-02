import pytest
import sigmo

from sigmo.config import get_default_queue
from sigmo.graph import make_csr_graph, toy_two_node_graph


@pytest.fixture(scope="session")
def q():
    """
    Queue SYCL condivisa tra i test.

    Usa la stessa logica della libreria: GPU se disponibile,
    altrimenti CPU. Lo scope di sessione evita di reinizializzare
    il device per ogni singolo test.
    """
    return get_default_queue()


@pytest.fixture
def ethane_graph():
    """Grafo CSR minimale: etano C-C."""
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
    """Lista di grafi CSR usata dai test low-level."""
    return [ethane_graph]


@pytest.fixture
def sig_simple(q, simple_graphs):
    """Signature coerente con il numero di nodi dei grafi di test."""
    n_nodes = sum(g["num_nodes"] for g in simple_graphs)
    return sigmo.Signature(q, n_nodes + 16, n_nodes + 16)


@pytest.fixture
def cand_simple(q, simple_graphs):
    """Candidates coerente con il numero di nodi dei grafi di test."""
    n_nodes = sum(g["num_nodes"] for g in simple_graphs)
    return sigmo.Candidates(q, n_nodes + 16, n_nodes + 16)


@pytest.fixture(scope="session")
def sample_smarts_file(tmp_path_factory):
    """File SMARTS temporaneo con righe nominate e non nominate."""
    content = "\n".join([
        "C1CCCCC1 Cyclohexane",
        "CC[NH+]CC",
        "[NX3][NX2]=[*] Query_1",
    ])
    fn = tmp_path_factory.mktemp("data") / "test.smarts"
    fn.write_text(content)
    return str(fn)


@pytest.fixture(scope="session")
def invalid_smarts_file(tmp_path_factory):
    """File SMARTS temporaneo con una riga non valida."""
    content = "\n".join([
        "C1CCCCC1 cyclohexane",
        "NOT_A_MOLECULE invalid",
        "CCCC butane",
    ])
    fn = tmp_path_factory.mktemp("data") / "invalid.smarts"
    fn.write_text(content)
    return str(fn)


def assert_match_result(result):
    """Helper comune: controlla che l'oggetto sia un MatchResult-like."""
    assert hasattr(result, "total_matches")
    assert hasattr(result, "matches")
    assert hasattr(result, "steps")
    assert hasattr(result, "warnings")
    assert hasattr(result, "summary")
    assert hasattr(result, "explain")
    assert isinstance(result.summary(), str)
    assert isinstance(result.explain(), str)
    return result