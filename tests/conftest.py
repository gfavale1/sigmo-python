import pytest
import dpctl
import sigmo
# Assicurati di importare le tue utility se sono in un file separato, 
# oppure incollale qui se servono solo ai test.
from sigmo.graph import toy_two_node_graph, make_csr_graph 

@pytest.fixture(scope="module")
def q():
    """Ritorna la coda SYCL per i test."""
    try:
        return dpctl.SyclQueue("gpu")
    except Exception:
        return dpctl.SyclQueue("cpu")

@pytest.fixture
def simple_graphs():
    """Ritorna una lista con un grafo giocattolo valido."""
    return [toy_two_node_graph()]

@pytest.fixture
def sig_simple(q, simple_graphs):
    """Crea l'oggetto Signature per i grafi semplici."""
    n_nodes = sum(g["num_nodes"] for g in simple_graphs)
    return sigmo.Signature(q, n_nodes, n_nodes)

@pytest.fixture
def invalid_graphs():
    """Ritorna un grafo con row_offsets errati per scatenare eccezioni C++."""
    return [
        make_csr_graph(
            row_offsets=[0, 1],   # ERRORE: con 2 nodi servono 3 offsets [0, x, y]
            column_indices=[1, 0],
            node_labels=[0, 1],
            edge_labels=[0, 0],
            num_nodes=2,
        )
    ]