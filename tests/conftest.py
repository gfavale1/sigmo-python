import pytest
import dpctl
import sigmo
import numpy as np
from sigmo.config import get_default_queue
from sigmo.graph import toy_two_node_graph

@pytest.fixture(scope="session")
def q():
    """
    Usa la logica centralizzata di config.py. 
    L'uso di scope="session" garantisce che la GPU venga inizializzata una sola volta per tutti i test.
    """
    return get_default_queue()

@pytest.fixture
def simple_graphs():
    """Restituisce un grafo CSR standard con tipi di dati espliciti."""
    g = toy_two_node_graph()
    # Assicuriamoci che i dati siano int32/uint32 per il C++
    return [g]

@pytest.fixture
def sig_simple(q, simple_graphs):
    """Crea una Signature basata sul numero reale di nodi."""
    n_nodes = sum(g["num_nodes"] for g in simple_graphs)
    # Importante: verificare se il binding si aspetta (q, query_nodes, target_nodes)
    return sigmo._core.Signature(q, n_nodes, n_nodes)

@pytest.fixture
def cand_simple(q, simple_graphs):
    """Fixture aggiuntiva per l'oggetto Candidates, spesso dimenticata."""
    n_nodes = sum(g["num_nodes"] for g in simple_graphs)
    return sigmo._core.Candidates(q, n_nodes, n_nodes)

@pytest.fixture(scope="session")
def sample_smarts_file(tmp_path_factory):
    """Crea un file SMARTS temporaneo per i test di caricamento."""
    content = "C1CCCCC1 Benzene\nCC[NH+]CC\n[NX3][NX2]=[*] Query_1"
    fn = tmp_path_factory.mktemp("data") / "test.smarts"
    fn.write_text(content)
    return str(fn)