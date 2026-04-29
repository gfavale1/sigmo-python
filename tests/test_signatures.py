import pytest
import sigmo


def test_generate_csr_signatures_data(q, simple_graphs, sig_simple):
    # ORDINE: queue, graphs, signatures_object, scope
    result = sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")

    assert isinstance(result, dict)
    assert result["num_graphs"] == 1
    assert result["total_nodes"] == 2
    assert result["scope"] == "data"
    # Se arriviamo qui senza SegFault, la memoria USM è stata scritta correttamente
    assert result.get("allocated_bytes", 0) > 0

def test_generate_csr_signatures_query(q, simple_graphs, sig_simple):
    result = sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "query")

    assert result["num_graphs"] == 1
    assert result["scope"] == "query"

# --- TEST DI REFINEMENT ---

def test_refine_csr_signatures_data(q, simple_graphs, sig_simple):
    VIEW_SIZE = 1
    # Prima generiamo (necessario per il refine)
    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")
    
    # Poi raffiniamo
    result = sigmo.refine_csr_signatures(q, simple_graphs, sig_simple, "data", VIEW_SIZE)

    assert result["num_graphs"] == 1
    assert result["scope"] == "data"
    # Verifica che il campo aggiunto nel wrapper esista
    assert "view_size" in result or result.get("view_size") == VIEW_SIZE

# --- TEST DI ROBUSTEZZA (FAILURE MODES) ---

def test_generate_csr_signatures_empty(q):
    # Anche con lista vuota, serve un oggetto Signature (anche minimo)
    sig = sigmo.Signature(q, 0, 0)
    result = sigmo.generate_csr_signatures(q, [], sig, "data")

    assert result["num_graphs"] == 0
    assert result["total_nodes"] == 0

def test_refine_csr_signatures_invalid_scope(q, simple_graphs, sig_simple):
    # Verifica che il C++ lanci l'eccezione correttamente per scope errati
    with pytest.raises(RuntimeError) as excinfo:
        sigmo.refine_csr_signatures(q, simple_graphs, sig_simple, "ciao", 1)
    
    assert "scope" in str(excinfo.value).lower()

def test_generate_csr_signatures_invalid_graph(q, invalid_graphs):
    # Calcolo nodi per il grafo invalido
    n_nodes = sum(g["num_nodes"] for g in invalid_graphs)
    sig = sigmo.Signature(q, n_nodes, n_nodes)
    
    # Verifica che le validazioni in api.cpp (to_sigmo_csr_graphs) blocchino il SegFault
    with pytest.raises(RuntimeError):
        sigmo.generate_csr_signatures(q, invalid_graphs, sig, "data")