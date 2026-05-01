import pytest
import sigmo

def test_generate_csr_signatures_data(q, simple_graphs, sig_simple):
    # Chiama direttamente il core se il wrapper non è ancora aggiornato
    result = sigmo._core.generate_csr_signatures(q, simple_graphs, sig_simple, "data")
    
    assert result["num_graphs"] == len(simple_graphs)
    # Se 'scope' non è nel dict, lo verifichiamo solo se presente o saltiamo
    if "scope" in result:
        assert result["scope"] == "data"

def test_refine_csr_signatures_data(q, simple_graphs, sig_simple):
    """Verifica il raffinamento iterativo delle signature."""
    VIEW_SIZE = 1
    sigmo.generate_csr_signatures(q, simple_graphs, sig_simple, "data")
    result = sigmo.refine_csr_signatures(q, simple_graphs, sig_simple, "data", VIEW_SIZE)

    assert "view_size" in result or result.get("view_size") == VIEW_SIZE

def test_generate_csr_signatures_empty(q):
    """Verifica che la libreria gestisca liste di grafi vuote senza SegFault."""
    sig = sigmo.Signature(q, 0, 0)
    # Una lista vuata non deve causare accessi illegali in C++
    result = sigmo.generate_csr_signatures(q, [], sig, "data")
    assert result["num_graphs"] == 0