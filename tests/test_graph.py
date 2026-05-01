import pytest
from sigmo.graph import smarts_to_csr

def test_smarts_loading_with_names(sample_smarts_file):
    """Verifica che i nomi vengano estratti correttamente o generati."""
    graphs = smarts_to_csr(sample_smarts_file)
    
    assert len(graphs) == 3
    # Verifica il nome esplicito
    assert "Benzene" in graphs[0]["name"]
    # Verifica il nome autogenerato per la riga senza nome
    assert "ID:1" in graphs[1]["name"]
    # Verifica che i dati CSR siano presenti
    assert "row_offsets" in graphs[0]
    assert len(graphs[0]["node_labels"]) > 0

def test_invalid_smarts():
    """Verifica che molecole scritte male vengano ignorate senza crash."""
    import os
    with open("invalid.smarts", "w") as f:
        f.write("C1CCCCC1\nNOT_A_MOLECULE\nCCCC")
    
    graphs = smarts_to_csr("invalid.smarts")
    assert len(graphs) == 2 # La riga centrale deve essere stata scartata
    os.remove("invalid.smarts")