import dpctl
import pytest
import sigmo 
from sigmo import Signature, Candidates 

def test_filter_candidates_exact_match():
    """
    Verifica che il filtro identifichi correttamente un match esatto tra 
    due grafi identici composti da 2 nodi e 1 arco.
    """
    try:
        q = dpctl.SyclQueue("gpu")
    except:
        pytest.skip("GPU non disponibile, salto il test.")

    # Grafo: (0) --[label 0]--> (1)
    # row_offsets: il nodo 0 ha un vicino (indici 0-1), il nodo 1 ne ha zero (indici 1-1)
    query_graphs = [{
        "row_offsets": [0, 1, 1], 
        "column_indices": [1], 
        "node_labels": [10, 20], # Etichette arbitrarie
        "edge_labels": [5], 
        "num_nodes": 2
    }]
    
    # Target identico alla query
    data_graphs = query_graphs 

    # 3. Allocazione Risorse
    # Usiamo le dimensioni reali (2 nodi) per testare la precisione dei limiti
    sig = Signature(q, 2, 2)
    cand = Candidates(q, 2, 2)

    # 4. Esecuzione Filtro
    try:
        stats = sigmo.filter_candidates(q, query_graphs, data_graphs, sig, cand)
        
        # 5. Asserzioni di Validità
        assert stats["num_query_graphs"] == 1
        assert stats["num_data_graphs"] == 1
        assert stats["total_query_nodes"] == 2
        assert stats["total_data_nodes"] == 2
        
        # In un match esatto con queste etichette, dovremmo avere 
        # almeno i 2 match diretti (Q0->D0 e Q1->D1)
        assert stats["candidates_count"] >= 2
        
        # Verifica memoria USM
        assert stats["allocated_bytes"] > 0

    except Exception as e:
        pytest.fail(f"Il filtro ha sollevato un'eccezione inaspettata: {e}")

def test_filter_mismatch_labels():
    """Verifica che etichette diverse producano zero candidati."""
    q = dpctl.SyclQueue("gpu")
    
    # Query con atomo 10, Target con atomo 99
    q_graphs = [{"row_offsets": [0, 0], "column_indices": [], "node_labels": [10], "edge_labels": [], "num_nodes": 1}]
    d_graphs = [{"row_offsets": [0, 0], "column_indices": [], "node_labels": [99], "edge_labels": [], "num_nodes": 1}]
    
    sig = Signature(q, 1, 1)
    cand = Candidates(q, 1, 1)
    
    stats = sigmo.filter_candidates(q, q_graphs, d_graphs, sig, cand)
    
    # Con etichette diverse, il bitset deve essere vuoto
    assert stats["candidates_count"] == 0