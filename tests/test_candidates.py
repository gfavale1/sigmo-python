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

    query_graphs = [{
        "row_offsets": [0, 1, 1], 
        "column_indices": [1], 
        "node_labels": [10, 20], # Etichette arbitrarie
        "edge_labels": [5], 
        "num_nodes": 2
    }]
    
    data_graphs = query_graphs 

    sig = Signature(q, 2, 2)
    cand = Candidates(q, 2, 2)

    try:
        stats = sigmo.filter_candidates(q, query_graphs, data_graphs, sig, cand)
        
        assert stats["num_query_graphs"] == 1
        assert stats["num_data_graphs"] == 1
        assert stats["total_query_nodes"] == 2
        assert stats["total_data_nodes"] == 2
        
        assert stats["candidates_count"] >= 2
        
        assert stats["allocated_bytes"] > 0

    except Exception as e:
        pytest.fail(f"Il filtro ha sollevato un'eccezione inaspettata: {e}")

def test_filter_mismatch_labels():
    """Verifica che etichette diverse producano zero candidati."""
    q = dpctl.SyclQueue("gpu")
    
    q_graphs = [{"row_offsets": [0, 0], "column_indices": [], "node_labels": [10], "edge_labels": [], "num_nodes": 1}]
    d_graphs = [{"row_offsets": [0, 0], "column_indices": [], "node_labels": [99], "edge_labels": [], "num_nodes": 1}]
    
    sig = Signature(q, 1, 1)
    cand = Candidates(q, 1, 1)
    
    stats = sigmo.filter_candidates(q, q_graphs, d_graphs, sig, cand)
    
    assert stats["candidates_count"] == 0

def test_refine_exact_match_preservation():
    q = dpctl.SyclQueue("gpu")
    graph = [{
        "row_offsets": [0, 2, 4, 6],
        "column_indices": [1, 2, 0, 2, 0, 1],
        "node_labels": [10, 10, 10],
        "edge_labels": [1, 1, 1, 1, 1, 1],
        "num_nodes": 3
    }]
    
    sig = Signature(q, 3, 3)
    cand = Candidates(q, 3, 3)
    
    sigmo.filter_candidates(q, graph, graph, sig, cand)
    
    stats = sigmo.refine_candidates(q, graph, graph, sig, cand)
    
    assert stats["candidates_count"] == 9

def test_refine_hard_mismatch_final():
    q = dpctl.SyclQueue("gpu")
    
    query = [{"row_offsets": [0, 1, 2], "column_indices": [1, 0], "node_labels": [1, 2], "edge_labels": [1, 1], "num_nodes": 2}]
    target = [{"row_offsets": [0, 1, 2], "column_indices": [1, 0], "node_labels": [1, 3], "edge_labels": [1, 1], "num_nodes": 2}]

    sig = Signature(q, 2, 2)
    cand = Candidates(q, 2, 2)

    res_base = sigmo.filter_candidates(q, query, target, sig, cand)
    count_base = res_base["candidates_count"] # Qui DEVE essere 1 (il nodo '1' matcha '1')
    
    res_refine = sigmo.refine_candidates(q, query, target, sig, cand)
    count_refine = res_refine["candidates_count"] # Qui DEVE essere 0 (perché i vicini 2 e 3 non matchano)

    print(f"DEBUG TEST: Base={count_base}, Refine={count_refine}")
    
    assert count_refine < count_base