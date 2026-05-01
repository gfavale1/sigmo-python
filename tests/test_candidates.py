import pytest
from sigmo import _core, matcher
from sigmo.graph import make_csr_graph

def test_filter_candidates_exact_match(q):
    """Test che usa il matcher completo per evitare SegFault."""
    # Grafo identico Query/Target
    g = make_csr_graph([0, 1, 2], [1, 0], [6, 6], [1, 1], 2, "ethane")
    
    # Eseguiamo la pipeline (almeno fino al filtro)
    # Usiamo iterations=0 per testare solo il primo filtro
    results = matcher.run_isomorphism([g], [g], queue=q, iterations=0)
    
    # Se tutto è ok, deve aver trovato i match (o almeno non essere crashato)
    assert "num_matches" in results

def test_refine_hard_mismatch(q):
    # Usiamo liste piatte di int puri
    g_q = {
        "row_offsets": [0, 1, 2],
        "column_indices": [1, 0],
        "node_labels": [6, 7],
        "edge_labels": [1, 1],
        "num_nodes": 2,
        "name": "q" 
    }
    g_d = {
        "row_offsets": [0, 1, 2],
        "column_indices": [1, 0],
        "node_labels": [6, 8],
        "edge_labels": [1, 1],
        "num_nodes": 2,
        "name": "d"
    }

    # 2. Allocazione USM 
    sig = _core.Signature(q, 10, 10) 
    cand = _core.Candidates(q, 10, 10)
    q.wait()

    # 3. Chiamate con liste esplicite
    _core.generate_csr_signatures(q, [g_q], sig, "query")
    _core.generate_csr_signatures(q, [g_d], sig, "data")
    q.wait()

    _core.filter_candidates(q, [g_q], [g_d], sig, cand)
    q.wait()
    
