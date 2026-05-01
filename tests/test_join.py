import pytest
import sigmo
from sigmo import _core

def test_join_logic(q):
    """
    Test specifico per il kernel di Join:
    Verifica che dati due grafi identici, il join produca almeno un match.
    """
    from sigmo.graph import make_csr_graph
    
    g = make_csr_graph(
        row_offsets=[0, 1, 2], 
        column_indices=[1, 0], 
        node_labels=[6, 6], 
        edge_labels=[1, 1], 
        name="test_g"
    )
    
    sig = _core.Signature(q, 20, 20)
    cand = _core.Candidates(q, 20, 20)
    gmcr = _core.GMCR(q)
    q.wait()

    _core.generate_csr_signatures(q, [g], sig, "query")
    _core.generate_csr_signatures(q, [g], sig, "data")
    _core.filter_candidates(q, [g], [g], sig, cand)
    q.wait()

    results = _core.join_candidates(q, [g], [g], cand, gmcr, True)
    q.wait()

    assert isinstance(results, dict), "Il Join deve restituire un dizionario"
    assert "num_matches" in results, "Il risultato deve contenere il conteggio dei match"
    assert results["num_matches"] >= 1, f"Atteso almeno 1 match per grafi identici, trovato: {results['num_matches']}"
    
