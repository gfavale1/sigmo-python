import pytest
import sigmo
from sigmo import matcher
from sigmo.graph import smarts_to_csr_from_string

def test_gold_standard_rdkit(q):
    q_graphs = [smarts_to_csr_from_string("C=O")]
    d_graphs = [smarts_to_csr_from_string("CC=O")]
    
    # 2. Esecuzione con iterations=0
    results = matcher.run_isomorphism(q_graphs, d_graphs, q, True, iterations=0)
    
    assert results["num_matches"] >= 1
