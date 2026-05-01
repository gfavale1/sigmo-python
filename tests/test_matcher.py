import pytest
import sigmo
from sigmo import matcher

def test_simple_match(q):
    """Test di base: Etano (C-C) contro se stesso."""
    graph = {
        "row_offsets": [0, 1, 2],
        "column_indices": [1, 0],
        "node_labels": [6, 6], 
        "edge_labels": [1, 1],
        "num_nodes": 2,
        "name": "ethane"
    }
    
    results = matcher.run_isomorphism([graph], [graph], queue=q, iterations=0)
    
    print(f"\nDEBUG - Match trovati: {results.get('num_matches')}")
    assert results['num_matches'] >= 1