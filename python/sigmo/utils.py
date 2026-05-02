from __future__ import annotations

from typing import Any, Dict, List

from .result import build_match_result


def format_matches(raw_results: Dict[str, Any], q_graphs: List[Dict[str, Any]], d_graphs: List[Dict[str, Any]]):
    """
    Utility retrocompatibile: trasforma il risultato grezzo in MatchResult.

    Preferisci pero' usare direttamente sigmo.match(), sigmo.search() o SIGMoMatcher.
    """
    return build_match_result(raw_results, q_graphs, d_graphs)
