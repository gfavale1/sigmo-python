from __future__ import annotations

from typing import Any, Dict, List

from .result import MatchResult, build_match_result


def format_matches(
    raw_results: Dict[str, Any],
    q_graphs: List[Dict[str, Any]],
    d_graphs: List[Dict[str, Any]],
) -> MatchResult:
    """
    Convert a raw native SIGMo result into a MatchResult.

    This helper is kept for backward compatibility. New code should prefer the
    high-level APIs:

        sigmo.match(...)
        sigmo.search(...)
        sigmo.SIGMoMatcher(...)

    Args:
        raw_results: Raw dictionary returned by the native join kernel.
        q_graphs: Query graphs used in the search.
        d_graphs: Data graphs used in the search.

    Returns:
        A MatchResult instance.
    """
    return build_match_result(raw_results, q_graphs, d_graphs)