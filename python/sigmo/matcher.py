from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph import load_molecules
from .pipeline import PipelineContext
from .result import MatchResult

__all__ = [
    "match",
    "match_smarts",
    "search",
    "run_isomorphism",
    "SIGMoMatcher",
]


def match(
    query: str,
    target: str,
    *,
    input_format: str = "auto",
    iterations: int = 1,
    find_first: bool = True,
    device: str = "auto",
    queue: Any = None,
) -> MatchResult:
    """
    Match a single query molecule/pattern against a single target molecule.

    Args:
        query: Query molecule or pattern.
        target: Target molecule.
        input_format: One of "auto", "smarts", or "smiles".
        iterations: Number of requested refinement iterations.
        find_first: If True, return only the first match for each query-target pair
            when supported by the native backend.
        device: SYCL device selector, for example "auto" or "cuda:gpu".
        queue: Optional pre-created dpctl.SyclQueue.

    Returns:
        A MatchResult object containing matches, kernel timings, warnings and
        execution metadata.
    """
    q_graphs = load_molecules([query], input_format=input_format)
    d_graphs = load_molecules([target], input_format=input_format)

    return run_isomorphism(
        q_graphs,
        d_graphs,
        queue=queue,
        device=device,
        find_first=find_first,
        iterations=iterations,
    )


def match_smarts(
    query_smarts: str,
    target_smarts: str,
    **kwargs: Any,
) -> MatchResult:
    """
    Match a SMARTS query against a SMARTS target.

    This is a convenience alias for match(..., input_format="smarts").
    """
    return match(
        query_smarts,
        target_smarts,
        input_format="smarts",
        **kwargs,
    )


def search(
    queries: Any,
    database: Any,
    *,
    input_format: str = "auto",
    iterations: int = 1,
    find_first: bool = True,
    device: str = "auto",
    queue: Any = None,
    strict: bool = False,
) -> MatchResult:
    """
    Run batch subgraph isomorphism over multiple query and data graphs.

    Args:
        queries: File path, chemical string, list of strings, RDKit Mol objects,
            or SIGMo-compatible CSR graphs.
        database: File path, chemical string, list of strings, RDKit Mol objects,
            or SIGMo-compatible CSR graphs.
        input_format: One of "auto", "smarts", or "smiles".
        iterations: Number of requested refinement iterations.
        find_first: If True, return only the first match for each pair when
            supported by the native backend.
        device: SYCL device selector.
        queue: Optional pre-created dpctl.SyclQueue.
        strict: If True, fail on the first invalid input item.

    Returns:
        A MatchResult object.
    """
    q_graphs = load_molecules(
        queries,
        input_format=input_format,
        strict=strict,
    )
    d_graphs = load_molecules(
        database,
        input_format=input_format,
        strict=strict,
    )

    return run_isomorphism(
        q_graphs,
        d_graphs,
        queue=queue,
        device=device,
        find_first=find_first,
        iterations=iterations,
    )


def run_isomorphism(
    q_graphs: List[Dict[str, Any]],
    d_graphs: List[Dict[str, Any]],
    *,
    queue: Any = None,
    device: str = "auto",
    find_first: bool = True,
    iterations: int = 1,
) -> MatchResult:
    """
    Execute SIGMo on already-converted CSR query and data graphs.

    Use this function when you already have CSR graphs and do not need the
    loading/parsing convenience provided by match() or search().
    """
    context = PipelineContext(
        q_graphs,
        d_graphs,
        queue=queue,
        device=device,
    )

    return context.run(
        iterations=iterations,
        find_first=find_first,
    )


class SIGMoMatcher:
    """
    Object-oriented high-level matcher.

    SIGMoMatcher is useful when the same device, input format and execution
    configuration should be reused across multiple experiments.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        queue: Any = None,
        iterations: int = 1,
        find_first: bool = True,
        input_format: str = "auto",
        strict: bool = False,
    ) -> None:
        self.device = device
        self.queue = queue
        self.iterations = int(iterations)
        self.find_first = bool(find_first)
        self.input_format = input_format
        self.strict = bool(strict)

        self.query_graphs: List[Dict[str, Any]] = []
        self.data_graphs: List[Dict[str, Any]] = []

        self.last_context: Optional[PipelineContext] = None
        self.last_result: Optional[MatchResult] = None

    def set_queries(self, queries: Any) -> "SIGMoMatcher":
        """
        Load and store query graphs.
        """
        self.query_graphs = load_molecules(
            queries,
            input_format=self.input_format,
            strict=self.strict,
        )
        return self

    def set_database(self, database: Any) -> "SIGMoMatcher":
        """
        Load and store data graphs.
        """
        self.data_graphs = load_molecules(
            database,
            input_format=self.input_format,
            strict=self.strict,
        )
        return self

    def run(
        self,
        queries: Any = None,
        database: Any = None,
        *,
        iterations: Optional[int] = None,
        find_first: Optional[bool] = None,
    ) -> MatchResult:
        """
        Run SIGMo using stored or newly provided query/data inputs.
        """
        if queries is not None:
            self.set_queries(queries)

        if database is not None:
            self.set_database(database)

        if not self.query_graphs:
            raise ValueError(
                "No query graphs loaded. Use set_queries() or pass queries to run()."
            )

        if not self.data_graphs:
            raise ValueError(
                "No data graphs loaded. Use set_database() or pass database to run()."
            )

        self.last_context = PipelineContext(
            self.query_graphs,
            self.data_graphs,
            queue=self.queue,
            device=self.device,
        )

        self.last_result = self.last_context.run(
            iterations=self.iterations if iterations is None else int(iterations),
            find_first=self.find_first if find_first is None else bool(find_first),
        )

        return self.last_result

    def create_context(self) -> PipelineContext:
        """
        Create a PipelineContext from the currently loaded query/data graphs.

        This exposes the step-by-step pipeline API while preserving the matcher
        configuration.
        """
        if not self.query_graphs or not self.data_graphs:
            raise ValueError("Load query and database graphs before creating a context.")

        self.last_context = PipelineContext(
            self.query_graphs,
            self.data_graphs,
            queue=self.queue,
            device=self.device,
        )

        return self.last_context