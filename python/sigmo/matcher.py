from __future__ import annotations

from typing import Any, Dict, List, Optional

from .graph import load_molecules, smarts_to_csr_from_string
from .pipeline import PipelineContext
from .result import MatchResult
from .validation import validate_result_with_rdkit

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
    validate_with_rdkit: bool = False,
) -> MatchResult:
    """
    Entry point piu' semplice: confronta una query con un target.

    Esempio:
        result = sigmo.match("c1ccccc1", "CCOC(=O)c1ccccc1")
        print(result.summary())
    """
    q_graphs = load_molecules([query], input_format=input_format)
    d_graphs = load_molecules([target], input_format=input_format)
    return run_isomorphism(q_graphs, d_graphs, queue=queue, device=device, find_first=find_first, iterations=iterations, validate_with_rdkit=validate_with_rdkit)


def match_smarts(query_smarts: str, target_smarts: str, **kwargs: Any) -> MatchResult:
    """Alias esplicito per utenti che lavorano con SMARTS."""
    return match(query_smarts, target_smarts, input_format="smarts", **kwargs)


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
    validate_with_rdkit: bool = False,
) -> MatchResult:
    """
    Entry point batch: accetta file, liste, stringhe, RDKit Mol o grafi CSR.
    """
    q_graphs = load_molecules(queries, input_format=input_format, strict=strict)
    d_graphs = load_molecules(database, input_format=input_format, strict=strict)
    return run_isomorphism(q_graphs, d_graphs, queue=queue, device=device, find_first=find_first, iterations=iterations, validate_with_rdkit=validate_with_rdkit)


def run_isomorphism(
    q_graphs,
    d_graphs,
    *,
    queue=None,
    device="auto",
    find_first=True,
    iterations=1,
    validate_with_rdkit=False,
):
    ctx = PipelineContext(q_graphs, d_graphs, queue=queue, device=device)

    result = ctx.run(
        iterations=iterations,
        find_first=find_first,
    )

    if validate_with_rdkit:
        result = validate_result_with_rdkit(
            result,
            q_graphs,
            d_graphs,
        )

    return result


class SIGMoMatcher:
    """
    Interfaccia object-oriented ad alta usabilita'.

    Consigliata quando si vuole riusare la stessa configurazione device/iterazioni
    su piu' esperimenti.
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
        validate_with_rdkit: bool = False,
    ) -> None:
        self.device = device
        self.queue = queue
        self.iterations = int(iterations)
        self.find_first = bool(find_first)
        self.input_format = input_format
        self.strict = strict
        self.validate_with_rdkit = bool(validate_with_rdkit)
        self.query_graphs: List[Dict[str, Any]] = []
        self.data_graphs: List[Dict[str, Any]] = []
        self.last_context: Optional[PipelineContext] = None
        self.last_result: Optional[MatchResult] = None

    def set_queries(self, queries: Any) -> "SIGMoMatcher":
        self.query_graphs = load_molecules(queries, input_format=self.input_format, strict=self.strict)
        return self

    def set_database(self, database: Any) -> "SIGMoMatcher":
        self.data_graphs = load_molecules(database, input_format=self.input_format, strict=self.strict)
        return self

    def run(
        self,
        queries: Any = None,
        database: Any = None,
        *,
        iterations: Optional[int] = None,
        find_first: Optional[bool] = None,
    ) -> MatchResult:
        if queries is not None:
            self.set_queries(queries)
        if database is not None:
            self.set_database(database)

        if not self.query_graphs:
            raise ValueError("Nessuna query caricata. Usa set_queries() oppure passa queries a run().")
        if not self.data_graphs:
            raise ValueError("Nessun database caricato. Usa set_database() oppure passa database a run().")

        self.last_context = PipelineContext(self.query_graphs, self.data_graphs, queue=self.queue, device=self.device)
        self.last_result = self.last_context.run(
            iterations=self.iterations if iterations is None else int(iterations),
            find_first=self.find_first if find_first is None else bool(find_first),
            )
        if getattr(self, "validate_with_rdkit", False):
            self.last_result = validate_result_with_rdkit(
                self.last_result,
                self.query_graphs,
                self.data_graphs,
            )
        return self.last_result

    # Metodi avanzati: espongono i singoli step in modo controllato.
    def create_context(self) -> PipelineContext:
        if not self.query_graphs or not self.data_graphs:
            raise ValueError("Carica prima query e database.")
        self.last_context = PipelineContext(self.query_graphs, self.data_graphs, queue=self.queue, device=self.device)
        return self.last_context
