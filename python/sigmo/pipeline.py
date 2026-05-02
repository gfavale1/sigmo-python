from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from . import _core
from .config import describe_queue, get_sycl_queue
from .result import KernelStep, MatchResult, build_match_result


class PipelineContext:
    """
    API avanzata: permette di eseguire la pipeline SIGMo step-by-step.

    Serve a chi vuole controllare i singoli kernel senza manipolare direttamente
    tutti gli oggetti C++ esposti dal binding.
    """

    def __init__(
        self,
        query_graphs: List[Dict[str, Any]],
        data_graphs: List[Dict[str, Any]],
        *,
        queue: Any = None,
        device: str = "auto",
        memory_padding: int = 16,
    ) -> None:
        self.query_graphs = list(query_graphs)
        self.data_graphs = list(data_graphs)
        self.queue = queue if queue is not None else get_sycl_queue(device)
        self.memory_padding = int(memory_padding)

        self.signature = None
        self.candidates = None
        self.gmcr = None

        self.steps: List[KernelStep] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.executed_iterations = 0
        self.raw_join_result: Dict[str, Any] = {}

    @property
    def device_name(self) -> str:
        return describe_queue(self.queue)

    @property
    def total_query_nodes(self) -> int:
        return sum(int(g.get("num_nodes", 0)) for g in self.query_graphs)

    @property
    def total_data_nodes(self) -> int:
        return sum(int(g.get("num_nodes", 0)) for g in self.data_graphs)

    def allocate(self) -> "PipelineContext":
        """Alloca gli oggetti stateful della pipeline."""
        if not self.query_graphs:
            raise ValueError("Nessun query graph fornito.")
        if not self.data_graphs:
            raise ValueError("Nessun data graph fornito.")

        total_q = self.total_query_nodes + self.memory_padding
        total_d = self.total_data_nodes + self.memory_padding

        self.signature = _core.Signature(self.queue, total_d, total_q)
        self.candidates = _core.Candidates(self.queue, total_q, total_d)
        self.gmcr = _core.GMCR(self.queue)
        self.queue.wait()
        return self

    def generate_signatures(self) -> "PipelineContext":
        self._ensure_allocated()
        self._run_step(
            "generate_query_signatures",
            lambda: _core.generate_csr_signatures(self.queue, self.query_graphs, self.signature, "query"),
        )
        self._run_step(
            "generate_data_signatures",
            lambda: _core.generate_csr_signatures(self.queue, self.data_graphs, self.signature, "data"),
        )
        return self

    def filter_candidates(self) -> Dict[str, Any]:
        self._ensure_allocated()

        stats = self._run_step(
            "filter_candidates",
            lambda: _core.filter_candidates(
                self.queue,
                self.query_graphs,
                self.data_graphs,
                self.signature,
                self.candidates,
            ),
        )

        self.last_candidates_count = stats.get(
            "candidates_count",
            stats.get("total_candidates"),
        )

        return stats

    def refine_once(self, iteration: int) -> Dict[str, Any]:
        """Esegue una singola iterazione di raffinamento firme+candidati."""
        self._ensure_allocated()
        view_size = 1 + int(iteration)

        def _refine_all() -> Dict[str, Any]:
            q_stats = _core.refine_csr_signatures(self.queue, self.query_graphs, self.signature, "query", view_size)
            self.queue.wait()
            d_stats = _core.refine_csr_signatures(self.queue, self.data_graphs, self.signature, "data", view_size)
            self.queue.wait()
            c_stats = _core.refine_candidates(
                self.queue,
                self.query_graphs,
                self.data_graphs,
                self.signature,
                self.candidates,
            )
            self.queue.wait()

            merged = {
                "view_size": view_size,
                "query_signature_stats": q_stats,
                "data_signature_stats": d_stats,
                "candidate_stats": c_stats,
            }
            if isinstance(c_stats, dict):
                for key in ("candidates_count", "total_candidates", "num_candidates", "candidate_count"):
                    if key in c_stats:
                        merged["candidates_count"] = c_stats[key]
                        break
            return merged

        stats = self._run_step(f"refine_iteration_{iteration + 1}", _refine_all)
        self.executed_iterations += 1
        return stats

    def _record_step_safe(self, name: str, duration_seconds: float, stats: dict):
        """
        Registra uno step kernel nella lista self.steps.

        Questo helper è difensivo: prova prima a usare KernelStep della
        nuova interfaccia. Se la struttura del dataclass cambia, non fa
        fallire la pipeline, ma salva comunque un warning.
        """
        try:
            from .result import KernelStep

            try:
                step = KernelStep(
                    name=name,
                    duration_seconds=duration_seconds,
                    stats=stats or {},
                )
            except TypeError:
                step = KernelStep(
                    name=name,
                    elapsed_seconds=duration_seconds,
                    stats=stats or {},
                )

            self.steps.append(step)

        except Exception as exc:
            self.warnings.append(
                f"Impossibile registrare lo step {name}: {exc}"
            )

    def refine(
        self,
        iterations: int = 1,
        *,
        start_view_size: int = 1,
        stop_on_fixed_point: bool = True,
    ) -> Dict[str, Any]:
        """
        Esegue il refinement chiamando esplicitamente i tre kernel:

        1. refine_csr_signatures(query)
        2. refine_csr_signatures(data)
        3. refine_candidates()

        Questa versione evita letture dirette potenzialmente instabili da Candidates,
        usando soltanto le statistiche restituite da refine_candidates().
        """
        self._ensure_allocated()

        max_iterations = int(iterations)

        if max_iterations <= 0:
            self.executed_iterations = 0
            return {
                "executed_iterations": 0,
                "reason": "iterations <= 0",
            }

        current_count = getattr(self, "last_candidates_count", None)
        last_stats: Dict[str, Any] = {}
        self.executed_iterations = 0

        for i in range(max_iterations):
            view_size = int(start_view_size) + i

            rq_stats = self._run_step(
                "refine_query_signatures",
                lambda view_size=view_size: _core.refine_csr_signatures(
                    self.queue,
                    self.query_graphs,
                    self.signature,
                    "query",
                    view_size,
                ),
            )

            rd_stats = self._run_step(
                "refine_data_signatures",
                lambda view_size=view_size: _core.refine_csr_signatures(
                    self.queue,
                    self.data_graphs,
                    self.signature,
                    "data",
                    view_size,
                ),
            )

            rc_stats = self._run_step(
                "refine_candidates",
                lambda: _core.refine_candidates(
                    self.queue,
                    self.query_graphs,
                    self.data_graphs,
                    self.signature,
                    self.candidates,
                ),
            )

            new_count = rc_stats.get("candidates_count")
            last_stats = rc_stats
            self.executed_iterations = i + 1

            if new_count is not None:
                self.last_candidates_count = new_count

            if (
                stop_on_fixed_point
                and current_count is not None
                and new_count is not None
                and new_count == current_count
            ):
                self.warnings.append(
                    "Refinement fermato al punto fisso dopo "
                    f"{i + 1} iterazione/i: candidates_count={new_count}."
                )
                break

            current_count = new_count

        return {
            "executed_iterations": self.executed_iterations,
            "last_candidates_count": getattr(self, "last_candidates_count", None),
            "last_stats": last_stats,
        }

    def join(self, *, find_first: bool = True) -> Dict[str, Any]:
        self._ensure_allocated()
        self.raw_join_result = self._run_step(
            "join_candidates",
            lambda: _core.join_candidates(
                self.queue,
                self.query_graphs,
                self.data_graphs,
                self.candidates,
                self.gmcr,
                find_first,
            ),
        )
        return self.raw_join_result

    def run(
        self,
        *,
        iterations: int = 1,
        find_first: bool = True,
        disable_refine_for_small_graphs: bool = True,
        min_refine_nodes: int = 6,
    ) -> MatchResult:
        """Esegue l'intera pipeline e restituisce MatchResult."""
        requested_iterations = int(iterations)
        effective_iterations = requested_iterations

        if disable_refine_for_small_graphs and effective_iterations > 0:
            min_q = min((int(g.get("num_nodes", 0)) for g in self.query_graphs), default=0)
            min_d = min((int(g.get("num_nodes", 0)) for g in self.data_graphs), default=0)

            if min_q < min_refine_nodes or min_d < min_refine_nodes:
                effective_iterations = 0
                self.warnings.append(
                    "Refinement disabilitato: almeno un grafo ha meno di "
                    f"{min_refine_nodes} nodi. La pipeline resta stabile ma meno selettiva."
                )

        try:
            self.allocate()
            self.generate_signatures()
            self.filter_candidates()

            if effective_iterations > 0:
                self.refine(
                    effective_iterations,
                    start_view_size=1,
                    stop_on_fixed_point=True,
                )

            self.join(find_first=find_first)

        except Exception as exc:
            self.errors.append(str(exc))
            self.raw_join_result = {
                "num_matches": 0,
                "error": str(exc),
                "status": "pipeline_failed",
            }

        return build_match_result(
            self.raw_join_result,
            self.query_graphs,
            self.data_graphs,
            steps=self.steps,
            warnings=self.warnings,
            errors=self.errors,
            device=self.device_name,
            requested_iterations=requested_iterations,
            executed_iterations=self.executed_iterations,
        )

    def current_candidates_count(self, *, default: Optional[int] = 0) -> Optional[int]:
        if self.candidates is None:
            return default
        try:
            return int(self.candidates.get_candidates_count(0))
        except Exception:
            return default

    def _run_step(self, name: str, fn) -> Dict[str, Any]:
        start = time.perf_counter()
        stats = fn()
        self.queue.wait()
        elapsed = time.perf_counter() - start
        if stats is None:
            stats = {}
        if not isinstance(stats, dict):
            stats = {"value": stats}
        self.steps.append(KernelStep(name=name, elapsed_seconds=elapsed, stats=stats))
        return stats

    def _ensure_allocated(self) -> None:
        if self.signature is None or self.candidates is None or self.gmcr is None:
            raise RuntimeError("Pipeline non allocata. Chiama allocate() prima di eseguire i kernel.")


def _candidate_count_from_stats(stats: Dict[str, Any]) -> Optional[int]:
    for key in ("candidates_count", "total_candidates", "num_candidates", "candidate_count"):
        if key in stats:
            try:
                return int(stats[key])
            except Exception:
                return None
    nested = stats.get("candidate_stats")
    if isinstance(nested, dict):
        return _candidate_count_from_stats(nested)
    return None
