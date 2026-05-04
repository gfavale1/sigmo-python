from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from . import _core
from .config import describe_queue, get_sycl_queue
from .result import KernelStep, MatchResult, build_match_result


class PipelineContext:
    """
    Stateful SIGMo kernel pipeline.

    PipelineContext owns the SYCL queue and the native SIGMo objects required
    by the pipeline:

        - Signature
        - Candidates
        - GMCR

    The class exposes both a step-by-step API and a high-level run() method.
    It is intended for users who need more control than the high-level
    match()/search() functions, without manually managing all native objects.

    Important:
        The native objects are intentionally kept alive across all pipeline
        steps. Recreating Signature/Candidates/GMCR between steps would break
        the stateful execution model expected by the native SIGMo backend.
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
        self.last_candidates_count: Optional[int] = None
        self.raw_join_result: Dict[str, Any] = {}

    @property
    def device_name(self) -> str:
        """Return a human-readable name for the selected SYCL device."""
        return describe_queue(self.queue)

    @property
    def total_query_nodes(self) -> int:
        """Return the total number of query nodes."""
        return sum(int(graph.get("num_nodes", 0)) for graph in self.query_graphs)

    @property
    def total_data_nodes(self) -> int:
        """Return the total number of data nodes."""
        return sum(int(graph.get("num_nodes", 0)) for graph in self.data_graphs)

    def allocate(self) -> "PipelineContext":
        """
        Allocate native SIGMo objects.

        This creates Signature, Candidates and GMCR objects and stores them
        inside the context. These objects are reused by all subsequent steps.
        """
        if not self.query_graphs:
            raise ValueError("No query graphs provided.")

        if not self.data_graphs:
            raise ValueError("No data graphs provided.")

        total_q = self.total_query_nodes + self.memory_padding
        total_d = self.total_data_nodes + self.memory_padding

        self.signature = _core.Signature(self.queue, total_d, total_q)
        self.candidates = _core.Candidates(self.queue, total_q, total_d)
        self.gmcr = _core.GMCR(self.queue)

        self.queue.wait()
        return self

    def generate_signatures(self) -> "PipelineContext":
        """
        Generate initial query and data signatures.
        """
        self._ensure_allocated()

        self._run_step(
            "generate_query_signatures",
            lambda: _core.generate_csr_signatures(
                self.queue,
                self.query_graphs,
                self.signature,
                "query",
            ),
        )

        self._run_step(
            "generate_data_signatures",
            lambda: _core.generate_csr_signatures(
                self.queue,
                self.data_graphs,
                self.signature,
                "data",
            ),
        )

        return self

    def filter_candidates(self) -> Dict[str, Any]:
        """
        Run the initial SIGMo candidate filtering step.

        Returns:
            Kernel statistics returned by the native binding.
        """
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

        self.last_candidates_count = _candidate_count_from_stats(stats)
        return stats

    def refine_once(self, iteration: int) -> Dict[str, Any]:
        """
        Run one refinement iteration.

        This method is mainly useful for debugging. The standard refine()
        method records each kernel step separately.
        """
        self._ensure_allocated()

        view_size = 1 + int(iteration)

        def _refine_all() -> Dict[str, Any]:
            query_stats = _core.refine_csr_signatures(
                self.queue,
                self.query_graphs,
                self.signature,
                "query",
                view_size,
            )
            self.queue.wait()

            data_stats = _core.refine_csr_signatures(
                self.queue,
                self.data_graphs,
                self.signature,
                "data",
                view_size,
            )
            self.queue.wait()

            candidate_stats = _core.refine_candidates(
                self.queue,
                self.query_graphs,
                self.data_graphs,
                self.signature,
                self.candidates,
            )
            self.queue.wait()

            merged = {
                "view_size": view_size,
                "query_signature_stats": query_stats,
                "data_signature_stats": data_stats,
                "candidate_stats": candidate_stats,
            }

            count = _candidate_count_from_stats(merged)
            if count is not None:
                merged["candidates_count"] = count
                self.last_candidates_count = count

            return merged

        stats = self._run_step(f"refine_iteration_{iteration + 1}", _refine_all)
        self.executed_iterations += 1
        return stats

    def refine(
        self,
        iterations: int = 1,
        *,
        start_view_size: int = 1,
        stop_on_fixed_point: bool = True,
    ) -> Dict[str, Any]:
        """
        Run iterative signature and candidate refinement.

        Each iteration executes the following native kernels:

            1. refine_csr_signatures(query)
            2. refine_csr_signatures(data)
            3. refine_candidates()

        Args:
            iterations: Maximum number of refinement iterations.
            start_view_size: Initial signature view size.
            stop_on_fixed_point: If True, stop when the candidate count no
                longer changes.

        Returns:
            A summary dictionary containing executed_iterations,
            last_candidates_count and last_stats.
        """
        self._ensure_allocated()

        max_iterations = int(iterations)

        if max_iterations <= 0:
            self.executed_iterations = 0
            return {
                "executed_iterations": 0,
                "reason": "iterations <= 0",
            }

        current_count = self.last_candidates_count
        last_stats: Dict[str, Any] = {}
        self.executed_iterations = 0

        for idx in range(max_iterations):
            view_size = int(start_view_size) + idx

            self._run_step(
                "refine_query_signatures",
                lambda view_size=view_size: _core.refine_csr_signatures(
                    self.queue,
                    self.query_graphs,
                    self.signature,
                    "query",
                    view_size,
                ),
            )

            self._run_step(
                "refine_data_signatures",
                lambda view_size=view_size: _core.refine_csr_signatures(
                    self.queue,
                    self.data_graphs,
                    self.signature,
                    "data",
                    view_size,
                ),
            )

            candidate_stats = self._run_step(
                "refine_candidates",
                lambda: _core.refine_candidates(
                    self.queue,
                    self.query_graphs,
                    self.data_graphs,
                    self.signature,
                    self.candidates,
                ),
            )

            new_count = _candidate_count_from_stats(candidate_stats)
            last_stats = candidate_stats
            self.executed_iterations = idx + 1

            if new_count is not None:
                self.last_candidates_count = new_count

            if (
                stop_on_fixed_point
                and current_count is not None
                and new_count is not None
                and new_count == current_count
            ):
                self.warnings.append(
                    "Refinement stopped at fixed point after "
                    f"{idx + 1} iteration(s): candidates_count={new_count}."
                )
                break

            current_count = new_count

        return {
            "executed_iterations": self.executed_iterations,
            "last_candidates_count": self.last_candidates_count,
            "last_stats": last_stats,
        }

    def join(self, *, find_first: bool = True) -> Dict[str, Any]:
        """
        Run the final SIGMo join/isomorphism step.

        Args:
            find_first: If True, stop after the first match per pair whenever
                the native backend supports this behavior.

        Returns:
            Raw join result returned by the native binding.
        """
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
        """
        Execute the full SIGMo pipeline and return a MatchResult.

        The executed pipeline is:

            allocate()
            generate_signatures()
            filter_candidates()
            refine()        optional
            join()

        Args:
            iterations: Requested number of refinement iterations.
            find_first: Whether the join step should stop at the first match.
            disable_refine_for_small_graphs: If True, skip refinement when
                very small graphs are present. This keeps the high-level API
                stable on datasets containing micro-molecules.
            min_refine_nodes: Minimum number of nodes required to enable
                refinement when disable_refine_for_small_graphs=True.

        Returns:
            A MatchResult containing matches, kernel statistics, warnings and
            errors.
        """
        requested_iterations = int(iterations)
        effective_iterations = requested_iterations
        self.executed_iterations = 0

        if disable_refine_for_small_graphs and effective_iterations > 0:
            min_q = min(
                (int(graph.get("num_nodes", 0)) for graph in self.query_graphs),
                default=0,
            )
            min_d = min(
                (int(graph.get("num_nodes", 0)) for graph in self.data_graphs),
                default=0,
            )

            if min_q < min_refine_nodes or min_d < min_refine_nodes:
                effective_iterations = 0
                self.warnings.append(
                    "Refinement disabled for stability: at least one graph has "
                    f"fewer than {min_refine_nodes} nodes. The pipeline remains "
                    "stable but less selective."
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

    def _run_step(
        self,
        name: str,
        fn: Callable[[], Any],
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        stats = fn()
        self.queue.wait()
        elapsed = time.perf_counter() - start

        if stats is None:
            stats = {}

        if not isinstance(stats, dict):
            stats = {"value": stats}

        self.steps.append(
            KernelStep(
                name=name,
                elapsed_seconds=elapsed,
                stats=stats,
            )
        )

        return stats

    def _ensure_allocated(self) -> None:
        if self.signature is None or self.candidates is None or self.gmcr is None:
            raise RuntimeError(
                "Pipeline is not allocated. Call allocate() before running kernels."
            )


def _candidate_count_from_stats(stats: Dict[str, Any]) -> Optional[int]:
    """
    Extract a candidate count from a native kernel statistics dictionary.
    """
    for key in (
        "candidates_count",
        "total_candidates",
        "num_candidates",
        "candidate_count",
    ):
        if key in stats:
            try:
                return int(stats[key])
            except Exception:
                return None

    nested = stats.get("candidate_stats")
    if isinstance(nested, dict):
        return _candidate_count_from_stats(nested)

    return None