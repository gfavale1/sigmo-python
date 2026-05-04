from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class KernelStep:
    """
    Explainable statistics for one SIGMo kernel step.
    """

    name: str
    elapsed_seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Match:
    """
    Python-friendly representation of one query-data match.
    """

    query_index: int
    data_index: int
    query_name: str
    data_name: str
    query_input: Optional[str] = None
    data_input: Optional[str] = None
    query_num_nodes: Optional[int] = None
    data_num_nodes: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        """
        Convert the match to a dictionary suitable for CSV/JSON export.
        """
        return asdict(self)


@dataclass
class MatchResult:
    """
    Structured result returned by the SIGMo Python pipeline.

    MatchResult avoids exposing users directly to raw dictionaries returned by
    the native C++/SYCL binding. It stores matches, kernel steps, warnings,
    errors and execution metadata in a Python-friendly format.
    """

    total_matches: int
    matches: List[Match] = field(default_factory=list)
    query_count: int = 0
    data_count: int = 0
    device: Optional[str] = None
    requested_iterations: int = 0
    executed_iterations: int = 0
    steps: List[KernelStep] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    raw_result: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """
        Return True if the pipeline completed without recorded errors.
        """
        return len(self.errors) == 0

    def summary(self) -> str:
        """
        Return a compact human-readable execution summary.
        """
        lines = [
            "SIGMo search summary",
            "--------------------",
            f"Status: {'OK' if self.ok else 'ERROR'}",
            f"Device: {self.device or 'unknown'}",
            f"Queries: {self.query_count}",
            f"Database graphs: {self.data_count}",
            f"Matches found: {self.total_matches}",
            f"Refinement iterations: {self.executed_iterations}/{self.requested_iterations}",
        ]

        if self.steps:
            lines.append("Kernel steps:")
            for step in self.steps:
                candidate_info = _extract_candidate_info(step.stats)
                suffix = f" | {candidate_info}" if candidate_info else ""
                lines.append(f"  - {step.name}: {step.elapsed_seconds:.4f}s{suffix}")

        if self.warnings:
            lines.append("Warnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")

        if self.errors:
            lines.append("Errors:")
            for error in self.errors:
                lines.append(f"  - {error}")

        return "\n".join(lines)

    def explain(self) -> str:
        """
        Explain the executed pipeline in sequential, user-readable form.
        """
        lines = [
            "SIGMo pipeline explanation",
            "--------------------------",
            f"1. Loaded {self.query_count} query graph(s) and {self.data_count} data graph(s).",
            "2. Converted the input molecules/graphs into the CSR representation expected by SIGMo.",
        ]

        step_number = 3

        for step in self.steps:
            if step.name == "generate_query_signatures":
                lines.append(f"{step_number}. Generated structural signatures for query graphs.")

            elif step.name == "generate_data_signatures":
                lines.append(f"{step_number}. Generated structural signatures for data graphs.")

            elif step.name == "filter_candidates":
                lines.append(
                    f"{step_number}. Applied the initial candidate filtering step"
                    f"{_format_stats_suffix(step.stats)}."
                )

            elif step.name == "refine_query_signatures":
                lines.append(
                    f"{step_number}. Refined query signatures"
                    f"{_format_stats_suffix(step.stats)}."
                )

            elif step.name == "refine_data_signatures":
                lines.append(
                    f"{step_number}. Refined data signatures"
                    f"{_format_stats_suffix(step.stats)}."
                )

            elif step.name == "refine_candidates":
                lines.append(
                    f"{step_number}. Refined candidate pairs"
                    f"{_format_stats_suffix(step.stats)}."
                )

            elif step.name.startswith("refine_iteration_"):
                lines.append(
                    f"{step_number}. Refined signatures and candidates ({step.name})"
                    f"{_format_stats_suffix(step.stats)}."
                )

            elif step.name == "join_candidates":
                lines.append(
                    f"{step_number}. Ran the final join/isomorphism phase and "
                    f"found {self.total_matches} match(es)."
                )

            else:
                lines.append(
                    f"{step_number}. Executed step '{step.name}'"
                    f"{_format_stats_suffix(step.stats)}."
                )

            step_number += 1

        if self.warnings:
            lines.append("Warnings considered during execution:")
            for warning in self.warnings:
                lines.append(f"- {warning}")

        if self.errors:
            lines.append("Errors raised during execution:")
            for error in self.errors:
                lines.append(f"- {error}")

        return "\n".join(lines)

    def to_records(self) -> List[Dict[str, Any]]:
        """
        Convert all materialized matches to a list of dictionaries.
        """
        return [match.to_record() for match in self.matches]

    def to_dataframe(self):
        """
        Convert materialized matches to a pandas.DataFrame.

        Requires:
            pandas
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Install pandas to use MatchResult.to_dataframe()."
            ) from exc

        return pd.DataFrame(self.to_records())

    def to_csv(self, path: str, **kwargs: Any) -> None:
        """
        Export materialized matches to CSV without requiring pandas.
        """
        records = self.to_records()

        default_fields = [
            "query_index",
            "query_name",
            "query_input",
            "data_index",
            "data_name",
            "data_input",
        ]

        if records:
            fieldnames: List[str] = []
            for record in records:
                for key in record.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        else:
            fieldnames = default_fields

        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, **kwargs)
            writer.writeheader()
            writer.writerows(records)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """
        Serialize the full MatchResult to JSON.

        If path is provided, the JSON payload is also written to disk.
        """
        payload = {
            "total_matches": self.total_matches,
            "query_count": self.query_count,
            "data_count": self.data_count,
            "device": self.device,
            "requested_iterations": self.requested_iterations,
            "executed_iterations": self.executed_iterations,
            "matches": self.to_records(),
            "steps": [asdict(step) for step in self.steps],
            "warnings": self.warnings,
            "errors": self.errors,
            "raw_result": self.raw_result,
        }

        text = json.dumps(payload, indent=indent, ensure_ascii=False)

        if path is not None:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)

        return text


def build_match_result(
    raw_result: Dict[str, Any],
    q_graphs: List[Dict[str, Any]],
    d_graphs: List[Dict[str, Any]],
    *,
    steps: Optional[List[KernelStep]] = None,
    warnings: Optional[List[str]] = None,
    errors: Optional[List[str]] = None,
    device: Optional[str] = None,
    requested_iterations: int = 0,
    executed_iterations: int = 0,
) -> MatchResult:
    """
    Convert a raw native SIGMo result dictionary into a MatchResult.
    """
    raw_result = raw_result or {}
    warnings = list(warnings or [])
    errors = list(errors or [])

    if raw_result.get("error"):
        errors.append(str(raw_result["error"]))

    matches = _extract_matches(raw_result, q_graphs, d_graphs)
    total_matches = int(raw_result.get("num_matches", len(matches)))

    return MatchResult(
        total_matches=total_matches,
        matches=matches,
        query_count=len(q_graphs),
        data_count=len(d_graphs),
        device=device,
        requested_iterations=requested_iterations,
        executed_iterations=executed_iterations,
        steps=list(steps or []),
        warnings=warnings,
        errors=errors,
        raw_result=raw_result,
    )


def _extract_matches(
    raw_result: Dict[str, Any],
    q_graphs: List[Dict[str, Any]],
    d_graphs: List[Dict[str, Any]],
) -> List[Match]:
    """
    Extract matches from raw native output.

    Supported native formats:
        - matches_dict: {query_index: [data_index, ...]}
        - matches: [(query_index, data_index), ...]
        - matches: [{"query_index": ..., "data_index": ...}, ...]
    """
    matches: List[Match] = []

    if isinstance(raw_result.get("matches_dict"), dict):
        for q_idx, data_indices in raw_result["matches_dict"].items():
            for d_idx in data_indices:
                matches.append(
                    _make_match(
                        int(q_idx),
                        int(d_idx),
                        q_graphs,
                        d_graphs,
                    )
                )
        return matches

    raw_matches = raw_result.get("matches", []) or []

    for item in raw_matches:
        if isinstance(item, dict):
            q_idx = item.get("query_index", item.get("q_idx", item.get("query", 0)))
            d_idx = item.get("data_index", item.get("d_idx", item.get("data", 0)))

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            q_idx = item[0]
            d_idx = item[1]

        else:
            continue

        matches.append(
            _make_match(
                int(q_idx),
                int(d_idx),
                q_graphs,
                d_graphs,
            )
        )

    return matches


def _make_match(
    q_idx: int,
    d_idx: int,
    q_graphs: List[Dict[str, Any]],
    d_graphs: List[Dict[str, Any]],
) -> Match:
    query_graph = q_graphs[q_idx] if 0 <= q_idx < len(q_graphs) else {}
    data_graph = d_graphs[d_idx] if 0 <= d_idx < len(d_graphs) else {}

    return Match(
        query_index=q_idx,
        data_index=d_idx,
        query_name=str(query_graph.get("name", f"query_{q_idx}")),
        data_name=str(data_graph.get("name", f"data_{d_idx}")),
        query_input=query_graph.get(
            "input",
            query_graph.get("smarts", query_graph.get("smiles")),
        ),
        data_input=data_graph.get(
            "input",
            data_graph.get("smarts", data_graph.get("smiles")),
        ),
        query_num_nodes=query_graph.get("num_nodes"),
        data_num_nodes=data_graph.get("num_nodes"),
    )


def _extract_candidate_info(stats: Dict[str, Any]) -> str:
    for key in (
        "candidates_count",
        "total_candidates",
        "num_candidates",
        "candidate_count",
    ):
        if key in stats:
            return f"candidates={stats[key]}"

    return ""


def _format_stats_suffix(stats: Dict[str, Any]) -> str:
    info = _extract_candidate_info(stats)
    return f" ({info})" if info else ""