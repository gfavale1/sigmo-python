from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional
import json
import csv


@dataclass
class KernelStep:
    """Statistiche explainable di uno step della pipeline SIGMo."""

    name: str
    elapsed_seconds: float
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Match:
    """Singolo match query-data in forma Python-friendly."""

    query_index: int
    data_index: int
    query_name: str
    data_name: str
    query_input: Optional[str] = None
    data_input: Optional[str] = None
    query_num_nodes: Optional[int] = None
    data_num_nodes: Optional[int] = None

    def to_record(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MatchResult:
    """
    Risultato strutturato della pipeline SIGMo.

    L'obiettivo e' evitare che l'utente debba interpretare direttamente
    dizionari grezzi provenienti dal binding C++.
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
    validation: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        """Restituisce un riepilogo testuale compatto e leggibile."""
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

        if self.validation:
            passed = self.validation.get("passed")
            status = "PASSED" if passed else "FAILED"
            checked = self.validation.get("checked_pairs", 0)
            agreements = self.validation.get("agreements", 0)
            disagreements = len(self.validation.get("disagreements", []))
            lines.append("Validation:")
            lines.append(f"  - Method: {self.validation.get('method', 'unknown')}")
            lines.append(f"  - Status: {status}")
            lines.append(f"  - Agreements: {agreements}/{checked}")
            lines.append(f"  - Disagreements: {disagreements}")

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
        Spiega in modo sequenziale cosa e' successo nella pipeline.
        Utile per notebook, debug e utenti non HPC.
        """
        lines = [
            "SIGMo pipeline explanation",
            "--------------------------",
            f"1. Loaded {self.query_count} query graph(s) and {self.data_count} data graph(s).",
            "2. Converted the input molecules/graphs into the CSR representation expected by SIGMo.",
        ]

        n = 3
        for step in self.steps:
            if step.name == "generate_query_signatures":
                lines.append(f"{n}. Generated structural signatures for query graphs.")
            elif step.name == "generate_data_signatures":
                lines.append(f"{n}. Generated structural signatures for data graphs.")
            elif step.name == "filter_candidates":
                lines.append(f"{n}. Applied the initial candidate filtering step{_format_stats_suffix(step.stats)}.")
            elif step.name.startswith("refine_iteration_"):
                lines.append(f"{n}. Refined signatures and candidates ({step.name}){_format_stats_suffix(step.stats)}.")
            elif step.name == "join_candidates":
                lines.append(f"{n}. Ran the final join/isomorphism phase and found {self.total_matches} match(es).")
            else:
                lines.append(f"{n}. Executed step '{step.name}'{_format_stats_suffix(step.stats)}.")
            n += 1

        if self.validation:
            checked = self.validation.get("checked_pairs", 0)
            agreements = self.validation.get("agreements", 0)
            disagreements = self.validation.get("disagreements", [])
            lines.append(
                f"{n}. Validated the result against RDKit "
                f"({agreements}/{checked} checked pair(s) agree)."
            )
            n += 1

            if disagreements:
                lines.append("Validation disagreements:")
                for item in disagreements:
                    lines.append(
                        "- Query "
                        f"{item.get('query_name')} vs data {item.get('data_name')}: "
                        f"SIGMo={item.get('sigmo')}, RDKit={item.get('rdkit')}"
                    )

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
        return [match.to_record() for match in self.matches]

    def to_dataframe(self):
        """Restituisce un pandas.DataFrame, se pandas e' installato."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Installa pandas per usare MatchResult.to_dataframe().") from exc
        return pd.DataFrame(self.to_records())

    def to_csv(self, path, **kwargs):
        """
        Esporta i match in CSV.

        A differenza di to_dataframe(), questo metodo non richiede pandas.
        Questo rende l'export CSV disponibile anche in ambienti minimali.
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
            fieldnames = []
            for record in records:
                for key in record.keys():
                    if key not in fieldnames:
                        fieldnames.append(key)
        else:
            fieldnames = default_fields

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, **kwargs)
            writer.writeheader()
            writer.writerows(records)

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        """Serializza tutto il risultato in JSON. Se path e' fornito, salva anche su file."""
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
            "validation": self.validation,
        }
        text = json.dumps(payload, indent=indent, ensure_ascii=False)
        if path is not None:
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
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
    """Converte l'output grezzo del binding in MatchResult."""
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
    Supporta piu' formati possibili del binding:
    - matches_dict: {q_idx: [d_idx, ...]}
    - matches: [(q_idx, d_idx), ...] oppure [{query_index, data_index}, ...]
    """
    out: List[Match] = []

    if isinstance(raw_result.get("matches_dict"), dict):
        for q_idx, data_indices in raw_result["matches_dict"].items():
            for d_idx in data_indices:
                out.append(_make_match(int(q_idx), int(d_idx), q_graphs, d_graphs))
        return out

    raw_matches = raw_result.get("matches", []) or []
    for item in raw_matches:
        if isinstance(item, dict):
            q_idx = item.get("query_index", item.get("q_idx", item.get("query", 0)))
            d_idx = item.get("data_index", item.get("d_idx", item.get("data", 0)))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            q_idx, d_idx = item[0], item[1]
        else:
            continue
        out.append(_make_match(int(q_idx), int(d_idx), q_graphs, d_graphs))

    return out


def _make_match(
    q_idx: int,
    d_idx: int,
    q_graphs: List[Dict[str, Any]],
    d_graphs: List[Dict[str, Any]],
) -> Match:
    q = q_graphs[q_idx] if 0 <= q_idx < len(q_graphs) else {}
    d = d_graphs[d_idx] if 0 <= d_idx < len(d_graphs) else {}
    return Match(
        query_index=q_idx,
        data_index=d_idx,
        query_name=str(q.get("name", f"query_{q_idx}")),
        data_name=str(d.get("name", f"data_{d_idx}")),
        query_input=q.get("input", q.get("smarts", q.get("smiles"))),
        data_input=d.get("input", d.get("smarts", d.get("smiles"))),
        query_num_nodes=q.get("num_nodes"),
        data_num_nodes=d.get("num_nodes"),
    )


def _extract_candidate_info(stats: Dict[str, Any]) -> str:
    for key in ("candidates_count", "total_candidates", "num_candidates", "candidate_count"):
        if key in stats:
            return f"candidates={stats[key]}"
    return ""


def _format_stats_suffix(stats: Dict[str, Any]) -> str:
    info = _extract_candidate_info(stats)
    return f" ({info})" if info else ""
