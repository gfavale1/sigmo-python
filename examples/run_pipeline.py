"""
Advanced example: step-by-step execution of the SIGMo pipeline.

This script exposes the SIGMo pipeline at kernel level while still using the
Python interface:

- sigmo.load_molecules() loads SMARTS/SMILES files without duplicating RDKit parsing;
- sigmo.PipelineContext manages Signature, Candidates, GMCR and the SYCL queue;
- sigmo.result.build_match_result() converts raw native output into MatchResult;
- summary(), explain(), to_csv(), and to_json() provide user-friendly output.

Recommended execution from the project root:

    PYTHONPATH=python python examples/run_pipeline.py

Example with options:

    PYTHONPATH=python python examples/run_pipeline.py \
        --base-dir benchmarks/datasets \
        --query-file query.smarts \
        --data-file data.smarts \
        --query-limit 5 \
        --data-limit 20 \
        --iterations 6 \
        --device auto \
        --csv examples/outputs/matches.csv \
        --json examples/outputs/matches.json

Notes:
    This example intentionally does not perform RDKit validation. RDKit is used
    by the package for parsing and visualization, while this script focuses on
    executing and inspecting the SIGMo pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import sigmo
from sigmo.result import build_match_result


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"

    if isinstance(value, int):
        return f"{value:,}"

    if isinstance(value, dict):
        compact = []
        for key, item in value.items():
            if isinstance(item, int):
                compact.append(f"{key}={item:,}")
            elif isinstance(item, float):
                compact.append(f"{key}={item:.6f}")
            else:
                compact.append(f"{key}={item}")
        return "{" + ", ".join(compact) + "}"

    return str(value)


def print_stats(title: str, stats: Optional[Dict[str, Any]]) -> None:
    """
    Print kernel statistics without dumping huge match structures.
    """
    print(f"  [KERNEL] {title}:")

    if not stats:
        print("    - no statistics available")
        return

    hidden_keys = {"matches_dict", "matches", "raw_matches"}

    for key, value in stats.items():
        if key in hidden_keys:
            if isinstance(value, dict):
                total_pairs = sum(len(items) for items in value.values())
                print(f"    - {key}: <hidden: {total_pairs:,} match(es)>")
            elif isinstance(value, list):
                print(f"    - {key}: <hidden: {len(value):,} item(s)>")
            else:
                print(f"    - {key}: <hidden>")
            continue

        print(f"    - {key}: {_format_value(value)}")


def print_graph_overview(label: str, graphs: List[Dict[str, Any]]) -> None:
    """
    Print a compact overview of loaded CSR graphs.
    """
    total_nodes = sum(int(graph.get("num_nodes", 0)) for graph in graphs)
    total_edges_directed = sum(len(graph.get("column_indices", [])) for graph in graphs)

    print(f"[*] {label}:")
    print(f"    - graphs: {len(graphs):,}")
    print(f"    - total nodes: {total_nodes:,}")
    print(f"    - directed CSR edges: {total_edges_directed:,}")

    if graphs:
        preview = ", ".join(
            graph.get("name", f"graph_{idx}")
            for idx, graph in enumerate(graphs[:3])
        )

        if len(graphs) > 3:
            preview += ", ..."

        print(f"    - preview: {preview}")


def _iter_raw_match_pairs(raw_join: Dict[str, Any]) -> Iterable[Tuple[int, int]]:
    """
    Yield raw query/data match pairs from supported native output formats.
    """
    matches_dict = raw_join.get("matches_dict", {}) or {}

    if isinstance(matches_dict, dict) and matches_dict:
        for q_idx, d_indices in matches_dict.items():
            for d_idx in d_indices:
                yield int(q_idx), int(d_idx)
        return

    raw_matches = raw_join.get("matches", []) or []

    for item in raw_matches:
        if isinstance(item, dict):
            q_idx = item.get("query_index", item.get("q_idx", item.get("query")))
            d_idx = item.get("data_index", item.get("d_idx", item.get("data")))

            if q_idx is None or d_idx is None:
                continue

            yield int(q_idx), int(d_idx)

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            yield int(item[0]), int(item[1])


def print_matches(
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    max_matches: int = 30,
) -> None:
    """
    Print only a preview of the matches.

    Complete results should be exported to CSV/JSON.
    """
    num_matches = int(raw_join.get("num_matches", 0))

    print("\n[MATCH DETAILS]")
    print(f"  Total matches reported by SIGMo: {num_matches:,}")
    print(f"  Showing at most: {max_matches:,}")

    if num_matches == 0:
        print("  No matches found.")
        return

    printed = 0

    for q_idx, d_idx in _iter_raw_match_pairs(raw_join):
        if printed >= max_matches:
            remaining = max(num_matches - printed, 0)
            print(f"  ... output truncated. Remaining matches not printed: {remaining:,}")
            print("  Use --csv or --json to export results.")
            return

        q_graph = query_graphs[q_idx] if 0 <= q_idx < len(query_graphs) else {}
        d_graph = data_graphs[d_idx] if 0 <= d_idx < len(data_graphs) else {}

        q_name = q_graph.get("name", f"Q-{q_idx}")
        d_name = d_graph.get("name", f"D-{d_idx}")

        q_input = q_graph.get("input", "")
        d_input = d_graph.get("input", "")

        print(f"  - [{q_idx}] {q_name} MATCHES [{d_idx}] {d_name}")

        if q_input:
            print(f"      query: {q_input}")

        if d_input:
            preview = d_input[:140] + "..." if len(d_input) > 140 else d_input
            print(f"      data:  {preview}")

        printed += 1


def _step_stats(ctx: sigmo.PipelineContext, step_name: str) -> Optional[Dict[str, Any]]:
    """
    Return the statistics for the last KernelStep with the given name.
    """
    for step in reversed(getattr(ctx, "steps", [])):
        if getattr(step, "name", None) == step_name:
            return getattr(step, "stats", None)

    return None


def load_graphs(
    path: Path,
    *,
    input_format: str,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Load SIGMo CSR graphs from a molecule file and optionally apply a limit.
    """
    graphs = sigmo.load_molecules(
        str(path),
        input_format=input_format,
    )

    if limit is not None:
        graphs = graphs[:limit]

    return graphs


def iter_match_records(
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
):
    """
    Yield match records one by one without materializing a huge list.
    """
    for q_idx, d_idx in _iter_raw_match_pairs(raw_join):
        q_graph = query_graphs[q_idx] if 0 <= q_idx < len(query_graphs) else {}
        d_graph = data_graphs[d_idx] if 0 <= d_idx < len(data_graphs) else {}

        yield {
            "query_index": q_idx,
            "query_name": q_graph.get("name", f"query_{q_idx}"),
            "query_input": q_graph.get("input", ""),
            "data_index": d_idx,
            "data_name": d_graph.get("name", f"data_{d_idx}"),
            "data_input": d_graph.get("input", ""),
        }


def write_matches_csv_streaming(
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    path: Path,
) -> int:
    """
    Write matches to CSV in streaming mode.

    This avoids building a large Python list and is suitable for very large
    result sets.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_index",
        "query_name",
        "query_input",
        "data_index",
        "data_name",
        "data_input",
    ]

    count = 0

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for record in iter_match_records(raw_join, query_graphs, data_graphs):
            writer.writerow(record)
            count += 1

            if count % 1_000_000 == 0:
                print(f"[*] CSV streaming: wrote {count:,} matches...", flush=True)

    return count


def write_summary_json(
    *,
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    ctx: sigmo.PipelineContext,
    result: sigmo.MatchResult,
    path: Path,
) -> None:
    """
    Write a lightweight JSON summary.

    This does not include all matches, because full JSON output can be too large
    for big datasets. Use CSV streaming for complete match export.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "OK" if not result.errors else "ERROR",
        "device": result.device,
        "query_count": len(query_graphs),
        "data_count": len(data_graphs),
        "total_matches": raw_join.get("num_matches", 0),
        "requested_iterations": result.requested_iterations,
        "executed_iterations": result.executed_iterations,
        "warnings": result.warnings,
        "errors": result.errors,
        "kernel_steps": [
            {
                "name": getattr(step, "name", None),
                "duration_seconds": getattr(
                    step,
                    "duration_seconds",
                    getattr(step, "elapsed_seconds", None),
                ),
                "stats": {
                    key: value
                    for key, value in getattr(step, "stats", {}).items()
                    if key not in {"matches_dict", "matches", "raw_matches"}
                },
            }
            for step in getattr(ctx, "steps", [])
        ],
        "note": (
            "This JSON contains only metadata and kernel statistics. "
            "Complete matches should be exported through the streaming CSV output."
        ),
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def run_kernel_pipeline(
    *,
    query_path: Path,
    data_path: Path,
    input_format: str = "auto",
    device: str = "auto",
    iterations: int = 6,
    find_first: bool = True,
    query_limit: Optional[int] = 5,
    data_limit: Optional[int] = 20,
    csv_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    force_refine: bool = False,
    max_print_matches: int = 30,
    large_result_threshold: int = 1_000_000,
) -> sigmo.MatchResult:
    """
    Execute the SIGMo kernel-level pipeline step by step.
    """
    print("[*] SIGMo kernel-level pipeline")
    print(f"[*] Query file: {query_path}")
    print(f"[*] Data file:  {data_path}")

    query_graphs = load_graphs(
        query_path,
        input_format=input_format,
        limit=query_limit,
    )
    data_graphs = load_graphs(
        data_path,
        input_format=input_format,
        limit=data_limit,
    )

    if not query_graphs:
        raise ValueError(f"No valid query graphs loaded from: {query_path}")

    if not data_graphs:
        raise ValueError(f"No valid data graphs loaded from: {data_path}")

    print_graph_overview("Query graphs", query_graphs)
    print_graph_overview("Data graphs", data_graphs)

    small_query_graphs = [
        (idx, graph.get("name", f"query_{idx}"), graph.get("num_nodes", 0))
        for idx, graph in enumerate(query_graphs)
        if graph.get("num_nodes", 0) < 6
    ]

    small_data_graphs = [
        (idx, graph.get("name", f"data_{idx}"), graph.get("num_nodes", 0))
        for idx, graph in enumerate(data_graphs)
        if graph.get("num_nodes", 0) < 6
    ]

    if small_query_graphs or small_data_graphs:
        print("\n[WARNING] Small graphs detected. Refinement may be unstable:")
        for idx, name, nodes in small_query_graphs[:10]:
            print(f"  - query[{idx}] {name}: {nodes} node(s)")
        for idx, name, nodes in small_data_graphs[:10]:
            print(f"  - data[{idx}] {name}: {nodes} node(s)")

    ctx = sigmo.PipelineContext(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        device=device,
    )

    print(f"\n[*] Device: {ctx.device_name}")

    print("\n[*] Allocating SIGMo memory and native structures")
    ctx.allocate()
    print("  - Signature, Candidates and GMCR allocated successfully")

    print("\n[*] 1/4 - Generating initial signatures")
    ctx.generate_signatures()
    print_stats("Generate Query Signatures", _step_stats(ctx, "generate_query_signatures"))
    print_stats("Generate Data Signatures", _step_stats(ctx, "generate_data_signatures"))

    print("\n[*] 2/4 - Filtering candidates")
    filter_stats = ctx.filter_candidates()
    print_stats("Initial Filter", filter_stats)

    print("\n[*] 3/4 - Iterative refinement")

    min_query_nodes = min(graph.get("num_nodes", 0) for graph in query_graphs)
    min_data_nodes = min(graph.get("num_nodes", 0) for graph in data_graphs)
    min_nodes = min(min_query_nodes, min_data_nodes)

    safe_iterations = int(iterations)

    if iterations > 0 and min_nodes < 6 and not force_refine:
        warning_msg = (
            "Refinement disabled for stability: at least one graph has "
            f"{min_nodes} node(s) (< 6). The pipeline continues with "
            "filter + join. Use --force-refine to override this behavior."
        )

        print(f"  [WARNING] {warning_msg}")
        ctx.warnings.append(warning_msg)

        safe_iterations = 0

    elif iterations > 0 and min_nodes < 6 and force_refine:
        warning_msg = (
            "Force refine enabled: refinement will run even though small graphs "
            f"are present. Minimum node count: {min_nodes}. This mode may cause "
            "native C++/SYCL instability on some datasets."
        )

        print(f"  [WARNING] {warning_msg}")
        ctx.warnings.append(warning_msg)

        safe_iterations = int(iterations)

    if safe_iterations <= 0:
        print("  - Refinement disabled.")
        executed_iterations = 0

    else:
        print(
            f"  - Running refinement for at most {safe_iterations} iteration(s) "
            f"(force_refine={force_refine})"
        )

        before_steps = len(ctx.steps)

        try:
            refine_stats = ctx.refine(
                safe_iterations,
                start_view_size=1,
                stop_on_fixed_point=True,
            )
        except TypeError:
            refine_stats = ctx.refine(max_iterations=safe_iterations)

        executed_iterations = getattr(ctx, "executed_iterations", None)

        if executed_iterations is None:
            new_steps = ctx.steps[before_steps:]
            executed_iterations = sum(
                1
                for step in new_steps
                if getattr(step, "name", None) == "refine_candidates"
            )

        if isinstance(refine_stats, dict):
            print_stats("Refinement", refine_stats)
        else:
            print(f"  - Executed iterations: {executed_iterations}")

    print("\n[*] 4/4 - Final join / isomorphism")
    raw_join = ctx.join(find_first=find_first)
    print_stats("Join", raw_join)

    num_matches = int(raw_join.get("num_matches", 0))
    large_result = num_matches > large_result_threshold

    if max_print_matches > 0:
        print_matches(
            raw_join,
            query_graphs,
            data_graphs,
            max_matches=max_print_matches,
        )
    else:
        print("\n[MATCH DETAILS]")
        print("  Match detail printing disabled: --max-print-matches 0")

    if large_result:
        large_warning = (
            f"Very large result: {num_matches:,} matches. "
            "To avoid excessive memory usage, complete matches are not "
            "materialized inside MatchResult. Use streaming CSV export for "
            "complete output."
        )

        print(f"\n[WARNING] {large_warning}")
        ctx.warnings.append(large_warning)

        raw_join_for_result = dict(raw_join)
        raw_join_for_result["matches_dict"] = {}

        result = build_match_result(
            raw_join_for_result,
            query_graphs,
            data_graphs,
            steps=ctx.steps,
            warnings=ctx.warnings,
            errors=ctx.errors,
            device=ctx.device_name,
            requested_iterations=iterations,
            executed_iterations=executed_iterations,
        )

        result.total_matches = num_matches

    else:
        result = build_match_result(
            raw_join,
            query_graphs,
            data_graphs,
            steps=ctx.steps,
            warnings=ctx.warnings,
            errors=ctx.errors,
            device=ctx.device_name,
            requested_iterations=iterations,
            executed_iterations=executed_iterations,
        )

    print("\n" + result.summary())
    print("\n" + result.explain())

    if csv_path is not None:
        if large_result:
            print(f"\n[*] Exporting CSV in streaming mode: {csv_path}", flush=True)
            written = write_matches_csv_streaming(
                raw_join,
                query_graphs,
                data_graphs,
                csv_path,
            )
            print(f"[*] CSV exported: {written:,} matches written.", flush=True)
        else:
            result.to_csv(str(csv_path))
            print(f"\n[*] CSV exported to: {csv_path}")

    if json_path is not None:
        if large_result:
            print(f"[*] Exporting JSON summary: {json_path}", flush=True)
            write_summary_json(
                raw_join=raw_join,
                query_graphs=query_graphs,
                data_graphs=data_graphs,
                ctx=ctx,
                result=result,
                path=json_path,
            )
            print(f"[*] JSON summary exported to: {json_path}", flush=True)
        else:
            result.to_json(str(json_path))
            print(f"[*] JSON exported to: {json_path}")

    print("\n[*] Pipeline completed.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the SIGMo kernel-level pipeline through the Python interface."
    )

    parser.add_argument(
        "--base-dir",
        default="benchmarks/datasets",
        help="Directory containing query/data files.",
    )
    parser.add_argument(
        "--query-file",
        default="query.smarts",
        help="Query SMARTS/SMILES file.",
    )
    parser.add_argument(
        "--data-file",
        default="data.smarts",
        help="Database SMARTS/SMILES file.",
    )
    parser.add_argument(
        "--input-format",
        default="auto",
        choices=["auto", "smiles", "smarts"],
        help="Input format used by sigmo.load_molecules().",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="SYCL device selector: auto, gpu, cpu, cuda, cuda:gpu, etc.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help=(
            "Maximum number of refinement iterations. Default: 0 to avoid "
            "native backend instability on datasets containing very small graphs."
        ),
    )
    parser.add_argument(
        "--find-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the join at the first match per pair when supported.",
    )
    parser.add_argument(
        "--query-limit",
        type=int,
        default=5,
        help="Maximum number of query graphs to load. Use -1 for no limit.",
    )
    parser.add_argument(
        "--data-limit",
        type=int,
        default=20,
        help="Maximum number of data graphs to load. Use -1 for no limit.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV path for exporting matches.",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Optional JSON path for exporting the result or summary.",
    )
    parser.add_argument(
        "--force-refine",
        action="store_true",
        help=(
            "Force refinement even when small graphs are present. Experimental: "
            "may cause native C++/SYCL instability on some datasets."
        ),
    )
    parser.add_argument(
        "--max-print-matches",
        type=int,
        default=30,
        help="Maximum number of matches printed to the terminal. Use 0 to disable.",
    )

    return parser.parse_args()


def _normalize_limit(value: int) -> Optional[int]:
    return None if value is None or value < 0 else value


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    query_path = base_dir / args.query_file
    data_path = base_dir / args.data_file

    run_kernel_pipeline(
        query_path=query_path,
        data_path=data_path,
        input_format=args.input_format,
        device=args.device,
        iterations=args.iterations,
        find_first=args.find_first,
        query_limit=_normalize_limit(args.query_limit),
        data_limit=_normalize_limit(args.data_limit),
        csv_path=Path(args.csv) if args.csv else None,
        json_path=Path(args.json) if args.json else None,
        force_refine=args.force_refine,
        max_print_matches=args.max_print_matches,
    )


if __name__ == "__main__":
    main()