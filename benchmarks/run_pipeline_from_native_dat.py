from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import sigmo
from sigmo.graph import make_csr_graph
from sigmo.result import KernelStep, build_match_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERY_SMARTS = PROJECT_ROOT / "benchmarks" / "datasets" / "query.smarts"
DEFAULT_DATA_SMARTS = PROJECT_ROOT / "benchmarks" / "datasets" / "data.smarts"

DEFAULT_NATIVE_CONVERTER = (
    PROJECT_ROOT / "external" / "sigmo" / "scripts" / "smile2graph.py"
)

DEFAULT_CACHE_DIR = PROJECT_ROOT / "benchmarks" / "native_dat_cache"


def parse_prefixed_int(token: str, prefix: str) -> int:
    if not token.startswith(prefix):
        raise ValueError(
            f"Invalid SIGMo .dat token '{token}': expected prefix '{prefix}'."
        )

    try:
        return int(token[len(prefix):])
    except ValueError as exc:
        raise ValueError(
            f"Invalid integer token '{token}' in SIGMo .dat file."
        ) from exc


def source_records(source_path: Path) -> list[dict[str, Any]]:
    """
    Reproduce the native converter input policy:
    empty lines are ignored, while all non-empty lines are treated as SMARTS.
    """
    records: list[dict[str, Any]] = []

    with source_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            value = line.strip()

            if not value:
                continue

            records.append(
                {
                    "input": value,
                    "line": line_no,
                    "source_file": str(source_path),
                }
            )

    return records


def sigmo_dat_line_to_csr(
    line: str,
    *,
    name: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Parse one native SIGMo .dat graph line and reproduce the logic of
    IntermediateGraph::toCSRGraph() from the native C++ implementation.

    The .dat file stores every bond once as an undirected edge.
    The returned CSR stores both directions, exactly as the C++ reader does.
    """
    tokens = line.split()

    if len(tokens) < 3:
        raise ValueError(f"Invalid SIGMo .dat graph line: '{line}'.")

    cursor = 0

    num_nodes = parse_prefixed_int(tokens[cursor], "n#")
    cursor += 1

    # This is read for format compatibility. The current Python graph object
    # does not need to store the number of label classes separately.
    _num_labels = parse_prefixed_int(tokens[cursor], "l#")
    cursor += 1

    node_labels = [0] * num_nodes

    for _ in range(num_nodes):
        if cursor + 1 >= len(tokens):
            raise ValueError("Unexpected end of line while reading node labels.")

        node_id = int(tokens[cursor])
        node_label = int(tokens[cursor + 1])
        cursor += 2

        if not 0 <= node_id < num_nodes:
            raise ValueError(
                f"Node id {node_id} is outside [0, {num_nodes - 1}]."
            )

        node_labels[node_id] = node_label

    if cursor >= len(tokens):
        raise ValueError("Missing edge-count token in SIGMo .dat graph.")

    num_edges = parse_prefixed_int(tokens[cursor], "e#")
    cursor += 1

    expected_edge_tokens = num_edges * 3
    remaining_tokens = len(tokens) - cursor

    if remaining_tokens != expected_edge_tokens:
        raise ValueError(
            "Invalid SIGMo .dat edge section: "
            f"expected {expected_edge_tokens} tokens, found {remaining_tokens}."
        )

    edges: list[tuple[int, int, int]] = []

    for _ in range(num_edges):
        u = int(tokens[cursor])
        v = int(tokens[cursor + 1])
        edge_label = int(tokens[cursor + 2])
        cursor += 3

        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            raise ValueError(
                f"Invalid edge ({u}, {v}) for graph with {num_nodes} nodes."
            )

        edges.append((u, v, edge_label))

    # Exact logical equivalent of IntermediateGraph::toCSRGraph().
    row_offsets = [0] * (num_nodes + 1)

    for u, v, _ in edges:
        row_offsets[u + 1] += 1
        row_offsets[v + 1] += 1

    for index in range(1, len(row_offsets)):
        row_offsets[index] += row_offsets[index - 1]

    total_directed_edges = row_offsets[-1]

    column_indices = [0] * total_directed_edges
    edge_labels = [0] * total_directed_edges
    current_positions = [0] * num_nodes

    for u, v, edge_label in edges:
        u_position = row_offsets[u] + current_positions[u]
        column_indices[u_position] = v
        edge_labels[u_position] = edge_label
        current_positions[u] += 1

        v_position = row_offsets[v] + current_positions[v]
        column_indices[v_position] = u
        edge_labels[v_position] = edge_label
        current_positions[v] += 1

    return make_csr_graph(
        row_offsets=row_offsets,
        column_indices=column_indices,
        node_labels=node_labels,
        edge_labels=edge_labels,
        num_nodes=num_nodes,
        name=name,
        input_format="sigmo_dat",
        **metadata,
    )


def load_sigmo_dat(
    dat_path: Path,
    *,
    original_source: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Load graphs from a native SIGMo .dat file into the CSR dictionaries
    accepted by the Python binding.

    If original_source is provided, original SMARTS strings are preserved
    only as metadata for inspection and visualization purposes.
    """
    if not dat_path.exists():
        raise FileNotFoundError(f"SIGMo .dat file not found: {dat_path}")

    original_records = (
        source_records(original_source)
        if original_source is not None
        else []
    )

    graphs: list[dict[str, Any]] = []

    with dat_path.open("r", encoding="utf-8") as handle:
        for graph_index, line in enumerate(handle):
            line = line.strip()

            if not line:
                continue

            metadata: dict[str, Any] = {
                "original_index": graph_index,
                "source_file": str(dat_path),
            }

            if graph_index < len(original_records):
                metadata.update(original_records[graph_index])

            graph = sigmo_dat_line_to_csr(
                line,
                name=f"{dat_path.stem}:{graph_index}",
                metadata=metadata,
            )
            graphs.append(graph)

    if original_records and len(graphs) != len(original_records):
        print(
            "Warning: the number of graphs in the .dat file differs from "
            "the number of non-empty source lines.",
            file=sys.stderr,
        )

    return graphs


def generate_native_dat(
    *,
    converter_path: Path,
    source_path: Path,
    output_path: Path,
) -> None:
    """
    Invoke Antonio's actual smile2graph.py script.

    This is intentionally not a reimplementation: it guarantees that the
    generated .dat file follows the same preprocessing logic used by SIGMo.
    """
    if not converter_path.exists():
        raise FileNotFoundError(
            "Native SIGMo converter not found: "
            f"{converter_path}"
        )

    if not source_path.exists():
        raise FileNotFoundError(f"Input SMARTS file not found: {source_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_text = source_path.read_text(encoding="utf-8")

    command = [
        sys.executable,
        str(converter_path),
        "-f",
        "SIGMO",
        "-o",
        str(output_path),
    ]

    print(f"Generating native-compatible .dat file: {output_path}")

    completed = subprocess.run(
        command,
        input=source_text,
        text=True,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Native SIGMo converter failed for {source_path} "
            f"with exit code {completed.returncode}."
        )


def run_pipeline(
    query_graphs: list[dict[str, Any]],
    data_graphs: list[dict[str, Any]],
    *,
    device: str,
    iterations: int,
    find_first: bool,
):
    ctx = sigmo.PipelineContext(
        query_graphs,
        data_graphs,
        device=device,
    )

    pipeline_start = time.perf_counter()

    allocation_start = time.perf_counter()
    ctx.allocate()
    allocation_time = time.perf_counter() - allocation_start

    # Add allocation to the same execution trace printed by MatchResult.
    ctx.steps.insert(
        0,
        KernelStep(
            name="allocate",
            elapsed_seconds=allocation_time,
        ),
    )

    ctx.generate_signatures()
    ctx.filter_candidates()
    ctx.refine(
        iterations,
        start_view_size=1,
        stop_on_fixed_point=False,
    )

    raw_join = ctx.join(find_first=find_first)

    pipeline_time = time.perf_counter() - pipeline_start

    result = build_match_result(
        raw_join,
        query_graphs,
        data_graphs,
        steps=ctx.steps,
        warnings=ctx.warnings,
        errors=ctx.errors,
        device=ctx.device_name,
        requested_iterations=iterations,
        executed_iterations=ctx.executed_iterations,
    )

    return result, pipeline_time


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Python SIGMo binding using CSR graphs loaded from "
            "native SIGMo .dat files."
        )
    )

    parser.add_argument(
        "--query-smarts",
        type=Path,
        default=DEFAULT_QUERY_SMARTS,
    )
    parser.add_argument(
        "--data-smarts",
        type=Path,
        default=DEFAULT_DATA_SMARTS,
    )
    parser.add_argument(
        "--query-dat",
        type=Path,
        default=DEFAULT_CACHE_DIR / "query.dat",
    )
    parser.add_argument(
        "--data-dat",
        type=Path,
        default=DEFAULT_CACHE_DIR / "data.dat",
    )
    parser.add_argument(
        "--native-converter",
        type=Path,
        default=DEFAULT_NATIVE_CONVERTER,
    )
    parser.add_argument(
        "--regenerate-dat",
        action="store_true",
        help="Regenerate .dat files through the native SIGMo converter.",
    )
    parser.add_argument(
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--find-first",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    args = parser.parse_args()

    total_start = time.perf_counter()

    if args.regenerate_dat or not args.query_dat.exists():
        query_conversion_start = time.perf_counter()
        generate_native_dat(
            converter_path=args.native_converter,
            source_path=args.query_smarts,
            output_path=args.query_dat,
        )
        query_conversion_time = time.perf_counter() - query_conversion_start
    else:
        query_conversion_time = 0.0

    if args.regenerate_dat or not args.data_dat.exists():
        data_conversion_start = time.perf_counter()
        generate_native_dat(
            converter_path=args.native_converter,
            source_path=args.data_smarts,
            output_path=args.data_dat,
        )
        data_conversion_time = time.perf_counter() - data_conversion_start
    else:
        data_conversion_time = 0.0

    query_loading_start = time.perf_counter()
    query_graphs = load_sigmo_dat(
        args.query_dat,
        original_source=args.query_smarts,
    )
    query_loading_time = time.perf_counter() - query_loading_start

    data_loading_start = time.perf_counter()
    data_graphs = load_sigmo_dat(
        args.data_dat,
        original_source=args.data_smarts,
    )
    data_loading_time = time.perf_counter() - data_loading_start

    result, pipeline_time = run_pipeline(
        query_graphs,
        data_graphs,
        device=args.device,
        iterations=args.iterations,
        find_first=args.find_first,
    )

    total_time = time.perf_counter() - total_start

    print(result.summary())
    print("\nNative .dat preparation and execution")
    print("------------------------------------")
    print(f"Query .dat generation: {query_conversion_time:.3f} s")
    print(f"Data .dat generation:  {data_conversion_time:.3f} s")
    print(f"Query .dat loading:    {query_loading_time:.3f} s")
    print(f"Data .dat loading:     {data_loading_time:.3f} s")
    print(f"Pipeline execution:    {pipeline_time:.3f} s")
    print(f"End-to-end total:      {total_time:.3f} s")


if __name__ == "__main__":
    main()

