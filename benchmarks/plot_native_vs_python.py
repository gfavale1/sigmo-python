#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import sigmo


DEFAULT_QUERY_PATH = "benchmarks/datasets/query.smarts"
DEFAULT_DATA_PATH = "benchmarks/datasets/data.smarts"


def step_sum(ctx, name: str) -> float:
    return sum(step.elapsed_seconds for step in ctx.steps if step.name == name)


def run_python_pipeline(
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    iterations: int,
    find_all: bool,
    device: str,
) -> Dict[str, float]:
    find_first = not find_all

    total_start = time.perf_counter()

    ctx = sigmo.PipelineContext(query_graphs, data_graphs, device=device)

    allocate_start = time.perf_counter()
    ctx.allocate()
    allocate_end = time.perf_counter()

    ctx.generate_signatures()
    ctx.filter_candidates()

    if iterations > 0:
        ctx.refine(
            iterations,
            start_view_size=1,
            stop_on_fixed_point=False,
        )

    ctx.join(find_first=find_first)

    total_end = time.perf_counter()

    generate_time = (
        step_sum(ctx, "generate_query_signatures")
        + step_sum(ctx, "generate_data_signatures")
    )

    refine_time = (
        step_sum(ctx, "refine_query_signatures")
        + step_sum(ctx, "refine_data_signatures")
        + step_sum(ctx, "refine_candidates")
    )

    return {
        "Allocate": allocate_end - allocate_start,
        "Generate": generate_time,
        "Filter": step_sum(ctx, "filter_candidates"),
        "Refine": refine_time,
        "Join": step_sum(ctx, "join_candidates"),
        "Total": total_end - total_start,
    }


def get_native_breakdown(
    native_csv: Path,
    iterations: int,
    find_all: bool,
) -> Dict[str, float]:
    df = pd.read_csv(native_csv)

    rows = df[
        (df["n_refinement_steps"] == iterations)
        & (df["find_all"].astype(bool) == bool(find_all))
    ]

    if rows.empty:
        raise ValueError(
            f"No native row found for n_refinement_steps={iterations}, "
            f"find_all={find_all}"
        )

    row = rows.mean(numeric_only=True)

    baseline_rows = df[
        (df["n_refinement_steps"] == 0)
        & (df["find_all"].astype(bool) == bool(find_all))
    ]

    if baseline_rows.empty:
        baseline_filter_ms = 0.0
    else:
        baseline_filter_ms = float(
            baseline_rows.mean(numeric_only=True)["filter_host_time"]
        )

    total_filter_ms = float(row["filter_host_time"])

    filter_ms = baseline_filter_ms
    refine_ms = max(0.0, total_filter_ms - baseline_filter_ms)

    # The native SIGMo total_time does not include setup_data_host_time in the
    # same way the plotted end-to-end total does, so we add it explicitly.
    allocate_ms = float(row["setup_data_host_time"])
    total_ms = allocate_ms + float(row["total_time"])

    generate_ms = (
        float(row.get("query_signature_gpu_time", 0.0))
        + float(row.get("data_signature_gpu_time", 0.0))
    )

    return {
        "Allocate": allocate_ms / 1000.0,
        "Generate": generate_ms / 1000.0,
        "Filter": filter_ms / 1000.0,
        "Refine": refine_ms / 1000.0,
        "Join": float(row["join_host_time"]) / 1000.0,
        "Total": total_ms / 1000.0,
    }


def plot_comparison(
    native: Dict[str, float],
    python: Dict[str, float],
    output_png: Path,
    output_pdf: Path | None = None,
) -> None:
    categories = ["Allocate", "Generate", "Filter", "Refine", "Join", "Total"]

    native_values = [native[c] for c in categories]
    python_values = [python[c] for c in categories]

    x = list(range(len(categories)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    native_bars = ax.bar(
        [i - width / 2 for i in x],
        native_values,
        width,
        label="Native SIGMo",
    )

    python_bars = ax.bar(
        [i + width / 2 for i in x],
        python_values,
        width,
        label="Python binding",
    )

    ax.set_ylabel("Execution time (s)")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    for bars in (native_bars, python_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300)

    if output_pdf is not None:
        fig.savefig(output_pdf)

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--native-csv",
        required=True,
        help="CSV generated from native SIGMo results.",
    )

    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY_PATH,
        help="Query SMARTS file.",
    )

    parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Data SMILES file.",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=7,
        help="Number of refinement iterations.",
    )

    parser.add_argument(
        "--find-all",
        action="store_true",
        help="Use find_all=True. Default is False, corresponding to find_first=True.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="SYCL device selector for the Python binding.",
    )

    parser.add_argument(
        "--out",
        default="benchmarks/results/native_vs_python_times.png",
        help="Output PNG path.",
    )

    parser.add_argument(
        "--excel-out",
        default="benchmarks/results/native_vs_python_times.xlsx",
        help="Output Excel path.",
    )

    args = parser.parse_args()

    native_csv = Path(args.native_csv)
    output_png = Path(args.out)
    output_pdf = output_png.with_suffix(".pdf")
    excel_out = Path(args.excel_out)

    print("Loading datasets...")
    query_graphs = sigmo.load_molecules(args.query, input_format="smarts")
    data_graphs = sigmo.load_molecules(args.data, input_format="smiles")

    print("Running Python binding pipeline...")
    python_times = run_python_pipeline(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        iterations=args.iterations,
        find_all=args.find_all,
        device=args.device,
    )

    print("Loading native SIGMo timings...")
    native_times = get_native_breakdown(
        native_csv=native_csv,
        iterations=args.iterations,
        find_all=args.find_all,
    )

    rows = []

    for phase in ["Allocate", "Generate", "Filter", "Refine", "Join", "Total"]:
        rows.append(
            {
                "phase": phase,
                "native_sigmo_s": native_times[phase],
                "python_binding_s": python_times[phase],
                "delta_s": python_times[phase] - native_times[phase],
                "ratio_python_over_native": (
                    python_times[phase] / native_times[phase]
                    if native_times[phase] > 0
                    else None
                ),
            }
        )

    df = pd.DataFrame(rows)

    excel_out.parent.mkdir(parents=True, exist_ok=True)

    csv_out = excel_out.with_suffix(".csv")

    try:
        with pd.ExcelWriter(excel_out, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="comparison", index=False)

        print(f"Saved Excel: {excel_out}")

    except ModuleNotFoundError:
        df.to_csv(csv_out, index=False)

        print(
            "openpyxl is not installed, so the Excel file could not be written. "
            f"Saved CSV instead: {csv_out}"
        )

    plot_comparison(
        native=native_times,
        python=python_times,
        output_png=output_png,
        output_pdf=output_pdf,
    )

    print()
    print(df.to_string(index=False))
    print()
    print(f"Saved plot:  {output_png}")
    print(f"Saved PDF:   {output_pdf}")
    print(f"Saved Excel: {excel_out}")


if __name__ == "__main__":
    main()