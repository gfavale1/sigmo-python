"""
Visualization examples for the SIGMo Python interface.

This script demonstrates:
1. drawing a single molecule;
2. drawing a query-target pair with highlighted substructure match;
3. drawing a small internal SIGMo CSR graph;
4. drawing multiple real molecules from the benchmark dataset;
5. drawing real query-target match pairs found by SIGMo.
"""

from pathlib import Path
from typing import Tuple

from rdkit import Chem

import sigmo
from sigmo.visualize import draw_graph, draw_match_pair, draw_molecule

OUTPUT_DIR = Path("examples/outputs")
DATA_FILE = Path("benchmarks/datasets/data.smarts")
QUERY_FILE = Path("benchmarks/datasets/query.smarts")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def split_dataset_line(line: str, fallback_name: str) -> Tuple[str, str]:
    """
    Split a dataset line into (molecule_input, name).

    The expected format is:

        MOLECULE_STRING optional_name

    Blank lines and comments return an empty molecule string.
    """
    stripped = line.strip()

    if not stripped or stripped.startswith("#"):
        return "", fallback_name

    parts = stripped.split(maxsplit=1)
    molecule_input = parts[0]
    name = parts[1].strip() if len(parts) > 1 else fallback_name

    return molecule_input, name


def find_dataset_graph_samples(
    path: Path,
    *,
    count: int = 3,
    min_nodes: int = 12,
    max_nodes: int = 40,
    max_scan: int = 10000,
):
    """
    Find multiple real molecules from the benchmark dataset.

    The selected graphs are large enough to test visualization,
    but not too large to make the CSR debug plot unreadable.
    """
    samples = []

    if not path.exists():
        print(f"    dataset not found: {path}")
        return samples

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if line_idx >= max_scan:
                break

            molecule_input, name = split_dataset_line(
                line,
                fallback_name=f"data:{line_idx + 1}",
            )

            if not molecule_input:
                continue

            try:
                graphs = sigmo.load_molecules(
                    [molecule_input],
                    input_format="smiles",
                )
            except Exception:
                continue

            if not graphs:
                continue

            graph = graphs[0]
            num_nodes = int(graph.get("num_nodes", 0))

            if min_nodes <= num_nodes <= max_nodes:
                graph["name"] = f"dataset sample - {name}"
                graph["input"] = molecule_input
                graph["original_index"] = line_idx
                samples.append(graph)

                if len(samples) >= count:
                    break

    return samples


def find_sigmo_match_samples(
    query_path: Path,
    data_path: Path,
    *,
    count: int = 3,
    query_limit: int = 20,
    data_limit: int = 200,
    iterations: int = 0,
    device: str = "auto",
):
    """
    Run SIGMo on a subset of the benchmark dataset and return
    a few real query-target matches for visualization.
    """
    if not query_path.exists():
        print(f"    query dataset not found: {query_path}")
        return []

    if not data_path.exists():
        print(f"    data dataset not found: {data_path}")
        return []

    query_graphs = sigmo.load_molecules(query_path, input_format="auto")
    data_graphs = sigmo.load_molecules(data_path, input_format="auto")

    if query_limit is not None and query_limit > 0:
        query_graphs = query_graphs[:query_limit]

    if data_limit is not None and data_limit > 0:
        data_graphs = data_graphs[:data_limit]

    if not query_graphs or not data_graphs:
        print("    no graphs available for SIGMo matching")
        return []

    try:
        result = sigmo.run_isomorphism(
            query_graphs,
            data_graphs,
            iterations=iterations,
            find_first=False,
            device=device,
        )
    except Exception as exc:
        print(f"    SIGMo matching failed: {exc}")
        return []

    if not getattr(result, "matches", None):
        print("    no real matches found in the selected subset")
        return []

    samples = []
    seen = set()

    for match in result.matches:
        key = (match.query_index, match.data_index)
        if key in seen:
            continue
        seen.add(key)

        q_graph = query_graphs[match.query_index]
        d_graph = data_graphs[match.data_index]

        q_input = q_graph.get("input", "")
        d_input = d_graph.get("input", "")

        q_mol = Chem.MolFromSmarts(q_input)
        d_mol = Chem.MolFromSmiles(d_input)

        if q_mol is None or d_mol is None:
            continue

        # Keep only samples that RDKit can also highlight cleanly.
        if not d_mol.HasSubstructMatch(q_mol):
            continue

        samples.append(
            {
                "query_index": match.query_index,
                "data_index": match.data_index,
                "query_name": q_graph.get("name", f"query:{match.query_index + 1}"),
                "data_name": d_graph.get("name", f"data:{match.data_index + 1}"),
                "query_input": q_input,
                "data_input": d_input,
            }
        )

        if len(samples) >= count:
            break

    return samples


def main():
    query = "[nH]1cnoc1=O"
    target = "[O-]C(=O)[C@H]([NH3+])CN1OC(=O)NC1=O"

    print("[1] Drawing single molecule")
    draw_molecule(
        target,
        input_format="smiles",
        output_path=OUTPUT_DIR / "molecule.png",
        legend="Target molecule",
    )
    print(f"    saved: {OUTPUT_DIR / 'molecule.png'}")

    print("[2] Drawing query-target pair")
    draw_match_pair(
        query,
        target,
        query_format="smarts",
        target_format="smiles",
        output_path=OUTPUT_DIR / "match_pair.png",
        legends=("Query", "Target"),
        highlight=True,
    )
    print(f"    saved: {OUTPUT_DIR / 'match_pair.png'}")

    print("[3] Drawing internal SIGMo CSR graph (small example)")
    try:
        graphs_small = sigmo.load_molecules(["CCO"], input_format="smiles")
        draw_graph(
            graphs_small[0],
            output_path=OUTPUT_DIR / "csr_graph_small.png",
            layout="spring",
        )
        print(f"    saved: {OUTPUT_DIR / 'csr_graph_small.png'}")
    except ImportError as exc:
        print(f"    skipped: {exc}")

    print("[4] Drawing multiple real molecules from benchmark dataset")
    sample_graphs = find_dataset_graph_samples(
        DATA_FILE,
        count=3,
        min_nodes=12,
        max_nodes=40,
        max_scan=10000,
    )

    if not sample_graphs:
        print("    no suitable dataset graph found")
    else:
        for i, graph in enumerate(sample_graphs, start=1):
            molecule_input = graph.get("input", "")
            name = graph.get("name", f"dataset sample {i}")

            print(
                f"    sample {i}: nodes={graph['num_nodes']}, "
                f"directed_edges={len(graph['column_indices'])}, "
                f"name={name}"
            )

            mol_path = OUTPUT_DIR / f"dataset_molecule_{i}.png"
            draw_molecule(
                molecule_input,
                input_format="smiles",
                output_path=mol_path,
                legend=f"Dataset molecule {i}",
            )
            print(f"      saved molecule: {mol_path}")

            csr_path = OUTPUT_DIR / f"csr_graph_dataset_sample_{i}.png"
            draw_graph(
                graph,
                output_path=csr_path,
                layout="spring",
                figsize=(9, 7),
            )
            print(f"      saved CSR graph: {csr_path}")

    print("[5] Drawing real match pairs from benchmark dataset")
    match_samples = find_sigmo_match_samples(
        QUERY_FILE,
        DATA_FILE,
        count=3,
        query_limit=20,
        data_limit=200,
        iterations=0,
        device="auto",
    )

    if not match_samples:
        print("    no match pairs available")
    else:
        for i, sample in enumerate(match_samples, start=1):
            print(f"    match {i}: {sample['query_name']} -> {sample['data_name']}")
            print(f"      query: {sample['query_input']}")
            print(f"      data:  {sample['data_input']}")

            output_path = OUTPUT_DIR / f"real_dataset_match_pair_{i}.png"

            draw_match_pair(
                sample["query_input"],
                sample["data_input"],
                query_format="smarts",
                target_format="smiles",
                output_path=output_path,
                legends=(
                    f"Query: {sample['query_name']}",
                    f"Target: {sample['data_name']}",
                ),
                highlight=True,
            )

            print(f"      saved: {output_path}")


if __name__ == "__main__":
    main()