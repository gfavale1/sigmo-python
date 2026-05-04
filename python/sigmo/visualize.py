"""
Visualization utilities for SIGMo Python.

This module provides optional visualization helpers based on RDKit and,
when available, NetworkX/Matplotlib.

The visualization layer is intentionally separate from the core SIGMo pipeline.
It is meant for inspection, debugging and examples, not for executing native
SIGMo kernels.

Main features:
    - draw individual molecules;
    - draw query-target molecule pairs;
    - highlight query substructures inside target molecules using RDKit;
    - convert SIGMo CSR graphs to NetworkX graphs;
    - draw internal SIGMo CSR graphs for debugging.

Notes:
    Match highlighting is RDKit-based. Current SIGMo results are pair-level
    results, meaning that SIGMo reports that query graph i matches data graph j,
    but it does not expose an atom-level mapping for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from rdkit import Chem
from rdkit.Chem import Draw


def mol_from_input(
    molecule: Union[str, Chem.Mol],
    *,
    input_format: str = "auto",
    role: str = "data",
) -> Chem.Mol:
    """
    Convert a SMILES/SMARTS string or RDKit Mol into an RDKit Mol.

    Args:
        molecule: SMILES string, SMARTS string, or existing RDKit Mol.
        input_format: One of "auto", "smiles", or "smarts".
        role: Either "query" or "data". In auto mode, query inputs prefer
            SMARTS parsing, while data inputs prefer SMILES parsing.

    Returns:
        An RDKit Mol object.

    Raises:
        ValueError: If the input is empty, uses an unsupported format, or
        cannot be parsed.
    """
    if isinstance(molecule, Chem.Mol):
        return molecule

    text = str(molecule).strip()
    input_format = input_format.lower()
    role = role.lower()

    if not text:
        raise ValueError("Empty molecule input.")

    if input_format == "smiles":
        mol = Chem.MolFromSmiles(text)

    elif input_format == "smarts":
        mol = Chem.MolFromSmarts(text)

    elif input_format == "auto":
        if role == "query":
            mol = Chem.MolFromSmarts(text)
            if mol is None:
                mol = Chem.MolFromSmiles(text)
        else:
            mol = Chem.MolFromSmiles(text)
            if mol is None:
                mol = Chem.MolFromSmarts(text)

    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    if mol is None:
        raise ValueError(f"Could not parse molecule: {text}")

    return mol


def draw_molecule(
    molecule: Union[str, Chem.Mol],
    *,
    input_format: str = "auto",
    role: str = "data",
    output_path: Optional[Union[str, Path]] = None,
    size: Tuple[int, int] = (400, 300),
    legend: Optional[str] = None,
):
    """
    Draw a single molecule.

    Args:
        molecule: SMILES string, SMARTS string, or RDKit Mol.
        input_format: One of "auto", "smiles", or "smarts".
        role: Either "query" or "data".
        output_path: Optional path where the generated image is saved.
        size: Output image size.
        legend: Optional molecule legend.

    Returns:
        The generated RDKit/PIL image object.
    """
    mol = mol_from_input(
        molecule,
        input_format=input_format,
        role=role,
    )

    image = Draw.MolToImage(
        mol,
        size=size,
        legend=legend or "",
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return image


def draw_match_pair(
    query: Union[str, Chem.Mol],
    target: Union[str, Chem.Mol],
    *,
    query_format: str = "auto",
    target_format: str = "auto",
    output_path: Optional[Union[str, Path]] = None,
    sub_img_size: Tuple[int, int] = (450, 300),
    legends: Tuple[str, str] = ("Query", "Target"),
    highlight: bool = True,
):
    """
    Draw a query-target molecule pair.

    If highlight=True, RDKit computes one substructure match and highlights
    the matched atoms inside the target molecule.

    Important:
        The highlight is computed by RDKit and is used only for visualization.
        It is not an atom-level mapping returned by SIGMo.

    Args:
        query: Query molecule or pattern.
        target: Target molecule.
        query_format: Format used to parse the query.
        target_format: Format used to parse the target.
        output_path: Optional path where the generated image is saved.
        sub_img_size: Size of each molecule image in the grid.
        legends: Pair of labels shown under query and target.
        highlight: Whether to highlight the query inside the target.

    Returns:
        The generated RDKit/PIL image object.
    """
    query_mol = mol_from_input(
        query,
        input_format=query_format,
        role="query",
    )

    target_mol = mol_from_input(
        target,
        input_format=target_format,
        role="data",
    )

    highlight_lists: List[List[int]] = [[], []]

    if highlight:
        match = target_mol.GetSubstructMatch(query_mol)
        if match:
            highlight_lists[1] = list(match)

    image = Draw.MolsToGridImage(
        [query_mol, target_mol],
        molsPerRow=2,
        subImgSize=sub_img_size,
        legends=list(legends),
        highlightAtomLists=highlight_lists,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)

    return image


def to_networkx(graph: Dict[str, Any]):
    """
    Convert a SIGMo CSR graph dictionary to a NetworkX graph.

    Nodes receive a "label" attribute from node_labels.
    Edges receive a "label" attribute from edge_labels.

    Requires:
        networkx
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "NetworkX is required for to_networkx(). "
            "Install it with: pip install networkx"
        ) from exc

    row_offsets = graph.get("row_offsets")
    column_indices = graph.get("column_indices")
    node_labels = graph.get("node_labels")
    edge_labels = graph.get("edge_labels")

    if row_offsets is None or column_indices is None:
        raise ValueError("CSR graph must contain row_offsets and column_indices.")

    num_nodes = int(graph.get("num_nodes", len(row_offsets) - 1))

    networkx_graph = nx.Graph()

    for node_idx in range(num_nodes):
        label = (
            node_labels[node_idx]
            if node_labels is not None and node_idx < len(node_labels)
            else None
        )

        networkx_graph.add_node(
            node_idx,
            label=label,
        )

    for src in range(num_nodes):
        start = int(row_offsets[src])
        end = int(row_offsets[src + 1])

        for pos in range(start, end):
            dst = int(column_indices[pos])

            edge_label = (
                edge_labels[pos]
                if edge_labels is not None and pos < len(edge_labels)
                else None
            )

            if src <= dst:
                networkx_graph.add_edge(
                    src,
                    dst,
                    label=edge_label,
                )

    networkx_graph.graph["name"] = graph.get("name", "")
    networkx_graph.graph["input"] = graph.get("input", "")

    return networkx_graph


def _format_node_debug_label(node: int, label: Any) -> str:
    """
    Format node labels for CSR debug visualization.

    Examples:
        node=0, label=6 -> "0:C"
        node=2, label=8 -> "2:O"

    If the label is not a known atomic number, it is shown as-is.
    """
    atomic_symbols = {
        1: "H",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        15: "P",
        16: "S",
        17: "Cl",
        35: "Br",
        53: "I",
    }

    try:
        label_int = int(label)
        symbol = atomic_symbols.get(label_int, str(label_int))
    except Exception:
        symbol = str(label)

    return f"{node}:{symbol}"


def draw_graph(
    graph: Dict[str, Any],
    *,
    output_path: Optional[Union[str, Path]] = None,
    with_labels: bool = True,
    show_node_labels: bool = True,
    show_edge_labels: bool = True,
    layout: str = "spring",
    figsize: Tuple[int, int] = (7, 5),
):
    """
    Draw a SIGMo CSR graph using NetworkX and Matplotlib.

    This visualization is intended for debugging the internal CSR graph
    representation. It is not intended to replace chemically accurate molecule
    rendering.

    Node labels are shown as:

        node_index:atomic_label

    Examples:
        0:C
        1:C
        2:O

    Requires:
        networkx
        matplotlib

    Returns:
        The NetworkX graph used for drawing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt
        import networkx as nx

    except ImportError as exc:
        raise ImportError(
            "draw_graph() requires networkx and matplotlib. "
            "Install them with: pip install networkx matplotlib"
        ) from exc

    networkx_graph = to_networkx(graph)

    if layout == "spring":
        pos = nx.spring_layout(networkx_graph, seed=42, k=1.2)
    elif layout == "shell":
        pos = nx.shell_layout(networkx_graph)
    elif layout == "planar":
        try:
            pos = nx.planar_layout(networkx_graph)
        except Exception:
            pos = nx.kamada_kawai_layout(networkx_graph)
    else:
        pos = nx.kamada_kawai_layout(networkx_graph)

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(
        networkx_graph,
        pos,
        node_size=950,
        ax=ax,
    )

    nx.draw_networkx_edges(
        networkx_graph,
        pos,
        width=1.8,
        ax=ax,
    )

    if with_labels and show_node_labels:
        labels = {
            node: _format_node_debug_label(node, data.get("label", ""))
            for node, data in networkx_graph.nodes(data=True)
        }

        nx.draw_networkx_labels(
            networkx_graph,
            pos,
            labels=labels,
            font_size=9,
            font_color="white",
            font_weight="bold",
            ax=ax,
        )

    if show_edge_labels:
        edge_labels = {
            (u, v): data.get("label", "")
            for u, v, data in networkx_graph.edges(data=True)
            if data.get("label", "") is not None
        }

        if edge_labels:
            nx.draw_networkx_edge_labels(
                networkx_graph,
                pos,
                edge_labels=edge_labels,
                font_size=8,
                label_pos=0.5,
                ax=ax,
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "fc": "white",
                    "ec": "none",
                    "alpha": 0.8,
                },
            )

    title = graph.get("name") or graph.get("input") or "SIGMo CSR graph"
    ax.set_title(title)
    ax.axis("off")

    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.03)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    return networkx_graph