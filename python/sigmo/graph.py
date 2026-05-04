from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rdkit import Chem

CSRGraph = Dict[str, Any]


def make_csr_graph(
    row_offsets: Sequence[int],
    column_indices: Sequence[int],
    node_labels: Sequence[int],
    edge_labels: Sequence[int],
    num_nodes: Optional[int] = None,
    name: str = "graph",
    **metadata: Any,
) -> CSRGraph:
    """
    Create a SIGMo-compatible CSR graph dictionary.

    The native SIGMo backend expects graphs to be represented through:
        - row_offsets
        - column_indices
        - node_labels
        - edge_labels
        - num_nodes

    Additional metadata is preserved inside the returned dictionary.
    """
    graph = {
        "row_offsets": list(row_offsets),
        "column_indices": list(column_indices),
        "node_labels": list(node_labels),
        "edge_labels": list(edge_labels),
        "num_nodes": int(num_nodes if num_nodes is not None else len(node_labels)),
        "name": str(name),
    }
    graph.update(metadata)
    return graph


def chemical_string_to_csr(
    value: str,
    *,
    name: Optional[str] = None,
    input_format: str = "auto",
    index: Optional[int] = None,
) -> CSRGraph:
    """
    Convert a SMILES/SMARTS string into a SIGMo-compatible CSR graph.

    Args:
        value: Input chemical string.
        name: Optional graph name.
        input_format: One of "auto", "smarts", or "smiles".
            - "auto": try SMARTS first, then SMILES.
            - "smarts": parse as SMARTS.
            - "smiles": parse as SMILES.
        index: Optional original index used for metadata.

    Returns:
        A CSR graph dictionary.

    Raises:
        ValueError: If the input string is empty or cannot be parsed.
    """
    value = str(value).strip()
    if not value:
        raise ValueError("Empty chemical string.")

    mol, parsed_as = _parse_chemical_string(value, input_format=input_format)
    if mol is None:
        raise ValueError(f"Invalid chemical string ({input_format}): {value}")

    return rdkit_mol_to_csr(
        mol,
        name=name or _default_name(value, index),
        input=value,
        input_format=parsed_as,
        original_index=index,
    )


def smarts_to_csr_from_string(smarts: str) -> CSRGraph:
    """
    Backward-compatible alias for converting a SMARTS/SMILES string to CSR.
    """
    return chemical_string_to_csr(smarts, input_format="auto")


def rdkit_mol_to_csr(
    mol: Chem.Mol,
    *,
    name: str = "molecule",
    **metadata: Any,
) -> CSRGraph:
    """
    Convert an RDKit Mol object into a SIGMo-compatible CSR graph.

    Notes:
        The resulting CSR graph is a structural representation. It does not
        preserve the full SMARTS semantics used internally by RDKit. Atom labels
        are currently encoded as atomic numbers, and bond labels are encoded as
        backend-safe integer bond labels.
    """
    if mol is None:
        raise ValueError("Invalid RDKit Mol: None.")

    num_nodes = mol.GetNumAtoms()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    node_labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_labels_map: Dict[Tuple[int, int], int] = {}

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        label = _bond_label(bond)

        adj[u].append(v)
        adj[v].append(u)

        edge_labels_map[(u, v)] = label
        edge_labels_map[(v, u)] = label

    row_offsets = [0]
    column_indices: List[int] = []
    edge_labels: List[int] = []

    for node in range(num_nodes):
        for neigh in sorted(adj[node]):
            column_indices.append(neigh)
            edge_labels.append(edge_labels_map[(node, neigh)])
        row_offsets.append(len(column_indices))

    return make_csr_graph(
        row_offsets,
        column_indices,
        node_labels,
        edge_labels,
        num_nodes,
        name,
        **metadata,
    )


def load_molecules(
    source: Union[str, os.PathLike, Sequence[Any]],
    *,
    input_format: str = "auto",
    strict: bool = False,
    return_report: bool = False,
) -> Union[List[CSRGraph], Tuple[List[CSRGraph], Dict[str, Any]]]:
    """
    Load molecules or graphs and convert them to SIGMo-compatible CSR graphs.

    Supported inputs:
        - file path containing one molecule per line;
        - single SMILES/SMARTS string;
        - sequence of SMILES/SMARTS strings;
        - sequence of CSR dictionaries;
        - sequence of RDKit Mol objects.

    File lines are expected to have the format:

        MOLECULE_STRING optional_name

    Args:
        source: File path, chemical string, list of strings, CSR graphs,
            or RDKit Mol objects.
        input_format: One of "auto", "smarts", or "smiles".
        strict: If True, raise on the first invalid item. If False, skip
            invalid items and include them in the report.
        return_report: If True, return both graphs and a parsing report.

    Returns:
        A list of CSR graphs, or a tuple (graphs, report) if return_report=True.
    """
    items = _normalise_source(source)
    graphs: List[CSRGraph] = []
    invalid: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        try:
            graph = _item_to_csr(item, idx=idx, input_format=input_format)
            graphs.append(graph)
        except Exception as exc:
            invalid.append(
                {
                    "index": idx,
                    "item": _safe_repr(item),
                    "error": str(exc),
                }
            )
            if strict:
                raise

    report = {
        "loaded": len(graphs),
        "invalid": len(invalid),
        "invalid_items": invalid,
    }

    if return_report:
        return graphs, report

    return graphs


def smarts_to_csr(file_path: Union[str, os.PathLike]) -> List[CSRGraph]:
    """
    Backward-compatible alias for loading a SMARTS/SMILES file as CSR graphs.
    """
    return load_molecules(file_path, input_format="auto")  # type: ignore[return-value]


def toy_two_node_graph() -> CSRGraph:
    """
    Return a minimal C-C toy graph used in tests and examples.
    """
    return make_csr_graph(
        [0, 1, 2],
        [1, 0],
        [6, 6],
        [1, 1],
        2,
        "ethane",
        input="CC",
        input_format="smiles",
    )


def to_networkx(graph: CSRGraph):
    """
    Convert a SIGMo CSR graph into a networkx.Graph.

    Requires:
        networkx
    """
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("Install networkx to use to_networkx().") from exc

    g = nx.Graph(name=graph.get("name", "graph"))

    for idx, label in enumerate(graph["node_labels"]):
        g.add_node(idx, label=label, atomic_num=label)

    row_offsets = graph["row_offsets"]
    cols = graph["column_indices"]
    edge_labels = graph["edge_labels"]

    for u in range(graph["num_nodes"]):
        for pos in range(row_offsets[u], row_offsets[u + 1]):
            v = cols[pos]
            if u <= v:
                g.add_edge(u, v, label=edge_labels[pos])

    return g


def from_networkx(nx_graph: Any, *, name: Optional[str] = None) -> CSRGraph:
    """
    Convert a networkx.Graph into a SIGMo-compatible CSR graph.

    Node attributes:
        - atomic_num, or
        - label

    Edge attributes:
        - bond_type, or
        - label
    """
    nodes = sorted(nx_graph.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}

    adj: List[List[int]] = [[] for _ in nodes]
    edge_labels_map: Dict[Tuple[int, int], int] = {}

    node_labels = []
    for node in nodes:
        attrs = nx_graph.nodes[node]
        node_labels.append(int(attrs.get("atomic_num", attrs.get("label", 0))))

    for u_raw, v_raw, attrs in nx_graph.edges(data=True):
        u = node_to_idx[u_raw]
        v = node_to_idx[v_raw]
        label = int(attrs.get("bond_type", attrs.get("label", 1)))

        adj[u].append(v)
        adj[v].append(u)

        edge_labels_map[(u, v)] = label
        edge_labels_map[(v, u)] = label

    row_offsets = [0]
    column_indices: List[int] = []
    edge_labels: List[int] = []

    for u in range(len(nodes)):
        for v in sorted(adj[u]):
            column_indices.append(v)
            edge_labels.append(edge_labels_map[(u, v)])
        row_offsets.append(len(column_indices))

    return make_csr_graph(
        row_offsets,
        column_indices,
        node_labels,
        edge_labels,
        len(nodes),
        name or getattr(nx_graph, "name", "networkx_graph"),
        input_format="networkx",
    )


def _parse_chemical_string(
    value: str,
    *,
    input_format: str,
) -> Tuple[Optional[Chem.Mol], str]:
    fmt = (input_format or "auto").lower()

    if fmt == "smarts":
        return Chem.MolFromSmarts(value), "smarts"

    if fmt == "smiles":
        return Chem.MolFromSmiles(value), "smiles"

    if fmt != "auto":
        raise ValueError(f"Unsupported input format: {input_format}")

    mol = Chem.MolFromSmarts(value)
    if mol is not None:
        return mol, "smarts"

    mol = Chem.MolFromSmiles(value)
    if mol is not None:
        return mol, "smiles"

    return None, "unknown"


def _bond_label(bond: Chem.Bond) -> int:
    """
    Convert an RDKit bond into the integer label expected by SIGMo.

    Current backend-safe policy:
        - single bond  -> 1
        - double bond  -> 2
        - triple bond  -> 3
        - aromatic bond -> int(1.5) = 1

    RDKit represents aromatic bonds with bond order 1.5. The current native
    backend expects stable integer labels and may become unstable with
    unsupported labels. Therefore, aromatic bonds are intentionally collapsed
    to label 1.

    This makes the CSR representation less chemically expressive than RDKit's
    full SMARTS semantics, but keeps the SIGMo backend stable.
    """
    return int(bond.GetBondTypeAsDouble())


def _normalise_source(source: Union[str, os.PathLike, Sequence[Any]]) -> List[Any]:
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        if path.exists() and path.is_file():
            return _read_molecule_file(path)
        return [str(source)]

    return list(source)


def _read_molecule_file(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            parts = line.split(maxsplit=1)
            value = parts[0]
            name = parts[1] if len(parts) > 1 else f"{path.stem}:{line_no}"

            items.append(
                {
                    "value": value,
                    "name": name,
                    "line": line_no,
                    "source_file": str(path),
                }
            )

    return items


def _item_to_csr(item: Any, *, idx: int, input_format: str) -> CSRGraph:
    if isinstance(item, dict) and _looks_like_csr(item):
        graph = dict(item)
        graph.setdefault("name", f"graph_{idx}")
        graph.setdefault("original_index", idx)
        return graph

    if isinstance(item, dict) and "value" in item:
        graph = chemical_string_to_csr(
            item["value"],
            name=item.get("name"),
            input_format=input_format,
            index=idx,
        )

        metadata = {key: value for key, value in item.items() if key not in {"value", "name"}}
        graph.update(metadata)
        return graph

    # RDKit Mol: avoid strict isinstance checks for compatibility across RDKit builds.
    if hasattr(item, "GetAtoms") and hasattr(item, "GetBonds"):
        return rdkit_mol_to_csr(
            item,
            name=f"mol_{idx}",
            input_format="rdkit",
            original_index=idx,
        )

    return chemical_string_to_csr(
        str(item),
        input_format=input_format,
        index=idx,
    )


def _looks_like_csr(item: Dict[str, Any]) -> bool:
    required = {
        "row_offsets",
        "column_indices",
        "node_labels",
        "edge_labels",
        "num_nodes",
    }
    return required.issubset(item.keys())


def _default_name(value: str, index: Optional[int]) -> str:
    prefix = f"ID:{index}" if index is not None else "molecule"
    preview = value[:20] + "..." if len(value) > 20 else value
    return f"{prefix} ({preview})"


def _safe_repr(item: Any, max_len: int = 120) -> str:
    text = repr(item)
    return text if len(text) <= max_len else text[:max_len] + "..."