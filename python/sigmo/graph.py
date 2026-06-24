from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from rdkit import Chem

CSRGraph = Dict[str, Any]

SIGMO_NATIVE_ATOM_LABELS = { "N": 0, "Cl": 1, "*": 2, "Br": 3, "I": 4, "P": 5, "H": 6, "O": 7, "C": 8, "S": 9, "F": 10, } 

def _atom_label(atom: Chem.Atom) -> int: 
    """ Return the node label used by the native SIGMo preprocessing script. 
    Unknown symbols intentionally fall back to 0, matching smile2graph.py. """ 
    return SIGMO_NATIVE_ATOM_LABELS.get(atom.GetSymbol(), 0)

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

    The CSR construction reproduces the ordering used by the native SIGMo
    preprocessing workflow:

    1. RDKit bonds are inserted in their original order;
    2. each undirected edge is emitted once following node order and
       neighbour insertion order;
    3. the CSR is filled by inserting both directions of every emitted edge
       in that global edge order, as done by SIGMo's C++ reader.
    """
    if mol is None:
        raise ValueError("Invalid RDKit Mol: None.")

    num_nodes = mol.GetNumAtoms()
    node_labels = [_atom_label(atom) for atom in mol.GetAtoms()]

    # Keep RDKit bond insertion order for each endpoint.
    insertion_adj: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_labels_map: Dict[Tuple[int, int], int] = {}

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        label = _bond_label(bond)

        insertion_adj[u].append(v)
        insertion_adj[v].append(u)

        edge_labels_map[(u, v)] = label
        edge_labels_map[(v, u)] = label

    # Reproduce the edge sequence emitted by the native preprocessing path:
    # visit nodes in index order and emit each undirected edge only once.
    native_edge_order: List[Tuple[int, int, int]] = []
    seen_edges: set[Tuple[int, int]] = set()

    for u in range(num_nodes):
        for v in insertion_adj[u]:
            edge_key = (u, v) if u < v else (v, u)

            if edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)
            native_edge_order.append(
                (u, v, edge_labels_map[(u, v)])
            )

    # Same CSR fill strategy used by SIGMo's IntermediateGraph::toCSRGraph().
    row_offsets = [0] * (num_nodes + 1)

    for u, v, _ in native_edge_order:
        row_offsets[u + 1] += 1
        row_offsets[v + 1] += 1

    for node in range(1, len(row_offsets)):
        row_offsets[node] += row_offsets[node - 1]

    total_directed_edges = row_offsets[-1]

    column_indices = [0] * total_directed_edges
    edge_labels = [0] * total_directed_edges
    current_positions = [0] * num_nodes

    for u, v, label in native_edge_order:
        u_pos = row_offsets[u] + current_positions[u]
        column_indices[u_pos] = v
        edge_labels[u_pos] = label
        current_positions[u] += 1

        v_pos = row_offsets[v] + current_positions[v]
        column_indices[v_pos] = u
        edge_labels[v_pos] = label
        current_positions[v] += 1

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

    Alternative version that respects SIGMo native parser:

    bond_type = bond.GetBondType() 
    if bond_type == Chem.rdchem.BondType.AROMATIC: 
        return 4 
    return int(bond.GetBondTypeAsDouble())
    """
    bond_type = bond.GetBondType() 

    if bond_type == Chem.rdchem.BondType.AROMATIC: 
        return 4 
    
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