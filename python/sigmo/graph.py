from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
    """Crea il dizionario CSR standard accettato dal binding SIGMo."""
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
    Converte una stringa SMARTS/SMILES in CSR.

    input_format:
        - "auto": prova SMARTS e poi SMILES
        - "smarts": interpreta come SMARTS
        - "smiles": interpreta come SMILES
    """
    value = str(value).strip()
    if not value:
        raise ValueError("Stringa chimica vuota.")

    mol, parsed_as = _parse_chemical_string(value, input_format=input_format)
    if mol is None:
        raise ValueError(f"Stringa chimica non valida ({input_format}): {value}")

    csr = rdkit_mol_to_csr(
        mol,
        name=name or _default_name(value, index),
        input=value,
        input_format=parsed_as,
        original_index=index,
    )
    return csr


def smarts_to_csr_from_string(smarts: str) -> CSRGraph:
    """Alias retrocompatibile: accetta SMARTS/SMILES e restituisce CSR."""
    return chemical_string_to_csr(smarts, input_format="auto")


def rdkit_mol_to_csr(mol: Chem.Mol, *, name: str = "molecule", **metadata: Any) -> CSRGraph:
    """Converte un oggetto RDKit Mol in CSR."""
    if mol is None:
        raise ValueError("RDKit Mol non valido: None.")

    num_nodes = mol.GetNumAtoms()
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    node_labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_labels_map: Dict[Tuple[int, int], int] = {}

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
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

    return make_csr_graph(row_offsets, column_indices, node_labels, edge_labels, num_nodes, name, **metadata)


def load_molecules(
    source: Union[str, os.PathLike, Sequence[Any]],
    *,
    input_format: str = "auto",
    strict: bool = False,
    return_report: bool = False,
) -> Union[List[CSRGraph], Tuple[List[CSRGraph], Dict[str, Any]]]:
    """
    Carica molecole/grafi in formato CSR.

    Supporta:
        - path a file .smi/.smiles/.smarts/.txt con righe "MOLECOLA nome opzionale"
        - singola stringa chimica
        - lista di stringhe chimiche
        - lista di dizionari CSR gia' pronti
        - lista di RDKit Mol

    Se return_report=True restituisce anche un report explainable di parsing.
    """
    items = _normalise_source(source)
    graphs: List[CSRGraph] = []
    invalid: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        try:
            graph = _item_to_csr(item, idx=idx, input_format=input_format)
            graphs.append(graph)
        except Exception as exc:
            invalid.append({"index": idx, "item": _safe_repr(item), "error": str(exc)})
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
    """Alias retrocompatibile per caricare un file SMARTS/SMILES."""
    return load_molecules(file_path, input_format="auto")  # type: ignore[return-value]


def toy_two_node_graph() -> CSRGraph:
    """Grafo di test semplice: C-C (etano)."""
    return make_csr_graph([0, 1, 2], [1, 0], [6, 6], [1, 1], 2, "ethane", input="CC", input_format="smiles")


def to_networkx(graph: CSRGraph):
    """Converte un CSR SIGMo in networkx.Graph. Richiede networkx installato."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError("Installa networkx per usare to_networkx().") from exc

    g = nx.Graph(name=graph.get("name", "graph"))
    for i, label in enumerate(graph["node_labels"]):
        g.add_node(i, label=label, atomic_num=label)

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
    Converte un networkx.Graph in CSR.

    Convenzione:
        - nodo: attributo 'atomic_num' oppure 'label'
        - arco: attributo 'bond_type' oppure 'label'
    """
    nodes = sorted(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    adj: List[List[int]] = [[] for _ in nodes]
    edge_labels_map: Dict[Tuple[int, int], int] = {}

    node_labels = []
    for node in nodes:
        attrs = nx_graph.nodes[node]
        node_labels.append(int(attrs.get("atomic_num", attrs.get("label", 0))))

    for u_raw, v_raw, attrs in nx_graph.edges(data=True):
        u, v = node_to_idx[u_raw], node_to_idx[v_raw]
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


def _parse_chemical_string(value: str, *, input_format: str) -> Tuple[Optional[Chem.Mol], str]:
    fmt = (input_format or "auto").lower()
    if fmt == "smarts":
        return Chem.MolFromSmarts(value), "smarts"
    if fmt == "smiles":
        return Chem.MolFromSmiles(value), "smiles"
    if fmt != "auto":
        raise ValueError(f"Formato input non supportato: {input_format}")

    mol = Chem.MolFromSmarts(value)
    if mol is not None:
        return mol, "smarts"
    mol = Chem.MolFromSmiles(value)
    if mol is not None:
        return mol, "smiles"
    return None, "unknown"


def _bond_label(bond: Chem.Bond) -> int:
    """
    Converte il tipo di legame RDKit nella label usata da SIGMo.

    Nota progettuale:
    per ora manteniamo la stessa politica del prototipo originale:
    - singolo  -> 1
    - doppio   -> 2
    - triplo   -> 3
    - aromatico RDKit -> int(1.5) = 1

    Questa scelta è meno informativa dal punto di vista chimico, perché
    collassa i legami aromatici sui legami singoli, ma riduce il rischio
    di mandare al kernel C++ etichette non previste, come 12.

    In futuro si potrà introdurre una modalità:
        aromatic_policy="explicit"
    solo dopo aver verificato che il backend SIGMo supporti davvero
    la label 12.
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
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            value = parts[0]
            name = parts[1] if len(parts) > 1 else f"{path.stem}:{line_no}"
            items.append({"value": value, "name": name, "line": line_no, "source_file": str(path)})
    return items


def _item_to_csr(item: Any, *, idx: int, input_format: str) -> CSRGraph:
    if isinstance(item, dict) and _looks_like_csr(item):
        graph = dict(item)
        graph.setdefault("name", f"graph_{idx}")
        graph.setdefault("original_index", idx)
        return graph

    if isinstance(item, dict) and "value" in item:
        return chemical_string_to_csr(
            item["value"],
            name=item.get("name"),
            input_format=input_format,
            index=idx,
        ) | {k: v for k, v in item.items() if k not in {"value", "name"}}

    # RDKit Mol: evitiamo isinstance stretto per compatibilita' tra build RDKit.
    if hasattr(item, "GetAtoms") and hasattr(item, "GetBonds"):
        return rdkit_mol_to_csr(item, name=f"mol_{idx}", input_format="rdkit", original_index=idx)

    return chemical_string_to_csr(str(item), input_format=input_format, index=idx)


def _looks_like_csr(item: Dict[str, Any]) -> bool:
    required = {"row_offsets", "column_indices", "node_labels", "edge_labels", "num_nodes"}
    return required.issubset(item.keys())


def _default_name(value: str, index: Optional[int]) -> str:
    prefix = f"ID:{index}" if index is not None else "molecule"
    preview = value[:20] + "..." if len(value) > 20 else value
    return f"{prefix} ({preview})"


def _safe_repr(item: Any, max_len: int = 120) -> str:
    text = repr(item)
    return text if len(text) <= max_len else text[:max_len] + "..."
