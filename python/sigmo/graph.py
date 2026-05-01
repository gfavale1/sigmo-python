import os
from rdkit import Chem

def make_csr_graph(row_offsets, column_indices, node_labels, edge_labels, num_nodes=None, name="graph"):
    """Crea il dizionario standard usando liste semplici (compatibile con la tua pipeline)."""
    return {
        "row_offsets": list(row_offsets),
        "column_indices": list(column_indices),
        "node_labels": list(node_labels),
        "edge_labels": list(edge_labels),
        "num_nodes": int(num_nodes if num_nodes is not None else len(node_labels)),
        "name": str(name)
    }

def smarts_to_csr_from_string(smarts):
    """Converte una stringa SMARTS/SMILES in formato CSR per SIGMo."""
    mol = Chem.MolFromSmarts(smarts)
    if mol is None: mol = Chem.MolFromSmiles(smarts)
    if mol is None: raise ValueError(f"Stringa chimica non valida: {smarts}")

    num_nodes = mol.GetNumAtoms()
    adj = [[] for _ in range(num_nodes)]
    node_labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    edge_labels_map = {}

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Riconoscimento tipi di legame (1=SINGOLO, 2=DOPPIO, 3=TRIPLO, 12=AROMATICO)
        label = int(bond.GetBondTypeAsDouble()) 
        adj[u].append(v)
        adj[v].append(u)
        edge_labels_map[(u, v)] = edge_labels_map[(v, u)] = label

    row_offsets = [0]
    column_indices = []
    edge_labels = []
    for j in range(num_nodes):
        for n in sorted(adj[j]):
            column_indices.append(n)
            edge_labels.append(edge_labels_map[(j, n)])
        row_offsets.append(len(column_indices))

    return make_csr_graph(row_offsets, column_indices, node_labels, edge_labels, num_nodes, smarts)

def toy_two_node_graph():
    """Grafo di test semplice: C-C (etano)."""
    return make_csr_graph([0, 1, 2], [1, 0], [6, 6], [1, 1], 2, "ethane")

def smarts_to_csr(file_path):
    """Legge un file SMARTS e restituisce una lista di grafi CSR."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File non trovato: {file_path}")
    
    graphs = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # Gestisce il formato: "SMARTS Nome" o solo "SMARTS"
            parts = line.split(maxsplit=1)
            smarts = parts[0]
            name = parts[1] if len(parts) > 1 else f"ID:{i}"
            
            try:
                graph = smarts_to_csr_from_string(smarts)
                graph["name"] = name
                graphs.append(graph)
            except ValueError:
                # Ignora le righe non valide 
                continue
    return graphs