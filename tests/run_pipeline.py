import dpctl
import os
import time
from rdkit import Chem
from sigmo import Signature, Candidates, filter_candidates

def smarts_to_csr(filepath):
    """Converte SMARTS/SMILES in strutture CSR per la GPU."""
    graphs = []
    if not os.path.exists(filepath): return graphs
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        smarts = line.strip()
        if not smarts: continue
        mol = Chem.MolFromSmarts(smarts)
        if mol is None: mol = Chem.MolFromSmiles(smarts)
        if mol is None: continue
        num_nodes = mol.GetNumAtoms()
        adj = [[] for _ in range(num_nodes)]
        node_labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        edge_labels_map = {}
        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            label = int(bond.GetBondTypeAsDouble())
            adj[u].append(v); adj[v].append(u)
            edge_labels_map[(u, v)] = label; edge_labels_map[(v, u)] = label
        row_offsets = [0]; column_indices = []; edge_labels = []
        for i in range(num_nodes):
            neighbors = sorted(adj[i])
            for n in neighbors:
                column_indices.append(n)
                edge_labels.append(edge_labels_map[(i, n)])
            row_offsets.append(len(column_indices))
        graphs.append({
            "row_offsets": row_offsets, "column_indices": column_indices,
            "node_labels": node_labels, "edge_labels": edge_labels, "num_nodes": num_nodes
        })
    return graphs

def run_sigmo_pipeline():
    q = dpctl.SyclQueue("gpu")

    base_dir = "benchmarks/datasets"
    print(f"[*] Dispositivo in uso: {q.sycl_device.name}")

    query_graphs = smarts_to_csr(os.path.join(base_dir, "query.smarts"))
    data_graphs = smarts_to_csr(os.path.join(base_dir, "data.smarts"))

    max_q_nodes = sum(g['num_nodes'] for g in query_graphs)
    max_d_nodes = sum(g['num_nodes'] for g in data_graphs)

    sig = Signature(q, max_d_nodes, max_q_nodes)
    cand = Candidates(q, max_q_nodes, max_d_nodes)

    print(f"[*] Esecuzione Filtro Iniziale su {len(data_graphs)} target...")
    t0 = time.perf_counter()
    results = filter_candidates(q, query_graphs, data_graphs, sig, cand)
    t1 = time.perf_counter()

    # Calcolo statistiche base
    total_space = max_q_nodes * max_d_nodes
    initial_cand = results['candidates_count']
    
    print("\n" + "="*50)
    print(f"{'REPORT SIGMO':^50}")
    print("="*50)
    print(f"Tempo Kernel:      {(t1 - t0)*1000:.2f} ms")
    print(f"Candidati:         {initial_cand:,}")
    print(f"Selettività:       {(initial_cand/total_space)*100:.2f}%")
    print(f"Memoria Occupata:  {results['allocated_bytes'] / 1024 / 1024:.2f} MB")
    print("-" * 50)
    print("="*50)
    print("[*] Pipeline completata con successo.")

if __name__ == "__main__":
    run_sigmo_pipeline()