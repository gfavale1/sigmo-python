import os
import time
import dpctl
import sigmo
import warnings
from rdkit import Chem

from sigmo import (
    Signature,
    Candidates,
    GMCR,
    generate_csr_signatures,
    refine_csr_signatures,
    filter_candidates,
    refine_candidates,
    join_candidates,
)

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*dpctl.tensor is deprecated*")

def smarts_to_csr(filepath):
    graphs = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File non trovato: {filepath}")

    with open(filepath, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines): # Aggiunto enumerate per avere l'indice
        parts = line.strip().split()
        if not parts:
            continue

        smarts = parts[0]
        
        # MODIFICATO: Se non c'è un nome, usiamo l'indice + un'anteprima della molecola
        if len(parts) > 1:
            mol_name = parts[1]
        else:
            # Prende i primi 10 caratteri della stringa chimica per riconoscerla
            preview = smarts[:10] + "..." if len(smarts) > 10 else smarts
            # Esempio: "ID:0 (Clc1cc2[nH+...)"
            mol_name = f"ID:{i} ({preview})"

        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            mol = Chem.MolFromSmiles(smarts)
        if mol is None:
            continue

        num_nodes = mol.GetNumAtoms()
        adj = [[] for _ in range(num_nodes)]
        node_labels = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        edge_labels_map = {}

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            label = int(bond.GetBondTypeAsDouble())

            adj[u].append(v)
            adj[v].append(u)

            edge_labels_map[(u, v)] = label
            edge_labels_map[(v, u)] = label

        row_offsets = [0]
        column_indices = []
        edge_labels = []

        for i in range(num_nodes):
            for n in sorted(adj[i]):
                column_indices.append(n)
                edge_labels.append(edge_labels_map[(i, n)])
            row_offsets.append(len(column_indices))

        graphs.append({
            "row_offsets": row_offsets,
            "column_indices": column_indices,
            "node_labels": node_labels,
            "edge_labels": edge_labels,
            "num_nodes": num_nodes,
            "name": mol_name, # AGGIUNTO: Ora il nome fa parte del dizionario del grafo
        })

    return graphs


def print_stats(title, stats):
    """Utility per stampare i log dei kernel in modo pulito"""
    print(f"  [KERNEL] {title}:")
    for key, value in stats.items():
        if isinstance(value, int):
            print(f"    - {key}: {value:,}")
        else:
            print(f"    - {key}: {value}")

def run_pipeline():
    q = dpctl.SyclQueue("gpu")
    print(f"[*] Device: {q.sycl_device.name}")

    test_graph = [{
        "num_nodes": 2,
        "row_offsets": [0, 1, 2],
        "column_indices": [1, 0],
        "node_labels": [1, 1],
        "edge_labels": [1, 1]
    }]
    
    # sig_test = Signature(q, 2, 2)
    # stats = generate_csr_signatures(q, test_graph, sig_test, "query")
    # print(f"[*] Signature popolata. Size: {stats['allocated_bytes']}")
    # print("TEST MINIMO PASSATO!")

    base_dir = "benchmarks/datasets"
    query_graphs = smarts_to_csr(os.path.join(base_dir, "query.smarts"))
    data_graphs = smarts_to_csr(os.path.join(base_dir, "data.smarts"))
    query_names = [g["name"] for g in query_graphs]
    data_names = [g["name"] for g in data_graphs]

    limit = 10000 
    data_graphs = data_graphs[:500]
    query_graphs = query_graphs[:100]

    # Opzionale: limita anche le query se sono tante
    # query_graphs = query_graphs[:limit]

    total_q_nodes = sum(g["num_nodes"] for g in query_graphs)
    total_d_nodes = sum(g["num_nodes"] for g in data_graphs)

    print(f"[*] Graphs Loaded -> Query: {len(query_graphs)} | Data: {len(data_graphs)}")
    print(f"[*] Total Nodes   -> Query: {total_q_nodes:,} | Data: {total_d_nodes:,}")

    print("\n[*] Allocazione Memoria USM su Device...")
    sig = Signature(q, total_d_nodes, total_q_nodes)
    cand = Candidates(q, total_q_nodes, total_d_nodes)
    q.wait() 

    print("\n[*] 1/4 - Generazione Firme Iniziali")
    
    # Nota: L'ordine deve essere (q, graphs, sig, scope)
    try:
        q_gen_stats = generate_csr_signatures(q, query_graphs, sig, "query")
        q.wait()
        print_stats("Generate Query Sigs", q_gen_stats)

        d_gen_stats = generate_csr_signatures(q, data_graphs, sig, "data")
        q.wait()
        print_stats("Generate Data Sigs", d_gen_stats)
    except RuntimeError as e:
        print(f"!!! CRASH in Generation: {e}")
        return


    # Verifica manuale delle etichette
    query_labels = set()
    for g in query_graphs:
        query_labels.update(g['node_labels'])

    data_labels = set()
    for g in data_graphs[:100]: # controlla solo i primi 100 per velocità
        data_labels.update(g['node_labels'])

    #print(f"DEBUG: Etichette Query trovate: {query_labels}")
    #print(f"DEBUG: Etichette Data trovate: {data_labels}")
    #print(f"DEBUG: Intersezione: {query_labels.intersection(data_labels)}")

    print("\n[*] 2/4 - Filtraggio Candidati (Initial Filter)")
    try:
        f_stats = filter_candidates(q, query_graphs, data_graphs, sig, cand)
        q.wait()
        initial_count = f_stats.get("total_candidates", 0)
        print_stats("Initial Filter", f_stats)
    except RuntimeError as e:
        print(f"!!! CRASH in Filter: {e}")
        return

    print("\n[*] 3/4 - Ciclo di Raffinamento Iterativo")
    NUM_ITERATIONS = 6
    VIEW_SIZE = 1 # Raggio dell'intorno per le firme
    current_count = initial_count

    for i in range(NUM_ITERATIONS):
        print(f"\n  [Iterazione {i+1}/{NUM_ITERATIONS}]")
        t_start = time.time()

        try:
            var = VIEW_SIZE + i
            # A. Raffina Firme Query
            rq_stats = refine_csr_signatures(q, query_graphs, sig, "query", var)
            q.wait()

            # B. Raffina Firme Data
            rd_stats = refine_csr_signatures(q, data_graphs, sig, "data", var)
            q.wait()

            # C. Raffina Matrice Candidati
            rc_stats = refine_candidates(q, query_graphs, data_graphs, sig, cand)
            q.wait()

            new_count = rc_stats["candidates_count"]

            delta = current_count - new_count
            current_count = new_count

            print(f"    - Candidati superstiti: {current_count:,}")
            print(f"    - Riduzione (Delta):    {delta:,}")
            print(f"    - Tempo iterazione:     {time.time() - t_start:.2f}s")

            # Se non tolgo piu nulla esco 
            if delta == 0 and i > 0:
                print("    [!] Punto fisso raggiunto. Fermo il raffinamento.")
                break
                
        except Exception as e:
            print(f"    !!! ERRORE CRITICO nell'iterazione {i+1}: {e}")
            break

    gmcr = GMCR(q)
    
    print("\n[*] 4/4 - Fase di Join (Isomorfismo)")
    
    try:        
        j_stats = join_candidates(q, query_graphs, data_graphs, cand, gmcr, True)
        
        print(f"  - Match totali trovati: {j_stats['num_matches']:,}")
        print(f"  - Tempo Kernel:         {j_stats['execution_time']:.2f} ms")
        print(f"  - Grafi Query processati: {j_stats['num_query_graph']}")
        print(f"  - Grafi Data processati:  {j_stats['num_data_graph']}")

        matches = j_stats.get('matches_dict', {})

        print("\n[DETTAGLIO MATCH FOUND]:")

        # Iteriamo sugli items del dizionario: q_idx è la chiave, d_indices è la lista
        for q_idx, d_indices in matches.items():
            # Per ogni query, iteriamo su tutti i grafi data che hanno matchato
            for d_idx in d_indices:
                
                # Recupera i nomi usando gli indici (che ora sono semplici interi)
                q_name = query_names[q_idx] if q_idx < len(query_names) else f"Q-{q_idx}"
                d_name = data_names[d_idx] if d_idx < len(data_names) else f"D-{d_idx}"
                
                print(f"  - {q_name} MATCH con {d_name}")
                        
    except Exception as e:
        print(f"!!! ERRORE nella fase di Join: {e}")
        
    finally:
        print("\n[*] Pipeline terminata.")

if __name__ == "__main__":
    run_pipeline()