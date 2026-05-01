from . import _core
from .utils import format_matches
from .config import get_default_queue

# Espongo le funzioni principali all'utente
__all__ = ['run_isomorphism', 'match_smarts', 'filter_candidates', 'refine_candidates', 'join_candidates']

# Espongo 3 livelli di funzione, pensando alla user experience e scaricando le responsabilità dell'utente

# Top level --> prende direttamente stringhe chimiche e restituisce i risultati. 
# Prendo il numero di iterazioni per la refine e find first
def match(query_smarts, target_smarts, iterations=1):
    """
    Funzione 'Entry Point' consigliata per l'utente finale.
    """
    from .graph import smarts_to_csr_from_string
    q = get_default_queue()
    q_csr = [smarts_to_csr_from_string(query_smarts)]
    t_csr = [smarts_to_csr_from_string(target_smarts)]

    return run_isomorphism(q_graphs=q_csr, d_graphs=t_csr, queue=q, find_first=True, iterations=iterations)

# Middle level --> serve a chi ha gia i grafi pronti (tipo caricati da un database)
# e vuole gestire manualmente la coda SYCL
def run_isomorphism(q_graphs, d_graphs, queue=None, find_first=True, iterations=1):
    """Pipeline completa con protezione anti-crash per micro-molecole."""
    if queue is None:
        queue = get_default_queue()

    # 1. ANALISI DIMENSIONI PER STABILITÀ
    # Calcoliamo il numero minimo di nodi tra tutti i grafi coinvolti
    min_nodes_q = min((g["num_nodes"] for g in q_graphs), default=0)
    min_nodes_d = min((g["num_nodes"] for g in d_graphs), default=0)
    
    # SOGLIA DI SICUREZZA: Se una molecola ha meno di 6 atomi (es. Formaldeide, Etanolo)
    # il raffinamento del dottorando è instabile. Lo disattiviamo silenziosamente.
    if iterations > 0 and (min_nodes_q < 6 or min_nodes_d < 6):
        # Log di debug opzionale (puoi rimuoverlo per la release)
        # print(f"[SIGMo System] Warning: Micro-molecules detected. Refinement disabled for stability.")
        iterations = 0

    total_q = sum(g["num_nodes"] for g in q_graphs)
    total_d = sum(g["num_nodes"] for g in d_graphs)

    # 2. ALLOCAZIONE USM (con un piccolo padding extra di sicurezza)
    sig = _core.Signature(queue, total_d + 16, total_q + 16)
    cand = _core.Candidates(queue, total_q + 16, total_d + 16)
    gmcr = _core.GMCR(queue)
    queue.wait()

    # 3. GENERAZIONE FIRME
    _core.generate_csr_signatures(queue, q_graphs, sig, "query")
    _core.generate_csr_signatures(queue, d_graphs, sig, "data")
    queue.wait()

    # 4. FILTRO INIZIALE
    _core.filter_candidates(queue, q_graphs, d_graphs, sig, cand)
    queue.wait()

    # 5. RAFFINAMENTO (Eseguito solo se iterations è rimasto > 0)
    for i in range(iterations):
        _core.refine_csr_signatures(queue, q_graphs, sig, "query", 1 + i)
        _core.refine_csr_signatures(queue, d_graphs, sig, "data", 1 + i)
        _core.refine_candidates(queue, q_graphs, d_graphs, sig, cand)
        queue.wait()
        
        if cand.get_candidates_count(0) == 0:
            break

    # 6. JOIN FINALE
    try:
        return _core.join_candidates(queue, q_graphs, d_graphs, cand, gmcr, find_first)
    except Exception as e:
        return {"num_matches": 0, "error": str(e), "status": "failed_at_join"}