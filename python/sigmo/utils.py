def format_matches(raw_results, q_graphs, d_graphs):
    """
    Trasforma i risultati grezzi del C++ in un formato Pythonico.
    Esempio: associa gli indici dei match ai nomi delle molecole.
    """
    # Se il matcher ha restituito un errore o 0 match
    if raw_results.get("num_matches", 0) == 0:
        return {"total_matches": 0, "details": []}

    formatted = {
        "total_matches": raw_results["num_matches"],
        "results": raw_results.get("matches", [])
    }
    return formatted