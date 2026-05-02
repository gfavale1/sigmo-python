from __future__ import annotations

from typing import Any, Dict, List, Tuple

from rdkit import Chem

from .result import MatchResult


def validate_result_with_rdkit(
    result: MatchResult,
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
) -> MatchResult:
    """
    Valida un MatchResult confrontando SIGMo con RDKit.

    La validazione confronta, per ogni coppia query-data, se SIGMo e RDKit
    concordano sull'esistenza di un match di sottostruttura.

    Il risultato viene aggiunto direttamente in result.validation.
    """
    sigmo_pairs = {
        (match.query_index, match.data_index)
        for match in result.matches
    }

    checked_pairs = 0
    agreements = 0
    disagreements = []
    skipped = []

    for q_idx, q_graph in enumerate(query_graphs):
        for d_idx, d_graph in enumerate(data_graphs):
            rdkit_ok, reason = _rdkit_has_substructure_match(q_graph, d_graph)

            if reason is not None:
                skipped.append({
                    "query_index": q_idx,
                    "data_index": d_idx,
                    "reason": reason,
                })
                continue

            checked_pairs += 1

            sigmo_ok = (q_idx, d_idx) in sigmo_pairs

            if sigmo_ok == rdkit_ok:
                agreements += 1
            else:
                disagreements.append({
                    "query_index": q_idx,
                    "data_index": d_idx,
                    "query_name": q_graph.get("name", f"query_{q_idx}"),
                    "data_name": d_graph.get("name", f"data_{d_idx}"),
                    "query_input": q_graph.get("input"),
                    "data_input": d_graph.get("input"),
                    "sigmo": sigmo_ok,
                    "rdkit": rdkit_ok,
                })

    result.validation = {
        "enabled": True,
        "method": "RDKit HasSubstructMatch",
        "checked_pairs": checked_pairs,
        "agreements": agreements,
        "disagreements": disagreements,
        "skipped": skipped,
        "passed": len(disagreements) == 0 and checked_pairs > 0,
    }

    if disagreements:
        result.warnings.append(
            f"Validazione RDKit: trovate {len(disagreements)} divergenze tra SIGMo e RDKit."
        )
    elif checked_pairs > 0:
        result.warnings.append(
            f"Validazione RDKit completata: {agreements}/{checked_pairs} coppie concordano."
        )
    else:
        result.warnings.append(
            "Validazione RDKit non eseguita: nessuna coppia validabile."
        )

    return result


def _rdkit_has_substructure_match(
    query_graph: Dict[str, Any],
    data_graph: Dict[str, Any],
) -> Tuple[bool, str | None]:
    query_input = query_graph.get("input")
    data_input = data_graph.get("input")

    if not query_input:
        return False, "query graph senza campo 'input'"
    if not data_input:
        return False, "data graph senza campo 'input'"

    query_mol = _parse_query_mol(str(query_input))
    data_mol = _parse_data_mol(str(data_input))

    if query_mol is None:
        return False, f"query non interpretabile da RDKit: {query_input}"
    if data_mol is None:
        return False, f"data molecule non interpretabile da RDKit: {data_input}"

    return bool(data_mol.HasSubstructMatch(query_mol)), None


def _parse_query_mol(value: str):
    """
    Per la query conviene provare prima SMARTS, perché una query di
    sottostruttura può contenere espressioni SMARTS non valide come SMILES.
    """
    mol = Chem.MolFromSmarts(value)
    if mol is not None:
        return mol
    return Chem.MolFromSmiles(value)


def _parse_data_mol(value: str):
    """
    Per il target conviene provare prima SMILES, perché rappresenta una
    molecola concreta. Se fallisce, proviamo SMARTS come fallback.
    """
    mol = Chem.MolFromSmiles(value)
    if mol is not None:
        return mol
    return Chem.MolFromSmarts(value)