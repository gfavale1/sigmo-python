"""
Esempio avanzato: esecuzione step-by-step della pipeline SIGMo.

Questo script mostra esplicitamente le fasi della pipeline a livello di kernel,
ma usa la nuova interfaccia Python:

- sigmo.load_molecules() per caricare SMARTS/SMILES senza duplicare parsing RDKit;
- sigmo.PipelineContext per gestire Signature, Candidates, GMCR e queue SYCL;
- sigmo.result.build_match_result() per ottenere un MatchResult spiegabile;
- summary(), explain(), to_csv(), to_json() per output user-friendly.

Esecuzione consigliata dalla root del progetto:

    PYTHONPATH=python python examples/run_pipeline.py

Esempio con opzioni:

    PYTHONPATH=python python examples/run_pipeline.py \
        --base-dir benchmarks/datasets \
        --query-file query.smarts \
        --data-file data.smarts \
        --query-limit 5 \
        --data-limit 20 \
        --iterations 6 \
        --device auto \
        --validate-with-rdkit \
        --csv matches.csv \
        --json matches.json
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import sigmo
from sigmo.result import build_match_result

try:
    from sigmo.validation import validate_result_with_rdkit
except Exception:  # pragma: no cover - fallback difensivo
    validate_result_with_rdkit = None

import csv
import json

def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def print_stats(title: str, stats: Optional[Dict[str, Any]]) -> None:
    """
    Stampa le statistiche restituite dai kernel.

    Evita di stampare strutture potenzialmente enormi come matches_dict,
    perché su dataset grandi possono produrre output giganteschi e rendere
    instabile l'esecuzione.
    """
    print(f"  [KERNEL] {title}:")

    if not stats:
        print("    - nessuna statistica disponibile")
        return

    hidden_keys = {"matches_dict", "matches", "raw_matches"}

    for key, value in stats.items():
        if key in hidden_keys:
            if isinstance(value, dict):
                total_pairs = sum(len(v) for v in value.values())
                print(f"    - {key}: <nascosto: {total_pairs:,} match>")
            elif isinstance(value, list):
                print(f"    - {key}: <nascosto: {len(value):,} elementi>")
            else:
                print(f"    - {key}: <nascosto>")
            continue

        print(f"    - {key}: {_format_value(value)}")


def print_graph_overview(label: str, graphs: List[Dict[str, Any]]) -> None:
    """Stampa informazioni sintetiche sui grafi caricati."""
    total_nodes = sum(int(g.get("num_nodes", 0)) for g in graphs)
    total_edges_directed = sum(len(g.get("column_indices", [])) for g in graphs)

    print(f"[*] {label}:")
    print(f"    - grafi: {len(graphs):,}")
    print(f"    - nodi totali: {total_nodes:,}")
    print(f"    - archi direzionati CSR: {total_edges_directed:,}")

    if graphs:
        preview = ", ".join(g.get("name", f"graph_{i}") for i, g in enumerate(graphs[:3]))
        if len(graphs) > 3:
            preview += ", ..."
        print(f"    - preview: {preview}")


def print_matches(
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    max_matches: int = 30,
) -> None:
    """
    Stampa solo una preview dei match trovati.

    I risultati completi devono essere esportati in CSV/JSON.
    """
    matches_dict = raw_join.get("matches_dict", {}) or {}

    total_pairs = sum(len(v) for v in matches_dict.values())

    print("\n[DETTAGLIO MATCH FOUND]")
    print(f"  Match totali nel dizionario: {total_pairs:,}")
    print(f"  Mostro al massimo: {max_matches:,}")

    if total_pairs == 0:
        print("  Nessun match trovato.")
        return

    printed = 0

    for q_idx, d_indices in matches_dict.items():
        for d_idx in d_indices:
            if printed >= max_matches:
                remaining = total_pairs - printed
                print(f"  ... output troncato. Altri match non stampati: {remaining:,}")
                print("  Usa --csv oppure --json per esportare i risultati completi.")
                return

            q_name = (
                query_graphs[q_idx].get("name", f"Q-{q_idx}")
                if q_idx < len(query_graphs)
                else f"Q-{q_idx}"
            )
            d_name = (
                data_graphs[d_idx].get("name", f"D-{d_idx}")
                if d_idx < len(data_graphs)
                else f"D-{d_idx}"
            )

            q_input = (
                query_graphs[q_idx].get("input", "")
                if q_idx < len(query_graphs)
                else ""
            )
            d_input = (
                data_graphs[d_idx].get("input", "")
                if d_idx < len(data_graphs)
                else ""
            )

            print(f"  - [{q_idx}] {q_name} MATCH con [{d_idx}] {d_name}")

            if q_input:
                print(f"      query: {q_input}")

            if d_input:
                preview = d_input[:140] + "..." if len(d_input) > 140 else d_input
                print(f"      data:  {preview}")

            printed += 1


def _step_stats(ctx: sigmo.PipelineContext, step_name: str) -> Optional[Dict[str, Any]]:
    """
    Recupera le statistiche associate all'ultimo KernelStep con un certo nome.

    La struttura KernelStep della nuova interfaccia espone almeno name e stats.
    Questa funzione usa getattr per restare robusta se il dataclass cambia leggermente.
    """
    for step in reversed(getattr(ctx, "steps", [])):
        if getattr(step, "name", None) == step_name:
            return getattr(step, "stats", None)
    return None


def load_graphs(path: Path, *, input_format: str, limit: Optional[int]) -> List[Dict[str, Any]]:
    graphs = sigmo.load_molecules(str(path), input_format=input_format)
    if limit is not None:
        graphs = graphs[:limit]
    return graphs

def iter_match_records(
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
):
    """
    Genera i match uno alla volta, senza materializzare una lista enorme.

    Questa funzione è pensata per dataset grandi.
    """
    matches_dict = raw_join.get("matches_dict", {}) or {}

    for q_idx, d_indices in matches_dict.items():
        for d_idx in d_indices:
            q_graph = query_graphs[q_idx] if q_idx < len(query_graphs) else {}
            d_graph = data_graphs[d_idx] if d_idx < len(data_graphs) else {}

            yield {
                "query_index": q_idx,
                "query_name": q_graph.get("name", f"query_{q_idx}"),
                "query_input": q_graph.get("input", ""),
                "data_index": d_idx,
                "data_name": d_graph.get("name", f"data_{d_idx}"),
                "data_input": d_graph.get("input", ""),
            }


def write_matches_csv_streaming(
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    path: Path,
) -> int:
    """
    Scrive i match in CSV in modalità streaming.

    Non crea una lista Python con tutti i match, quindi è adatto anche
    a milioni di risultati.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "query_index",
        "query_name",
        "query_input",
        "data_index",
        "data_name",
        "data_input",
    ]

    count = 0

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for record in iter_match_records(raw_join, query_graphs, data_graphs):
            writer.writerow(record)
            count += 1

            if count % 1_000_000 == 0:
                print(f"[*] CSV streaming: scritti {count:,} match...", flush=True)

    return count


def write_summary_json(
    *,
    raw_join: Dict[str, Any],
    query_graphs: List[Dict[str, Any]],
    data_graphs: List[Dict[str, Any]],
    ctx,
    result,
    path: Path,
) -> None:
    """
    Scrive un JSON leggero con metadati e statistiche.

    Non salva tutti i match, perché su dataset grandi il JSON completo
    può saturare la memoria.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "OK" if not result.errors else "ERROR",
        "device": result.device,
        "query_count": len(query_graphs),
        "data_count": len(data_graphs),
        "total_matches": raw_join.get("num_matches", 0),
        "requested_iterations": result.requested_iterations,
        "executed_iterations": result.executed_iterations,
        "warnings": result.warnings,
        "errors": result.errors,
        "kernel_steps": [
            {
                "name": getattr(step, "name", None),
                "duration_seconds": getattr(step, "duration_seconds", getattr(step, "elapsed_seconds", None)),
                "stats": {
                    k: v
                    for k, v in getattr(step, "stats", {}).items()
                    if k not in {"matches_dict", "matches", "raw_matches"}
                },
            }
            for step in getattr(ctx, "steps", [])
        ],
        "note": (
            "Questo JSON contiene solo metadati e statistiche. "
            "I match completi sono esportati nel CSV streaming."
        ),
    }

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
def run_kernel_pipeline(
    *,
    query_path: Path,
    data_path: Path,
    input_format: str = "auto",
    device: str = "auto",
    iterations: int = 6,
    find_first: bool = True,
    query_limit: Optional[int] = 5,
    data_limit: Optional[int] = 20,
    validate_with_rdkit: bool = False,
    csv_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
    force_refine: bool = False,
    max_print_matches: int = 30,
    large_result_threshold: int = 1_000_000,
) -> sigmo.MatchResult:
    print("[*] SIGMo kernel-level pipeline")
    print(f"[*] Query file: {query_path}")
    print(f"[*] Data file:  {data_path}")

    query_graphs = load_graphs(query_path, input_format=input_format, limit=query_limit)
    data_graphs = load_graphs(data_path, input_format=input_format, limit=data_limit)

    if not query_graphs:
        raise ValueError(f"Nessun grafo query valido caricato da: {query_path}")
    if not data_graphs:
        raise ValueError(f"Nessun grafo data valido caricato da: {data_path}")

    print_graph_overview("Query graphs", query_graphs)
    print_graph_overview("Data graphs", data_graphs)

    small_query_graphs = [
        (i, g.get("name", f"query_{i}"), g.get("num_nodes", 0))
        for i, g in enumerate(query_graphs)
        if g.get("num_nodes", 0) < 6
    ]

    small_data_graphs = [
        (i, g.get("name", f"data_{i}"), g.get("num_nodes", 0))
        for i, g in enumerate(data_graphs)
        if g.get("num_nodes", 0) < 6
    ]

    if small_query_graphs or small_data_graphs:
        print("\n[WARNING] Rilevati grafi piccoli potenzialmente instabili per il refinement:")
        for idx, name, nodes in small_query_graphs[:10]:
            print(f"  - query[{idx}] {name}: {nodes} nodi")
        for idx, name, nodes in small_data_graphs[:10]:
            print(f"  - data[{idx}] {name}: {nodes} nodi")

    ctx = sigmo.PipelineContext(
        query_graphs=query_graphs,
        data_graphs=data_graphs,
        device=device,
    )

    print(f"\n[*] Device: {ctx.device_name}")

    print("\n[*] Allocazione memoria e strutture SIGMo")
    ctx.allocate()
    print("  - Signature, Candidates e GMCR allocati correttamente")

    print("\n[*] 1/4 - Generazione firme iniziali")
    ctx.generate_signatures()
    print_stats("Generate Query Signatures", _step_stats(ctx, "generate_query_signatures"))
    print_stats("Generate Data Signatures", _step_stats(ctx, "generate_data_signatures"))

    print("\n[*] 2/4 - Filtraggio candidati")
    filter_stats = ctx.filter_candidates()
    print_stats("Initial Filter", filter_stats)

    print("\n[*] 3/4 - Raffinamento iterativo")

    min_query_nodes = min(g.get("num_nodes", 0) for g in query_graphs)
    min_data_nodes = min(g.get("num_nodes", 0) for g in data_graphs)
    min_nodes = min(min_query_nodes, min_data_nodes)

    safe_iterations = iterations

    if iterations > 0 and min_nodes < 6 and not force_refine:
        warning_msg = (
            "Raffinamento disabilitato per stabilità: "
            f"trovato almeno un grafo con {min_nodes} nodi (< 6). "
            "La pipeline prosegue con filter + join, evitando il kernel di refine "
            "che può essere instabile su micro-molecole. "
            "Per forzare comunque l'esecuzione usa --force-refine."
        )

        print(f"  [WARNING] {warning_msg}")
        ctx.warnings.append(warning_msg)

        safe_iterations = 0

    elif iterations > 0 and min_nodes < 6 and force_refine:
        warning_msg = (
            "Force refine attivo: il refinement verra' eseguito anche se sono presenti "
            f"grafi piccoli. Nodo minimo rilevato: {min_nodes}. "
            "Questa modalità può causare segmentation fault lato C++/SYCL."
        )

        print(f"  [WARNING] {warning_msg}")
        ctx.warnings.append(warning_msg)

        safe_iterations = iterations

    if safe_iterations <= 0:
        print("  - Raffinamento disabilitato.")
        executed_iterations = 0
    else:
        print(
            f"  - Eseguo refinement per massimo {safe_iterations} iterazione/i "
            f"(force_refine={force_refine})"
        )

        before_steps = len(ctx.steps)

        # Compatibile sia con refine(iterations=...) sia con vecchie versioni refine(max_iterations=...)
        try:
            refine_stats = ctx.refine(
                safe_iterations,
                start_view_size=1,
                stop_on_fixed_point=True,
            )
        except TypeError:
            refine_stats = ctx.refine(max_iterations=safe_iterations)

        executed_iterations = getattr(ctx, "executed_iterations", None)

        if executed_iterations is None:
            new_steps = ctx.steps[before_steps:]
            executed_iterations = sum(
                1 for step in new_steps
                if getattr(step, "name", None) == "refine_candidates"
            )

        if isinstance(refine_stats, dict):
            print_stats("Refinement", refine_stats)
        else:
            print(f"  - Iterazioni eseguite: {executed_iterations}")

    print("\n[*] 4/4 - Join finale / isomorfismo")
    raw_join = ctx.join(find_first=find_first)
    print_stats("Join", raw_join)

    num_matches = int(raw_join.get("num_matches", 0))
    large_result = num_matches > large_result_threshold

    if max_print_matches > 0:
        print_matches(
            raw_join,
            query_graphs,
            data_graphs,
            max_matches=max_print_matches,
        )
    else:
        print("\n[DETTAGLIO MATCH FOUND]")
        print("  Stampa dettagli disabilitata: --max-print-matches 0")

    if large_result:
        large_warning = (
            f"Risultato molto grande: {num_matches:,} match. "
            "Per evitare consumo eccessivo di memoria, i match completi non vengono "
            "materializzati dentro MatchResult. Usa il CSV streaming per esportarli."
        )

        print(f"\n[WARNING] {large_warning}")
        ctx.warnings.append(large_warning)

        raw_join_for_result = dict(raw_join)
        raw_join_for_result["matches_dict"] = {}

        result = build_match_result(
            raw_join_for_result,
            query_graphs,
            data_graphs,
            steps=ctx.steps,
            warnings=ctx.warnings,
            errors=ctx.errors,
            device=ctx.device_name,
            requested_iterations=iterations,
            executed_iterations=executed_iterations,
        )

        # Manteniamo il numero reale di match anche se non materializziamo la lista.
        result.total_matches = num_matches

    else:
        result = build_match_result(
            raw_join,
            query_graphs,
            data_graphs,
            steps=ctx.steps,
            warnings=ctx.warnings,
            errors=ctx.errors,
            device=ctx.device_name,
            requested_iterations=iterations,
            executed_iterations=executed_iterations,
        )

    if validate_with_rdkit:
        if large_result:
            validation_warning = (
                "Validazione RDKit saltata: il risultato è troppo grande per una "
                "validazione completa coppia-per-coppia in questa modalità."
            )
            print(f"[WARNING] {validation_warning}")
            result.warnings.append(validation_warning)

        elif validate_result_with_rdkit is None:
            result.warnings.append(
                "Validazione RDKit richiesta, ma sigmo.validation non e' disponibile."
            )

        else:
            result = validate_result_with_rdkit(result, query_graphs, data_graphs)

    print("\n" + result.summary())
    print("\n" + result.explain())

    if csv_path is not None:
        if large_result:
            print(f"\n[*] Esporto CSV in streaming: {csv_path}", flush=True)
            written = write_matches_csv_streaming(
                raw_join,
                query_graphs,
                data_graphs,
                csv_path,
            )
            print(f"[*] CSV esportato: {written:,} match scritti.", flush=True)
        else:
            result.to_csv(csv_path)
            print(f"\n[*] CSV esportato in: {csv_path}")

    if json_path is not None:
        if large_result:
            print(f"[*] Esporto JSON summary: {json_path}", flush=True)
            write_summary_json(
                raw_join=raw_join,
                query_graphs=query_graphs,
                data_graphs=data_graphs,
                ctx=ctx,
                result=result,
                path=json_path,
            )
            print(f"[*] JSON summary esportato in: {json_path}", flush=True)
        else:
            result.to_json(json_path)
            print(f"[*] JSON esportato in: {json_path}")

    print("\n[*] Pipeline terminata.")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Esegue la pipeline SIGMo step-by-step usando la nuova interfaccia Python."
    )

    parser.add_argument("--base-dir", default="benchmarks/datasets", help="Cartella contenente query/data file.")
    parser.add_argument("--query-file", default="query.smarts", help="File query SMARTS/SMILES.")
    parser.add_argument("--data-file", default="data.smarts", help="File database SMARTS/SMILES.")
    parser.add_argument("--input-format", default="auto", choices=["auto", "smiles", "smarts"], help="Formato input.")
    parser.add_argument("--device", default="auto", help="Device SYCL: auto, gpu, cpu o filtro dpctl.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=0,
        help=(
            "Numero massimo di iterazioni di refinement. "
            "Default: 0 per evitare crash del backend su dataset con molecole piccole. "
            "Usa valori > 0 solo dopo aver verificato la stabilità del refinement."
        ),
    )    
    parser.add_argument(
        "--find-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ferma il join al primo match per coppia, se supportato.",
    )
    parser.add_argument("--query-limit", type=int, default=5, help="Limite massimo di query da caricare. Usa -1 per nessun limite.")
    parser.add_argument("--data-limit", type=int, default=20, help="Limite massimo di grafi data da caricare. Usa -1 per nessun limite.")
    parser.add_argument("--validate-with-rdkit", action="store_true", help="Valida il risultato confrontandolo con RDKit.")
    parser.add_argument("--csv", default=None, help="Percorso CSV opzionale per esportare i match.")
    parser.add_argument("--json", default=None, help="Percorso JSON opzionale per esportare il risultato completo.")
    parser.add_argument(
        "--force-refine",
        action="store_true",
        help=(
            "Forza l'esecuzione del refinement anche se sono presenti grafi piccoli. "
            "Modalità sperimentale: può causare segmentation fault se il backend SIGMo "
            "non gestisce correttamente alcune micro-molecole."
        ),
    )
    parser.add_argument(
        "--max-print-matches",
        type=int,
        default=30,
        help="Numero massimo di match da stampare a terminale. Usa 0 per non stampare dettagli.",
    )
    return parser.parse_args()


def _normalize_limit(value: int) -> Optional[int]:
    return None if value is None or value < 0 else value


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir)
    query_path = base_dir / args.query_file
    data_path = base_dir / args.data_file

    run_kernel_pipeline(
        query_path=query_path,
        data_path=data_path,
        input_format=args.input_format,
        device=args.device,
        iterations=args.iterations,
        find_first=args.find_first,
        query_limit=_normalize_limit(args.query_limit),
        data_limit=_normalize_limit(args.data_limit),
        validate_with_rdkit=args.validate_with_rdkit,
        csv_path=Path(args.csv) if args.csv else None,
        json_path=Path(args.json) if args.json else None,
        force_refine=args.force_refine,
        max_print_matches=args.max_print_matches,
    )


if __name__ == "__main__":
    main()