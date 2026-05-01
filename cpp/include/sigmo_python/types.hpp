#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @namespace sigmo_python
 * @brief Namespace che raccoglie i tipi e le API pubbliche del binding Python per SIGMo.
 *
 * In questo file sono definiti i tipi dati semplici usati come interfaccia
 * tra il lato Python e il lato C++ del binding.
 */
namespace sigmo_python
{

    /**
     * @brief Rappresentazione host-side di un grafo nel formato CSR.
     *
     * Questa struct è usata come formato intermedio tra Python e C++:
     * il lato Python costruisce un grafo come dizionario/lista, il binding
     * lo converte in HostCSRGraphInput e successivamente in `sigmo::CSRGraph`.
     *
     * I campi seguono la rappresentazione CSR (Compressed Sparse Row):
     * - @p row_offsets contiene gli offset di inizio/fine della lista di adiacenza di ciascun nodo
     * - @p column_indices contiene la lista concatenata dei vicini
     * - @p node_labels contiene le etichette dei nodi
     * - @p edge_labels contiene le etichette degli archi
     * - @p num_nodes rappresenta il numero totale di nodi del grafo
     */
    struct HostCSRGraphInput
    {
        /// Offset CSR che delimitano l'inizio e la fine dei vicini di ciascun nodo.
        std::vector<uint32_t> row_offsets;

        /// Lista concatenata degli indici dei nodi adiacenti.
        std::vector<uint32_t> column_indices;

        /// Etichette associate ai nodi del grafo.
        std::vector<uint8_t> node_labels;

        /// Etichette associate agli archi del grafo.
        std::vector<uint8_t> edge_labels;

        /// Numero totale di nodi del grafo.
        std::size_t num_nodes;
    };

    /**
     * @brief Statistiche restituite dal binding dopo aver processato un batch di grafi.
     *
     * Questa struct è usata come valore di ritorno semplice e Python-friendly:
     * invece di esporre direttamente le strutture interne di SIGMo, il binding
     * restituisce solo informazioni aggregate utili per verificare il corretto
     * funzionamento del batch e l'uso di memoria.
     */
    struct GraphBatchStats
    {
        /// Numero di grafi contenuti nel batch.
        std::uint32_t num_graphs;

        /// Numero totale di nodi nel batch.
        std::size_t total_nodes;

        /// Numero totale di archi nel batch.
        std::size_t total_edges;

        /// Quantità di memoria allocata sul device per la struttura processata.
        std::size_t allocated_bytes;
    };

    /**
     * @brief Statistiche e metadati risultanti dall'operazione di filtraggio dei candidati.
     *
     * Questa struttura aggrega le informazioni raccolte durante l'esecuzione della pipeline
     * di filtraggio su GPU, includendo le dimensioni del dataset processato, il conteggio
     * totale dei candidati identificati e l'impatto sulla memoria USM (Unified Shared Memory).
     */
    struct FilterCandidatesStats
    {
        /// Numero totale di grafi contenuti nel dataset di query.
        std::uint32_t num_query_graphs;

        /// Numero totale di grafi contenuti nel dataset target (data).
        std::uint32_t num_data_graphs;

        /// Somma totale dei nodi presenti in tutti i grafi query processati.
        std::size_t total_query_nodes;

        /// Somma totale dei nodi presenti in tutti i grafi target (data) processati.
        std::size_t total_data_nodes;

        /// Rappresenta il numero totale di bit impostati a 1 nella matrice dei candidati dopo l'esecuzione del kernel di filtraggio.
        std::size_t total_candidates;

        /// Indica la quantità di memoria USM utilizzata per memorizzare le matrici dei candidati e le strutture dati ausiliarie durante il processo.
        std::size_t allocated_bytes;
    };

    /**
     * @brief Statistiche e metadati risultanti dall'operazione di refine dei candidati.
     *
     * Questa struttura aggrega le informazioni raccolte durante l'esecuzione della pipeline
     * di refine su GPU, includendo le dimensioni del dataset processato, il conteggio
     * totale dei candidati identificati e l'impatto sulla memoria USM (Unified Shared Memory).
     */
    struct RefineCandidatesStats
    {
        std::uint32_t num_query_graphs;

        std::uint32_t num_data_graphs;

        std::size_t total_query_nodes;

        std::size_t total_data_nodes;

        std::size_t total_candidates;

        std::size_t allocated_bytes;
    };

    struct JoinCandidatesStats
    {
        std::size_t num_matches;

        double execution_time;

        std::uint32_t total_query_graph;

        std::uint32_t total_data_graph;

        std::unordered_map<uint32_t, std::vector<uint32_t>> matches_dict;
    };

} // namespace sigmo_python