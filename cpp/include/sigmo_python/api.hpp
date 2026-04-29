#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

#include "sigmo_python/types.hpp"

#include "device.hpp"
#include "types.hpp"
#include "graph.hpp"
#include "signature.hpp"
#include "candidates.hpp"
#include "gmcr.hpp"
/**
 * @namespace sigmo_python
 * @brief Namespace che raccoglie l'interfaccia pubblica del binding Python per SIGMo.
 */
namespace sigmo_python
{

    /**
     * @brief Genera le CSR signatures per un batch di grafi.
     *
     * La funzione riceve in input un device SYCL, un insieme di grafi nel formato
     * host-side del binding e uno scope che specifica se generare le signatures
     * per il caso "data" oppure "query".
     *
     * Internamente il flusso previsto è:
     * 1 - conversione dei grafi in `sigmo::CSRGraph`
     * 2 - creazione del `DeviceBatchedCSRGraph`
     * 3 - generazione delle signatures CSR sul device
     * 4 - distruzione del batch device
     *
     * @param dev Device SYCL sul quale eseguire la computazione.
     * @param graphs Batch di grafi nel formato HostCSRGraphInput.
     * @param scope Scope delle signatures da generare: "data" oppure "query".
     *
     * @return Una struct GraphBatchStats contenente statistiche sul batch processato,
     *         come numero di grafi, numero totale di nodi, numero totale di archi
     *         e memoria allocata.
     *
     * @throws std::runtime_error Se i grafi in input non sono validi oppure
     *         se lo scope passato non è supportato.
     */
    GraphBatchStats generate_csr_signatures(
        sycl::queue &queue,
        const std::vector<HostCSRGraphInput> &graphs,
        sigmo::signature::Signature<> &signatures,
        const std::string &scope);

    /**
     * @brief Genera e raffina le CSR signatures per un batch di grafi.
     *
     * La funzione esegue prima la generazione delle signatures di base e poi
     * applica il refinement usando il parametro @p view_size.
     *
     * Lo scope determina se operare sulle signatures di tipo "data" oppure "query".
     * In questa fase del progetto il refinement di interesse è quello sul path CSR.
     *
     * @param dev Device SYCL sul quale eseguire la computazione.
     * @param graphs Batch di grafi nel formato HostCSRGraphInput.
     * @param scope Scope delle signatures da raffinare: "data" oppure "query".
     * @param view_size Parametro di refinement che controlla l'ampiezza della vista
     *                  usata nell'algoritmo di refine.
     *
     * @return Una struct GraphBatchStats contenente statistiche sul batch processato,
     *         come numero di grafi, numero totale di nodi, numero totale di archi
     *         e memoria allocata.
     *
     * @throws std::runtime_error Se i grafi in input non sono validi oppure
     *         se lo scope passato non è supportato.
     */
    GraphBatchStats refine_csr_signatures(
        sycl::queue &queue,
        const std::vector<HostCSRGraphInput> &graphs,
        sigmo::signature::Signature<> &signatures,
        const std::string &scope,
        std::size_t view_size);

    /**
     * @brief Esegue il filtro dei candidati seguendo la stessa struttura della
     *        `sigmo::isomorphism::filter::filterCandidates` originale.
     *
     * Il chiamante fornisce esplicitamente queue, signatures e struttura dei
     * candidati, mentre il wrapper si occupa di convertire i grafi host-side
     * del binding nei `DeviceBatchedCSRGraph` richiesti dalla libreria SIGMo.
     *
     * @param queue Queue SYCL usata per l'esecuzione.
     * @param query_graph Batch di grafi query nel formato HostCSRGraphInput.
     * @param data_graph Batch di grafi data nel formato HostCSRGraphInput.
     * @param signatures Oggetto SIGMo che contiene le firme query/data gia'
     *                   allocate dal chiamante.
     * @param candidates Oggetto SIGMo che contiene la struttura device dei
     *                   candidati gia' allocata dal chiamante.
     *
     * @return Statistiche aggregate sull'esecuzione del filtro.
     */
    FilterCandidatesStats filter_candidates(
        sycl::queue &queue,
        const std::vector<HostCSRGraphInput> &query_graph,
        const std::vector<HostCSRGraphInput> &data_graph,
        sigmo::signature::Signature<> &signatures,
        sigmo::candidates::Candidates &candidates);

    /**
     * @brief Esegue il refine dei candidati seguendo la stessa struttura della
     *        `sigmo::isomorphism::filter::refineCandidates` originale.
     *
     * Il chiamante fornisce esplicitamente queue, signatures e struttura dei
     * candidati, mentre il wrapper si occupa di convertire i grafi host-side
     * del binding nei `DeviceBatchedCSRGraph` richiesti dalla libreria SIGMo.
     *
     * @param queue Queue SYCL usata per l'esecuzione.
     * @param query_graph Batch di grafi query nel formato HostCSRGraphInput.
     * @param data_graph Batch di grafi data nel formato HostCSRGraphInput.
     * @param signatures Oggetto SIGMo che contiene le firme query/data gia'
     *                   allocate dal chiamante.
     * @param candidates Oggetto SIGMo che contiene la struttura device dei
     *                   candidati gia' allocata dal chiamante.
     *
     * @return Statistiche aggregate sull'esecuzione della refine.
     */

    RefineCandidatesStats refine_candidates(
        sycl::queue &queue,
        const std::vector<HostCSRGraphInput> &query_graph,
        const std::vector<HostCSRGraphInput> &data_graph,
        sigmo::signature::Signature<> &signatures,
        sigmo::candidates::Candidates &candidates);

    /**
     * @brief Esegue la fase di join per identificare gli isomorfismi reali tra i grafi.
     *
     * * Il wrapper gestisce il caricamento temporaneo dei grafi host-side sulla GPU 
     * (DeviceBatchedCSRGraph) e coordina la generazione della struttura GMCR 
     * (Global Matching Candidates Record) necessaria per l'esplorazione dello spazio degli stati.
     * 
     * @param queue Queue SYCL usata per il caricamento dei dati e l'esecuzione dei kernel.
     * @param query_graph Batch di grafi query nel formato HostCSRGraphInput provenienti dal binding.
     * @param data_graph Batch di grafi target (database) nel formato HostCSRGraphInput.
     * @param candidates Oggetto SIGMo dei candidati, aggiornato dalle fasi precedenti (Filter/Refine).
     * @param gmcr Oggetto per la gestione del mapping dei candidati, fondamentale per la fase di join.
     * @param num_matches Riferimento (output) in cui viene memorizzato il numero totale di isomorfismi trovati.
     * @param find_first Se true, il kernel si interrompe al primo match trovato per ogni coppia (migliora le performance 
     * se interessa solo la presenza del sottografo e non tutte le occorrenze).
     * 
     * * @return Statistiche aggregate sull'esecuzione (tempi di calcolo e conteggio grafi processati).
     */

    JoinCandidatesStats join_candidates ( 
        sycl::queue &queue,
        const std::vector<HostCSRGraphInput> &query_graph,
        const std::vector<HostCSRGraphInput> &data_graph,
        sigmo::candidates::Candidates &candidates,
        sigmo::isomorphism::mapping::GMCR &gmcr,
        std::size_t &num_matches,
        bool find_first = true
    );
} // namespace sigmo_python
