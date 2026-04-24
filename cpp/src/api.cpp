#include "sigmo_python/api.hpp"
#include "sigmo_python/exceptions.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

#include "types.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "signature.hpp"
#include "candidates.hpp"
#include "isomorphism.hpp"

namespace sigmo_python
{

    // Questa funzione mi serve per convertire i grafi ricevuti dal binding
    // nel formato HostCSRGraphInput nei veri sigmo::CSRGraph usati dalla libreria.
    //
    // Qui faccio anche i controlli minimi di consistenza, così se lato Python
    // arriva un grafo corrotto me ne accorgo subito e lancio un errore chiaro.
    static std::vector<sigmo::CSRGraph> to_sigmo_csr_graphs(
        const std::vector<HostCSRGraphInput> &graphs)
    {
        std::vector<sigmo::CSRGraph> csr_graphs;
        csr_graphs.reserve(graphs.size());

        for (const auto &g : graphs)
        {
            // In CSR gli offset devono essere esattamente num_nodes + 1.
            if (g.row_offsets.size() != g.num_nodes + 1)
            {
                throw InvalidGraphInputError("row_offsets must have length num_nodes + 1");
            }

            // Devo avere una label per ogni nodo.
            if (g.node_labels.size() != g.num_nodes)
            {
                throw InvalidGraphInputError("node_labels must have length num_nodes");
            }

            // Ogni arco deve avere la sua label, quindi i due vettori devono avere la stessa lunghezza.
            if (g.column_indices.size() != g.edge_labels.size())
            {
                throw InvalidGraphInputError("column_indices and edge_labels must have the same length");
            }

            // Nell'ultima posizione di row_offsets deve esserci il numero totale
            // di elementi in column_indices, altrimenti la struttura CSR non è coerente.
            if (!g.row_offsets.empty() &&
                static_cast<std::size_t>(g.row_offsets.back()) != g.column_indices.size())
            {
                throw InvalidGraphInputError("row_offsets.back() must equal len(column_indices)");
            }

            // Cast espliciti per convertire dai tipi di input a quelli usati da sigmo.
            std::vector<sigmo::types::row_offset_t> row_offsets(
                g.row_offsets.begin(), g.row_offsets.end());
            std::vector<sigmo::types::col_index_t> column_indices(
                g.column_indices.begin(), g.column_indices.end());
            std::vector<sigmo::types::label_t> node_labels(
                g.node_labels.begin(), g.node_labels.end());
            std::vector<sigmo::types::label_t> edge_labels(
                g.edge_labels.begin(), g.edge_labels.end());

            // Costruisco il grafo CSR di sigmo e lo aggiungo alla lista.
            csr_graphs.emplace_back(
                row_offsets,
                column_indices,
                node_labels,
                edge_labels,
                g.num_nodes);
        }

        return csr_graphs;
    }

    // Creo una queue SYCL in_order così mantengo un'esecuzione più prevedibile
    // e non devo gestire complicazioni inutili con l'ordine delle operazioni
    static sycl::queue make_queue(const sycl::device &dev)
    {
        return sycl::queue(dev, sycl::property::queue::in_order{});
    }

    // Mi serve per interpretare lo scope passato dal lato Python.
    // Accetto "data" oppure "query", in modo case-insensitive.
    static bool is_query_scope(std::string scope)
    {
        std::transform(
            scope.begin(),
            scope.end(),
            scope.begin(),
            [](unsigned char c)
            { return static_cast<char>(std::tolower(c)); });

        if (scope == "query")
        {
            return true;
        }

        if (scope == "data")
        {
            return false;
        }

        throw InvalidScopeError("scope must be either 'data' or 'query'");
    }

    // Wrapper del path generateCSRSignatures.
    //
    // L'idea qui è:
    // 1. ricevo i grafi lato host
    // 2. li converto in sigmo::CSRGraph
    // 3. creo il DeviceBatchedCSRGraph
    // 4. genero le signatures sul device
    // 5. restituisco solo statistiche semplici lato Python
    GraphBatchStats generate_csr_signatures(
        const sycl::device &dev,
        const std::vector<HostCSRGraphInput> &graphs,
        const std::string &scope)
    {
        sycl::queue q = make_queue(dev);

        // Se non ho grafi, ritorno subito stats vuote.
        if (graphs.empty())
        {
            return GraphBatchStats{0, 0, 0, 0};
        }

        const bool query_scope = is_query_scope(scope);

        // Conversione host-side -> CSRGraph di SIGMo.
        auto csr_graphs = to_sigmo_csr_graphs(graphs);

        // Creo il batch device a partire dai grafi CSR.
        sigmo::DeviceBatchedCSRGraph device_graph;
        try
        {
            device_graph = sigmo::createDeviceCSRGraph(q, csr_graphs);
        }
        catch (const std::bad_alloc &)
        {
            throw OutOfDeviceMemoryError("Failed to allocate CSR graphs on GPU");
        }

        try
        {
            GraphBatchStats stats{0, 0, 0, 0};

            if (query_scope)
            {
                // Per lo scope query alloco solo la parte query delle signatures.
                sigmo::signature::Signature<> signatures(
                    q,
                    0,
                    device_graph.total_nodes);

                // Chiamo il path CSR per generare le signatures query.
                signatures.generateQuerySignatures(device_graph).wait();

                stats = GraphBatchStats{
                    device_graph.num_graphs,
                    static_cast<std::size_t>(device_graph.total_nodes),
                    static_cast<std::size_t>(device_graph.total_edges),
                    signatures.getQuerySignatureAllocationSize()};
            }
            else
            {
                // Per lo scope data alloco solo la parte data delle signatures.
                sigmo::signature::Signature<> signatures(
                    q,
                    device_graph.total_nodes,
                    0);

                // Chiamo il path CSR per generare le signatures data.
                signatures.generateDataSignatures(device_graph).wait();

                stats = GraphBatchStats{
                    device_graph.num_graphs,
                    static_cast<std::size_t>(device_graph.total_nodes),
                    static_cast<std::size_t>(device_graph.total_edges),
                    signatures.getDataSignatureAllocationSize()};
            }

            // Alla fine libero esplicitamente il batch sul device.
            sigmo::destroyDeviceCSRGraph(device_graph, q);

            // Aspetto che tutte le operazioni sulla queue siano davvero concluse
            // e faccio propagare eventuali errori SYCL.
            q.wait_and_throw();

            return stats;
        }
        catch (const sycl::exception &e)
        {
            sigmo::destroyDeviceCSRGraph(device_graph, q);
            throw DeviceRuntimeError(e.what());
        }
    }

    // Wrapper del path refineCSRSignatures.
    //
    // Qui il flusso è lo stesso di prima, ma con un passo in più:
    // prima genero le signatures base e poi applico il refinement.
    GraphBatchStats refine_csr_signatures(
        const sycl::device &dev,
        const std::vector<HostCSRGraphInput> &graphs,
        const std::string &scope,
        std::size_t view_size)
    {
        sycl::queue q = make_queue(dev);

        if (graphs.empty())
        {
            return GraphBatchStats{0, 0, 0, 0};
        }

        const bool query_scope = is_query_scope(scope);

        // Conversione e Validazione
        auto csr_graphs = to_sigmo_csr_graphs(graphs);

        // Allocazione Device
        sigmo::DeviceBatchedCSRGraph device_graph;
        try
        {
            device_graph = sigmo::createDeviceCSRGraph(q, csr_graphs);
        }
        catch (const std::bad_alloc &)
        {
            throw OutOfDeviceMemoryError("Incapable of allocating CSR graphs for refinement on GPU");
        }

        try
        {
            GraphBatchStats stats{0, 0, 0, 0};

            if (query_scope)
            {
                sigmo::signature::Signature<> signatures(q, 0, device_graph.total_nodes);

                // Generazione base + Raffinamento
                signatures.generateQuerySignatures(device_graph).wait();
                signatures.refineQuerySignatures(device_graph, view_size).wait();

                stats = {device_graph.num_graphs, (size_t)device_graph.total_nodes,
                         (size_t)device_graph.total_edges, signatures.getQuerySignatureAllocationSize()};
            }
            else
            {
                sigmo::signature::Signature<> signatures(q, device_graph.total_nodes, 0);

                signatures.generateDataSignatures(device_graph).wait();
                signatures.refineDataSignatures(device_graph, view_size).wait();

                stats = {device_graph.num_graphs, (size_t)device_graph.total_nodes,
                         (size_t)device_graph.total_edges, signatures.getDataSignatureAllocationSize()};
            }

            sigmo::destroyDeviceCSRGraph(device_graph, q);
            q.wait_and_throw();
            return stats;
        }
        catch (const sycl::exception &e)
        {
            // Cleanup di emergenza se il kernel fallisce
            sigmo::destroyDeviceCSRGraph(device_graph, q);
            throw DeviceRuntimeError(std::string("Refinement Kernel failed: ") + e.what());
        }
    }

    FilterCandidatesStats filter_candidates(
        sycl::queue &q,
        const std::vector<HostCSRGraphInput> &query_input,
        const std::vector<HostCSRGraphInput> &data_input,
        sigmo::signature::Signature<> &signatures,
        sigmo::candidates::Candidates &candidates)
    {
        // Validazione preventiva
        if (query_input.empty() || data_input.empty())
        {
            throw InvalidGraphInputError("Filter aborted: query or data input is empty");
        }

        auto csr_q = to_sigmo_csr_graphs(query_input);
        auto csr_d = to_sigmo_csr_graphs(data_input);

        // Allocazione Device (Monitoraggio memoria USM)
        sigmo::DeviceBatchedCSRGraph dev_q, dev_d;
        try
        {
            dev_q = sigmo::createDeviceCSRGraph(q, csr_q);
            dev_d = sigmo::createDeviceCSRGraph(q, csr_d);
        }
        catch (const std::bad_alloc &)
        {
            throw OutOfDeviceMemoryError("GPU Memory exhausted during graph upload for filtering");
        }

        // Bypass del riferimento corrotto tramite oggetto locale
        // Per via del fatto che i puntatori di antonio sono per riferimento e non per valore
        sigmo::signature::Signature<> local_sig(q, dev_d.total_nodes, dev_q.total_nodes);
        uint64_t total_candidates = 0;

        try
        {
            // Pipeline: Generazione firme -> Filtraggio
            local_sig.generateQuerySignatures(dev_q).wait();
            local_sig.generateDataSignatures(dev_d).wait();

            auto event = sigmo::isomorphism::filter::filterCandidates<sigmo::CandidatesDomain::Query>(
                q, dev_q, dev_d, local_sig, candidates);
            event.wait();
            q.wait_and_throw();

            // Trasferimento e Popcount dei risultati
            auto cand_dev = candidates.getCandidatesDevice();
            if (cand_dev.candidates != nullptr)
            {
                const std::size_t total_words = cand_dev.source_nodes * cand_dev.single_node_size;
                std::vector<sigmo::types::candidates_t> host_buffer(total_words);

                q.copy(cand_dev.candidates, host_buffer.data(), total_words).wait();

                // Bit a bit sempre perchè i puntatori
                for (auto word : host_buffer)
                {
                    total_candidates += __builtin_popcountll(static_cast<unsigned long long>(word));
                }
            }
        }
        catch (const sycl::exception &e)
        {
            sigmo::destroyDeviceCSRGraph(dev_q, q);
            sigmo::destroyDeviceCSRGraph(dev_d, q);
            throw DeviceRuntimeError(std::string("Filter Kernel failed: ") + e.what());
        }

        // Estrazione statistiche prima della distruzione dei grafi
        uint32_t allocated_bytes = static_cast<uint32_t>(candidates.getAllocationSize());
        FilterCandidatesStats stats{
            static_cast<uint32_t>(dev_q.num_graphs),
            static_cast<uint32_t>(dev_d.num_graphs),
            static_cast<size_t>(dev_q.total_nodes),
            static_cast<size_t>(dev_d.total_nodes),
            static_cast<size_t>(total_candidates),
            allocated_bytes};

        sigmo::destroyDeviceCSRGraph(dev_q, q);
        sigmo::destroyDeviceCSRGraph(dev_d, q);
        q.wait_and_throw();

        return stats;
    }

    RefineCandidatesStats refine_candidates(
        sycl::queue &q,
        const std::vector<HostCSRGraphInput> &query_input,
        const std::vector<HostCSRGraphInput> &data_input,
        sigmo::signature::Signature<> &signatures,
        sigmo::candidates::Candidates &candidates)
    {
        if (query_input.empty() || data_input.empty())
        {
            throw InvalidGraphInputError("Filter aborted: query or data input is empty");
        }

        auto csr_q = to_sigmo_csr_graphs(query_input);
        auto csr_d = to_sigmo_csr_graphs(data_input);

        sigmo::DeviceBatchedCSRGraph dev_q, dev_d;
        try
        {
            dev_q = sigmo::createDeviceCSRGraph(q, csr_q);
            dev_d = sigmo::createDeviceCSRGraph(q, csr_d);
        }
        catch (const std::bad_alloc &)
        {
            throw OutOfDeviceMemoryError("GPU Memory exhausted during graph upload for filtering");
        }

        sigmo::signature::Signature<> local_sig(q, dev_d.total_nodes, dev_q.total_nodes);
        uint64_t total_candidates = 0;

        try
        {
            local_sig.generateQuerySignatures(dev_q).wait();
            local_sig.generateDataSignatures(dev_d).wait();
            local_sig.refineQuerySignatures(dev_q).wait();
            local_sig.refineDataSignatures(dev_d).
            wait();

            auto event = sigmo::isomorphism::filter::refineCandidates<sigmo::CandidatesDomain::Query>(
                q, dev_q, dev_d, local_sig, candidates);
            event.wait();
            q.wait_and_throw();

            // Trasferimento e Popcount dei risultati
            auto cand_dev = candidates.getCandidatesDevice();
            if (cand_dev.candidates != nullptr)
            {
                const std::size_t total_words = cand_dev.source_nodes * cand_dev.single_node_size;
                std::vector<sigmo::types::candidates_t> host_buffer(total_words);

                q.copy(cand_dev.candidates, host_buffer.data(), total_words).wait();

                for (auto word : host_buffer)
                {
                    total_candidates += __builtin_popcountll(static_cast<unsigned long long>(word));
                }
            }
        }
        catch (const sycl::exception &e)
        {
            sigmo::destroyDeviceCSRGraph(dev_q, q);
            sigmo::destroyDeviceCSRGraph(dev_d, q);
            throw DeviceRuntimeError(std::string("Filter Kernel failed: ") + e.what());
        }

        uint32_t allocated_bytes = static_cast<uint32_t>(candidates.getAllocationSize());
        RefineCandidatesStats stats{
            static_cast<uint32_t>(dev_q.num_graphs),
            static_cast<uint32_t>(dev_d.num_graphs),
            static_cast<size_t>(dev_q.total_nodes),
            static_cast<size_t>(dev_d.total_nodes),
            static_cast<size_t>(total_candidates),
            allocated_bytes};

        sigmo::destroyDeviceCSRGraph(dev_q, q);
        sigmo::destroyDeviceCSRGraph(dev_d, q);
        q.wait_and_throw();

        return stats;
    }

} // namespace sigmo_python
