#include "sigmo_python/api.hpp"
#include "sigmo_python/exceptions.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>

#include "types.hpp"
#include "device.hpp"
#include "graph.hpp"
#include "gmcr.hpp"
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
            if (g.row_offsets.size() != g.num_nodes + 1)
            {
                throw InvalidGraphInputError("row_offsets must have length num_nodes + 1");
            }

            if (g.node_labels.size() != g.num_nodes)
            {
                throw InvalidGraphInputError("node_labels must have length num_nodes");
            }

            if (g.column_indices.size() != g.edge_labels.size())
            {
                throw InvalidGraphInputError("column_indices and edge_labels must have the same length");
            }

            if (!g.row_offsets.empty() &&
                static_cast<std::size_t>(g.row_offsets.back()) != g.column_indices.size())
            {
                throw InvalidGraphInputError("row_offsets.back() must equal len(column_indices)");
            }

            for (std::size_t i = 0; i + 1 < g.row_offsets.size(); ++i)
            {
                if (g.row_offsets[i] > g.row_offsets[i + 1])
                {
                    throw InvalidGraphInputError("row_offsets must be monotonic");
                }
            }

            for (auto col : g.column_indices)
            {
                if (col >= g.num_nodes)
                {
                    throw InvalidGraphInputError(
                        "column_indices contains node index out of range");
                }
            }

            std::vector<sigmo::types::row_offset_t> row_offsets(
                g.row_offsets.begin(), g.row_offsets.end());

            std::vector<sigmo::types::col_index_t> column_indices(
                g.column_indices.begin(), g.column_indices.end());

            std::vector<sigmo::types::label_t> node_labels(
                g.node_labels.begin(), g.node_labels.end());

            std::vector<sigmo::types::label_t> edge_labels(
                g.edge_labels.begin(), g.edge_labels.end());

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
    static bool is_query_scope(const std::string &scope)
    {
        if (scope.find("query") != std::string::npos)
            return true;
        if (scope.find("data") != std::string::npos)
            return false;

        // Invece di lanciare l'eccezione subito, stampa cosa hai ricevuto!
        std::cerr << "[SIGMO DEBUG] Received scope: '" << scope << "'" << std::endl;
        throw std::runtime_error("Invalid scope: " + scope);
    }

    std::size_t count_candidates_on_host(sycl::queue &q, sigmo::candidates::Candidates &candidates)
    {
        auto cand_dev = candidates.getCandidatesDevice();
        // Invece di dedurre la taglia, usiamo quella dichiarata dall'oggetto
        std::size_t total_words = cand_dev.source_nodes * cand_dev.single_node_size;

        std::vector<sigmo::types::candidates_t> host_buffer(total_words);
        q.copy(cand_dev.candidates, host_buffer.data(), total_words).wait();

        std::size_t total = 0;
        for (std::size_t i = 0; i < total_words; ++i)
        {
            // Usiamo un cast esplicito al tipo definito nella libreria
            // per evitare errori di dimensione (32 vs 64 bit)
            auto word = static_cast<uint64_t>(host_buffer[i]);
            total += __builtin_popcountll(word);
        }
        return total;
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
        sycl::queue &q,
        const std::vector<HostCSRGraphInput> &graphs,
        sigmo::signature::Signature<> &signatures,
        const std::string &scope)
    {
        if (graphs.empty())
            return GraphBatchStats{0, 0, 0, 0};

        const bool query_scope = is_query_scope(scope);

        // 1. Conversione efficiente in strutture SIGMo
        std::vector<sigmo::CSRGraph> sigmo_graphs;
        sigmo_graphs.reserve(graphs.size());

        for (const auto &g : graphs)
        {
            if (g.num_nodes == 0)
                continue;

            sigmo_graphs.emplace_back(
                g.row_offsets,
                g.column_indices,
                g.node_labels,
                g.edge_labels,
                g.num_nodes);
        }

        // 2. Trasferimento Batch su Device (USM)
        sigmo::DeviceBatchedCSRGraph device_graph;
        try
        {
            device_graph = sigmo::createDeviceCSRGraph(q, sigmo_graphs);
            q.wait_and_throw();
        }
        catch (const std::exception &e)
        {
            throw OutOfDeviceMemoryError(std::string("Errore allocazione GPU: ") + e.what());
        }

        // 3. Lancio del Kernel di calcolo Firme
        try
        {
            if (query_scope)
            {
                // .wait() è fondamentale per assicurare che le firme siano
                // scritte prima di distruggere il device_graph
                signatures.generateQuerySignatures(device_graph).wait();
            }
            else
            {
                signatures.generateDataSignatures(device_graph).wait();
            }

            // Recuperiamo le statistiche prima della distruzione
            GraphBatchStats stats{
                static_cast<uint32_t>(device_graph.num_graphs),
                static_cast<std::size_t>(device_graph.total_nodes),
                static_cast<std::size_t>(device_graph.total_edges),
                query_scope ? signatures.getQuerySignatureAllocationSize()
                            : signatures.getDataSignatureAllocationSize()};

            // 4. Pulizia e sincronizzazione finale
            sigmo::destroyDeviceCSRGraph(device_graph, q);
            q.wait_and_throw();

            return stats;
        }
        catch (const std::exception &e)
        {
            sigmo::destroyDeviceCSRGraph(device_graph, q);
            throw DeviceRuntimeError(std::string("GPU Kernel Error: ") + e.what());
        }
    }

    // Wrapper del path refineCSRSignatures.
    //
    // Qui il flusso è lo stesso di prima, ma con un passo in più:
    // prima genero le signatures base e poi applico il refinement.
    GraphBatchStats refine_csr_signatures(
        sycl::queue &q,
        const std::vector<HostCSRGraphInput> &graphs,
        sigmo::signature::Signature<> &signatures,
        const std::string &scope,
        std::size_t view_size)
    {
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
                // Generazione base + Raffinamento
                signatures.refineQuerySignatures(device_graph, view_size).wait();

                stats = {device_graph.num_graphs, (size_t)device_graph.total_nodes,
                         (size_t)device_graph.total_edges, signatures.getQuerySignatureAllocationSize()};
            }
            else
            {
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
        if (query_input.empty() || data_input.empty())
            throw InvalidGraphInputError("Filter aborted: query or data input is empty");

        auto convert = [](const std::vector<HostCSRGraphInput> &in)
        {
            std::vector<sigmo::CSRGraph> out;
            out.reserve(in.size());
            for (const auto &g : in)
            {
                if (g.num_nodes > 0)
                    out.emplace_back(g.row_offsets, g.column_indices, g.node_labels, g.edge_labels, g.num_nodes);
            }
            return out;
        };

        auto csr_q = convert(query_input);
        auto csr_d = convert(data_input);

        sigmo::DeviceBatchedCSRGraph dev_q, dev_d;
        try
        {
            dev_q = sigmo::createDeviceCSRGraph(q, csr_q);
            dev_d = sigmo::createDeviceCSRGraph(q, csr_d);
            q.wait_and_throw();
        }
        catch (const std::exception &e)
        {
            throw OutOfDeviceMemoryError(std::string("GPU Upload failed: ") + e.what());
        }

        try
        {
            auto event = sigmo::isomorphism::filter::filterCandidates<sigmo::CandidatesDomain::Query>(
                q, dev_q, dev_d, signatures, candidates);

            event.wait();
            q.wait_and_throw();

            uint64_t total_candidates_count = count_candidates_on_host(q, candidates);

            auto cand_device_info = candidates.getCandidatesDevice();

            FilterCandidatesStats stats{
                static_cast<uint32_t>(dev_q.num_graphs),
                static_cast<uint32_t>(dev_d.num_graphs),
                dev_q.total_nodes,
                dev_d.total_nodes,
                total_candidates_count,
                static_cast<uint32_t>(candidates.getAllocationSize())};

            sigmo::destroyDeviceCSRGraph(dev_q, q);
            sigmo::destroyDeviceCSRGraph(dev_d, q);
            return stats;
        }
        catch (const std::exception &e)
        {
            sigmo::destroyDeviceCSRGraph(dev_q, q);
            sigmo::destroyDeviceCSRGraph(dev_d, q);
            throw DeviceRuntimeError(std::string("Filter Kernel failed: ") + e.what());
        }
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

        uint64_t total_candidates = 0;

        try
        {
            uint64_t pre_refine = count_candidates_on_host(q, candidates);

            // Esecuzione del kernel
            auto event = sigmo::isomorphism::filter::refineCandidates<sigmo::CandidatesDomain::Query>(
                q, dev_q, dev_d, signatures, candidates);
            event.wait();
            q.wait_and_throw();

            auto cand_dev = candidates.getCandidatesDevice();

            total_candidates = count_candidates_on_host(q, candidates);
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

    JoinCandidatesStats join_candidates(
        sycl::queue &q,
        const std::vector<HostCSRGraphInput> &query_input,
        const std::vector<HostCSRGraphInput> &data_input,
        sigmo::candidates::Candidates &candidates,
        sigmo::isomorphism::mapping::GMCR &gmcr,
        std::size_t &num_matches,
        bool find_first)
    {
        if (query_input.empty() || data_input.empty())
        {
            throw InvalidGraphInputError("Join aborted: query or data input is empty");
        }

        auto csr_q = to_sigmo_csr_graphs(query_input);
        auto csr_d = to_sigmo_csr_graphs(data_input);

        sigmo::DeviceBatchedCSRGraph dev_q = {};
        sigmo::DeviceBatchedCSRGraph dev_d = {};
        double elapsed_ms = 0.0;

        size_t max_capacity = 50000000;
        sigmo::types::MatchPair *d_buffer = sycl::malloc_device<sigmo::types::MatchPair>(max_capacity, q);
        // d_count deve essere allocato per contare i match scritti
        size_t *d_count = sycl::malloc_device<size_t>(1, q);
        q.fill(d_count, size_t(0), 1).wait();

        sigmo::types::MatchResultsDevice out_results{d_buffer, d_count, max_capacity};
        size_t *d_num_matches = nullptr;

        try
        {
            dev_q = sigmo::createDeviceCSRGraph(q, csr_q);
            dev_d = sigmo::createDeviceCSRGraph(q, csr_d);
            d_num_matches = sycl::malloc_shared<size_t>(1, q);
            d_num_matches[0] = 0;

            gmcr.generateGMCR(dev_q, dev_d, candidates);
            q.wait_and_throw();

            auto gmcr_dev = gmcr.getGMCRDevice();

            if (gmcr_dev.total_query_indices > 0)
            {
                auto start = std::chrono::high_resolution_clock::now();

                sigmo::isomorphism::join::joinCandidates(
                    q, dev_q, dev_d, candidates, gmcr, out_results, d_num_matches, find_first);

                q.wait_and_throw();
                auto end = std::chrono::high_resolution_clock::now();
                elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
            }
            num_matches = d_num_matches[0];
        }
        catch (const std::exception &e)
        {
            if (d_num_matches)
                sycl::free(d_num_matches, q);

            if (dev_q.graph_offsets)
                sigmo::destroyDeviceCSRGraph(dev_q, q);

            if (dev_d.graph_offsets)
                sigmo::destroyDeviceCSRGraph(dev_d, q);
            
            if(d_buffer)
                sycl::free(d_buffer, q);
            
            if(d_count)
                sycl::free(d_count, q);
            throw;
        }

        JoinCandidatesStats stats;
        stats.num_matches = num_matches;
        stats.execution_time = elapsed_ms;
        stats.total_query_graph = dev_q.num_graphs;
        stats.total_data_graph = dev_d.num_graphs;

        // Leggiamo quanti match effettivi sono nel buffer (dal contatore atomico d_count)
        size_t actual_match_count = 0;
        q.memcpy(&actual_match_count, d_count, sizeof(size_t)).wait();

        if (actual_match_count > 0)
        {
            std::vector<sigmo::types::MatchPair> h_matches(actual_match_count);
            q.memcpy(h_matches.data(), d_buffer, actual_match_count * sizeof(sigmo::types::MatchPair)).wait();

            for (const auto &match : h_matches)
            {
                stats.matches_dict[match.query_id].push_back(match.data_id);
            }
        }

        sycl::free(d_num_matches, q);
        sycl::free(d_buffer, q);
        sycl::free(d_count, q);
        sigmo::destroyDeviceCSRGraph(dev_q, q);
        sigmo::destroyDeviceCSRGraph(dev_d, q);

        return stats;
    }

} // namespace sigmo_python
