#include "sigmo_python/api.hpp"
#include "sigmo_python/exceptions.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

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

    std::size_t count_candidates_on_device(
        sycl::queue &q,
        sigmo::candidates::Candidates &candidates)
    {
        auto cand_dev = candidates.getCandidatesDevice();

        const std::size_t total_words =
            cand_dev.source_nodes * cand_dev.single_node_size;

        if (total_words == 0)
        {
            return 0;
        }

        constexpr std::size_t WG_SIZE = 256;

        const std::size_t num_groups =
            (total_words + WG_SIZE - 1) / WG_SIZE;

        std::uint64_t *partials =
            sycl::malloc_shared<std::uint64_t>(num_groups, q);

        if (!partials)
        {
            throw OutOfDeviceMemoryError(
                "Unable to allocate partial counter buffer for device-side candidate counting");
        }

        q.submit([&](sycl::handler &cgh)
                 {
        sycl::local_accessor<std::uint64_t, 1> local_counts(
            sycl::range<1>(WG_SIZE),
            cgh
        );

        cgh.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(num_groups * WG_SIZE),
                sycl::range<1>(WG_SIZE)
            ),
            [=](sycl::nd_item<1> item) {
                const std::size_t gid = item.get_global_linear_id();
                const std::size_t lid = item.get_local_linear_id();
                const std::size_t group_id = item.get_group_linear_id();

                std::uint64_t value = 0;

                if (gid < total_words) {
                    auto word = cand_dev.candidates[gid];
                    value = static_cast<std::uint64_t>(sycl::popcount(word));
                }

                local_counts[lid] = value;
                sycl::group_barrier(item.get_group());

                for (std::size_t stride = WG_SIZE / 2; stride > 0; stride /= 2) {
                    if (lid < stride) {
                        local_counts[lid] += local_counts[lid + stride];
                    }

                    sycl::group_barrier(item.get_group());
                }

                if (lid == 0) {
                    partials[group_id] = local_counts[0];
                }
            }
        ); })
            .wait();

        q.wait_and_throw();

        std::size_t total = 0;

        for (std::size_t i = 0; i < num_groups; ++i)
        {
            total += static_cast<std::size_t>(partials[i]);
        }

        sycl::free(partials, q);

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
            auto event = sigmo::isomorphism::filter::refineCandidates(
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

            if (d_buffer)
                sycl::free(d_buffer, q);

            if (d_count)
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

    NativePipeline::NativePipeline(
        sycl::queue queue,
        const std::vector<HostCSRGraphInput> &query_graphs,
        const std::vector<HostCSRGraphInput> &data_graphs,
        std::size_t memory_padding)
        : queue_(queue),
          memory_padding_(memory_padding)
    {
        if (query_graphs.empty() || data_graphs.empty())
        {
            throw InvalidGraphInputError(
                "NativePipeline requires non-empty query and data graph batches");
        }

        auto csr_query = to_sigmo_csr_graphs(query_graphs);
        auto csr_data = to_sigmo_csr_graphs(data_graphs);

        try
        {
            dev_query_ = sigmo::createDeviceCSRGraph(queue_, csr_query);
            dev_data_ = sigmo::createDeviceCSRGraph(queue_, csr_data);
            queue_.wait_and_throw();
            device_graphs_valid_ = true;
        }
        catch (const std::exception &e)
        {
            throw OutOfDeviceMemoryError(
                std::string("GPU graph upload failed in NativePipeline: ") + e.what());
        }

        const std::size_t total_q =
            static_cast<std::size_t>(dev_query_.total_nodes) + memory_padding_;

        const std::size_t total_d =
            static_cast<std::size_t>(dev_data_.total_nodes) + memory_padding_;

        try
        {
            signatures_ = std::make_unique<sigmo::signature::Signature<>>(
                queue_,
                total_d,
                total_q);

            candidates_ = std::make_unique<sigmo::candidates::Candidates>(
                queue_,
                total_q,
                total_d);

            gmcr_ = std::make_unique<sigmo::isomorphism::mapping::GMCR>(
                queue_);
        }
        catch (const std::exception &e)
        {
            if (device_graphs_valid_)
            {
                sigmo::destroyDeviceCSRGraph(dev_query_, queue_);
                sigmo::destroyDeviceCSRGraph(dev_data_, queue_);
                device_graphs_valid_ = false;
            }

            throw OutOfDeviceMemoryError(
                std::string("NativePipeline native object allocation failed: ") + e.what());
        }
    }

    NativePipeline::~NativePipeline()
    {
        try
        {
            if (device_graphs_valid_)
            {
                sigmo::destroyDeviceCSRGraph(dev_query_, queue_);
                sigmo::destroyDeviceCSRGraph(dev_data_, queue_);
                queue_.wait();
                device_graphs_valid_ = false;
            }
        }
        catch (...)
        {
            // Destructor must not throw.
        }
    }

    GraphBatchStats NativePipeline::generate_query_signatures()
    {
        try
        {
            signatures_->generateQuerySignatures(dev_query_).wait();
            queue_.wait_and_throw();

            return GraphBatchStats{
                static_cast<std::uint32_t>(dev_query_.num_graphs),
                static_cast<std::size_t>(dev_query_.total_nodes),
                static_cast<std::size_t>(dev_query_.total_edges),
                signatures_->getQuerySignatureAllocationSize()};
        }
        catch (const std::exception &e)
        {
            throw DeviceRuntimeError(
                std::string("generate_query_signatures failed: ") + e.what());
        }
    }

    GraphBatchStats NativePipeline::generate_data_signatures()
    {
        try
        {
            signatures_->generateDataSignatures(dev_data_).wait();
            queue_.wait_and_throw();

            return GraphBatchStats{
                static_cast<std::uint32_t>(dev_data_.num_graphs),
                static_cast<std::size_t>(dev_data_.total_nodes),
                static_cast<std::size_t>(dev_data_.total_edges),
                signatures_->getDataSignatureAllocationSize()};
        }
        catch (const std::exception &e)
        {
            throw DeviceRuntimeError(
                std::string("generate_data_signatures failed: ") + e.what());
        }
    }

    GraphBatchStats NativePipeline::refine_query_signatures(std::size_t view_size)
    {
        try
        {
            signatures_->refineQuerySignatures(dev_query_, view_size).wait();
            queue_.wait_and_throw();

            return GraphBatchStats{
                static_cast<std::uint32_t>(dev_query_.num_graphs),
                static_cast<std::size_t>(dev_query_.total_nodes),
                static_cast<std::size_t>(dev_query_.total_edges),
                signatures_->getQuerySignatureAllocationSize()};
        }
        catch (const std::exception &e)
        {
            throw DeviceRuntimeError(
                std::string("refine_query_signatures failed: ") + e.what());
        }
    }

    GraphBatchStats NativePipeline::refine_data_signatures(std::size_t view_size)
    {
        try
        {
            signatures_->refineDataSignatures(dev_data_, view_size).wait();
            queue_.wait_and_throw();

            return GraphBatchStats{
                static_cast<std::uint32_t>(dev_data_.num_graphs),
                static_cast<std::size_t>(dev_data_.total_nodes),
                static_cast<std::size_t>(dev_data_.total_edges),
                signatures_->getDataSignatureAllocationSize()};
        }
        catch (const std::exception &e)
        {
            throw DeviceRuntimeError(
                std::string("refine_data_signatures failed: ") + e.what());
        }
    }

    FilterCandidatesStats NativePipeline::filter_candidates()
    {
        try
        {
            auto event = sigmo::isomorphism::filter::filterCandidates(
                queue_,
                dev_query_,
                dev_data_,
                *signatures_,
                *candidates_);

            event.wait();
            queue_.wait_and_throw();

            const std::size_t total_candidates =
                count_candidates_on_device(queue_, *candidates_);

            return FilterCandidatesStats{
                static_cast<std::uint32_t>(dev_query_.num_graphs),
                static_cast<std::uint32_t>(dev_data_.num_graphs),
                static_cast<std::size_t>(dev_query_.total_nodes),
                static_cast<std::size_t>(dev_data_.total_nodes),
                total_candidates,
                static_cast<std::size_t>(candidates_->getAllocationSize())};
        }
        catch (const std::exception &e)
        {
            throw DeviceRuntimeError(
                std::string("filter_candidates failed: ") + e.what());
        }
    }

    RefineCandidatesStats NativePipeline::refine_candidates()
    {
        try
        {
            auto event = sigmo::isomorphism::filter::refineCandidates(
                queue_,
                dev_query_,
                dev_data_,
                *signatures_,
                *candidates_);

            event.wait();
            queue_.wait_and_throw();

            const std::size_t total_candidates =
                count_candidates_on_device(queue_, *candidates_);

            return RefineCandidatesStats{
                static_cast<std::uint32_t>(dev_query_.num_graphs),
                static_cast<std::uint32_t>(dev_data_.num_graphs),
                static_cast<std::size_t>(dev_query_.total_nodes),
                static_cast<std::size_t>(dev_data_.total_nodes),
                total_candidates,
                static_cast<std::size_t>(candidates_->getAllocationSize())};
        }
        catch (const std::exception &e)
        {
            throw DeviceRuntimeError(
                std::string("refine_candidates failed: ") + e.what());
        }
    }

    JoinCandidatesStats NativePipeline::join_candidates(bool find_first)
    {
        auto total_start = std::chrono::high_resolution_clock::now();

        JoinCandidatesStats stats{};
        std::size_t num_matches = 0;
        double elapsed_ms = 0.0;

        /*
         * We do not materialize huge match sets into Python.
         * The total number of matches is still computed exactly through d_num_matches.
         */
        constexpr std::size_t MATCH_MATERIALIZATION_LIMIT = 100000;
        constexpr std::size_t max_capacity = MATCH_MATERIALIZATION_LIMIT;

        auto alloc_start = std::chrono::high_resolution_clock::now();

        sigmo::types::MatchPair *d_buffer =
            sycl::malloc_device<sigmo::types::MatchPair>(max_capacity, queue_);

        std::size_t *d_count =
            sycl::malloc_device<std::size_t>(1, queue_);

        std::size_t *d_num_matches =
            sycl::malloc_device<std::size_t>(1, queue_);

        auto alloc_end = std::chrono::high_resolution_clock::now();

        if (!d_buffer || !d_count || !d_num_matches)
        {
            if (d_buffer)
                sycl::free(d_buffer, queue_);
            if (d_count)
                sycl::free(d_count, queue_);
            if (d_num_matches)
                sycl::free(d_num_matches, queue_);

            throw OutOfDeviceMemoryError(
                "Unable to allocate join output buffers");
        }

        try
        {
            queue_.fill(d_count, std::size_t(0), 1).wait();
            queue_.fill(d_num_matches, std::size_t(0), 1).wait();

            sigmo::types::MatchResultsDevice out_results{
                d_buffer,
                d_count,
                max_capacity};

            auto gmcr_start = std::chrono::high_resolution_clock::now();

            gmcr_->generateGMCR(dev_query_, dev_data_, *candidates_);
            queue_.wait_and_throw();

            auto gmcr_end = std::chrono::high_resolution_clock::now();

            auto gmcr_dev = gmcr_->getGMCRDevice();

            if (gmcr_dev.total_query_indices > 0)
            {
                auto kernel_start = std::chrono::high_resolution_clock::now();

                sigmo::isomorphism::join::joinCandidates(
                    queue_,
                    dev_query_,
                    dev_data_,
                    *candidates_,
                    *gmcr_,
                    out_results,
                    d_num_matches,
                    find_first);

                queue_.wait_and_throw();

                auto kernel_end = std::chrono::high_resolution_clock::now();

                elapsed_ms =
                    std::chrono::duration<double, std::milli>(
                        kernel_end - kernel_start)
                        .count();
            }

            auto count_copy_start = std::chrono::high_resolution_clock::now();

            queue_.memcpy(&num_matches, d_num_matches, sizeof(std::size_t)).wait();

            auto count_copy_end = std::chrono::high_resolution_clock::now();

            stats.num_matches = num_matches;
            stats.execution_time = elapsed_ms;
            stats.total_query_graph = static_cast<std::uint32_t>(dev_query_.num_graphs);
            stats.total_data_graph = static_cast<std::uint32_t>(dev_data_.num_graphs);

            auto materialization_start = std::chrono::high_resolution_clock::now();

            if (num_matches <= MATCH_MATERIALIZATION_LIMIT)
            {
                std::size_t actual_match_count = 0;
                queue_.memcpy(&actual_match_count, d_count, sizeof(std::size_t)).wait();

                const std::size_t safe_match_count =
                    std::min(actual_match_count, max_capacity);

                if (safe_match_count > 0)
                {
                    std::vector<sigmo::types::MatchPair> h_matches(safe_match_count);

                    queue_.memcpy(
                              h_matches.data(),
                              d_buffer,
                              safe_match_count * sizeof(sigmo::types::MatchPair))
                        .wait();

                    for (const auto &match : h_matches)
                    {
                        stats.matches_dict[match.query_id].push_back(match.data_id);
                    }
                }
            }

            auto materialization_end = std::chrono::high_resolution_clock::now();

            auto free_start = std::chrono::high_resolution_clock::now();

            sycl::free(d_buffer, queue_);
            sycl::free(d_count, queue_);
            sycl::free(d_num_matches, queue_);

            auto free_end = std::chrono::high_resolution_clock::now();
            auto total_end = std::chrono::high_resolution_clock::now();

            auto ms = [](auto start, auto end)
            {
                return std::chrono::duration<double, std::milli>(end - start).count();
            };

            //std::cerr << "[JOIN DEBUG] "
            //          << "alloc_ms=" << ms(alloc_start, alloc_end)
            //         << " gmcr_ms=" << ms(gmcr_start, gmcr_end)
            //          << " kernel_ms=" << elapsed_ms
            //          << " count_copy_ms=" << ms(count_copy_start, count_copy_end)
            //         << " materialization_ms=" << ms(materialization_start, materialization_end)
            //          << " free_ms=" << ms(free_start, free_end)
            //          << " total_cpp_ms=" << ms(total_start, total_end)
            //          << std::endl;

            return stats;
        }
        catch (...)
        {
            sycl::free(d_buffer, queue_);
            sycl::free(d_count, queue_);
            sycl::free(d_num_matches, queue_);
            throw;
        }
    }

    std::size_t NativePipeline::total_query_nodes() const
    {
        return static_cast<std::size_t>(dev_query_.total_nodes);
    }

    std::size_t NativePipeline::total_data_nodes() const
    {
        return static_cast<std::size_t>(dev_data_.total_nodes);
    }

    std::size_t NativePipeline::total_query_graphs() const
    {
        return static_cast<std::size_t>(dev_query_.num_graphs);
    }

    std::size_t NativePipeline::total_data_graphs() const
    {
        return static_cast<std::size_t>(dev_data_.num_graphs);
    }

} // namespace sigmo_python
