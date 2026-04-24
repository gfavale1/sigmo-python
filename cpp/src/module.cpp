#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dpctl4pybind11.hpp"
#include "sigmo_python/api.hpp"
#include "sigmo_python/types.hpp"

#include "signature.hpp"
#include "candidates.hpp"

namespace py = pybind11;

// Funzione helper interna per convertire i grafi Python (dict) in C++
static std::vector<sigmo_python::HostCSRGraphInput> parse_graph_batch(const py::list &graphs_py)
{
    std::vector<sigmo_python::HostCSRGraphInput> graphs;
    graphs.reserve(py::len(graphs_py));

    for (const auto &item : graphs_py)
    {
        py::dict g = py::cast<py::dict>(item);
        sigmo_python::HostCSRGraphInput graph;

        graph.row_offsets = py::cast<std::vector<std::uint32_t>>(g["row_offsets"]);
        graph.column_indices = py::cast<std::vector<std::uint32_t>>(g["column_indices"]);
        graph.node_labels = py::cast<std::vector<std::uint32_t>>(g["node_labels"]);
        graph.edge_labels = py::cast<std::vector<std::uint32_t>>(g["edge_labels"]);

        if (g.contains("num_nodes"))
        {
            graph.num_nodes = py::cast<std::size_t>(g["num_nodes"]);
        }
        else
        {
            graph.num_nodes = static_cast<std::size_t>(py::len(g["node_labels"]));
        }

        graphs.push_back(std::move(graph));
    }
    return graphs;
}

static py::dict to_python_dict(const sigmo_python::GraphBatchStats &stats)
{
    py::dict out;
    out["num_graphs"] = stats.num_graphs;
    out["total_nodes"] = stats.total_nodes;
    out["total_edges"] = stats.total_edges;
    out["allocated_bytes"] = stats.allocated_bytes;
    return out;
}

PYBIND11_MODULE(_core, m)
{
    // Definizione classe Signature di sigmo base
    py::class_<sigmo::signature::Signature<>, std::unique_ptr<sigmo::signature::Signature<>, py::nodelete>>(m, "Signature")
        .def(py::init<sycl::queue &, std::size_t, std::size_t>(),
             py::keep_alive<1, 2>(),
             py::arg("queue"), py::arg("data_nodes"), py::arg("query_nodes"))
        .def("get_data_signature_allocation_size", &sigmo::signature::Signature<>::getDataSignatureAllocationSize)
        .def("get_query_signature_allocation_size", &sigmo::signature::Signature<>::getQuerySignatureAllocationSize)
        .def("get_max_labels", &sigmo::signature::Signature<>::getMaxLabels);

    // Definizione classe Candidates di sigmo base
    py::class_<sigmo::candidates::Candidates, std::unique_ptr<sigmo::candidates::Candidates, py::nodelete>>(m, "Candidates")
        .def(py::init<sycl::queue &, std::size_t, std::size_t>(),
             py::keep_alive<1, 2>(),
             py::arg("queue"), py::arg("source_nodes"), py::arg("target_nodes"))
        .def("get_allocation_size", &sigmo::candidates::Candidates::getAllocationSize)
        .def("get_candidates_count", py::overload_cast<sigmo::types::node_t>(&sigmo::candidates::Candidates::getCandidatesCount, py::const_),
             py::arg("source_node"));

    m.def("generate_csr_signatures", [](const sycl::device &dev, const py::list &graphs_py, const std::string &scope)
          {
            auto graphs = parse_graph_batch(graphs_py);
            auto stats = sigmo_python::generate_csr_signatures(dev, graphs, scope);
            
            py::dict out = to_python_dict(stats);
            out["scope"] = scope;
            return out; }, py::arg("device"), py::arg("graphs"), py::arg("scope") = "data");

    m.def("refine_csr_signatures", [](const sycl::device &dev, const py::list &graphs_py, const std::string &scope, std::size_t view_size)
          {
            auto graphs = parse_graph_batch(graphs_py);
            auto stats = sigmo_python::refine_csr_signatures(dev, graphs, scope, view_size);

            py::dict out = to_python_dict(stats);
            out["scope"] = scope;
            out["view_size"] = view_size;
            return out; }, py::arg("device"), py::arg("graphs"), py::arg("scope") = "data", py::arg("view_size") = 1);

    m.def("filter_candidates", [](sycl::queue &q, const py::list query_graph_py, const py::list data_graph_py, sigmo::signature::Signature<> &sig, sigmo::candidates::Candidates &cand)
          {
        auto query_batch = parse_graph_batch(query_graph_py);
        auto data_batch = parse_graph_batch(data_graph_py);
        auto stats = sigmo_python::filter_candidates(q, query_batch, data_batch, sig, cand);

        py::dict out;
        out["num_query_graphs"] = (uint32_t)stats.num_query_graphs;
        out["num_data_graphs"] = (uint32_t)stats.num_data_graphs;
        out["total_query_nodes"] = (uint32_t)stats.total_query_nodes;
        out["total_data_nodes"] = (uint32_t)stats.total_data_nodes;
        out["candidates_count"] = (uint64_t)stats.total_candidates;
        out["allocated_bytes"] = (uint64_t)stats.allocated_bytes;
        return out; }, py::arg("queue"), py::arg("query_graphs"), py::arg("data_graphs"), py::arg("signatures"), py::arg("candidates"), py::keep_alive<4, 1>(), py::keep_alive<5, 1>());

    m.def("refine_candidates", [](sycl::queue &q, const py::list &query_graph_py, const py::list &data_graph_py, sigmo::signature::Signature<> &sig, sigmo::candidates::Candidates &cand)
    {
        auto query_batch = parse_graph_batch(query_graph_py);
        auto data_batch = parse_graph_batch(data_graph_py);
        auto stats = sigmo_python::refine_candidates(q, query_batch, data_batch, sig, cand);

        py::dict out;
        out["num_query_graphs"] = stats.num_query_graphs;
        out["num_data_graphs"] = stats.num_data_graphs;
        out["total_query_nodes"] = stats.total_query_nodes;
        out["total_data_nodes"] = stats.total_data_nodes;
        out["candidates_count"] = stats.total_candidates;
        out["allocated_bytes"] = stats.allocated_bytes;
        
        return out;
    }, 
    py::arg("queue"), py::arg("query_graphs"), py::arg("data_graphs"), py::arg("signatures"), py::arg("candidates"), py::keep_alive<4, 1>(), py::keep_alive<5, 1>());
}