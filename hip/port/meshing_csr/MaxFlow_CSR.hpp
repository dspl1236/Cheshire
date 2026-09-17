// Cheshire: the s-t graph for Meshing's max-flow as a compressed sparse row graph.
//
// Upstream's MaxFlow_AdjList builds a boost::adjacency_list with a std::vector of out-edges per
// cell: 21.7 M cells and 195 M directed edges on the 107-photo engine bay job take 66 s to build
// and 25 s to free, around 114 s of Boykov-Kolmogorov. This class records the same addNode /
// addEdge calls and lays the graph out as a compressed_sparse_row_graph at compute() time, with
// every node's out-edges in exactly the order the adjacency list would have held them (the s/t edge
// first, then the facet edges and reverse edges in insertion order), so Boykov-Kolmogorov visits the
// same edges in the same order and the flow and the labelling are the ones the adjacency list gives.
// CHESHIRE_MAXFLOW_ADJLIST=1 in GraphFiller::binarize selects upstream's class.
#pragma once

#include <aliceVision/system/Logger.hpp>

#include <boost/graph/compressed_sparse_row_graph.hpp>
#include <boost/graph/boykov_kolmogorov_max_flow.hpp>
#include <boost/property_map/property_map.hpp>

#include <cassert>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace aliceVision {
namespace fuseCut {

class MaxFlow_CSR
{
  public:
    using NodeType = int;
    using ValueType = float;
    using EdgeDesc = boost::detail::csr_edge_descriptor<std::size_t, std::size_t>;

    struct Edge
    {
        ValueType capacity{};
        ValueType residual{};
        EdgeDesc reverse;
    };

    using Graph = boost::compressed_sparse_row_graph<boost::directedS, boost::no_property, Edge, boost::no_property, std::size_t, std::size_t>;

    explicit MaxFlow_CSR(std::size_t numNodes)
      : _numNodes(numNodes),
        _S(numNodes),
        _T(numNodes + 1)
    {
        _nodes.reserve(numNodes);
        _edges.reserve(numNodes * 4);
    }

    inline void addNode(NodeType n, ValueType source, ValueType sink)
    {
        assert(source >= 0 && sink >= 0);
        _nodes.push_back({std::uint32_t(n), source - sink});
    }

    inline void addEdge(NodeType n1, NodeType n2, ValueType capacity, ValueType reverseCapacity)
    {
        assert(capacity >= 0 && reverseCapacity >= 0);
        _edges.push_back({std::uint32_t(n1), std::uint32_t(n2), capacity, reverseCapacity});
    }

    ValueType compute()
    {
        build();
        ALICEVISION_LOG_INFO("# vertices: " << boost::num_vertices(_graph) << ", edges: " << boost::num_edges(_graph) << " (CSR)");
        ALICEVISION_LOG_INFO("Compute boykov_kolmogorov_max_flow.");
        const std::size_t nbVertices = boost::num_vertices(_graph);
        _color.assign(nbVertices, boost::white_color);
        std::vector<EdgeDesc> pred(nbVertices);
        std::vector<std::size_t> dist(nbVertices);
        const ValueType v = boost::boykov_kolmogorov_max_flow(_graph,
                                                              boost::get(&Edge::capacity, _graph),
                                                              boost::get(&Edge::residual, _graph),
                                                              boost::get(&Edge::reverse, _graph),
                                                              &pred[0],
                                                              &_color[0],
                                                              &dist[0],
                                                              boost::get(boost::vertex_index, _graph),
                                                              _S,
                                                              _T);
        std::size_t nbBlack = 0, nbWhite = 0, nbGray = 0;
        for (std::size_t i = 0; i < nbVertices; ++i)
        {
            if (_color[i] == boost::black_color) ++nbBlack;
            else if (_color[i] == boost::white_color) ++nbWhite;
            else ++nbGray;
        }
        ALICEVISION_LOG_INFO("Full (white): " << nbWhite << ", Empty (black): " << nbBlack << ", Undefined (gray): " << nbGray);
        _graph = Graph();   // the labelling is all that is needed from here on
        return v;
    }

    /// is full
    inline bool isSource(NodeType n) const { return (_color[n] == boost::black_color); }
    inline bool isTarget(NodeType n) const { return (_color[n] == boost::white_color); }

  private:
    struct NodeRec { std::uint32_t n; ValueType score; };
    struct EdgeRec { std::uint32_t n1, n2; ValueType cap, rcap; };

    // the recorded calls replayed into per-node out-edge lists in adjacency-list order
    struct Laid
    {
        std::vector<std::size_t> rowstart;   // V + 1
        std::vector<std::uint32_t> target;   // E
        std::vector<ValueType> cap;          // E
        std::vector<std::uint32_t> partner;  // E: global index of the reverse edge
        std::vector<std::uint32_t> psrc;     // E: source node of the reverse edge
    };

    // forward iterators over the laid-out edges, in global (row-major) order
    struct EdgeIt
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<std::size_t, std::size_t>;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = value_type;
        const Laid* l = nullptr; std::size_t pos = 0, row = 0;
        mutable value_type cur;
        EdgeIt() = default;
        EdgeIt(const Laid* laid, std::size_t p) : l(laid), pos(p), row(0) { if (l) while (row + 1 < l->rowstart.size() && l->rowstart[row + 1] <= pos) ++row; }
        reference operator*() const { return {row, std::size_t(l->target[pos])}; }
        pointer operator->() const { cur = **this; return &cur; }
        EdgeIt& operator++() { ++pos; while (row + 1 < l->rowstart.size() && l->rowstart[row + 1] <= pos) ++row; return *this; }
        EdgeIt operator++(int) { EdgeIt t = *this; ++*this; return t; }
        bool operator==(const EdgeIt& o) const { return pos == o.pos; }
        bool operator!=(const EdgeIt& o) const { return pos != o.pos; }
    };
    struct PropIt
    {
        using iterator_category = std::forward_iterator_tag;
        using value_type = Edge;
        using difference_type = std::ptrdiff_t;
        using pointer = const Edge*;
        using reference = Edge;
        const Laid* l = nullptr; std::size_t pos = 0;
        PropIt() = default;
        PropIt(const Laid* laid, std::size_t p) : l(laid), pos(p) {}
        reference operator*() const { Edge e; e.capacity = l->cap[pos]; e.residual = ValueType(0); e.reverse = EdgeDesc(l->psrc[pos], l->partner[pos]); return e; }
        PropIt& operator++() { ++pos; return *this; }
        PropIt operator++(int) { PropIt t = *this; ++*this; return t; }
        bool operator==(const PropIt& o) const { return pos == o.pos; }
        bool operator!=(const PropIt& o) const { return pos != o.pos; }
    };

    void build()
    {
        const std::size_t V = _numNodes + 2;
        Laid l;
        // out-degrees: the s/t edge and its reverse, then one edge per addEdge on each side
        std::vector<std::uint32_t> deg(V, 0);
        for (const NodeRec& r : _nodes)
        {
            if (r.score > 0) { ++deg[_S]; ++deg[r.n]; }
            else { ++deg[r.n]; ++deg[_T]; }
        }
        for (const EdgeRec& r : _edges) { ++deg[r.n1]; ++deg[r.n2]; }
        l.rowstart.resize(V + 1);
        l.rowstart[0] = 0;
        for (std::size_t i = 0; i < V; ++i) l.rowstart[i + 1] = l.rowstart[i] + deg[i];
        const std::size_t E = l.rowstart[V];
        deg.clear(); deg.shrink_to_fit();
        l.target.resize(E); l.cap.resize(E); l.partner.resize(E); l.psrc.resize(E);
        std::vector<std::size_t> fill(l.rowstart.begin(), l.rowstart.end() - 1);
        auto app = [&](std::size_t from, std::size_t to, ValueType c) { const std::size_t pos = fill[from]++; l.target[pos] = std::uint32_t(to); l.cap[pos] = c; return pos; };
        auto pair = [&](std::size_t e, std::size_t esrc, std::size_t r, std::size_t rsrc) {
            l.partner[e] = std::uint32_t(r); l.psrc[e] = std::uint32_t(rsrc);
            l.partner[r] = std::uint32_t(e); l.psrc[r] = std::uint32_t(esrc);
        };
        // the same sequence of appends as MaxFlow_AdjList::addNode / addEdge produce
        for (const NodeRec& r : _nodes)
        {
            if (r.score > 0)
            {
                const std::size_t e = app(_S, r.n, r.score), rv = app(r.n, _S, r.score);
                pair(e, _S, rv, r.n);
            }
            else
            {
                const std::size_t e = app(r.n, _T, -r.score), rv = app(_T, r.n, -r.score);
                pair(e, r.n, rv, _T);
            }
        }
        for (const EdgeRec& r : _edges)
        {
            const std::size_t e = app(r.n1, r.n2, r.cap), rv = app(r.n2, r.n1, r.rcap);
            pair(e, r.n1, rv, r.n2);
        }
        fill.clear(); fill.shrink_to_fit();
        _nodes.clear(); _nodes.shrink_to_fit();
        _edges.clear(); _edges.shrink_to_fit();
        _graph = Graph(boost::edges_are_sorted, EdgeIt(&l, 0), EdgeIt(&l, E), PropIt(&l, 0), V, E);
    }

    std::size_t _numNodes;
    std::size_t _S, _T;
    std::vector<NodeRec> _nodes;
    std::vector<EdgeRec> _edges;
    Graph _graph;
    std::vector<boost::default_color_type> _color;
};

}  // namespace fuseCut
}  // namespace aliceVision
