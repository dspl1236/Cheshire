// Cheshire: standalone harness for the GPU min-cut against a graph dumped by MaxFlow_CSR
// (CHESHIRE_MAXFLOW_DUMP): loads the CSR graph and Boykov-Kolmogorov's flow value and colours,
// runs cheshire::maxflow::minCut and reports the flow value, the disagreeing cells and the time.
//
//   maxflow_test <dump.bin> [verbose]
#include "../port/gpu_maxflow/maxflowGPU.hpp"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 2) { std::fprintf(stderr, "usage: maxflow_test <dump.bin> [verbose]\n"); return 2; }
    const int verbose = argc > 2 ? std::atoi(argv[2]) : 1;
    std::ifstream f(argv[1], std::ios::binary);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
    char magic[4]; f.read(magic, 4);
    if (std::memcmp(magic, "CHMF", 4) != 0) { std::fprintf(stderr, "not a max-flow dump\n"); return 2; }
    std::uint64_t hdr[4]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    const std::uint64_t V = hdr[0], E = hdr[1], S = hdr[2], T = hdr[3];
    if (V >= 0xffffffffull || E >= 0xffffffffull) { std::fprintf(stderr, "graph too large for 32-bit indices\n"); return 2; }
    std::vector<std::uint64_t> rowstart64(V + 1);
    f.read(reinterpret_cast<char*>(rowstart64.data()), std::streamsize((V + 1) * 8));
    std::vector<std::uint32_t> rowstart(V + 1), target(E), partner(E);
    std::vector<float> cap(E);
    for (std::uint64_t i = 0; i <= V; ++i) rowstart[i] = std::uint32_t(rowstart64[i]);
    rowstart64.clear(); rowstart64.shrink_to_fit();
    f.read(reinterpret_cast<char*>(target.data()), std::streamsize(E * 4));
    f.read(reinterpret_cast<char*>(cap.data()), std::streamsize(E * 4));
    f.read(reinterpret_cast<char*>(partner.data()), std::streamsize(E * 4));
    float bkFlow = 0; std::vector<std::uint8_t> bkColor(V);
    f.read(reinterpret_cast<char*>(&bkFlow), 4);
    f.read(reinterpret_cast<char*>(bkColor.data()), std::streamsize(V));
    const bool haveBk = bool(f);
    std::fprintf(stderr, "graph: %llu nodes, %llu edges, source %llu (%u edges), sink %llu (%u edges)%s\n", (unsigned long long)V, (unsigned long long)E,
                 (unsigned long long)S, rowstart[S + 1] - rowstart[S], (unsigned long long)T, rowstart[T + 1] - rowstart[T], haveBk ? ", BK result present" : ", no BK result");

    cheshire::maxflow::Graph g;
    g.nbNodes = std::uint32_t(V); g.nbEdges = std::uint32_t(E); g.source = std::uint32_t(S); g.sink = std::uint32_t(T);
    g.rowstart = rowstart.data(); g.target = target.data(); g.capacity = cap.data(); g.partner = partner.data();
    std::vector<std::uint8_t> sinkSide;
    cheshire::maxflow::Stats st;
    if (!cheshire::maxflow::minCut(g, sinkSide, st, verbose)) { std::fprintf(stderr, "minCut failed\n"); return 1; }
    std::size_t nbSink = 0;
    for (std::uint8_t v : sinkSide) nbSink += v;
    std::printf("gpu: flow %.7g, sink side %zu of %llu, %d pulses, %d global relabels, %.2f s\n", st.flow, nbSink, (unsigned long long)V, st.pulses, st.globalRelabels, st.seconds);
    // the value of a cut, in double: capacities of the edges leaving the source side
    auto cutValue = [&](auto sinkOf) {
        double v = 0;
        for (std::uint64_t u = 0; u < V; ++u) {
            if (sinkOf(u)) continue;
            for (std::uint32_t e = rowstart[u]; e < rowstart[u + 1]; ++e)
                if (sinkOf(target[e])) v += double(cap[e]);
        }
        return v;
    };
    std::printf("gpu: cut value (double, from the labelling) %.10g\n", cutValue([&](std::uint64_t u) { return u == T || (u != S && sinkSide[u]); }));
    if (haveBk) {
        std::printf("bk : cut value (double, from the labelling) %.10g\n", cutValue([&](std::uint64_t u) { return u == T || (u != S && bkColor[u] == 0); }));
        std::size_t white = 0, gray = 0, diffWhite = 0, diffBlack = 0;
        for (std::uint64_t i = 0; i < V; ++i) {
            if (i == S || i == T) continue;
            const bool bkWhite = bkColor[i] == 0;
            white += bkWhite; gray += bkColor[i] == 1;
            if (bkWhite && !sinkSide[i]) ++diffWhite;
            if (!bkWhite && sinkSide[i]) ++diffBlack;
        }
        std::printf("bk : flow %.7g, white %zu, gray %zu\n", bkFlow, white, gray);
        std::printf("cells labelled differently: %zu (BK white, GPU not) + %zu (GPU sink side, BK not) of %llu; flow ratio %.9f\n", diffWhite, diffBlack, (unsigned long long)V, double(st.flow) / double(bkFlow));
    }
    return 0;
}
