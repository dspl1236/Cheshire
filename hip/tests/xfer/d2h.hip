// Scratch (step 14b): a GPU matcher search's result download, 22,600 queries x 2 neighbours of (int index, float distance)
// = 362 KB, the False Door's average, the ways it could be copied to the host vectors the matcher fills, each followed
// by the copy into those vectors (what the CPU then reads). Every way must give the same bytes; the times decide.
//   a. two pageable hipMemcpy D2H (idx, then dist) straight into the vectors - the matcher today
//   b. one pageable hipMemcpy of a contiguous [idx | dist] buffer into a host staging vector, then two memcpy
//   c. the same into default pinned memory (hipHostMalloc: coherent, uncached for the CPU), then two memcpy
//   d. the same into non-coherent pinned memory (hipHostMallocNonCoherent: cached), then two memcpy
//   e. hipMemcpyAsync of the contiguous buffer into non-coherent pinned memory + hipStreamSynchronize, then two memcpy
// Build: build-d2h.cmd. d2h [queries] [reps]
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using clk = std::chrono::steady_clock;

int main(int argc, char** argv)
{
    const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 22600;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 2000;
    const size_t half = n * 2 * sizeof(int);  // idx; dist is the same size
    void* dev = nullptr;
    if (hipMalloc(&dev, 2 * half) != hipSuccess) return 1;
    std::vector<unsigned char> init(2 * half);
    for (size_t k = 0; k < init.size(); ++k) init[k] = (unsigned char)(k * 2654435761u >> 13);
    hipMemcpy(dev, init.data(), 2 * half, hipMemcpyHostToDevice);
    void* devIdx = dev;
    void* devDist = (char*)dev + half;
    std::vector<int> idx(n * 2), ref_idx(n * 2);
    std::vector<float> dist(n * 2), ref_dist(n * 2);
    std::memcpy(ref_idx.data(), init.data(), half);
    std::memcpy(ref_dist.data(), init.data() + half, half);
    std::vector<unsigned char> stage(2 * half);
    void* pinC = nullptr;
    void* pinN = nullptr;
    hipHostMalloc(&pinC, 2 * half, hipHostMallocDefault);
    hipHostMalloc(&pinN, 2 * half, hipHostMallocNonCoherent);
    hipStream_t s;
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    auto check = [&](const char* name) {
        const bool ok = std::memcmp(idx.data(), ref_idx.data(), half) == 0 && std::memcmp(dist.data(), ref_dist.data(), half) == 0;
        std::memset(idx.data(), 0, half);
        std::memset(dist.data(), 0, half);
        if (!ok) std::printf("%s: WRONG BYTES\n", name);
        return ok;
    };
    auto timeIt = [&](const char* name, auto&& body) {
        body();
        if (!check(name)) return;
        body();
        const auto t0 = clk::now();
        for (int r = 0; r < reps; ++r) body();
        const double us = std::chrono::duration<double, std::micro>(clk::now() - t0).count() / reps;
        check(name);
        std::printf("%-58s %7.1f us\n", name, us);
    };
    timeIt("a. two pageable copies (today)", [&] {
        hipMemcpy(idx.data(), devIdx, half, hipMemcpyDeviceToHost);
        hipMemcpy(dist.data(), devDist, half, hipMemcpyDeviceToHost);
    });
    timeIt("b. one pageable copy + two memcpy", [&] {
        hipMemcpy(stage.data(), dev, 2 * half, hipMemcpyDeviceToHost);
        std::memcpy(idx.data(), stage.data(), half);
        std::memcpy(dist.data(), stage.data() + half, half);
    });
    timeIt("c. one copy to coherent pinned + two memcpy", [&] {
        hipMemcpy(pinC, dev, 2 * half, hipMemcpyDeviceToHost);
        std::memcpy(idx.data(), pinC, half);
        std::memcpy(dist.data(), (char*)pinC + half, half);
    });
    timeIt("d. one copy to non-coherent pinned + two memcpy", [&] {
        hipMemcpy(pinN, dev, 2 * half, hipMemcpyDeviceToHost);
        std::memcpy(idx.data(), pinN, half);
        std::memcpy(dist.data(), (char*)pinN + half, half);
    });
    timeIt("e. async copy to non-coherent pinned + sync + two memcpy", [&] {
        hipMemcpyAsync(pinN, dev, 2 * half, hipMemcpyDeviceToHost, s);
        hipStreamSynchronize(s);
        std::memcpy(idx.data(), pinN, half);
        std::memcpy(dist.data(), (char*)pinN + half, half);
    });
    return 0;
}
