// Cheshire memory bridge v2: VRAM first, mapped system RAM when VRAM is short, and the
// choice of *what* spills made by buffer class instead of by arrival order.
//
// Header-only, and used by both backends. On HIP it is wired in through cuda_to_hip.h, which
// shadows AliceVision's cudaMalloc / cudaMallocPitch / cudaMalloc3D / cudaFree so memory.hpp
// needs no source change. That trick relies on the CUDA names being absent from the HIP
// headers, so it cannot work in reverse: the CUDA build routes memory.hpp's five device
// alloc/free sites to this bridge explicitly (apply_hip_patch.py step 6), and picks up the HIP
// entry points below through cheshire/hip_to_cuda.h.
//
// Classes (docs/02-memory-bridge.md):
//   Volume  3D pitched allocations: SGM / Refine similarity volumes. Streamed by every
//           aggregation pass: the biggest class, but coalesced, so the cheapest per byte behind PCIe.
//   Map     2D pitched allocations: depth / sim / normal maps and per-tile working buffers.
//   Image   camera mipmap levels (tagged by the mipmap emulation, mipmap_emu.h). Sampled as
//           random texture fetches by every similarity kernel: by far the worst class to spill.
//   Other   everything else (1D buffers, small tables).
//
// Placement:
//   * a soft VRAM cap (default 90 % of the VRAM free when the bridge first runs, so WDDM on
//     Windows never pages behind our back and Linux never hits a hard OOM);
//   * Image is VRAM-only by default: measured on the RX 9070, camera images behind PCIe cost
//     13x (random texture fetches, no cache reuse), volumes 5.6x for 30x the bytes, maps 1.9x;
//     the planner keeps a VRAM reserve for them (setReserve) that the other classes respect;
//   * Volume / Map / Other take VRAM while the cap minus that reserve allows, then mapped host
//     memory (volumes are the cheapest class per byte to spill: streamed, coalesced);
//   * a registry maps device pointers of spilled blocks to their host pointers so
//     cudaFree() releases the right thing; per-class live-byte counts for the planner and logs.
//
// Environment knobs:
//   CHESHIRE_BRIDGE=0                    disable (plain HIP behaviour, no cap, no spill)
//   CHESHIRE_BRIDGE_VRAM_MB=N            soft cap on bytes placed in VRAM (0 = no cap, v1 behaviour)
//   CHESHIRE_BRIDGE_VRAM_FRACTION=F      default cap = F * free VRAM at init (default 0.9)
//   CHESHIRE_BRIDGE_HOST_MB=N            cap on spilled host bytes (default 25 % of RAM)
//   CHESHIRE_BRIDGE_HOST_CLASSES=a,b     force these classes to host memory (image,map,volume,other)
//   CHESHIRE_BRIDGE_VRAM_ONLY_CLASSES=.. never spill these classes (fail with OOM instead); default: image
//   CHESHIRE_BRIDGE_IMAGE_SPILL=1        allow camera images to spill after all (experiments)
//   CHESHIRE_BRIDGE_PLANNER=0|1|2        DepthMapEstimator tile planner: 0 upstream heuristic on
//                                        hipMemGetInfo, 1 upstream formula on the bridge's VRAM
//                                        budget + image reserve, never fewer than one tile (default),
//                                        2 tile buffers get all the VRAM and images go to host
//   CHESHIRE_BRIDGE_MIN_SPILL_MB=N       allocations under N MB always stay in VRAM (default 4):
//                                        spilling a sub-megabyte map saves nothing and puts every
//                                        kernel read of it on the PCIe link (the 41-view set at
//                                        downscale 1 under a 2 GB cap spilled 3240 maps of 0 MB)
//   CHESHIRE_BRIDGE_LOG=1                log placement decisions and a per-class summary at exit
#pragma once
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#  include <hip/hip_runtime.h>
#else
// relative, not <cheshire/...>: the CUDA build must never get the shim directory on its
// include path, because it also holds cuda_runtime.h / cuda_fp16.h shims that would shadow the
// real CUDA headers. bridge.h and hip_to_cuda.h always sit side by side.
#  include "hip_to_cuda.h"
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#endif

namespace cheshire { namespace bridge {

enum class Tier { Vram, Host };
enum class Class { Other = 0, Map = 1, Volume = 2, Image = 3 };
constexpr int kClasses = 4;
inline const char* className(Class c) {
    static const char* n[kClasses] = {"other", "map", "volume", "image"};
    return n[int(c)];
}

struct ClassStats {
    size_t vramBytes = 0, hostBytes = 0;     // live
    size_t vramAllocs = 0, hostAllocs = 0;   // cumulative
    size_t vramPeak = 0, hostPeak = 0;
};
struct Stats {
    size_t vramBytes = 0;      // live bytes this process placed in VRAM through the bridge
    size_t hostBytes = 0;      // live bytes spilled to host
    size_t spills = 0;         // number of allocations that went to host
    size_t spillFailures = 0;  // host fallback also failed
    ClassStats cls[kClasses];
};
struct Budget {
    size_t vramCap = 0, vramUsed = 0, hostCap = 0, hostUsed = 0;
    bool enabled = false, imagesMaySpill = false;
    int planner = 1;
};

namespace detail {
struct Rec { void* host; size_t bytes; Class cls; };
struct VRec { size_t bytes; Class cls; };

inline Class& hint() { static thread_local Class c = Class::Other; return c; }

inline bool listHas(const char* list, const char* name) {
    if (!list) return false;
    std::string s(list); std::string n(name);
    size_t i = 0;
    while (i <= s.size()) {
        size_t j = s.find(',', i); if (j == std::string::npos) j = s.size();
        if (s.compare(i, j - i, n) == 0) return true;
        i = j + 1;
    }
    return false;
}

struct State {
    std::mutex m;
    std::unordered_map<void*, Rec> spilled;   // device ptr -> host record
    std::map<uintptr_t, size_t> spilledRanges; // device ptr -> bytes, for containment lookups
    std::unordered_map<void*, VRec> vram;     // device ptr -> bytes/class (for the soft cap)
    Stats stats;
    bool enabled = true, log = false, capInit = false;
    bool hostFirst[kClasses] = {false, false, false, false};
    bool vramOnly[kClasses] = {false, false, false, false};
    size_t vramCap = ~size_t(0), hostCap = 0, reserve = 0;
    size_t minSpill = size_t(4) << 20;
    double vramFraction = 0.9;
    long long vramCapEnv = -1;   // -1 unset, 0 no cap, >0 MB
    int planner = 1;
    State() {
        const char* e = std::getenv("CHESHIRE_BRIDGE");
        enabled = !(e && e[0] == '0');
        log = std::getenv("CHESHIRE_BRIDGE_LOG") != nullptr;
        if (const char* v = std::getenv("CHESHIRE_BRIDGE_VRAM_MB")) vramCapEnv = std::strtoll(v, nullptr, 10);
        if (const char* f = std::getenv("CHESHIRE_BRIDGE_VRAM_FRACTION")) vramFraction = std::strtod(f, nullptr);
        if (const char* f = std::getenv("CHESHIRE_BRIDGE_MIN_SPILL_MB")) minSpill = size_t(std::strtod(f, nullptr) * (1 << 20));
        if (const char* p = std::getenv("CHESHIRE_BRIDGE_PLANNER")) planner = std::atoi(p);
        const char* hc = std::getenv("CHESHIRE_BRIDGE_HOST_CLASSES");
        const char* vc = std::getenv("CHESHIRE_BRIDGE_VRAM_ONLY_CLASSES");
        for (int c = 0; c < kClasses; ++c) {
            hostFirst[c] = listHas(hc, className(Class(c)));
            vramOnly[c] = listHas(vc, className(Class(c)));
        }
        // camera images stay in VRAM unless explicitly allowed out (13x slower behind PCIe)
        const char* is = std::getenv("CHESHIRE_BRIDGE_IMAGE_SPILL");
        if (!(is && is[0] == '1') && planner != 2 && !hostFirst[int(Class::Image)]) vramOnly[int(Class::Image)] = true;
        size_t ram = 0;
#if defined(_WIN32)
        MEMORYSTATUSEX ms; ms.dwLength = sizeof(ms); if (GlobalMemoryStatusEx(&ms)) ram = size_t(ms.ullTotalPhys);
#else
        long pages = sysconf(_SC_PHYS_PAGES), psz = sysconf(_SC_PAGE_SIZE); if (pages > 0 && psz > 0) ram = size_t(pages) * size_t(psz);
#endif
        hostCap = ram / 4;
        if (const char* h = std::getenv("CHESHIRE_BRIDGE_HOST_MB")) hostCap = size_t(std::strtoull(h, nullptr, 10)) << 20;
    }
    ~State() {
        if (!log) return;
        std::fprintf(stderr, "[cheshire] bridge summary: %zu spills, %zu spill failures\n", stats.spills, stats.spillFailures);
        for (int c = 0; c < kClasses; ++c) {
            const ClassStats& s = stats.cls[c];
            if (s.vramAllocs + s.hostAllocs == 0) continue;
            std::fprintf(stderr, "[cheshire]   %-6s vram: %zu allocs, peak %zu MB | host: %zu allocs, peak %zu MB\n",
                         className(Class(c)), s.vramAllocs, s.vramPeak >> 20, s.hostAllocs, s.hostPeak >> 20);
        }
    }
    // the VRAM cap needs a device context; resolve it on first use (under the lock)
    void initCap() {
        if (capInit) return;
        capInit = true;
        size_t freeB = 0, totalB = 0;
        const bool haveInfo = (hipMemGetInfo(&freeB, &totalB) == hipSuccess && freeB > 0);
        if (vramCapEnv == 0) { vramCap = ~size_t(0); }
        else if (vramCapEnv > 0) {
            vramCap = size_t(vramCapEnv) << 20;
            // An explicit cap above what the device actually has is not reachable, and it is
            // actively harmful: the cap is also the tile planner's budget, so the planner commits
            // more concurrent tiles than the card can hold and the bridge fills VRAM to the brim
            // trying to honour it. Allocations the bridge owns then spill correctly, but anything
            // it does NOT own has nowhere to go - on CUDA the camera mipmaps are real
            // cudaMipmappedArrays that never pass through cudaMalloc, so they hit a hard OOM with
            // a couple of hundred MB left (measured 2026-09-20, docs/18). Clamp to the same
            // fraction of free VRAM the default uses, which leaves that headroom.
            // Only a cap above the device's TOTAL memory is clamped - that one is unreachable by
            // definition. A cap below total is a deliberate budget and is left exactly as asked,
            // including the ones the bridge-v2 matrices in docs/validation pin down to 500 MB.
            if (haveInfo && vramCap > totalB) {
                const size_t ceiling = size_t(double(freeB) * vramFraction);
                if (log)
                    std::fprintf(stderr, "[cheshire] bridge: requested vram cap %zu MB exceeds this device (%zu MB total); clamping to %zu MB\n",
                                 vramCap >> 20, totalB >> 20, ceiling >> 20);
                vramCap = ceiling;
            }
        }
        else {
            if (haveInfo) vramCap = size_t(double(freeB) * vramFraction);
            else vramCap = ~size_t(0);
        }
        if (log) {
            std::fprintf(stderr, "[cheshire] bridge: vram cap %s MB, host cap %zu MB, planner %d, host-first:",
                         vramCap == ~size_t(0) ? "none" : std::to_string(vramCap >> 20).c_str(), hostCap >> 20, planner);
            for (int c = 0; c < kClasses; ++c) if (hostFirst[c]) std::fprintf(stderr, " %s", className(Class(c)));
            std::fprintf(stderr, " vram-only:");
            for (int c = 0; c < kClasses; ++c) if (vramOnly[c]) std::fprintf(stderr, " %s", className(Class(c)));
            std::fprintf(stderr, "\n");
        }
    }
};
inline State& st() { static State s; return s; }

inline size_t alignUp(size_t v, size_t a) { return (v + a - 1) / a * a; }

inline size_t pitchAlign() {
    static size_t a = [] {
        hipDeviceProp_t p{}; int d = 0;
        if (hipGetDevice(&d) == hipSuccess && hipGetDeviceProperties(&p, d) == hipSuccess && p.texturePitchAlignment > 0)
            return size_t(p.texturePitchAlignment) > 256 ? size_t(p.texturePitchAlignment) : size_t(256);
        return size_t(256);
    }();
    return a;
}

// Allocate `bytes` in mapped host memory; returns the device-visible pointer.
inline hipError_t hostAlloc(void** devPtr, size_t bytes, Class cls, const char* what) {
    State& s = st();
    std::lock_guard<std::mutex> g(s.m);
    if (s.stats.hostBytes + bytes > s.hostCap) { s.stats.spillFailures++; return hipErrorOutOfMemory; }
    void* h = nullptr;
    // Non-coherent (coarse-grained, device-cached) on purpose: GPU atomics to fine-grained
    // system memory are silently wrong on platforms without PCIe atomics (hip/tests/
    // host_atomics.hip on an RX 6750 XT / Linux: atomicMin wrong on every element, atomicAdd
    // fine, both correct on non-coherent memory; the fused SGM aggregation atomicMins into a
    // spilled map). The CPU only touches spilled blocks through hipMemcpy after a sync.
    hipError_t e = hipHostMalloc(&h, bytes, hipHostMallocMapped | hipHostMallocNonCoherent);
    if (e != hipSuccess) { s.stats.spillFailures++; return e; }
    void* d = nullptr;
    e = hipHostGetDevicePointer(&d, h, 0);
    if (e != hipSuccess) { (void)hipHostFree(h); s.stats.spillFailures++; return e; }
    s.spilled[d] = Rec{h, bytes, cls};
    s.spilledRanges[reinterpret_cast<uintptr_t>(d)] = bytes;
    s.stats.hostBytes += bytes; s.stats.spills++;
    ClassStats& cs = s.stats.cls[int(cls)];
    cs.hostBytes += bytes; cs.hostAllocs++; if (cs.hostBytes > cs.hostPeak) cs.hostPeak = cs.hostBytes;
    if (s.log) std::fprintf(stderr, "[cheshire] bridge: %s %s %zu MB -> host (host total %zu MB, vram total %zu MB)\n",
                            className(cls), what, bytes >> 20, s.stats.hostBytes >> 20, s.stats.vramBytes >> 20);
    *devPtr = d;
    return hipSuccess;
}

// Decide whether `bytes` of class `cls` may be placed in VRAM under the soft cap.
inline bool vramAllowed(size_t bytes, Class cls) {
    State& s = st();
    std::lock_guard<std::mutex> g(s.m);
    s.initCap();
    if (!s.enabled) return true;
    if (s.vramOnly[int(cls)]) return true;
    if (s.hostFirst[int(cls)]) return false;
    // The reserve is *for* images and is the planner's estimate of what they will need; the
    // images already resident are counted in vramBytes, so only the part not yet loaded is held
    // back. Without this a 2 GB cap with 0.5 GB in use spilled every 93 MB map to the host
    // while 1.4 GB of VRAM sat empty (RX 6750 XT, 41 views at downscale 1).
    size_t reserve = 0;
    if (cls != Class::Image) {
        const size_t liveImages = s.stats.cls[int(Class::Image)].vramBytes;
        reserve = s.reserve > liveImages ? s.reserve - liveImages : 0;
    }
    if (s.vramCap == ~size_t(0)) return true;
    if (bytes < s.minSpill) return true;   // small blocks: the cap is soft, and a host-resident 4 KB map costs a PCIe round trip per read
    return s.stats.vramBytes + bytes + reserve <= s.vramCap;
}
inline bool mayFallBack(Class cls) { return !st().vramOnly[int(cls)]; }
inline void noteVram(void* p, size_t bytes, Class cls) {
    State& s = st(); std::lock_guard<std::mutex> g(s.m);
    s.vram[p] = VRec{bytes, cls}; s.stats.vramBytes += bytes;
    ClassStats& cs = s.stats.cls[int(cls)];
    cs.vramBytes += bytes; cs.vramAllocs++; if (cs.vramBytes > cs.vramPeak) cs.vramPeak = cs.vramBytes;
}
inline Class classFor(Class shapeDefault) {
    const Class h = hint();
    return h == Class::Other ? shapeDefault : h;
}
}  // namespace detail

// Tag every allocation made while the scope is alive (e.g. camera mipmap levels).
struct ClassScope {
    Class prev;
    explicit ClassScope(Class c) : prev(detail::hint()) { detail::hint() = c; }
    ~ClassScope() { detail::hint() = prev; }
};

inline bool enabled() { return detail::st().enabled; }
inline Stats stats() { auto& s = detail::st(); std::lock_guard<std::mutex> g(s.m); return s.stats; }
// Is `p` inside a spilled (mapped host) block? Copies are often issued from offset pointers.
inline bool inSpilled(const void* p) {
    auto& s = detail::st();
    if (!s.enabled) return false;
    std::lock_guard<std::mutex> g(s.m);
    if (s.spilledRanges.empty()) return false;
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    auto it = s.spilledRanges.upper_bound(a);
    if (it == s.spilledRanges.begin()) return false;
    --it;
    return a < it->first + it->second;
}
inline Tier tierOf(const void* devPtr) {
    auto& s = detail::st(); std::lock_guard<std::mutex> g(s.m);
    return s.spilled.count(const_cast<void*>(devPtr)) ? Tier::Host : Tier::Vram;
}
// VRAM the planner wants kept free for camera images (they are allocated after the tile buffers).
inline void setReserve(size_t bytes) { auto& s = detail::st(); std::lock_guard<std::mutex> g(s.m); s.reserve = bytes; }

// VRAM the bridge does not own but must count. On CUDA the camera mipmaps are real
// cudaMipmappedArrays, allocated by the driver and never through cudaMalloc, so until 0.3.2 they
// were invisible here: no "image" line in the summary and a budget the card could not honour on a
// small card. The depth-map code notes each array's estimated bytes after creating it and forgets
// them before freeing it; the bytes count towards vramUsed and the class peaks like any other
// VRAM placement, and free() never sees these keys (forget removes them first).
inline void noteExternal(const void* key, size_t bytes, Class cls) {
    auto& s = detail::st();
    if (!s.enabled) return;
    detail::noteVram(const_cast<void*>(key), bytes, cls);
}
inline void forgetExternal(const void* key) {
    auto& s = detail::st(); std::lock_guard<std::mutex> g(s.m);
    auto v = s.vram.find(const_cast<void*>(key));
    if (v == s.vram.end()) return;
    s.stats.vramBytes -= v->second.bytes; s.stats.cls[int(v->second.cls)].vramBytes -= v->second.bytes;
    s.vram.erase(v);
}
inline Budget budget() {
    auto& s = detail::st(); std::lock_guard<std::mutex> g(s.m);
    s.initCap();
    Budget b;
    b.enabled = s.enabled;
    b.vramCap = s.vramCap; b.vramUsed = s.stats.vramBytes;
    b.hostCap = s.hostCap; b.hostUsed = s.stats.hostBytes;
    b.imagesMaySpill = s.enabled && !s.vramOnly[int(Class::Image)] && s.hostCap > 0;
    b.planner = s.enabled ? s.planner : 0;
    if (b.vramCap == ~size_t(0)) {   // no cap: report what the device says
        size_t freeB = 0, totalB = 0;
        if (hipMemGetInfo(&freeB, &totalB) == hipSuccess) { b.vramCap = freeB + s.stats.vramBytes; }
    }
    return b;
}

inline hipError_t malloc(void** devPtr, size_t bytes) {
    if (!enabled()) return hipMalloc(devPtr, bytes);
    const Class cls = detail::classFor(Class::Other);
    hipError_t e = hipErrorOutOfMemory;
    if (detail::vramAllowed(bytes, cls)) { e = hipMalloc(devPtr, bytes); if (e == hipSuccess) { detail::noteVram(*devPtr, bytes, cls); return e; } (void)hipGetLastError(); }
    if (!detail::mayFallBack(cls)) return e;
    return detail::hostAlloc(devPtr, bytes, cls, "1D");
}

inline hipError_t mallocPitch(void** devPtr, size_t* pitch, size_t widthBytes, size_t height) {
    if (!enabled()) return hipMallocPitch(devPtr, pitch, widthBytes, height);
    const Class cls = detail::classFor(Class::Map);
    const size_t p = detail::alignUp(widthBytes, detail::pitchAlign());
    const size_t bytes = p * height;
    hipError_t e = hipErrorOutOfMemory;
    if (detail::vramAllowed(bytes, cls)) {
        e = hipMallocPitch(devPtr, pitch, widthBytes, height);
        if (e == hipSuccess) { detail::noteVram(*devPtr, *pitch * height, cls); return e; }
        (void)hipGetLastError();
    }
    if (!detail::mayFallBack(cls)) return e;
    e = detail::hostAlloc(devPtr, bytes, cls, "2D pitched");
    if (e == hipSuccess) *pitch = p;
    return e;
}

inline hipError_t malloc3D(hipPitchedPtr* pp, hipExtent extent) {
    if (!enabled()) return hipMalloc3D(pp, extent);
    const Class cls = detail::classFor(Class::Volume);
    const size_t p = detail::alignUp(extent.width, detail::pitchAlign());
    const size_t bytes = p * extent.height * extent.depth;
    hipError_t e = hipErrorOutOfMemory;
    if (detail::vramAllowed(bytes, cls)) {
        e = hipMalloc3D(pp, extent);
        if (e == hipSuccess) { detail::noteVram(pp->ptr, pp->pitch * extent.height * extent.depth, cls); return e; }
        (void)hipGetLastError();
    }
    if (!detail::mayFallBack(cls)) return e;
    void* d = nullptr;
    e = detail::hostAlloc(&d, bytes, cls, "3D volume");
    if (e == hipSuccess) { pp->ptr = d; pp->pitch = p; pp->xsize = extent.width; pp->ysize = extent.height; }
    return e;
}

inline hipError_t free(void* devPtr) {
    if (devPtr == nullptr) return hipSuccess;
    auto& s = detail::st();
    void* host = nullptr; size_t bytes = 0;
    {
        std::lock_guard<std::mutex> g(s.m);
        auto it = s.spilled.find(devPtr);
        if (it != s.spilled.end()) {
            host = it->second.host; bytes = it->second.bytes;
            s.stats.hostBytes -= bytes; s.stats.cls[int(it->second.cls)].hostBytes -= bytes;
            s.spilled.erase(it);
            s.spilledRanges.erase(reinterpret_cast<uintptr_t>(devPtr));
        } else {
            auto v = s.vram.find(devPtr);
            if (v != s.vram.end()) { s.stats.vramBytes -= v->second.bytes; s.stats.cls[int(v->second.cls)].vramBytes -= v->second.bytes; s.vram.erase(v); }
        }
    }
    return host ? hipHostFree(host) : hipFree(devPtr);
}

inline void logSummary(const char* tag = "") {
    Stats s = stats();
    std::fprintf(stderr, "[cheshire] bridge%s%s: vram %zu MB live, host %zu MB live, %zu spills, %zu spill failures\n",
                 tag[0] ? " " : "", tag, s.vramBytes >> 20, s.hostBytes >> 20, s.spills, s.spillFailures);
    for (int c = 0; c < kClasses; ++c) {
        const ClassStats& cs = s.cls[c];
        if (cs.vramAllocs + cs.hostAllocs == 0) continue;
        std::fprintf(stderr, "[cheshire]   %-6s vram %zu MB live (peak %zu), host %zu MB live (peak %zu)\n",
                     className(Class(c)), cs.vramBytes >> 20, cs.vramPeak >> 20, cs.hostBytes >> 20, cs.hostPeak >> 20);
    }
}

}}  // namespace cheshire::bridge
