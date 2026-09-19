// Cheshire: pick the runtime family and GPU target for the bundled Windows package (docs/16).
//
// Shared by cheshire-detect.exe (the standalone probe) and meshroom-pair-launcher.exe, so the
// two cannot disagree about which payload a card should get - a disagreement would be invisible
// until it produced wrong output.
//
// The bundle carries several payloads and choosing wrong is the worst failure this project has:
// a package built for one chip enumerates another, runs every kernel and returns wrong data with
// no error. So the choice is made by asking the card, not by a table of PCI device IDs - a table
// would need an entry for hardware that does not exist yet, which is exactly the case it has to
// get right.
//
// Both runtimes export hipGetDevicePropertiesR0600, the ABI-versioned entry point, so one probe
// loads either. The families are tried newest first, and **whichever runtime enumerates the card
// is the family to use**: the HIP 7.2 runtime answers hipErrorNoDevice for RX 6000 and older, and
// the HIP 6 runtime the driver ships answers for them. Detection and family selection are one
// question.
#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// hipDeviceProp_tR0600 as of ROCm 6.0 - the layout the R0600 symbol promises. Only gcnArchName is
// read, but the fields before it must match exactly, so this mirrors hip_runtime_api.h rather than
// guessing an offset. Declared here so the probe needs no HIP headers to build.
struct hipUUID_t { char bytes[16]; };
struct DevProp0600 {
    char name[256];
    hipUUID_t uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseHipArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingHipArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved[63];
    int hipReserved[32];
    char gcnArchName[256];
    // ... more fields follow; not read here.
};

typedef int (*FnCount)(int*);
typedef int (*FnProps)(DevProp0600*, int);

struct Family { const wchar_t* dir; const wchar_t* dll; };
// Newest first: if the 7.2 runtime enumerates the card, it is the better one to use.
static const Family kFamilies[] = {
    { L"rocm7.2", L"amdhip64_7.dll" },
    { L"hip6.2",  L"amdhip64_6.dll" },
};

static std::wstring exeDir() {
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 4);
    std::wstring p(buf, n);
    size_t i = p.find_last_of(L"\\/");
    return i == std::wstring::npos ? L"." : p.substr(0, i);
}

static bool dirExists(const std::wstring& p) {
    DWORD a = GetFileAttributesW(p.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

// "gfx1031:xnack-" -> "gfx1031". The runtime appends feature flags to the arch name.
static std::string baseArch(const char* s) {
    std::string a(s ? s : "");
    size_t c = a.find(':');
    if (c != std::string::npos) a.resize(c);
    return a;
}

// gfx1031 -> gfx10-3-generic, gfx1201 -> gfx12-generic, gfx1151 -> gfx11-generic.
// Generic targets are grouped by major version, and within gfx10 by the first minor digit,
// which is how AMD names them (gfx10-1-generic covers gfx101x, gfx10-3-generic gfx103x).
static std::string genericFor(const std::string& arch) {
    if (arch.rfind("gfx", 0) != 0) return "";
    std::string d = arch.substr(3);
    if (d.size() < 3) return "";
    if (d.rfind("10", 0) == 0 && d.size() >= 3) return std::string("gfx10-") + d[2] + "-generic";
    if (d.rfind("11", 0) == 0) return "gfx11-generic";
    if (d.rfind("12", 0) == 0) return "gfx12-generic";
    if (d.rfind("13", 0) == 0) return "gfx13-generic";   // whatever comes next, if AMD keeps the scheme
    return "";
}

static std::wstring widen(const std::string& s) { return std::wstring(s.begin(), s.end()); }

// Choose the payload for the card in this machine. Returns true and fills family/target, or false
// if no family enumerated a device it has a payload for. Reasons go to stderr when verbose.
// Returns false with *layoutMismatch set if a runtime answered but its property layout no longer
// matches DevProp0600 - the caller must not fall back to guessing in that case.
static bool cheshireSelectPayload(const std::wstring& root, bool verbose,
                                  std::wstring* family, std::wstring* target,
                                  bool* layoutMismatch = nullptr) {
    if (layoutMismatch) *layoutMismatch = false;
    const std::wstring gpu = root + L"\gpu";
    for (const auto& fam : kFamilies) {
        const std::wstring famDir = gpu + L"\\" + fam.dir;
        if (!dirExists(famDir)) { if (verbose) fwprintf(stderr, L"[detect] %ls: no payload
", fam.dir); continue; }

        // The runtime LoadLibrary's its code-object manager from beside itself, so add the family
        // directory to the search path before loading rather than relying on the working directory.
        AddDllDirectory(famDir.c_str());
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        HMODULE h = LoadLibraryExW((famDir + L"\\" + fam.dll).c_str(), nullptr,
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (!h) { if (verbose) fwprintf(stderr, L"[detect] %ls: %ls did not load (%lu)
", fam.dir, fam.dll, GetLastError()); continue; }

        FnCount count = (FnCount)GetProcAddress(h, "hipGetDeviceCount");
        FnProps props = (FnProps)GetProcAddress(h, "hipGetDevicePropertiesR0600");
        int n = 0;
        if (!count || !props || count(&n) != 0 || n <= 0) {
            if (verbose) fwprintf(stderr, L"[detect] %ls: no device (%d)
", fam.dir, n);
            FreeLibrary(h);
            continue;
        }
        DevProp0600 p{};
        if (props(&p, 0) != 0) { if (verbose) fwprintf(stderr, L"[detect] %ls: properties failed
", fam.dir); FreeLibrary(h); continue; }

        const std::string arch = baseArch(p.gcnArchName);
        if (verbose) fwprintf(stderr, L"[detect] %ls: %d device(s), arch %ls
", fam.dir, n, widen(arch).c_str());
        // DevProp0600 is transcribed from hip_runtime_api.h, and a future runtime that moved a field
        // would have gcnArchName read some unrelated bytes. Every AMD arch name starts with "gfx",
        // so anything else means the layout no longer matches - fail loudly rather than pick a
        // payload from garbage, which is the exact failure this exists to prevent.
        if (arch.rfind("gfx", 0) != 0) {
            fwprintf(stderr, L"[detect] %ls: arch name \"%ls\" is not a gfx target - the runtime's "
                             L"device-property layout no longer matches this build; refusing to guess
",
                     fam.dir, widen(arch).c_str());
            FreeLibrary(h);
            if (layoutMismatch) *layoutMismatch = true;
            return false;
        }

        // Exact chip directory first, then the generic that covers it.
        std::vector<std::string> tries{arch};
        const std::string g = genericFor(arch);
        if (!g.empty()) tries.push_back(g);
        for (const auto& t : tries) {
            if (dirExists(famDir + L"\\" + widen(t))) {
                *family = fam.dir;
                *target = widen(t);
                FreeLibrary(h);
                return true;
            }
        }
        if (verbose) fwprintf(stderr, L"[detect] %ls: no payload for %ls
", fam.dir, widen(arch).c_str());
        FreeLibrary(h);
        // Fall through: another family may carry this chip.
    }
    return false;
}
