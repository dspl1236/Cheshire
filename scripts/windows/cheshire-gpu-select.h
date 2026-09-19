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

// Probe the families in this process. `onlyFamily`, when not -1, restricts it to one - which is how
// the isolated path below uses it, one family per child process.
//
// Do not call this with more than one family available unless you are the child: loading a HIP
// runtime that does not support the installed driver can take the process down rather than report
// no device. On an RX 6750 XT under a scheduled task, loading amdhip64_7.dll exits 0xC0000005
// before a single line of output; the same call over an interactive session returns "no device"
// politely. That is why cheshireSelectPayload spawns instead (docs/16).
static bool cheshireProbeFamilies(const std::wstring& root, bool verbose, int onlyFamily,
                                  std::wstring* family, std::wstring* target,
                                  bool* layoutMismatch = nullptr) {
    if (layoutMismatch) *layoutMismatch = false;
    // Unbuffered: when this is redirected to a file and something below dies inside a vendor
    // runtime, block buffering loses every line that would say where. The volume is a few lines.
    if (verbose) setvbuf(stderr, nullptr, _IONBF, 0);
    // AddDllDirectory takes absolute paths only, and LoadLibraryExW with the SEARCH flags will not
    // take a relative one either: a relative root fails with ERROR_MOD_NOT_FOUND, which reads like
    // a missing dependency rather than a bad argument. Normalise instead of failing obscurely.
    wchar_t full[MAX_PATH * 4];
    const DWORD fn = GetFullPathNameW(root.c_str(), MAX_PATH * 4, full, nullptr);
    const std::wstring base = (fn > 0 && fn < MAX_PATH * 4) ? std::wstring(full, fn) : root;
    const std::wstring gpu = base + L"\\gpu";
    for (int fi = 0; fi < (int)(sizeof(kFamilies) / sizeof(kFamilies[0])); ++fi) {
        if (onlyFamily >= 0 && fi != onlyFamily) continue;
        const Family& fam = kFamilies[fi];
        const std::wstring famDir = gpu + L"\\" + fam.dir;
        if (!dirExists(famDir)) {
            if (verbose) fwprintf(stderr, L"[detect] %ls: no payload\n", fam.dir);
            continue;
        }

        // The runtime LoadLibrary's its code-object manager from beside itself, so add the family
        // directory to the search path before loading rather than relying on the working directory.
        AddDllDirectory(famDir.c_str());
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        HMODULE h = LoadLibraryExW((famDir + L"\\" + fam.dll).c_str(), nullptr,
                                   LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS);
        if (!h) {
            if (verbose) fwprintf(stderr, L"[detect] %ls: %ls did not load (%lu)\n",
                                  fam.dir, fam.dll, GetLastError());
            continue;
        }

        FnCount count = (FnCount)GetProcAddress(h, "hipGetDeviceCount");
        FnProps props = (FnProps)GetProcAddress(h, "hipGetDevicePropertiesR0600");
        int n = 0;
        if (!count || !props || count(&n) != 0 || n <= 0) {
            if (verbose) fwprintf(stderr, L"[detect] %ls: no device (%d)\n", fam.dir, n);
            FreeLibrary(h);
            continue;
        }
        // The runtime writes the WHOLE hipDeviceProp_t, and DevProp0600 above stops at gcnArchName
        // because that is the last field anything here reads. Handing it a DevProp0600 on the stack
        // therefore lets it write past the end - which segfaulted the launcher while the standalone
        // probe survived on stack-layout luck. Give it a buffer far larger than the struct can be,
        // so the size of the tail never matters; the "gfx" check below still catches a layout that
        // moved gcnArchName itself.
        std::vector<unsigned char> propbuf(16384, 0);
        if (props(reinterpret_cast<DevProp0600*>(propbuf.data()), 0) != 0) {
            if (verbose) fwprintf(stderr, L"[detect] %ls: properties failed\n", fam.dir);
            FreeLibrary(h);
            continue;
        }
        const DevProp0600& p = *reinterpret_cast<const DevProp0600*>(propbuf.data());

        const std::string arch = baseArch(p.gcnArchName);
        if (verbose) fwprintf(stderr, L"[detect] %ls: %d device(s), arch %ls\n",
                              fam.dir, n, widen(arch).c_str());

        // DevProp0600 is transcribed from hip_runtime_api.h, and a future runtime that moved a
        // field would have gcnArchName read some unrelated bytes. Every AMD arch name starts with
        // "gfx", so anything else means the layout no longer matches - fail loudly rather than pick
        // a payload out of garbage, which is the exact failure this exists to prevent.
        if (arch.rfind("gfx", 0) != 0) {
            fwprintf(stderr, L"[detect] %ls: arch name \"%ls\" is not a gfx target - the runtime's "
                             L"device-property layout no longer matches this build; refusing to "
                             L"guess\n", fam.dir, widen(arch).c_str());
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
        if (verbose) fwprintf(stderr, L"[detect] %ls: no payload for %ls\n",
                              fam.dir, widen(arch).c_str());
        FreeLibrary(h);
        // Fall through: another family may carry this chip.
    }
    return false;
}

// Run "<root>\cheshire-detect.exe --family <n> <root>" and read its one line of stdout.
// Returns true and fills `line` when the child exits 0 with output. A child that crashes, hangs or
// exits non-zero is simply "this family did not answer".
static bool cheshireAskChild(const std::wstring& root, int famIndex, bool verbose, std::wstring* line) {
    const std::wstring exe = root + L"\\cheshire-detect.exe";
    if (GetFileAttributesW(exe.c_str()) == INVALID_FILE_ATTRIBUTES) return false;

    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    std::wstring cmd = L"\"" + exe + L"\" --family " + std::to_wstring(famIndex) + L" \"" + root + L"\"";
    if (verbose) cmd += L" -v";   // so the child says why, not just whether
    std::vector<wchar_t> buf(cmd.begin(), cmd.end());
    buf.push_back(0);

    STARTUPINFOW si{};
    si.cb = sizeof si;
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = wr;
    // The child's stderr goes to ours when verbose, and to nothing otherwise, so a crash message
    // from a vendor runtime does not land in the middle of a Meshroom node's log.
    si.hStdError = verbose ? GetStdHandle(STD_ERROR_HANDLE) : nullptr;
    si.hStdInput = nullptr;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), buf.data(), nullptr, nullptr, TRUE,
                        CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) {
        CloseHandle(rd); CloseHandle(wr);
        return false;
    }
    CloseHandle(wr);   // our copy, so the read below ends when the child exits

    std::string out;
    char chunk[512];
    DWORD got = 0;
    while (ReadFile(rd, chunk, sizeof chunk, &got, nullptr) && got) out.append(chunk, got);
    CloseHandle(rd);

    // A runtime that wedges must not wedge the caller: a node run would hang with no output.
    DWORD code = 1;
    if (WaitForSingleObject(pi.hProcess, 60000) == WAIT_TIMEOUT) {
        if (verbose) fwprintf(stderr, L"[detect] family %d: timed out, killed\n", famIndex);
        TerminateProcess(pi.hProcess, 1);
    }
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);

    if (code != 0) {
        if (verbose) {
            // 0xC0000005 and friends: the runtime took the child down. That is a normal answer
            // here, not an error to report - it means this family cannot drive this card.
            fwprintf(stderr, L"[detect] family %d: no answer (exit 0x%08lX)\n", famIndex, code);
        }
        return false;
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    if (out.empty()) return false;
    *line = std::wstring(out.begin(), out.end());
    return true;
}

// Choose the payload for the card in this machine, one family per child process. Returns true and
// fills family/target, or false if no family answered.
//
// Isolation is the point: a HIP runtime built for a driver that is not installed can crash on load
// instead of reporting no device, and the family that would have worked is the one after it. In
// this process that is fatal; in a child it is just a non-zero exit.
static bool cheshireSelectPayload(const std::wstring& root, bool verbose,
                                  std::wstring* family, std::wstring* target,
                                  bool* layoutMismatch = nullptr) {
    if (layoutMismatch) *layoutMismatch = false;
    wchar_t full[MAX_PATH * 4];
    const DWORD fn = GetFullPathNameW(root.c_str(), MAX_PATH * 4, full, nullptr);
    const std::wstring base = (fn > 0 && fn < MAX_PATH * 4) ? std::wstring(full, fn) : root;

    const int n = (int)(sizeof(kFamilies) / sizeof(kFamilies[0]));
    bool spawned = false;
    for (int fi = 0; fi < n; ++fi) {
        std::wstring line;
        if (!cheshireAskChild(base, fi, verbose, &line)) continue;
        spawned = true;
        const size_t sp = line.find(L' ');
        if (sp == std::wstring::npos) continue;
        *family = line.substr(0, sp);
        *target = line.substr(sp + 1);
        return true;
    }
    // No child produced an answer. Either none could, or cheshire-detect.exe is not beside us - in
    // which case fall back to probing here, accepting the crash risk rather than refusing to run.
    if (!spawned && GetFileAttributesW((base + L"\\cheshire-detect.exe").c_str()) == INVALID_FILE_ATTRIBUTES) {
        if (verbose) fwprintf(stderr, L"[detect] no cheshire-detect.exe beside the package; probing in-process\n");
        return cheshireProbeFamilies(base, verbose, -1, family, target, layoutMismatch);
    }
    return false;
}
