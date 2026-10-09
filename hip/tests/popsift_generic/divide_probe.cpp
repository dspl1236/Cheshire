// Packaging I item 3 (0.4.4): where does PopSIFT built for a generic code object divide by zero? docs/16 records that a
// gfx12-generic popsift.dll exits 0xC0000094 on the first photograph while the same source built for gfx1201 works.
// This drives PopSIFT directly (as ImageDescriber_SIFT_popSIFT does) with a process-wide vectored exception handler: on
// the fault it prints the faulting module and offset, then the stack (dbghelp, module + offset per frame, a symbol name
// where a PDB is found next to the module), and exits 3. Exit 0 and a feature count when it runs through.
// Build: build.cmd. Run with the popsift.dll to test first on PATH.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <dbghelp.h>
#include <popsift/popsift.h>
#include <popsift/features.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <vector>

#pragma comment(lib, "dbghelp.lib")

static void describe(DWORD64 addr)
{
    HMODULE mod = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCSTR)addr, &mod))
        GetModuleFileNameA(mod, name, MAX_PATH);
    char symBuf[sizeof(SYMBOL_INFO) + 256] = {};
    SYMBOL_INFO* sym = (SYMBOL_INFO*)symBuf;
    sym->SizeOfStruct = sizeof(SYMBOL_INFO);
    sym->MaxNameLen = 255;
    DWORD64 disp = 0;
    const bool haveSym = SymFromAddr(GetCurrentProcess(), addr, &disp, sym);
    std::printf("  %016llx  %s+0x%llx  %s%s\n", (unsigned long long)addr, name, (unsigned long long)(addr - (DWORD64)mod),
                haveSym ? sym->Name : "", haveSym ? "" : "(no symbol)");
}

static LONG WINAPI onFault(EXCEPTION_POINTERS* e)
{
    const DWORD code = e->ExceptionRecord->ExceptionCode;
    if (code != EXCEPTION_INT_DIVIDE_BY_ZERO && code != EXCEPTION_ACCESS_VIOLATION)
        return EXCEPTION_CONTINUE_SEARCH;
    std::printf("FAULT 0x%08lx in thread %lu at\n", code, GetCurrentThreadId());
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    SymInitialize(GetCurrentProcess(), nullptr, TRUE);
    describe((DWORD64)e->ExceptionRecord->ExceptionAddress);
    std::printf("stack:\n");
    CONTEXT ctx = *e->ContextRecord;
    STACKFRAME64 f = {};
    f.AddrPC.Offset = ctx.Rip; f.AddrPC.Mode = AddrModeFlat;
    f.AddrFrame.Offset = ctx.Rbp; f.AddrFrame.Mode = AddrModeFlat;
    f.AddrStack.Offset = ctx.Rsp; f.AddrStack.Mode = AddrModeFlat;
    for (int i = 0; i < 24 && StackWalk64(IMAGE_FILE_MACHINE_AMD64, GetCurrentProcess(), GetCurrentThread(), &f, &ctx, nullptr,
                                         SymFunctionTableAccess64, SymGetModuleBase64, nullptr); ++i)
        describe(f.AddrPC.Offset);
    std::fflush(stdout);
    ExitProcess(3);
    return EXCEPTION_CONTINUE_SEARCH;
}

int main(int argc, char** argv)
{
    const int maxExtrema = argc > 1 ? std::atoi(argv[1]) : 0;  // >0: AliceVision's grid filter (FilterMaxExtrema, LargestScaleFirst)
    AddVectoredExceptionHandler(1, onFault);
    const int w = 1500, h = 1000;
    std::vector<float> img(size_t(w) * h);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            img[size_t(y) * w + x] = 0.5f + 0.25f * float(((x / 37) + (y / 29)) % 2) + 0.1f * u(rng);  // edges and noise
    popsift::Config config;
    config.setNormalizationMultiplier(9);
    if (maxExtrema > 0)
    {
        config.setFilterMaxExtrema(maxExtrema);
        config.setFilterSorting(popsift::Config::LargestScaleFirst);
    }
    std::printf("PopSift, filter %d ...\n", maxExtrema);
    std::fflush(stdout);
    PopSift ps(config, popsift::Config::ExtractingMode, PopSift::FloatImages);
    std::unique_ptr<SiftJob> job(ps.enqueue(w, h, img.data()));
    std::unique_ptr<popsift::FeaturesHost> feats(job->get());
    std::printf("ran through: %d features, %d descriptors\n", feats->getFeatureCount(), feats->getDescriptorCount());
    ps.uninit();
    return 0;
}
