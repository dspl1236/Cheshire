// Cheshire: Meshroom pairing launcher for Windows.
//
// Installed by meshroom-pair.cmd under the name of a Meshroom node binary in
// <Meshroom>\aliceVision\bin (aliceVision_depthMapEstimation.exe, aliceVision_featureMatching.exe),
// with Meshroom's own binary kept beside it as <name>.cuda.exe and the Cheshire package's path in
// <name>.cheshire.txt. Meshroom runs its nodes as `aliceVision_<node> {allParams}` through a shell,
// so they land here; the launcher takes its role from its own file name.
//
// Per run: NVIDIA present (nvidia-smi answers) -> Meshroom's binary (CUDA DepthMap, CPU matcher);
// otherwise the Cheshire build, with the package's bin first on PATH and ALICEVISION_ROOT pointing
// at the package so it finds its own DLLs and share/ tree, not Meshroom's.
// CHESHIRE_BACKEND=auto|cheshire|meshroom forces one for every paired binary; CHESHIRE_DEPTHMAP=cuda|hip
// is the older spelling and still works. Meshroom 2023.3 still passes --sgmFilteringAxes to DepthMap,
// which upstream AliceVision removed (the Cheshire build is based on 2026 upstream and always filters
// YX): dropped on the Cheshire path, kept on Meshroom's.
//
// Build: scripts\windows\build-launcher.cmd (clang-cl, static CRT). Plain Win32 CreateProcessW,
// exit code passed through.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
// Payload selection for a bundled package, shared with cheshire-detect.exe so the two cannot
// disagree about which payload a card gets. Brings in windows.h itself, and exeDir/dirExists.
#include "cheshire-gpu-select.h"
#include <string>
#include <vector>
#include <cstdio>
#include <cwchar>

static std::wstring exePath() {
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 4);
    return std::wstring(buf, n);
}
// exeDir() and dirExists() come from cheshire-gpu-select.h.
// "aliceVision_depthMapEstimation" from ...\aliceVision_depthMapEstimation.exe
static std::wstring exeStem() {
    const std::wstring p = exePath();
    size_t i = p.find_last_of(L"\\/");
    std::wstring name = i == std::wstring::npos ? p : p.substr(i + 1);
    size_t d = name.find_last_of(L'.');
    return d == std::wstring::npos ? name : name.substr(0, d);
}

static std::wstring readLine(const std::wstring& path) {
    FILE* f = _wfopen(path.c_str(), L"rt, ccs=UTF-8");
    if (!f) return L"";
    wchar_t line[MAX_PATH * 4] = {};
    if (!fgetws(line, MAX_PATH * 4, f)) { fclose(f); return L""; }
    fclose(f);
    std::wstring s(line);
    while (!s.empty() && (s.back() == L'\n' || s.back() == L'\r' || s.back() == L' ')) s.pop_back();
    return s;
}

static std::wstring envVar(const wchar_t* name) {
    DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (n == 0) return L"";
    std::wstring v(n, L'\0');
    GetEnvironmentVariableW(name, &v[0], n);
    v.resize(n - 1);
    return v;
}

static std::wstring quote(const std::wstring& a) {
    if (!a.empty() && a.find_first_of(L" \t\"") == std::wstring::npos) return a;
    std::wstring q = L"\"";
    for (size_t i = 0; i < a.size(); ++i) {
        size_t bs = 0;
        while (i < a.size() && a[i] == L'\\') { ++bs; ++i; }
        if (i == a.size()) { q.append(bs * 2, L'\\'); break; }
        if (a[i] == L'"') { q.append(bs * 2 + 1, L'\\'); q += L'"'; }
        else { q.append(bs, L'\\'); q += a[i]; }
    }
    return q + L"\"";
}

static int run(const std::wstring& exe, const std::vector<std::wstring>& args) {
    std::wstring cmd = quote(exe);
    for (const auto& a : args) { cmd += L' '; cmd += quote(a); }
    std::vector<wchar_t> buf(cmd.begin(), cmd.end()); buf.push_back(0);
    STARTUPINFOW si{}; si.cb = sizeof si;
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(exe.c_str(), buf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
        fwprintf(stderr, L"[cheshire] cannot start %ls (error %lu)\n", exe.c_str(), GetLastError());
        return 127;
    }
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    return int(code);
}

static bool nvidiaPresent() {
    // nvidia-smi lives in System32 once the NVIDIA driver is installed; a clean exit means a card answered.
    STARTUPINFOW si{}; si.cb = sizeof si; si.dwFlags = STARTF_USESTDHANDLES;   // no inherited stdio: keep the node log clean
    PROCESS_INFORMATION pi{};
    wchar_t cmd[] = L"nvidia-smi.exe -L";
    if (!CreateProcessW(nullptr, cmd, nullptr, nullptr, FALSE, CREATE_NO_WINDOW, nullptr, nullptr, &si, &pi)) return false;
    WaitForSingleObject(pi.hProcess, 20000);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    return code == 0;
}

int wmain() {
    int argc = 0;
    wchar_t** argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    std::vector<std::wstring> args(argv + 1, argv + argc);
    const std::wstring dir = exeDir(), stem = exeStem();
    const std::wstring orig = dir + L"\\" + stem + L".cuda.exe";
    const std::wstring pkg = readLine(dir + L"\\" + stem + L".cheshire.txt");
    const bool isDepthMap = stem == L"aliceVision_depthMapEstimation";
    // Which binary runs: Meshroom's own, or the paired Cheshire package's. CHESHIRE_BACKEND decides -
    // cheshire, meshroom, or auto. auto (the default) hands the node back to Meshroom whenever an
    // NVIDIA card is present, which was the right default while Cheshire was AMD-only and is exactly
    // wrong when the paired package is itself a CUDA build, so a CUDA package must be asked for.
    //
    // CHESHIRE_DEPTHMAP=cuda|hip is the older spelling and is still honoured. It named the two
    // backends at a time when "hip" and "the Cheshire package" were the same thing; they are not any
    // more, which is why the switch is no longer named after either a node or a GPU vendor.
    std::wstring mode = envVar(L"CHESHIRE_BACKEND");
    if (mode.empty()) {
        const std::wstring legacy = envVar(L"CHESHIRE_DEPTHMAP");
        if (legacy == L"cuda")     mode = L"meshroom";
        else if (legacy == L"hip") mode = L"cheshire";
        else                       mode = legacy;
    }
    if (mode.empty()) mode = L"auto";
    if (mode != L"auto" && mode != L"cheshire" && mode != L"meshroom")
        fwprintf(stderr, L"[cheshire] %ls: CHESHIRE_BACKEND=%ls is not one of auto|cheshire|meshroom;"
                         L" using the Cheshire package\n", stem.c_str(), mode.c_str());
    const bool useOrig = mode == L"meshroom" || (mode == L"auto" && nvidiaPresent());
    if (useOrig) {
        fwprintf(stderr, L"[cheshire] %ls: Meshroom's own binary (%ls)\n", stem.c_str(), orig.c_str());
        return run(orig, args);
    }
    if (pkg.empty()) {
        fwprintf(stderr, L"[cheshire] %ls: no Cheshire package path in %ls\\%ls.cheshire.txt -"
                         L" re-run meshroom-pair.cmd\n", stem.c_str(), dir.c_str(), stem.c_str());
        return 2;
    }
    // Two package shapes. A flat package has bin/ at its root and one GPU target compiled in. A
    // bundle (docs/16) has gpu/<family>/<target>/ and picks the payload from the card, so the
    // layered PATH has to be composed per run. Tell them apart by the gpu/ directory rather than by
    // a setting, so an existing paired install keeps working untouched.
    std::wstring pkgExe = pkg + L"\\bin\\" + stem + L".exe";
    std::wstring root = pkg, path = pkg + L"\\bin";
    if (dirExists(pkg + L"\\gpu")) {
        std::wstring fam, tgt;
        bool layoutMismatch = false;
        if (!cheshireSelectPayload(pkg, false, &fam, &tgt, &layoutMismatch)) {
            fwprintf(stderr, layoutMismatch
                ? L"[cheshire] %ls: the HIP runtime's device-property layout is not the one this "
                  L"launcher was built against; refusing to guess a payload\n"
                : L"[cheshire] %ls: no AMD GPU with a matching payload in %ls\\gpu "
                  L"(cheshire-detect.exe -v explains)\n",
                stem.c_str(), pkg.c_str());
            return 2;
        }
        // The GPU directory first: its five DLLs must win over the same names behind them.
        const std::wstring famBin = pkg + L"\\fam\\" + fam + L"\\bin";
        root = pkg + L"\\common";
        path = pkg + L"\\gpu\\" + fam + L"\\" + tgt + L";" + pkg + L"\\gpu\\" + fam + L";"
             + famBin + L";" + pkg + L"\\common\\bin";
        // A bundle carrying one family has nothing to differ against, so everything lands in common.
        pkgExe = famBin + L"\\" + stem + L".exe";
        if (GetFileAttributesW(pkgExe.c_str()) == INVALID_FILE_ATTRIBUTES)
            pkgExe = pkg + L"\\common\\bin\\" + stem + L".exe";
        fwprintf(stderr, L"[cheshire] %ls: bundle payload %ls/%ls\n", stem.c_str(), fam.c_str(), tgt.c_str());
    }
    SetEnvironmentVariableW(L"ALICEVISION_ROOT", root.c_str());
    SetEnvironmentVariableW(L"PATH", (path + L";" + envVar(L"PATH")).c_str());
    std::vector<std::wstring> kept;
    for (size_t i = 0; i < args.size(); ++i) {
        if (isDepthMap && args[i] == L"--sgmFilteringAxes") { ++i; continue; }   // removed upstream; YX is the only behaviour
        kept.push_back(args[i]);
    }
    fwprintf(stderr, L"[cheshire] %ls: Cheshire build (%ls)\n", stem.c_str(), pkgExe.c_str());
    return run(pkgExe, kept);
}
