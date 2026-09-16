// Cheshire: Meshroom pairing launcher for Windows.
//
// Installed by meshroom-pair.cmd as <Meshroom>\aliceVision\bin\aliceVision_depthMapEstimation.exe,
// with the original CUDA binary kept beside it as aliceVision_depthMapEstimation.cuda.exe and the
// HIP package's path in aliceVision_depthMapEstimation.cheshire.txt. Meshroom runs the DepthMap
// node as `aliceVision_depthMapEstimation {allParams}` through a shell, so it lands here.
//
// Per run: NVIDIA present (nvidia-smi answers) -> the CUDA binary; otherwise the HIP build, with
// the Cheshire package's bin first on PATH and ALICEVISION_ROOT pointing at the package so it finds
// its own DLLs and share/ tree, not Meshroom's. CHESHIRE_DEPTHMAP=cuda|hip forces one. Meshroom
// 2023.3 still passes --sgmFilteringAxes, which upstream AliceVision removed (the HIP build is based
// on 2026 upstream and always filters YX): dropped on the HIP path, kept on the CUDA path.
//
// Build (from scripts\env.cmd): clang-cl /O2 /EHsc meshroom-pair-launcher.cpp /Fe:launcher.exe
// No dependencies beyond kernel32/shell32: plain Win32 CreateProcessW, exit code passed through.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#include <string>
#include <vector>
#include <cstdio>
#include <cwchar>

static std::wstring exeDir() {
    wchar_t buf[MAX_PATH * 4];
    DWORD n = GetModuleFileNameW(nullptr, buf, MAX_PATH * 4);
    std::wstring p(buf, n);
    size_t i = p.find_last_of(L"\\/");
    return i == std::wstring::npos ? L"." : p.substr(0, i);
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
    const std::wstring dir = exeDir();
    const std::wstring cuda = dir + L"\\aliceVision_depthMapEstimation.cuda.exe";
    const std::wstring pkg = readLine(dir + L"\\aliceVision_depthMapEstimation.cheshire.txt");
    std::wstring mode = envVar(L"CHESHIRE_DEPTHMAP");
    if (mode.empty()) mode = L"auto";
    const bool useCuda = mode == L"cuda" || (mode == L"auto" && nvidiaPresent());
    if (useCuda) {
        fwprintf(stderr, L"[cheshire] DepthMap backend: CUDA (%ls)\n", cuda.c_str());
        return run(cuda, args);
    }
    if (pkg.empty()) {
        fwprintf(stderr, L"[cheshire] no NVIDIA card and no Cheshire package path in %ls\\aliceVision_depthMapEstimation.cheshire.txt\n", dir.c_str());
        return 2;
    }
    const std::wstring hip = pkg + L"\\bin\\aliceVision_depthMapEstimation.exe";
    SetEnvironmentVariableW(L"ALICEVISION_ROOT", pkg.c_str());
    SetEnvironmentVariableW(L"PATH", (pkg + L"\\bin;" + envVar(L"PATH")).c_str());
    std::vector<std::wstring> kept;
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == L"--sgmFilteringAxes") { ++i; continue; }   // removed upstream; YX is the only behaviour
        kept.push_back(args[i]);
    }
    fwprintf(stderr, L"[cheshire] DepthMap backend: HIP (%ls)\n", hip.c_str());
    return run(hip, kept);
}
