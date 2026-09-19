// Cheshire: print which bundled payload this machine's card needs (docs/16).
//
//   cheshire-detect.exe [bundle root] [-v]
//
// Prints "<family> <target>" (e.g. "rocm7.2 gfx12-generic") and exits 0; exits 1 if no family
// enumerated a device it has a payload for, or 2 if a runtime answered but its device-property
// layout no longer matches what this was built against.
//
// The selection itself lives in cheshire-gpu-select.h, shared with meshroom-pair-launcher.exe so
// the two cannot disagree about which payload a card should get.
//
// Build: scripts\windows\build-launcher.cmd (clang-cl, static CRT).
#include "cheshire-gpu-select.h"

int wmain(int argc, wchar_t** argv) {
    bool verbose = false;
    std::wstring root;
    for (int i = 1; i < argc; ++i) {
        if (!wcscmp(argv[i], L"-v")) verbose = true;
        else root = argv[i];
    }
    if (root.empty()) root = exeDir();

    std::wstring family, target;
    bool layoutMismatch = false;
    if (cheshireSelectPayload(root, verbose, &family, &target, &layoutMismatch)) {
        wprintf(L"%ls %ls\n", family.c_str(), target.c_str());
        return 0;
    }
    if (layoutMismatch) return 2;
    fwprintf(stderr, L"[detect] no AMD GPU with a matching payload under %ls\\gpu\n", root.c_str());
    return 1;
}
