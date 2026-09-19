// Cheshire: print which bundled payload this machine's card needs (docs/16).
//
//   cheshire-detect.exe [bundle root] [-v]        choose a payload
//   cheshire-detect.exe --family <n> <root>       probe one family only (used internally)
//
// Prints "<family> <target>" (e.g. "rocm7.2 gfx12-generic") and exits 0; exits 1 if no family
// enumerated a device it has a payload for, or 2 if a runtime answered but its device-property
// layout no longer matches what this was built against.
//
// The --family form is what makes the normal form safe. A HIP runtime built for a driver that is
// not installed can crash on load rather than report no device - on an RX 6750 XT under a
// scheduled task, loading amdhip64_7.dll exits 0xC0000005 before printing anything, while the same
// call over an interactive session politely says "no device". So the default form spawns this
// program once per family, and a family that takes its child down is simply one that did not
// answer. The selection itself lives in cheshire-gpu-select.h, shared with
// meshroom-pair-launcher.exe so the two cannot disagree about which payload a card should get.
//
// Build: scripts\windows\build-launcher.cmd (clang-cl, static CRT).
#include "cheshire-gpu-select.h"
#include <cstdlib>

int wmain(int argc, wchar_t** argv) {
    bool verbose = false;
    int onlyFamily = -1;
    std::wstring root;
    for (int i = 1; i < argc; ++i) {
        if (!wcscmp(argv[i], L"-v")) verbose = true;
        else if (!wcscmp(argv[i], L"--family") && i + 1 < argc) onlyFamily = _wtoi(argv[++i]);
        else root = argv[i];
    }
    if (root.empty()) root = exeDir();

    std::wstring family, target;
    bool layoutMismatch = false;

    if (onlyFamily >= 0) {
        // Child: one family, in this process, and whatever happens happens to this process only.
        if (cheshireProbeFamilies(root, verbose, onlyFamily, &family, &target, &layoutMismatch)) {
            wprintf(L"%ls %ls\n", family.c_str(), target.c_str());
            return 0;
        }
        return layoutMismatch ? 2 : 1;
    }

    if (cheshireSelectPayload(root, verbose, &family, &target, &layoutMismatch)) {
        wprintf(L"%ls %ls\n", family.c_str(), target.c_str());
        return 0;
    }
    if (layoutMismatch) return 2;
    fwprintf(stderr, L"[detect] no AMD GPU with a matching payload under %ls\\gpu\n", root.c_str());
    return 1;
}
