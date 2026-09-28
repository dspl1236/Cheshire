# Task: 0.3.6 tooling debts (scripts only, no GPU needed)

The work is scripts only. The owner runs the hardware gates after merge, so every change here must
be checkable without a GPU: `python -m py_compile`, `bash -n`, and a dry run or unit-style test
where one is possible. Keep each fix small and put its reason in a comment next to it, in the
repository's style: plain sentences, the measurement or incident that motivated it, `file:line`
references. The items come from docs/roadmap.md, "Packaging and platforms".

## 1. End-to-end gate gaps (`scripts/verify_end_to_end.py`)

- **`cpufallback` never forces CPU SIFT.** It turns every `CHESHIRE_GPU_*` switch off, but its
  overrides keep `FeatureExtraction:forceCpuExtraction=False`, so feature extraction stays on the
  GPU. Add `FeatureExtraction:forceCpuExtraction=True`. Its markers then need FeatureExtraction
  without the GPU SIFT lines: see `DEFAULT_MARKERS`, which does the same for the `defaults`
  configuration.
- **PrepareDenseScene is in neither `BINARY` nor `GPU_MARKERS`**, so a run whose PrepareDenseScene
  is Meshroom's own binary passes unnoticed.
  - Add it to `BINARY`. The launcher prints `[cheshire] aliceVision_prepareDenseScene: Cheshire
    build (...)` or `...: Meshroom's own binary`.
  - Give it a port marker taken from the success branch of the direct 8-bit read (step 6n in
    `scripts/apply_hip_patch.py`). Follow the marker rule at the top of the file: the line must not
    also match the disabled branch.
- **No check that SfM's "edges skipped" warning is absent.** Step 5k in `scripts/apply_hip_patch.py`
  makes the local-BA graph skip an edge it cannot hold instead of throwing, and logs a warning when
  it does. A gate run that hits it should say so. Treat the warning as an unmet check in every
  configuration.
- **A never-paired node is caught only at the end.** `main()` runs the pairing script and prints
  its output. If any expected node reports "not paired", fail fast, before `meshroom_batch` starts,
  and name the node and the script's reason. Today `wrong_binary()` only notices Meshroom's own
  binary once that node has run.

## 2. `CHESHIRE_POPSIFT` defaults to OFF in the HIP build scripts

See `scripts/build-alicevision.cmd:16` and `scripts/linux/build-alicevision.sh:25`. The same trap
shipped a CPU-SIFT CUDA zip in v0.3.2. Make the default auto: ON when a PopSift install is found
where the scripts look for one (`CHESHIRE_POPSIFT_INSTALL`, or the default install paths the scripts
already use), OFF with a printed warning otherwise. An explicit `CHESHIRE_POPSIFT=ON|OFF` still wins.

Then add the packaging check the CUDA packager already has
(`scripts/windows/package-cuda.ps1`, "GPU SIFT: aliceVision_feature.dll imports popsift.dll") to:

- `scripts/package_windows.py`: read the import table with the `llvm-objdump` it is given;
- `scripts/linux/wsl-pack-bundle.sh`: `readelf -d` on `libaliceVision_feature.so*`, whose NEEDED
  entries must include `libpopsift`.

A package without GPU SIFT should fail packaging, not ship.

## 3. `scripts/linux/meshroom-pair.sh`: `CHESHIRE_BACKEND` set in the bundle's `env.sh` never takes effect

The generated launcher decides the backend (lines around 63-68, `CHESHIRE_MODE`) before it sources
the bundle's `env.sh` (around line 80). Source `env.sh` first, when it exists, so a backend set
there is honoured. Keep the per-run decision: swapping cards must not need re-pairing. Check the
generated launcher text with `bash -n` after writing one to a temporary directory.

## 4. `scripts/verify_packages.py`: retire it

It stops at v0.2.16 with five markers (`:21-32`), and `verify_end_to_end.py` plus the stage gates
replaced it. Remove it. Replace the references to it: `grep -rn verify_packages` in scripts/, docs/
and README.md; docs/04 mentions it for step 10. Say in docs/04's newest section that it was retired
and why.

## Done means

- Each item is its own commit, and the message says what changed and how it was checked.
- `python -m py_compile scripts/*.py scripts/*/*.py` and `bash -n scripts/linux/*.sh` pass.
- A short note at the end of docs/04 lists what changed, for the owner's next gate run.
- Do not change anything under `hip/`, `third_party/`, or the generator's port steps.
