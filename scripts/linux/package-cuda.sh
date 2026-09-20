#!/usr/bin/env bash
# Package the Linux CUDA bundle for release, with the checks that matter for a CUDA package.
#
# Naming follows the HIP releases: cheshire-alicevision-<backend>-linux-x64-<runtime>.tar.gz
#
# The checks are not decoration. Each one corresponds to a way a bundle has actually gone out
# wrong, or nearly has:
#   * libcuda must NOT be present - it is the driver library and belongs to the host. Shipping the
#     toolkit's link stub gives a package that loads and then fails on the first API call.
#   * libcudart MUST be present, or the package depends on a toolkit the user does not have.
#   * libpopsift must be a CUDA build. Every one on the dev machines was a HIP build for months
#     (they declare libamdhip64), which is why GPU SIFT was off entirely.
#   * no libamdhip64 anywhere - that would mean a HIP artefact leaked into a CUDA package.
set -uo pipefail
BUNDLE="${1:-/opt/AliceVision_cuda/bundle}"
OUT="${2:-$PWD}"
CUDA_VER="${CUDA_VER:-12.9}"
NAME="cheshire-alicevision-cuda-linux-x64-cuda${CUDA_VER}"

[ -d "$BUNDLE/bin" ] || { echo "no bundle at $BUNDLE"; exit 1; }
fail=0
note() { printf "  %-6s %s\n" "$1" "$2"; [ "$1" = "FAIL" ] && fail=1; return 0; }

echo "=== checking $BUNDLE"
printf "  %-6s %s\n" "info" "$(ls "$BUNDLE/bin" | wc -l) binaries, $(du -sh "$BUNDLE" | cut -f1)"

if ls "$BUNDLE"/lib/libcuda.so* >/dev/null 2>&1; then
  note FAIL "libcuda.so is in the bundle - it belongs to the host driver, remove it"
else
  note ok "no libcuda (correct: the host driver provides it)"
fi

if ls "$BUNDLE"/lib/libcudart.so* >/dev/null 2>&1; then
  note ok "libcudart present: $(basename "$(ls "$BUNDLE"/lib/libcudart.so.*.* 2>/dev/null | head -1)")"
else
  note FAIL "no libcudart - the package would need a toolkit installed"
fi

P=$(ls "$BUNDLE"/lib/libpopsift.so* 2>/dev/null | head -1)
if [ -z "$P" ]; then
  note WARN "no libpopsift - GPU SIFT will fall back to the CPU extractor"
elif ldd "$P" 2>/dev/null | grep -qi amdhip; then
  note FAIL "libpopsift is a HIP build (declares libamdhip64) - wrong backend"
else
  note ok "libpopsift is a CUDA build"
fi

if find "$BUNDLE" -name "*amdhip*" -o -name "*rocm*" 2>/dev/null | grep -q .; then
  note FAIL "HIP/ROCm artefacts leaked into a CUDA bundle:"
  find "$BUNDLE" -name "*amdhip*" -o -name "*rocm*" 2>/dev/null | head -3 | sed 's/^/         /'
else
  note ok "no HIP/ROCm artefacts"
fi

# anything the loader cannot resolve from inside the bundle is a missing dependency for the user
UNRES=$(cd "$BUNDLE" && LD_LIBRARY_PATH="$PWD/lib" ldd bin/aliceVision_depthMapEstimation 2>/dev/null | grep -c "not found")
if [ "${UNRES:-0}" -eq 0 ]; then
  note ok "depthMapEstimation resolves every dependency from the bundle"
else
  note FAIL "$UNRES unresolved dependencies in depthMapEstimation"
  (cd "$BUNDLE" && LD_LIBRARY_PATH="$PWD/lib" ldd bin/aliceVision_depthMapEstimation 2>/dev/null | grep "not found" | head -4 | sed 's/^/         /')
fi

[ "$fail" -ne 0 ] && { echo; echo "not packaging: fix the FAILs above"; exit 1; }

echo
echo "=== packaging"
TARBALL="$OUT/$NAME.tar.gz"
tar czf "$TARBALL" -C "$(dirname "$BUNDLE")" "$(basename "$BUNDLE")"
echo "  $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo "  sha256: $(sha256sum "$TARBALL" | cut -d' ' -f1)"
