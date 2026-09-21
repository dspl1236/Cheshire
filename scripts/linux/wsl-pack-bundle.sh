#!/usr/bin/env bash
# Pack a Linux HIP bundle built on the WSL box into the release tarball, after the checks that
# have each caught a bad bundle before.
#
#   scripts/linux/wsl-pack-bundle.sh <install prefix> <out.tar.gz>
#   e.g.  wsl-pack-bundle.sh /opt/AliceVision_hip_031 /mnt/d/MMI/cheshire/build/release/0.3.1/cheshire-alicevision-hip-linux-x64-rocm7.2.tar.gz
#
# Promoted from build/wsl_pack_release_*.sh, which was gitignored - the only record of how the
# Linux artifact was packed lived outside the repository.
#
#  * The HSA runtime. A WSL build box carries the WSL flavour of libhsa-runtime64 (it probes
#    /dev/dxg, never /dev/kfd, and is under 3 MB); a bundle shipping it fails hsa_init with
#    OUT_OF_RESOURCES on a real Linux box (docs/06). build-alicevision.sh swaps it at bundle time;
#    this checks the swap took and does it again from the standard runtime if not.
#  * PopSift must be present, be the HIP one, and carry code objects - the pairing script declines
#    GPU SIFT silently without it, and v0.3.0's CUDA bundle shipped with a name the script did not
#    match.
#  * The tarball root is bundle/, which everything downstream (house-pc's tooling, meshroom-pair.sh)
#    expects.
set -u
PREFIX="${1:?install prefix, e.g. /opt/AliceVision_hip_031}"
OUT="${2:?output tarball path}"
B="$PREFIX/bundle"
STD="${CHESHIRE_HSA_RUNTIME:-/mnt/d/MMI/cheshire/build/libhsa-runtime64.so.1.18.70200}"
[ -d "$B/bin" ] || { echo "no bundle at $B"; exit 1; }
fail=0
note() { printf "  %-6s %s\n" "$1" "$2"; [ "$1" = FAIL ] && fail=1; return 0; }

echo "=== checking $B"
rt=$(ls "$B"/lib/libhsa-runtime64.so.1.* 2>/dev/null | grep -v '\.so\.1$' | head -1)
if [ -z "$rt" ]; then
  note FAIL "no libhsa-runtime64 in the bundle"
elif grep -a -q "/dev/dxg" "$rt" && [ "$(stat -c %s "$rt")" -lt 3000000 ]; then
  if [ -f "$STD" ]; then
    cp -f "$STD" "$rt" && ln -sfn "$(basename "$rt")" "$B/lib/libhsa-runtime64.so.1"
    note ok "HSA runtime was the WSL one; replaced with $(basename "$STD") ($(stat -c %s "$rt") bytes)"
  else
    note FAIL "HSA runtime is the WSL one and no standard runtime at $STD to swap in"
  fi
else
  note ok "HSA runtime is the standard one ($(basename "$rt"), $(stat -c %s "$rt") bytes)"
fi

ps="$B/lib/libpopsift.so"
if [ ! -f "$ps" ]; then
  note FAIL "no libpopsift.so - GPU SIFT will not pair"
elif readelf -d "$ps" 2>/dev/null | grep -q cudart; then
  note FAIL "libpopsift.so needs libcudart: that is the CUDA PopSift in a HIP bundle"
else
  n=$(grep -ac 'amdhsa--' "$ps")
  [ "$n" -gt 0 ] && note ok "libpopsift.so is HIP, $n code-object markers, $(stat -c %s "$ps") bytes" \
                 || note FAIL "libpopsift.so carries no HIP code objects"
fi

dm="$B/lib/libaliceVision_depthMap_cuda.so.3.4"
if [ -f "$dm" ]; then
  note ok "depthMap: amdhsa=$(grep -ac 'amdhsa--' "$dm") clamp=$(grep -ac 'exceeds this device' "$dm") ($(stat -c %s "$dm") bytes)"
else
  note FAIL "no $dm"
fi
# Anything in the bundle that needs the CUDA runtime, or the CUDA PopSift's versioned soname, is a
# CUDA object in a HIP bundle. Every ELF file, not just the featureExtraction executable: the CUDA
# linkage from the 2026-09-20 poisoned tree sat in libaliceVision_feature.so (NEEDED
# libpopsift.so.0.10.0), and the first version of this check read only the executable - which
# needs neither - and passed a contaminated bundle on its first run.
bad_elf=$(for f in "$B"/bin/* "$B"/lib/*.so*; do
  [ -f "$f" ] && [ ! -L "$f" ] || continue
  readelf -d "$f" 2>/dev/null | grep -qE 'NEEDED.*(libcudart|libpopsift\.so\.[0-9])' && basename "$f"
done | tr '\n' ' ')
[ -z "$bad_elf" ] && note ok "no ELF in the bundle needs libcudart or a versioned libpopsift" \
                  || note FAIL "CUDA linkage in a HIP bundle: $bad_elf"
cg=$(ls "$B"/lib/libamd_comgr.so.* 2>/dev/null | head -1)
[ -n "$cg" ] && note ok "comgr: $(basename "$cg")" || note FAIL "no libamd_comgr (HIP dlopens it; hipGetDeviceCount finds no GPU without it)"

[ "$fail" = 1 ] && { echo "=== not packing"; exit 1; }
echo "=== packing -> $OUT"
mkdir -p "$(dirname "$OUT")"
rm -f "$OUT"
tar -C "$PREFIX" -czf "$OUT" bundle
printf "  %s  (%d MB)\n" "$OUT" "$(( $(stat -c %s "$OUT") / 1048576 ))"
echo "  sha256: $(sha256sum "$OUT" | cut -c1-64)"
