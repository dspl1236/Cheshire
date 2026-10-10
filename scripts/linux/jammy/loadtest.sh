#!/usr/bin/env bash
# Cheshire 0.5.0: does a Linux bundle load on a system with nothing but the base OS? Extracts the tarball and, in a
# fresh container of the given image (default ubuntu:22.04, glibc 2.35; no ROCm, no CUDA, no compiler), runs ldd on
# every ELF file - none may have a library "not found" - and starts every aliceVision_* program with --help, which
# must not die of a loader error (exit 127, or "GLIBC_... not found" / "version ... not found").
#
#   scripts/linux/jammy/loadtest.sh <bundle.tar.gz> [image]
set -u
TAR="${1:?bundle tarball}"
IMG="${2:-ubuntu:22.04}"
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
tar -xzf "$TAR" -C "$W" || { echo "extract failed"; exit 1; }
docker run --rm --network host -v "$W:/w:ro" "$IMG" bash -c '
  . /etc/os-release; echo "== $PRETTY_NAME, $(ldd --version | head -1)"
  B=/w/bundle; bad=0; n=0
  # as every user of a bundle runs it (meshroom-pair.sh, node-amd-setup.sh): its lib/ on the library path
  export LD_LIBRARY_PATH="$B/lib" ALICEVISION_ROOT="$B"
  while IFS= read -r -d "" f; do
    head -c4 "$f" | grep -q "ELF" || continue
    n=$((n+1))
    miss=$(ldd "$f" 2>&1 | grep -E "not found|version .* not found")
    [ -n "$miss" ] && { echo "  LDD $f: $miss" | head -3; bad=1; }
  done < <(find "$B" -type f -print0)
  echo "  $n ELF files checked by ldd"
  p=0
  for x in "$B"/bin/aliceVision_*; do
    [ -x "$x" ] && [ -f "$x" ] || continue
    out=$("$x" --help 2>&1 | head -40); rc=$?
    if echo "$out" | grep -q -E "GLIBC|GLIBCXX|CXXABI|version .* not found|error while loading shared libraries"; then
      echo "  RUN $(basename "$x"): $(echo "$out" | grep -m1 -E "GLIBC|not found|error while loading")"; bad=1
    fi
    p=$((p+1))
  done
  echo "  $p programs started"
  [ $bad = 0 ] && echo "  LOADS: every library resolves, every program starts" || echo "  DOES NOT LOAD"
  exit $bad
'
