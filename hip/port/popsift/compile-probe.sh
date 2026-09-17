#!/bin/bash
# First contact: compile every PopSIFT library source as HIP for gfx1201 and collect the errors.
cd /d/MMI/cheshire || exit 1
R=D:/MMI/cheshire
ROCM="$R/tools/venv-rocm/Lib/site-packages/_rocm_sdk_devel"
LLVM="$ROCM/lib/llvm/bin"
OUT="$R/build/popsift-probe"
rm -rf "$OUT"; mkdir -p "$OUT"
SRC="$R/third_party/popsift/src"
INC="-I$SRC -I$SRC/popsift -I$R/build/popsift-gen -I$R/build/popsift-gen/popsift -I$R/hip/compat/include"
FLAGS="-x hip --offload-arch=gfx1201 --rocm-path=$ROCM --rocm-device-lib-path=$ROCM/lib/llvm/amdgcn/bitcode -std=c++17 -O1 -fgpu-rdc -c -o "$OUT/$b.o" -include popsift_hip.h -I$R/hip/port/popsift -DCHESHIRE_NATIVE_MIPMAP"
ok=0; fail=0
for f in $(ls $SRC/popsift/*.cu $SRC/popsift/common/*.cu 2>/dev/null); do
  b=$(basename $f)
  if "$LLVM/clang++.exe" $FLAGS $INC "$f" > "$OUT/$b.log" 2>&1; then
    ok=$((ok+1)); echo "OK    $b"
  else
    fail=$((fail+1)); n=$(grep -c "error:" "$OUT/$b.log"); echo "FAIL  $b ($n errors)"
  fi
done
echo "=== $ok compiled, $fail failed"
echo "=== distinct error texts across all files:"
cat "$OUT"/*.log 2>/dev/null | grep -oE "error: .*" | sed 's/[0-9]\+/N/g' | sort | uniq -c | sort -rn | head -25
echo "=== popsift probe done"
