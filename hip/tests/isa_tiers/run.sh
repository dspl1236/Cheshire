#!/usr/bin/env bash
# Build eigen_tiers.cpp at x86-64 v1, v2, v3 (contraction off) and v3 (gcc's default contraction), run each, and say which
# results have the v1 build's bits. Run in the WSL build box: bash run.sh [eigen include dir]
set -u
cd "$(dirname "$0")"
E="${1:-/opt/AliceVision_deps/include/eigen3}"
for t in "v1|-march=x86-64" "v2|-march=x86-64-v2" "v3|-march=x86-64-v3 -ffp-contract=off" "v3c|-march=x86-64-v3" "v3p|-march=x86-64-v3 -ffp-contract=off -U__AVX__ -U__AVX2__ -U__FMA__ -U__AVX512F__"; do
  n="${t%%|*}"; f="${t#*|}"
  g++ -O2 -std=c++17 $f -I"$E" eigen_tiers.cpp -o "/tmp/et_$n" && "/tmp/et_$n" > "/tmp/et_$n.txt" || echo "build/run $n failed"
done
printf '%-30s %-6s %-6s %-15s %s\n' result v2 v3 "v3, contracted" "v3, Eigen at SSE width"
paste /tmp/et_v1.txt /tmp/et_v2.txt /tmp/et_v3.txt /tmp/et_v3c.txt /tmp/et_v3p.txt | while IFS=$'\t' read -r a b c d e; do
  name="${a% *}"; ha="${a##* }"
  s() { [ "${1##* }" = "$ha" ] && echo same || echo DIFF; }
  printf '%-30s %-6s %-6s %-15s %s\n' "$name" "$(s "$b")" "$(s "$c")" "$(s "$d")" "$(s "$e")"
done
g++ --version | head -1
