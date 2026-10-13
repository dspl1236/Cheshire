#!/usr/bin/env bash
# Cheshire vs stock Meshroom, end to end, on house-pc's GTX 1080 Ti (2026-10). The scratch Meshroom 2025.1 install in
# /data/tests (never the node app's), its default "photogrammetry" pipeline with Meshroom's default settings, run once
# unpaired (stock: Meshroom's own CUDA build) and once paired with the published 0.4.5 Linux CUDA package. Per set the
# stock run goes first, then Cheshire's. Each run: /data/tests/bench/runs/<set>-<arm>/ (project.mg, MeshroomCache,
# out/, batch.log, run.json with wall time and exit code). summarize.py reads them.
#
#   bench.sh <set> <photos dir> [stock|cheshire|cpu|both|all] [Meshroom paramOverrides...]
#     cpu: the Cheshire package with every GPU port but DepthMap switched off (the gate's cpufallback switches, without
#     its forced CPU feature extraction: the default describer, DSP-SIFT, is a CPU one anyway), so stock against cpu
#     shows the CPU-side work alone and cpu against cheshire the GPU ports. all = stock cpu cheshire.
#
# Stock runs until a node fails. If that node is one Cheshire can stand in for alone, the run resumes with only that
# node's binary paired: StructureFromMotion (upstream's crash on large sets, Meshroom #2344) with every other Cheshire
# SfM change switched back to upstream, so it is stock plus the fix (step 5k); DepthMap (no CUDA device). Every other
# node stays Meshroom's own. run.json records where stock stopped and what filled in.
set -u
SET="${1:?set}"; PHOTOS="${2:?photos}"; ARMS="${3:-both}"; shift 3 2>/dev/null || shift $#
OVERRIDES=("$@")
M=/data/tests/Meshroom-2025.1.0
BIN="$M/aliceVision/bin"
PAIR=/data/tests/bench/meshroom-pair.sh
PKG=/home/house/apps/cheshire/b045cuda/bundle
R=/data/tests/bench/runs
mkdir -p "$R"
[ "$ARMS" = both ] && ARMS="stock cheshire"
[ "$ARMS" = all ] && ARMS="stock cpu cheshire"
CPU_OFF="CHESHIRE_GPU_MATCHER CHESHIRE_GPU_FILTER CHESHIRE_GPU_VOTE CHESHIRE_GPU_TEX CHESHIRE_GPU_BLUR CHESHIRE_GPU_MAXFLOW CHESHIRE_GPU_VIS CHESHIRE_UNDISTORT_DEVICE"
# Cheshire's SfM changes back to upstream's behaviour, all but the pending-BA fix (USING.md, the switches table)
UPSTREAM_SFM="CHESHIRE_BA_SOLVER=ceres CHESHIRE_BA_JACOBIANS=autodiff CHESHIRE_BA_FUSED_PROJECTION=0 CHESHIRE_BA_PERSIST=0 CHESHIRE_BA_LOG_COST=1 CHESHIRE_SFM_TASK_SEED=0 CHESHIRE_SFM_FILTER_THREADS=1 CHESHIRE_ACR_BOUND=0 CHESHIRE_ACR_RESIDUALS=0 CHESHIRE_ACR_SVD_LANES=0"
binary_of() { case "$1" in StructureFromMotion) echo aliceVision_incrementalSfM ;; DepthMap) echo aliceVision_depthMapEstimation ;; *) echo "" ;; esac; }

keep_only() {  # pair, then give every node but the binaries in $1 back to Meshroom, and its DepthMap node file unless DepthMap is kept
  bash "$PAIR" "$M" "$PKG" >> "$plog" 2>&1 || return 1
  for c in "$BIN"/aliceVision_*.cuda; do
    n=$(basename "$c" .cuda)
    case " $1 " in *" $n "*) ;; *) mv -f "$c" "$BIN/$n" ;; esac
  done
  case " $1 " in *" aliceVision_depthMapEstimation "*) ;; *)
    for f in "$M"/aliceVision/share/meshroom/aliceVision/DepthMap.py "$M"/lib/meshroom/nodes/aliceVision/DepthMap.py; do
      [ -f "$f.meshroom" ] && mv -f "$f.meshroom" "$f"
    done ;;
  esac
  echo "kept paired: $1" >> "$plog"
}

for arm in $ARMS; do
  tag="$SET-$arm"; [ ${#OVERRIDES[@]} -gt 0 ] && tag="$tag-ovr"
  out="$R/$tag"; plog="$R/$tag.pair.log"; : > "$plog"
  for v in $CPU_OFF $UPSTREAM_SFM; do unset "${v%%=*}"; done
  bash "$PAIR" "$M" --unpair >> "$plog" 2>&1
  if [ "$arm" = cheshire ] || [ "$arm" = cpu ]; then
    bash "$PAIR" "$M" "$PKG" >> "$plog" 2>&1 || { echo "$tag: pairing failed" >> "$R/bench.done"; continue; }
    export CHESHIRE_BACKEND=cheshire
    if [ "$arm" = cpu ]; then for v in $CPU_OFF; do export "$v=0"; done; fi
  else
    unset CHESHIRE_BACKEND
  fi
  rm -rf "$out"; mkdir -p "$out/out"
  args=(--input "$PHOTOS" --output "$out/out" --pipeline photogrammetry --save "$out/project.mg")
  [ ${#OVERRIDES[@]} -gt 0 ] && args+=(--paramOverrides "${OVERRIDES[@]}")
  t0=$(date +%s.%N)
  "$M/meshroom_batch" "${args[@]}" > "$out/batch.log" 2>&1
  rc=$?
  wall=$(echo "$(date +%s.%N) - $t0" | bc)
  notes=""; filled=""
  pass=1
  while [ "$arm" = stock ] && [ $rc -ne 0 ] && [ $pass -le 3 ]; do
    failed=$(python3 - "$out/MeshroomCache" <<'EOF'
import json, sys
from pathlib import Path
bad = set()
for st in Path(sys.argv[1]).glob("*/*/*status"):
    try:
        if json.loads(st.read_text(errors="replace")).get("status") == "ERROR":
            bad.add(st.parent.parent.name)
    except ValueError:
        pass
print(" ".join(sorted(bad)))
EOF
)
    b=""; for n in $failed; do b="$b $(binary_of "$n")"; done; b=$(echo $b)
    if [ -z "$b" ]; then notes="$notes; stock failed at [$failed], nothing to fill in"; break; fi
    notes="$notes; stock failed at [$failed] after $(printf '%.0f' "$wall") s, rc $rc"
    # keep the failed chunks' logs, then clear every chunk that did not succeed so Meshroom computes it again
    python3 - "$out" <<'EOF'
import json, shutil, sys
from pathlib import Path
out = Path(sys.argv[1])
for st in (out / "MeshroomCache").glob("*/*/*status"):
    try:
        state = json.loads(st.read_text(errors="replace")).get("status")
    except ValueError:
        state = None
    if state != "SUCCESS":
        log = st.with_name(st.name.replace("status", "log"))
        if state == "ERROR" and log.exists():
            shutil.copy2(log, out / f"stock-failure-{st.parent.parent.name}-{log.name}")
        log.unlink(missing_ok=True)
        st.unlink()
EOF
    filled="$(echo $filled $b)"
    keep_only "$filled" || { notes="$notes; pairing for the fill-in failed"; break; }
    export CHESHIRE_BACKEND=cheshire
    case " $filled " in *" aliceVision_incrementalSfM "*) for v in $UPSTREAM_SFM; do export "$v"; done ;; esac
    t1=$(date +%s.%N)
    "$M/meshroom_batch" "${args[@]}" > "$out/batch-resume$pass.log" 2>&1
    rc=$?
    wall=$(echo "$wall + $(date +%s.%N) - $t1" | bc)
    pass=$((pass + 1))
  done
  [ -n "$filled" ] && notes="$notes; filled in by Cheshire: $filled"
  printf '{"set": "%s", "arm": "%s", "photos": "%s", "overrides": "%s", "rc": %d, "wall_s": %.1f, "start": "%s", "notes": "%s"}\n' \
    "$SET" "$arm" "$PHOTOS" "${OVERRIDES[*]:-}" "$rc" "$wall" "$(date -d @${t0%.*} -Is)" "${notes#; }" > "$out/run.json"
  echo "$tag rc=$rc wall=$(printf '%.0f' "$wall") s${notes:+ ($( echo "${notes#; }"))}" >> "$R/bench.done"
done
bash "$PAIR" "$M" --unpair > "$R/unpair-last.log" 2>&1
