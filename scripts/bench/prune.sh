#!/usr/bin/env bash
# Keeps /data from filling during the benchmark: every run listed in bench.done (so finished) loses its large
# intermediates - depth and similarity maps, PrepareDenseScene's images, features, matches, Alembic clouds, textures -
# and keeps what summarize.py reads (chunk statuses, logs, statistics, cameras.sfm, mesh.obj) plus project.mg and
# run.json. Runs until "all done" appears, checking every 5 minutes.
R=/data/tests/bench/runs
while :; do
  for d in "$R"/*/; do
    tag=$(basename "$d")
    [ -f "$d/run.json" ] && [ ! -f "$d/.pruned" ] || continue
    grep -q "^$tag " "$R/bench.done" 2>/dev/null || continue
    find "$d" -type f \( -name '*.exr' -o -name '*.png' -o -name '*.feat' -o -name '*.desc' -o -name '*.abc' -o -name '*.jpg' \
      -o -name '*.JPG' -o -name 'texturedMesh.obj' -o -name '*.matches.txt' -o -name '*.bin' -o -name 'tetrahedralization*' \) -delete
    touch "$d/.pruned"
    echo "$(date -Is) pruned $tag: $(du -sh "$d" | cut -f1) left; /data $(df -h /data | tail -1 | awk '{print $4}') free" >> "$R/prune.log"
  done
  grep -q "^all done" "$R/bench.done" 2>/dev/null && break
  sleep 300
done
