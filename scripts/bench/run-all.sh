#!/usr/bin/env bash
# The whole benchmark, smallest set first: stock, Cheshire with only DepthMap on the GPU (cpu), and Cheshire, on mini6,
# the 41 views, the engine bay and the zoo; then stock and Cheshire on the False Door. Needs the GTX 1080 Ti in house-pc.
# Started detached; progress in /data/tests/bench/runs/bench.done, tables from summarize.py.
set -u
B=/data/tests/bench
nvidia-smi -L > "$B/runs/gpu.txt" 2>&1 || { mkdir -p "$B/runs"; echo "no NVIDIA card" >> "$B/runs/bench.done"; exit 1; }
bash "$B/bench.sh" mini6 /data/tests/mini6 all
bash "$B/bench.sh" monstree41 /home/house/apps/cheshire/data/monstree/full all
bash "$B/bench.sh" enginebay "$B/enginebay/photos" all
bash "$B/bench.sh" zoo "$B/zoo/photos" all
bash "$B/bench.sh" falsedoor "$B/falsedoor" both
echo "all done $(date -Is)" >> "$B/runs/bench.done"
