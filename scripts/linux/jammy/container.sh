#!/usr/bin/env bash
# Cheshire 0.4.5: the jammy build container on the WSL build box (Docker Engine in the WSL distro).
#
#   scripts/linux/jammy/container.sh up        create and provision it (provision.sh), once
#   scripts/linux/jammy/container.sh sync      refresh its source tree from the box's /root/cheshire (origin/main)
#   scripts/linux/jammy/container.sh run CMD   run CMD in it, in /src/cheshire
#
# The container is persistent (cheshire-jammy): its toolchains, /opt/AliceVision_deps and build trees live in it. The
# source is a clone of the box's own checkout (mounted read-only at /host), never the checkout itself, so a build in the
# container cannot touch the noble trees; third_party/popsift, a clone of its own, is cloned the same way. Packages go
# to /out, which is build/jammy on the Windows side.
set -euo pipefail
NAME=cheshire-jammy
HOST_SRC="${CHESHIRE_HOST_SRC:-/root/cheshire}"
OUT="${CHESHIRE_JAMMY_OUT:-/mnt/d/MMI/cheshire/build/jammy}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cmd="${1:-}"; shift || true

case "$cmd" in
  up)
    mkdir -p "$OUT"
    if ! docker container inspect "$NAME" >/dev/null 2>&1; then
      docker pull ubuntu:22.04
      # the host's network: the WSL distro has IPv4 forwarding off, so a bridged container cannot reach the mirrors
      docker create --name "$NAME" --network host -v "$HOST_SRC:/host:ro" -v "$OUT:/out" -v "$HERE:/jammy:ro" ubuntu:22.04 sleep infinity
    fi
    docker start "$NAME" >/dev/null
    docker exec "$NAME" bash /jammy/provision.sh
    "$0" sync
    ;;
  sync)
    docker start "$NAME" >/dev/null
    docker exec "$NAME" bash -euc '
      git config --global --add safe.directory "*"
      if [ ! -d /src/cheshire/.git ]; then mkdir -p /src && git clone -q /host /src/cheshire; fi
      cd /src/cheshire
      git fetch -q /host main && git reset -q --hard FETCH_HEAD
      # the AliceVision submodule from the host checkout'"'"'s own module repository, at the commit main pins
      git -c protocol.file.allow=always submodule set-url third_party/aliceVision /host/.git/modules/third_party/aliceVision
      git -c protocol.file.allow=always submodule update --init --quiet third_party/aliceVision
      git -C third_party/aliceVision checkout -q -- . && git -C third_party/aliceVision clean -fdq
      git checkout -q -- .gitmodules
      # PopSIFT: a clone of its own, not a submodule, at the commit scripts/apply_popsift_patch.py pins: taken
      # from the host checkout (which the host build has fetched), then checked against the pin
      if [ ! -d third_party/popsift/.git ]; then git init -q third_party/popsift; fi
      git -C third_party/popsift fetch -q /host/third_party/popsift HEAD
      git -C third_party/popsift checkout -q -f FETCH_HEAD && git -C third_party/popsift clean -fdq
      python3 scripts/apply_popsift_patch.py --fetch
      echo "[jammy] source at $(git log --oneline -1); aliceVision $(git -C third_party/aliceVision rev-parse --short HEAD); popsift $(git -C third_party/popsift rev-parse --short HEAD)"
    '
    ;;
  run)
    docker start "$NAME" >/dev/null
    docker exec -w /src/cheshire "$NAME" bash -lc "$*"
    ;;
  *)
    sed -n '2,12p' "$0"; exit 2 ;;
esac
