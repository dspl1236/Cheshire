#!/usr/bin/env bash
# Cheshire 0.4.5: provision the Ubuntu 22.04 (jammy, glibc 2.35) build container in which both Linux bundles are built,
# so they load on glibc 2.35 and newer (Ubuntu 22.04, Debian 12, Mint 21 and later) instead of 2.38 and newer.
# Runs as root INSIDE the container (scripts/linux/jammy/container.sh starts it). The package set follows upstream's own
# docker/Dockerfile_ubuntu_deps (jammy, GCC 12, Kitware's CMake 3.31); ROCm and CUDA are the versions the noble build
# box uses (7.2 and 12.9), from the vendors' jammy repositories. Idempotent: a step whose result is present is skipped.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! command -v g++-12 >/dev/null || ! cmake --version 2>/dev/null | grep -q "3\.31"; then
  apt-get update
  apt-get install -y --no-install-recommends ca-certificates gpg wget software-properties-common
  wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor > /usr/share/keyrings/kitware-archive-keyring.gpg
  echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' \
    > /etc/apt/sources.list.d/kitware.list
  apt-get update
  apt-get install -y --no-install-recommends \
    automake bison build-essential cmake=3.31.11-0kitware1ubuntu22.04.1 cmake-data=3.31.11-0kitware1ubuntu22.04.1 \
    cpp-12 curl file g++-12 gcc-12 gfortran-12 git libpcre2-dev libssl-dev libtool nasm ninja-build patchelf pkg-config \
    python3 python3-dev python3-venv texinfo unzip wget yasm binutils
fi
# GCC 12 as gcc/g++/gfortran (jammy's default is 11); idempotent, so it runs every time
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 --slave /usr/bin/g++ g++ /usr/bin/g++-12 \
  --slave /usr/bin/gfortran gfortran /usr/bin/gfortran-12 >/dev/null
echo "[jammy] $(gcc --version | head -1); $(cmake --version | head -1); $(ldd --version | head -1)"

# ROCm 7.2: the compiler, HIP headers and runtime libraries the HIP build links against. Nothing here runs a GPU.
if [ ! -x /opt/rocm/lib/llvm/bin/clang++ ]; then
  wget -q https://repo.radeon.com/amdgpu-install/7.2/ubuntu/jammy/amdgpu-install_7.2.70200-1_all.deb -O /tmp/amdgpu-install.deb
  apt-get install -y /tmp/amdgpu-install.deb
  amdgpu-install -y --usecase=rocm --no-dkms
fi
echo "[jammy] rocm: $(cat /opt/rocm/.info/version 2>/dev/null || ls -d /opt/rocm-*)"

# CUDA 12.9: the toolkit alone (no driver), as on the noble box.
if [ ! -x /usr/local/cuda-12.9/bin/nvcc ]; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
  apt-get install -y /tmp/cuda-keyring.deb
  apt-get update
  apt-get install -y --no-install-recommends cuda-toolkit-12-9
fi
echo "[jammy] $(/usr/local/cuda-12.9/bin/nvcc --version | sed -n 4p)"
echo "[jammy] provisioned"
