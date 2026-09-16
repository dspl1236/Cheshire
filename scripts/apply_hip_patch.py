#!/usr/bin/env python3
"""Apply Cheshire's HIP-backend changes to the AliceVision submodule working tree.

Idempotent: run it again after a submodule update and it re-applies (or reports
anchors that moved). Also regenerates patches/0002-hip-backend-cmake.patch from the
resulting `git diff` so the change set stays reviewable / upstreamable.

What it does
  * src/cmake/config.hpp.in           + ALICEVISION_HAVE_HIP()
  * src/CMakeLists.txt                + ALICEVISION_USE_HIP option, HIP detection after the
                                        CUDA block. A HIP build sets ALICEVISION_HAVE_CUDA=1
                                        (the CUDA sources are what gets compiled) and routes
                                        ALICEVISION_CUDA_LIBRARIES to hip::host.
  * src/aliceVision/depthMap/CMakeLists.txt
                                      + on HIP: device sources = one unity TU, host sources
                                        compiled as HIP, CUDA::cudart -> ${ALICEVISION_CUDA_LIBRARIES}
  * src/aliceVision/depthMap/cuda/hip/  new: compat shims + unity TU (copied from hip/compat, hip/port)
"""
from __future__ import annotations
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AV = ROOT / "third_party" / "aliceVision"
MARK = "# --- cheshire HIP backend ---"
NL = chr(10)


def patch(path: Path, anchor: str, new: str, *, after: bool = True, once_marker: str = MARK) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return  # already applied (exact block present)
    if anchor not in text:
        sys.exit(f"anchor not found in {path}:\n{anchor}")
    text = text.replace(anchor, (anchor + new) if after else (new + anchor), 1)
    path.write_text(text, encoding="utf-8", newline="\n")


TRACKED = [
    "src/cmake/config.hpp.in",
    "src/CMakeLists.txt",
    "src/aliceVision/depthMap/CMakeLists.txt",
    "src/aliceVision/mvsData/ROI.hpp",
    "src/aliceVision/depthMap/BufPtr.hpp",
    "src/aliceVision/sfm/pipeline/expanding/DistanceWeighting.cpp",
    "src/aliceVision/depthMap/cuda/host/memory.hpp",
    "src/aliceVision/depthMap/cuda/planeSweeping/deviceSimilarityVolume.cu",
    "src/aliceVision/depthMap/cuda/planeSweeping/deviceSimilarityVolumeKernels.cuh",
    "src/aliceVision/depthMap/cuda/imageProcessing/deviceMipmappedArray.cu",
    "src/aliceVision/depthMap/Sgm.cpp",
    "src/aliceVision/depthMap/Refine.cpp",
    "src/aliceVision/mvsUtils/ImagesCache.hpp",
    "src/aliceVision/mvsUtils/ImagesCache.cpp",
    "src/aliceVision/depthMap/DepthMapEstimator.cpp",
    "src/aliceVision/matching/RegionsMatcher.cpp",
    "src/aliceVision/matching/CMakeLists.txt",
    "src/software/pipeline/main_featureMatching.cpp",
    "src/aliceVision/fuseCut/Fuser.cpp",
    "src/aliceVision/fuseCut/CMakeLists.txt",
]


def main() -> None:
    # 0. always start from pristine upstream files so re-runs never stack edits
    subprocess.run(["git", "checkout", "--", *TRACKED], cwd=AV, check=True)

    # 1. new files: compat shims + unity TU inside the tree
    dst = AV / "src" / "aliceVision" / "depthMap" / "cuda" / "hip"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ["cuda_runtime.h", "cuda_fp16.h", "math_constants.h"]:
        shutil.copy2(ROOT / "hip" / "compat" / "include" / f, dst / f)
    (dst / "cheshire").mkdir(exist_ok=True)
    for f in ["cuda_to_hip.h", "bridge.h", "mipmap_emu.h"]:
        shutil.copy2(ROOT / "hip" / "compat" / "include" / "cheshire" / f, dst / "cheshire" / f)
    shutil.copy2(ROOT / "hip" / "port" / "unity" / "depthmap_device_unity.hip", dst / "depthmap_device_unity.hip")
    # header overlay (2-line change) applied in place
    for rel in ["src/aliceVision/mvsData/ROI.hpp", "src/aliceVision/depthMap/BufPtr.hpp"]:
        p = AV / rel
        t = p.read_text(encoding="utf-8")
        t2 = t.replace("#if defined(__NVCC__)\n", "#if defined(__NVCC__) || defined(__HIPCC__)\n") \
              .replace("#if !defined(__NVCC__)\n", "#if !defined(__NVCC__) && !defined(__HIPCC__)\n")
        if t2 != t:
            p.write_text(t2, encoding="utf-8", newline="\n")

    # 1a. HIP on Windows (ROCm 7.2.1, RX 9070) samples 16-bit float (half4) texture arrays as
    #     zeros (hip/tests/half_tex.hip); float4 arrays work. Use AliceVision's float4 texture
    #     path in HIP builds (2x camera-image VRAM, handled by the memory bridge).
    mh = AV / "src/aliceVision/depthMap/cuda/host/memory.hpp"
    t = mh.read_text(encoding="utf-8")
    t2 = t.replace("// #define ALICEVISION_DEPTHMAP_TEXTURE_USE_UCHAR\n#define ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF\n",
                   "#if defined(__HIP_PLATFORM_AMD__) && defined(CHESHIRE_TEXTURE_FLOAT4)\n"
                   "// cheshire escape hatch: 16-byte float4 camera textures (no half conversions anywhere)\n"
                   "#else\n"
                   "// half4 textures: on HIP-Windows the mip levels are built through a buffer copy because\n"
                   "// surf2Dwrite into 16-bit float arrays is broken there (deviceMipmappedArray.cu, CHESHIRE_HIP)\n"
                   "#define ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF\n"
                   "#endif\n", 1)
    assert t2 != t, "memory.hpp texture defines changed upstream"
    if t2 != t:
        mh.write_text(t2, encoding="utf-8", newline="\n")

    # 1b. clang (OpenMP) cannot capture a structured binding inside an omp region
    #     (DistanceWeighting.cpp: "capturing a structured binding is not yet supported in OpenMP")
    dw = AV / "src/aliceVision/sfm/pipeline/expanding/DistanceWeighting.cpp"
    t = dw.read_text(encoding="utf-8")
    t2 = t.replace("    for (auto & [idView, pointCloud] : perViewObservations)\n    {\n",
                   "    for (auto & viewObs : perViewObservations)\n    {\n"
                   "        const auto& idView = viewObs.first;   // not a structured binding: clang/OpenMP cannot capture those\n"
                   "        auto& pointCloud = viewObs.second;\n", 1)
    if t2 != t:
        dw.write_text(t2, encoding="utf-8", newline="\n")


    # 1d. fused SGM path aggregation: one kernel per volume row instead of three
    #     (profiled on the RX 9070: bestZ 43 %, slice copy 26 %, aggregate 30 % of SGM optimize).
    #     Kernel text: hip/port/sgm_fused/kernel.cuh.txt; host loop: hip/port/sgm_fused/loop.cu.txt.
    #     The original three-kernel loop is kept under #else for TSIM_USE_FLOAT / CHESHIRE_SGM_LEGACY.
    kh = AV / "src/aliceVision/depthMap/cuda/planeSweeping/deviceSimilarityVolumeKernels.cuh"
    t = kh.read_text(encoding="utf-8")
    if "volume_agregateCostVolumeAtXinSlicesFused_kernel" not in t:
        ktxt = (ROOT / "hip/port/sgm_fused/kernel.cuh.txt").read_text(encoding="utf-8")
        end = t.rfind("} // namespace depthMap")
        assert end > 0, "namespace end not found in deviceSimilarityVolumeKernels.cuh"
        t = t[:end] + ktxt + t[end:]
        kh.write_text(t, encoding="utf-8", newline="\n")
    sv = AV / "src/aliceVision/depthMap/cuda/planeSweeping/deviceSimilarityVolume.cu"
    t = sv.read_text(encoding="utf-8")
    if "Fused_kernel<<<" not in t:
        start_marker = "    CudaDeviceMemoryPitched<TSimAcc, 2>* xzSliceForY_dmpPtr   = &inout_volSliceAccA_dmp; // Y slice\n"
        end_marker = "        std::swap(xzSliceForYm1_dmpPtr, xzSliceForY_dmpPtr);\n    }\n"
        i0 = t.find(start_marker); i1 = t.find(end_marker, i0)
        assert i0 > 0 and i1 > i0, "aggregation loop markers not found in deviceSimilarityVolume.cu"
        i1 += len(end_marker)
        original = t[i0:i1]
        fused = (ROOT / "hip/port/sgm_fused/loop.cu.txt").read_text(encoding="utf-8")
        t = t[:i0] + "#if !defined(TSIM_USE_FLOAT) && !defined(CHESHIRE_SGM_LEGACY)\n" + fused + "#else\n" + original + "#endif\n" + t[i1:]
        sv.write_text(t, encoding="utf-8", newline="\n")

    # 1f. mip levels via buffer + cudaMemcpy2DToArray instead of surf2Dwrite (HIP-Windows drops
    #     surface stores into 16-bit float arrays: hip/tests/surf_probe.hip)
    ma = AV / "src/aliceVision/depthMap/cuda/imageProcessing/deviceMipmappedArray.cu"
    t = ma.read_text(encoding="utf-8")
    if "createMipmappedArrayLevelToBuffer_kernel" not in t:
        anchor = "__host__ void cuda_createMipmappedArrayFromImage("
        assert anchor in t
        t = t.replace(anchor, (ROOT / "hip/port/sgm_fused/miplevel_kernel.cu.txt").read_text(encoding="utf-8") + anchor, 1)
        s0 = "        cudaSurfaceObject_t currentLevel_surf;\n"
        s1 = "        CHECK_CUDA_RETURN_ERROR(cudaDestroyTextureObject(previousLevel_tex));\n"
        i0 = t.find(s0); i1 = t.find(s1, i0)
        assert i0 > 0 and i1 > i0, "mip level surface block not found"
        i1 += len(s1)
        original = t[i0:i1]
        t = t[:i0] + "#ifdef CHESHIRE_HIP\n" + (ROOT / "hip/port/sgm_fused/miplevel_host.cu.txt").read_text(encoding="utf-8") + "#else\n" + original + "#endif\n" + t[i1:]
        ma.write_text(t, encoding="utf-8", newline="\n")

    # 1g. opt-in stage-level syncs (CHESHIRE_PROFILE_SGM=1) before the "... done." log lines in
    #     Sgm.cpp / Refine.cpp so the log timestamps measure GPU stage time without serializing
    #     every launch (scripts/profile_log.py parses them).
    import re as _re
    stage_defs = (
        "#ifdef CHESHIRE_HIP\n#include <cstdlib>\n"
        "static bool cheshire_stage_prof() { static int v = -1; if (v < 0) { const char* e = std::getenv(\"CHESHIRE_PROFILE_SGM\"); v = (e && e[0] == '1') ? 1 : 0; } return v == 1; }\n"
        "#define CHESHIRE_STAGE_SYNC(s) if (cheshire_stage_prof()) cudaStreamSynchronize(s)\n"
        "#else\n#define CHESHIRE_STAGE_SYNC(s)\n#endif\n")
    for fname, hdr in [("Sgm.cpp", '#include "Sgm.hpp"\n'), ("Refine.cpp", '#include "Refine.hpp"\n')]:
        fp = AV / "src/aliceVision/depthMap" / fname
        t = fp.read_text(encoding="utf-8")
        if "CHESHIRE_STAGE_SYNC" not in t:
            assert hdr in t
            t = t.replace(hdr, hdr + stage_defs, 1)
            t = _re.sub(r'^(\s*)(ALICEVISION_LOG_INFO\(tile << "(?:SGM |Refine |Color optimize )[^"]* done\."\);)',
                        r'\1CHESHIRE_STAGE_SYNC(_stream);\n\1\2', t, flags=_re.M)
            fp.write_text(t, encoding="utf-8", newline="\n")

    # 1h. parallel image prefetch per batch (image cache slot lock + omp prefetch loop)
    ich = AV / "src/aliceVision/mvsUtils/ImagesCache.hpp"
    t = ich.read_text(encoding="utf-8")
    if "_slotMutex" not in t:
        assert "    std::vector<std::mutex> _imagesMutexes;\n" in t
        t = t.replace("    std::vector<std::mutex> _imagesMutexes;\n",
                      "    std::vector<std::mutex> _imagesMutexes;\n    std::mutex _slotMutex;  // cheshire: slot bookkeeping, allows parallel loads of different cameras\n", 1)
        ich.write_text(t, encoding="utf-8", newline="\n")
    icc = AV / "src/aliceVision/mvsUtils/ImagesCache.cpp"
    t = icc.read_text(encoding="utf-8")
    if "_slotMutex" not in t:
        f0 = t.find("void ImagesCache<Image>::refreshData(int camId)\n{")
        f1 = t.find("template<typename Image>\nvoid ImagesCache<Image>::refreshImage_sync", f0)
        assert f0 > 0 and f1 > f0, "refreshData not found"
        t = t[:f0] + (ROOT / "hip/port/sgm_fused/imagescache_refresh.cpp.txt").read_text(encoding="utf-8") + "\n" + t[f1:]
        icc.write_text(t, encoding="utf-8", newline="\n")
    dme = AV / "src/aliceVision/depthMap/DepthMapEstimator.cpp"
    t = dme.read_text(encoding="utf-8")
    if "batchCams" not in t:
        anchor = "        // load tile R and corresponding T cameras in device cache\n"
        assert anchor in t, "prefetch anchor not found"
        t = t.replace(anchor, (ROOT / "hip/port/sgm_fused/prefetch.cpp.txt").read_text(encoding="utf-8") + anchor, 1)
        t = t.replace('#include "DepthMapEstimator.hpp"\n', '#include "DepthMapEstimator.hpp"\n#include <aliceVision/alicevision_omp.hpp>\n#include <algorithm>\n', 1)
        dme.write_text(t, encoding="utf-8", newline="\n")

    # 1i. bridge v2 planner: tile parallelism from the VRAM budget, camera images may spill
    #     (hip/port/bridge_v2/planner.cpp.txt replaces the hipMemGetInfo * 0.8 block)
    t = dme.read_text(encoding="utf-8")
    if "cheshire::bridge::budget" not in t:
        p0 = "    // available device memory\n    double deviceMemoryMB;\n"
        p1 = "        nbRemainingTiles = static_cast<int>(std::max(0.0, remainingMemoryMB - rcCamsCostMB) / tileCostMB);\n    }\n"
        i0 = t.find(p0); i1 = t.find(p1, i0)
        assert i0 > 0 and i1 > i0, "planner block not found in DepthMapEstimator.cpp"
        i1 += len(p1)
        original = t[i0:i1]
        t = t[:i0] + "#ifdef CHESHIRE_HIP\n" + (ROOT / "hip/port/bridge_v2/planner.cpp.txt").read_text(encoding="utf-8") + "#else\n" + original + "#endif\n" + t[i1:]
        dme.write_text(t, encoding="utf-8", newline="\n")

    # 1e. block-height override for the occupancy-derived launch shape (CHESHIRE_BLOCK_Y)
    t = sv.read_text(encoding="utf-8")
    old_blk = ("    if(recommendedBlockSize > 32)\n    {\n        const dim3 recommendedBlock(32, divUp(recommendedBlockSize, 32), 1);\n"
               "        return recommendedBlock;\n    }\n")
    if "CHESHIRE_BLOCK_Y" not in t:
        assert old_blk in t, "getMaxPotentialBlockSize body changed upstream"
        t = t.replace(old_blk, (ROOT / "hip/port/sgm_fused/blocksize.cu.txt").read_text(encoding="utf-8"), 1)
        sv.write_text(t, encoding="utf-8", newline="\n")

    # 1c. opt-in SGM aggregation profiling (CHESHIRE_PROFILE_SGM=1): per-kernel wall time with
    #     device syncs around the three per-row launches. HIP builds only (CHESHIRE_HIP).
    sv = AV / "src/aliceVision/depthMap/cuda/planeSweeping/deviceSimilarityVolume.cu"
    t = sv.read_text(encoding="utf-8")
    prof_defs = """#ifdef CHESHIRE_HIP
static double cheshire_prof_acc[3]; static int cheshire_prof_calls; static std::mutex cheshire_prof_mutex;
static bool cheshire_prof_on() { static int v = -1; if (v < 0) { const char* e = std::getenv("CHESHIRE_PROFILE_SGM"); v = (e && e[0] == '1') ? 1 : 0; } return v == 1; }
static thread_local std::chrono::steady_clock::time_point _cp_t0;
#define CHESHIRE_PROF_BEGIN() if (cheshire_prof_on()) { cudaDeviceSynchronize(); _cp_t0 = std::chrono::steady_clock::now(); }
#define CHESHIRE_PROF_END(i) if (cheshire_prof_on()) { cudaDeviceSynchronize(); std::lock_guard<std::mutex> _cp_g(cheshire_prof_mutex); cheshire_prof_acc[i] += std::chrono::duration<double>(std::chrono::steady_clock::now() - _cp_t0).count(); }
#define CHESHIRE_PROF_REPORT() if (cheshire_prof_on() && (++cheshire_prof_calls % 24 == 0)) std::fprintf(stderr, "[cheshire-prof] SGM aggregate cumulative after %d passes: bestZ %.3f s, getSlice %.3f s, aggregate %.3f s\\n", cheshire_prof_calls, cheshire_prof_acc[0], cheshire_prof_acc[1], cheshire_prof_acc[2]);
#else
#define CHESHIRE_PROF_BEGIN()
#define CHESHIRE_PROF_END(i)
#define CHESHIRE_PROF_REPORT()
#endif
"""
    if "#define CHESHIRE_PROF_BEGIN" not in t:
        t = t.replace('#include "deviceSimilarityVolume.hpp"\n',
                      '#include "deviceSimilarityVolume.hpp"\n#include <chrono>\n#include <cstdlib>\n#include <cstdio>\n', 1)
        t = t.replace("__host__ void cuda_volumeAggregatePath(", prof_defs + "__host__ void cuda_volumeAggregatePath(", 1)
        sv.write_text(t, encoding="utf-8", newline="\n")

    # 2. config.hpp.in
    patch(AV / "src/cmake/config.hpp.in",
          "#define ALICEVISION_HAVE_SYCL() @ALICEVISION_HAVE_SYCL@\n",
          "\n// --- cheshire HIP backend ---\n// HIP build: the CUDA depth-map sources are compiled through a CUDA->HIP compat header,\n"
          "// so ALICEVISION_HAVE_CUDA stays 1 and this flag marks the AMD runtime underneath.\n"
          "#define ALICEVISION_HAVE_HIP() @ALICEVISION_HAVE_HIP@\n")

    # 3. src/CMakeLists.txt: option + detection block after the CUDA block
    patch(AV / "src/CMakeLists.txt",
          'trilean_option(ALICEVISION_USE_SYCL "Enable SYCL" AUTO)\n',
          f'{MARK}\ntrilean_option(ALICEVISION_USE_HIP "Enable HIP (AMD GPUs): compiles the CUDA depth-map backend through a CUDA->HIP compat layer" AUTO)\n')
    hip_block = f'''
# ==============================================================================
{MARK}
# HIP (AMD ROCm). Only considered when no CUDA toolkit was found. The existing CUDA
# depth-map sources are compiled as HIP through src/aliceVision/depthMap/cuda/hip/,
# so from the code's point of view ALICEVISION_HAVE_CUDA is 1.
# ==============================================================================
set(ALICEVISION_HAVE_HIP 0)
if (NOT ALICEVISION_HAVE_CUDA AND NOT ALICEVISION_USE_HIP STREQUAL "OFF")
    include(CheckLanguage)
    check_language(HIP)
    if (NOT CMAKE_HIP_COMPILER)
        if (ALICEVISION_USE_HIP STREQUAL "ON")
            message(SEND_ERROR "Failed to find a HIP compiler (set CMAKE_HIP_COMPILER to ROCm's clang++/clang-cl).")
        endif()
    else()
        enable_language(HIP)
        find_package(hip QUIET)
        if (hip_FOUND)
            set(ALICEVISION_HAVE_HIP 1)
            set(ALICEVISION_HAVE_CUDA 1)
            list(APPEND ALICEVISION_CUDA_LIBRARIES hip::host)
            set(ALICEVISION_HIP_DIR "${{CMAKE_CURRENT_SOURCE_DIR}}/aliceVision/depthMap/cuda/hip")
            # cuda_runtime.h / cuda_fp16.h / math_constants.h shims must win over any real CUDA headers
            include_directories(BEFORE "${{ALICEVISION_HIP_DIR}}")
            add_compile_definitions(__HIP_PLATFORM_AMD__=1)
            # nvcc pre-includes cuda_runtime.h into every .cu; clang does not
            if (CMAKE_HIP_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
                add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:/FI${{ALICEVISION_HIP_DIR}}/cheshire/cuda_to_hip.h>")
                add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:/bigobj>")
            else()
                add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:-include${{ALICEVISION_HIP_DIR}}/cheshire/cuda_to_hip.h>")
            endif()
            add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:-Wno-ignored-attributes>" "$<$<COMPILE_LANGUAGE:HIP>:-Wno-unknown-attributes>")
            if (WIN32)
                # the device-side pass rejects Boost.WinAPI's own __stdcall prototypes next to windows.h
                add_compile_definitions("$<$<COMPILE_LANGUAGE:HIP>:BOOST_USE_WINDOWS_H>" "$<$<COMPILE_LANGUAGE:HIP>:WIN32_LEAN_AND_MEAN>" "$<$<COMPILE_LANGUAGE:HIP>:NOMINMAX>")
            endif()
            option(ALICEVISION_HIP_RDC "HIP relocatable device code (-fgpu-rdc) instead of the unity device TU; broken on Windows ROCm 7.2.1" OFF)
            message(STATUS "HIP found: ${{hip_VERSION}} (architectures: ${{CMAKE_HIP_ARCHITECTURES}})")
        elseif (ALICEVISION_USE_HIP STREQUAL "ON")
            message(SEND_ERROR "Failed to find the hip CMake package (add the ROCm root to CMAKE_PREFIX_PATH).")
        endif()
    endif()
endif()
'''
    patch(AV / "src/CMakeLists.txt",
          "# ==============================================================================\n# SYCL/AdaptiveCpp\n",
          hip_block, after=False)

    # 4. depthMap/CMakeLists.txt: HIP source handling + link line
    dm = AV / "src/aliceVision/depthMap/CMakeLists.txt"
    patch(dm,
          "alicevision_add_library(aliceVision_depthMap_cuda\n",
          f'''{MARK}
if (ALICEVISION_HAVE_HIP)
    set(depthMap_hip_device_sources cuda/hip/depthmap_device_unity.hip)
    if (ALICEVISION_HIP_RDC)
        set(depthMap_hip_device_sources
            cuda/device/DeviceCameraParams.cu cuda/device/DevicePatchPattern.cu
            cuda/imageProcessing/deviceGaussianFilter.cu cuda/imageProcessing/deviceColorConversion.cu
            cuda/imageProcessing/deviceMipmappedArray.cu
            cuda/planeSweeping/deviceDepthSimilarityMap.cu cuda/planeSweeping/deviceSimilarityVolume.cu)
        add_compile_options("$<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>")
        add_link_options(-fgpu-rdc --hip-link)
    else()
        # the .cu files are pulled into the unity TU; keep them out of the build
        set_source_files_properties(
            cuda/device/DeviceCameraParams.cu cuda/device/DevicePatchPattern.cu
            cuda/imageProcessing/deviceGaussianFilter.cu cuda/imageProcessing/deviceColorConversion.cu
            cuda/imageProcessing/deviceMipmappedArray.cu
            cuda/planeSweeping/deviceDepthSimilarityMap.cu cuda/planeSweeping/deviceSimilarityVolume.cu
            cuda/host/DeviceCache.cpp cuda/host/patchPattern.cpp
            PROPERTIES HEADER_FILE_ONLY true)
    endif()
    # everything that touches the runtime is compiled by the HIP compiler (host + device passes)
    set_source_files_properties(${{depthMap_hip_device_sources}} ${{depthMap_cuda_host_sources}} ${{depthMap_cuda_imageProcessing_sources}} ${{depthMap_cuda_planeSweeping_sources}} ${{depthMap_cuda_device_sources}}
        PROPERTIES LANGUAGE HIP)
    list(APPEND depthMap_cuda_files_sources ${{depthMap_hip_device_sources}})
endif()

''', after=False)
    t = dm.read_text(encoding="utf-8")
    if "        CUDA::cudart\n" in t:
        t = t.replace("        CUDA::cudart\n", "        ${ALICEVISION_CUDA_LIBRARIES}   # CUDA::cudart, or hip::host on a HIP build\n", 1)
        dm.write_text(t, encoding="utf-8", newline="\n")

    # 4b. GPU descriptor matcher (hip/port/gpu_matcher): exact brute-force 2-NN on the GPU behind
    #     RegionsMatcher, taken for ANN_L2 / BRUTE_FORCE_L2 whenever a device is present. Compiled as
    #     HIP here; the same source is CUDA for an NVIDIA build (ALICEVISION_HAVE_CUDA).
    gm_dst = AV / "src/aliceVision/matching/gpu"
    gm_dst.mkdir(parents=True, exist_ok=True)
    for f in ("gpuMatcher.hpp", "gpuMatcher.cu", "ArrayMatcher_gpuBruteForce.hpp"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_matcher" / f, gm_dst / f)
    rm = AV / "src/aliceVision/matching/RegionsMatcher.cpp"
    patch(rm, '#include "aliceVision/matching/ArrayMatcher_cascadeHashing.hpp"' + NL,
          '#ifdef ALICEVISION_HAVE_GPU_MATCHER' + NL + '#include "aliceVision/matching/gpu/ArrayMatcher_gpuBruteForce.hpp"  // cheshire' + NL + '#endif' + NL)
    gm_block = """#ifdef ALICEVISION_HAVE_GPU_MATCHER
            // cheshire: exact 2-NN on the GPU instead of the kd-tree / CPU brute force
            if ((matcherType == ANN_L2 || matcherType == BRUTE_FORCE_L2) && gpu::available() && gpu::supportsDim(regions.DescriptorLength()))
            {
                typedef feature::L2_Vectorized<SCALAR> MetricT;
                typedef ArrayMatcher_gpuBruteForce<SCALAR, MetricT> MatcherT;
                out.reset(new matching::RegionsMatcher<MatcherT>(randomNumberGenerator, regions, true));
                return out;
            }
#endif
"""
    patch(rm, "            // Build on the fly unsigned char based Matcher" + NL, gm_block.replace("SCALAR", "unsigned char"))
    patch(rm, "            // Build on the fly float based Matcher" + NL, gm_block.replace("SCALAR", "float"))
    mc = AV / "src/aliceVision/matching/CMakeLists.txt"
    patch(mc, "alicevision_add_library(aliceVision_matching" + NL, f"""{MARK} (GPU matcher)
set(matching_gpu_links "")
if (ALICEVISION_HAVE_CUDA OR ALICEVISION_HAVE_HIP)
    list(APPEND matching_files_headers gpu/gpuMatcher.hpp gpu/ArrayMatcher_gpuBruteForce.hpp)
    list(APPEND matching_files_sources gpu/gpuMatcher.cu)
    if (ALICEVISION_HAVE_HIP)
        set_source_files_properties(gpu/gpuMatcher.cu PROPERTIES LANGUAGE HIP)
    endif()
    set(matching_gpu_links ${{ALICEVISION_CUDA_LIBRARIES}})
endif()

""", after=False)
    patch(mc, "        ${FLANN_LIBRARIES}" + NL, "        ${matching_gpu_links}" + NL)
    patch(mc, "# Unit tests" + NL, f"""{MARK} (GPU matcher)
if (ALICEVISION_HAVE_CUDA OR ALICEVISION_HAVE_HIP)
    target_compile_definitions(aliceVision_matching PRIVATE ALICEVISION_HAVE_GPU_MATCHER=1)
endif()

""", after=False)

    # 4c. Meshroom 2023.3 chunk options for FeatureMatching: the 2023.3 node passes
    #     --rangeStart/--rangeSize (a range of *views*, chunk file = rangeStart/rangeSize) while this
    #     AliceVision chunks *pairs* with --rangeIteration/--rangeBlocksCount. Accept the old pair so
    #     the paired binary is a drop-in: keep the pairs whose first view's index is in the range.
    fm = AV / "src/software/pipeline/main_featureMatching.cpp"
    patch(fm, "    int rangeIteration = 0;" + NL,
          "    int rangeStart = -1;  // cheshire: Meshroom 2023.3 chunking" + NL + "    int rangeSize = -1;" + NL + "    int legacyChunkIndex = -1;" + NL)
    t = fm.read_text(encoding="utf-8")
    if "legacyChunkIndex >= 0" not in t:
        old_prefix = '    const std::string filePrefix = std::to_string(rangeIteration) + ".";'
        if old_prefix not in t:
            sys.exit("filePrefix line not found in main_featureMatching.cpp")
        t = t.replace(old_prefix, '    const std::string filePrefix = std::to_string(legacyChunkIndex >= 0 ? legacyChunkIndex : rangeIteration) + ".";  // cheshire', 1)
        fm.write_text(t, encoding="utf-8", newline=NL)
    patch(fm, '        ("rangeIteration", po::value<int>(&rangeIteration)->default_value(rangeIteration),' + NL,
          '        ("rangeStart", po::value<int>(&rangeStart)->default_value(rangeStart),' + NL
          + '         "cheshire: Meshroom 2023.3 chunking, first view index of the chunk (pairs are selected by their first view).")' + NL
          + '        ("rangeSize", po::value<int>(&rangeSize)->default_value(rangeSize),' + NL
          + '         "cheshire: Meshroom 2023.3 chunking, number of views in the chunk.")' + NL, after=False)
    patch(fm, "    int chunkStart, chunkEnd;" + NL,
          """    // cheshire: Meshroom 2023.3 chunking (--rangeStart/--rangeSize over views)
    if (rangeStart >= 0 && rangeSize > 0)
    {
        std::vector<IndexT> viewIds;
        viewIds.reserve(sfmData.getViews().size());
        for (const auto& v : sfmData.getViews()) viewIds.push_back(v.first);
        std::set<IndexT> chunkViews;
        for (int i = rangeStart; i < rangeStart + rangeSize && i < int(viewIds.size()); ++i) chunkViews.insert(viewIds[i]);
        PairSet chunkPairs;
        for (const auto& pair : allPairs) if (chunkViews.count(pair.first)) chunkPairs.insert(pair);
        ALICEVISION_LOG_INFO("Meshroom 2023.3 chunking: views " << rangeStart << " to " << rangeStart + rangeSize << " -> " << chunkPairs.size() << " of " << allPairs.size() << " pairs.");
        allPairs.swap(chunkPairs);
        legacyChunkIndex = rangeStart / rangeSize;   // output file prefix, as the 2023.3 binary named it
        rangeIteration = 0;                          // the pair range below is then "all of the chunk"
        rangeBlocksCount = 1;
        if (allPairs.empty())
        {
            ALICEVISION_LOG_INFO("No image pair in this chunk.");
            return EXIT_SUCCESS;
        }
    }
""", after=False)

    # 4d. GPU depth map filter (hip/port/gpu_filter): DepthMapFilter's group-vote pass on the GPU
    #     behind Fuser::filterGroupsRC, bit-identical to the CPU pass (double precision, no FMA).
    gf_dst = AV / "src/aliceVision/fuseCut/gpu"
    gf_dst.mkdir(parents=True, exist_ok=True)
    for f in ("depthMapFilterGPU.hpp", "depthMapFilterGPU.cu"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_filter" / f, gf_dst / f)
    fu = AV / "src/aliceVision/fuseCut/Fuser.cpp"
    patch(fu, '#include <aliceVision/mvsUtils/mapIO.hpp>' + NL,
          '#ifdef ALICEVISION_HAVE_GPU_FILTER' + NL + '#include "aliceVision/fuseCut/gpu/depthMapFilterGPU.hpp"  // cheshire' + NL + '#endif' + NL)
    patch(fu, "    StaticVector<int> tcams = _mp.findNearestCamsFromLandmarks(rc, nNearestCams);" + NL,
          """
#ifdef ALICEVISION_HAVE_GPU_FILTER
    // cheshire: the same votes on the GPU (one thread per tc pixel); the per-camera matrix
    // decompositions upstream redoes per pixel are done once here with upstream's own code
    if (gpu::available())
    {
        auto geom = [&](int cam) {
            gpu::CamGeom g{};
            const Matrix3x4& P = _mp.camArr[cam];
            const double pm[12] = {P.m11, P.m12, P.m13, P.m14, P.m21, P.m22, P.m23, P.m24, P.m31, P.m32, P.m33, P.m34};
            for (int i = 0; i < 12; ++i) g.P[i] = pm[i];
            const Matrix3x3& iC = _mp.iCamArr[cam];
            const double im[9] = {iC.m11, iC.m12, iC.m13, iC.m21, iC.m22, iC.m23, iC.m31, iC.m32, iC.m33};
            for (int i = 0; i < 9; ++i) g.iCam[i] = im[i];
            g.C[0] = _mp.CArr[cam].x; g.C[1] = _mp.CArr[cam].y; g.C[2] = _mp.CArr[cam].z;
            Point3d Co; Matrix3x3 Ro, iRo, Ko, iKo, iPo;
            _mp.decomposeProjectionMatrix(Co, Ro, iRo, Ko, iKo, iPo, P);
            const double rp[9] = {iPo.m11, iPo.m12, iPo.m13, iPo.m21, iPo.m22, iPo.m23, iPo.m31, iPo.m32, iPo.m33};
            for (int i = 0; i < 9; ++i) g.riP[i] = rp[i];
            g.rC[0] = Co.x; g.rC[1] = Co.y; g.rC[2] = Co.z;
            g.w = _mp.getWidth(cam); g.h = _mp.getHeight(cam);
            return g;
        };
        gpu::GroupFilter gf;
        bool ok = gf.setRc(depthMap.data(), simMap.data(), geom(rc));
        for (int c = 0; ok && c < tcams.size(); c++)
        {
            const int tc = tcams[c];
            image::Image<float> tcdepthMap;
            mvsUtils::readMap(tc, _mp, mvsUtils::EFileType::depthMap, tcdepthMap);
            if (std::getenv("CHESHIRE_GPU_FILTER_DEBUG") && tcdepthMap.height() > 0)
            {
                // one tc pixel through both implementations, printed side by side
                int x = tcdepthMap.width() / 2, y = tcdepthMap.height() / 2;
                while (y < tcdepthMap.height() && !(tcdepthMap(y, x) > 0.0f)) ++y;
                if (y < tcdepthMap.height())
                {
                    const float depth = tcdepthMap(y, x);
                    const Point3d p = _mp.CArr[tc] + (_mp.iCamArr[tc] * Point2d((float)x, (float)y)).normalize() * depth;
                    Pixel pix; _mp.getPixelFor3DPoint(&pix, p, rc);
                    const float pixDepth = (_mp.CArr[rc] - p).size();
                    const double avRcTc = _mp.getCamPixelSizeRcTc(p, rc, tc, 1.0f), avRc = _mp.getCamPixelSize(p, rc, 1.0f);
                    const float pixSize = pixToleranceFactor * _mp.getCamPixelSizePlaneSweepAlpha(p, rc, tc, 1, 1);
                    const float rcd = _mp.isPixelInImage(pix, rc) ? depthMap(pix.y, pix.x) : -999.f;
                    double o[10]; gf.probe(tcdepthMap.data(), geom(tc), x, y, pixToleranceFactor, pixSizeBall, pixSizeBallWSP, o);
                    std::fprintf(stderr, "[cheshire] filter probe rc=%d tc=%d tcpix=(%d,%d) depth=%.9g\\n  CPU: pix=(%d,%d) pixDepth=%.9g avRcTc=%.17g avRc=%.17g pixSize=%.9g rcDepth=%.9g p=(%.17g,%.17g,%.17g)\\n  GPU: pix=(%g,%g) pixDepth=%.9g avRcTc=%.17g avRc=%.17g pixSize=%.9g rcDepth=%.9g p=(%.17g,%.17g,%.17g)\\n",
                                 rc, tc, x, y, depth, pix.x, pix.y, pixDepth, avRcTc, avRc, pixSize, rcd, p.x, p.y, p.z,
                                 o[0], o[1], o[2], o[3], o[4], o[5], o[9], o[6], o[7], o[8]);
                }
            }
            if (tcdepthMap.height() > 0 && tcdepthMap.width() > 0)
                ok = gf.accumulate(tcdepthMap.data(), geom(tc), pixToleranceFactor, pixSizeBall, pixSizeBallWSP);
            if (ok && std::getenv("CHESHIRE_GPU_FILTER_DEBUG"))
            {
                long long cnt = -1; gf.voteCount(&cnt);
                std::fprintf(stderr, "[cheshire] filter votes GPU rc=%d tc=%d (%dx%d): %lld rc pixels voted\\n", rc, tc, tcdepthMap.width(), tcdepthMap.height(), cnt);
            }
        }
        if (ok && gf.result(numOfModalsMap.data()))
        {
            image::writeImageWithFloat(
              getFileNameFromIndex(_mp, rc, mvsUtils::EFileType::nmodMap),
              numOfModalsMap,
              image::ImageWriteOptions().toColorSpace(image::EImageColorSpace::LINEAR).storageDataType(image::EStorageDataType::Float));
            delete numOfPtsMap;
            ALICEVISION_LOG_DEBUG(rc << " solved (GPU).");
            mvsUtils::printfElapsedTime(t1);
            return true;
        }
        ALICEVISION_LOG_WARNING("cheshire: GPU depth map filter failed for camera " << rc << ", falling back to the CPU pass");
    }
#endif
""")
    patch(fu, """            for (int i = 0; i < w * h; i++)
            {
                numOfModalsMap(i) += static_cast<int>((*numOfPtsMap)[i] > 0);
            }
""", """            if (std::getenv("CHESHIRE_GPU_FILTER_DEBUG"))   // cheshire diagnostics
            {
                long long cnt = 0;
                for (int i = 0; i < w * h; i++) cnt += (*numOfPtsMap)[i] > 0;
                std::fprintf(stderr, "[cheshire] filter votes CPU rc=%d tc=%d (%dx%d): %lld rc pixels voted\\n", rc, tc, tcdepthMap.width(), tcdepthMap.height(), cnt);
            }
""")
    fc = AV / "src/aliceVision/fuseCut/CMakeLists.txt"
    patch(fc, "alicevision_add_library(aliceVision_fuseCut" + NL, f"""{MARK} (GPU depth map filter)
set(fuseCut_gpu_links "")
if (ALICEVISION_HAVE_CUDA OR ALICEVISION_HAVE_HIP)
    list(APPEND fuseCut_files_headers gpu/depthMapFilterGPU.hpp)
    list(APPEND fuseCut_files_sources gpu/depthMapFilterGPU.cu)
    if (ALICEVISION_HAVE_HIP)
        set_source_files_properties(gpu/depthMapFilterGPU.cu PROPERTIES LANGUAGE HIP)
    endif()
    set(fuseCut_gpu_links ${{ALICEVISION_CUDA_LIBRARIES}})
endif()

""", after=False)
    patch(fc, "        nanoflann::nanoflann" + NL, "        ${fuseCut_gpu_links}" + NL)
    t = fc.read_text(encoding="utf-8")
    if "ALICEVISION_HAVE_GPU_FILTER" not in t:
        # after the library block (ends at the first blank line following PRIVATE_LINKS)
        i = t.index("alicevision_add_library(aliceVision_fuseCut")
        j = t.index(NL + ")" + NL, i) + 3
        t = t[:j] + f"""{MARK} (GPU depth map filter)
if (ALICEVISION_HAVE_CUDA OR ALICEVISION_HAVE_HIP)
    target_compile_definitions(aliceVision_fuseCut PRIVATE ALICEVISION_HAVE_GPU_FILTER=1)
endif()
""" + t[j:]
        fc.write_text(t, encoding="utf-8", newline=NL)


    # 5. regenerate the reviewable patch
    subprocess.run(["git", "add", "-N", "src/aliceVision/depthMap/cuda/hip"], cwd=AV, check=True)
    diff = subprocess.run(["git", "diff", "--no-color"], cwd=AV, check=True, capture_output=True, text=True).stdout
    out = ROOT / "patches" / "0002-hip-backend-cmake.patch"
    out.write_text(diff, encoding="utf-8", newline="\n")
    print(f"applied; {len(diff.splitlines())} diff lines -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
