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
    "src/software/pipeline/main_depthMapFiltering.cpp",
    "src/aliceVision/fuseCut/GraphFiller.cpp",
    "src/aliceVision/mesh/Texturing.cpp",
    "src/aliceVision/mesh/CMakeLists.txt",
    "src/software/pipeline/main_texturing.cpp",
    "src/aliceVision/fuseCut/Tetrahedralization.cpp",
    "src/aliceVision/fuseCut/PointCloud.cpp",
    "src/aliceVision/fuseCut/Mesher.cpp",
    "src/aliceVision/mesh/Mesh.cpp",
    "src/aliceVision/mesh/MeshClean.cpp",
    "src/aliceVision/fuseCut/Kdtree.hpp",
    "src/aliceVision/fuseCut/GraphFiller.hpp",
    "src/software/pipeline/main_prepareDenseScene.cpp",
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
    # the pairing scripts detect the GPU pass from --help: say so in the program description
    df = AV / "src/software/pipeline/main_depthMapFiltering.cpp"
    t = df.read_text(encoding="utf-8")
    old_desc = '"AliceVision depthMapFiltering");'
    if "CHESHIRE_GPU_FILTER" not in t:
        if old_desc not in t:
            sys.exit("depthMapFiltering description not found")
        t = t.replace(old_desc, '"AliceVision depthMapFiltering (cheshire: the group votes run on the GPU when a device is present; CHESHIRE_GPU_FILTER=0 for the CPU pass)");', 1)
        df.write_text(t, encoding="utf-8", newline=NL)

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


    # 4e. GPU meshing votes (hip/port/gpu_vote): GraphFiller::fillGraph's ray marching on the GPU.
    gv_dst = AV / "src/aliceVision/fuseCut/gpu"
    gv_dst.mkdir(parents=True, exist_ok=True)
    for f in ("graphVoteGPU.hpp", "graphVoteGPU.cu"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_vote" / f, gv_dst / f)
    gfp = AV / "src/aliceVision/fuseCut/GraphFiller.cpp"
    patch(gfp, '#include <boost/atomic/atomic_ref.hpp>' + NL,
          '#ifdef ALICEVISION_HAVE_GPU_FILTER' + NL + '#include "aliceVision/fuseCut/gpu/graphVoteGPU.hpp"  // cheshire' + NL + '#include <cstdlib>' + NL + '#include <cmath>' + NL + '#endif' + NL)
    patch(gfp, """    // choose random order to prevent waiting
    const unsigned int seed = (unsigned int)_mp.userParams.get<unsigned int>("delaunaycut.seed", 0);
""", """#ifdef ALICEVISION_HAVE_GPU_FILTER
    // cheshire: the same rays on the GPU, one thread each (CHESHIRE_GPU_VOTE=0 for the CPU loop below)
    if (gpu::voteAvailable())
    {
        const uint32_t NONE = 0xffffffffu;
        const size_t nbV = _verticesCoords.size(), nbC = _tetrahedralization.nb_cells();
        std::vector<double> verts(3 * nbV);
        for (size_t i = 0; i < nbV; ++i) { verts[3 * i] = _verticesCoords[i].x; verts[3 * i + 1] = _verticesCoords[i].y; verts[3 * i + 2] = _verticesCoords[i].z; }
        std::vector<uint32_t> cv(4 * nbC), ca(4 * nbC);
        for (size_t c = 0; c < nbC; ++c)
            for (int k = 0; k < 4; ++k) { cv[4 * c + k] = (uint32_t)_tetrahedralization.cell_vertex(c, k); ca[4 * c + k] = (uint32_t)_tetrahedralization.cell_adjacent(c, k); }
        const auto& vcells = _tetrahedralization.getNeighboringCellsPerVertex();
        std::vector<uint32_t> vco(nbV + 1, 0), vcl;
        for (size_t i = 0; i < nbV; ++i) vco[i + 1] = vco[i] + (uint32_t)(i < vcells.size() ? vcells[i].size() : 0);
        vcl.reserve(vco[nbV]);
        for (size_t i = 0; i < nbV && i < vcells.size(); ++i) for (CellIndex c : vcells[i]) vcl.push_back((uint32_t)c);
        std::vector<uint32_t> rv, rc; std::vector<float> rw, rt; std::vector<double> rd, rcc;
        // the weakly-supported-surfaces pass (forceTedgesByGradientIJCV) runs on the GPU too, right after the votes
        const bool forceTEdge = _mp.userParams.get<bool>("delaunaycut.voteFilteringForWeaklySupportedSurfaces", true);
        for (size_t i = 0; i < nbV; ++i)
        {
            const GC_vertexInfo& v = _verticesAttr[i];
            if (!v.isReal()) continue;
            float weight = (float)v.nrc;
            weight = (float)_mp.userParams.get<double>("LargeScale.forceWeight", weight);
            const double maxDist = v.pixSize <= 0.0 ? 0.0 : nPixelSizeBehind * (double)v.pixSize;
            for (int c = 0; c < v.cams.size(); c++)
            {
                const int cam = v.cams[c];
                rv.push_back((uint32_t)i); rc.push_back(_camsVertexes[cam] < 0 ? NONE : (uint32_t)_camsVertexes[cam]);
                rw.push_back(weight); rd.push_back(maxDist);
                if (forceTEdge) rt.push_back((float)nPixelSizeBehind * _mp.getCamPixelSize(_verticesCoords[i], cam));
                rcc.push_back(_mp.CArr[cam].x); rcc.push_back(_mp.CArr[cam].y); rcc.push_back(_mp.CArr[cam].z);
            }
        }
        std::vector<float> attr(8 * nbC);
        for (size_t c = 0; c < nbC; ++c)
        {
            const GC_cellInfo& ci = _cellsAttr[c];
            attr[8 * c] = ci.cellSWeight; attr[8 * c + 1] = ci.cellTWeight;
            for (int k = 0; k < 4; ++k) attr[8 * c + 2 + k] = ci.gEdgeVisWeight[k];
            attr[8 * c + 6] = ci.emptinessScore; attr[8 * c + 7] = ci.on;
        }
        gpu::VoteInput in;
        in.vertices = verts.data(); in.nbVertices = (uint32_t)nbV; in.cellVertices = cv.data(); in.cellAdjacent = ca.data(); in.nbCells = (uint32_t)nbC;
        in.vertexCellsOffset = vco.data(); in.vertexCells = vcl.data();
        in.rayVertex = rv.data(); in.rayCamVertex = rc.data(); in.rayWeight = rw.data(); in.rayMaxDist = rd.data(); in.rayCamCenter = rcc.data();
        in.nbRays = (uint32_t)rv.size(); in.fullWeight = fullWeight; in.rayTedgeDist = forceTEdge ? rt.data() : nullptr;
        ALICEVISION_LOG_INFO("cheshire: " << in.nbRays << " rays, " << nbC << " cells, " << nbV << " vertices to the GPU" << (forceTEdge ? " (votes + weakly supported surfaces)." : "."));
        if (gpu::fillGraph(in, attr.data()))
        {
            for (size_t c = 0; c < nbC; ++c)
            {
                GC_cellInfo& ci = _cellsAttr[c];
                ci.cellSWeight = attr[8 * c]; ci.cellTWeight = attr[8 * c + 1];
                for (int k = 0; k < 4; ++k) ci.gEdgeVisWeight[k] = attr[8 * c + 2 + k];
                ci.emptinessScore = attr[8 * c + 6]; ci.on = attr[8 * c + 7];
            }
            return;
        }
        ALICEVISION_LOG_WARNING("cheshire: GPU meshing votes failed, falling back to the CPU loop");
    }
#endif
    // choose random order to prevent waiting
    const unsigned int seed = (unsigned int)_mp.userParams.get<unsigned int>("delaunaycut.seed", 0);
""", after=False)
    t = gfp.read_text(encoding="utf-8")
    # patch() inserted the block before the anchor and left the anchor: drop the duplicated two lines
    dup = """    // choose random order to prevent waiting
    const unsigned int seed = (unsigned int)_mp.userParams.get<unsigned int>("delaunaycut.seed", 0);
"""
    first = t.find(dup); second = t.find(dup, first + len(dup))
    if second != -1:
        t = t[:second] + t[second + len(dup):]
        gfp.write_text(t, encoding="utf-8", newline=NL)
    patch(gfp, "    const float forceTEdgeDelta = 0.1f;" + NL, """#ifdef ALICEVISION_HAVE_GPU_FILTER
    // cheshire: the ray loop below already ran on the GPU (gpu::fillGraph); only the final cellTWeight update is left.
    // CHESHIRE_GPU_TEDGE_CHECK=1: run the CPU loop as well and report how the two `on` vectors compare.
    std::vector<float> gpuOn;
    if (gpu::tedgesDone())
    {
        if (std::getenv("CHESHIRE_GPU_TEDGE_CHECK") == nullptr)
        {
            for (GC_cellInfo& c : _cellsAttr)
            {
                const float w = std::max(1.0f, c.cellTWeight) * c.on;
                c.cellTWeight = std::max(c.cellTWeight, std::min(1000000.0f, w));
            }
            return;
        }
        gpuOn.resize(_cellsAttr.size());
        for (size_t i = 0; i < _cellsAttr.size(); ++i) { gpuOn[i] = _cellsAttr[i].on; _cellsAttr[i].on = 0.0f; }
    }
#endif
""", after=False)
    patch(gfp, """    for (GC_cellInfo& c : _cellsAttr)
    {
        const float w = std::max(1.0f, c.cellTWeight) * c.on;
""", """#ifdef ALICEVISION_HAVE_GPU_FILTER
    if (!gpuOn.empty())
    {
        size_t cpuNz = 0, gpuNz = 0, mismatch = 0; double maxDiff = 0.0, sumCpu = 0.0, sumGpu = 0.0;
        for (size_t i = 0; i < _cellsAttr.size(); ++i)
        {
            const float c = _cellsAttr[i].on, g = gpuOn[i];
            cpuNz += c != 0.0f; gpuNz += g != 0.0f; sumCpu += c; sumGpu += g;
            const double d = std::fabs((double)c - (double)g);
            maxDiff = std::max(maxDiff, d);
            if (d > 1e-3 * std::max(1.0, (double)std::fabs(c))) ++mismatch;
            _cellsAttr[i].on = g;   // keep the GPU result, this was a check
        }
        ALICEVISION_LOG_INFO("cheshire: tedge check: cells with on != 0: cpu " << cpuNz << ", gpu " << gpuNz << "; sum cpu " << sumCpu << ", gpu " << sumGpu
                             << "; max |diff| " << maxDiff << "; cells beyond 1e-3 relative: " << mismatch);
    }
#endif
""", after=False)
    # the pairing scripts detect the GPU votes from --help: say so in the program description
    mm = AV / "src/software/pipeline/main_meshing.cpp"
    t = mm.read_text(encoding="utf-8")
    if "CHESHIRE_GPU_VOTE" not in t:
        old_desc = 'CmdLine cmdline("AliceVision meshing");'
        if old_desc not in t:
            sys.exit("meshing description not found")
        t = t.replace(old_desc, 'CmdLine cmdline("AliceVision meshing (cheshire: the graph-weight votes and the weakly-supported-surfaces pass run on the GPU when a device is present; CHESHIRE_GPU_VOTE=0 for the CPU passes)");', 1)
        mm.write_text(t, encoding="utf-8", newline=NL)
    fc = AV / "src/aliceVision/fuseCut/CMakeLists.txt"
    t = fc.read_text(encoding="utf-8")
    if "gpu/graphVoteGPU.cu" not in t:
        t = t.replace("    list(APPEND fuseCut_files_headers gpu/depthMapFilterGPU.hpp)", "    list(APPEND fuseCut_files_headers gpu/depthMapFilterGPU.hpp gpu/graphVoteGPU.hpp)", 1)
        t = t.replace("    list(APPEND fuseCut_files_sources gpu/depthMapFilterGPU.cu)", "    list(APPEND fuseCut_files_sources gpu/depthMapFilterGPU.cu gpu/graphVoteGPU.cu)", 1)
        t = t.replace("        set_source_files_properties(gpu/depthMapFilterGPU.cu PROPERTIES LANGUAGE HIP)", "        set_source_files_properties(gpu/depthMapFilterGPU.cu gpu/graphVoteGPU.cu PROPERTIES LANGUAGE HIP)", 1)
        fc.write_text(t, encoding="utf-8", newline=NL)


    # 4f. GPU texturing (hip/port/gpu_texturing): Texturing::generateTexturesSubSet's per-camera
    #     Laplacian pyramid + rasterisation and the final normalise/fuse on the GPU.
    gt_dst = AV / "src/aliceVision/mesh/gpu"
    gt_dst.mkdir(parents=True, exist_ok=True)
    for f in ("texturingGPU.hpp", "texturingGPU.cu"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_texturing" / f, gt_dst / f)
    tx = AV / "src/aliceVision/mesh/Texturing.cpp"
    patch(tx, '#include "Texturing.hpp"' + NL,
          '#include <cstdlib>  // cheshire' + NL + '#include <cstdint>' + NL + '#ifdef ALICEVISION_HAVE_GPU_TEX' + NL + '#include "aliceVision/mesh/gpu/texturingGPU.hpp"  // cheshire' + NL + '#include <chrono>' + NL + 'static int cheshirePrefetchDepth = 1;  // cameras read ahead of the one on the GPU (image cache slots - 1)' + NL + '#endif' + NL)
    # chunk size: what the card holds (the host keeps one atlas at a time on the GPU path)
    patch(tx, '    ALICEVISION_LOG_INFO("Total amount of available RAM: " << availableRam << " MB.");' + NL, """#ifdef ALICEVISION_HAVE_GPU_TEX
    // cheshire: the accumulators live in VRAM, so the chunk is what the card holds
    if (imageType != mvsUtils::EFileType::normalMap && gpu::texAvailable())
    {
        const int slots = gpu::maxAtlasSlots(texParams.textureSide, texParams.nbBand, mp.getMaxImageWidth(), mp.getMaxImageHeight(), (std::uint32_t)mesh->tris.size());
        if (slots >= 1)
        {
            nbAtlasMax = std::min(nbAtlas, slots);
            ALICEVISION_LOG_INFO("cheshire: " << slots << " atlas slots fit in VRAM, processing by chunks of " << nbAtlasMax);
            // the host holds no atlas pyramids on this path; if the images fit in RAM keep them all,
            // so a second chunk re-reads nothing (4 GB margin for the mesh, one atlas and the writer)
            const std::size_t allImagesMB = std::size_t(mp.ncams) * imageMaxMemSize;
            if (availableRam > 4096 && std::size_t(availableRam - 4096) > allImagesMB)
            {
                imageCache.setCacheSize(mp.ncams);
                cheshirePrefetchDepth = 4;
                ALICEVISION_LOG_INFO("cheshire: keeping all " << mp.ncams << " images in RAM (" << allImagesMB << " MB), reading 4 cameras ahead");
            }
            else if (availableRam > 4096 && std::size_t(availableRam - 4096) > 5 * imageMaxMemSize)
            {
                imageCache.setCacheSize(5);
                cheshirePrefetchDepth = 4;
                ALICEVISION_LOG_INFO("cheshire: reading 4 cameras ahead");
            }
        }
    }
#endif
""", after=False)
    patch(tx, '    ALICEVISION_LOG_INFO("Reading pixel color.");' + NL, """#ifdef ALICEVISION_HAVE_GPU_TEX
    // cheshire: per camera, upload the image; pyramid, rasterisation and the final fuse on the GPU
    if (imageType != mvsUtils::EFileType::normalMap && gpu::texAvailable())
    {
        const bool log = std::getenv("CHESHIRE_GPU_TEX_LOG") != nullptr;
        gpu::Texturer tex;
        bool ok = tex.init((int)atlasIDs.size(), texParams.textureSide, texParams.nbBand, texParams.multiBandDownscale, mp.getMaxImageWidth(), mp.getMaxImageHeight());
        if (ok)
        {
            // triangle tables for this chunk's atlases: 3D vertices and texture-pixel positions (UDIM tile remapped)
            const std::uint32_t nbTris = (std::uint32_t)mesh->tris.size();
            std::vector<double> tp(std::size_t(nbTris) * 9, 0.0), tpx(std::size_t(nbTris) * 6, 0.0);
            const StaticVector<Point2d>& uvCoords = mesh->uvCoords;
            for (const std::size_t atlasID : atlasIDs)
                for (std::size_t i = 0; i < _atlases[atlasID].size(); ++i)
                {
                    const int triangleId = _atlases[atlasID][i];
                    auto& triangleUvIds = mesh->trisUvIds[triangleId];
                    Point2d udimBL;
                    udimBL.x = std::floor(std::min({uvCoords[triangleUvIds[0]].x, uvCoords[triangleUvIds[1]].x, uvCoords[triangleUvIds[2]].x}));
                    udimBL.y = std::floor(std::min({uvCoords[triangleUvIds[0]].y, uvCoords[triangleUvIds[1]].y, uvCoords[triangleUvIds[2]].y}));
                    for (int k = 0; k < 3; ++k)
                    {
                        const Point3d& pt = mesh->pts[mesh->tris[triangleId].v[k]];
                        double* o = &tp[std::size_t(triangleId) * 9 + k * 3];
                        o[0] = pt.x; o[1] = pt.y; o[2] = pt.z;
                        Point2d uv = uvCoords[triangleUvIds.m[k]];
                        uv = uv - udimBL;
                        const Point2d pix = uv * texParams.textureSide;
                        tpx[std::size_t(triangleId) * 6 + k * 2] = pix.x;
                        tpx[std::size_t(triangleId) * 6 + k * 2 + 1] = pix.y;
                    }
                }
            ok = tex.setTriangles(nbTris, tp.data(), tpx.data());
        }
        double loadSec = 0.0;
        int prefetched = -1;   // highest camera id already handed to the cache's reader threads
        std::vector<std::uint32_t> ids;
        std::vector<float> scores;
        for (int camId = 0; ok && camId < (int)contributionsPerCamera.size(); ++camId)
        {
            const std::map<AtlasIndex, std::vector<ScorePerTriangle>>& cameraContributions = contributionsPerCamera[camId];
            if (cameraContributions.empty())
            {
                ALICEVISION_LOG_INFO("- camera " << mp.getViewId(camId) << " (" << camId + 1 << "/" << mp.ncams << ") unused.");
                continue;
            }
            ALICEVISION_LOG_INFO("- camera " << mp.getViewId(camId) << " (" << camId + 1 << "/" << mp.ncams << ") with contributions to "
                                             << cameraContributions.size() << " texture files:");
            const auto tl0 = std::chrono::steady_clock::now();
            auto imgPtr = imageCache.getImg_sync(camId);
            const image::Image<image::RGBfColor>& camImg = *imgPtr;
            loadSec += std::chrono::duration<double>(std::chrono::steady_clock::now() - tl0).count();

            if (camImg.width() != mp.getWidth(camId) || camImg.height() != mp.getHeight(camId))
            {
                ok = false;
                break;
            }
            const Matrix3x4& M = mp.camArr[camId];
            const double P[12] = {M.m11, M.m12, M.m13, M.m14, M.m21, M.m22, M.m23, M.m24, M.m31, M.m32, M.m33, M.m34};
            ok = tex.setCamera(reinterpret_cast<const float*>(camImg.data()), camImg.width(), camImg.height(), P);
            // read the next used cameras' images while this one is rasterised: the cache loads
            // different cameras concurrently, one thread each, up to its slot count minus this one
            for (int next = camId + 1, ahead = 0; next < (int)contributionsPerCamera.size() && ahead < cheshirePrefetchDepth; ++next)
                if (!contributionsPerCamera[next].empty())
                {
                    if (next > prefetched)
                    {
                        imageCache.refreshImage_async(next);
                        prefetched = next;
                    }
                    ++ahead;
                }
            for (const auto& c : cameraContributions)
            {
                if (!ok)
                    break;
                const AtlasIndex atlasID = c.first;
                const int slot = (int)(std::find(atlasIDs.begin(), atlasIDs.end(), atlasID) - atlasIDs.begin());
                ALICEVISION_LOG_INFO("  - Texture file: " << atlasID + 1);
                for (int band = 0; ok && band < (int)c.second.size(); ++band)
                {
                    const ScorePerTriangle& trianglesId = c.second[band];
                    ALICEVISION_LOG_INFO("      - band " << band + 1 << ": " << trianglesId.size() << " triangles.");
                    ids.resize(trianglesId.size());
                    scores.resize(trianglesId.size());
                    for (std::size_t i = 0; i < trianglesId.size(); ++i)
                    {
                        ids[i] = std::get<0>(trianglesId[i]);
                        scores[i] = texParams.useScore ? std::get<1>(trianglesId[i]) : 1.0f;
                    }
                    ok = tex.raster(slot, band, ids.data(), scores.data(), (std::uint32_t)ids.size());
                }
            }
        }
        for (std::size_t s = 0; ok && s < atlasIDs.size(); ++s)
        {
            const std::size_t atlasID = atlasIDs[s];
            ALICEVISION_LOG_INFO("Create texture " << atlasID + 1);
            AccuImage atlasTexture;
            atlasTexture.resize(texParams.textureSide, texParams.textureSide);
            ok = tex.finish((int)s, reinterpret_cast<float*>(atlasTexture.img.data()), atlasTexture.imgCount.data());
            if (!ok)
                break;
            writeTexture(atlasTexture, atlasID, outPath, textureFileType, -1, imageType);
        }
        if (log)
            ALICEVISION_LOG_INFO("cheshire texturing profile: image loads " << loadSec << " s, uploads " << tex.uploadSec << " s, pyramids " << tex.pyramidSec
                                                                            << " s, rasterisation " << tex.rasterSec << " s, finish " << tex.finishSec << " s");
        if (ok)
            return;
        ALICEVISION_LOG_WARNING("cheshire: GPU texturing failed, falling back to the CPU loop");
    }
#endif
""")
    # parallel triangle scoring (Texturing::generateTexturesSubSet): the single-threaded selection of
    # the best cameras per triangle scored into per-range lists in parallel and merged in triangle
    # order, so the lists are exactly the sequential ones
    patch(tx, """        // iterate over atlas' triangles
        for (size_t i = 0; i < _atlases[atlasID].size(); ++i)
        {
            int triangleID = _atlases[atlasID][i];
""", """        // cheshire: contiguous triangle ranges scored in parallel into their own lists, merged in order below
        const std::size_t nbAtlasTris = _atlases[atlasID].size();
        int nbParts = std::max(1, std::min(256, (int)(nbAtlasTris / 2048)));
        if (const char* pe = std::getenv("CHESHIRE_TEX_PARTS")) nbParts = std::max(1, std::atoi(pe));   // 1: the sequential loop
        std::vector<std::vector<std::map<AtlasIndex, std::vector<ScorePerTriangle>>>> partialContributions(
          nbParts, std::vector<std::map<AtlasIndex, std::vector<ScorePerTriangle>>>(mp.ncams));
#pragma omp parallel for schedule(static)
        for (int part = 0; part < nbParts; ++part)
        for (size_t i = nbAtlasTris * part / nbParts; i < nbAtlasTris * (part + 1) / nbParts; ++i)
        {
            auto& contributionsPerCamera = partialContributions[part];
            int triangleID = _atlases[atlasID][i];
""", after=False)
    t = tx.read_text(encoding="utf-8")
    dup = """        // iterate over atlas' triangles
        for (size_t i = 0; i < _atlases[atlasID].size(); ++i)
        {
            int triangleID = _atlases[atlasID][i];
"""
    if t.count(dup) == 1:
        t = t.replace(dup, "", 1)
        tx.write_text(t, encoding="utf-8", newline=NL)
    patch(tx, """                if (contrib + 1 == texParams.multiBandNbContrib[band])
                {
                    ++band;
                }
            }
        }
    }

    ALICEVISION_LOG_INFO("Reading pixel color.");""", """                if (contrib + 1 == texParams.multiBandNbContrib[band])
                {
                    ++band;
                }
            }
        }
        for (int part = 0; part < nbParts; ++part)
            for (int camId = 0; camId < mp.ncams; ++camId)
                for (auto& perAtlas : partialContributions[part][camId])
                {
                    auto& camContribution = contributionsPerCamera[camId];
                    if (camContribution.find(perAtlas.first) == camContribution.end())
                        camContribution[perAtlas.first].resize(texParams.nbBand);
                    auto& dst = camContribution.at(perAtlas.first);
                    for (std::size_t band = 0; band < perAtlas.second.size(); ++band)
                        dst[band].insert(dst[band].end(), perAtlas.second[band].begin(), perAtlas.second[band].end());
                }
    }
    if (std::getenv("CHESHIRE_GPU_TEX_LOG") != nullptr)
    {
        // order-sensitive checksum of every (camera, atlas, band) list, to compare CHESHIRE_TEX_PARTS=1 against the default
        std::uint64_t h = 1469598103934665603ull; std::size_t n = 0;
        for (std::size_t camId = 0; camId < contributionsPerCamera.size(); ++camId)
            for (const auto& perAtlas : contributionsPerCamera[camId])
                for (std::size_t band = 0; band < perAtlas.second.size(); ++band)
                    for (const auto& ts : perAtlas.second[band])
                    {
                        const std::uint64_t v[4] = {camId, perAtlas.first * 16 + band, ts.first, (std::uint64_t)ts.second};
                        for (std::uint64_t x : v) { h ^= x; h *= 1099511628211ull; }
                        ++n;
                    }
        ALICEVISION_LOG_INFO("cheshire: contributions checksum " << std::hex << h << std::dec << " over " << n << " entries");
    }

    ALICEVISION_LOG_INFO("Reading pixel color.");""", after=False)
    t = tx.read_text(encoding="utf-8")
    dup = """                if (contrib + 1 == texParams.multiBandNbContrib[band])
                {
                    ++band;
                }
            }
        }
    }

    ALICEVISION_LOG_INFO("Reading pixel color.");"""
    if t.count(dup) == 1:   # the original anchor left after the inserted block
        t = t.replace(dup, "", 1)
        tx.write_text(t, encoding="utf-8", newline=NL)
    # the pairing scripts detect the GPU pass from --help: say so in the program description
    mt = AV / "src/software/pipeline/main_texturing.cpp"
    t = mt.read_text(encoding="utf-8")
    if "CHESHIRE_GPU_TEX" not in t:
        old_desc = 'CmdLine cmdline("AliceVision texturing");'
        if old_desc not in t:
            sys.exit("texturing description not found")
        t = t.replace(old_desc, 'CmdLine cmdline("AliceVision texturing (cheshire: the per-camera pyramid and rasterisation run on the GPU when a device is present; CHESHIRE_GPU_TEX=0 for the CPU pass)");', 1)
        mt.write_text(t, encoding="utf-8", newline=NL)
    mc = AV / "src/aliceVision/mesh/CMakeLists.txt"
    patch(mc, "alicevision_add_library(aliceVision_mesh" + NL, MARK + """ (GPU texturing)
set(mesh_gpu_links "")
if (ALICEVISION_HAVE_CUDA OR ALICEVISION_HAVE_HIP)
    list(APPEND mesh_files_headers gpu/texturingGPU.hpp)
    list(APPEND mesh_files_sources gpu/texturingGPU.cu)
    if (ALICEVISION_HAVE_HIP)
        set_source_files_properties(gpu/texturingGPU.cu PROPERTIES LANGUAGE HIP)
    endif()
    set(mesh_gpu_links ${ALICEVISION_CUDA_LIBRARIES})
endif()

""", after=False)
    patch(mc, "        OpenMeshCore" + NL, "        ${mesh_gpu_links}" + NL)
    t = mc.read_text(encoding="utf-8")
    if "ALICEVISION_HAVE_GPU_TEX" not in t:
        i = t.index("alicevision_add_library(aliceVision_mesh")
        j = t.index(NL + ")" + NL, i) + 3
        t = t[:j] + MARK + """ (GPU texturing)
if (ALICEVISION_HAVE_CUDA OR ALICEVISION_HAVE_HIP)
    target_compile_definitions(aliceVision_mesh PRIVATE ALICEVISION_HAVE_GPU_TEX=1)
endif()
""" + t[j:]
        mc.write_text(t, encoding="utf-8", newline=NL)


    # 4g. Meshing CPU quick wins, all exact:
    #     - the cells-around-each-vertex table by counting instead of a map of sets (47 s -> ~1 s on
    #       21.7 M cells; same ascending, duplicate-free lists; CHESHIRE_MESH_OLD_NEIGHBOURS=1 keeps
    #       upstream's construction, and CHESHIRE_GPU_VOTE_LOG=1 logs a checksum of the table)
    #     - nanoflann index builds on every core (1.9 builds the same tree from any thread count)
    #     - timestamps around the graph build, the max-flow and the labelling in binarize
    tz = AV / "src/aliceVision/fuseCut/Tetrahedralization.cpp"
    patch(tz, "void Tetrahedralization::updateVertexToCellsCache(const size_t verticesCount)" + NL + "{" + NL,
          """    // cheshire: cells are visited in ascending order and each cell lists a vertex once, so
    // counting then appending gives exactly the ascending, duplicate-free lists the map of sets did
    if (std::getenv("CHESHIRE_MESH_OLD_NEIGHBOURS") == nullptr)
    {
        _neighboringCellsPerVertex.clear();
        _neighboringCellsPerVertex.resize(verticesCount);
        std::vector<std::uint32_t> counts(verticesCount, 0);
        const CellIndex nbC = nb_cells();
        for (CellIndex ci = 0; ci < nbC; ++ci)
            for (VertexIndex k = 0; k < 4; ++k)
            {
                const VertexIndex vi = cell_vertex(ci, k);
                if (vi == GEO::NO_VERTEX || vi >= verticesCount)
                    continue;
                ++counts[vi];
            }
        for (size_t vi = 0; vi < verticesCount; ++vi)
            _neighboringCellsPerVertex[vi].reserve(counts[vi]);
        for (CellIndex ci = 0; ci < nbC; ++ci)
            for (VertexIndex k = 0; k < 4; ++k)
            {
                const VertexIndex vi = cell_vertex(ci, k);
                if (vi == GEO::NO_VERTEX || vi >= verticesCount)
                    continue;
                _neighboringCellsPerVertex[vi].push_back(ci);
            }
        if (std::getenv("CHESHIRE_GPU_VOTE_LOG") != nullptr)
        {
            // build upstream's table as well and compare list for list
            std::map<VertexIndex, std::set<CellIndex>> tmp;
            for (CellIndex ci = 0; ci < nbC; ++ci)
                for (VertexIndex k = 0; k < 4; ++k)
                {
                    const VertexIndex vi = cell_vertex(ci, k);
                    if (vi == GEO::NO_VERTEX || vi >= verticesCount)
                        continue;
                    tmp[vi].insert(ci);
                }
            std::size_t bad = 0, n = 0;
            for (size_t vi = 0; vi < verticesCount; ++vi)
            {
                const auto it = tmp.find(vi);
                const std::vector<CellIndex> ref = it == tmp.end() ? std::vector<CellIndex>() : std::vector<CellIndex>(it->second.begin(), it->second.end());
                n += ref.size();
                if (ref != _neighboringCellsPerVertex[vi]) ++bad;
            }
            ALICEVISION_LOG_INFO("cheshire: neighbour table check: " << verticesCount << " vertices, " << n << " entries, lists differing from upstream's construction: " << bad);
        }
        return;
    }
""")
    patch(tz, """    _neighboringCellsPerVertex.resize(verticesCount);
    for (const auto& it : neighboringCellsPerVertexTmp)
    {
        const std::set<CellIndex>& input = it.second;
        std::vector<CellIndex>& output = _neighboringCellsPerVertex[it.first];
        output.assign(input.begin(), input.end());
    }
""", """""")
    t = tz.read_text(encoding="utf-8")
    if "#include <cstdlib>  // cheshire" not in t:
        i = t.index("#include")
        t = t[:i] + "#include <cstdlib>  // cheshire" + NL + "#include <cstdint>" + NL + "#include <aliceVision/system/Logger.hpp>" + NL + t[i:]
        tz.write_text(t, encoding="utf-8", newline=NL)
    for rel, old in (("src/aliceVision/fuseCut/PointCloud.cpp", "nanoflann::KDTreeSingleIndexAdaptorParams(MAX_LEAF_ELEMENTS)"),
                     ("src/aliceVision/fuseCut/Kdtree.hpp", "nanoflann::KDTreeSingleIndexAdaptorParams(MAX_LEAF_ELEMENTS)")):
        f = AV / rel
        t = f.read_text(encoding="utf-8")
        if "n_thread_build" not in t and "cheshireKdThreads()" not in t:
            new = "nanoflann::KDTreeSingleIndexAdaptorParams(MAX_LEAF_ELEMENTS, nanoflann::KDTreeSingleIndexAdaptorFlags::None, std::max(1u, std::thread::hardware_concurrency()))  /* cheshire: parallel build */"
            t = t.replace(old, new)
            i = t.index("#include")
            t = t[:i] + "#include <thread>  // cheshire" + NL + "#include <algorithm>" + NL + t[i:]
            f.write_text(t, encoding="utf-8", newline=NL)
    patch(gfp, "    const float CONSTalphaVIS = 1.0f;" + NL, "    ALICEVISION_LOG_INFO(\"cheshire: s-t edges added.\");" + NL, after=False)
    patch(gfp, "    // Find graph-cut solution" + NL, "    ALICEVISION_LOG_INFO(\"cheshire: graph built.\");" + NL, after=False)
    patch(gfp, "    _cellIsFull.resize(nbCells);" + NL, "    ALICEVISION_LOG_INFO(\"cheshire: max-flow done, labelling cells.\");" + NL, after=False)
    patch(mm, "                    gfiller.binarize();" + NL, "                    ALICEVISION_LOG_INFO(\"cheshire: binarize returned.\");" + NL)


    # 4h. CSR max-flow graph (hip/port/meshing_csr/MaxFlow_CSR.hpp): the same s-t graph laid out as a
    #     compressed sparse row graph with every node's out-edges in adjacency-list order, so
    #     Boykov-Kolmogorov sees the same edges in the same order; 66 s of graph build and 25 s of
    #     teardown become a few seconds. CHESHIRE_MAXFLOW_ADJLIST=1 keeps upstream's class,
    #     CHESHIRE_MAXFLOW_CHECK=1 runs both on the same graph and compares.
    shutil.copy2(ROOT / "hip" / "port" / "meshing_csr" / "MaxFlow_CSR.hpp", AV / "src/aliceVision/fuseCut/MaxFlow_CSR.hpp")
    # the GPU min-cut (hip/port/gpu_maxflow) next to the other fuseCut GPU sources
    for f in ("maxflowGPU.hpp", "maxflowGPU.cu"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_maxflow" / f, gv_dst / f)
    fc = AV / "src/aliceVision/fuseCut/CMakeLists.txt"
    t = fc.read_text(encoding="utf-8")
    if "gpu/maxflowGPU.cu" not in t:
        t = t.replace("gpu/depthMapFilterGPU.hpp gpu/graphVoteGPU.hpp)", "gpu/depthMapFilterGPU.hpp gpu/graphVoteGPU.hpp gpu/maxflowGPU.hpp)", 1)
        t = t.replace("gpu/depthMapFilterGPU.cu gpu/graphVoteGPU.cu)", "gpu/depthMapFilterGPU.cu gpu/graphVoteGPU.cu gpu/maxflowGPU.cu)", 1)
        t = t.replace("set_source_files_properties(gpu/depthMapFilterGPU.cu gpu/graphVoteGPU.cu PROPERTIES LANGUAGE HIP)", "set_source_files_properties(gpu/depthMapFilterGPU.cu gpu/graphVoteGPU.cu gpu/maxflowGPU.cu PROPERTIES LANGUAGE HIP)", 1)
        if "gpu/maxflowGPU.cu" not in t:
            sys.exit("fuseCut CMake GPU source lists not found")
        fc.write_text(t, encoding="utf-8", newline=NL)
    gfh = AV / "src/aliceVision/fuseCut/GraphFiller.hpp"
    patch(gfh, "    void binarize();" + NL, "    template<class MaxFlowT> float binarizeImpl(std::vector<bool>& cellIsFull);  // cheshire" + NL)
    patch(gfp, "#include <aliceVision/fuseCut/MaxFlow_AdjList.hpp>" + NL, "#include <aliceVision/fuseCut/MaxFlow_CSR.hpp>  // cheshire" + NL)
    t = gfp.read_text(encoding="utf-8")
    old_head = """void GraphFiller::binarize()
{
    const std::size_t nbCells = _cellsAttr.size();

    MaxFlow_AdjList maxFlowGraph(nbCells);
"""
    new_head = """void GraphFiller::binarize()
{
    // cheshire: the CSR graph by default (the same edges in the same order, a fraction of the build
    // and teardown time); CHESHIRE_MAXFLOW_ADJLIST=1 keeps upstream's adjacency list, and
    // CHESHIRE_MAXFLOW_CHECK=1 runs both on the same graph and compares flow value and labelling
    if (std::getenv("CHESHIRE_MAXFLOW_CHECK") != nullptr)
    {
        std::vector<bool> csrFull, adjFull;
        const float fCsr = binarizeImpl<MaxFlow_CSR>(csrFull);
        const float fAdj = binarizeImpl<MaxFlow_AdjList>(adjFull);
        std::size_t diff = 0;
        for (std::size_t i = 0; i < csrFull.size(); ++i)
            diff += (csrFull[i] != adjFull[i]);
        ALICEVISION_LOG_INFO("cheshire: max-flow check: CSR flow " << fCsr << ", adjacency-list flow " << fAdj << (fCsr == fAdj ? " (identical)" : " (DIFFERENT)")
                                                                  << "; cells labelled differently: " << diff << " of " << csrFull.size());
        _cellIsFull.swap(csrFull);
    }
    else if (std::getenv("CHESHIRE_MAXFLOW_ADJLIST") != nullptr)
        binarizeImpl<MaxFlow_AdjList>(_cellIsFull);
    else
        binarizeImpl<MaxFlow_CSR>(_cellIsFull);
    _cellsAttr.clear();
}

template<class MaxFlowT>
float GraphFiller::binarizeImpl(std::vector<bool>& cellIsFull)
{
    const std::size_t nbCells = _cellsAttr.size();

    MaxFlowT maxFlowGraph(nbCells);
"""
    if old_head in t:
        t = t.replace(old_head, new_head, 1)
        t = t.replace("""    //Clear graph
    _cellsAttr.clear();

""", "", 1)
        old_tail = """    _cellIsFull.resize(nbCells);
    std::size_t nbFullCells = 0;
    for (CellIndex ci = 0; ci < nbCells; ++ci)
    {
        _cellIsFull[ci] = maxFlowGraph.isTarget(ci);
        nbFullCells += _cellIsFull[ci];
    }
}"""
        new_tail = """    cellIsFull.resize(nbCells);
    std::size_t nbFullCells = 0;
    for (CellIndex ci = 0; ci < nbCells; ++ci)
    {
        cellIsFull[ci] = maxFlowGraph.isTarget(ci);
        nbFullCells += cellIsFull[ci];
    }
    ALICEVISION_LOG_INFO("cheshire: max-flow value " << totalFlow << ", full cells " << nbFullCells);
    return totalFlow;
}"""
        if old_tail not in t:
            sys.exit("binarize tail not found")
        t = t.replace(old_tail, new_tail, 1)
        gfp.write_text(t, encoding="utf-8", newline=NL)
    elif "binarizeImpl<MaxFlow_CSR>" not in t:
        sys.exit("binarize head not found")


    # 4i. parallel facet weights in binarize: two circumsphere centres and a few normalisations per
    #     facet, 87 M facets single-threaded upstream (30 s); computed in parallel into a table and
    #     the edges added in the same order with the same values (CHESHIRE_GPU_VOTE_LOG=1 recomputes
    #     them sequentially and compares).
    t = gfp.read_text(encoding="utf-8")
    old_loop = """    // fill u-v directed edges
    for (CellIndex ci = 0; ci < nbCells; ++ci)
    {
        for (VertexIndex k = 0; k < 4; ++k)
        {
            Facet fu(ci, k);
            Facet fv = _tetrahedralization.mirrorFacet(fu);
            if (_tetrahedralization.isInvalidOrInfiniteCell(fv.cellIndex))
            {
                continue;
            }

            float a1 = 0.0f;
            float a2 = 0.0f;
            if ((!_tetrahedralization.isInfiniteCell(fu.cellIndex)) && (!_tetrahedralization.isInfiniteCell(fv.cellIndex)))
            {
                // Score for each facet based on the quality of the topology
                a1 = _tetrahedralization.getFaceWeight(fu);
                a2 = _tetrahedralization.getFaceWeight(fv);
            }

            // In output of maxflow the cuts will become the surface.
            // High weight on some facets will avoid cutting them.
            float wFvFu = _cellsAttr[fu.cellIndex].gEdgeVisWeight[fu.localVertexIndex] * CONSTalphaVIS + a1 * CONSTalphaPHOTO;
            float wFuFv = _cellsAttr[fv.cellIndex].gEdgeVisWeight[fv.localVertexIndex] * CONSTalphaVIS + a2 * CONSTalphaPHOTO;

            maxFlowGraph.addEdge(fu.cellIndex, fv.cellIndex, wFuFv, wFvFu);
        }
    }
"""
    new_loop = """    // cheshire: the facet weights computed in parallel into a table (the same expressions per
    // facet), then the u-v edges added in upstream's order
    {
        struct FacetEdge { std::uint32_t fv; float wFuFv, wFvFu; };
        std::vector<FacetEdge> table(nbCells * 4);
        auto computeFacet = [&](CellIndex ci, VertexIndex k, FacetEdge& out) {
            Facet fu(ci, k);
            Facet fv = _tetrahedralization.mirrorFacet(fu);
            if (_tetrahedralization.isInvalidOrInfiniteCell(fv.cellIndex))
            {
                out.fv = 0xffffffffu; out.wFuFv = 0.0f; out.wFvFu = 0.0f;
                return;
            }
            float a1 = 0.0f;
            float a2 = 0.0f;
            if ((!_tetrahedralization.isInfiniteCell(fu.cellIndex)) && (!_tetrahedralization.isInfiniteCell(fv.cellIndex)))
            {
                // Score for each facet based on the quality of the topology
                a1 = _tetrahedralization.getFaceWeight(fu);
                a2 = _tetrahedralization.getFaceWeight(fv);
            }
            // In output of maxflow the cuts will become the surface.
            // High weight on some facets will avoid cutting them.
            out.fv = (std::uint32_t)fv.cellIndex;
            out.wFvFu = _cellsAttr[fu.cellIndex].gEdgeVisWeight[fu.localVertexIndex] * CONSTalphaVIS + a1 * CONSTalphaPHOTO;
            out.wFuFv = _cellsAttr[fv.cellIndex].gEdgeVisWeight[fv.localVertexIndex] * CONSTalphaVIS + a2 * CONSTalphaPHOTO;
        };
#pragma omp parallel for schedule(static)
        for (long long ci = 0; ci < (long long)nbCells; ++ci)
            for (VertexIndex k = 0; k < 4; ++k)
                computeFacet((CellIndex)ci, k, table[std::size_t(ci) * 4 + k]);
        if (std::getenv("CHESHIRE_GPU_VOTE_LOG") != nullptr)
        {
            std::size_t bad = 0;
            for (CellIndex ci = 0; ci < nbCells; ++ci)
                for (VertexIndex k = 0; k < 4; ++k)
                {
                    FacetEdge seq;
                    computeFacet(ci, k, seq);
                    const FacetEdge& par = table[std::size_t(ci) * 4 + k];
                    if (seq.fv != par.fv || seq.wFuFv != par.wFuFv || seq.wFvFu != par.wFvFu)
                        ++bad;
                }
            ALICEVISION_LOG_INFO("cheshire: facet weight check: " << nbCells * 4 << " facets, differing from the sequential computation: " << bad);
        }
        for (CellIndex ci = 0; ci < nbCells; ++ci)
            for (VertexIndex k = 0; k < 4; ++k)
            {
                const FacetEdge& e = table[std::size_t(ci) * 4 + k];
                if (e.fv == 0xffffffffu)
                    continue;
                maxFlowGraph.addEdge(ci, e.fv, e.wFuFv, e.wFvFu);
            }
    }
"""
    if old_loop in t:
        t = t.replace(old_loop, new_loop, 1)
        gfp.write_text(t, encoding="utf-8", newline=NL)
    elif "facet weight check" not in t:
        sys.exit("binarize edge loop not found")


    # 4j. DepthMapFilter depth-map cache: every neighbour depth map is decoded from EXR once per
    #     process instead of once per reference camera that lists it (about ten times); the values
    #     are the same bytes. CHESHIRE_FILTER_CACHE_MB caps it (default 4096, 0 disables).
    fz = AV / "src/aliceVision/fuseCut/Fuser.cpp"
    t = fz.read_text(encoding="utf-8")
    if "CheshireDepthMapCache" not in t:
        i = t.index(NL + "namespace aliceVision")   # after every include
        t = t[:i] + NL + """#include <memory>
#include <mutex>
#include <unordered_map>
namespace {
// cheshire: decoded neighbour depth maps shared across the reference cameras of this process
struct CheshireDepthMapCache
{
    struct Entry { std::shared_ptr<std::once_flag> once; std::shared_ptr<aliceVision::image::Image<float>> img; };
    std::mutex mutex;
    std::unordered_map<int, Entry> entries;
    std::size_t bytes = 0;
    std::size_t capBytes = 4096ull << 20;
    CheshireDepthMapCache() { if (const char* e = std::getenv("CHESHIRE_FILTER_CACHE_MB")) capBytes = std::size_t(std::atoll(e)) << 20; }
    std::shared_ptr<aliceVision::image::Image<float>> get(int tc, const aliceVision::mvsUtils::MultiViewParams& mp)
    {
        if (capBytes == 0)
        {
            auto img = std::make_shared<aliceVision::image::Image<float>>();
            aliceVision::mvsUtils::readMap(tc, mp, aliceVision::mvsUtils::EFileType::depthMap, *img);
            return img;
        }
        Entry e;
        {
            std::lock_guard<std::mutex> g(mutex);
            Entry& slot = entries[tc];
            if (!slot.once) { slot.once = std::make_shared<std::once_flag>(); slot.img = std::make_shared<aliceVision::image::Image<float>>(); }
            e = slot;
        }
        std::call_once(*e.once, [&] {
            aliceVision::mvsUtils::readMap(tc, mp, aliceVision::mvsUtils::EFileType::depthMap, *e.img);
            std::lock_guard<std::mutex> g(mutex);
            bytes += std::size_t(e.img->size()) * sizeof(float);
            if (bytes > capBytes)   // over budget: hand this one out but do not keep it
            {
                bytes -= std::size_t(e.img->size()) * sizeof(float);
                entries.erase(tc);
            }
        });
        return e.img;
    }
};
CheshireDepthMapCache& cheshireDepthMaps() { static CheshireDepthMapCache c; return c; }
}  // namespace
""" + t[i:]
        fz.write_text(t, encoding="utf-8", newline=NL)
    t = fz.read_text(encoding="utf-8")
    import re
    t2, n = re.subn(r"( +)image::Image<float> tcdepthMap;\n\1mvsUtils::readMap\(tc, _mp, mvsUtils::EFileType::depthMap, tcdepthMap\);\n",
                    lambda m: f"{m.group(1)}const std::shared_ptr<image::Image<float>> tcdepthMapPtr = cheshireDepthMaps().get(tc, _mp);  // cheshire\n{m.group(1)}image::Image<float>& tcdepthMap = *tcdepthMapPtr;\n", t)
    if n:
        fz.write_text(t2, encoding="utf-8", newline=NL)
    elif "cheshireDepthMaps().get" not in t:
        sys.exit("tc depth map reads not found in Fuser.cpp")


    # 4k. PrepareDenseScene: the per-image loop (read, exposure, undistort, EXR write) is pinned to
    #     three threads upstream; every image is independent, so run it on every core
    #     (CHESHIRE_PDS_THREADS overrides). Same bytes out.
    pds = AV / "src/software/pipeline/main_prepareDenseScene.cpp"
    t = pds.read_text(encoding="utf-8")
    if "CHESHIRE_PDS_THREADS" not in t:
        old = "#pragma omp parallel for num_threads(3)" + NL + "    for (int i = 0; i < viewIds.size(); ++i)" + NL
        if old not in t:
            sys.exit("prepareDenseScene loop not found")
        new = ("    // cheshire: one image per thread on every core (upstream: three threads); CHESHIRE_PDS_THREADS overrides" + NL
               + "    int cheshirePdsThreads = omp_get_max_threads();" + NL
               + "    if (const char* e = std::getenv(\"CHESHIRE_PDS_THREADS\")) cheshirePdsThreads = std::max(1, std::atoi(e));" + NL
               + "#pragma omp parallel for num_threads(cheshirePdsThreads)" + NL + "    for (int i = 0; i < viewIds.size(); ++i)" + NL)
        t = t.replace(old, new, 1)
        i = t.index("#include")
        t = t[:i] + "#include <cstdlib>  // cheshire" + NL + "#include <algorithm>" + NL + t[i:]
        pds.write_text(t, encoding="utf-8", newline=NL)
    # the pairing scripts detect it from --help
    t = pds.read_text(encoding="utf-8")
    if "CHESHIRE_PDS_THREADS for the thread count" not in t:
        old_desc = 'CmdLine cmdline("AliceVision prepareDenseScene");'
        if old_desc not in t:
            sys.exit("prepareDenseScene description not found")
        t = t.replace(old_desc, 'CmdLine cmdline("AliceVision prepareDenseScene (cheshire: one image per thread on every core; CHESHIRE_PDS_THREADS for the thread count)");', 1)
        pds.write_text(t, encoding="utf-8", newline=NL)


    # 4l. The visibility passes' nearest-neighbour search on the GPU (hip/port/gpu_knn): nanoflann's
    #     tree copied node for node and walked on the device exactly as nanoflann walks it, every
    #     query of a pass answered from one snapshot of the coordinates, votes applied in pixel
    #     order (deterministic, where upstream's locked updates race with its own tree reads).
    #     CHESHIRE_GPU_VIS=0 keeps upstream, CHESHIRE_GPU_VIS_CHECK=1 compares every query.
    for f in ("knnGPU.hpp", "knnGPU.cu", "visibilitiesGPU.inc"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_knn" / f, gv_dst / f)
    t = fc.read_text(encoding="utf-8")
    if "gpu/knnGPU.cu" not in t:
        t = t.replace("gpu/maxflowGPU.hpp)", "gpu/maxflowGPU.hpp gpu/knnGPU.hpp)", 1)
        t = t.replace("gpu/maxflowGPU.cu)", "gpu/maxflowGPU.cu gpu/knnGPU.cu)", 1)
        t = t.replace("gpu/maxflowGPU.cu PROPERTIES LANGUAGE HIP)", "gpu/maxflowGPU.cu gpu/knnGPU.cu PROPERTIES LANGUAGE HIP)", 1)
        if t.count("gpu/knnGPU.cu") != 2:
            sys.exit("fuseCut CMake GPU source lists not found for knnGPU")
        fc.write_text(t, encoding="utf-8", newline=NL)
    pc = AV / "src/aliceVision/fuseCut/PointCloud.cpp"
    patch(pc, "#include <aliceVision/fuseCut/Kdtree.hpp>" + NL,
          '#ifdef ALICEVISION_HAVE_GPU_FILTER' + NL + '#include "aliceVision/fuseCut/gpu/knnGPU.hpp"  // cheshire' + NL
          + '#include <chrono>' + NL + '#include <future>' + NL + '#include <cstdint>' + NL + '#include <cstdlib>' + NL + '#include <limits>' + NL + '#endif' + NL)
    t = pc.read_text(encoding="utf-8")
    if "createVerticesWithVisibilitiesUpstream" not in t:
        old = "void createVerticesWithVisibilities(const StaticVector<int>& cams,"
        if t.count(old) != 1:
            sys.exit("createVerticesWithVisibilities definition not found once")
        t = t.replace(old, "void createVerticesWithVisibilitiesUpstream(const StaticVector<int>& cams,  // cheshire: gpu/visibilitiesGPU.inc keeps the name", 1)
        end = '    ALICEVISION_LOG_INFO("Visibilities created.");' + NL + "}" + NL
        if t.count(end) != 1:
            sys.exit("end of createVerticesWithVisibilities not found once")
        t = t.replace(end, end + '#include "aliceVision/fuseCut/gpu/visibilitiesGPU.inc"  // cheshire' + NL, 1)
        pc.write_text(t, encoding="utf-8", newline=NL)

    # 4n. Direct OBJ writer in Mesh::save (upstream builds an Assimp scene and exports through it:
    #     6.6 s for the 1.2 M-vertex engine bay mesh, single-threaded text formatting plus a scene
    #     copy). The same numbers are written the same way (float, 9 significant digits, y and z
    #     negated as upstream does, 1-based faces), without the scene: a fraction of a second.
    #     CHESHIRE_OBJ_ASSIMP=1 keeps upstream; CHESHIRE_OBJ_CHECK=1 also writes upstream's file
    #     next to it (<file>.assimp.obj) for comparison.
    mc = AV / "src/aliceVision/mesh/Mesh.cpp"
    patch(mc, '    ALICEVISION_LOG_INFO("Saving " << fileTypeStr << " mesh file using Assimp.");' + NL, r"""
    // cheshire: OBJ straight to the file (see scripts/apply_hip_patch.py, step 4n)
    static thread_local bool cheshireForceAssimp = false;
    if (fileType == EFileType::OBJ && !cheshireForceAssimp && std::getenv("CHESHIRE_OBJ_ASSIMP") == nullptr)
    {
        ALICEVISION_LOG_INFO("Saving obj mesh file (cheshire direct writer): " << pts.size() << " vertices, " << tris.size() << " faces.");
        FILE* f = std::fopen(filepath.c_str(), "wb");
        if (f == nullptr)
            throw std::runtime_error("Cannot open mesh file for writing: " + filepath);
        std::vector<char> buf(1 << 22);
        std::size_t used = 0;
        auto flush = [&]() {
            if (used > 0)
            {
                std::fwrite(buf.data(), 1, used, f);
                used = 0;
            }
        };
        char line[160];
        int n = std::snprintf(line, sizeof(line), "# %zu vertex positions, %zu faces (cheshire)%c", pts.size(), tris.size(), 10);
        std::memcpy(buf.data(), line, n);
        used = n;
        for (const auto& p : pts)
        {
            const float x = p.x, y = -p.y, z = -p.z;
            n = std::snprintf(line, sizeof(line), "v %.9g %.9g %.9g%c", x, y, z, 10);
            if (used + n > buf.size()) flush();
            std::memcpy(buf.data() + used, line, n);
            used += n;
        }
        for (const auto& t : tris)
        {
            n = std::snprintf(line, sizeof(line), "f %d %d %d%c", t.v[0] + 1, t.v[1] + 1, t.v[2] + 1, 10);
            if (used + n > buf.size()) flush();
            std::memcpy(buf.data() + used, line, n);
            used += n;
        }
        flush();
        std::fclose(f);
        if (std::getenv("CHESHIRE_OBJ_CHECK") == nullptr)
            return;
        ALICEVISION_LOG_INFO("cheshire: CHESHIRE_OBJ_CHECK: also writing " << filepath << ".assimp.obj through Assimp");
        cheshireForceAssimp = true;
        save(filepath + ".assimp.obj");
        cheshireForceAssimp = false;
        return;
    }
""")
    t = mc.read_text(encoding="utf-8")
    if "#include <cstdio>  // cheshire" not in t:
        i = t.index("#include")
        t = t[:i] + "#include <cstdio>  // cheshire" + NL + "#include <cstdlib>" + NL + "#include <cstring>" + NL + t[i:]
        mc.write_text(t, encoding="utf-8", newline=NL)

    # 4o. Mesh cleaner: MeshClean::cleanMesh(maxIters) runs three consistency tests before the loop
    #     and after every iteration (14 full passes over the mesh on the engine bay); they only emit
    #     debug-level log lines and never change state. Off unless CHESHIRE_MESHCLEAN_TESTS=1; the
    #     iteration timing is logged.
    mcl = AV / "src/aliceVision/mesh/MeshClean.cpp"
    t = mcl.read_text(encoding="utf-8")
    if "CHESHIRE_MESHCLEAN_TESTS" not in t:
        old = """int MeshClean::cleanMesh(int maxIters)
{
    testPtsNeighTrisSortedAsc();
    testEdgesNeighTris();
"""
        if t.count(old) != 1:
            sys.exit("MeshClean::cleanMesh(int) head not found")
        new = """int MeshClean::cleanMesh(int maxIters)
{
    // cheshire: the consistency tests only log at debug level and never change the mesh; they are
    // 14 full passes over the mesh per call. CHESHIRE_MESHCLEAN_TESTS=1 keeps them.
    const bool cheshireTests = std::getenv("CHESHIRE_MESHCLEAN_TESTS") != nullptr;
    if (cheshireTests)
    {
        testPtsNeighTrisSortedAsc();
        testEdgesNeighTris();
    }
"""
        t = t.replace(old, new, 1)
        old2 = """        nupd = cleanMesh();
        testPtsNeighTrisSortedAsc();
        testEdgesNeighTris();
        testPtsNeighPtsOrdered();
"""
        if t.count(old2) != 1:
            sys.exit("MeshClean::cleanMesh(int) loop not found")
        new2 = """        const auto cheshireT0 = std::chrono::steady_clock::now();
        nupd = cleanMesh();
        const double cheshireMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cheshireT0).count();
        ALICEVISION_LOG_INFO("cheshire: cleanMesh iteration " << iter << ": " << cheshireMs << " ms");
        if (cheshireTests)
        {
            testPtsNeighTrisSortedAsc();
            testEdgesNeighTris();
            testPtsNeighPtsOrdered();
        }
"""
        t = t.replace(old2, new2, 1)
        i = t.index("#include")
        t = t[:i] + "#include <cstdlib>  // cheshire" + NL + "#include <chrono>" + NL + t[i:]
        mcl.write_text(t, encoding="utf-8", newline=NL)

    # 4p. The dense point cloud stage (42 s of Meshing on the engine bay: 20 s reading depth maps,
    #     20 s building a nanoflann tree over all 61 M raw slots, 1 s of queries).
    #     (a) the load loop runs one camera per thread on every core instead of three cameras with a
    #         nested inner loop; every point already lands in a fixed preallocated slot, so the
    #         result does not change. CHESHIRE_FUSION_THREADS overrides.
    #     (b) filterByPixSize decides in ascending score order in rounds (one legitimate execution of
    #         upstream's racy loop, and the same answer whatever the thread count) over a tree built
    #         from the candidate points only. hip/port/fusion_filter/filterFusion.inc has the
    #         reasoning and the margin check that makes leaving the rest out exact.
    #         CHESHIRE_FILTER_OLD=1 keeps upstream, CHESHIRE_FILTER_CHECK=1 runs both and reports.
    shutil.copy2(ROOT / "hip" / "port" / "fusion_filter" / "filterFusion.inc", AV / "src/aliceVision/fuseCut/filterFusion.inc")
    pc = AV / "src/aliceVision/fuseCut/PointCloud.cpp"
    t = pc.read_text(encoding="utf-8")
    if "filterByPixSizeUpstream" not in t:
        old = "void filterByPixSize(const std::vector<Point3d>& verticesCoordsPrepare,"
        if t.count(old) != 1:
            sys.exit("filterByPixSize definition not found once")
        t = t.replace(old, "void filterByPixSizeUpstream(const std::vector<Point3d>& verticesCoordsPrepare,  // cheshire: filterFusion.inc keeps the name", 1)
        end = '    ALICEVISION_LOG_INFO("Filtering done.");' + NL + "}" + NL
        if t.count(end) != 1:
            sys.exit("end of filterByPixSize not found once")
        t = t.replace(end, end + '#include "aliceVision/fuseCut/filterFusion.inc"  // cheshire' + NL, 1)
        pc.write_text(t, encoding="utf-8", newline=NL)
    t = pc.read_text(encoding="utf-8")
    if "CHESHIRE_FUSION_THREADS" not in t:
        old = ("        omp_set_nested(1);" + NL
               + "#pragma omp parallel for num_threads(3)" + NL
               + "        for (int c = 0; c < cams.size(); c++)" + NL
               + "        {" + NL
               + "            image::Image<float> depthMap;" + NL)
        if t.count(old) != 1:
            sys.exit("dense point cloud load loop not found once")
        new = ("        // cheshire: one camera per thread on every core (upstream: three cameras at a time, each with a" + NL
               + "        // nested parallel loop over blocks). Every point lands in a fixed preallocated slot," + NL
               + "        // index = startIndex[c] + sy * sxMax + sx, so the values do not depend on the thread count." + NL
               + "        // CHESHIRE_FUSION_THREADS overrides; with one camera per thread the inner loop stays serial." + NL
               + "        int cheshireFusionThreads = omp_get_max_threads();" + NL
               + "        if (const char* e = std::getenv(\"CHESHIRE_FUSION_THREADS\"))" + NL
               + "            cheshireFusionThreads = std::max(1, std::atoi(e));" + NL
               + "        ALICEVISION_LOG_INFO(\"cheshire: loading depth maps on \" << cheshireFusionThreads << \" threads.\");" + NL
               + "        omp_set_nested(cheshireFusionThreads > 1 ? 0 : 1);" + NL
               + "#pragma omp parallel for num_threads(cheshireFusionThreads)" + NL
               + "        for (int c = 0; c < cams.size(); c++)" + NL
               + "        {" + NL
               + "            image::Image<float> depthMap;" + NL)
        t = t.replace(old, new, 1)
        pc.write_text(t, encoding="utf-8", newline=NL)
    t = pc.read_text(encoding="utf-8")
    if "#include <string>  // cheshire" not in t:
        i = t.index("#include")
        t = t[:i] + "#include <string>  // cheshire" + NL + t[i:]
        pc.write_text(t, encoding="utf-8", newline=NL)

    # 4q. Grid helper points (PointCloud::addGridHelperPoints). Three things make the points differ
    #     from run to run: the generator is seeded from std::random_device whenever
    #     --seed is 0 (the default), one shared std::mt19937 is then drawn from inside an omp
    #     parallel region, so the order of the draws follows the scheduling (and concurrent calls to
    #     a generator are a data race), and the results are written into a std::vector<bool>, whose
    #     neighbouring elements share a word. The draws are now made in index order before the loop,
    #     which is the sequence a single thread produces, an unset seed uses a fixed value, and the
    #     flags are one byte each. A checksum of the points is logged so runs can be compared.
    #     CHESHIRE_GRID_RANDOM=1 restores random seeding, CHESHIRE_GRID_OLD=1 restores the shared
    #     generator (the torn-write fix stays either way).
    pc = AV / "src/aliceVision/fuseCut/PointCloud.cpp"
    t = pc.read_text(encoding="utf-8")
    if "CHESHIRE_GRID_RANDOM" not in t:
        old = ("    const unsigned int seed = (unsigned int)_mp.userParams.get<unsigned int>(\"delaunaycut.seed\", 0);" + NL
               + "    std::mt19937 generator(seed != 0 ? seed : std::random_device{}());" + NL)
        if t.count(old) != 1:
            sys.exit("addGridHelperPoints generator not found once")
        new = ("    const unsigned int seed = (unsigned int)_mp.userParams.get<unsigned int>(\"delaunaycut.seed\", 0);" + NL
               + "    // cheshire: an unset seed drew from std::random_device, so the helper points, and every" + NL
               + "    // mesh built on them, differed from run to run. A fixed value is used instead;" + NL
               + "    // --seed and CHESHIRE_GRID_RANDOM=1 both still ask for something else." + NL
               + "    const bool cheshireGridOld = std::getenv(\"CHESHIRE_GRID_OLD\") != nullptr;" + NL
               + "    const bool cheshireGridRandom = std::getenv(\"CHESHIRE_GRID_RANDOM\") != nullptr;" + NL
               + "    const unsigned int cheshireSeed = seed != 0 ? seed : (cheshireGridRandom ? std::random_device{}() : 1u);" + NL
               + "    std::mt19937 generator(cheshireSeed);" + NL)
        t = t.replace(old, new, 1)

        old = "    std::vector<bool> valid(gridVerticesCoords.size());" + NL
        if t.count(old) != 1:
            sys.exit("grid valid vector not found once")
        # neighbouring bits of a vector<bool> share a word, so the parallel writes below tear
        new = ("    std::vector<char> valid(gridVerticesCoords.size(), 0);  // cheshire: one byte each, the bits tore" + NL
               + "    // cheshire: the noise, drawn in index order before the loop. The loop below consumes three" + NL
               + "    // draws per grid vertex and visits the vertices in increasing index, so this is the sequence" + NL
               + "    // one thread produces; upstream drew from the shared generator inside the parallel region." + NL
               + "    std::vector<Point3d> cheshireNoise;" + NL
               + "    if (!cheshireGridOld)" + NL
               + "    {" + NL
               + "        cheshireNoise.resize(gridVerticesCoords.size());" + NL
               + "        for (std::size_t k = 0; k < cheshireNoise.size(); ++k)" + NL
               + "        {" + NL
               + "            const double nx = maxNoiseSize.x * rand();" + NL
               + "            const double ny = maxNoiseSize.y * rand();" + NL
               + "            const double nz = maxNoiseSize.z * rand();" + NL
               + "            cheshireNoise[k] = Point3d(nx, ny, nz);" + NL
               + "        }" + NL
               + "    }" + NL)
        t = t.replace(old, new, 1)

        old = "                const Point3d noise(maxNoiseSize.x * rand(), maxNoiseSize.y * rand(), maxNoiseSize.z * rand());" + NL
        if t.count(old) != 1:
            sys.exit("grid noise expression not found once")
        new = ("                const Point3d noise = cheshireGridOld" + NL
               + "                                        ? Point3d(maxNoiseSize.x * rand(), maxNoiseSize.y * rand(), maxNoiseSize.z * rand())" + NL
               + "                                        : cheshireNoise[i];  // cheshire" + NL)
        t = t.replace(old, new, 1)

        old = ("        _verticesCoords.push_back(gridVerticesCoords[i]);" + NL)
        if t.count(old) != 1:
            sys.exit("grid insert not found once")
        new = (old
               + "        {  // cheshire: a checksum of the points, so two runs can be compared" + NL
               + "            const unsigned char* bytes = reinterpret_cast<const unsigned char*>(gridVerticesCoords[i].m);" + NL
               + "            for (std::size_t b = 0; b < 3 * sizeof(double); ++b)" + NL
               + "                cheshireGridHash = (cheshireGridHash ^ (std::uint64_t)bytes[b]) * 1099511628211ull;" + NL
               + "        }" + NL)
        t = t.replace(old, new, 1)

        old = ("    _verticesAttr.reserve(_verticesAttr.size() + gridVerticesCoords.size());" + NL
               + "    int addedPoints = 0;" + NL)
        if t.count(old) != 1:
            sys.exit("grid addedPoints not found once")
        t = t.replace(old, old + "    std::uint64_t cheshireGridHash = 14695981039346656037ull;  // cheshire" + NL, 1)

        old = ('    ALICEVISION_LOG_WARNING("Add " << addedPoints << " new helper points for a 3D grid of "')
        if t.count(old) != 1:
            sys.exit("grid log not found once")
        new = ('    ALICEVISION_LOG_INFO("cheshire: grid helper points: seed " << cheshireSeed << ", " << addedPoints'
               + ' << " added, checksum " << std::hex << cheshireGridHash << std::dec);' + NL + old)
        t = t.replace(old, new, 1)
        pc.write_text(t, encoding="utf-8", newline=NL)

    # 4r. Checksums either side of the Delaunay tetrahedralisation. Two runs can produce the same
    #     number of cells with a different numbering, which changes every per-cell weight and the
    #     mesh that follows; these two lines say whether the input point order or geogram itself is
    #     responsible. Measured with them: the input is byte-identical on every run and the cells are
    #     not, and neither resetting geogram's generator nor disabling its threading changes that.
    #     Always logged: one pass over the vertices and the cells.
    tz = AV / "src/aliceVision/fuseCut/Tetrahedralization.cpp"
    t = tz.read_text(encoding="utf-8")
    if "cheshireTetraHash" not in t:
        old = "    tetrahedralization->set_vertices(_vertices.size(), _vertices.front().m);" + NL
        if t.count(old) != 1:
            sys.exit("set_vertices not found once")
        new = ("    // cheshire: geogram inserts the points in a biased randomised order, drawing from its own" + NL
               + "    // generator, so the same points came out as the same tetrahedra numbered differently on every" + NL
               + "    // run, and every per-cell weight and the mesh that follows moved with them. Resetting the" + NL
               + "    // generator fixes the insertion order. CHESHIRE_TETRA_RANDOM=1 restores geogram's own state." + NL
               + "    if (std::getenv(\"CHESHIRE_TETRA_RANDOM\") == nullptr)" + NL
               + "    {" + NL
               + "        GEO::Numeric::random_reset();" + NL
               + "        if (std::getenv(\"CHESHIRE_TETRA_SINGLE_THREAD\") != nullptr)" + NL
               + "            GEO::Process::enable_multithreading(false);" + NL
               + "    }" + NL
               + "    // a checksum of the points handed to geogram, in order" + NL
               + "    auto cheshireTetraHash = [](const void* data, std::size_t bytes) {" + NL
               + "        const unsigned char* p = static_cast<const unsigned char*>(data);" + NL
               + "        std::uint64_t h = 14695981039346656037ull;" + NL
               + "        for (std::size_t i = 0; i < bytes; ++i)" + NL
               + "            h = (h ^ (std::uint64_t)p[i]) * 1099511628211ull;" + NL
               + "        return h;" + NL
               + "    };" + NL
               + "    ALICEVISION_LOG_INFO(\"cheshire: tetrahedralization input: \" << _vertices.size() << \" points, checksum \"" + NL
               + "                         << std::hex << cheshireTetraHash(_vertices.front().m, _vertices.size() * 3 * sizeof(double)) << std::dec);" + NL
               + old)
        t = t.replace(old, new, 1)
        old = "    //Remove geogram data" + NL
        if t.count(old) != 1:
            sys.exit("geogram teardown not found once")
        new = ("    // cheshire: and of the cells geogram produced, in order" + NL
               + "    ALICEVISION_LOG_INFO(\"cheshire: tetrahedralization output: \" << _mesh.size() << \" cells, checksum \"" + NL
               + "                         << std::hex << cheshireTetraHash(_mesh.data(), _mesh.size() * sizeof(Cell)) << std::dec);" + NL
               + old)
        t = t.replace(old, new, 1)
        tz.write_text(t, encoding="utf-8", newline=NL)

    # 4s. Per-pass timing of the graph-cut post-processing block (9.5 s of Meshing on the engine bay,
    #     and until now one opaque number). Six passes, each timed.
    ms = AV / "src/aliceVision/fuseCut/Mesher.cpp"
    t = ms.read_text(encoding="utf-8")
    if "cheshirePostMark" not in t:
        old = '    ALICEVISION_LOG_INFO("Graph cut post-processing.");' + NL
        if t.count(old) != 1:
            sys.exit("post-processing banner not found once")
        new = (old
               + "    // cheshire: each pass of this block timed separately" + NL
               + "    auto cheshirePostMark = std::chrono::steady_clock::now();" + NL
               + "    auto cheshirePostMs = [&cheshirePostMark]() {" + NL
               + "        const auto now = std::chrono::steady_clock::now();" + NL
               + "        const double ms = std::chrono::duration<double, std::milli>(now - cheshirePostMark).count();" + NL
               + "        cheshirePostMark = now;" + NL
               + "        return ms;" + NL
               + "    };" + NL)
        t = t.replace(old, new, 1)
        for call, label in (("    removeBubbles();", "removeBubbles"),
                            ("    removeDust(minSegmentSize);", "removeDust"),
                            ("    invertFullStatusForSmallLabels();", "invertFullStatusForSmallLabels"),
                            ("    cellsStatusFilteringBySolidAngleRatio(nbSolidAngleFilteringIterations, minSolidAngleRatio);", "solidAngleFiltering")):
            if t.count(call + NL) != 1:
                sys.exit("post-cut call not found once: " + label)
            t = t.replace(call + NL,
                          call + NL + '    ALICEVISION_LOG_INFO("cheshire: post-cut ' + label + ': " << cheshirePostMs() << " ms");' + NL, 1)
        # the camera-vertex freeing loop and the neighbour-inversion rounds are inline blocks
        old = "    removeDust(minSegmentSize);" + NL
        t = t.replace(old, '    ALICEVISION_LOG_INFO("cheshire: post-cut cameraCells: " << cheshirePostMs() << " ms");' + NL + old, 1)
        old = "    cellsStatusFilteringBySolidAngleRatio(nbSolidAngleFilteringIterations, minSolidAngleRatio);" + NL
        t = t.replace(old, '    ALICEVISION_LOG_INFO("cheshire: post-cut neighbourInversion: " << cheshirePostMs() << " ms");' + NL + old, 1)
        if "#include <chrono>" not in t:
            i = t.index("#include")
            t = t[:i] + "#include <chrono>  // cheshire" + NL + t[i:]
        ms.write_text(t, encoding="utf-8", newline=NL)

    # 4t. Two exact reductions in the post-cut block (8.4 s on the engine bay, measured per pass by
    #     step 4s: solid-angle filtering 3.3 s, the three segmentation passes 4.9 s together).
    #     (a) segmentFullOrFree colours a cell when it is pushed rather than when it is popped, so a
    #         cell enters the stack once instead of up to four times. The colouring is identical: the
    #         seed order is unchanged and a cell's colour is the one its component's seed carries.
    #     (b) cellsStatusFilteringBySolidAngleRatio stops heap-allocating inside its inner loop. It
    #         built a three-element std::vector per neighbouring cell per surface vertex, tens of
    #         millions of allocations per round, and a fresh facet vector per vertex. Same values,
    #         same order: a fixed array and one buffer reused across the vertex's cells.
    ms = AV / "src/aliceVision/fuseCut/Mesher.cpp"
    t = ms.read_text(encoding="utf-8")
    if "cheshireTriangle" not in t:
        old = ("            buff.push_back(ci);" + NL + NL
               + "            while (buff.size() > 0)" + NL
               + "            {" + NL
               + "                CellIndex tmp_ci = buff.pop();" + NL + NL
               + "                out_fullSegsColor[tmp_ci] = col;" + NL)
        if t.count(old) != 1:
            sys.exit("segmentFullOrFree flood fill not found once")
        new = ("            buff.push_back(ci);" + NL
               + "            out_fullSegsColor[ci] = col;  // cheshire: colour on push, so a cell enters the stack once" + NL + NL
               + "            while (buff.size() > 0)" + NL
               + "            {" + NL
               + "                CellIndex tmp_ci = buff.pop();" + NL)
        t = t.replace(old, new, 1)
        old = ("                    if ((!_tetrahedralization.isInfiniteCell(nci)) && (out_fullSegsColor[nci] == -1) && (_cellIsFull[nci] == full))" + NL
               + "                    {" + NL
               + "                        buff.push_back(nci);" + NL)
        if t.count(old) != 1:
            sys.exit("segmentFullOrFree neighbour push not found once")
        new = ("                    if ((!_tetrahedralization.isInfiniteCell(nci)) && (out_fullSegsColor[nci] == -1) && (_cellIsFull[nci] == full))" + NL
               + "                    {" + NL
               + "                        out_fullSegsColor[nci] = col;  // cheshire" + NL
               + "                        buff.push_back(nci);" + NL)
        t = t.replace(old, new, 1)

        old = ("            const std::vector<CellIndex>& neighboringCells = neighboringCellsPerVertex[vi];" + NL
               + "            std::vector<Facet> neighboringFacets;" + NL
               + "            neighboringFacets.reserve(neighboringCells.size());" + NL)
        if t.count(old) != 1:
            sys.exit("solid angle facet vector not found once")
        new = ("            const std::vector<CellIndex>& neighboringCells = neighboringCellsPerVertex[vi];" + NL
               + "            // cheshire: one buffer per thread, reused across vertices, instead of an allocation each" + NL
               + "            static thread_local std::vector<Facet> neighboringFacets;" + NL
               + "            neighboringFacets.clear();" + NL
               + "            neighboringFacets.reserve(neighboringCells.size());" + NL)
        t = t.replace(old, new, 1)

        old = ("                std::vector<VertexIndex> triangle;" + NL
               + "                triangle.reserve(3);" + NL)
        if t.count(old) != 1:
            sys.exit("solid angle triangle vector not found once")
        new = ("                // cheshire: this ran once per neighbouring cell of every surface vertex" + NL
               + "                VertexIndex cheshireTriangle[4];" + NL
               + "                std::size_t cheshireTriangleSize = 0;" + NL)
        t = t.replace(old, new, 1)
        old = ("                    if (currentVertex != vi)" + NL
               + "                        triangle.push_back(currentVertex);" + NL
               + "                    else" + NL)
        if t.count(old) != 1:
            sys.exit("triangle push_back not found once")
        new = ("                    if (currentVertex != vi)" + NL
               + "                    {" + NL
               + "                        if (cheshireTriangleSize < 4)  // cheshire" + NL
               + "                            cheshireTriangle[cheshireTriangleSize] = currentVertex;" + NL
               + "                        ++cheshireTriangleSize;" + NL
               + "                    }" + NL
               + "                    else" + NL)
        t = t.replace(old, new, 1)
        old = "                if (triangle.size() != 3)" + NL
        if t.count(old) != 1:
            sys.exit("triangle size test not found once")
        t = t.replace(old, "                if (cheshireTriangleSize != 3)" + NL, 1)
        for k in (0, 1, 2):
            old = "triangle[%d]" % k
            if t.count(old) != 1:
                sys.exit("triangle[%d] not found once" % k)
            t = t.replace(old, "cheshireTriangle[%d]" % k, 1)
        # CHESHIRE_SEGMENT_CHECK=1: recompute the segmentation the way upstream does (colour at pop,
        # so a cell can enter the stack up to four times) and require the same colours and count
        old = "    out_nsegments = col;" + NL + "}" + NL
        if t.count(old) != 1:
            sys.exit("segmentFullOrFree tail not found once")
        new = ("    out_nsegments = col;" + NL
               + "    if (std::getenv(\"CHESHIRE_SEGMENT_CHECK\") != nullptr)" + NL
               + "    {" + NL
               + "        // upstream's order: colour when the cell is popped" + NL
               + "        StaticVector<int> refColour;" + NL
               + "        refColour.reserve(_cellIsFull.size());" + NL
               + "        refColour.resize_with(_cellIsFull.size(), -1);" + NL
               + "        StaticVector<CellIndex> refBuff;" + NL
               + "        refBuff.reserve(_cellIsFull.size());" + NL
               + "        int refCol = 0;" + NL
               + "        for (CellIndex ci = 0; ci < _cellIsFull.size(); ++ci)" + NL
               + "        {" + NL
               + "            if ((!_tetrahedralization.isInfiniteCell(ci)) && (refColour[ci] == -1) && (_cellIsFull[ci] == full))" + NL
               + "            {" + NL
               + "                refBuff.resize(0);" + NL
               + "                refBuff.push_back(ci);" + NL
               + "                while (refBuff.size() > 0)" + NL
               + "                {" + NL
               + "                    CellIndex tci = refBuff.pop();" + NL
               + "                    refColour[tci] = refCol;" + NL
               + "                    for (int k = 0; k < 4; ++k)" + NL
               + "                    {" + NL
               + "                        const CellIndex nci = _tetrahedralization.cell_adjacent(tci, k);" + NL
               + "                        if (nci == GEO::NO_CELL)" + NL
               + "                            continue;" + NL
               + "                        if ((!_tetrahedralization.isInfiniteCell(nci)) && (refColour[nci] == -1) && (_cellIsFull[nci] == full))" + NL
               + "                            refBuff.push_back(nci);" + NL
               + "                    }" + NL
               + "                }" + NL
               + "                ++refCol;" + NL
               + "            }" + NL
               + "        }" + NL
               + "        std::size_t differing = 0;" + NL
               + "        for (CellIndex ci = 0; ci < _cellIsFull.size(); ++ci)" + NL
               + "            differing += (out_fullSegsColor[ci] != refColour[ci]) ? 1 : 0;" + NL
               + "        if (differing == 0 && refCol == col)" + NL
               + "            ALICEVISION_LOG_INFO(\"cheshire: segmentFullOrFree check: identical to upstream on all \"" + NL
               + "                                 << _cellIsFull.size() << \" cells (\" << col << \" segments)\");" + NL
               + "        else" + NL
               + "            ALICEVISION_LOG_WARNING(\"cheshire: segmentFullOrFree check: \" << differing << \" of \"" + NL
               + "                                    << _cellIsFull.size() << \" cells differ, segments \" << col << \" vs \" << refCol);" + NL
               + "    }" + NL
               + "}" + NL)
        t = t.replace(old, new, 1)
        ms.write_text(t, encoding="utf-8", newline=NL)

    # 5. regenerate the reviewable patch
    subprocess.run(["git", "add", "-N", "src/aliceVision/depthMap/cuda/hip"], cwd=AV, check=True)
    diff = subprocess.run(["git", "diff", "--no-color"], cwd=AV, check=True, capture_output=True, text=True).stdout
    out = ROOT / "patches" / "0002-hip-backend-cmake.patch"
    out.write_text(diff, encoding="utf-8", newline="\n")
    print(f"applied; {len(diff.splitlines())} diff lines -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
