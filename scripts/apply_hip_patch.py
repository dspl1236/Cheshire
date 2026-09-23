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
import os
import re
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
    "CMakeLists.txt",
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
    "src/aliceVision/depthMap/cuda/host/DeviceCache.hpp",
    "src/aliceVision/depthMap/cuda/host/DeviceCache.cpp",
    "src/software/pipeline/main_depthMapEstimation.cpp",
    "src/aliceVision/image/io.cpp",
    "src/aliceVision/image/CMakeLists.txt",
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
    "src/aliceVision/feature/CMakeLists.txt",
    "src/aliceVision/fuseCut/Mesher.cpp",
    "src/aliceVision/mesh/Mesh.cpp",
    "src/aliceVision/mesh/MeshClean.cpp",
    "src/aliceVision/fuseCut/Kdtree.hpp",
    "src/aliceVision/fuseCut/GraphFiller.hpp",
    "src/software/pipeline/main_prepareDenseScene.cpp",
    "src/software/pipeline/main_meshing.cpp",
    "src/aliceVision/mesh/UVAtlas.hpp",
    "src/aliceVision/mesh/UVAtlas.cpp",
    "src/aliceVision/robustEstimation/ACRansac.hpp",
    "src/aliceVision/multiview/RelativePoseKernel.hpp",
    "src/aliceVision/image/imageAlgo.hpp",
    "src/aliceVision/image/imageAlgo.cpp",
    "src/aliceVision/numeric/algebra.hpp",
    "src/aliceVision/multiview/relativePose/Fundamental7PSolver.cpp",
    "src/aliceVision/sfm/bundle/BundleAdjustmentCeres.cpp",
    "src/aliceVision/sfm/bundle/BundleAdjustmentCeres.hpp",
    "src/aliceVision/sfm/pipeline/ReconstructionEngine.hpp",
    "src/aliceVision/sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.cpp",
    "src/aliceVision/sfm/bundle/costfunctions/intrinsicsProject.hpp",
    "src/aliceVision/sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.hpp",
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
    for f in ["cuda_to_hip.h", "bridge.h", "hip_to_cuda.h", "managed_cuda.h", "mipmap_emu.h", "devalloc.h"]:
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

    # 1h. parallel image prefetch per batch (image cache slot lock + omp prefetch loop); the
    #     prefetch block itself is hip/port/sgm_fused/prefetch.cpp.txt, since step 5t the
    #     once-per-batch loader (needs the accessors 5t adds)
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
        t = t[:i0] + "#if defined(CHESHIRE_HIP) || defined(CHESHIRE_BRIDGE_CUDA)\n" \
            + (ROOT / "hip/port/bridge_v2/planner.cpp.txt").read_text(encoding="utf-8") \
            + "#else\n" + original + "#endif\n" + t[i1:]
        # The planner needs cheshire::bridge and the macro that says which backend wired it up.
        # On HIP both arrive through cuda_to_hip.h via the shimmed <cuda_runtime.h>; on CUDA
        # memory.hpp does it (step 1j). Including memory.hpp here makes that true in both cases
        # rather than relying on a transitive include through DeviceCache.hpp.
        inc = '#include "DepthMapEstimator.hpp"\n'
        assert t.count(inc) == 1, "DepthMapEstimator.cpp self-include not found once"
        t = t.replace(inc, inc + "#include <aliceVision/depthMap/cuda/host/memory.hpp>  // cheshire: memory bridge\n", 1)
        dme.write_text(t, encoding="utf-8", newline="\n")

    # 1j. CUDA backend: route memory.hpp's device allocations through the memory bridge.
    #     cuda_to_hip.h cannot do this job here. It works on HIP only because the CUDA names are
    #     absent from the HIP headers, so its inline cudaMalloc/cudaMallocPitch/... shadow
    #     nothing; in a CUDA build <cuda_runtime.h> declares the real ones and they would clash.
    #     A function-like macro is not an option either: AliceVision calls
    #     cudaMallocPitch<Type>(&buf, ...), and a macro does not expand when the next token is
    #     '<' rather than '(', so that call - the main one - would silently bypass the bridge.
    #     Declaring the overloads inside namespace aliceVision::depthMap works because
    #     unqualified lookup searches the enclosing namespaces before the global one, so every
    #     call site in this file binds to them with no source change.
    #
    #     Note what the bridge does and does not cover on CUDA: camera mipmaps are real
    #     cudaMipmappedArrays here (HIP-Windows emulates them with buffers, mipmap_emu.h), and
    #     array allocations do not pass through cudaMalloc. So the Image class stays empty and
    #     the bridge manages volumes and maps - which docs/02 measured as the cheap classes to
    #     spill (5.6x and 1.9x) rather than the expensive one (images, 13x).
    mh = AV / "src/aliceVision/depthMap/cuda/host/memory.hpp"
    t = mh.read_text(encoding="utf-8")
    if "CHESHIRE_BRIDGE_CUDA" not in t:
        # (a) pull the bridge in at file scope. This must happen OUTSIDE the namespace: bridge.h
        #     includes <map>, <mutex>, <string> and friends, and including it inside
        #     aliceVision::depthMap would nest all of those in that namespace.
        inc_anchor = "#include <cuda_runtime.h>\n"
        assert t.count(inc_anchor) == 1, "memory.hpp <cuda_runtime.h> include not found once"
        t = t.replace(inc_anchor, inc_anchor +
                      "\n"
                      "// --- cheshire memory bridge, CUDA backend --------------------------------------\n"
                      "// CHESHIRE_HIP is defined by cuda_to_hip.h, which a HIP build reaches through the\n"
                      "// shimmed <cuda_runtime.h> just included; there the bridge is already wired in and\n"
                      "// this must stay out of the way. Anything else is a real CUDA toolkit.\n"
                      "#ifndef CHESHIRE_HIP\n"
                      "#define CHESHIRE_BRIDGE_CUDA 1\n"
                      "#include \"../hip/cheshire/bridge.h\"\n"
                      "#include \"../hip/cheshire/managed_cuda.h\"\n"
                      "namespace cheshire { namespace devmem {\n"
                      "// Two placement strategies, picked once per process. The bridge places whole buffers\n"
                      "// by class; CUDA unified memory migrates pages on fault. CHESHIRE_CUDA_MANAGED=1\n"
                      "// selects the latter (docs/18). The mode is constant for the process, so an\n"
                      "// allocation and its free always agree on who owns the pointer.\n"
                      "inline cudaError_t malloc(void** p, size_t n)\n"
                      "{ return managed::enabled() ? managed::malloc(p, n) : bridge::malloc(p, n); }\n"
                      "inline cudaError_t mallocPitch(void** p, size_t* pitch, size_t w, size_t h)\n"
                      "{ return managed::enabled() ? managed::mallocPitch(p, pitch, w, h) : bridge::mallocPitch(p, pitch, w, h); }\n"
                      "inline cudaError_t malloc3D(cudaPitchedPtr* pp, cudaExtent e)\n"
                      "{ return managed::enabled() ? managed::malloc3D(pp, e) : bridge::malloc3D(pp, e); }\n"
                      "inline cudaError_t free(void* p)\n"
                      "{ return managed::enabled() ? managed::free(p) : bridge::free(p); }\n"
                      "}}  // namespace cheshire::devmem\n"
                      "#endif\n"
                      "\n"
                      "#if defined(CHESHIRE_BRIDGE_CUDA)\n"
                      "// CHESHIRE_BRIDGE=0 falls back to plain cudaMalloc; CHESHIRE_CUDA_MANAGED=1 to unified memory.\n"
                      "#define CHESHIRE_DEV_MALLOC(p, n)               cheshire::devmem::malloc(reinterpret_cast<void**>(p), (n))\n"
                      "#define CHESHIRE_DEV_MALLOC_PITCH(p, pi, w, h)  cheshire::devmem::mallocPitch(reinterpret_cast<void**>(p), (pi), (w), (h))\n"
                      "#define CHESHIRE_DEV_MALLOC_3D(pp, e)           cheshire::devmem::malloc3D((pp), (e))\n"
                      "#define CHESHIRE_DEV_FREE(p)                    cheshire::devmem::free(p)\n"
                      "#elif defined(CHESHIRE_HIP)\n"
                      "// Device allocations go through the bridge; CHESHIRE_BRIDGE=0 disables it at runtime.\n"
                      "#define CHESHIRE_DEV_MALLOC(p, n)               cheshire::bridge::malloc(reinterpret_cast<void**>(p), (n))\n"
                      "#define CHESHIRE_DEV_MALLOC_PITCH(p, pi, w, h)  cheshire::bridge::mallocPitch(reinterpret_cast<void**>(p), (pi), (w), (h))\n"
                      "#define CHESHIRE_DEV_MALLOC_3D(pp, e)           cheshire::bridge::malloc3D((pp), (e))\n"
                      "#define CHESHIRE_DEV_FREE(p)                    cheshire::bridge::free(p)\n"
                      "#else\n"
                      "#define CHESHIRE_DEV_MALLOC(p, n)               cudaMalloc(reinterpret_cast<void**>(p), (n))\n"
                      "#define CHESHIRE_DEV_MALLOC_PITCH(p, pi, w, h)  cudaMallocPitch(reinterpret_cast<void**>(p), (pi), (w), (h))\n"
                      "#define CHESHIRE_DEV_MALLOC_3D(pp, e)           cudaMalloc3D((pp), (e))\n"
                      "#define CHESHIRE_DEV_FREE(p)                    cudaFree(p)\n"
                      "#endif\n"
                      "// -------------------------------------------------------------------------------\n", 1)
        # (b) name the bridge at the call sites. Declaring overloads inside
        #     aliceVision::depthMap does NOT work: the arguments (float2, __half, cudaPitchedPtr,
        #     cudaExtent) all live in the global namespace, so ADL adds :: back into the
        #     candidate set and every call becomes ambiguous with the real CUDA declaration.
        #     Unqualified lookup stopping at the enclosing namespace does not suppress ADL, it
        #     merges with it. Macros at the five call sites are unambiguous and keep one code
        #     path for both backends (on HIP the bridge is the same function cuda_to_hip.h
        #     would have routed to).
        sites = [
            ("cudaMallocPitch<Type>(&buffer, &this->getPitchRef(), this->getUnpaddedBytesInRow(), this->getUnitsInDim(1))",
             "CHESHIRE_DEV_MALLOC_PITCH(&buffer, &this->getPitchRef(), this->getUnpaddedBytesInRow(), this->getUnitsInDim(1))"),
            ("cudaMalloc3D(&pitchDevPtr, extent)", "CHESHIRE_DEV_MALLOC_3D(&pitchDevPtr, extent)"),
            ("cudaError_t err = cudaFree(buffer);", "cudaError_t err = CHESHIRE_DEV_FREE(buffer);"),
            ("cudaMalloc(&buffer, this->getBytesUnpadded())", "CHESHIRE_DEV_MALLOC(&buffer, this->getBytesUnpadded())"),
            ("CHECK_CUDA_RETURN_ERROR(cudaFree(buffer));", "CHECK_CUDA_RETURN_ERROR(CHESHIRE_DEV_FREE(buffer));"),
        ]
        for old, new in sites:
            assert t.count(old) == 1, f"memory.hpp: expected exactly one '{old}'"
            t = t.replace(old, new, 1)
        mh.write_text(t, encoding="utf-8", newline="\n")

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

    # 3c. Bundle search paths. Two upstream defects in one line of the `bundle` target:
    #     it passes -DBUNDLE_LIBS_PATHS=${BUNDLE_LIBS_PATHS} unquoted AND without VERBATIM.
    #     Unquoted, the CMake list expands into separate command-line arguments, so
    #     MakeBundle.cmake receives only the FIRST path and silently drops the rest - the CUDA
    #     toolkit's lib64 has never actually been on that search path. Quoting alone is not
    #     enough either: ninja runs the command through sh, where the semicolons separate
    #     commands, and sh then tries to execute the paths ("Permission denied"). The COMMAND
    #     form with VERBATIM makes CMake escape each argument for the native shell, which is
    #     also correct for the Windows bundles.
    #
    #     Invisible for as long as every dependency was resolvable some other way; PopSIFT,
    #     installed outside the deps prefix, is the first one that was not.
    #     Upstream bug, not ours - candidate for a PR alongside #2179 / #2181.
    top = AV / "CMakeLists.txt"
    t = top.read_text(encoding="utf-8")
    if "VERBATIM" not in t:
        nl_top = "\r\n" if "\r\n" in t else "\n"
        old_bundle = (
            "add_custom_target(bundle" + nl_top
            + "    ${CMAKE_COMMAND}" + nl_top
            + "    -DBUNDLE_INSTALL_PREFIX=${ALICEVISION_BUNDLE_PREFIX}" + nl_top
            + "    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}" + nl_top
            + "    -DBUNDLE_LIBS_PATHS=${BUNDLE_LIBS_PATHS}" + nl_top
            + "    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}" + nl_top
            + "    -P ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/MakeBundle.cmake" + nl_top
            + ")" + nl_top)
        assert t.count(old_bundle) == 1, "bundle target block not found once in CMakeLists.txt"
        new_bundle = (
            "add_custom_target(bundle" + nl_top
            + "    COMMAND ${CMAKE_COMMAND}" + nl_top
            + "    -DBUNDLE_INSTALL_PREFIX=${ALICEVISION_BUNDLE_PREFIX}" + nl_top
            + "    -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}" + nl_top
            + "    \"-DBUNDLE_LIBS_PATHS=${BUNDLE_LIBS_PATHS}\"" + nl_top
            + "    -DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}" + nl_top
            + "    -P ${CMAKE_CURRENT_SOURCE_DIR}/src/cmake/MakeBundle.cmake" + nl_top
            + "    VERBATIM" + nl_top
            + ")" + nl_top)
        t = t.replace(old_bundle, new_bundle, 1)
        top.write_text(t, encoding="utf-8", newline="")

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
          '#include <cstdlib>  // cheshire' + NL + '#include <cstdint>' + NL + '#ifdef ALICEVISION_HAVE_GPU_TEX' + NL + '#include "aliceVision/mesh/gpu/texturingGPU.hpp"  // cheshire' + NL + '#include <chrono>' + NL + 'static int cheshirePrefetchDepth = 1;  // cameras read ahead of the one on the GPU (image cache slots - 1)' + NL + 'static bool cheshireAtlasPadded = false;  // the GPU already ran writeTexture edge padding on this atlas' + NL + 'static aliceVision::image::Image<aliceVision::image::RGBfColor>* cheshireResizedAtlas = nullptr;  // the GPU already downscaled this atlas (step 5j)' + NL + '#endif' + NL)
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
            // cheshire: the edge padding runs on the device before the download (step 5d)
            const int gpuPad = (!texParams.fillHoles && texParams.padding > 0) ? int(texParams.padding) * 3 : 0;
            // cheshire: the downscale too (step 5j): OIIO's lanczos3 resize transcribed on the device
            image::Image<image::RGBfColor> cheshireSmall;
            float* cheshireSmallPtr = nullptr;
            if (texParams.downscale > 1 && std::getenv(\"CHESHIRE_GPU_RESIZE\") == nullptr)
            {
                const int smallSide = int(texParams.textureSide) / int(texParams.downscale);   // not 'small': windows.h defines it
                cheshireSmall.resize(smallSide, smallSide);
                cheshireSmallPtr = reinterpret_cast<float*>(cheshireSmall.data());
            }
            ok = tex.finish((int)s, reinterpret_cast<float*>(atlasTexture.img.data()), atlasTexture.imgCount.data(), gpuPad, int(texParams.downscale), cheshireSmallPtr);
            if (!ok)
                break;
            cheshireAtlasPadded = gpuPad > 0;
            cheshireResizedAtlas = cheshireSmallPtr ? &cheshireSmall : nullptr;
            writeTexture(atlasTexture, atlasID, outPath, textureFileType, -1, imageType);
            cheshireAtlasPadded = false;
            cheshireResizedAtlas = nullptr;
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
    # vcpkg's OpenMeshConfig.cmake exports INTERFACE_COMPILE_OPTIONS "/bigobj" with no language
    # guard (pybind11 does the same thing correctly, as $<$<COMPILE_LANGUAGE:CXX>:/bigobj>), so it
    # reaches this target's CUDA sources too. nvcc on Windows reads a leading '/' as a file path
    # and dies with "A single input file is required for a non-link phase when an outputfile is
    # specified". mesh is the only CUDA-bearing target that links OpenMesh, which is why nothing
    # hit this before the GPU texturing port. Re-state the option, guarded.
    foreach(_om_target OpenMeshCore OpenMeshTools)
        if (TARGET ${_om_target})
            get_target_property(_om_opts ${_om_target} INTERFACE_COMPILE_OPTIONS)
            if (_om_opts AND "/bigobj" IN_LIST _om_opts)
                list(REMOVE_ITEM _om_opts "/bigobj")
                list(APPEND _om_opts "$<$<COMPILE_LANGUAGE:C,CXX>:/bigobj>")
                set_target_properties(${_om_target} PROPERTIES INTERFACE_COMPILE_OPTIONS "${_om_opts}")
            endif()
        endif()
    endforeach()
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
    CheshireDepthMapCache()
    {
        if (const char* e = std::getenv("CHESHIRE_FILTER_CACHE_MB")) capBytes = std::size_t(std::atoll(e)) << 20;
        if (capBytes == 0)
            ALICEVISION_LOG_INFO("cheshire: depth map filter cache: disabled by CHESHIRE_FILTER_CACHE_MB=0");
        else
            ALICEVISION_LOG_INFO("cheshire: depth map filter cache: cap " << (capBytes >> 20) << " MB (CHESHIRE_FILTER_CACHE_MB)");
    }
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
          + '#include <chrono>' + NL + '#include <future>' + NL + '#include <cstdint>' + NL + '#include <cstdlib>' + NL + '#include <limits>' + NL
          # the visibility pass's digest and CHECK comparison (visibilitiesGPU.inc). Not <cmath> or
          # <cstring>: step 4y adds those unconditionally, but only when the file does not already
          # name them, and naming them here, inside this #ifdef, would suppress its copies.
          + '#include <algorithm>' + NL + '#include <cstdio>' + NL + '#include <string>' + NL + '#endif' + NL)
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

    # 4u. The 4-neighbour inversion count. main_meshing.cpp puts an int under
    #     hallucinationsFiltering.invertTetrahedronBasedOnNeighborsNbIterations, and Mesher reads it
    #     back with get<bool>: boost's bool translator rejects "10", the default is returned and
    #     collapses to true, so the loop runs ONCE however many rounds were asked for. Read as an int.
    #     CHESHIRE_INVERT_OLD=1 restores upstream's single round. Each round's moves are logged.
    ms = AV / "src/aliceVision/fuseCut/Mesher.cpp"
    t = ms.read_text(encoding="utf-8")
    if "CHESHIRE_INVERT_OLD" not in t:
        old = ("    int invertTetrahedronBasedOnNeighborsNbIterations =" + NL
               + '      _mp.userParams.get<bool>("hallucinationsFiltering.invertTetrahedronBasedOnNeighborsNbIterations", 10);' + NL)
        if t.count(old) != 1:
            sys.exit("inversion iteration count not found once")
        new = ("    // cheshire: upstream reads this count with get<bool> while the command line stores an int, so" + NL
               + "    // boost returns the default and it collapses to 1: the loop below ran once whatever was asked" + NL
               + "    // for. CHESHIRE_INVERT_OLD=1 restores that." + NL
               + "    int invertTetrahedronBasedOnNeighborsNbIterations =" + NL
               + '      (std::getenv("CHESHIRE_INVERT_OLD") != nullptr)' + NL
               + '        ? (int)_mp.userParams.get<bool>("hallucinationsFiltering.invertTetrahedronBasedOnNeighborsNbIterations", 10)' + NL
               + '        : _mp.userParams.get<int>("hallucinationsFiltering.invertTetrahedronBasedOnNeighborsNbIterations", 10);' + NL
               + '    ALICEVISION_LOG_INFO("cheshire: neighbour inversion rounds requested: " << invertTetrahedronBasedOnNeighborsNbIterations);' + NL)
        t = t.replace(old, new, 1)
        old = ("                if (_cellIsFull[ci])" + NL
               + "                    ++movedToFull;" + NL
               + "                else" + NL
               + "                    ++movedToEmpty;" + NL
               + "            }" + NL)
        if t.count(old) != 1:
            sys.exit("inversion round tail not found once")
        new = (old
               + '            ALICEVISION_LOG_INFO("cheshire: neighbour inversion round " << i << ": " << movedToFull'
               + ' << " to full, " << movedToEmpty << " to empty");' + NL)
        t = t.replace(old, new, 1)
        if "#include <cstdlib>" not in t:
            i = t.index("#include")
            t = t[:i] + "#include <cstdlib>  // cheshire" + NL + t[i:]
        ms.write_text(t, encoding="utf-8", newline=NL)

    # 4v. GPU SIFT. PopSIFT's public headers include <cuda_runtime.h>, which resolves to this
    #     project's shim in depthMap/cuda/hip, and that pulls in <hip/hip_runtime.h>.
    #     ImageDescriber_SIFT_popSIFT.cpp is ordinary host C++, so the ROCm include directory is not
    #     on its path and the shim cannot be resolved. Hand it to the target that compiles the file.
    fc = AV / "src/aliceVision/feature/CMakeLists.txt"
    t = fc.read_text(encoding="utf-8")
    if "cheshire: GPU SIFT" not in t:
        old = "    target_link_libraries(aliceVision_feature PRIVATE PopSift::popsift)" + NL
        if t.count(old) != 1:
            sys.exit("PopSift link line not found once in feature/CMakeLists.txt")
        new = (old
               + "    # cheshire: GPU SIFT. PopSIFT's headers reach <cuda_runtime.h>, which is this project's" + NL
               + "    # shim, and that needs the HIP headers; this file is host C++, so add them here." + NL
               + "    if (DEFINED ENV{ROCM_PATH})" + NL
               + "        target_include_directories(aliceVision_feature PRIVATE \"$ENV{ROCM_PATH}/include\")" + NL
               + "        # and the runtime itself: the describer calls cudaDeviceReset()" + NL
               + "        if (WIN32)" + NL
               + "            target_link_libraries(aliceVision_feature PRIVATE \"$ENV{ROCM_PATH}/lib/amdhip64.lib\")" + NL
               + "        else()" + NL
               + "            target_link_libraries(aliceVision_feature PRIVATE \"$ENV{ROCM_PATH}/lib/libamdhip64.so\")" + NL
               + "        endif()" + NL
               + "    endif()" + NL)
        t = t.replace(old, new, 1)
        fc.write_text(t, encoding="utf-8", newline=NL)

    # 4w. AC-RANSAC's residual sort. Geometric filtering is 95 % of FeatureMatching's wall clock on
    #     the engine bay, and profiling AC-RANSAC put 48 % of that in one std::sort: the loop builds
    #     a std::pair<double,size_t> array per candidate model and sorts it so bestNFA can read the
    #     residuals in order. bestNFA reads e[k-1].first alone. The indices matter only when the
    #     model improves on minNFA, which is 72,381 of 278,454,716 sorts - 0.026 %.
    #
    #     So sort 8-byte keys on the common path and fall into the original pair sort on the rare
    #     one. Sorting the values alone gives the same value sequence as sorting the pairs, because
    #     pair ordering breaks ties by index and tied values are equal. The sort itself is a stable
    #     LSD radix over the double's bit pattern, valid because squared epipolar distances are
    #     non-negative; a sign bit anywhere falls back to std::sort.
    #
    #     Measured on the 107-photo engine bay, byte-identical match file: the sort phase 3437 s ->
    #     1035 s of CPU time and the stage 597 s -> 370 s. Iterations, sorts and improving models
    #     all came out identical, so the search followed the same path, not merely the same result.
    #     CHESHIRE_ACR_PAIRSORT=1 restores upstream's per-model pair sort.
    ac = AV / "src/aliceVision/robustEstimation/ACRansac.hpp"
    t = ac.read_text(encoding="utf-8")
    if "cheshireRadixSortResiduals" not in t:
        old = "#include <algorithm>" + NL
        if t.count(old) != 1:
            sys.exit("<algorithm> include not found once in ACRansac.hpp")
        t = t.replace(old, "#include <atomic>  // cheshire" + NL + "#include <bit>  // cheshire" + NL
                      + "#include <cstdint>  // cheshire" + NL + "#include <cstdio>  // cheshire" + NL
                      + "#include <cstdlib>  // cheshire" + NL + old, 1)

        old = ("/**" + NL + " * @brief Find best NFA and its index wrt square error threshold in e." + NL + " */" + NL)
        if t.count(old) != 1:
            sys.exit("bestNFA doc comment not found once")
        new = ("/**" + NL
               + " * @brief cheshire: sort squared residuals ascending, as their IEEE bit patterns." + NL
               + " *" + NL
               + " * A non-negative double orders the same as its bit pattern read as a uint64_t, so a stable" + NL
               + " * LSD radix over the eight bytes sorts the values with no comparisons and no branches to" + NL
               + " * mispredict. Byte positions where every key agrees are skipped, which covers most of the" + NL
               + " * exponent for a typical residual spread." + NL
               + " *" + NL
               + " * Anything the bit-pattern order cannot represent goes to std::sort instead: a negative" + NL
               + " * value, -0.0 or a NaN, all of which have a key above 0x7FF0000000000000 (+inf is exactly" + NL
               + " * equal to it and orders correctly). NaN is the one that matters. It breaks the strict weak" + NL
               + " * ordering std::sort requires, so upstream's order on such an array is unspecified - but it" + NL
               + " * is what upstream produces, and reproducing it is the point. Geometric filtering never sees" + NL
               + " * one (4.7 M residuals, zero non-finite); SfM's relative pose does, through" + NL
               + " * RelativePoseKernel_K, which builds F from E and can divide by a zero squaredNorm." + NL
               + " */" + NL
               + "/// cheshire: how many arrays went to std::sort instead. CHESHIRE_ACR_NANLOG=1 reports it." + NL
               + "struct CheshireAcrDeferred" + NL
               + "{" + NL
               + "    std::atomic<long long> n{0};" + NL
               + "    ~CheshireAcrDeferred()" + NL
               + "    {" + NL
               + "        if (std::getenv(\"CHESHIRE_ACR_NANLOG\") && n.load())" + NL
               + "            std::fprintf(stderr, \"[cheshire acransac] %lld residual arrays deferred to std::sort\\n\", n.load());" + NL
               + "    }" + NL
               + "};" + NL
               + "inline std::atomic<long long>& cheshireAcrDeferred() { static CheshireAcrDeferred d; return d.n; }" + NL
               + NL
               + "inline void cheshireRadixSortResiduals(std::vector<std::uint64_t>& keys, std::vector<std::uint64_t>& scratch)" + NL
               + "{" + NL
               + "    const std::size_t n = keys.size();" + NL
               + "    if (n < 2)" + NL
               + "        return;" + NL
               + NL
               + "    std::uint64_t* a = keys.data();" + NL
               + "    std::uint32_t hist[8][256] = {};" + NL
               + "    bool defer = false;" + NL
               + "    for (std::size_t i = 0; i < n; ++i)" + NL
               + "    {" + NL
               + "        const std::uint64_t k = a[i];" + NL
               + "        defer |= (k > 0x7FF0000000000000ULL);  // negative, -0.0 or NaN" + NL
               + "        for (int p = 0; p < 8; ++p)" + NL
               + "            ++hist[p][(k >> (p * 8)) & 0xFF];" + NL
               + "    }" + NL
               + NL
               + "    if (defer)" + NL
               + "    {" + NL
               + "        cheshireAcrDeferred().fetch_add(1, std::memory_order_relaxed);" + NL
               + "        std::sort(keys.begin(), keys.end(), [](std::uint64_t x, std::uint64_t y) {" + NL
               + "            return std::bit_cast<double>(x) < std::bit_cast<double>(y);" + NL
               + "        });" + NL
               + "        return;" + NL
               + "    }" + NL
               + NL
               + "    scratch.resize(n);" + NL
               + "    std::uint64_t* b = scratch.data();" + NL
               + "    for (int p = 0; p < 8; ++p)" + NL
               + "    {" + NL
               + "        if (hist[p][(a[0] >> (p * 8)) & 0xFF] == n)  // every key agrees: the pass is the identity" + NL
               + "            continue;" + NL
               + NL
               + "        std::uint32_t off[256];" + NL
               + "        std::uint32_t sum = 0;" + NL
               + "        for (int d = 0; d < 256; ++d)" + NL
               + "        {" + NL
               + "            off[d] = sum;" + NL
               + "            sum += hist[p][d];" + NL
               + "        }" + NL
               + "        for (std::size_t i = 0; i < n; ++i)" + NL
               + "        {" + NL
               + "            const std::uint64_t k = a[i];" + NL
               + "            b[off[(k >> (p * 8)) & 0xFF]++] = k;" + NL
               + "        }" + NL
               + "        std::swap(a, b);" + NL
               + "    }" + NL
               + "    if (a != keys.data())" + NL
               + "        std::copy(a, a + n, keys.data());" + NL
               + "}" + NL
               + NL
               + "/// cheshire: residual of a sorted entry, whether it carries its index or is a bare key." + NL
               + "inline double acrResidual(const ErrorIndex& e) { return e.first; }" + NL
               + "inline double acrResidual(std::uint64_t e) { return std::bit_cast<double>(e); }" + NL
               + NL
               + old)
        t = t.replace(old, new, 1)

        # bestNFA reads through acrResidual, so it takes either array
        old = ("inline ErrorIndex bestNFA(int startIndex,  // number of point required for estimation" + NL
               + "                          double logalpha0," + NL
               + "                          const std::vector<ErrorIndex>& e," + NL)
        if t.count(old) != 1:
            sys.exit("bestNFA signature not found once")
        t = t.replace(old,
                      "template<typename ResidualT>  // cheshire: ErrorIndex, or a bare residual key" + NL
                      + "inline ErrorIndex bestNFA(int startIndex,  // number of point required for estimation" + NL
                      + "                          double logalpha0," + NL
                      + "                          const std::vector<ResidualT>& e," + NL, 1)

        old = ("    for (size_t k = startIndex + 1; k <= n && e[k - 1].first <= maxThreshold; ++k)" + NL
               + "    {" + NL
               + "        double squaredResidual = e[k - 1].first;" + NL)
        if t.count(old) != 1:
            sys.exit("bestNFA loop not found once")
        t = t.replace(old,
                      "    for (size_t k = startIndex + 1; k <= n && acrResidual(e[k - 1]) <= maxThreshold; ++k)" + NL
                      + "    {" + NL
                      + "        double squaredResidual = acrResidual(e[k - 1]);" + NL, 1)

        old = ("    std::vector<ErrorIndex> vec_residuals(nData);  // [residual,index]" + NL
               + "    std::vector<double> vec_residuals_(nData);" + NL)
        if t.count(old) != 1:
            sys.exit("residual buffers not found once")
        t = t.replace(old,
                      "    std::vector<ErrorIndex> vec_residuals(nData);  // [residual,index], cheshire: only for a better model" + NL
                      + "    std::vector<double> vec_residuals_(nData);" + NL
                      + "    // cheshire: residual bits, sorted for every model; see cheshireRadixSortResiduals" + NL
                      + "    std::vector<std::uint64_t> vec_keys(nData), vec_keysScratch;" + NL
                      + "    const bool cheshirePairSort = (std::getenv(\"CHESHIRE_ACR_PAIRSORT\") != nullptr);" + NL, 1)

        # the common path
        old = ("                for (size_t i = 0; i < nData; ++i)" + NL
               + "                {" + NL
               + "                    const double error = vec_residuals_[i];" + NL
               + "                    vec_residuals[i] = ErrorIndex(error, i);" + NL
               + "                }" + NL
               + "                std::sort(vec_residuals.begin(), vec_residuals.end());" + NL
               + NL
               + "                // Most meaningful discrimination inliers/outliers" + NL
               + "                const ErrorIndex best =" + NL
               + "                  bestNFA(sizeSample, kernel.logalpha0(), vec_residuals, loge0, maxThreshold, vec_logc_n, vec_logc_k, kernel.errorVectorDimension());" + NL)
        if t.count(old) != 1:
            sys.exit("residual ordering block not found once")
        new = ("                // cheshire: bestNFA reads residuals only, so sort 8-byte keys rather than" + NL
               + "                // (residual, index) pairs. The value sequence is the same either way." + NL
               + "                ErrorIndex best;" + NL
               + "                if (cheshirePairSort)" + NL
               + "                {" + NL
               + "                    for (size_t i = 0; i < nData; ++i)" + NL
               + "                    {" + NL
               + "                        const double error = vec_residuals_[i];" + NL
               + "                        vec_residuals[i] = ErrorIndex(error, i);" + NL
               + "                    }" + NL
               + "                    std::sort(vec_residuals.begin(), vec_residuals.end());" + NL
               + "                    best = bestNFA(sizeSample, kernel.logalpha0(), vec_residuals, loge0, maxThreshold, vec_logc_n, vec_logc_k, kernel.errorVectorDimension());" + NL
               + "                }" + NL
               + "                else" + NL
               + "                {" + NL
               + "                    for (size_t i = 0; i < nData; ++i)" + NL
               + "                        vec_keys[i] = std::bit_cast<std::uint64_t>(vec_residuals_[i]);" + NL
               + "                    cheshireRadixSortResiduals(vec_keys, vec_keysScratch);" + NL
               + "                    // Most meaningful discrimination inliers/outliers" + NL
               + "                    best = bestNFA(sizeSample, kernel.logalpha0(), vec_keys, loge0, maxThreshold, vec_logc_n, vec_logc_k, kernel.errorVectorDimension());" + NL
               + "                }" + NL)
        t = t.replace(old, new, 1)

        # the rare path: the inlier indices have to come out in (residual, index) order
        old = ("                    // A better model was found" + NL
               + "                    better = true;" + NL)
        if t.count(old) != 1:
            sys.exit("better-model branch not found once")
        new = ("                    // A better model was found" + NL
               + "                    // cheshire: the inlier indices below have to come out in (residual, index)" + NL
               + "                    // order, which the key sort does not carry. Pay for the pair sort here, on" + NL
               + "                    // the 0.026 % of models that reach this branch." + NL
               + "                    if (!cheshirePairSort)" + NL
               + "                    {" + NL
               + "                        for (size_t i = 0; i < nData; ++i)" + NL
               + "                            vec_residuals[i] = ErrorIndex(vec_residuals_[i], i);" + NL
               + "                        std::sort(vec_residuals.begin(), vec_residuals.end());" + NL
               + "                    }" + NL
               + "                    better = true;" + NL)
        t = t.replace(old, new, 1)
        ac.write_text(t, encoding="utf-8", newline=NL)

    # 4x. The residual loop itself. PointFittingKernel::errors() calls the virtual error() once per
    #     correspondence, which stops the compiler inlining a ~15-flop epipolar distance and stops it
    #     vectorising the loop. RelativePoseKernel does not override error(), so going straight to the
    #     estimator is the same arithmetic in the same order: 558 s -> 383 s of CPU time on the engine
    #     bay, match file unchanged.
    rp = AV / "src/aliceVision/multiview/RelativePoseKernel.hpp"
    t = rp.read_text(encoding="utf-8")
    if "cheshire: the inherited errors()" not in t:
        old = ("    double logalpha0() const override { return _logalpha0; }" + NL
               + "    double errorVectorDimension() const override { return (_pointToLine) ? 1.0 : 2.0; }" + NL)
        if t.count(old) != 1:
            sys.exit("RelativePoseKernel accessors not found once")
        new = ("    // cheshire: the inherited errors() calls the virtual error() once per correspondence," + NL
               + "    // which stops the compiler inlining a ~15-flop functor and stops it vectorising the" + NL
               + "    // loop. This class does not override error(), so going straight to the estimator is the" + NL
               + "    // same arithmetic." + NL
               + "    void errors(const ModelT_& model, std::vector<double>& errors) const override" + NL
               + "    {" + NL
               + "        const std::size_t n = PFRansacKernel::PFKernel::_x1.cols();" + NL
               + "        errors.resize(n);" + NL
               + "        for (std::size_t i = 0; i < n; ++i)" + NL
               + "            errors[i] = PFRansacKernel::PFKernel::_errorEstimator.error(" + NL
               + "              model, PFRansacKernel::PFKernel::_x1.col(i), PFRansacKernel::PFKernel::_x2.col(i));" + NL
               + "    }" + NL
               + NL
               + old)
        t = t.replace(old, new, 1)
        rp.write_text(t, encoding="utf-8", newline=NL)

    # 4y. The similarity-map gaussian of point-cloud fusion, on the GPU. Profiling the block it sits
    #     in (CHESHIRE_FUSION_PROFILE=1) put 108.7 of 180.8 thread-seconds there - 60 %, against 65.5
    #     for all three EXR reads together, in a block named "Load depth maps and add points". OIIO's
    #     ImageBufAlgo::convolve does not exploit separability, so it is 121 taps per pixel over
    #     2016x1134 maps, 107 times, and at twelve threads it is memory bound: 0.144 s isolated
    #     against about 1.0 thread-second in the pipeline. See hip/port/gpu_blur/simBlurGPU.hpp for
    #     what had to be matched to reproduce OIIO's output. CHESHIRE_GPU_BLUR=0 keeps OIIO.
    gb_dst = AV / "src/aliceVision/fuseCut/gpu"
    gb_dst.mkdir(parents=True, exist_ok=True)
    for f in ("simBlurGPU.hpp", "simBlurGPU.cu"):
        shutil.copy2(ROOT / "hip" / "port" / "gpu_blur" / f, gb_dst / f)

    # the weights OIIO builds, exposed so fuseCut can reproduce the convolution without OIIO
    ia = AV / "src/aliceVision/image/imageAlgo.hpp"
    t = ia.read_text(encoding="utf-8")
    if "gaussianKernelWeights" not in t:
        old = "void convolveImage(const image::Image<float>& inBuffer," + NL
        if t.count(old) != 1:
            sys.exit("convolveImage(float) declaration not found once in imageAlgo.hpp")
        new = ("// cheshire: the weights OIIO's make_kernel produces, so a caller can reproduce" + NL
               + "// convolveImage on another device without depending on OpenImageIO. The origin is" + NL
               + "// relative to the output pixel and is negative, as OIIO centres the kernel." + NL
               + "void gaussianKernelWeights(float kernelWidth," + NL
               + "                           float kernelHeight," + NL
               + "                           std::vector<float>& weights," + NL
               + "                           int& kw," + NL
               + "                           int& kh," + NL
               + "                           int& kx0," + NL
               + "                           int& ky0);" + NL
               + NL
               + old)
        t = t.replace(old, new, 1)
        if "#include <vector>" not in t:
            t = t.replace("#pragma once" + NL, "#pragma once" + NL + "#include <vector>  // cheshire" + NL, 1)
        ia.write_text(t, encoding="utf-8", newline=NL)

    ic = AV / "src/aliceVision/image/imageAlgo.cpp"
    t = ic.read_text(encoding="utf-8")
    if "gaussianKernelWeights" not in t:
        old = "void convolveImage(const image::Image<unsigned char>& inBuffer," + NL
        if t.count(old) != 1:
            sys.exit("convolveImage(uchar) definition not found once in imageAlgo.cpp")
        new = ("// cheshire: see the header." + NL
               + "void gaussianKernelWeights(float kernelWidth, float kernelHeight," + NL
               + "                           std::vector<float>& weights, int& kw, int& kh, int& kx0, int& ky0)" + NL
               + "{" + NL
               + "    oiio::ImageBuf K = oiio::ImageBufAlgo::make_kernel(\"gaussian\", kernelWidth, kernelHeight);" + NL
               + "    const oiio::ImageSpec& s = K.spec();" + NL
               + "    kw = s.width;" + NL
               + "    kh = s.height;" + NL
               + "    kx0 = s.x;" + NL
               + "    ky0 = s.y;" + NL
               + "    weights.resize(static_cast<std::size_t>(kw) * kh);" + NL
               + "    K.get_pixels(oiio::ROI(kx0, kx0 + kw, ky0, ky0 + kh, 0, 1, 0, 1), oiio::TypeDesc::FLOAT, weights.data());" + NL
               + "}" + NL
               + NL
               + old)
        t = t.replace(old, new, 1)
        ic.write_text(t, encoding="utf-8", newline=NL)

    # fuseCut builds the new device source alongside the existing ones
    fc = AV / "src/aliceVision/fuseCut/CMakeLists.txt"
    t = fc.read_text(encoding="utf-8")
    if "simBlurGPU" not in t:
        # earlier steps append their own sources to these lists, so add to whatever is there rather
        # than matching a fixed set: insert before the closing paren of each line.
        def appendTo(text, prefix, item, before=")"):
            i = text.index(prefix)
            j = text.index(before, i)
            return text[:j] + " " + item + text[j:]

        # the properties line ends with "... PROPERTIES LANGUAGE HIP)", so the file goes before the
        # keyword, not before the paren
        for prefix, item, before in (("    list(APPEND fuseCut_files_headers gpu/", "gpu/simBlurGPU.hpp", ")"),
                                     ("    list(APPEND fuseCut_files_sources gpu/", "gpu/simBlurGPU.cu", ")"),
                                     ("        set_source_files_properties(gpu/", "gpu/simBlurGPU.cu", " PROPERTIES")):
            if prefix not in t:
                sys.exit(f"fuseCut CMake line not found: {prefix.strip()}")
            t = appendTo(t, prefix, item, before)
        fc.write_text(t, encoding="utf-8", newline=NL)

    # and the call site
    pc = AV / "src/aliceVision/fuseCut/PointCloud.cpp"
    t = pc.read_text(encoding="utf-8")
    if "simBlurGPU.hpp" not in t:
        old = '#include "aliceVision/fuseCut/gpu/knnGPU.hpp"  // cheshire' + NL
        if t.count(old) != 1:
            sys.exit("knnGPU include not found once in PointCloud.cpp")
        t = t.replace(old, old + '#include "aliceVision/fuseCut/gpu/simBlurGPU.hpp"  // cheshire' + NL, 1)

        # build the kernel once: simGaussianSizeInit does not change between cameras
        old = '        ALICEVISION_LOG_INFO("cheshire: loading depth maps on " << cheshireFusionThreads << " threads.");' + NL
        if t.count(old) != 1:
            sys.exit("fusion thread log line not found once")
        new = (old
               + "        // cheshire: the gaussian below is 60 % of this block. Take the weights OIIO would" + NL
               + "        // build, once, and let the device apply them; CHESHIRE_GPU_BLUR=0 keeps OIIO." + NL
               + "        std::vector<float> chBlurKern;" + NL
               + "        int chBlurKw = 0, chBlurKh = 0, chBlurKx0 = 0, chBlurKy0 = 0;" + NL
               + "        const bool chBlurGpu = gpu::blurAvailable();" + NL
               + "        // CHESHIRE_GPU_BLUR_CHECK=1 also runs OIIO and reports how far apart they were." + NL
               + "        const bool chBlurCheck = std::getenv(\"CHESHIRE_GPU_BLUR_CHECK\") != nullptr;" + NL
               + "        std::atomic<long long> chBlurPix{0}, chBlurDiff{0};" + NL
               + "        double chBlurWorst = 0.0;" + NL
               + "        if (chBlurGpu)" + NL
               + "            imageAlgo::gaussianKernelWeights(params.simGaussianSizeInit, params.simGaussianSizeInit," + NL
               + "                                             chBlurKern, chBlurKw, chBlurKh, chBlurKx0, chBlurKy0);" + NL)
        t = t.replace(old, new, 1)

        old = ("                    image::Image<float> simMapTmp;" + NL
               + "                    imageAlgo::convolveImage(simMap, simMapTmp, \"gaussian\", params.simGaussianSizeInit, params.simGaussianSizeInit);" + NL
               + "                    simMap.swap(simMapTmp);" + NL)
        if t.count(old) != 1:
            sys.exit("sim map convolution not found once")
        new = ("                    image::Image<float> simMapTmp;" + NL
               + "                    bool chBlurred = false;" + NL
               + "                    if (chBlurGpu)" + NL
               + "                    {" + NL
               + "                        simMapTmp.resize(simMap.width(), simMap.height());" + NL
               + "                        chBlurred = gpu::blurGaussian(simMap.data(), simMapTmp.data()," + NL
               + "                                                      simMap.width(), simMap.height()," + NL
               + "                                                      chBlurKern.data(), chBlurKw, chBlurKh," + NL
               + "                                                      chBlurKx0, chBlurKy0);" + NL
               + "                    }" + NL
               + "                    if (!chBlurred)" + NL
               + "                        imageAlgo::convolveImage(simMap, simMapTmp, \"gaussian\", params.simGaussianSizeInit, params.simGaussianSizeInit);" + NL
               + "                    else if (chBlurCheck)" + NL
               + "                    {" + NL
               + "                        image::Image<float> chRef;" + NL
               + "                        imageAlgo::convolveImage(simMap, chRef, \"gaussian\", params.simGaussianSizeInit, params.simGaussianSizeInit);" + NL
               + "                        const std::size_t nPix = static_cast<std::size_t>(simMap.width()) * simMap.height();" + NL
               + "                        long long diff = 0;" + NL
               + "                        double worst = 0.0;" + NL
               + "                        for (std::size_t i = 0; i < nPix; ++i)" + NL
               + "                        {" + NL
               + "                            const float g = simMapTmp(i), o = chRef(i);" + NL
               + "                            if (std::memcmp(&g, &o, sizeof(float)) != 0)" + NL
               + "                            {" + NL
               + "                                ++diff;" + NL
               + "                                worst = std::max(worst, std::fabs((double)g - (double)o));" + NL
               + "                            }" + NL
               + "                        }" + NL
               + "                        chBlurPix.fetch_add((long long)nPix, std::memory_order_relaxed);" + NL
               + "                        chBlurDiff.fetch_add(diff, std::memory_order_relaxed);" + NL
               + "                        _Pragma(\"omp critical(cheshireBlurCheck)\")" + NL
               + "                        chBlurWorst = std::max(chBlurWorst, worst);" + NL
               + "                    }" + NL
               + "                    simMap.swap(simMapTmp);" + NL)
        t = t.replace(old, new, 1)

        old = "        omp_set_nested(0);" + NL
        if t.count(old) != 1:
            sys.exit("fusion block tail not found once")
        new = (old
               + "        if (chBlurCheck && chBlurPix.load() > 0)" + NL
               + "            ALICEVISION_LOG_INFO(\"cheshire: sim blur check: \" << chBlurDiff.load() << \" of \""
               + " << chBlurPix.load()" + NL
               + "                                 << \" pixels differ from OIIO, worst \" << chBlurWorst);" + NL)
        t = t.replace(old, new, 1)

        for inc in ("#include <atomic>", "#include <cstring>", "#include <cmath>"):
            if inc not in t:
                t = t.replace("#include <thread>  // cheshire" + NL,
                              "#include <thread>  // cheshire" + NL + inc + "  // cheshire" + NL, 1)
        pc.write_text(t, encoding="utf-8", newline=NL)

    # 4z. The chart packer. Texturing's Basic UV unwrap is 14 s, of which 6.28 s is
    #     createTextureAtlases: ChartRect::insert descends the whole tree for every chart, and
    #     occupied leaves return nullptr but are still visited, so each of 69,565 inserts re-walks
    #     thousands of full nodes. That is O(n^2) and about 90 us per insert for a tree descent.
    #
    #     Each node now carries the largest free extent anywhere beneath it, so the descent skips
    #     branches that provably cannot hold the chart. It is an upper bound - every free rectangle
    #     below has width <= maxFreeW and height <= maxFreeH - so no branch that could have fitted is
    #     ever skipped, and the same leaves are visited in the same depth-first order otherwise. The
    #     first fit is therefore the same one: createTextureAtlases 6.28 s -> 0.655 s, UVAtlas 14.01 s
    #     -> 9.20 s, and the 231 MB texturedMesh.obj is byte for byte the one upstream produces.
    uh = AV / "src/aliceVision/mesh/UVAtlas.hpp"
    t = uh.read_text(encoding="utf-8")
    if "maxFreeW" not in t:
        old = ("        Pixel LU;" + NL + "        Pixel RD;" + NL
               + "        void clear();" + NL
               + "        ChartRect* insert(Chart& chart, size_t gutter);" + NL)
        if t.count(old) != 1:
            sys.exit("ChartRect members not found once in UVAtlas.hpp")
        new = ("        Pixel LU;" + NL + "        Pixel RD;" + NL
               + "        // cheshire: the largest free extent anywhere in this subtree. insert() walked the" + NL
               + "        // whole tree for every chart, visiting thousands of already-full leaves, which is" + NL
               + "        // O(n^2) over 69,565 charts. These bound every free rectangle below, so a chart" + NL
               + "        // bigger than either cannot fit in any of them and the branch can be skipped. The" + NL
               + "        // same leaves are visited in the same order otherwise, so the packing is unchanged." + NL
               + "        std::size_t maxFreeW = 0;" + NL
               + "        std::size_t maxFreeH = 0;" + NL
               + "        void clear();" + NL
               + "        void refreshFree();" + NL
               + "        ChartRect* insert(Chart& chart, size_t gutter);" + NL)
        uh.write_text(t.replace(old, new, 1), encoding="utf-8", newline=NL)

    uc = AV / "src/aliceVision/mesh/UVAtlas.cpp"
    t = uc.read_text(encoding="utf-8")
    if "refreshFree" not in t:
        anchor = "UVAtlas::ChartRect* UVAtlas::ChartRect::insert(Chart& chart, size_t gutter)" + NL + "{" + NL
        if t.count(anchor) != 1:
            sys.exit("ChartRect::insert not found once")
        t = t.replace(anchor,
                      "// cheshire: recompute this node's bound from its children, or from itself when a leaf." + NL
                      + "void UVAtlas::ChartRect::refreshFree()" + NL
                      + "{" + NL
                      + "    if (child[0] || child[1])" + NL
                      + "    {" + NL
                      + "        maxFreeW = 0;" + NL
                      + "        maxFreeH = 0;" + NL
                      + "        for (ChartRect* ch : child)" + NL
                      + "            if (ch)" + NL
                      + "            {" + NL
                      + "                maxFreeW = std::max(maxFreeW, ch->maxFreeW);" + NL
                      + "                maxFreeH = std::max(maxFreeH, ch->maxFreeH);" + NL
                      + "            }" + NL
                      + "    }" + NL
                      + "    else" + NL
                      + "    {" + NL
                      + "        maxFreeW = c ? 0 : (std::size_t)(RD.x - LU.x);" + NL
                      + "        maxFreeH = c ? 0 : (std::size_t)(RD.y - LU.y);" + NL
                      + "    }" + NL
                      + "}" + NL + NL
                      + anchor, 1)

        old = ("    if (child[0] || child[1])  // not a leaf" + NL
               + "    {" + NL
               + "        if (child[0])" + NL
               + "            if (ChartRect* rect = child[0]->insert(chart, gutter))" + NL
               + "                return rect;" + NL
               + "        if (child[1])" + NL
               + "            if (ChartRect* rect = child[1]->insert(chart, gutter))" + NL
               + "                return rect;" + NL
               + "        return nullptr;" + NL
               + "    }" + NL)
        if t.count(old) != 1:
            sys.exit("insert descent not found once")
        new = ("    // cheshire: nothing below is big enough, so do not walk it" + NL
               + "    if (chart.targetWidth() + gutter * 2 > maxFreeW || chart.targetHeight() + gutter * 2 > maxFreeH)" + NL
               + "        return nullptr;" + NL + NL
               + "    if (child[0] || child[1])  // not a leaf" + NL
               + "    {" + NL
               + "        if (child[0])" + NL
               + "            if (ChartRect* rect = child[0]->insert(chart, gutter))" + NL
               + "            {" + NL
               + "                refreshFree();" + NL
               + "                return rect;" + NL
               + "            }" + NL
               + "        if (child[1])" + NL
               + "            if (ChartRect* rect = child[1]->insert(chart, gutter))" + NL
               + "            {" + NL
               + "                refreshFree();" + NL
               + "                return rect;" + NL
               + "            }" + NL
               + "        return nullptr;" + NL
               + "    }" + NL)
        t = t.replace(old, new, 1)

        old = ("        // insert chart" + NL + "        c = &chart;" + NL + "        return this;" + NL)
        if t.count(old) != 1:
            sys.exit("insert tail not found once")
        new = ("        // insert chart" + NL + "        c = &chart;" + NL
               + "        for (ChartRect* ch : child)" + NL
               + "            if (ch)" + NL
               + "                ch->refreshFree();" + NL
               + "        refreshFree();" + NL
               + "        return this;" + NL)
        t = t.replace(old, new, 1)

        old = ("        root->RD.x = _textureSide - 1;" + NL + "        root->RD.y = _textureSide - 1;" + NL)
        if t.count(old) != 1:
            sys.exit("atlas root extent not found once")
        t = t.replace(old, old + "        root->refreshFree();  // cheshire: seed the bound" + NL, 1)

        # packCharts: seven million single-element vectors, and an unreserved array that
        # reallocates its way to 7 M elements. See the comments the replacement carries.
        old = ("    // list mesh edges (with duplicates)" + NL
               + "    std::vector<Edge> alledges;" + NL
               + "    for (int i = 0; i < _mesh.tris.size(); ++i)" + NL
               + "    {" + NL
               + "        int a = _mesh.tris[i].v[0];" + NL
               + "        int b = _mesh.tris[i].v[1];" + NL
               + "        int c = _mesh.tris[i].v[2];" + NL
               + "        Edge e1;" + NL
               + "        e1.pointIDs = std::make_pair(std::min(a, b), std::max(a, b));" + NL
               + "        e1.triangleIDs.emplace_back(i);" + NL
               + "        alledges.emplace_back(e1);" + NL
               + "        Edge e2;" + NL
               + "        e2.pointIDs = std::make_pair(std::min(b, c), std::max(b, c));" + NL
               + "        e2.triangleIDs.emplace_back(i);" + NL
               + "        alledges.emplace_back(e2);" + NL
               + "        Edge e3;" + NL
               + "        e3.pointIDs = std::make_pair(std::min(c, a), std::max(c, a));" + NL
               + "        e3.triangleIDs.emplace_back(i);" + NL
               + "        alledges.emplace_back(e3);" + NL
               + "    }" + NL
               + "    std::sort(alledges.begin(), alledges.end());" + NL)
        if t.count(old) != 1:
            sys.exit("packCharts edge list not found once")
        new = ("    // cheshire: one contiguous array of 12-byte PODs rather than 7,029,609 Edge objects each" + NL
               + "    // holding a std::vector<int> with a single element. Same keys, same initial order, so" + NL
               + "    // std::sort makes the same comparisons and produces the same permutation - which matters" + NL
               + "    // because the merge below only compares adjacent entries, so on a non-manifold edge the" + NL
               + "    // tie order decides which pairs are emitted." + NL
               + "    struct RawEdge" + NL
               + "    {" + NL
               + "        int p0, p1, tri;" + NL
               + "    };" + NL
               + "    std::vector<RawEdge> alledges;" + NL
               + "    alledges.reserve(static_cast<std::size_t>(_mesh.tris.size()) * 3);" + NL
               + "    for (int i = 0; i < _mesh.tris.size(); ++i)" + NL
               + "    {" + NL
               + "        int a = _mesh.tris[i].v[0];" + NL
               + "        int b = _mesh.tris[i].v[1];" + NL
               + "        int c = _mesh.tris[i].v[2];" + NL
               + "        alledges.push_back({std::min(a, b), std::max(a, b), i});" + NL
               + "        alledges.push_back({std::min(b, c), std::max(b, c), i});" + NL
               + "        alledges.push_back({std::min(c, a), std::max(c, a), i});" + NL
               + "    }" + NL
               + "    // the same lexicographic order std::pair<int,int> gave" + NL
               + "    std::sort(alledges.begin(), alledges.end(), [](const RawEdge& x, const RawEdge& y) {" + NL
               + "        return x.p0 != y.p0 ? x.p0 < y.p0 : x.p1 < y.p1;" + NL
               + "    });" + NL)
        t = t.replace(old, new, 1)

        old = ("    // merge edges (no duplicate)" + NL
               + "    std::vector<Edge> edges;" + NL
               + "    auto eit = alledges.begin() + 1;" + NL
               + "    while (eit != alledges.end())" + NL
               + "    {" + NL
               + "        auto& a = *(eit - 1);" + NL
               + "        auto& b = *eit;" + NL
               + "        if (a == b)" + NL
               + "        {" + NL
               + "            a.triangleIDs.insert(a.triangleIDs.end(), b.triangleIDs.begin(), b.triangleIDs.end());" + NL
               + "            sort(a.triangleIDs.begin(), a.triangleIDs.end());" + NL
               + "            edges.push_back(a);" + NL
               + "        }" + NL
               + "        ++eit;" + NL
               + "    }" + NL
               + "    alledges.clear();" + NL)
        if t.count(old) != 1:
            sys.exit("packCharts merge not found once")
        new = ("    // merge edges (no duplicate)" + NL
               + "    // cheshire: upstream appended b's single id to a's list, sorted the two, and pushed a." + NL
               + "    // Each entry is the left of a pair exactly once and the right exactly once, and is read" + NL
               + "    // before it is written, so every emitted edge held exactly two ids in ascending order -" + NL
               + "    // the size() != 2 test below never fired. Store the pair directly." + NL
               + "    std::vector<std::pair<int, int>> edges;" + NL
               + "    for (std::size_t i = 1; i < alledges.size(); ++i)" + NL
               + "    {" + NL
               + "        const RawEdge& a = alledges[i - 1];" + NL
               + "        const RawEdge& b = alledges[i];" + NL
               + "        if (a.p0 == b.p0 && a.p1 == b.p1)" + NL
               + "            edges.emplace_back(std::min(a.tri, b.tri), std::max(a.tri, b.tri));" + NL
               + "    }" + NL
               + "    alledges.clear();" + NL
               + "    alledges.shrink_to_fit();" + NL)
        t = t.replace(old, new, 1)

        old = ("    for (auto& e : edges)" + NL
               + "    {" + NL
               + "        if (e.triangleIDs.size() != 2)" + NL
               + "            continue;" + NL
               + "        int chartIDA = findChart(e.triangleIDs[0]);" + NL
               + "        int chartIDB = findChart(e.triangleIDs[1]);" + NL)
        if t.count(old) != 1:
            sys.exit("packCharts chart merge not found once")
        new = ("    for (auto& e : edges)" + NL
               + "    {" + NL
               + "        int chartIDA = findChart(e.first);" + NL
               + "        int chartIDB = findChart(e.second);" + NL)
        t = t.replace(old, new, 1)

        old = "        auto cameras = trisCams[i];" + NL
        if t.count(old) != 1:
            sys.exit("createCharts camera list not found once")
        t = t.replace(old, "        // cheshire: a reference, not a copy of the camera list per triangle" + NL
               + "        const auto& cameras = trisCams[i];" + NL, 1)

        uc.write_text(t, encoding="utf-8", newline=NL)

    # 5a. The 7-point fundamental solver's nullspace. docs/15 left kernel.fit as the largest phase
    #     of FeatureMatching - 1589.4 thread-seconds, 39 % of the stage, 111.7 M calls - and it is
    #     Nullspace2 running Eigen's JacobiSVD with ComputeFullV on a 9x9.
    #
    #     The 9x9 has two zero rows. encodeEpipolarEquation writes one row per correspondence and
    #     the minimal case has seven, so this is a full iterative 9x9 SVD for a 7x9 problem whose
    #     nullspace is exactly two-dimensional. Householder-QR the 9x7 transpose instead: the last
    #     two columns of Q are orthonormal and orthogonal to every row of A, so they are a basis of
    #     that nullspace, from one non-iterative factorisation.
    #
    #     This is the first change here that is NOT bit-identical to upstream, and it cannot be: a
    #     different factorisation gives a different BASIS for the same nullspace. What it does not
    #     change is the answer. The solver then solves det(F1 + a*F2) = 0 over the pencil, and the
    #     pencil is a property of the subspace, not of the basis chosen for it, so the fundamental
    #     matrices agree to rounding.
    #
    #     Measured against upstream over 50,000 synthetic systems, using upstream's own cubic
    #     coefficients and root solver (scratch harness, docs/17):
    #       general position     nullspace 7.6x, whole fit 6.75x; worst model difference 3.96e-09
    #                            against a control of 3.97e-09 - the control being JacobiSVD on the
    #                            7x9 block versus the 9x9, the same algorithm on mathematically
    #                            identical input. QR is as close to upstream as upstream is to a
    #                            trivial reformulation of itself.
    #       near-degenerate      whole fit 6.66x. 275 root-count flips, of which 274 are the SAME
    #                            systems for QR and the control, one unique each; two systems give
    #                            NaN in upstream too. On the 49,722 stable systems the worst
    #                            difference is 1.183e-04 for QR against 1.176e-04 for the control.
    #                            The instability is the ill-conditioned system's, not the method's.
    #
    #     Ten reconstructions per leg settled the quality question, and reversed what three runs
    #     had suggested: QR loses 165 landmarks (p=0.0008) and adds 0.0024 to RMSE (p=0.032), both
    #     real, both about 0.12 %. A trade, not a free win - so upstream's JacobiSVD stays the
    #     DEFAULT and the QR path is opt-in with CHESHIRE_QR_NULLSPACE=1. The flag is read through
    #     a function-local static: solve() runs 111.7 M times and a getenv per call would cost more
    #     than the change saves.
    alg = AV / "src/aliceVision/numeric/algebra.hpp"
    t = alg.read_text(encoding="utf-8")
    if "Nullspace2RankDeficient" not in t:
        nl_alg = "\r\n" if "\r\n" in t else "\n"
        old = "#include <Eigen/Core>" + nl_alg + "#include <Eigen/SVD>" + nl_alg
        if t.count(old) != 1:
            sys.exit("Eigen includes not found once in algebra.hpp")
        t = t.replace(old, old + "#include <Eigen/QR>  // cheshire" + nl_alg, 1)

        anchor = "template<typename TMat, typename TVec1, typename TVec2>" + nl_alg + "inline double Nullspace2("
        if t.count(anchor) != 1:
            sys.exit("Nullspace2 template not found once in algebra.hpp")
        helper = (
            "/// cheshire: basis of the two-dimensional nullspace of a system that is exactly rank" + nl_alg
            + "/// deficient by two - the minimal 7-point case, seven independent rows in nine columns." + nl_alg
            + "/// Householder-QR the transpose: the last two columns of Q are orthonormal and" + nl_alg
            + "/// orthogonal to every row of A. Not for overdetermined systems, where the SVD's answer" + nl_alg
            + "/// is a least-squares fit that this does not reproduce." + nl_alg
            + "template<typename TMatA, typename TVec1, typename TVec2>" + nl_alg
            + "inline void Nullspace2RankDeficient(const TMatA& A, TVec1& x1, TVec2& x2)" + nl_alg
            + "{" + nl_alg
            + "    Eigen::Matrix<double, 9, 7> At = A.template topRows<7>().transpose();" + nl_alg
            + "    Eigen::HouseholderQR<Eigen::Matrix<double, 9, 7>> qr(At);" + nl_alg
            + "    const Eigen::Matrix<double, 9, 9> Q = qr.householderQ();" + nl_alg
            + "    x1 = Q.col(8);" + nl_alg
            + "    x2 = Q.col(7);" + nl_alg
            + "}" + nl_alg + nl_alg)
        t = t.replace(anchor, helper + anchor, 1)
        alg.write_text(t, encoding="utf-8", newline="")

    f7 = AV / "src/aliceVision/multiview/relativePose/Fundamental7PSolver.cpp"
    t = f7.read_text(encoding="utf-8")
    if "cheshireQrNullspace" not in t:
        nl_f7 = "\r\n" if "\r\n" in t else "\n"
        # The same two lines appear in the over-determined branch, where Nullspace2 is doing a
        # LEAST-SQUARES fit that a QR nullspace does not reproduce, and again in the spherical
        # solver. Anchor on the minimal branch's own preamble so only that one is touched.
        old = ("        Mat9 A = Mat::Zero(9, 9);" + nl_f7
               + "        encodeEpipolarEquation(x1, x2, &A);" + nl_f7)
        if t.count(old) != 1:
            sys.exit("minimal-branch preamble not found once in Fundamental7PSolver.cpp")
        tail = ("        // find the two F matrices in the nullspace of A." + nl_f7
                + "        Nullspace2(A, f1, f2);" + nl_f7)
        head_end = t.index(old) + len(old)
        rest = t[head_end:]
        if rest.count(tail) < 1:
            sys.exit("minimal-branch Nullspace2 call not found after the preamble")
        old = tail
        new = ("        // cheshire: A is 9x9 with two zero rows, so this is a 7x9 system whose nullspace is" + nl_f7
               + "        // exactly two-dimensional - a Householder QR of the transpose gives a basis for it" + nl_f7
               + "        // without an iterative SVD. A different basis of the same nullspace spans the same" + nl_f7
               + "        // pencil, so det(F1 + a*F2) = 0 has the same solutions to rounding." + nl_f7
               + "        static const bool cheshireQrNullspace = (std::getenv(\"CHESHIRE_QR_NULLSPACE\") != nullptr);" + nl_f7
               + "        static const bool cheshireQrAnnounced = []() {  // once per process, both paths (docs/04 marker rule)" + nl_f7
               + "            std::fprintf(stderr, cheshireQrNullspace ? \"[cheshire] 7-point nullspace: Householder QR (CHESHIRE_QR_NULLSPACE=1)\\n\"" + nl_f7
               + "                                                     : \"[cheshire] 7-point nullspace: SVD (default; CHESHIRE_QR_NULLSPACE=1 for QR)\\n\");" + nl_f7
               + "            return true; }();" + nl_f7
               + "        (void)cheshireQrAnnounced;" + nl_f7
               + "        if (cheshireQrNullspace)" + nl_f7
               + "            Nullspace2RankDeficient(A, f1, f2);" + nl_f7
               + "        else" + nl_f7
               + "            Nullspace2(A, f1, f2);" + nl_f7)
        t = t[:head_end] + rest.replace(old, new, 1)

        inc = "#include <aliceVision/numeric/polynomial.hpp>" + nl_f7
        if t.count(inc) != 1:
            sys.exit("polynomial.hpp include not found once in Fundamental7PSolver.cpp")
        t = t.replace(inc, inc + "#include <cstdlib>  // cheshire: CHESHIRE_QR_NULLSPACE" + nl_f7 + "#include <cstdio>   // cheshire: the announce line" + nl_f7, 1)
        f7.write_text(t, encoding="utf-8", newline="")

    # 5c. A stable order for GPU SIFT keypoints. PopSIFT returns the same keypoint set run to run
    #     (sorted diff 0 lines on 24,500 keypoints) in a different order each time, because octaves
    #     and orientations finish on the GPU in whatever order they finish. Downstream, order is
    #     not neutral: matching and the RANSAC seeds in SfM see it, so two runs on the same
    #     photographs pick different initial pairs and no pipeline is reproducible (docs/04, the
    #     skull turntable). Sorting by (x, y, scale, orientation), descriptor bytes as the tie-break,
    #     makes the .feat files a function of the image alone. CHESHIRE_SIFT_SORT=0 keeps the
    #     arrival order.
    ps = AV / "src/aliceVision/feature/sift/ImageDescriber_SIFT_popSIFT.cpp"
    t = ps.read_text(encoding="utf-8")
    if "cheshireSiftSort" not in t:
        inc = "#include <atomic>" + NL
        if t.count(inc) != 1:
            sys.exit("<atomic> include not found once in ImageDescriber_SIFT_popSIFT.cpp")
        t = t.replace(inc, inc + "#include <algorithm>  // cheshire: stable keypoint order" + NL
                      + "#include <cstdlib>" + NL + "#include <cstring>" + NL + "#include <numeric>" + NL, 1)
        tail = '    ALICEVISION_LOG_TRACE("aliceVision PopSIFT feature count : " << regionsCasted->RegionCount() << std::endl);' + NL
        if t.count(tail) != 1:
            sys.exit("PopSIFT feature-count trace not found once")
        sort_block = """    // cheshire: the keypoints arrive in GPU completion order, which differs run to run; put them
    // in an order that depends only on the image (see scripts/apply_hip_patch.py, step 5c)
    {
        static const bool cheshireSiftSort = []() {
            const char* e = std::getenv("CHESHIRE_SIFT_SORT");
            const bool on = !(e != nullptr && e[0] == '0');
            // if/else, not a ternary: ALICEVISION_LOG_INFO(a) expands to `stream << a` without
            // parentheses, so `stream << on ? A : B` logs the bool and discards both strings.
            if (on)
                ALICEVISION_LOG_INFO("cheshire: GPU SIFT keypoints in stable order (CHESHIRE_SIFT_SORT=0 for arrival order)");
            else
                ALICEVISION_LOG_INFO("cheshire: GPU SIFT keypoints in arrival order (CHESHIRE_SIFT_SORT=0)");
            return on;
        }();
        if (cheshireSiftSort)
        {
            auto& feats = regionsCasted->Features();
            auto& descs = regionsCasted->Descriptors();
            std::vector<std::size_t> order(feats.size());
            std::iota(order.begin(), order.end(), std::size_t(0));
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                const auto& fa = feats[a];
                const auto& fb = feats[b];
                if (fa.x() != fb.x()) return fa.x() < fb.x();
                if (fa.y() != fb.y()) return fa.y() < fb.y();
                if (fa.scale() != fb.scale()) return fa.scale() < fb.scale();
                if (fa.orientation() != fb.orientation()) return fa.orientation() < fb.orientation();
                return std::memcmp(descs[a].getData(), descs[b].getData(), 128) < 0;
            });
            std::vector<PointFeature> sortedFeats;
            std::vector<Descriptor<unsigned char, 128>> sortedDescs;
            sortedFeats.reserve(feats.size());
            sortedDescs.reserve(descs.size());
            for (std::size_t i : order)
            {
                sortedFeats.push_back(feats[i]);
                sortedDescs.push_back(descs[i]);
            }
            feats.swap(sortedFeats);
            descs.swap(sortedDescs);
        }
    }

"""
        t = t.replace(tail, sort_block.replace("\n", NL) + tail, 1)
        ps.write_text(t, encoding="utf-8", newline="")

    # 5d. Edge padding on the GPU. writeTexture's "dilate gutter" is two sequential sweeps over the
    #     whole atlas (67 M texels at 8192^2), run on the host after the port had already handed the
    #     atlas back; texturingGPU.cu now runs the same sweeps as wavefronts on the device before the
    #     download (the block above passes the padding to finish()), so writeTexture skips its own.
    #     The CPU fallback loop never sets the flag and pads as before.
    tx = AV / "src/aliceVision/mesh/Texturing.cpp"
    t = tx.read_text(encoding="utf-8")
    old_guard = "    if (!texParams.fillHoles && texParams.padding > 0 && level < 0)" + NL + "    {" + NL + "        const unsigned int padding = texParams.padding * 3;" + NL
    if "cheshireAtlasPadded)" not in t:
        if t.count(old_guard) != 1:
            sys.exit("writeTexture padding guard not found once")
        t = t.replace(old_guard,
                      "    if (cheshireAtlasPadded && !texParams.fillHoles && texParams.padding > 0 && level < 0)" + NL
                      + "        ALICEVISION_LOG_INFO(\"  - Edge padding (\" << texParams.padding * 3 << \" pixels) done on the GPU.\");" + NL
                      + "    if (!cheshireAtlasPadded && !texParams.fillHoles && texParams.padding > 0 && level < 0)" + NL
                      + "    {" + NL + "        const unsigned int padding = texParams.padding * 3;" + NL, 1)
        tx.write_text(t, encoding="utf-8", newline="")

    # 5e. The textured mesh straight to OBJ + MTL. saveAs builds an Assimp scene - every (vertex, uv)
    #     pair deduplicated through a std::map, positions and uvs copied into aiMesh arrays - and
    #     exports through Assimp's OBJ writer, 14 s for the engine bay's 1.2 M triangles. The same
    #     content written directly: every mesh vertex as `v` (y and z negated as upstream does), every
    #     uv as `vt u v 0`, then per atlas `usemtl material_<udim>` and `f v/vt v/vt v/vt`, 1-based,
    #     9 significant digits of the FLOAT value (the mesh holds doubles; Assimp's aiVector3D is float,
    #     so upstream's files carry float precision and the direct writer prints the same numbers); the
    #     MTL carries the same Kd/Ka/Ks/illum/map_Kd lines. OBJ only, and
    #     only when the material has no normal, bump or displacement maps (those keep Assimp).
    #     CHESHIRE_OBJ_ASSIMP=1 keeps upstream; CHESHIRE_OBJ_CHECK=1 also writes Assimp's file next to
    #     it as <basename>.assimp.obj for scripts/check_textured_obj.py.
    t = tx.read_text(encoding="utf-8")
    if "cheshireAssimpPath" not in t:
        head = '    ALICEVISION_LOG_INFO("Saving " << meshFileTypeStr << " mesh file using Assimp.");' + NL
        if t.count(head) != 1:
            sys.exit("saveAs Assimp log line not found once")
        block = r"""    // cheshire: OBJ and MTL straight to the files (scripts/apply_hip_patch.py, step 5e)
    std::string cheshireAssimpPath = filepath;
    if (meshFileType == EFileType::OBJ && std::getenv("CHESHIRE_OBJ_ASSIMP") == nullptr && !_atlases.empty()
        && material.getTextures(Material::TextureType::NORMAL).empty()
        && material.getTextures(Material::TextureType::BUMP).empty()
        && material.getTextures(Material::TextureType::DISPLACEMENT).empty())
    {
        const std::string mtlName = basename + ".mtl";
        const StaticVector<std::string>& diffuse = material.getTextures(Material::TextureType::DIFFUSE);
        ALICEVISION_LOG_INFO("Saving obj mesh file (cheshire direct writer): " << mesh->pts.size() << " vertices, " << mesh->uvCoords.size()
                                                                              << " uvs, " << mesh->tris.size() << " faces, " << _atlases.size() << " materials.");
        FILE* f = std::fopen(filepath.c_str(), "wb");
        if (f == nullptr)
            throw std::runtime_error("Cannot open mesh file for writing: " + filepath);
        std::vector<char> buf(1 << 22);
        std::size_t used = 0;
        auto flush = [&]() { if (used) { std::fwrite(buf.data(), 1, used, f); used = 0; } };
        auto put = [&](const char* fmt, auto... args) {
            if (used + 256 > buf.size()) flush();
            used += std::snprintf(buf.data() + used, buf.size() - used, fmt, args...);
        };
        put("# textured mesh, written by Cheshire (AliceVision)\n\nmtllib %s\n\n", mtlName.c_str());
        for (int i = 0; i < mesh->pts.size(); ++i)
            put("v %.9g %.9g %.9g\n", double(float(mesh->pts[i].x)), double(float(-mesh->pts[i].y)), double(float(-mesh->pts[i].z)));
        put("\n");
        for (int i = 0; i < mesh->uvCoords.size(); ++i)
            put("vt %.9g %.9g 0\n", double(float(mesh->uvCoords[i].x)), double(float(mesh->uvCoords[i].y)));
        for (int atlasId = 0; atlasId < _atlases.size(); ++atlasId)
        {
            put("\nusemtl material_%s\n", Material::textureId(atlasId).c_str());
            for (const int triangleId : _atlases[atlasId])
            {
                const auto& tri = mesh->tris[triangleId];
                const auto& uv = mesh->trisUvIds[triangleId];
                put("f %d/%d %d/%d %d/%d\n", tri.v[0] + 1, uv.m[0] + 1, tri.v[1] + 1, uv.m[1] + 1, tri.v[2] + 1, uv.m[2] + 1);
            }
        }
        flush();
        std::fclose(f);
        const std::string mtlPath = (dir / mtlName).string();
        FILE* m = std::fopen(mtlPath.c_str(), "wb");
        if (m == nullptr)
            throw std::runtime_error("Cannot open material file for writing: " + mtlPath);
        std::fprintf(m, "# textured mesh materials, written by Cheshire (AliceVision)\n\n");
        for (int atlasId = 0; atlasId < _atlases.size(); ++atlasId)
        {
            std::fprintf(m, "newmtl material_%s\n", Material::textureId(atlasId).c_str());
            std::fprintf(m, "Kd %.9g %.9g %.9g\n", double(material.diffuse.r), double(material.diffuse.g), double(material.diffuse.b));
            std::fprintf(m, "Ka %.9g %.9g %.9g\n", double(material.ambient.r), double(material.ambient.g), double(material.ambient.b));
            std::fprintf(m, "Ks %.9g %.9g %.9g\n", double(material.specular.r), double(material.specular.g), double(material.specular.b));
            std::fprintf(m, "illum 1\n");
            if (diffuse.size() == _atlases.size())
                std::fprintf(m, "map_Kd %s\n", diffuse[atlasId].c_str());
            std::fprintf(m, "\n");
        }
        std::fclose(m);
        if (std::getenv("CHESHIRE_OBJ_CHECK") == nullptr)
        {
            ALICEVISION_LOG_INFO("Mesh saved.");
            return;
        }
        cheshireAssimpPath = (dir / (basename + ".assimp." + meshFileTypeStr)).string();
        ALICEVISION_LOG_INFO("CHESHIRE_OBJ_CHECK: also writing Assimp's file to " << cheshireAssimpPath);
    }
"""
        t = t.replace(head, block.replace("\n", NL) + head, 1)
        old_export = "exporter.Export(&scene, pFormatId, filepath, pPreprocessing)"
        if t.count(old_export) != 1:
            sys.exit("Assimp Export call not found once")
        t = t.replace(old_export, "exporter.Export(&scene, pFormatId, cheshireAssimpPath, pPreprocessing)", 1)
        inc = "#include <cstdlib>  // cheshire" + NL
        if t.count(inc) != 1:
            sys.exit("cheshire cstdlib include not found once in Texturing.cpp")
        t = t.replace(inc, inc + "#include <cstdio>   // cheshire: direct OBJ writer" + NL + "#include <stdexcept>" + NL, 1)
        tx.write_text(t, encoding="utf-8", newline="")

    # 5f. CUDA camera mipmaps counted by the bridge. cudaMallocMipmappedArray is driver memory that
    #     never passes through cudaMalloc, so on CUDA the bridge saw no "image" bytes at all: no image
    #     line in the summary, and on a small card a budget the planner could commit but the card
    #     could not hold (bridge.h, the clamp comment). The array is noted with its estimated bytes
    #     (every level, the texel type the build uses) right after creation and forgotten before it
    #     is freed. Same on HIP when the platform's native mipmapped arrays are used
    #     (CHESHIRE_NATIVE_MIPMAP); the emulation counts its own levels in mipmap_emu.h, and forget
    #     of an unknown key is a no-op.
    dm = AV / "src/aliceVision/depthMap/cuda/imageProcessing/deviceMipmappedArray.cu"
    t = dm.read_text(encoding="utf-8")
    if "noteExternal" not in t:
        old = "    CHECK_CUDA_RETURN_ERROR(cudaMallocMipmappedArray(out_mipmappedArrayPtr, &desc, imgSize, levels));" + NL
        if t.count(old) != 1:
            sys.exit("cudaMallocMipmappedArray call not found once in deviceMipmappedArray.cu")
        t = t.replace(old, old + """#if !defined(CHESHIRE_EMULATE_MIPMAP)
    {
        // cheshire: count the array's VRAM in the bridge (scripts/apply_hip_patch.py, step 5f).
        // CUDA, and HIP with the platform's own mipmapped arrays: driver memory the bridge cannot
        // own. The HIP emulation (CHESHIRE_EMULATE_MIPMAP, cuda_to_hip.h) counts its own levels.
        size_t bytes = 0, w = in_imgSize.x(), h = in_imgSize.y();
        for (unsigned l = 0; l < levels; ++l)
        {
            bytes += w * h * sizeof(CudaRGBA)
#ifdef ALICEVISION_DEPTHMAP_TEXTURE_USE_HALF
                     / 2
#endif
                ;
            w = w > 1 ? w / 2 : 1;
            h = h > 1 ? h / 2 : 1;
        }
        cheshire::bridge::noteExternal(*out_mipmappedArrayPtr, bytes, cheshire::bridge::Class::Image);
    }
#endif
""".replace("\n", NL), 1)
        dm.write_text(t, encoding="utf-8", newline="")
    di = AV / "src/aliceVision/depthMap/cuda/host/DeviceMipmapImage.cpp"
    t = di.read_text(encoding="utf-8")
    if "forgetExternal" not in t:
        for old in ("        CHECK_CUDA_RETURN_ERROR_NOEXCEPT(cudaFreeMipmappedArray(_mipmappedArray));" + NL,
                    "        CHECK_CUDA_RETURN_ERROR(cudaFreeMipmappedArray(_mipmappedArray));" + NL):
            if t.count(old) != 1:
                sys.exit("cudaFreeMipmappedArray call not found once in DeviceMipmapImage.cpp")
            # braced: upstream's `if (_mipmappedArray != nullptr)` has no braces, so an inserted
            # statement would become the if-body and the free would run unconditionally (it did:
            # "invalid argument" from freeing a null handle, first build of this step).
            t = t.replace(old, "        {" + NL + "            cheshire::bridge::forgetExternal(_mipmappedArray);  // cheshire, step 5f" + NL
                          + "    " + old + "        }" + NL, 1)
        inc = "#include <aliceVision/depthMap/cuda/imageProcessing/deviceMipmappedArray.hpp>" + NL
        if t.count(inc) != 1:
            sys.exit("deviceMipmappedArray.hpp include not found once in DeviceMipmapImage.cpp")
        t = t.replace(inc, inc + "#include <aliceVision/depthMap/cuda/host/memory.hpp>  // cheshire: the bridge" + NL, 1)
        di.write_text(t, encoding="utf-8", newline="")

    # 5g. PrepareDenseScene phase profile. The node is read (JPEG decode), exposure + mask,
    #     undistort (per-pixel bilinear through the camera model) and an EXR write per view; which of
    #     the four costs what is not visible in any log. CHESHIRE_PDS_PROFILE=1 sums each phase over
    #     the threads and prints the four totals when the node finishes. Same bytes out.
    pds = AV / "src/software/pipeline/main_prepareDenseScene.cpp"
    t = pds.read_text(encoding="utf-8")
    if "cheshirePdsProfile" not in t:
        inc = "#include <aliceVision/camera/cameraUndistortImage.hpp>" + NL
        if t.count(inc) != 1:
            sys.exit("cameraUndistortImage include not found once in main_prepareDenseScene.cpp")
        t = t.replace(inc, inc + r"""#include <atomic>   // cheshire: CHESHIRE_PDS_PROFILE
#include <chrono>
#include <cstdio>
#include <cstdlib>
namespace {
// cheshire: per-phase seconds summed over threads (scripts/apply_hip_patch.py, step 5g)
struct CheshirePdsProfile
{
    bool on = std::getenv("CHESHIRE_PDS_PROFILE") != nullptr;
    std::atomic<long long> readUs{0}, prepUs{0}, undistortUs{0}, writeUs{0}, views{0};
    ~CheshirePdsProfile()
    {
        if (on)
            std::fprintf(stderr, "[cheshire] prepareDenseScene profile: %lld views; thread-seconds: read %.1f, exposure+mask %.1f, undistort %.1f, write %.1f\n",
                         views.load(), readUs.load() / 1e6, prepUs.load() / 1e6, undistortUs.load() / 1e6, writeUs.load() / 1e6);
    }
} cheshirePdsProfile;
inline long long cheshireUsSince(std::chrono::steady_clock::time_point t)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();
}
}  // namespace
""".replace("\n", NL), 1)
        old_read = "    ImageT image, image_ud;" + NL + "    readImage(srcImage, image, image::EImageColorSpace::LINEAR);" + NL
        if t.count(old_read) != 1:
            sys.exit("process() readImage not found once")
        t = t.replace(old_read,
                      "    ImageT image, image_ud;" + NL
                      + "    auto cheshireT = std::chrono::steady_clock::now();  // cheshire, step 5g" + NL
                      + "    readImage(srcImage, image, image::EImageColorSpace::LINEAR);" + NL
                      + "    cheshirePdsProfile.readUs += cheshireUsSince(cheshireT); cheshireT = std::chrono::steady_clock::now();" + NL, 1)
        old_mask = "    // mask" + NL + "    maskFunc(image);" + NL
        if t.count(old_mask) != 1:
            sys.exit("process() maskFunc not found once")
        t = t.replace(old_mask, old_mask + "    cheshirePdsProfile.prepUs += cheshireUsSince(cheshireT); cheshireT = std::chrono::steady_clock::now();" + NL, 1)
        old_ud = "        UndistortImage(image, cam, image_ud, pixZero);" + NL + "        writeImage(dstColorImage, image_ud, image::ImageWriteOptions(), metadata);" + NL
        if t.count(old_ud) != 1:
            sys.exit("process() undistort+write not found once")
        t = t.replace(old_ud,
                      "        UndistortImage(image, cam, image_ud, pixZero);" + NL
                      + "        cheshirePdsProfile.undistortUs += cheshireUsSince(cheshireT); cheshireT = std::chrono::steady_clock::now();" + NL
                      + "        writeImage(dstColorImage, image_ud, image::ImageWriteOptions(), metadata);" + NL
                      + "        cheshirePdsProfile.writeUs += cheshireUsSince(cheshireT); ++cheshirePdsProfile.views;" + NL, 1)
        old_w2 = "    else" + NL + "    {" + NL + "        writeImage(dstColorImage, image, image::ImageWriteOptions(), metadata);" + NL + "    }" + NL
        if t.count(old_w2) != 1:
            sys.exit("process() plain write not found once")
        t = t.replace(old_w2,
                      "    else" + NL + "    {" + NL
                      + "        writeImage(dstColorImage, image, image::ImageWriteOptions(), metadata);" + NL
                      + "        cheshirePdsProfile.writeUs += cheshireUsSince(cheshireT); ++cheshirePdsProfile.views;" + NL + "    }" + NL, 1)
        pds.write_text(t, encoding="utf-8", newline="")

    # 5h. Undistortion coordinate map, once per intrinsic. UndistortImage evaluates the camera model
    #     (ima2cam, the distortion polynomial, cam2ima) for every output pixel of every view; on the
    #     engine bay that is 153 of PrepareDenseScene's 367 thread-seconds, for 107 views that share
    #     ONE intrinsic. The distorted coordinate depends only on the intrinsic, the output size and
    #     the principal-point correction, so it is computed once per such key and kept; the per-view
    #     work is the bilinear sample alone. Same doubles, same contains() test, same sampler call:
    #     the same bytes out. CHESHIRE_UNDISTORT_MAP=0 disables; CHESHIRE_UNDISTORT_MAP_MB caps the
    #     cache (default 2048 MB; a 12 MP map is 195 MB); beyond the cap the original loop runs.
    cu = AV / "src/aliceVision/camera/cameraUndistortImage.hpp"
    t = cu.read_text(encoding="utf-8")
    if "cheshire_undistort" not in t:   # the namespace the helper defines; the guard must name something the patch emits
        inc = "#include <aliceVision/image/io.hpp>" + NL
        if t.count(inc) != 1:
            sys.exit("io.hpp include not found once in cameraUndistortImage.hpp")
        helper = r"""#include <cstdlib>   // cheshire: undistortion map cache (scripts/apply_hip_patch.py, step 5h)
#include <memory>
#include <mutex>
#include <vector>

namespace aliceVision {
namespace camera {
namespace cheshire_undistort {

struct MapKey
{
    std::vector<double> params;
    int type = 0, w = 0, h = 0, roiW = 0, roiH = 0, xOff = 0, yOff = 0;
    double ppx = 0, ppy = 0;
    bool operator==(const MapKey& o) const
    {
        return params == o.params && type == o.type && w == o.w && h == o.h && roiW == o.roiW && roiH == o.roiH
            && xOff == o.xOff && yOff == o.yOff && ppx == o.ppx && ppy == o.ppy;
    }
};
struct MapEntry
{
    MapKey key;
    std::shared_ptr<const std::vector<Vec2>> map;   // roiW x roiH distorted source coordinates
};
struct Cache
{
    std::mutex m;
    std::vector<MapEntry> entries;
    std::size_t bytes = 0;
    std::size_t capBytes = std::size_t(2048) << 20;
    bool enabled = true;
    bool announced = false;
    Cache()
    {
        if (const char* e = std::getenv("CHESHIRE_UNDISTORT_MAP")) enabled = !(e[0] == '0');
        if (const char* e = std::getenv("CHESHIRE_UNDISTORT_MAP_MB")) capBytes = std::size_t(std::atoll(e)) << 20;
    }
};
inline Cache& cache() { static Cache c; return c; }

// The map for this key, computed on first use with exactly the expression UndistortImage used
// per pixel; nullptr when the cache is off or full (the caller then runs the original loop).
inline std::shared_ptr<const std::vector<Vec2>> mapFor(const IntrinsicBase* intrinsicPtr, int roiW, int roiH,
                                                       int xOff, int yOff, const Vec2& ppCorrection)
{
    Cache& c = cache();
    if (!c.enabled) return nullptr;
    MapKey key;
    key.params = intrinsicPtr->getParameters();
    key.type = int(intrinsicPtr->getType());
    key.w = int(intrinsicPtr->w()); key.h = int(intrinsicPtr->h());
    key.roiW = roiW; key.roiH = roiH; key.xOff = xOff; key.yOff = yOff;
    key.ppx = ppCorrection(0); key.ppy = ppCorrection(1);
    {
        std::lock_guard<std::mutex> g(c.m);
        for (const MapEntry& e : c.entries)
            if (e.key == key) return e.map;
    }
    const std::size_t need = std::size_t(roiW) * roiH * sizeof(Vec2);
    {
        std::lock_guard<std::mutex> g(c.m);
        if (c.bytes + need > c.capBytes)
        {
            if (!c.announced)
            {
                c.announced = true;
                ALICEVISION_LOG_INFO("cheshire: undistort map: " << roiW << "x" << roiH << " would exceed CHESHIRE_UNDISTORT_MAP_MB, computing per view");
            }
            return nullptr;
        }
    }
    auto m = std::make_shared<std::vector<Vec2>>(std::size_t(roiW) * roiH);
    Vec2* out = m->data();
#pragma omp parallel for
    for (int y = 0; y < roiH; ++y)
    {
        for (int x = 0; x < roiW; ++x)
        {
            const Vec2 undisto_pix(x + xOff, y + yOff);
            out[std::size_t(y) * roiW + x] = intrinsicPtr->getDistortedPixel(undisto_pix + ppCorrection);
        }
    }
    std::lock_guard<std::mutex> g(c.m);
    for (const MapEntry& e : c.entries)   // another thread may have filled the same key meanwhile
        if (e.key == key) return e.map;
    c.entries.push_back(MapEntry{key, m});
    c.bytes += need;
    ALICEVISION_LOG_INFO("cheshire: undistort map: " << roiW << "x" << roiH << " for one intrinsic, computed once ("
                         << (need >> 20) << " MB; " << c.entries.size() << " cached, CHESHIRE_UNDISTORT_MAP=0 to disable)");
    return m;
}

}  // namespace cheshire_undistort
}  // namespace camera
}  // namespace aliceVision
"""
        t = t.replace(inc, inc + helper.replace("\n", NL), 1)
        old_loop = ("#pragma omp parallel for" + NL
                    + "    for (int y = 0; y < heightRoi; ++y)" + NL
                    + "    {" + NL
                    + "        for (int x = 0; x < widthRoi; ++x)" + NL
                    + "        {" + NL
                    + "            const Vec2 undisto_pix(x + xOffset, y + yOffset);" + NL
                    + "            // compute coordinates with distortion" + NL
                    + "            const Vec2 disto_pix = intrinsicPtr->getDistortedPixel(undisto_pix + ppCorrection);" + NL)
        if t.count(old_loop) != 1:
            sys.exit("UndistortImage (intrinsicPtr overload) pixel loop not found once")
        new_loop = ("    // cheshire: the distorted coordinates come from the per-intrinsic map when there is one" + NL
                    + "    const auto cheshireMap = cheshire_undistort::mapFor(intrinsicPtr, widthRoi, heightRoi, xOffset, yOffset, ppCorrection);" + NL
                    + "    const Vec2* cheshireLut = cheshireMap ? cheshireMap->data() : nullptr;" + NL
                    + "#pragma omp parallel for" + NL
                    + "    for (int y = 0; y < heightRoi; ++y)" + NL
                    + "    {" + NL
                    + "        for (int x = 0; x < widthRoi; ++x)" + NL
                    + "        {" + NL
                    + "            const Vec2 undisto_pix(x + xOffset, y + yOffset);" + NL
                    + "            // compute coordinates with distortion" + NL
                    + "            const Vec2 disto_pix = cheshireLut ? cheshireLut[std::size_t(y) * widthRoi + x]" + NL
                    + "                                              : intrinsicPtr->getDistortedPixel(undisto_pix + ppCorrection);" + NL)
        t = t.replace(old_loop, new_loop, 1)
        cu.write_text(t, encoding="utf-8", newline="")

    # 5i. Bundle adjustment profile. SfM on the engine bay is 63 % bundle adjustment (docs/04), and
    #     nothing says whether a run is Jacobian evaluation (autodiff over every observation) or the
    #     Schur solve. Ceres measures both; CHESHIRE_BA_PROFILE=1 prints its numbers after each solve.
    bac = AV / "src/aliceVision/sfm/bundle/BundleAdjustmentCeres.cpp"
    t = bac.read_text(encoding="utf-8")
    if "CHESHIRE_BA_PROFILE" not in t:
        old = "    ceres::Solve(options, &problem, &summary);" + NL
        if t.count(old) != 1:
            sys.exit("ceres::Solve call not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + r"""    // cheshire: Ceres' own time split, one line per solve (scripts/apply_hip_patch.py, step 5i)
    {
        static const bool cheshireBaProfile = std::getenv("CHESHIRE_BA_PROFILE") != nullptr;
        if (cheshireBaProfile)
            ALICEVISION_LOG_INFO("cheshire: BA profile: total " << summary.total_time_in_seconds << " s = residuals "
                                 << summary.residual_evaluation_time_in_seconds << " + jacobians " << summary.jacobian_evaluation_time_in_seconds
                                 << " + linear solver " << summary.linear_solver_time_in_seconds << " + other; "
                                 << summary.iterations.size() << " iterations, " << summary.num_threads_used << " threads, "
                                 << ceres::LinearSolverTypeToString(summary.linear_solver_type_used) << ", "
                                 << summary.num_residual_blocks << " residual blocks, " << summary.num_parameter_blocks << " parameter blocks");
    }
""".replace("\n", NL), 1)
        if "#include <cstdlib>" not in t:
            inc0 = t.index("#include")
            t = t[:inc0] + "#include <cstdlib>  // cheshire: CHESHIRE_BA_PROFILE" + NL + t[inc0:]
        bac.write_text(t, encoding="utf-8", newline="")

    # 5j. The texture downscale on the device. After padding, writeTexture ran imageAlgo::resizeImage
    #     (OIIO resize, lanczos3 for downsizing) on the host: 4.1 s per 8192^2 atlas on the RX 9070 box,
    #     the largest piece of the node once padding moved. finish() now produces the downscaled atlas
    #     (texturingGPU.cu: OIIO's tap weights computed on the host exactly as OIIO computes them, the
    #     accumulation on the device in OIIO's order) and writeTexture uses it; CHESHIRE_GPU_RESIZE=0
    #     keeps the host path, CHESHIRE_GPU_RESIZE_CHECK=1 runs both and counts differing texels.
    tx = AV / "src/aliceVision/mesh/Texturing.cpp"
    t = tx.read_text(encoding="utf-8")
    if "GPU resize check" not in t:
        old = ('        ALICEVISION_LOG_INFO("  - Downscaling texture (" << texParams.downscale << "x).");' + NL
               + "        imageAlgo::resizeImage(texParams.downscale, atlasTexture.img, resizedColorBuffer);" + NL)
        if t.count(old) != 1:
            sys.exit("writeTexture downscale call not found once")
        new = r"""        if (cheshireResizedAtlas != nullptr)
        {
            // cheshire: the device produced it (scripts/apply_hip_patch.py, step 5j)
            ALICEVISION_LOG_INFO("  - Downscaling texture (" << texParams.downscale << "x) done on the GPU.");
            if (std::getenv("CHESHIRE_GPU_RESIZE_CHECK") != nullptr)
            {
                image::Image<image::RGBfColor> hostResized;
                imageAlgo::resizeImage(texParams.downscale, atlasTexture.img, hostResized);
                std::size_t bad = 0;
                const std::size_t n = std::size_t(hostResized.width()) * hostResized.height();
                const float* a = reinterpret_cast<const float*>(hostResized.data());
                const float* b = reinterpret_cast<const float*>(cheshireResizedAtlas->data());
                if (hostResized.width() != cheshireResizedAtlas->width() || hostResized.height() != cheshireResizedAtlas->height())
                    bad = n;
                else
                    for (std::size_t i = 0; i < n * 3; ++i)
                        if (a[i] != b[i]) { ++bad; }
                ALICEVISION_LOG_INFO("cheshire: GPU resize check: texel channels differing from OpenImageIO: " << bad << " of " << n * 3);
            }
            resizedColorBuffer = *cheshireResizedAtlas;
        }
        else
        {
            ALICEVISION_LOG_INFO("  - Downscaling texture (" << texParams.downscale << "x).");
            imageAlgo::resizeImage(texParams.downscale, atlasTexture.img, resizedColorBuffer);
        }
"""
        t = t.replace(old, new.replace("\n", NL), 1)
        tx.write_text(t, encoding="utf-8", newline="")

    # 5k. Incremental SfM: the resection pass that ends without its bundle adjustment. The main
    #     loop runs a BA every N resected views; when findNextBestViews finds no candidate above its
    #     score threshold the loop exits with the views resected since the last BA still pending -
    #     never bundle-adjusted, never handed to the local-BA graph - and the next pass records them
    #     as previously reconstructed. They keep their poses and observe landmarks, so a later new
    #     view's edge to one of them is _nodePerViewId.at() on a view the graph never saw: the
    #     "[fatal] invalid map<K, T> key" of Meshroom #2344, reproduced three times on the 884-view
    #     False Door at views 830-834 (docs/04). Two changes: the pass finishes with the BA it owed
    #     (CHESHIRE_SFM_PENDING_BA=0 restores upstream), and the graph skips an edge whose endpoint
    #     it does not hold instead of throwing. Both announce.
    eng = AV / "src/aliceVision/sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.cpp"
    t = eng.read_text(encoding="utf-8")
    if "CHESHIRE_SFM_PENDING_BA" not in t:
        old = "    std::size_t globalIteration = 0;" + NL
        if t.count(old) != 1:
            sys.exit("globalIteration declaration not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, old + r"""    // cheshire: see the block after the resection loop (scripts/apply_hip_patch.py, step 5k)
    static const bool cheshirePendingBA = [] {
        const char* v = std::getenv("CHESHIRE_SFM_PENDING_BA");
        return v == nullptr || std::string(v) != "0";
    }();
    if (cheshirePendingBA)
        ALICEVISION_LOG_INFO("cheshire: incremental SfM: a resection pass that ends without a bundle adjustment gets one before the next pass (CHESHIRE_SFM_PENDING_BA=0 restores upstream)");
    else
        ALICEVISION_LOG_INFO("cheshire: incremental SfM: pending bundle adjustment disabled by CHESHIRE_SFM_PENDING_BA=0 (upstream behaviour)");
""".replace("\n", NL), 1)
        old = ('        if (_params.rig.useRigConstraint && !_sfmData.getRigs().empty())' + NL + '        {' + NL
               + '            ALICEVISION_LOG_INFO("Rig(s) calibration start");')
        if t.count(old) != 1:
            sys.exit("rig calibration block not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, r"""        // cheshire: the views resected since the last bundle adjustment, when the loop above ended
        // without one (no candidate reached the score threshold). Upstream leaves them with their
        // resection pose, no refinement and no node in the local-BA graph, and the next pass counts
        // them as old; a later edge to one of them throws. Finish the pass as a full group would.
        // (scripts/apply_hip_patch.py, step 5k)
        if (cheshirePendingBA)
        {
            std::set<IndexT> pendingViews;
            const std::set<IndexT> reconstructedNow = _sfmData.getValidViews();
            std::set_difference(reconstructedNow.begin(), reconstructedNow.end(),
                                prevReconstructedViews.begin(), prevReconstructedViews.end(),
                                std::inserter(pendingViews, pendingViews.end()));
            if (!pendingViews.empty())
            {
                ALICEVISION_LOG_INFO("cheshire: " << pendingViews.size() << " views resected since the last bundle adjustment were still pending when the resection loop ended; triangulating and bundle-adjusting them");
                triangulate(prevReconstructedViews, pendingViews);
                bundleAdjustment(pendingViews);
                prevReconstructedViews = _sfmData.getValidViews();
                registerChanges(linkedViewIds, pendingViews);
                std::set_union(potentials.begin(), potentials.end(), linkedViewIds.begin(), linkedViewIds.end(),
                               std::inserter(potentials, potentials.end()));
                ++_resectionId;
            }
        }

""".replace("\n", NL) + old, 1)
        for h in ("<cstdlib>", "<iterator>", "<string>"):
            if ("#include " + h) not in t:
                inc0 = t.index("#include")
                t = t[:inc0] + "#include " + h + "  // cheshire: step 5k" + NL + t[inc0:]
        eng.write_text(t, encoding="utf-8", newline="")

    lbg = AV / "src/aliceVision/sfm/LocalBundleAdjustmentGraph.cpp"
    t = lbg.read_text(encoding="utf-8")
    if "cheshireSkippedEdges" not in t:
        old = ("        for (const Pair& edge : newEdges)" + NL
               + "            _graph.addEdge(_nodePerViewId.at(edge.first), _nodePerViewId.at(edge.second));" + NL)
        if t.count(old) != 1:
            sys.exit("edge loop not found once in LocalBundleAdjustmentGraph.cpp")
        t = t.replace(old, r"""        // cheshire: an edge to a posed view the graph was never handed is skipped, not thrown on
        // (scripts/apply_hip_patch.py, step 5k; the engine now hands every view over, this is the guard)
        std::size_t cheshireSkippedEdges = 0;
        for (const Pair& edge : newEdges)
        {
            const auto a = _nodePerViewId.find(edge.first);
            const auto b = _nodePerViewId.find(edge.second);
            if (a == _nodePerViewId.end() || b == _nodePerViewId.end())
            {
                ++cheshireSkippedEdges;
                continue;
            }
            _graph.addEdge(a->second, b->second);
        }
        if (cheshireSkippedEdges != 0)
            ALICEVISION_LOG_WARNING("cheshire: local BA graph: " << cheshireSkippedEdges << " edges to posed views the graph was never handed were skipped (upstream throws here)");
""".replace("\n", NL), 1)
        lbg.write_text(t, encoding="utf-8", newline="")

    # the switch in the --help text, which is what meshroom-pair gates a package on
    msf = AV / "src/software/pipeline/main_incrementalSfM.cpp"
    t = msf.read_text(encoding="utf-8")
    if "CHESHIRE_SFM_PENDING_BA" not in t:
        old = 'CmdLine cmdline("Sequential/Incremental reconstruction.' + chr(92) + 'n"'
        if t.count(old) != 1:
            sys.exit("incrementalSfM CmdLine description not found once")
        t = t.replace(old, 'CmdLine cmdline("Sequential/Incremental reconstruction (cheshire: a resection pass that ends without a bundle adjustment gets one; CHESHIRE_SFM_PENDING_BA=0 restores upstream).' + chr(92) + 'n"', 1)
        msf.write_text(t, encoding="utf-8", newline="")

    # 5l. No FMA contraction in the ports on the CUDA backend either. Every port file carries
    #     `#pragma clang fp contract(off)` so a*b+c rounds twice as the CPU reference does; nvcc
    #     ignores that pragma and contracts by default (--fmad=true), so the CUDA build of the same
    #     kernels rounded once. Found by the GTX 1080 Ti gate on the first float-exact self-check that
    #     ran there: GPU resize 7,772,469 of 50,331,648 channels differing from OpenImageIO, against
    #     0 on every HIP card. Upstream's own depth-map kernels are left alone: they were built with
    #     contraction on, and byte-identity with the CUDA reference depends on it.
    fmad_re = re.compile(r"^(\s*)list\(APPEND (\w+)_files_sources ((?:gpu/\S+\.cu\s?)+)\)\s*$", re.M)
    for cml in (AV / "src/aliceVision/matching/CMakeLists.txt",
                AV / "src/aliceVision/fuseCut/CMakeLists.txt",
                AV / "src/aliceVision/mesh/CMakeLists.txt"):
        t = cml.read_text(encoding="utf-8")
        if "fmad=false" in t:
            continue
        def add_fmad(m):
            ind, files = m.group(1), m.group(3).strip()
            return (m.group(0) + NL
                    + ind + "if (ALICEVISION_HAVE_CUDA AND NOT ALICEVISION_HAVE_HIP)" + NL
                    + ind + "    # cheshire: the ports round a*b+c twice, as the CPU reference does (nvcc ignores the clang pragma)" + NL
                    + ind + "    set_source_files_properties(" + files + " PROPERTIES COMPILE_OPTIONS \"--fmad=false\")" + NL
                    + ind + "endif()")
        t2, n = fmad_re.subn(add_fmad, t)
        if n == 0:
            sys.exit(f"no port source list found in {cml}")
        cml.write_text(t2, encoding="utf-8", newline="")

    # 5m. Bundle adjustment: the projection residual's Jacobians without the autodiff passes.
    #     Upstream's projection cost is a DynamicAutoDiffCostFunction (Stride 4) around a functor
    #     that rotates the point with Jets and then calls the already-analytic CostIntrinsicsProject
    #     through DynamicCostFunctionToFunctorTmp; Ceres runs that functor once per 4 derivative
    #     components, so CostIntrinsicsProject::Evaluate computes all its Jacobian blocks 3 to 5
    #     times per residual block per Jacobian. hip/port/sfm_ba/projectionCheshire.hpp has the
    #     reasoning; CHESHIRE_BA_JACOBIANS=autodiff (default) | stride (one pass) | analytic (the
    #     chain rule by hand), CHESHIRE_BA_CHECK=1 evaluates a reference next to it and reports.
    shutil.copy2(ROOT / "hip" / "port" / "sfm_ba" / "projectionCheshire.hpp",
                 AV / "src/aliceVision/sfm/bundle/costfunctions/projectionCheshire.hpp")
    t = bac.read_text(encoding="utf-8")
    if "projectionCheshire.hpp" not in t:
        old = "#include <aliceVision/sfm/bundle/costfunctions/projection.hpp>" + NL
        if t.count(old) != 1:
            sys.exit("projection.hpp include not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + "#include <aliceVision/sfm/bundle/costfunctions/projectionCheshire.hpp>  // cheshire: step 5m" + NL, 1)
        old = "ceres::CostFunction* costFunction = ProjectionErrorFunctor::createCostFunction(intrinsic, observation);"
        if t.count(old) != 1:
            sys.exit("rig projection createCostFunction call not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, "ceres::CostFunction* costFunction = cheshire::createProjectionCost(true, intrinsic, observation);  // cheshire: step 5m", 1)
        old = "ceres::CostFunction* costFunction = ProjectionSimpleErrorFunctor::createCostFunction(intrinsic, observation);"
        if t.count(old) != 1:
            sys.exit("simple projection createCostFunction call not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, "ceres::CostFunction* costFunction = cheshire::createProjectionCost(false, intrinsic, observation);  // cheshire: step 5m", 1)
        old = "    ceres::Solve(options, &problem, &summary);" + NL
        if t.count(old) != 1:
            sys.exit("ceres::Solve call not found once in BundleAdjustmentCeres.cpp (5m)")
        t = t.replace(old, old + r"""    // cheshire: the projection cost check, one line per solve (scripts/apply_hip_patch.py, step 5m)
    if (cheshire::baCheckEnabled())
        ALICEVISION_LOG_INFO("cheshire: BA check (" << cheshire::baJacobiansName(cheshire::baJacobiansMode()) << " against " << cheshire::baCheckReferenceName() << "): " << cheshire::baCheckReport());
""".replace("\n", NL), 1)
        bac.write_text(t, encoding="utf-8", newline="")

    # 5n. Incremental SfM, reproducible. hip/port/sfm_ba/deterministic.hpp has the reasoning: a
    #     generator per task instead of one std::mt19937 shared across OpenMP threads, landmark
    #     blocks in one array and an ordering group per block so Ceres never orders by pointer, a
    #     total order in the next-best-views sort, and CHESHIRE_SFM_DETERMINISTIC=1 for one Ceres
    #     thread (CHESHIRE_BA_THREADS=n explicitly; CHESHIRE_SFM_TASK_SEED=0 restores upstream's
    #     shared generator).
    shutil.copy2(ROOT / "hip" / "port" / "sfm_ba" / "deterministic.hpp", AV / "src/aliceVision/sfm/deterministic.hpp")

    re_hpp = AV / "src/aliceVision/sfm/pipeline/ReconstructionEngine.hpp"
    t = re_hpp.read_text(encoding="utf-8")
    if "_cheshireSeed" not in t:
        old = "#include <random>" + NL
        if t.count(old) != 1:
            sys.exit("<random> include not found once in ReconstructionEngine.hpp")
        t = t.replace(old, old + "#include <aliceVision/sfm/deterministic.hpp>  // cheshire: step 5n" + NL, 1)
        old = "    void initRandomSeed(int seed) { _randomNumberGenerator.seed(seed == -1 ? std::random_device()() : seed); }"
        if t.count(old) != 1:
            sys.exit("initRandomSeed not found once in ReconstructionEngine.hpp")
        t = t.replace(old, r"""    void initRandomSeed(int seed)
    {
        // cheshire (step 5n): the effective seed is kept, every task derives its generator from it
        _cheshireSeed = (seed == -1) ? std::random_device()() : static_cast<unsigned>(seed);
        _randomNumberGenerator.seed(_cheshireSeed);
    }""".replace("\n", NL), 1)
        old = "    std::mt19937 _randomNumberGenerator;" + NL
        if t.count(old) != 1:
            sys.exit("_randomNumberGenerator member not found once in ReconstructionEngine.hpp")
        t = t.replace(old, old + "    // cheshire (step 5n): the seed the tasks derive their generators from" + NL
                      + "    unsigned _cheshireSeed = std::mt19937::default_seed;" + NL, 1)
        re_hpp.write_text(t, encoding="utf-8", newline="")

    t = eng.read_text(encoding="utf-8")
    if "cheshire::taskGenerator" not in t:
        # resection: a generator per (view, pass)
        anchor = "    const bool bResection = sfm::SfMLocalizer::localize("
        if t.count(anchor) != 1:
            sys.exit("localize call not found once in ReconstructionEngine_sequentialSfM.cpp")
        i = t.index(anchor)
        j = t.index("_randomNumberGenerator,", i)
        t = (t[:i]
             + "    // cheshire (step 5n): this view's own generator, the same on any thread" + NL
             + "    std::mt19937 cheshireTaskGenerator = cheshire::taskGenerator(_cheshireSeed, 1, viewId, _resectionId);" + NL
             + "    std::mt19937& cheshireGenerator = cheshire::taskSeedEnabled() ? cheshireTaskGenerator : _randomNumberGenerator;" + NL
             + t[i:j] + "cheshireGenerator," + t[j + len("_randomNumberGenerator,"):])
        # triangulation: a generator per (track, pass)
        old = "            multiview::TriangulateNViewLORANSAC(features, Ps, _randomNumberGenerator, X_homogeneous, &inliersIndex, 8.0);"
        if t.count(old) != 1:
            sys.exit("TriangulateNViewLORANSAC call not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, r"""            // cheshire (step 5n): this track's own generator, the same on any thread
            std::mt19937 cheshireTaskGenerator = cheshire::taskGenerator(_cheshireSeed, 2, trackId, _resectionId);
            std::mt19937& cheshireGenerator = cheshire::taskSeedEnabled() ? cheshireTaskGenerator : _randomNumberGenerator;
            multiview::TriangulateNViewLORANSAC(features, Ps, cheshireGenerator, X_homogeneous, &inliersIndex, 8.0);""".replace("\n", NL), 1)
        # next-best-views: a total order (score, then view id) instead of ties by arrival
        old = "        return std::get<2>(t1) > std::get<2>(t2);"
        if t.count(old) != 1:
            sys.exit("findConnectedViews comparator not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, "        // cheshire (step 5n): ties by view id, not by the order the threads finished in" + NL
                      + "        return std::get<2>(t1) != std::get<2>(t2) ? std::get<2>(t1) > std::get<2>(t2) : std::get<0>(t1) < std::get<0>(t2);", 1)
        eng.write_text(t, encoding="utf-8", newline="")

    bah = AV / "src/aliceVision/sfm/bundle/BundleAdjustmentCeres.hpp"
    t = bah.read_text(encoding="utf-8")
    if "OrderedBlocks" not in t:
        old = "#include <aliceVision/sfm/bundle/BundleAdjustment.hpp>" + NL
        if t.count(old) != 1:
            sys.exit("BundleAdjustment.hpp include not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, old + "#include <aliceVision/sfm/deterministic.hpp>  // cheshire: step 5n" + NL, 1)
        old = "    std::map<IndexT, std::array<double, 3>> _landmarksBlocks;" + NL
        if t.count(old) != 1:
            sys.exit("_landmarksBlocks member not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, "    cheshire::OrderedBlocks<std::array<double, 3>> _landmarksBlocks;  // cheshire (step 5n): one array, key order" + NL, 1)
        old = "    ceres::ParameterBlockOrdering _linearSolverOrdering;" + NL
        if t.count(old) != 1:
            sys.exit("_linearSolverOrdering member not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, old + r"""
    // cheshire (step 5n): one ordering group per parameter block, numbered by key, so the order
    // Ceres eliminates in and lays the reduced camera system out in is the key order and not the
    // address order of std::set<double*>. Poses from 1, rig sub-poses from 1e6, intrinsics from
    // 2e6, distortions from 3e6, the shared fake distortion block at 4e6.
    std::map<IndexT, int> _cheshirePoseGroup;
    std::map<IndexT, int> _cheshireIntrinsicGroup;
    std::map<std::pair<IndexT, IndexT>, int> _cheshireRigGroup;
    int cheshirePoseGroup(IndexT poseId) const { return _cheshirePoseGroup.at(poseId); }
    int cheshireIntrinsicGroup(IndexT intrinsicId) const { return _cheshireIntrinsicGroup.at(intrinsicId); }
    int cheshireDistortionGroup(IndexT intrinsicId, bool fake) const { return fake ? 4000000 : 1000000 + _cheshireIntrinsicGroup.at(intrinsicId); }
    int cheshireRigGroup(IndexT rigId, IndexT subPoseId)
    {
        int& g = _cheshireRigGroup[std::make_pair(rigId, subPoseId)];
        if (g == 0)
            g = 1000000 + static_cast<int>(_cheshireRigGroup.size());
        return g;
    }
""".replace("\n", NL), 1)
        bah.write_text(t, encoding="utf-8", newline="")

    t = bac.read_text(encoding="utf-8")
    if "_cheshirePoseGroup" not in t:
        old = "    // clear previously computed data" + NL + "    resetProblem();" + NL
        if t.count(old) != 1:
            sys.exit("createProblem's resetProblem call not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + r"""
    // cheshire (step 5n): landmark blocks in one array, and an ordering group per block by key
    _landmarksBlocks.reserve(sfmData.getLandmarks().size());
    {
        int g = 1;
        for (const auto& [poseId, pose] : sfmData.getPoses().valueRange())
            _cheshirePoseGroup[poseId] = g++;
        g = 2000000;
        for (const auto& [intrinsicId, intrinsicPtr] : sfmData.getIntrinsics())
            _cheshireIntrinsicGroup[intrinsicId] = g++;
    }
""".replace("\n", NL), 1)
        old = "    _linearSolverOrdering.Clear();" + NL
        if t.count(old) != 1:
            sys.exit("_linearSolverOrdering.Clear not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + "    _cheshirePoseGroup.clear();  // cheshire: step 5n" + NL
                      + "    _cheshireIntrinsicGroup.clear();" + NL + "    _cheshireRigGroup.clear();" + NL, 1)
        for old, new, n in (
            ("_linearSolverOrdering.AddElementToGroup(poseBlockPtr, 1);",
             "_linearSolverOrdering.AddElementToGroup(poseBlockPtr, cheshirePoseGroup(view.getPoseId()));  // cheshire: step 5n", 2),
            ("_linearSolverOrdering.AddElementToGroup(intrinsicBlockPtr, 2);",
             "_linearSolverOrdering.AddElementToGroup(intrinsicBlockPtr, cheshireIntrinsicGroup(intrinsicId));  // cheshire: step 5n", 2),
            ("_linearSolverOrdering.AddElementToGroup(distortionBlockPtr, 2);",
             "_linearSolverOrdering.AddElementToGroup(distortionBlockPtr, cheshireDistortionGroup(intrinsicId, distortionBlockPtr == fakeDistortionBlockPtr));  // cheshire: step 5n", 2),
            ("_linearSolverOrdering.AddElementToGroup(rigBlockPtr, 1);",
             "_linearSolverOrdering.AddElementToGroup(rigBlockPtr, cheshireRigGroup(view.getRigId(), view.getSubPoseId()));  // cheshire: step 5n", 1),
            ("_linearSolverOrdering.AddElementToGroup(poseBlockPtrs[frameIdx-firstViewWithPose], 1);",
             "_linearSolverOrdering.AddElementToGroup(poseBlockPtrs[frameIdx-firstViewWithPose], cheshirePoseGroup(poseIdsVec.at(frameIdx)));  // cheshire: step 5n", 1),
        ):
            if t.count(old) != n:
                sys.exit(f"expected {n} of: {old}")
            t = t.replace(old, new)
        # the reference pose's group needs its id where only the pointer was kept
        old = "        double * referencePoseBlockPtr = nullptr;" + NL
        if t.count(old) != 1:
            sys.exit("referencePoseBlockPtr declaration not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + "        IndexT cheshireReferencePoseId = UndefinedIndexT;  // cheshire: step 5n" + NL, 1)
        old = "            referencePoseBlockPtr = _posesBlocks.at(refview.getPoseId()).data();" + NL
        if t.count(old) != 1:
            sys.exit("referencePoseBlockPtr assignment not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + "            cheshireReferencePoseId = refview.getPoseId();  // cheshire: step 5n" + NL, 1)
        old = "_linearSolverOrdering.AddElementToGroup(referencePoseBlockPtr, 1);"
        if t.count(old) != 1:
            sys.exit("reference pose AddElementToGroup not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, "_linearSolverOrdering.AddElementToGroup(referencePoseBlockPtr, cheshirePoseGroup(cheshireReferencePoseId));  // cheshire: step 5n", 1)
        # the thread count Ceres gets
        for old in ("    solverOptions.num_threads = _ceresOptions.nbThreads;", "    solverOptions.num_linear_solver_threads = _ceresOptions.nbThreads;"):
            if t.count(old) != 1:
                sys.exit(f"not found once: {old}")
            t = t.replace(old, old.replace("_ceresOptions.nbThreads", "cheshire::baThreads(_ceresOptions.nbThreads)") + "  // cheshire: step 5n", 1)
        bac.write_text(t, encoding="utf-8", newline="")

    # 5o. What a bundle adjustment costs around Ceres' Solve. The 5i profile is Ceres' own clock;
    #     adjust() also builds the Problem (a cost function and four ordering-group inserts per
    #     observation), evaluates every landmark residual twice for two log lines, writes the
    #     solution back and destroys the Problem. CHESHIRE_BA_PROFILE=1 now prints those phases
    #     too ("cheshire: BA adjust: ..."), and the two log-only evaluations are skipped unless
    #     CHESHIRE_BA_LOG_COST=1: they are a full single-threaded residual pass each and feed
    #     nothing but "landmarksBlocks cost".
    t = bac.read_text(encoding="utf-8")
    if "cheshireAdjustClock" not in t:
        old = ("    problemOptions.evaluation_callback = this;" + NL
               + "    ceres::Problem problem(problemOptions);" + NL
               + "    createProblem(sfmData, refineOptions, problem, landmarksBlockIds, temporalConstraintBlockIds);" + NL)
        if t.count(old) != 1:
            sys.exit("Problem creation not found once in adjust()")
        t = t.replace(old, "    problemOptions.evaluation_callback = this;" + NL + r"""    // cheshire (step 5o): the phases around Ceres' Solve, printed with CHESHIRE_BA_PROFILE=1. The
    // guard is declared before the Problem so its destructor runs after the Problem's and times it.
    static const bool cheshireBaProfile = std::getenv("CHESHIRE_BA_PROFILE") != nullptr;
    static const bool cheshireBaLogCost = std::getenv("CHESHIRE_BA_LOG_COST") != nullptr;
    using cheshireAdjustClock = std::chrono::steady_clock;
    struct CheshireDestroyTimer
    {
        bool on;
        cheshireAdjustClock::time_point t;
        ~CheshireDestroyTimer()
        {
            if (on)
                ALICEVISION_LOG_INFO("cheshire: BA adjust: destroy " << std::chrono::duration<double>(cheshireAdjustClock::now() - t).count() << " s");
        }
    } cheshireDestroyTimer{false, cheshireAdjustClock::now()};
    const cheshireAdjustClock::time_point cheshireT0 = cheshireAdjustClock::now();
    ceres::Problem problem(problemOptions);
    createProblem(sfmData, refineOptions, problem, landmarksBlockIds, temporalConstraintBlockIds);
    const double cheshireBuildS = std::chrono::duration<double>(cheshireAdjustClock::now() - cheshireT0).count();
    const cheshireAdjustClock::time_point cheshireT1 = cheshireAdjustClock::now();
""".replace("\n", NL), 1)
        # the two log-only evaluations: the post-solve one first, its text is a substring of the pre-solve one once patched
        post = ('    ALICEVISION_LOG_INFO("landmarksBlockIds : " << landmarksBlockIds.size());' + NL
                + "    evaluateOptions.residual_blocks = landmarksBlockIds;" + NL
                + "    problem.Evaluate(evaluateOptions, &cost, NULL, NULL, NULL);" + NL
                + '    ALICEVISION_LOG_INFO("landmarksBlocks cost : " << cost);' + NL
                + NL
                + "    if (temporalConstraintBlockIds.size() != 0)" + NL
                + "    {" + NL
                + "        evaluateOptions.residual_blocks = temporalConstraintBlockIds;" + NL
                + "        problem.Evaluate(evaluateOptions, &cost, NULL, NULL, NULL);" + NL
                + '        ALICEVISION_LOG_INFO("temporalConstraintBlocks cost : " << cost);' + NL
                + "    }" + NL)
        if t.count(post) != 1:
            sys.exit("post-solve evaluation block not found once in adjust()")
        t = t.replace(post, ("    if (cheshireBaLogCost)  // cheshire (step 5o)" + NL + "    {" + NL + post + "    }" + NL
                             + "    const double cheshireEval2S = std::chrono::duration<double>(cheshireAdjustClock::now() - cheshireT3).count();" + NL), 1)
        pre = ('    ALICEVISION_LOG_INFO("landmarksBlockIds : " << landmarksBlockIds.size());' + NL
               + "    ceres::Problem::EvaluateOptions evaluateOptions;" + NL
               + "    evaluateOptions.residual_blocks = landmarksBlockIds;" + NL
               + "    double cost;" + NL
               + "    problem.Evaluate(evaluateOptions, &cost, NULL, NULL, NULL);" + NL
               + '    ALICEVISION_LOG_INFO("landmarksBlocks cost : " << cost);' + NL
               + NL
               + "    if (temporalConstraintBlockIds.size() != 0)" + NL
               + "    {" + NL
               + "        evaluateOptions.residual_blocks = temporalConstraintBlockIds;" + NL
               + "        problem.Evaluate(evaluateOptions, &cost, NULL, NULL, NULL);" + NL
               + '        ALICEVISION_LOG_INFO("temporalConstraintBlocks cost : " << cost);' + NL
               + "    }" + NL)
        if t.count(pre) != 1:
            sys.exit("pre-solve evaluation block not found once in adjust()")
        t = t.replace(pre, ("    ceres::Problem::EvaluateOptions evaluateOptions;" + NL
                            + "    double cost = 0.0;" + NL
                            + "    if (cheshireBaLogCost)  // cheshire (step 5o): a full residual pass for a log line, off unless asked" + NL
                            + "    {" + NL
                            + '    ALICEVISION_LOG_INFO("landmarksBlockIds : " << landmarksBlockIds.size());' + NL
                            + "    evaluateOptions.residual_blocks = landmarksBlockIds;" + NL
                            + "    problem.Evaluate(evaluateOptions, &cost, NULL, NULL, NULL);" + NL
                            + '    ALICEVISION_LOG_INFO("landmarksBlocks cost : " << cost);' + NL
                            + NL
                            + "    if (temporalConstraintBlockIds.size() != 0)" + NL
                            + "    {" + NL
                            + "        evaluateOptions.residual_blocks = temporalConstraintBlockIds;" + NL
                            + "        problem.Evaluate(evaluateOptions, &cost, NULL, NULL, NULL);" + NL
                            + '        ALICEVISION_LOG_INFO("temporalConstraintBlocks cost : " << cost);' + NL
                            + "    }" + NL
                            + "    }" + NL
                            + "    const double cheshireEval1S = std::chrono::duration<double>(cheshireAdjustClock::now() - cheshireT1).count();" + NL), 1)
        old = "    ceres::Solve(options, &problem, &summary);" + NL
        if t.count(old) != 1:
            sys.exit("ceres::Solve not found once in adjust() (5o)")
        t = t.replace(old, ("    const cheshireAdjustClock::time_point cheshireT2 = cheshireAdjustClock::now();" + NL + old
                            + "    const double cheshireSolveS = std::chrono::duration<double>(cheshireAdjustClock::now() - cheshireT2).count();" + NL
                            + "    const cheshireAdjustClock::time_point cheshireT3 = cheshireAdjustClock::now();" + NL), 1)
        old = "    updateFromSolution(sfmData, refineOptions);" + NL
        if t.count(old) != 1:
            sys.exit("updateFromSolution call not found once in adjust()")
        t = t.replace(old, "    const cheshireAdjustClock::time_point cheshireT4 = cheshireAdjustClock::now();" + NL + old, 1)
        old = '    ALICEVISION_LOG_INFO("BundleAdjustmentCeres::adjust end");' + NL
        if t.count(old) != 1:
            sys.exit("adjust end log not found once")
        t = t.replace(old, r"""    if (cheshireBaProfile)
    {
        ALICEVISION_LOG_INFO("cheshire: BA adjust: build " << cheshireBuildS << " s, log evaluations " << cheshireEval1S + cheshireEval2S
                             << " s, solve " << cheshireSolveS << " s (Ceres preprocessor " << summary.preprocessor_time_in_seconds << ", minimizer "
                             << summary.minimizer_time_in_seconds << ", postprocessor " << summary.postprocessor_time_in_seconds << "), update "
                             << std::chrono::duration<double>(cheshireAdjustClock::now() - cheshireT4).count() << " s; " << summary.num_residual_blocks
                             << " residual blocks");
        cheshireDestroyTimer.on = true;
        cheshireDestroyTimer.t = cheshireAdjustClock::now();
    }
""".replace("\n", NL) + old, 1)
        if "#include <chrono>" not in t:
            inc0 = t.index("#include")
            t = t[:inc0] + "#include <chrono>  // cheshire: step 5o" + NL + t[inc0:]
        bac.write_text(t, encoding="utf-8", newline="")

    # 5p. The problem build, measured by 5o at 9.2 s of a 27 s bundle-adjustment budget on 41 views
    #     (68 solves). Upstream adds every parameter block of every observation to the Ceres
    #     ordering - four AddElementToGroup calls per observation, each a std::map find over all
    #     the blocks, on a block that is almost always already there. Now a block enters the
    #     ordering once (an unordered_set of pointers says whether it has). And the cost functions'
    #     parameter-block-size vectors are reserved before the push_backs.
    t = bah.read_text(encoding="utf-8")
    if "_cheshireOrdered" not in t:
        old = "#include <memory>" + NL
        if t.count(old) != 1:
            sys.exit("<memory> include not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, old + "#include <unordered_set>  // cheshire: step 5p" + NL, 1)
        old = "    int cheshireRigGroup(IndexT rigId, IndexT subPoseId)" + NL
        if t.count(old) != 1:
            sys.exit("cheshireRigGroup not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, r"""    // cheshire (step 5p): a block enters the ordering once
    std::unordered_set<const double*> _cheshireOrdered;
    void cheshireOrder(double* block, int group)
    {
        if (_cheshireOrdered.insert(block).second)
            _linearSolverOrdering.AddElementToGroup(block, group);
    }
""".replace("\n", NL) + old, 1)
        bah.write_text(t, encoding="utf-8", newline="")

    t = bac.read_text(encoding="utf-8")
    if "cheshireOrder(" not in t:
        n = t.count("_linearSolverOrdering.AddElementToGroup(")
        if n != 10:
            sys.exit(f"expected 10 AddElementToGroup calls in BundleAdjustmentCeres.cpp, found {n}")
        t = t.replace("_linearSolverOrdering.AddElementToGroup(", "cheshireOrder(")
        old = "    _cheshireRigGroup.clear();" + NL
        if t.count(old) != 1:
            sys.exit("_cheshireRigGroup.clear not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, old + "    _cheshireOrdered.clear();  // cheshire: step 5p" + NL, 1)
        bac.write_text(t, encoding="utf-8", newline="")

    ip = AV / "src/aliceVision/sfm/bundle/costfunctions/intrinsicsProject.hpp"
    t = ip.read_text(encoding="utf-8")
    if "cheshire: step 5p" not in t:
        old = "        mutable_parameter_block_sizes()->push_back(intrinsics->getParametersSize());" + NL
        if t.count(old) != 1:
            sys.exit("CostIntrinsicsProject push_back not found once")
        t = t.replace(old, "        mutable_parameter_block_sizes()->reserve(3);  // cheshire: step 5p" + NL + old, 1)
        ip.write_text(t, encoding="utf-8", newline="")

    # 5q. The problem build, lean. Per observation upstream did nine std::map lookups (view, pose,
    #     intrinsic and distortion blocks, the intrinsic object), a shared_ptr copy, a heap-allocated
    #     std::vector for four pointers, and Ceres' safety checks (a sort of the pointers and a
    #     duplicate scan); measured by 5o at about 2 us per residual block, 9.2 s of 27 s on 41 views.
    #     Now: the per-view blocks are looked up once per view per solve (an unordered_map filled on
    #     first use, which also enters them into the ordering), the landmark enters the ordering once,
    #     the four pointers go through Ceres' array overload, and the problem is built with
    #     disable_all_safety_checks (the problem is well-formed by construction).
    t = bac.read_text(encoding="utf-8")
    if "CheshireViewBlocks" not in t:
        old = "    problemOptions.evaluation_callback = this;" + NL
        if t.count(old) != 1:
            sys.exit("evaluation_callback line not found once (5q)")
        t = t.replace(old, old + "    problemOptions.disable_all_safety_checks = true;  // cheshire (step 5q): the problem is well-formed by construction" + NL, 1)
        sig = "void BundleAdjustmentCeres::addLandmarksToProblem("
        if t.count(sig) != 1:
            sys.exit("addLandmarksToProblem definition not found once")
        i = t.index(sig)
        j = t.index("{" + NL, i) + len("{" + NL)
        t = t[:j] + """    // cheshire (step 5q): what an observation needs of its view, looked up once per view per solve
    struct CheshireViewBlocks
    {
        const sfmData::View* view = nullptr;
        IndexT intrinsicId = UndefinedIndexT;
        double* pose = nullptr;
        double* intrinsic = nullptr;
        double* distortion = nullptr;
        std::shared_ptr<IntrinsicBase> intrinsicObject;
        bool ok = false;
    };
    std::unordered_map<IndexT, CheshireViewBlocks> cheshireViews;
    cheshireViews.reserve(sfmData.getViews().size());
""".replace("\n", NL) + t[j:]
        old = "        _allParametersBlocks.push_back(landmarkBlockPtr);" + NL
        if t.count(old) != 1:
            sys.exit("_allParametersBlocks.push_back(landmarkBlockPtr) not found once")
        t = t.replace(old, old + "        if (_ceresOptions.useParametersOrdering)  // cheshire (step 5q): once per landmark, not per observation" + NL
                      + "            cheshireOrder(landmarkBlockPtr, 0);" + NL, 1)
        old = ("            const sfmData::View& view = sfmData.getView(viewId);" + NL
               + "            const IndexT intrinsicId = view.getIntrinsicId();" + NL)
        if t.count(old) != 1:
            sys.exit("observation view lookup not found once")
        i = t.index(old)
        end_marker = ("                cheshireOrder(distortionBlockPtr, cheshireDistortionGroup(intrinsicId, distortionBlockPtr == fakeDistortionBlockPtr));  // cheshire: step 5n" + NL
                      + "            }" + NL)
        k = t.index(end_marker, i)
        if t.count(end_marker) != 1:
            sys.exit("distortion ordering line count unexpected (5q)")
        k += len(end_marker)
        t = t[:i] + """            // cheshire (step 5q): one lookup per view per solve instead of nine per observation
            auto cheshireViewIt = cheshireViews.find(viewId);
            if (cheshireViewIt == cheshireViews.end())
            {
                CheshireViewBlocks e;
                e.view = &sfmData.getView(viewId);
                e.intrinsicId = e.view->getIntrinsicId();
                e.ok = sfmData.isPoseAndIntrinsicDefined(*e.view);
                if (!e.ok)
                {
                    ALICEVISION_LOG_ERROR("We should not have an undefined pose here");
                }
                else if (sfmData.getAbsolutePose(e.view->getPoseId()).getState() == EEstimatorParameterState::IGNORED)
                {
                    ALICEVISION_LOG_ERROR("We should not have an ignored pose here");
                    e.ok = false;
                }
                if (e.ok)
                {
                    e.pose = _posesBlocks.at(e.view->getPoseId()).data();
                    e.intrinsic = _intrinsicsBlocks.at(e.intrinsicId).data();
                    e.intrinsicObject = _intrinsicObjects[e.intrinsicId];
                    e.distortion = fakeDistortionBlockPtr;
                    if (_distortionsBlocks.find(e.intrinsicId) != _distortionsBlocks.end())
                        e.distortion = _distortionsBlocks.at(e.intrinsicId).data();
                    if (_ceresOptions.useParametersOrdering)
                    {
                        cheshireOrder(e.pose, cheshirePoseGroup(e.view->getPoseId()));
                        cheshireOrder(e.intrinsic, cheshireIntrinsicGroup(e.intrinsicId));
                        cheshireOrder(e.distortion, cheshireDistortionGroup(e.intrinsicId, e.distortion == fakeDistortionBlockPtr));
                    }
                }
                cheshireViewIt = cheshireViews.emplace(viewId, e).first;
            }
            const CheshireViewBlocks& cheshireView = cheshireViewIt->second;
            if (!cheshireView.ok)
                continue;
            const sfmData::View& view = *cheshireView.view;
            const IndexT intrinsicId = cheshireView.intrinsicId;
            double* poseBlockPtr = cheshireView.pose;
            double* intrinsicBlockPtr = cheshireView.intrinsic;
            const std::shared_ptr<IntrinsicBase>& intrinsic = cheshireView.intrinsicObject;
            double* distortionBlockPtr = cheshireView.distortion;
""".replace("\n", NL) + t[k:]
        old = ("                std::vector<double*> params;" + NL
               + "                params.push_back(intrinsicBlockPtr);" + NL
               + "                params.push_back(distortionBlockPtr);" + NL
               + "                params.push_back(poseBlockPtr);" + NL
               + "                params.push_back(landmarkBlockPtr);" + NL
               + NL
               + "                ceres::ResidualBlockId blockId = problem.AddResidualBlock(costFunction, weightedLossFunction, params);" + NL)
        if t.count(old) != 1:
            sys.exit("simple-case params block not found once (5q)")
        t = t.replace(old, ("                double* params[4] = {intrinsicBlockPtr, distortionBlockPtr, poseBlockPtr, landmarkBlockPtr};  // cheshire (step 5q): no vector per observation" + NL
                            + "                ceres::ResidualBlockId blockId = problem.AddResidualBlock(costFunction, weightedLossFunction, params, 4);" + NL), 1)
        if "#include <unordered_map>" not in t:
            inc0 = t.index("#include")
            t = t[:inc0] + "#include <unordered_map>  // cheshire: step 5q" + NL + t[inc0:]
        bac.write_text(t, encoding="utf-8", newline="")

    # 5r. A Problem that lives across solves. hip/port/sfm_ba/persistent.inc has the reasoning and
    #     the sync; the engine keeps one BundleAdjustmentCeres for the whole reconstruction and
    #     each adjust() applies the delta. CHESHIRE_BA_PERSIST=0 restores the rebuild,
    #     CHESHIRE_BA_PERSIST_CHECK=1 verifies the residual set against the scene after every sync.
    shutil.copy2(ROOT / "hip" / "port" / "sfm_ba" / "persistent.inc", AV / "src/aliceVision/sfm/bundle/persistent.inc")

    t = bah.read_text(encoding="utf-8")
    if "_cheshireProblem" not in t:
        old = "        bool useParametersOrdering = true;" + NL
        if t.count(old) != 1:
            sys.exit("useParametersOrdering option not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, old + "        bool cheshirePersist = true;  // cheshire (step 5r): keep the Problem across solves; the engine turns it off under the local strategy" + NL, 1)
        old = "    bool adjust(sfmData::SfMData& sfmData, ERefineOptions refineOptions = REFINE_ALL);" + NL
        if t.count(old) != 1:
            sys.exit("adjust declaration not found once in BundleAdjustmentCeres.hpp")
        t = t.replace(old, old + "    void setOptions(const CeresOptions& options);  // cheshire (step 5r): new solver options for a bundle adjuster kept across solves" + NL, 1)
        old = "    int cheshireRigGroup(IndexT rigId, IndexT subPoseId)" + NL
        if t.count(old) != 1:
            sys.exit("cheshireRigGroup not found once in BundleAdjustmentCeres.hpp (5r)")
        t = t.replace(old, r"""    // cheshire (step 5r): the Problem that lives across solves (persistent.inc)
    struct CheshireObs
    {
        IndexT viewId;
        ceres::ResidualBlockId id;
    };
    struct CheshireLandmarkRec
    {
        double* block = nullptr;  // its slot in _landmarksBlocks; no map lookup per landmark per solve
        bool active = false;
        bool constant = false;
        std::vector<CheshireObs> obs;
    };
    std::unique_ptr<ceres::Problem> _cheshireProblem;
    std::map<IndexT, CheshireLandmarkRec> _cheshireLandmarkRecs;
    std::size_t _cheshireSlab = 0;
    bool cheshirePersistAllowed(const sfmData::SfMData& sfmData, ERefineOptions refineOptions) const;
    void cheshireDropPersistent();
    bool cheshirePersistentBuild(const sfmData::SfMData& sfmData, ERefineOptions refineOptions, std::vector<ceres::ResidualBlockId>& landmarksBlockIds);
    void cheshirePersistCheck(const sfmData::SfMData& sfmData, const ceres::Problem& problem, bool preSync) const;
""".replace("\n", NL) + old, 1)
        bah.write_text(t, encoding="utf-8", newline="")

    t = bac.read_text(encoding="utf-8")
    if "cheshirePersistentBuild" not in t:
        # the block vectors must keep their buffers: the Problem holds their pointers across solves,
        # and "block = intrinsicPtr->getParameters()" move-assigns a fresh buffer under them
        old = "        intrinsicBlock = intrinsicPtr->getParameters();" + NL
        if t.count(old) != 1:
            sys.exit("intrinsic block assignment not found once (5r)")
        t = t.replace(old, ("        {" + NL
                            + "            // cheshire (step 5r): in place - a persistent Problem holds this vector's pointer" + NL
                            + "            const std::vector<double> cheshireParams = intrinsicPtr->getParameters();" + NL
                            + "            if (intrinsicBlock.size() == cheshireParams.size())" + NL
                            + "                std::copy(cheshireParams.begin(), cheshireParams.end(), intrinsicBlock.begin());" + NL
                            + "            else" + NL
                            + "                intrinsicBlock = cheshireParams;" + NL
                            + "        }" + NL), 1)
        old = "                distortionBlock = distortion->getParameters();" + NL
        if t.count(old) != 1:
            sys.exit("distortion block assignment not found once (5r)")
        t = t.replace(old, ("                {" + NL
                            + "                    // cheshire (step 5r): in place, as above" + NL
                            + "                    const std::vector<double>& cheshireParams = distortion->getParameters();" + NL
                            + "                    if (distortionBlock.size() == cheshireParams.size())" + NL
                            + "                        std::copy(cheshireParams.begin(), cheshireParams.end(), distortionBlock.begin());" + NL
                            + "                    else" + NL
                            + "                        distortionBlock = cheshireParams;" + NL
                            + "                }" + NL), 1)
        old = "    _intrinsicObjects[intrinsicId].reset(intrinsicPtr->clone());" + NL
        if t.count(old) != 1:
            sys.exit("intrinsic clone not found once in BundleAdjustmentCeres.cpp")
        t = t.replace(old, ("    if (!_intrinsicObjects[intrinsicId])  // cheshire (step 5r): kept across solves; PrepareForEvaluation keeps its parameters at the blocks" + NL
                            + "        _intrinsicObjects[intrinsicId].reset(intrinsicPtr->clone());" + NL), 1)
        old = ("    const cheshireAdjustClock::time_point cheshireT0 = cheshireAdjustClock::now();" + NL
               + "    ceres::Problem problem(problemOptions);" + NL
               + "    createProblem(sfmData, refineOptions, problem, landmarksBlockIds, temporalConstraintBlockIds);" + NL)
        if t.count(old) != 1:
            sys.exit("adjust problem creation not found once (5r)")
        t = t.replace(old, r"""    const cheshireAdjustClock::time_point cheshireT0 = cheshireAdjustClock::now();
    // cheshire (step 5r): the persistent Problem when the scene allows it, else the rebuild
    std::unique_ptr<ceres::Problem> cheshireLocalProblem;
    ceres::Problem* cheshireProblemPtr = nullptr;
    if (cheshirePersistAllowed(sfmData, refineOptions) && cheshirePersistentBuild(sfmData, refineOptions, landmarksBlockIds))
        cheshireProblemPtr = _cheshireProblem.get();
    else
    {
        cheshireDropPersistent();
        cheshireLocalProblem = std::make_unique<ceres::Problem>(problemOptions);
        createProblem(sfmData, refineOptions, *cheshireLocalProblem, landmarksBlockIds, temporalConstraintBlockIds);
        cheshireProblemPtr = cheshireLocalProblem.get();
    }
    ceres::Problem& problem = *cheshireProblemPtr;
""".replace("\n", NL), 1)
        old = "            sfmData::Landmark& landmark = sfmData.getLandmarks().at(idLandmark);" + NL
        if t.count(old) != 1:
            sys.exit("landmark write-back lookup not found once (5r)")
        t = t.replace(old, r"""            // cheshire (step 5r): the persistent slab keeps slots of landmarks the scene has dropped
            const auto cheshireLandmarkIt = sfmData.getLandmarks().find(idLandmark);
            if (cheshireLandmarkIt == sfmData.getLandmarks().end())
                continue;
            sfmData::Landmark& landmark = cheshireLandmarkIt->second;
""".replace("\n", NL), 1)
        t = t.rstrip() + NL + NL + '#include "aliceVision/sfm/bundle/persistent.inc"  // cheshire: step 5r' + NL
        bac.write_text(t, encoding="utf-8", newline="")

    eh = AV / "src/aliceVision/sfm/pipeline/sequential/ReconstructionEngine_sequentialSfM.hpp"
    t = eh.read_text(encoding="utf-8")
    if "_cheshireBA" not in t:
        old = "    IndexT _resectionId;" + NL
        if t.count(old) != 1:
            sys.exit("_resectionId member not found once in ReconstructionEngine_sequentialSfM.hpp")
        t = t.replace(old, old + "    // cheshire (step 5r): one bundle adjuster for the whole reconstruction; its Problem lives across solves" + NL
                      + "    std::shared_ptr<BundleAdjustmentCeres> _cheshireBA;" + NL, 1)
        old = "class ReconstructionEngine_sequentialSfM"
        if t.count(old) < 1:
            sys.exit("engine class declaration not found")
        i = t.index(old)
        t = t[:i] + "class BundleAdjustmentCeres;  // cheshire: step 5r" + NL + NL + t[i:]
        eh.write_text(t, encoding="utf-8", newline="")

    t = eng.read_text(encoding="utf-8")
    if "_cheshireBA" not in t:
        old = "    BundleAdjustmentCeres BA(options, _params.minNbCamerasToRefinePrincipalPoint);" + NL
        if t.count(old) != 1:
            sys.exit("BA construction not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, r"""    // cheshire (step 5r): the same bundle adjuster every time, so its Problem can live across solves;
    // persistence pays only while every landmark is active (persistent.inc), so the local strategy turns it off
    options.cheshirePersist = !enableLocalStrategy;
    if (!_cheshireBA)
        _cheshireBA = std::make_shared<BundleAdjustmentCeres>(options, _params.minNbCamerasToRefinePrincipalPoint);
    else
        _cheshireBA->setOptions(options);
    BundleAdjustmentCeres& BA = *_cheshireBA;
""".replace("\n", NL), 1)
        eng.write_text(t, encoding="utf-8", newline="")

    # 5s. The passes after every bundle-adjustment iteration - the pixel and angle outlier tests
    #     over every observation and the per-pose observation recount - walk the whole scene for a
    #     solve the local strategy mostly did not touch. hip/port/sfm_ba/postAdjust.inc restricts
    #     them, exactly, to the landmarks observed by a refined pose or through a refined intrinsic
    #     (found through the tracks-per-view index) and the poses that lost observations; the
    #     upstream passes run unchanged without the local strategy and after a pose is erased.
    #     OPT-IN, CHESHIRE_SFM_LOCAL_PASSES=1: about 40 s either way at 884 views, the case is
    #     thousands of views (docs/04). CHESHIRE_SFM_PROFILE=1 prints one line per iteration.
    shutil.copy2(ROOT / "hip" / "port" / "sfm_ba" / "postAdjust.inc", AV / "src/aliceVision/sfm/pipeline/sequential/postAdjust.inc")
    t = eng.read_text(encoding="utf-8")
    if "postAdjust.inc" not in t:
        old = "#include <aliceVision/utils/filesIO.hpp>" + NL
        if t.count(old) != 1:
            sys.exit("filesIO include not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, old + '#include "aliceVision/sfm/pipeline/sequential/postAdjust.inc"  // cheshire: step 5s' + NL, 1)
        old = ("        nbOutliers = removeOutliers();" + NL
               + NL
               + "        std::set<IndexT> removedViewsIdIteration;" + NL
               + "        eraseUnstablePosesAndObservations(this->_sfmData, _params.minPointsPerPose, _params.minTrackLength, &removedViewsIdIteration);" + NL)
        if t.count(old) != 1:
            sys.exit("post-adjust block not found once in ReconstructionEngine_sequentialSfM.cpp")
        t = t.replace(old, r"""        // cheshire (step 5s): the passes after the solve, proportional to what it touched (postAdjust.inc)
        std::set<IndexT> cheshireViewsWithErasures;
        std::size_t cheshireCandidates = 0, cheshirePosesChecked = 0;
        bool cheshireFullPass = true;
        const auto cheshirePostT0 = std::chrono::steady_clock::now();
        const std::size_t cheshireLandmarksBefore = _sfmData.getLandmarks().size();
        nbOutliers = cheshire::removeOutliersAfterAdjust(_sfmData, _map_tracksPerView, _params.featureConstraint, _params.maxReprojectionError,
                                                         _params.minAngleForLandmark, enableLocalStrategy, refineOptions, cheshireViewsWithErasures,
                                                         cheshireCandidates, cheshireFullPass);

        std::set<IndexT> removedViewsIdIteration;
        cheshire::eraseUnstableAfterAdjust(_sfmData, _map_tracksPerView, _params.minPointsPerPose, _params.minTrackLength,
                                           enableLocalStrategy && !cheshireFullPass, cheshireViewsWithErasures, newReconstructedViews,
                                           &removedViewsIdIteration, cheshirePosesChecked);
        if (cheshire::sfmProfileEnabled())
            ALICEVISION_LOG_INFO("cheshire: post-adjust: " << (cheshireFullPass ? "full" : "restricted") << ", candidates " << cheshireCandidates << " of "
                                                           << cheshireLandmarksBefore << " landmarks, poses recounted " << cheshirePosesChecked << ", "
                                                           << std::chrono::duration<double>(std::chrono::steady_clock::now() - cheshirePostT0).count() << " s");
""".replace("\n", NL), 1)
        eng.write_text(t, encoding="utf-8", newline="")

    # 5t. DepthMap: each image decoded once per batch, and only if the device lacks it. The 12-view
    #     chunks of the 884-view set spent 64 of every 78 s per batch decoding EXRs on house-pc
    #     (GPU busy 10 % of the wall): the host image cache holds 16 full-resolution images, a
    #     batch needs ~30, and the prefetch loaded them all - including the ones the device cache
    #     already held from the previous batch - then the upload re-decoded the evicted ones.
    #     hip/port/sgm_fused/prefetch.cpp.txt is the loader; this step adds the two accessors it
    #     needs and upgrades a tree that carries the old block. Results are cache-independent
    #     (maps bit-identical, docs/04).
    ich = AV / "src/aliceVision/mvsUtils/ImagesCache.hpp"
    t = ich.read_text(encoding="utf-8")
    if "getCacheSize" not in t:
        old = "    void setCacheSize(int nbPreload);" + NL
        if t.count(old) != 1:
            sys.exit("setCacheSize declaration not found once in ImagesCache.hpp")
        t = t.replace(old, old + "    int getCacheSize() const { return _N_PRELOADED_IMAGES; }  // cheshire: the loader decodes in groups of this size" + NL, 1)
        ich.write_text(t, encoding="utf-8", newline="")
    dch = AV / "src/aliceVision/depthMap/cuda/host/DeviceCache.hpp"
    t = dch.read_text(encoding="utf-8")
    if "hasMipmapImage" not in t:
        old = "    void addCameraParams(int camId, int downscale, const mvsUtils::MultiViewParams& mp);" + NL
        if t.count(old) != 1:
            sys.exit("addCameraParams declaration not found once in DeviceCache.hpp")
        t = t.replace(old, "    /**" + NL
                           + "     * @brief cheshire: does the current device's mipmap cache hold this camera? No LRU update." + NL
                           + "     */" + NL
                           + "    bool hasMipmapImage(int camId);" + NL + NL + old, 1)
        dch.write_text(t, encoding="utf-8", newline="")
    dcc = AV / "src/aliceVision/depthMap/cuda/host/DeviceCache.cpp"
    t = dcc.read_text(encoding="utf-8")
    if "hasMipmapImage" not in t:
        old = "void DeviceCache::addCameraParams(int camId, int downscale, const mvsUtils::MultiViewParams& mp)" + NL
        if t.count(old) != 1:
            sys.exit("addCameraParams definition not found once in DeviceCache.cpp")
        t = t.replace(old, "bool DeviceCache::hasMipmapImage(int camId)" + NL + "{" + NL
                           + "    // cheshire: a lookup only; LRUCache::getIndex does not touch the recency list" + NL
                           + "    return getCurrentDeviceCache().mipmapCache.getIndex(camId) >= 0;" + NL + "}" + NL + NL + old, 1)
        dcc.write_text(t, encoding="utf-8", newline="")
    dme = AV / "src/aliceVision/depthMap/DepthMapEstimator.cpp"
    t = dme.read_text(encoding="utf-8")
    if "cheshire: depth map batch" not in t:
        i0 = t.find("        // cheshire: prefetch this batch's images in parallel.")
        anchor = "        // load tile R and corresponding T cameras in device cache" + NL
        i1 = t.find(anchor, i0)
        if i0 < 0 or i1 < i0:
            sys.exit("old prefetch block not found in DepthMapEstimator.cpp")
        t = t[:i0] + (ROOT / "hip/port/sgm_fused/prefetch.cpp.txt").read_text(encoding="utf-8").replace("\n", NL) + t[i1:]
    if "#include <chrono>" not in t:
        old = "#include <algorithm>" + NL
        if t.count(old) != 1:
            sys.exit("algorithm include not found once in DepthMapEstimator.cpp")
        t = t.replace(old, old + "#include <chrono>" + NL, 1)
    dme.write_text(t, encoding="utf-8", newline="")

    # 5u. DepthMap: the chunk's cameras in a nearest-neighbour tour over their centres. A 12-view
    #     chunk in index (view-id hash) order was 12 unrelated cameras needing 114 distinct images;
    #     in tour order consecutive R cameras share their T cameras, so the once-per-batch loader
    #     (5t) has something to reuse. Outputs are per view id and unchanged (bit-identical maps).
    #     CHESHIRE_DEPTHMAP_ORDER=0 restores the index order.
    mde = AV / "src/software/pipeline/main_depthMapEstimation.cpp"
    t = mde.read_text(encoding="utf-8")
    if "cheshireCameraOrder" not in t:
        func = (ROOT / "hip/port/sgm_fused/camera_order.cpp.txt").read_text(encoding="utf-8").replace("\n", NL)
        inc = "#include <boost/program_options.hpp>" + NL
        if t.count(inc) != 1:
            sys.exit("program_options include not found once in main_depthMapEstimation.cpp")
        t = t.replace(inc, inc + "#include <cstdlib>" + NL + "#include <numeric>" + NL + "#include <string>" + NL, 1)
        anchor = "int aliceVision_main(int argc, char* argv[])" + NL
        if t.count(anchor) != 1:
            sys.exit("aliceVision_main not found once in main_depthMapEstimation.cpp")
        t = t.replace(anchor, func + anchor, 1)
        old = "    // camera list" + NL + "    std::vector<int> cams;" + NL
        if t.count(old) != 1:
            sys.exit("camera list block not found once in main_depthMapEstimation.cpp")
        t = t.replace(old, "    const std::vector<int> cheshireOrder = cheshireCameraOrder(mp);  // cheshire: tour order" + NL + old, 1)
        old = "        for (int rc = 0; rc < mp.ncams; ++rc)  // process all cameras" + NL + "            cams.push_back(rc);" + NL
        if t.count(old) != 1:
            sys.exit("all-cameras loop not found once in main_depthMapEstimation.cpp")
        t = t.replace(old, "        for (int rc = 0; rc < mp.ncams; ++rc)  // process all cameras" + NL + "            cams.push_back(cheshireOrder[rc]);" + NL, 1)
        old = "        for (int rc = rangeStart; rc < std::min(rangeStart + rangeSize, mp.ncams); ++rc)" + NL + "            cams.push_back(rc);" + NL
        if t.count(old) != 1:
            sys.exit("range loop not found once in main_depthMapEstimation.cpp")
        t = t.replace(old, "        for (int rc = rangeStart; rc < std::min(rangeStart + rangeSize, mp.ncams); ++rc)" + NL + "            cams.push_back(cheshireOrder[rc]);" + NL, 1)
        mde.write_text(t, encoding="utf-8", newline="")

    # 5u (continued). The estimator assumed a batch's cameras are consecutive indices: the batch
    #     slot was tile.rc % nbRcPerBatch and the write loop walked the index range firstRc..lastRc.
    #     Both go by position in the tile list now (hip/port/sgm_fused/batch_by_position.py.txt
    #     holds the two replacements).
    t = dme.read_text(encoding="utf-8")
    if "by position in the batch" not in t:
        rep = (ROOT / "hip/port/sgm_fused/batch_by_position.py.txt").read_text(encoding="utf-8")
        oldSlot, newSlot, oldWrite, newWrite = [x.replace("\n", NL) for x in rep.split("=====\n")]
        if t.count(oldSlot) != 1 or t.count(oldWrite) != 1:
            sys.exit("batch slot / write block not found once in DepthMapEstimator.cpp")
        t = t.replace(oldSlot, newSlot, 1).replace(oldWrite, newWrite, 1)
        dme.write_text(t, encoding="utf-8", newline="")

    # 5v. EXR files read through OpenEXR directly. OpenImageIO's ImageBuf::read of the 73 MB half
    #     RGBA EXRs PrepareDenseScene writes takes 1.8-2.0 s each on the RX 9070 box regardless of
    #     threads; Imf::InputFile with the OpenEXR pool takes 0.3 s, identical values. The depth-map
    #     node decodes ~10 per view, texturing every one per sheet. The reader is its own
    #     translation unit of the image library (main_cameraInit.cpp includes io.cpp directly, so
    #     OpenEXR headers cannot live there); io.cpp calls it for float RGB/RGBA reads in the stored
    #     colour space and keeps OpenImageIO for everything else. CHESHIRE_EXR_DIRECT=0 disables it.
    shutil.copy2(ROOT / "hip" / "port" / "sgm_fused" / "cheshireExr.hpp.txt", AV / "src/aliceVision/image/cheshireExr.hpp")
    shutil.copy2(ROOT / "hip" / "port" / "sgm_fused" / "cheshireExr.cpp.txt", AV / "src/aliceVision/image/cheshireExr.cpp")
    icm = AV / "src/aliceVision/image/CMakeLists.txt"
    t = icm.read_text(encoding="utf-8")
    if "cheshireExr" not in t:
        for old, new in (("    io.hpp" + NL, "    io.hpp" + NL + "    cheshireExr.hpp" + NL), ("    io.cpp" + NL, "    io.cpp" + NL + "    cheshireExr.cpp" + NL)):
            if t.count(old) != 1:
                sys.exit("image/CMakeLists.txt source list anchor not found once")
            t = t.replace(old, new, 1)
        icm.write_text(t, encoding="utf-8", newline="")
    iop = AV / "src/aliceVision/image/io.cpp"
    t = iop.read_text(encoding="utf-8")
    if "cheshireReadExr" not in t:
        inc = "#include <aliceVision/image/io.hpp>" + NL
        if t.count(inc) != 1:
            sys.exit("io.hpp include not found once in image/io.cpp")
        t = t.replace(inc, inc + "#include <aliceVision/image/cheshireExr.hpp>  // cheshire: step 5v" + NL, 1)
        two = "        ALICEVISION_THROW_ERROR(\"Load of 2 channels is not supported. Image file: '\" + path + \"'.\")"
        anchor = two + NL + NL + "    oiio::ImageSpec configSpec;" + NL
        if t.count(anchor) != 1:
            sys.exit("readImage configSpec anchor not found once in image/io.cpp")
        call = (ROOT / "hip/port/sgm_fused/exr_call.cpp.txt").read_text(encoding="utf-8").replace("\n", NL)
        t = t.replace(anchor, two + NL + NL + call + "    oiio::ImageSpec configSpec;" + NL, 1)
        iop.write_text(t, encoding="utf-8", newline="")

    # 5w. OpenImageIO's resize with one thread when called from inside a parallel region. The
    #     depth-map node downscales every image on the host right after reading it (mp process
    #     downscale), inside the prefetch loop: twelve resizes at once, each with OpenImageIO's
    #     twelve workers, took 10 s per batch of 16 images against 0.7 s of decoding; the result
    #     does not depend on the thread count (docs/04, 0.3.4 "the depth-map node was decoding").
    iac = AV / "src/aliceVision/image/imageAlgo.cpp"
    t = iac.read_text(encoding="utf-8")
    if "step 5w" not in t:
        old = "#include <OpenImageIO/imagebufalgo.h>" + NL
        if t.count(old) != 1:
            sys.exit("imagebufalgo include not found once in imageAlgo.cpp")
        t = t.replace(old, old + "#include <aliceVision/alicevision_omp.hpp>  // cheshire: step 5w" + NL
                      + "#if !ALICEVISION_IS_DEFINED(ALICEVISION_HAVE_OPENMP)" + NL + "inline int omp_in_parallel() { return 0; }" + NL + "#endif" + NL, 1)
        old = "    oiio::ImageBufAlgo::resize(outBuf, inBuf, filter, filterSize, oiio::ROI::All());" + NL
        if t.count(old) != 1:
            sys.exit("resize call not found once in imageAlgo.cpp")
        t = t.replace(old, "    // cheshire (step 5w): one thread per call when the caller is already parallel. The depth-map" + NL
                      + "    // node downscales every image it reads, inside its prefetch loop: twelve of these at once," + NL
                      + "    // each handing OpenImageIO twelve workers, took 10 s per batch of 16 images for 0.7 s of" + NL
                      + "    // decoding. The result does not depend on the thread count." + NL
                      + "    oiio::ImageBufAlgo::resize(outBuf, inBuf, filter, filterSize, oiio::ROI::All(), omp_in_parallel() ? 1 : 0);" + NL, 1)
        iac.write_text(t, encoding="utf-8", newline="")

    # 5b. Let the CUDA architecture list be chosen. Upstream FORCEs "all-major", which on CUDA 12.9
    #     means real code for sm_50/60/70/80/90 plus PTX - five device compilations of every .cu
    #     when the cards in front of us are both compute 6.1. FORCE beats -D on the command line, so
    #     this has to be patched rather than passed. Upstream's default is kept when the variable is
    #     unset, so the HIP build and anyone building for a spread of cards is unaffected.
    av = AV / "src/CMakeLists.txt"
    t = av.read_text(encoding="utf-8")
    if "CHESHIRE_CUDA_ARCHS" not in t:
        nl_av = "\r\n" if "\r\n" in t else "\n"
        old = ('    set(CMAKE_CUDA_ARCHITECTURES' + nl_av
               + '        "all-major"' + nl_av
               + '        CACHE STRING "CUDA architectures used to build AliceVision" FORCE' + nl_av
               + '    )' + nl_av)
        if t.count(old) != 1:
            sys.exit("CMAKE_CUDA_ARCHITECTURES block not found once in src/CMakeLists.txt")
        new = ('    # cheshire: CHESHIRE_CUDA_ARCHS overrides the architecture list (e.g. 61 for' + nl_av
               + '    # Pascal). Upstream FORCEs all-major, so -D on the command line cannot win.' + nl_av
               + '    if (DEFINED ENV{CHESHIRE_CUDA_ARCHS})' + nl_av
               + '        set(CMAKE_CUDA_ARCHITECTURES' + nl_av
               + '            "$ENV{CHESHIRE_CUDA_ARCHS}"' + nl_av
               + '            CACHE STRING "CUDA architectures used to build AliceVision" FORCE' + nl_av
               + '        )' + nl_av
               + '    else()' + nl_av
               + old.rstrip(nl_av) + nl_av
               + '    endif()' + nl_av)
        t = t.replace(old, new, 1)
        av.write_text(t, encoding="utf-8", newline="")

    # 5. regenerate the reviewable patch.
    # CHESHIRE_SKIP_PATCH_EXPORT=1 leaves it alone. The submodule is normally cloned on Windows with
    # autocrlf, so its working tree has CRLF endings while the index has LF; a git diff taken from
    # WSL, where autocrlf is off, then reports every line of every file as changed and the artifact
    # balloons from 180 KB to 123 MB. The Linux build sets this.
    if os.environ.get("CHESHIRE_SKIP_PATCH_EXPORT"):
        print("applied; patch export skipped (CHESHIRE_SKIP_PATCH_EXPORT)")
        return
    subprocess.run(["git", "add", "-N", "src/aliceVision/depthMap/cuda/hip"], cwd=AV, check=True)
    # surrogateescape both ways: some upstream sources are not UTF-8 (a 0xf6 in a Latin-1 comment),
    # and the default decode fails outright on a UTF-8 locale. Windows survived it only because its
    # preferred encoding accepts the byte. This keeps the byte instead of replacing or dropping it.
    diff = subprocess.run(["git", "diff", "--no-color"], cwd=AV, check=True, capture_output=True,
                          text=True, encoding="utf-8", errors="surrogateescape").stdout
    out = ROOT / "patches" / "0002-hip-backend-cmake.patch"
    out.write_text(diff, encoding="utf-8", errors="surrogateescape", newline="\n")
    # The export is a blanket diff of the submodule, so anything edited by hand ends up in it. Step 0
    # resets TRACKED, which makes those safe; a file outside that list keeps whatever was done to it
    # and ships silently. A UVAtlas.cpp profiler reached the patch that way. Name them.
    touched = {l[len("+++ b/"):].strip() for l in diff.splitlines() if l.startswith("+++ b/")}
    generated = ("src/aliceVision/depthMap/cuda/hip", "src/aliceVision/fuseCut/gpu")
    stray = sorted(f for f in touched
                   if f not in TRACKED and not any(f.startswith(g + "/") for g in generated))
    if stray:
        print("  WARNING: in the patch but not reset by step 0, so hand edits persist:")
        for f in stray:
            print(f"    {f}")

    print(f"applied; {len(diff.splitlines())} diff lines -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
