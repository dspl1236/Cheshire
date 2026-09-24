// Cheshire: single device translation unit for AliceVision's depthMap CUDA sources.
// Windows HIP (ROCm 7.2.1) cannot do relocatable device code (-fgpu-rdc: the
// clang-offload-bundler/llvm-objcopy step fails on COFF objects), and the kernels
// reference __constant__ symbols (camera params, patch pattern, gaussian tables)
// defined in other .cu files. Compiling them as one TU resolves that without RDC.
// Host .cpp files stay separate: cudaMemcpyToSymbol works across TUs without RDC.
#include <aliceVision/depthMap/cuda/device/DeviceCameraParams.cu>
#include <aliceVision/depthMap/cuda/device/DevicePatchPattern.cu>
#include <aliceVision/depthMap/cuda/imageProcessing/deviceGaussianFilter.cu>
#include <aliceVision/depthMap/cuda/imageProcessing/deviceColorConversion.cu>
#include <aliceVision/depthMap/cuda/imageProcessing/deviceMipmappedArray.cu>
#include <aliceVision/depthMap/cuda/planeSweeping/deviceDepthSimilarityMap.cu>
#include <aliceVision/depthMap/cuda/planeSweeping/deviceSimilarityVolume.cu>
// Host files that write the __constant__ symbols (cudaMemcpyToSymbol). Without RDC the
// host-side shadow variables are TU-local too, so these must live in the same TU.
#include <aliceVision/depthMap/cuda/host/DeviceCache.cpp>
#include <aliceVision/depthMap/cuda/host/patchPattern.cpp>
// Cheshire step 6m: CheshireEXR (hip/port/cheshireexr, copied under cuda/hip/cheshireexr) and the
// loader that decodes the depth-map node's images with it
#include <aliceVision/depthMap/cuda/hip/cheshireexr/exrHost.cpp>
#include <aliceVision/depthMap/cuda/hip/cheshireexr/exrGPU.cu>
#include <aliceVision/depthMap/cuda/imageProcessing/cheshireDeviceExr.cu>
// Cheshire step 6d: last, because it (and the resize header it includes) turns fp contraction off
// for the rest of the translation unit.
#include <aliceVision/depthMap/cuda/imageProcessing/cheshireDownscale.cu>
