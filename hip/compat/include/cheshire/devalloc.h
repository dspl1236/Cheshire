// Cheshire: device allocation for the GPU ports (matcher, depth-map filter, votes, max-flow,
// knn, blur, texturing), routed through the memory bridge on BOTH backends.
//
// On HIP every port already reached the bridge: cuda_to_hip.h is force-included into each HIP
// translation unit and defines cudaMalloc/cudaFree as bridge calls. The CUDA build has no such
// force-include (bridge.h reaches it through depthMap's memory.hpp only), so the ports' plain
// cudaMalloc calls bypassed the bridge there: nothing counted them, the per-class summary showed
// no "other" line, and a CUDA card's VRAM had a population the planner could not see. Item 4 of
// 0.3.2 (docs/04). The ports call these instead, and the two names below are the same bridge
// functions on either backend; bridge.h supplies hipError_t as cudaError_t on CUDA through
// hip_to_cuda.h and cuda_to_hip.h maps the other way on HIP.
//
// Reached as <aliceVision/depthMap/cuda/hip/cheshire/devalloc.h>: the generator copies this
// directory into the tree, and the CUDA build must never see hip/compat/include on its include
// path (it holds cuda_runtime.h shims that would shadow the real headers).
#pragma once

#include "bridge.h"

namespace cheshire {

inline hipError_t devMalloc(void** devPtr, size_t bytes) { return bridge::malloc(devPtr, bytes); }
// the typed form the ports use (cudaMalloc has the same templated overload in cuda_runtime.h)
template <class T>
inline hipError_t devMalloc(T** devPtr, size_t bytes) { return bridge::malloc(reinterpret_cast<void**>(devPtr), bytes); }
inline hipError_t devFree(void* devPtr) { return bridge::free(devPtr); }

}  // namespace cheshire
