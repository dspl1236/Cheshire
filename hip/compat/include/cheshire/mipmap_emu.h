// Cheshire: mipmapped-array emulation for HIP stacks without hipMallocMipmappedArray
// (Linux ROCm 7.2 reports "Mipmap not supported on one of the devices", e.g. RX 5500 XT).
//
// A "mipmapped array" becomes one plain 2D level per mip level; a texture object over it becomes
// one texture object per level whose descriptors are packed into a fixed-stride device buffer,
// and the 64-bit handle AliceVision carries around (cudaTextureObject_t) is that buffer. Device code samples
// with tex2DLod(handle, u, v, level) -> tex2D(slot[int(level + 0.5)], u, v): AliceVision only
// samples at integer levels (level = log2(scale / minDownscale)), so bilinear at that level is
// exactly what trilinear at an integer LOD returns. Fractional levels (custom patch-pattern
// subparts) are rounded to the nearest level.
//
// Level storage (runtime, CHESHIRE_MIPMAP_STORAGE=array|linear, default linear):
//   array   one hipArray per level (opaque, always VRAM);
//   linear  one pitched linear buffer per level, allocated through the memory bridge as class
//           Image and sampled through a hipResourceTypePitch2D texture. Linear levels are the
//           only camera-image storage the bridge can spill to system RAM (bridge.h, v2).
//
// Active when CHESHIRE_EMULATE_MIPMAP is defined (the default under HIP); define
// CHESHIRE_NATIVE_MIPMAP to use hipMallocMipmappedArray where the platform supports it.
#pragma once
#include <hip/hip_runtime.h>

namespace cheshire { namespace mip {

constexpr int kMaxLevels = 16;
// Packed level table. On AMD a texture object is a pointer to HIP_TEXTURE_OBJECT_SIZE_DWORD
// dwords of constant memory (image descriptor, then sampler descriptor) and nothing else, so the
// per-level descriptors are copied into one buffer of kMaxLevels fixed-stride slots; the fake
// handle is the buffer base and level l's texture object is base + l * kSlotBytes. Slots past the
// last level repeat the last level, so the device clamps against a constant and never loads a
// table entry: the descriptor address is arithmetic on kernel arguments, as with a native
// mipmapped texture, instead of two dependent loads per sample (the v0.2 table: handle -> TexSet
// -> hipTextureObject_t -> descriptor).
constexpr int kSlotBytes = 128;  // >= HIP_TEXTURE_OBJECT_SIZE_DWORD * 4 (80), cache-line aligned
static_assert(kSlotBytes >= HIP_TEXTURE_OBJECT_SIZE_DWORD * 4, "texture object slot too small");

}}  // namespace cheshire::mip

// Slot addressing in the constant address space, the way tex2D reads the descriptor.
#if defined(__HIP_DEVICE_COMPILE__)
typedef const unsigned int __attribute__((address_space(4)))* cheshire_cslot;
// Per-lane level (rare: only under useConsistentScale, where AliceVision derives a per-pixel
// level in computeRcTcMipmapLevels). Kept out of line so the common path stays small: the
// compiler emits a waterfall loop over the distinct descriptors here.
template<class T>
__device__ __attribute__((noinline)) T cheshire_tex2DLod_divergent(hipTextureObject_t handle, float u, float v, int l)
{
    return tex2D<T>((hipTextureObject_t)((cheshire_cslot)handle + size_t(l) * (cheshire::mip::kSlotBytes / 4)), u, v);
}
#endif

template<class T>
__device__ inline T cheshire_tex2DLod(hipTextureObject_t handle, float u, float v, float level)
{
    int l = int(level + 0.5f);
    l = l < 0 ? 0 : (l > cheshire::mip::kMaxLevels - 1 ? cheshire::mip::kMaxLevels - 1 : l);
#if defined(__HIP_DEVICE_COMPILE__)
    // The level is normally a kernel argument (wave-uniform), but it reaches here through float
    // VALU ops, so the backend cannot see that and would fetch the descriptor with vector loads
    // plus a readfirstlane per dword at every sample (121 of them in
    // volume_computeSimilarity_kernel). Check uniformity at run time instead: when every lane
    // agrees, address the slot with the scalar value, which costs one scalar descriptor load.
    const int first = __builtin_amdgcn_readfirstlane(l);
    if (__builtin_expect(__all(l == first), 1))
        return tex2D<T>((hipTextureObject_t)((cheshire_cslot)handle + size_t(first) * (cheshire::mip::kSlotBytes / 4)), u, v);
    return cheshire_tex2DLod_divergent<T>(handle, u, v, l);
#else
    const char* base = reinterpret_cast<const char*>(handle);
    return tex2D<T>(reinterpret_cast<hipTextureObject_t>(const_cast<char*>(base + l * cheshire::mip::kSlotBytes)), u, v);
#endif
}

// host-side implementation (plain __host__ functions: parsed in both compilation passes)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <cheshire/bridge.h>
#include "env.h"

namespace cheshire { namespace mip {

enum class Storage { Array, Linear };

// One mip level. `arr` for array storage; `dev`/`pitch` for linear storage (the fake hipArray_t
// handed to AliceVision is then the address of this record).
struct Level { hipArray_t arr = nullptr; void* dev = nullptr; size_t pitch = 0; size_t w = 0, h = 0; hipChannelFormatDesc desc{}; size_t elemBytes = 0; };
struct MipRec { std::vector<Level*> levels; hipChannelFormatDesc desc{}; size_t w = 0, h = 0; size_t elemBytes = 0; Storage st = Storage::Array; };
struct TexRec { std::vector<hipTextureObject_t> levels; void* dev; };  // dev: packed slot buffer (kMaxLevels * kSlotBytes)

struct Registry {
    std::mutex m;
    std::unordered_map<void*, MipRec> mips;      // fake hipMipmappedArray_t -> levels
    std::unordered_map<void*, Level*> linear;    // fake hipArray_t (Level*) -> level (linear storage only)
    std::unordered_map<void*, TexRec> texs;      // fake texture handle (device TexSet*) -> level textures
    Storage storage = Storage::Linear;   // same speed as arrays, and the bridge can account for it
    Registry() {
        const std::string s = ::cheshire::env::text("CHESHIRE_MIPMAP_STORAGE");
        if (s[0] == 'l' || s[0] == 'L') storage = Storage::Linear;   // s[0] is '\0' when unset or empty
        else if (s[0] == 'a' || s[0] == 'A') storage = Storage::Array;
    }
};
inline Registry& reg() { static Registry r; return r; }
inline Storage storage() { return reg().storage; }

inline hipArray_t handleOf(Level* l) { return l->arr ? l->arr : reinterpret_cast<hipArray_t>(l); }

// linear level lookup (nullptr when `a` is a real array)
inline Level* linearLevel(hipArray_t a) {
    std::lock_guard<std::mutex> g(reg().m);
    auto it = reg().linear.find(reinterpret_cast<void*>(a));
    return it == reg().linear.end() ? nullptr : it->second;
}

inline hipError_t mallocMipmappedArray(hipMipmappedArray_t* out, const hipChannelFormatDesc* desc, hipExtent extent, unsigned int levels, unsigned int flags) {
    (void)flags;  // surfaces are not used with the emulated levels (buffer-copy mip builder)
    if (levels == 0 || levels > (unsigned)kMaxLevels) return hipErrorInvalidValue;
    MipRec rec; rec.desc = *desc; rec.w = extent.width; rec.h = extent.height;
    rec.elemBytes = size_t(desc->x + desc->y + desc->z + desc->w) / 8;
    rec.st = storage();
    size_t w = extent.width, h = extent.height;
    auto fail = [&](hipError_t e) {
        for (Level* l : rec.levels) {
            if (l->arr) { cheshire::bridge::forgetExternal(l->arr); (void)hipFreeArray(l->arr); }
            if (l->dev) (void)cheshire::bridge::free(l->dev);
            delete l;
        }
        return e;
    };
    for (unsigned l = 0; l < levels; ++l) {
        Level* lv = new Level; lv->w = w ? w : 1; lv->h = h ? h : 1; lv->desc = *desc; lv->elemBytes = rec.elemBytes;
        hipError_t e;
        if (rec.st == Storage::Linear) {
            cheshire::bridge::ClassScope sc(cheshire::bridge::Class::Image);
            e = cheshire::bridge::mallocPitch(&lv->dev, &lv->pitch, lv->w * rec.elemBytes, lv->h);
        } else {
            e = hipMallocArray(&lv->arr, desc, lv->w, lv->h, hipArrayDefault);
            // array storage is driver memory the bridge cannot own; count it (0.3.2 item 5)
            if (e == hipSuccess) cheshire::bridge::noteExternal(lv->arr, lv->w * lv->h * rec.elemBytes, cheshire::bridge::Class::Image);
        }
        if (e != hipSuccess) { delete lv; return fail(e); }
        rec.levels.push_back(lv);
        w /= 2; h /= 2;
    }
    if (::cheshire::env::flag("CHESHIRE_BRIDGE_LOG"))
        std::fprintf(stderr, "[cheshire] mip: %zux%zu, %u levels, %zu B/texel, storage=%s\n", rec.w, rec.h, levels, rec.elemBytes, rec.st == Storage::Linear ? "linear" : "array");
    void* handle = rec.levels[0];  // the level-0 record doubles as the fake mipmap handle
    { std::lock_guard<std::mutex> g(reg().m);
      if (rec.st == Storage::Linear) for (Level* l : rec.levels) reg().linear[l] = l;
      reg().mips[handle] = std::move(rec); }
    *out = reinterpret_cast<hipMipmappedArray_t>(handle);
    return hipSuccess;
}

inline hipError_t getMipmappedArrayLevel(hipArray_t* out, hipMipmappedArray_t mip, unsigned int level) {
    std::lock_guard<std::mutex> g(reg().m);
    auto it = reg().mips.find(reinterpret_cast<void*>(mip));
    if (it == reg().mips.end() || level >= it->second.levels.size()) return hipErrorInvalidValue;
    *out = handleOf(it->second.levels[level]);
    return hipSuccess;
}

inline hipError_t freeMipmappedArray(hipMipmappedArray_t mip) {
    MipRec rec;
    { std::lock_guard<std::mutex> g(reg().m);
      auto it = reg().mips.find(reinterpret_cast<void*>(mip));
      if (it == reg().mips.end()) return hipErrorInvalidValue;
      rec = std::move(it->second); reg().mips.erase(it);
      for (Level* l : rec.levels) reg().linear.erase(l); }
    hipError_t err = hipSuccess;
    for (Level* l : rec.levels) {
        if (l->arr) cheshire::bridge::forgetExternal(l->arr);
        hipError_t e = l->arr ? hipFreeArray(l->arr) : cheshire::bridge::free(l->dev);
        if (e != hipSuccess) err = e;
        delete l;
    }
    return err;
}

// hipArrayGetInfo may be unavailable on some stacks; answer from the registry for our levels.
inline hipError_t arrayGetInfo(hipChannelFormatDesc* desc, hipExtent* extent, unsigned int* flags, hipArray_t array) {
    std::lock_guard<std::mutex> g(reg().m);
    for (auto& kv : reg().mips) {
        const MipRec& r = kv.second;
        for (size_t l = 0; l < r.levels.size(); ++l) if (handleOf(r.levels[l]) == array) {
            if (desc) *desc = r.desc;
            if (extent) { extent->width = r.levels[l]->w; extent->height = r.levels[l]->h; extent->depth = 0; }
            if (flags) *flags = 0;
            return hipSuccess;
        }
    }
    return hipArrayGetInfo(desc, extent, flags, array);
}

// Copies into a level: real arrays go to the HIP array copy, linear levels to hipMemcpy2D.
inline hipError_t memcpy2DToArray(hipArray_t dst, size_t wOffset, size_t hOffset, const void* src, size_t spitch, size_t width, size_t height, hipMemcpyKind kind) {
    if (Level* l = linearLevel(dst))
        return hipMemcpy2D(static_cast<char*>(l->dev) + hOffset * l->pitch + wOffset, l->pitch, src, spitch, width, height, kind);
    return hipMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}
inline hipError_t memcpy3D(const hipMemcpy3DParms* p) {
    if (p->dstArray) {
        if (Level* l = linearLevel(p->dstArray)) {
            const size_t elemBytes = l->elemBytes;
            if (elemBytes == 0) return hipErrorInvalidValue;
            return hipMemcpy2D(static_cast<char*>(l->dev) + p->dstPos.y * l->pitch + p->dstPos.x * elemBytes, l->pitch,
                               static_cast<const char*>(p->srcPtr.ptr) + p->srcPos.y * p->srcPtr.pitch + p->srcPos.x,
                               p->srcPtr.pitch, p->extent.width * elemBytes, p->extent.height, p->kind);
        }
    }
    return hipMemcpy3D(p);
}

inline hipError_t createTextureObject(hipTextureObject_t* out, const hipResourceDesc* rd, const hipTextureDesc* td, const hipResourceViewDesc* vd) {
    if (rd->resType == hipResourceTypeArray) {
        // a single level asked for by "array": the mip builder samples the previous level this way
        if (Level* l = linearLevel(rd->res.array.array)) {
            hipResourceDesc lrd{}; lrd.resType = hipResourceTypePitch2D; lrd.res.pitch2D.devPtr = l->dev; lrd.res.pitch2D.desc = l->desc;
            lrd.res.pitch2D.width = l->w; lrd.res.pitch2D.height = l->h; lrd.res.pitch2D.pitchInBytes = l->pitch;
            return hipCreateTextureObject(out, &lrd, td, vd);
        }
        return hipCreateTextureObject(out, rd, td, vd);
    }
    if (rd->resType != hipResourceTypeMipmappedArray) return hipCreateTextureObject(out, rd, td, vd);
    MipRec* rec = nullptr;
    { std::lock_guard<std::mutex> g(reg().m);
      auto it = reg().mips.find(reinterpret_cast<void*>(rd->res.mipmap.mipmap));
      if (it != reg().mips.end()) rec = &it->second; }
    if (!rec) return hipCreateTextureObject(out, rd, td, vd);  // a native mipmapped array
    TexRec tr; tr.dev = nullptr;
    hipTextureDesc ld = *td; ld.mipmapFilterMode = hipFilterModePoint; ld.maxMipmapLevelClamp = 0; ld.minMipmapLevelClamp = 0; ld.mipmapLevelBias = 0;
    for (Level* l : rec->levels) {
        hipResourceDesc lrd{};
        if (l->arr) { lrd.resType = hipResourceTypeArray; lrd.res.array.array = l->arr; }
        else {
            lrd.resType = hipResourceTypePitch2D; lrd.res.pitch2D.devPtr = l->dev; lrd.res.pitch2D.desc = rec->desc;
            lrd.res.pitch2D.width = l->w; lrd.res.pitch2D.height = l->h; lrd.res.pitch2D.pitchInBytes = l->pitch;
        }
        hipTextureObject_t t = 0;
        hipError_t e = hipCreateTextureObject(&t, &lrd, &ld, nullptr);
        if (e != hipSuccess) { for (auto x : tr.levels) (void)hipDestroyTextureObject(x); return e; }
        tr.levels.push_back(t);
    }
    // pack the per-level descriptors into fixed-stride slots (see cheshire_tex2DLod); the
    // descriptors only reference the level storage, which outlives the texture, so the level
    // texture objects are kept solely to be destroyed together with the packed copy.
    const int n = int(tr.levels.size());
    hipError_t e = hipMalloc(&tr.dev, size_t(kMaxLevels) * kSlotBytes);
    for (int i = 0; e == hipSuccess && i < kMaxLevels; ++i) {
        const hipTextureObject_t src = tr.levels[i < n ? i : n - 1];
        e = hipMemcpy(static_cast<char*>(tr.dev) + size_t(i) * kSlotBytes, reinterpret_cast<const void*>(src), HIP_TEXTURE_OBJECT_SIZE_DWORD * 4, hipMemcpyDefault);
    }
    if (e != hipSuccess) { for (auto x : tr.levels) (void)hipDestroyTextureObject(x); if (tr.dev) (void)hipFree(tr.dev); return e; }
    { std::lock_guard<std::mutex> g(reg().m); reg().texs[tr.dev] = tr; }
    *out = reinterpret_cast<hipTextureObject_t>(tr.dev);
    return hipSuccess;
}

inline hipError_t destroyTextureObject(hipTextureObject_t tex) {
    TexRec tr; bool ours = false;
    { std::lock_guard<std::mutex> g(reg().m);
      auto it = reg().texs.find(reinterpret_cast<void*>(tex));
      if (it != reg().texs.end()) { tr = it->second; reg().texs.erase(it); ours = true; } }
    if (!ours) return hipDestroyTextureObject(tex);
    hipError_t err = hipSuccess;
    for (auto t : tr.levels) { hipError_t e = hipDestroyTextureObject(t); if (e != hipSuccess) err = e; }
    if (hipFree(tr.dev) != hipSuccess) err = hipErrorInvalidValue;
    return err;
}

}}  // namespace cheshire::mip
