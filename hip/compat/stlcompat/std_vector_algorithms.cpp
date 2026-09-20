// Cheshire toolchain shim, part 2: MSVC STL vectorized-algorithm helpers needed by the CUDA
// Windows build specifically.
//
// std_minmax_element.cpp covers what the clang-cl / HIP build needs against MSVC 14.50. The CUDA
// build has to use MSVC 14.44, because CUDA 12.9's crt/host_config.h rejects _MSC_VER >= 1950 and
// 14.50 is exactly 1950 (docs/18). 14.44 is *older* than 14.50, so its msvcp140 exports even less
// of what the prebuilt vcpkg archive (alicevision/vcpkg 2026.09.01, built with a newer STL) was
// compiled against - these turned up as LNK2019 in CoinUtils when linking
// aliceVision_lInftyComputerVision.
//
// Semantics follow microsoft/STL vector_algorithms.cpp. The _N suffix is the element width in
// bytes; the STL only routes trivially-comparable types here, so a bitwise compare of the
// corresponding unsigned integer is the correct equality.
#include <cstddef>
#include <cstdint>

extern "C" {

// First position p in [first, last) with p[0] == p[1]; last if there is none.
// Referenced by std::_Adjacent_find_vectorized<const int>(const int*, const int*).
const void* __stdcall __std_adjacent_find_4(const void* first, const void* last) noexcept {
    const uint32_t* p = static_cast<const uint32_t*>(first);
    const uint32_t* e = static_cast<const uint32_t*>(last);
    if (p == e) return last;
    for (const uint32_t* n = p + 1; n != e; ++p, ++n)
        if (*p == *n) return p;
    return last;
}

const void* __stdcall __std_adjacent_find_8(const void* first, const void* last) noexcept {
    const uint64_t* p = static_cast<const uint64_t*>(first);
    const uint64_t* e = static_cast<const uint64_t*>(last);
    if (p == e) return last;
    for (const uint64_t* n = p + 1; n != e; ++p, ++n)
        if (*p == *n) return p;
    return last;
}

// In-place removal of consecutive duplicates; returns the new logical end.
// Referenced by std::_Unique_vectorized<int>(int*, int*).
void* __stdcall __std_unique_4(void* first, void* last) noexcept {
    uint32_t* p = static_cast<uint32_t*>(first);
    uint32_t* e = static_cast<uint32_t*>(last);
    if (p == e) return last;
    uint32_t* out = p;
    for (uint32_t* n = p + 1; n != e; ++n)
        if (*n != *out) *++out = *n;
    return out + 1;
}

void* __stdcall __std_unique_8(void* first, void* last) noexcept {
    uint64_t* p = static_cast<uint64_t*>(first);
    uint64_t* e = static_cast<uint64_t*>(last);
    if (p == e) return last;
    uint64_t* out = p;
    for (uint64_t* n = p + 1; n != e; ++n)
        if (*n != *out) *++out = *n;
    return out + 1;
}

// 1- and 2-byte widths exist too and cost nothing to provide; the linker drops what is unused.
const void* __stdcall __std_adjacent_find_1(const void* first, const void* last) noexcept {
    const uint8_t* p = static_cast<const uint8_t*>(first);
    const uint8_t* e = static_cast<const uint8_t*>(last);
    if (p == e) return last;
    for (const uint8_t* n = p + 1; n != e; ++p, ++n)
        if (*p == *n) return p;
    return last;
}

const void* __stdcall __std_adjacent_find_2(const void* first, const void* last) noexcept {
    const uint16_t* p = static_cast<const uint16_t*>(first);
    const uint16_t* e = static_cast<const uint16_t*>(last);
    if (p == e) return last;
    for (const uint16_t* n = p + 1; n != e; ++p, ++n)
        if (*p == *n) return p;
    return last;
}

void* __stdcall __std_unique_1(void* first, void* last) noexcept {
    uint8_t* p = static_cast<uint8_t*>(first);
    uint8_t* e = static_cast<uint8_t*>(last);
    if (p == e) return last;
    uint8_t* out = p;
    for (uint8_t* n = p + 1; n != e; ++n)
        if (*n != *out) *++out = *n;
    return out + 1;
}

void* __stdcall __std_unique_2(void* first, void* last) noexcept {
    uint16_t* p = static_cast<uint16_t*>(first);
    uint16_t* e = static_cast<uint16_t*>(last);
    if (p == e) return last;
    uint16_t* out = p;
    for (uint16_t* n = p + 1; n != e; ++n)
        if (*n != *out) *++out = *n;
    return out + 1;
}

}  // extern "C"
