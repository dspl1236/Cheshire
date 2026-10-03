// Cheshire: the RDNA4 (gfx12) INT8 matrix instruction's register layout, measured on the card before a kernel relies
// on it. v_wmma_i32_16x16x16_iu8 (wave32): D[16x16 int32] = A[16x16 u8] * B[16x16 u8] + C. The hypothesis tested
// (RDNA4 ISA, "WMMA" operand layouts), per lane l of the wave:
//   A: 8 bytes = row m = l % 16, k = 8 * (l / 16) .. +7      (two VGPRs, byte j of the 8 = k offset j)
//   B: 8 bytes = column n = l % 16, k = 8 * (l / 16) .. +7
//   D: 8 int32 = column n = l % 16, rows m = 8 * (l / 16) + v for v = 0..7
// The probe fills A and B with distinct bytes, runs one instruction, and compares every D element the lanes return
// with the CPU's product. Unsigned on both sides, so bytes 128..255 are tested too.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

typedef int v2i __attribute__((ext_vector_type(2)));
typedef int v8i __attribute__((ext_vector_type(8)));

__global__ void probe(const uint8_t* A, const uint8_t* B, int* D)
{
    const int l = threadIdx.x;   // one wave of 32
    const int r = l % 16, k0 = 8 * (l / 16);
    uint8_t a[8], b[8];
    for (int j = 0; j < 8; ++j) { a[j] = A[r * 16 + k0 + j]; b[j] = B[(k0 + j) * 16 + r]; }   // A row-major MxK, B KxN
    v2i av, bv;   // little-endian: byte j of the 8 is bits 8*(j%4) of word j/4
    av[0] = int(uint32_t(a[0]) | uint32_t(a[1]) << 8 | uint32_t(a[2]) << 16 | uint32_t(a[3]) << 24);
    av[1] = int(uint32_t(a[4]) | uint32_t(a[5]) << 8 | uint32_t(a[6]) << 16 | uint32_t(a[7]) << 24);
    bv[0] = int(uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24);
    bv[1] = int(uint32_t(b[4]) | uint32_t(b[5]) << 8 | uint32_t(b[6]) << 16 | uint32_t(b[7]) << 24);
    v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
    v8i d = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(false, av, false, bv, c, false);
    for (int v = 0; v < 8; ++v) D[l * 8 + v] = d[v];
}

int main()
{
    uint8_t A[256], B[256];
    for (int i = 0; i < 256; ++i) { A[i] = uint8_t((i * 37 + 11) & 0xff); B[i] = uint8_t((i * 91 + 200) & 0xff); }
    int ref[16][16];
    for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
            int s = 0;
            for (int k = 0; k < 16; ++k) s += int(A[m * 16 + k]) * int(B[k * 16 + n]);
            ref[m][n] = s;
        }
    uint8_t *dA, *dB; int* dD; int D[256];
    hipMalloc(&dA, 256); hipMalloc(&dB, 256); hipMalloc(&dD, 256 * sizeof(int));
    hipMemcpy(dA, A, 256, hipMemcpyHostToDevice); hipMemcpy(dB, B, 256, hipMemcpyHostToDevice);
    hipLaunchKernelGGL(probe, dim3(1), dim3(32), 0, 0, dA, dB, dD);
    if (hipDeviceSynchronize() != hipSuccess) { std::printf("kernel failed\n"); return 2; }
    hipMemcpy(D, dD, sizeof D, hipMemcpyDeviceToHost);
    int ok = 0, bad = 0;
    for (int l = 0; l < 32; ++l)
        for (int v = 0; v < 8; ++v) {
            const int n = l % 16, m = 8 * (l / 16) + v;
            if (D[l * 8 + v] == ref[m][n]) ++ok; else if (bad++ < 6) std::printf("lane %d v %d: got %d, expected D[%d][%d] = %d\n", l, v, D[l * 8 + v], m, n, ref[m][n]);
        }
    std::printf("layout hypothesis: %d of 256 elements match, %d differ\n", ok, bad);
    if (bad) {   // where does each returned value sit in the reference, if anywhere?
        for (int l = 0; l < 4; ++l)
            for (int v = 0; v < 8; ++v)
                for (int m = 0; m < 16; ++m)
                    for (int n = 0; n < 16; ++n)
                        if (ref[m][n] == D[l * 8 + v]) { std::printf("  lane %d v %d = ref[%d][%d]\n", l, v, m, n); m = 16; break; }
    }
    return bad ? 1 : 0;
}
