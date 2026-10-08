// Scratch: the cost of one depth map image upload on the device-downscale path (6000x3376 float RGBA, 324 MB):
// hipMalloc + pageable hipMemcpy2D + hipFree, against a pinned source and against a kept buffer.
// RESULT (2026-10-08, RX 9070): alloc 0 ms after the first, pageable copy 25-26 ms, free 0 ms, pinned copy 11.5 ms: of an
// upload's ~45 ms the copy is the larger part, and the rest is the mipmap build and its device-wide waits (roadmap, 0.4.3).
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using clk = std::chrono::steady_clock;
static double ms(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }
int main() {
  const size_t w = 6000, h = 3376, row = w * 16, bytes = row * h;
  std::vector<char> pageable(bytes, 1);
  void* pinned = nullptr; hipHostMalloc(&pinned, bytes); std::memset(pinned, 1, bytes);
  void* kept = nullptr; size_t pitch = 0; hipMallocPitch(&kept, &pitch, row, h);
  for (int rep = 0; rep < 3; ++rep) {
    auto t = clk::now(); void* d = nullptr; size_t p = 0; hipMallocPitch(&d, &p, row, h); double tA = ms(t);
    t = clk::now(); hipMemcpy2D(d, p, pageable.data(), row, row, h, hipMemcpyHostToDevice); double tC = ms(t);
    t = clk::now(); hipFree(d); double tF = ms(t);
    t = clk::now(); hipMemcpy2D(kept, pitch, pageable.data(), row, row, h, hipMemcpyHostToDevice); double tK = ms(t);
    t = clk::now(); hipMemcpy2D(kept, pitch, pinned, row, row, h, hipMemcpyHostToDevice); double tP = ms(t);
    printf("alloc %.1f ms, pageable copy %.1f ms, free %.1f ms | kept buffer pageable %.1f ms | pinned %.1f ms\n", tA, tC, tF, tK, tP);
  }
  return 0;
}
