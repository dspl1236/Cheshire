#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
    hipDeviceProp_t p; hipGetDeviceProperties(&p, 0);
    printf("maxTexture2DLayered: %d %d %d\n", p.maxTexture2DLayered[0], p.maxTexture2DLayered[1], p.maxTexture2DLayered[2]);
    struct { int w, h, l; } cases[] = {{2048,2048,3},{4032,3024,3},{4032,3024,8},{8192,8192,3}};
    for (auto c : cases) {
        hipArray_t arr = nullptr;
        hipChannelFormatDesc fd = hipCreateChannelDesc<float>();
        hipExtent ext = make_hipExtent(c.w, c.h, c.l);
        hipError_t e = hipMalloc3DArray(&arr, &fd, ext, hipArrayLayered);
        printf("%5d x %5d x %d -> %s%s\n", c.w, c.h, c.l, hipGetErrorString(e), (e==hipSuccess && arr) ? " (ok)" : " (FAILED)");
        if (e == hipSuccess && arr) hipFreeArray(arr);
    }
    return 0;
}
