// Test helper: take all free GPU memory, hold it for N seconds.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
int main(int argc, char** argv) {
    size_t total = 0; void* p;
    for (size_t chunk : {size_t(64) << 20, size_t(4) << 20, size_t(1) << 20, size_t(64) << 10})
        while (cudaMalloc(&p, chunk) == cudaSuccess) total += chunk;
    size_t fr, tot; cudaGetLastError(); cudaMemGetInfo(&fr, &tot);
    printf("hog: took %zu MiB, %zu KiB free now\n", total >> 20, fr >> 10); fflush(stdout);
    sleep(argc > 1 ? atoi(argv[1]) : 60);
    return 0;
}
