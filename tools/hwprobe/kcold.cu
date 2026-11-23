// Does it matter that each kernel reads a *different* region of a big allocation (cold cache / TLB)?
// N kernels, each reading `mb` MB with 16-byte loads; region i starts at i*stride.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
__global__ void k_read(const uint4* w, float* a, int per_thread) {
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x); unsigned acc = 0; size_t nthreads = (size_t)gridDim.x * blockDim.x;
    for (int i = 0; i < per_thread; i++) acc += w[base + (size_t)i * nthreads].x;      // coalesced: neighbours read neighbours
    if (threadIdx.x == 0) a[blockIdx.x] = (float)acc;
}
int main() {
    cudaStream_t st; cudaStreamCreate(&st);
    size_t GB = (size_t)1 << 30, total = 16 * GB; uint4* w; float* a;
    if (cudaMalloc(&w, total)) { printf("alloc failed\n"); return 1; }
    cudaMalloc(&a, 1 << 20); cudaMemset(w, 1, total);
    const int N = 400;
    auto run = [&](const char* name, int blocks, int threads, int per_thread, size_t stride) {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < N; i++) k_read<<<blocks, threads, 0, st>>>((const uint4*)((const char*)w + (size_t)i * stride), a, per_thread);
        cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
        std::vector<double> t;
        for (int r = 0; r < 7; r++) { auto t0 = std::chrono::steady_clock::now(); cudaGraphLaunch(ge, st); cudaStreamSynchronize(st); t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); }
        std::sort(t.begin(), t.end());
        double mb = (double)blocks * threads * per_thread * 16 / 1e6;
        printf("%-34s %6.1f MB/kernel, regions %5.1f MB apart: %7.2f us per kernel (%6.0f GB/s)\n", name, mb, stride / 1e6, t[3] / N * 1e6, mb / 1e3 / (t[3] / N));
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    };
    run("small, same region", 8, 512, 9, 0);
    run("small, 1 MB apart", 8, 512, 9, 1 << 20);
    run("small, 39 MB apart", 8, 512, 9, (size_t)39 << 20);
    run("router-like 0.6MB, 39 MB apart", 8, 512, 9, (size_t)39 << 20);
    run("qkvr-like, same region", 538, 512, 9, 0);
    run("qkvr-like, 39.7 MB apart", 538, 512, 9, (size_t)39 << 20);
    run("qkvr-like 2x threads/half loads", 1076, 512, 5, (size_t)39 << 20);
    run("qkvr-like, 1 load per thread", 4838, 512, 1, (size_t)39 << 20);
    run("expert-like 23MB, 39 MB apart", 576, 256, 10, (size_t)39 << 20);
    run("expert-like, 1 load per thread", 5760, 256, 1, (size_t)39 << 20);
    return 0;
}
