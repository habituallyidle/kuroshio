// What does one kernel cost when replayed from a CUDA graph? Chains of N dependent kernels of various shapes.
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
__global__ void k_tiny(float* a) { if (threadIdx.x == 0 && blockIdx.x == 0) a[0] += 1.0f; }
__global__ void k_sync(float* a) { __shared__ float s[256]; s[threadIdx.x] = a[threadIdx.x]; __syncthreads(); if (threadIdx.x == 0 && blockIdx.x == 0) a[0] = s[1] + 1.0f; }
__global__ void k_loads(const float* w, float* a, int n) {   // n dependent-free loads per thread, then one write
    float acc = 0; size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * (size_t)n;
    for (int i = 0; i < n; i++) acc += w[base + i];
    if (threadIdx.x == 0) a[blockIdx.x] = acc;
}
int main() {
    cudaStream_t st; cudaStreamCreate(&st);
    float *a, *w; cudaMalloc(&a, 1 << 20); size_t WB = (size_t)1 << 30; cudaMalloc(&w, WB); cudaMemset(w, 0, WB); cudaMemset(a, 0, 1 << 20);
    const int N = 500;
    auto run = [&](const char* name, auto launch) {
        cudaGraph_t g; cudaGraphExec_t ge;
        cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < N; i++) launch();
        cudaStreamEndCapture(st, &g); cudaGraphInstantiate(&ge, g, 0);
        std::vector<double> t;
        for (int r = 0; r < 9; r++) { auto t0 = std::chrono::steady_clock::now(); cudaGraphLaunch(ge, st); cudaStreamSynchronize(st); t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); }
        std::sort(t.begin(), t.end());
        printf("%-58s %6.2f us per kernel\n", name, t[4] / N * 1e6);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    };
    run("1 block x 32 threads, trivial", [&] { k_tiny<<<1, 32, 0, st>>>(a); });
    run("1 block x 256 threads, one barrier", [&] { k_sync<<<1, 256, 0, st>>>(a); });
    run("1 block x 1024 threads, trivial", [&] { k_tiny<<<1, 1024, 0, st>>>(a); });
    run("48 blocks x 256 threads, trivial", [&] { k_tiny<<<48, 256, 0, st>>>(a); });
    run("538 blocks x 512 threads, trivial", [&] { k_tiny<<<538, 512, 0, st>>>(a); });
    run("2304 blocks x 256 threads, trivial", [&] { k_tiny<<<2304, 256, 0, st>>>(a); });
    run("12504 blocks x 512 threads, trivial", [&] { k_tiny<<<12504, 512, 0, st>>>(a); });
    for (int n : {1, 4, 16, 64}) {
        char nm[96]; snprintf(nm, 96, "538 blocks x 512 threads, %d float loads/thread (%.1f MB)", n, 538.0 * 512 * n * 4 / 1e6);
        run(nm, [&] { k_loads<<<538, 512, 0, st>>>(w, a, n); });
    }
    for (int n : {1, 16, 64}) {
        char nm[96]; snprintf(nm, 96, "1 block x 256 threads, %d float loads/thread", n);
        run(nm, [&] { k_loads<<<1, 256, 0, st>>>(w, a, n); });
    }
    return 0;
}
