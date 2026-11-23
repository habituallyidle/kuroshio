// Can idle memory time be put to work? The 5090 has a 96 MB L2 cache. While a small compute-only kernel runs,
// a second kernel on a parallel graph branch touches the weights the NEXT kernel will read.
//   A: [glue ~G us] -> [read 39.7 MB cold]                       (what the engine does today)
//   B: same, with a prefetch of that region running alongside the glue kernel
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include <algorithm>
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA error %s line %d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
__global__ void k_read(const uint4* w, float* a, int per_thread) {
    size_t base = ((size_t)blockIdx.x * blockDim.x + threadIdx.x), nt = (size_t)gridDim.x * blockDim.x; unsigned acc = 0;
    for (int i = 0; i < per_thread; i++) acc += w[base + (size_t)i * nt].x;
    if (threadIdx.x == 0) a[blockIdx.x] = (float)acc;
}
__global__ void k_touch(const unsigned* w, size_t bytes, int stride) {        // one 4-byte load per `stride` bytes
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i < bytes) { volatile unsigned v = w[i / 4]; (void)v; }
}
__global__ void k_glue(float* a, int iters) {                                 // compute only, no global memory traffic
    float x = (float)threadIdx.x * 1e-3f; for (int i = 0; i < iters; i++) x = sinf(x) + 1.0f;
    if (threadIdx.x == 0) a[1 << 16] = x;
}
// Same glue kernel, but its extra blocks touch memory: blocks of ONE kernel are guaranteed to run side by side.
__global__ void k_glue_pf(float* a, int iters, const unsigned* w, size_t bytes) {
    if (blockIdx.x > 0) { size_t i = ((size_t)(blockIdx.x - 1) * blockDim.x + threadIdx.x) * 32; if (i < bytes) { volatile unsigned v = w[i / 4]; (void)v; } return; }
    float x = (float)threadIdx.x * 1e-3f; for (int i = 0; i < iters; i++) x = sinf(x) + 1.0f;
    if (threadIdx.x == 0) a[1 << 16] = x;
}
int main() {
    cudaStream_t A, B; CK(cudaStreamCreate(&A)); CK(cudaStreamCreate(&B)); cudaEvent_t ev; CK(cudaEventCreate(&ev));
    size_t GB = (size_t)1 << 30, total = 3 * GB, stride_reg = (size_t)40 << 20; uint4* w; float* a;
    CK(cudaMalloc(&w, total)); CK(cudaMalloc(&a, 1 << 20)); CK(cudaMemset(w, 1, total));
    const int N = 70; const size_t bytes = (size_t)538 * 512 * 9 * 16;
    auto run = [&](const char* name, int glue_iters, int mode, int tstride) -> int {
        cudaGraph_t g; cudaGraphExec_t ge;
        CK(cudaStreamBeginCapture(A, cudaStreamCaptureModeGlobal));
        for (int i = 0; i < N; i++) {
            const char* reg = (const char*)w + (size_t)i * stride_reg;
            if (mode == 1 || mode == 3) {                                             // fork: B may start now, runs beside the glue kernel
                CK(cudaEventRecord(ev, A)); CK(cudaStreamWaitEvent(B, ev));
                size_t nthreads = bytes / tstride;
                k_touch<<<(unsigned)((nthreads + 1023) / 1024), 1024, 0, B>>>((const unsigned*)reg, bytes, tstride);
            }
            if (mode == 4 || mode == 5) { size_t pb = (size_t)(bytes * (tstride / 100.0)); k_glue_pf<<<1 + (unsigned)((pb / 32 + 767) / 768), 768, 0, A>>>(a, glue_iters, (const unsigned*)reg, pb); }
            else if (glue_iters) k_glue<<<1, 768, 0, A>>>(a, glue_iters);
            if (mode != 2 && mode != 3 && mode != 5) k_read<<<538, 512, 0, A>>>((const uint4*)reg, a, 9);
        }
        if (mode == 1 || mode == 3) { CK(cudaEventRecord(ev, B)); CK(cudaStreamWaitEvent(A, ev)); }
        CK(cudaStreamEndCapture(A, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
        std::vector<double> t;
        for (int r = 0; r < 7; r++) { auto t0 = std::chrono::steady_clock::now(); CK(cudaGraphLaunch(ge, A)); CK(cudaStreamSynchronize(A)); t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count()); }
        std::sort(t.begin(), t.end());
        printf("%-64s %6.2f us per iteration\n", name, t[3] / N * 1e6);
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g); return 0;
    };
    for (int gi : {400}) {
        char nm[128];
        snprintf(nm, 128, "glue only (%d iterations)", gi); if (run(nm, gi, 2, 0)) return 1;
        snprintf(nm, 128, "A: glue(%d) -> cold read 39.7 MB", gi); if (run(nm, gi, 0, 0)) return 1;
        snprintf(nm, 128, "C: glue with parallel prefetch (32 B), no read"); if (run(nm, gi, 3, 32)) return 1;
        for (int ts : {32}) { snprintf(nm, 128, "B: + parallel prefetch, one touch per %d B", ts); if (run(nm, gi, 1, ts)) return 1; }
    }
    for (int gi : {60, 150, 400}) for (int pct : {25, 50, 100}) {
        char nm[128]; snprintf(nm, 128, "D: glue(%d) with in-kernel prefetch of %d%%, no read", gi, pct); if (run(nm, gi, 5, pct)) return 1;
        snprintf(nm, 128, "E: glue(%d) with in-kernel prefetch of %d%% -> read", gi, pct); if (run(nm, gi, 4, pct)) return 1;
    }
    for (int gi : {60, 150}) { char nm[128]; snprintf(nm, 128, "glue only (%d)", gi); if (run(nm, gi, 2, 0)) return 1; snprintf(nm, 128, "A: glue(%d) -> cold read", gi); if (run(nm, gi, 0, 0)) return 1; }
    if (run("read only, cold", 0, 0, 0)) return 1;
    if (run("prefetch only (32 B), then read (no glue)", 0, 1, 32)) return 1;
    return 0;
}
