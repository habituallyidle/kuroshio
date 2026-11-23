// Measures what the 5090 can really do: memory bandwidth (decode is bound by this)
// and FP16 matmul throughput (prefill is bound by this).
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#define CK(x) do { auto e_ = (x); if (e_ != 0) { fprintf(stderr, "CUDA error %d at %s:%d\n", (int)e_, __FILE__, __LINE__); exit(1); } } while (0)
using clk = std::chrono::steady_clock;
static double secs(clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); }

// Read-only streaming pass: every thread sums a strided slice. This is the access
// pattern of a decode GEMV (read weights once, write almost nothing).
__global__ void read_sum(const unsigned long long* p, size_t n, unsigned long long* out) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    unsigned long long s = 0;
    for (size_t i = tid; i < n; i += stride) s += p[i];
    if (s == 0xdeadbeefULL) out[0] = s;  // keep the loads alive without contended writes
}

int main() {
    size_t freeb, totalb; CK(cudaMemGetInfo(&freeb, &totalb));
    printf("VRAM free %.1f / %.1f GiB\n", freeb / 1073741824.0, totalb / 1073741824.0);
    const size_t GB = 1ull << 30, N = 8 * GB;
    void *a, *b; CK(cudaMalloc(&a, N)); CK(cudaMalloc(&b, N));
    CK(cudaMemset(a, 1, N)); CK(cudaMemset(b, 2, N)); CK(cudaDeviceSynchronize());
    std::vector<double> t;
    for (int i = 0; i < 7; i++) { auto s = clk::now(); CK(cudaMemcpy(b, a, N, cudaMemcpyDeviceToDevice)); CK(cudaDeviceSynchronize()); t.push_back(secs(s)); }
    std::sort(t.begin(), t.end());
    printf("copy 8GiB: median %.1f GB/s copied (=%.1f GB/s read+write traffic), best %.1f\n", N / 1e9 / t[3], 2 * N / 1e9 / t[3], N / 1e9 / t[0]);
    unsigned long long* out; CK(cudaMalloc(&out, 8));
    for (int threads : {1 << 16, 1 << 18, 1 << 20, 1 << 22}) {
        t.clear();
        for (int i = 0; i < 7; i++) { auto s = clk::now(); read_sum<<<threads / 256, 256>>>((unsigned long long*)a, N / 8, out); CK(cudaDeviceSynchronize()); t.push_back(secs(s)); }
        std::sort(t.begin(), t.end());
        printf("read-only 8GiB, %8d threads: median %.1f GB/s, best %.1f\n", threads, N / 1e9 / t[3], N / 1e9 / t[0]);
    }
    CK(cudaFree(a)); CK(cudaFree(b));
    cublasHandle_t h; CK(cublasCreate(&h));
    for (int n : {4096, 8192, 16384}) {
        __half *A, *B, *C; size_t bytes = (size_t)n * n * sizeof(__half);
        CK(cudaMalloc(&A, bytes)); CK(cudaMalloc(&B, bytes)); CK(cudaMalloc(&C, bytes));
        CK(cudaMemset(A, 0, bytes)); CK(cudaMemset(B, 0, bytes));
        __half one = __float2half(1.f), zero = __float2half(0.f);
        t.clear();
        for (int i = 0; i < 6; i++) {
            auto s = clk::now();
            CK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, A, CUDA_R_16F, n, B, CUDA_R_16F, n, &zero, C, CUDA_R_16F, n, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            CK(cudaDeviceSynchronize()); if (i) t.push_back(secs(s));
        }
        std::sort(t.begin(), t.end());
        printf("FP16 GEMM %5d^3: median %.1f TFLOPS, best %.1f\n", n, 2.0 * n * n * n / 1e12 / t[2], 2.0 * n * n * n / 1e12 / t[0]);
        CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C));
    }
    return 0;
}
