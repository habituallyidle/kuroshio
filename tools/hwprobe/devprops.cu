// What this GPU offers, plus probes of two Blackwell-only conversion instructions (float -> FP4 / FP8 pairs).
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
__global__ void k_cvt(const float* in, uint32_t* out) {
    float a = in[2 * threadIdx.x], b = in[2 * threadIdx.x + 1]; uint32_t r4, r8;
    asm volatile("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u32.u8 %0, t;\n}" : "=r"(r4) : "f"(a), "f"(b));
    asm volatile("{\n .reg .b16 t;\n cvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\n cvt.u32.u16 %0, t;\n}" : "=r"(r8) : "f"(a), "f"(b));
    out[2 * threadIdx.x] = r4; out[2 * threadIdx.x + 1] = r8;
}
int main() {
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("%s: %d SMs, compute %d.%d, L2 cache %.0f MB, shared mem/SM %zu KB (per block max %zu KB, opt-in %zu KB), max threads/SM %d, regs/SM %d\n",
           p.name, p.multiProcessorCount, p.major, p.minor, p.l2CacheSize / 1048576.0, p.sharedMemPerMultiprocessor / 1024, p.sharedMemPerBlock / 1024, p.sharedMemPerBlockOptin / 1024,
           p.maxThreadsPerMultiProcessor, p.regsPerMultiprocessor);
    const int N = 12; float in[2 * N] = {0.26f, -0.74f, 1.25f, 1.26f, 2.5f, 2.51f, 3.5f, -5.0f, 5.01f, 7.9f, 0.0f, -0.24f, 0.75f, 1.75f, 100.f, -100.f, 0.1f, 449.f, 0.0625f, 0.07f, 3.3f, -17.0f, 1e-3f, 448.f};
    float* di; uint32_t* dout; uint32_t out[2 * N];
    cudaMalloc(&di, sizeof in); cudaMalloc(&dout, sizeof out); cudaMemcpy(di, in, sizeof in, cudaMemcpyHostToDevice);
    k_cvt<<<1, N>>>(di, dout); if (cudaDeviceSynchronize()) { printf("cvt probe: CUDA error\n"); return 1; }
    cudaMemcpy(out, dout, sizeof out, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) printf("  (%g, %g) -> e2m1x2 byte 0x%02x   e4m3x2 word 0x%04x\n", in[2 * i], in[2 * i + 1], out[2 * i], out[2 * i + 1]);
    return 0;
}
