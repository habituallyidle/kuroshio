// Layout probes for two more tensor-core instructions used by the engine:
//   mode 0: bf16 x bf16 -> f32, m16n8k16
//   mode 1: MXFP8: e4m3 x e4m3 with one power-of-two (ue8m0) scale per 32 elements, m16n8k32
// D[m][n] = sum_k A[m][k] * B[k][n] (times the scales). Compared with host arithmetic.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

__global__ void k_bf16(const uint16_t* A /*[16][16]*/, const uint16_t* B /*[16][8]*/, float* D) {
    int t = threadIdx.x, q = t & 3, g = t >> 2;
    auto pk = [](uint16_t lo, uint16_t hi) { return (uint32_t)lo | ((uint32_t)hi << 16); };
    uint32_t a0 = pk(A[g * 16 + 2 * q], A[g * 16 + 2 * q + 1]), a1 = pk(A[(g + 8) * 16 + 2 * q], A[(g + 8) * 16 + 2 * q + 1]);
    uint32_t a2 = pk(A[g * 16 + 8 + 2 * q], A[g * 16 + 9 + 2 * q]), a3 = pk(A[(g + 8) * 16 + 8 + 2 * q], A[(g + 8) * 16 + 9 + 2 * q]);
    uint32_t b0 = pk(B[(2 * q) * 8 + g], B[(2 * q + 1) * 8 + g]), b1 = pk(B[(8 + 2 * q) * 8 + g], B[(9 + 2 * q) * 8 + g]);
    float d0, d1, d2, d3, c = 0;
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c), "f"(c), "f"(c), "f"(c));
    D[g * 8 + 2 * q] = d0; D[g * 8 + 2 * q + 1] = d1; D[(g + 8) * 8 + 2 * q] = d2; D[(g + 8) * 8 + 2 * q + 1] = d3;
}

__global__ void k_fp8(const uint8_t* A /*[16][32]*/, const uint8_t* B /*[32][8]*/, const uint8_t* sfa /*[16]*/, const uint8_t* sfb /*[8]*/, float* D) {
    int t = threadIdx.x, i = t & 3, j = t >> 2;
    uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
    for (int v = 0; v < 4; v++) {
        a[0] |= (uint32_t)A[j * 32 + 4 * i + v] << (8 * v); a[1] |= (uint32_t)A[(j + 8) * 32 + 4 * i + v] << (8 * v);
        a[2] |= (uint32_t)A[j * 32 + 16 + 4 * i + v] << (8 * v); a[3] |= (uint32_t)A[(j + 8) * 32 + 16 + 4 * i + v] << (8 * v);
        b[0] |= (uint32_t)B[(4 * i + v) * 8 + j] << (8 * v); b[1] |= (uint32_t)B[(16 + 4 * i + v) * 8 + j] << (8 * v);
    }
    uint32_t sa = sfa[j + 8 * (t & 1)], sb = sfb[j];
    float d0, d1, d2, d3, c = 0; uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c), "f"(c), "f"(c), "f"(c), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
    D[j * 8 + 2 * i] = d0; D[j * 8 + 2 * i + 1] = d1; D[(j + 8) * 8 + 2 * i] = d2; D[(j + 8) * 8 + 2 * i + 1] = d3;
}

static float e4m3(int c) { int m = c & 7, e = (c >> 3) & 15; float v = e ? (8 + m) * ldexpf(1, e - 10) : m * ldexpf(1, -9); return (c & 0x80) ? -v : v; }
static float bf(uint16_t w) { uint32_t u = (uint32_t)w << 16; float f; memcpy(&f, &u, 4); return f; }
template <class T> T* up(const T* h, size_t n) { T* d; cudaMalloc(&d, n * sizeof(T)); cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice); return d; }

int report(const char* name, const float* D, const float* ref) {
    double worst = 0; int bad = 0;
    for (int q = 0; q < 128; q++) { double d = fabs(D[q] - ref[q]); if (d > 1e-3 * (1 + fabs(ref[q]))) bad++; if (d > worst) worst = d; }
    printf("%s: %d/128 mismatches, max abs diff %.4g (sample got %.4f want %.4f)\n", name, bad, worst, D[77], ref[77]);
    return bad;
}

int main() {
    srand(11); float D[128], ref[128]; float* dD; cudaMalloc(&dD, sizeof D); int bad = 0;
    {
        uint16_t A[256], B[128];
        for (auto& x : A) { float f = (rand() % 2001 - 1000) / 256.0f; uint32_t u; memcpy(&u, &f, 4); x = u >> 16; }
        for (auto& x : B) { float f = (rand() % 2001 - 1000) / 512.0f; uint32_t u; memcpy(&u, &f, 4); x = u >> 16; }
        for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) { double s = 0; for (int k = 0; k < 16; k++) s += (double)bf(A[m * 16 + k]) * bf(B[k * 8 + n]); ref[m * 8 + n] = (float)s; }
        k_bf16<<<1, 32>>>(up(A, 256), up(B, 128), dD);
        if (cudaDeviceSynchronize()) { printf("bf16: CUDA error\n"); return 1; }
        cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost); bad += report("bf16 m16n8k16", D, ref);
    }
    {
        uint8_t A[512], B[256], sa[16], sb[8];
        for (auto& x : A) { x = rand() & 0xFF; if ((x & 0x7F) > 0x7E) x = 0x30; }
        for (auto& x : B) { x = rand() & 0xFF; if ((x & 0x7F) > 0x7E) x = 0x30; }
        for (auto& x : sa) x = 120 + rand() % 12; for (auto& x : sb) x = 122 + rand() % 10;
        for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) { double s = 0; for (int k = 0; k < 32; k++) s += (double)e4m3(A[m * 32 + k]) * e4m3(B[k * 8 + n]); ref[m * 8 + n] = (float)(s * ldexp(1.0, sa[m] - 127) * ldexp(1.0, sb[n] - 127)); }
        k_fp8<<<1, 32>>>(up(A, 512), up(B, 256), up(sa, 16), up(sb, 8), dD);
        if (cudaDeviceSynchronize()) { printf("fp8: CUDA error\n"); return 1; }
        cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost); bad += report("mxfp8 m16n8k32", D, ref);
    }
    return bad != 0;
}
