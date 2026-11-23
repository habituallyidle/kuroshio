// Empirical check of the Blackwell block-scaled FP4 tensor-core instruction (m16n8k64, NVFP4).
// D[m][n] = sum_k A[m][k]*sfa[m][k/16] * B[k][n]*sfb[n][k/16]   (A: 16x64, B: 64x8, 4-bit E2M1; scales UE4M3)
// We pack operands according to the layout derived from CUTLASS's traits and compare with host math.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#ifndef ORDER
#define ORDER 0
#endif
#if ORDER == 0
#define MMA "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
#else
#define MMA "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
#endif

__global__ void k_probe(const uint8_t* A /*[16][64] codes*/, const uint8_t* B /*[64][8]*/, const uint8_t* sfa /*[16][4]*/,
                        const uint8_t* sfb /*[8][4]*/, float* D /*[16][8]*/) {
    int t = threadIdx.x, i = t & 3, j = t >> 2;
    uint32_t a[4] = {0, 0, 0, 0}, b[2] = {0, 0};
    for (int v = 0; v < 8; v++) {
        a[0] |= (uint32_t)A[j * 64 + 8 * i + v] << (4 * v);
        a[1] |= (uint32_t)A[(j + 8) * 64 + 8 * i + v] << (4 * v);
        a[2] |= (uint32_t)A[j * 64 + 32 + 8 * i + v] << (4 * v);
        a[3] |= (uint32_t)A[(j + 8) * 64 + 32 + 8 * i + v] << (4 * v);
        b[0] |= (uint32_t)B[(8 * i + v) * 8 + j] << (4 * v);
        b[1] |= (uint32_t)B[(32 + 8 * i + v) * 8 + j] << (4 * v);
    }
    int ma = j + 8 * (t & 1);
    uint32_t sa = 0, sb = 0;
    for (int q = 0; q < 4; q++) { sa |= (uint32_t)sfa[ma * 4 + q] << (8 * q); sb |= (uint32_t)sfb[j * 4 + q] << (8 * q); }
    float d0, d1, d2, d3, c = 0.0f;
    uint16_t z = 0;
    asm volatile(MMA
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(c), "f"(c), "f"(c), "f"(c),
          "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
    D[j * 8 + 2 * i] = d0; D[j * 8 + 2 * i + 1] = d1; D[(j + 8) * 8 + 2 * i] = d2; D[(j + 8) * 8 + 2 * i + 1] = d3;
}

static float e2m1(int c) { static const float t[8] = {0, .5f, 1, 1.5f, 2, 3, 4, 6}; return (c & 8) ? -t[c & 7] : t[c & 7]; }
static float ue4m3(int c) { int m = c & 7, e = (c >> 3) & 15; return e ? (8 + m) * ldexpf(1, e - 10) : m * ldexpf(1, -9); }

int main(int argc, char** argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    uint8_t A[16 * 64], B[64 * 8], sfa[64], sfb[32]; float D[128], ref[128];
    srand(7);
    for (auto& x : A) x = rand() & 15; for (auto& x : B) x = rand() & 15;
    for (auto& x : sfa) x = mode == 0 ? 0x38 : 0x30 + (rand() % 24);     // 0x38 = 1.0
    for (auto& x : sfb) x = mode == 0 ? 0x38 : 0x30 + (rand() % 24);
    for (int m = 0; m < 16; m++) for (int n = 0; n < 8; n++) {
        double s = 0; for (int k = 0; k < 64; k++) s += (double)e2m1(A[m * 64 + k]) * ue4m3(sfa[m * 4 + k / 16]) * e2m1(B[k * 8 + n]) * ue4m3(sfb[n * 4 + k / 16]);
        ref[m * 8 + n] = (float)s;
    }
    uint8_t *dA, *dB, *dsa, *dsb; float* dD;
    cudaMalloc(&dA, sizeof A); cudaMalloc(&dB, sizeof B); cudaMalloc(&dsa, 64); cudaMalloc(&dsb, 32); cudaMalloc(&dD, sizeof D);
    cudaMemcpy(dA, A, sizeof A, cudaMemcpyHostToDevice); cudaMemcpy(dB, B, sizeof B, cudaMemcpyHostToDevice);
    cudaMemcpy(dsa, sfa, 64, cudaMemcpyHostToDevice); cudaMemcpy(dsb, sfb, 32, cudaMemcpyHostToDevice);
    k_probe<<<1, 32>>>(dA, dB, dsa, dsb, dD);
    cudaError_t e = cudaDeviceSynchronize(); if (e) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 1; }
    cudaMemcpy(D, dD, sizeof D, cudaMemcpyDeviceToHost);
    double worst = 0; int bad = 0;
    for (int q = 0; q < 128; q++) { double d = fabs(D[q] - ref[q]); if (d > 1e-3 * (1 + fabs(ref[q]))) bad++; if (d > worst) worst = d; }
    printf("mode %d (%s scales): %d/128 mismatches, max abs diff %.4g; sample got %.3f %.3f %.3f want %.3f %.3f %.3f\n", mode,
           mode ? "random" : "unit", bad, worst, D[0], D[1], D[77], ref[0], ref[1], ref[77]);
    return bad != 0;
}
