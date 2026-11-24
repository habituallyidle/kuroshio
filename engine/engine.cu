// Inkling 2 Mini engine, v0: correctness first.
//
// One kernel per mathematical operation, no fusion, no tuning. The goal of this version is to
// reproduce ref/inkling_ref.py; speed work starts from measurements of this baseline.
// Section numbers refer to docs/ARCHITECTURE.md.
//
// Conventions: big weight matrices stay in their checkpoint format on the GPU (bf16, packed
// 4-bit, fp8). All activations are fp32 buffers; where the model rounds to bf16 we round the
// fp32 value in place (bf16r). T = number of consecutive tokens of one sequence in this call.
#include <cuda_runtime.h>
#include <cuda.h>              // driver API: virtual memory management (address ranges mapped on demand)
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)

namespace {
cudaStream_t g_stream = nullptr;
const int D = 2304, H = 24, HKV = 8, DH = 128, DR = 16, EXT = 1024, NE = 128, TOPK = 8, I2 = 1152;
const int V0 = 200058, NVIS = 48, TMAX = 512;               // TMAX: most rows of one step (a prompt batch)
const int HMAX = 128;                                     // most rows of a step that uses the output head (decoding); also the size of an attention
                                                          // sub-batch: a bigger step runs its attention in halves (see forward_body)
const int QKVR = 8600;                                   // rows of the fused q|r|gate|sink|k|v projection
const int R_Q = 0, R_R = 3072, R_GATE = 3456, R_SINK = 6528, R_K = 6552, R_V = 7576;
const int RING = 1152;                                   // local KV ring: window 1024 + batch headroom
const int DBATCH = 4;                                     // conversations the drafter handles in one pass
const int MAXSEQ = 16;                                    // most conversations (slots) one engine can hold
const int GSUB = 16;                                      // tokens per launch in full-attention layers (bounds the scratch: GSUB * 24 * max_ctx floats)
const int CR = 40;                                        // conv-input history kept between calls, indexed by position mod CR. Lags inside a batch are
                                                          // read from the batch itself. Runs of up to CR - 3 tokens store every token in the kernel that
                                                          // computes it (no slot is read and written in one launch); longer runs store their last CR - 3
                                                          // tokens with a separate kernel afterwards. So a step can be rewound into (drafts rejected) only
                                                          // if its run was at most CR - 3 rows: the 3 positions before the restart point are then intact.
// One row of a step: a token of conversation `seq` at absolute position `pos`. Consecutive rows of the same
// conversation form a run (run0 = index of its first row, runlen = its length); only within a run may a row take
// its convolution history from the batch itself.
struct Row { int pos, seq, run0, runlen; };
const float EPS = 1e-6f, QK_CLIP = 22.627416997969522f, REL_CLIP = 8.0f;

// An MXFP8 copy of a bf16 row-major matrix, in tile layout. Does not free the source.
struct Fp8Mat { uint8_t* w = nullptr; uint8_t* s = nullptr; int R = 0, C = 0; };

// ------------------------------------------------------------------ device helpers
__device__ __forceinline__ float bf16r(float x) {       // round fp32 to bf16 (nearest even), keep as fp32
    uint32_t u = __float_as_uint(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return __uint_as_float(u & 0xFFFF0000u);
}
__device__ __forceinline__ float bf16w(uint16_t w) { return __uint_as_float((uint32_t)w << 16); }
__device__ __forceinline__ uint16_t bf16bits(float x) { return (uint16_t)(__float_as_uint(bf16r(x)) >> 16); }
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float e4m3(uint8_t c) {      // FP8 E4M3 code -> value
    int m = c & 7, e = (c >> 3) & 15;
    float mag = e ? scalbnf((float)(8 + m), e - 10) : scalbnf((float)m, -9);
    return (c & 0x80) ? -mag : mag;
}
__device__ __forceinline__ float e4m3_rn(float x) {     // round x in [0, 448] to the nearest E4M3 value
    if (x <= 0.0f) return 0.0f;
    float r;
    if (x < 0.015625f) r = rintf(x * 512.0f) * (1.0f / 512.0f);
    else { int e = ilogbf(x); r = scalbnf(rintf(scalbnf(x, 3 - e)), e - 3); }
    return fminf(r, 448.0f);
}
__device__ __forceinline__ float e2m1(int code) {       // FP4 E2M1 code -> value
    const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = t[code & 7];
    return (code & 8) ? -v : v;
}
__device__ __forceinline__ float e2m1_rn(float v) {     // round to nearest E2M1 value, saturating (6.2)
    float a = fabsf(v), m;
    if (a <= 0.25f) m = 0.0f; else if (a < 0.75f) m = 0.5f; else if (a <= 1.25f) m = 1.0f; else if (a < 1.75f) m = 1.5f;
    else if (a <= 2.5f) m = 2.0f; else if (a < 3.5f) m = 3.0f; else if (a <= 5.0f) m = 4.0f; else m = 6.0f;
    return copysignf(m, v);
}

// ---- Key/value cache layout for tensor-core attention. Keys are grouped 16 slots at a time; for each KV head
// the group's 16 keys x 128 dims are stored as 8 tiles of 16 x 16 values, each tile in the per-thread register
// order of the bf16 matrix-multiply instruction (see tools/mmaprobe/probe2.cu):
//   K tile  (rows = keys, columns = dims): scores  = K . q
//   V tile  (rows = dims, columns = keys): outputs = V^T . P
// Byte offset of element (row r, column c), both 0..15, inside a 512-byte tile:
__device__ __forceinline__ size_t tile_elem(int r, int c) { return (size_t)(((r & 7) * 4 + ((c & 7) >> 1)) * 16 + ((r >> 3) + 2 * (c >> 3)) * 4 + (c & 1) * 2); }
__device__ __forceinline__ size_t ktile_off(size_t slot, int j, int d) { return (((slot >> 4) * HKV + j) * 8 + (d >> 4)) * 512 + tile_elem((int)(slot & 15), d & 15); }
__device__ __forceinline__ size_t vtile_off(size_t slot, int j, int d) { return (((slot >> 4) * HKV + j) * 8 + (d >> 4)) * 512 + tile_elem(d & 15, (int)(slot & 15)); }
// 8-bit key/value cache (INK_KV_FP8): the same tiles with one E4M3 byte per value (256-byte tiles, 8 bytes per thread),
// stored "unit scale" - keys are L2-clipped to a norm of 22.6 and values are RMS-normalised, so no scale factor is needed.
__device__ __forceinline__ size_t tile_elem8(int r, int c) { return (size_t)(((r & 7) * 4 + ((c & 7) >> 1)) * 8 + ((r >> 3) + 2 * (c >> 3)) * 2 + (c & 1)); }
__device__ __forceinline__ size_t ktile_off8(size_t slot, int j, int d) { return (((slot >> 4) * HKV + j) * 8 + (d >> 4)) * 256 + tile_elem8((int)(slot & 15), d & 15); }
__device__ __forceinline__ size_t vtile_off8(size_t slot, int j, int d) { return (((slot >> 4) * HKV + j) * 8 + (d >> 4)) * 256 + tile_elem8(d & 15, (int)(slot & 15)); }
__device__ __forceinline__ uint32_t e4m3_to_bf16(uint32_t c) {            // exact: E4M3 has 3 mantissa bits, bf16 has 7
    uint32_t e = (c >> 3) & 15u, m = c & 7u, sgn = (c & 0x80u) << 8;
    // subnormals (e == 0): m * 2^-9 -> {0, 2^-9, 2^-8, 1.5*2^-8, 2^-7, 1.25*2^-7, 1.5*2^-7, 1.75*2^-7} as bf16 bit patterns
    const uint64_t lo = 0x3BC03B803B000000ull, hi = 0x3C603C403C203C00ull;
    uint32_t sub = (uint32_t)(((m < 4 ? lo : hi) >> ((m & 3u) * 16u)) & 0xFFFFu);
    return sgn | (e ? ((e + 120u) << 7) | (m << 4) : sub);
}
// 8 E4M3 bytes (this thread's share of a tile) -> the four 32-bit registers of the bf16 multiply.
__device__ __forceinline__ uint4 tile8_to_bf16(uint64_t u) {
    uint4 w;
    w.x = e4m3_to_bf16((uint32_t)u & 255u) | (e4m3_to_bf16((uint32_t)(u >> 8) & 255u) << 16);
    w.y = e4m3_to_bf16((uint32_t)(u >> 16) & 255u) | (e4m3_to_bf16((uint32_t)(u >> 24) & 255u) << 16);
    w.z = e4m3_to_bf16((uint32_t)(u >> 32) & 255u) | (e4m3_to_bf16((uint32_t)(u >> 40) & 255u) << 16);
    w.w = e4m3_to_bf16((uint32_t)(u >> 48) & 255u) | (e4m3_to_bf16((uint32_t)(u >> 56) & 255u) << 16);
    return w;
}
__device__ __forceinline__ uint32_t e2m1_code(float v) { // exact E2M1 value -> 4-bit code
    float a = fabsf(v); uint32_t c = a <= 2.0f ? (uint32_t)(a * 2.0f) : (a < 3.5f ? 5u : (a < 5.0f ? 6u : 7u));
    return c | (signbit(v) ? 8u : 0u);
}
__device__ __forceinline__ uint8_t e4m3_code(float s) {  // exact non-negative E4M3 value -> code
    if (s < 0.015625f) return (uint8_t)(s * 512.0f);
    int e = ilogbf(s); int m = (int)((scalbnf(s, -e) - 1.0f) * 8.0f);
    return (uint8_t)(((e + 7) << 3) | m);
}
// Blackwell tensor cores: D[16x8] += A[16x64] * B[64x8] with 4-bit E2M1 operands and one E4M3 scale per
// 16 elements along k on both sides. Operand layout verified by tools/mmaprobe/probe.cu.
__device__ __forceinline__ void mma_fp4(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}
// Sum / max of one value per thread across the block. blockDim.x must be a power of two.
__device__ float bsum(float v, float* sh) {
    int t = threadIdx.x; sh[t] = v; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) sh[t] += sh[t + s]; __syncthreads(); }
    float r = sh[0]; __syncthreads(); return r;
}
__device__ float bmax(float v, float* sh) {
    int t = threadIdx.x; sh[t] = v; __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]); __syncthreads(); }
    float r = sh[0]; __syncthreads(); return r;
}

// ------------------------------------------------------------------ kernels
// out[t] = bf16(rmsnorm(embed[token[t]]) * w)
__global__ void k_embed(const int* tok, const uint16_t* embed, const float* w, float* out) {
    __shared__ float sh[256];
    int t = blockIdx.x; const uint16_t* row = embed + (size_t)tok[t] * D;
    float ss = 0; for (int c = threadIdx.x; c < D; c += 256) { float v = bf16w(row[c]); ss += v * v; }
    float inv = rsqrtf(bsum(ss, sh) / D + EPS);
    for (int c = threadIdx.x; c < D; c += 256) out[(size_t)t * D + c] = bf16r(bf16w(row[c]) * inv * w[c]);
}

// out[t] = bf16(rmsnorm(x[t]) * w)
__global__ void k_rms(const float* x, const float* w, float* out) {
    __shared__ float sh[256];
    const float* r = x + (size_t)blockIdx.x * D;
    float ss = 0; for (int c = threadIdx.x; c < D; c += 256) ss += r[c] * r[c];
    float inv = rsqrtf(bsum(ss, sh) / D + EPS);
    for (int c = threadIdx.x; c < D; c += 256) out[(size_t)blockIdx.x * D + c] = bf16r(r[c] * inv * w[c]);
}

// Block attention residual (4). One block per (token, query).
struct MixArgs { const float* src[6]; const float* add; int n, Q; const float* q; const float* norm; float* out; };



// Conv history of long runs: the last CR - 3 rows of every run longer than that (shorter runs store in-kernel).
__global__ void k_ring_put(const float* src, int sstride, int soff, float* ring, int C, const Row* rows) {
    int t = blockIdx.x; const Row rw = rows[t];
    if (rw.runlen <= CR - 3 || rw.run0 + rw.runlen - 1 - t >= CR - 3) return;
    float* dst = ring + ((size_t)rw.seq * CR + (size_t)(rw.pos % CR)) * C; const float* s = src + (size_t)t * sstride + soff;
    for (int c = threadIdx.x; c < C; c += blockDim.x) dst[c] = s[c];
}

// Static short convolution over the token axis (3.4). w[c][k], k = 0 is the oldest tap.
__global__ void k_sconv(const float* ring, const float* w, int C, const int* posp, float* out) {
    const int pos0 = *posp;
    int t = blockIdx.x, p = pos0 + t;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float acc = 0;
        for (int k = 0; k < 4; k++) { int s = p - 3 + k; if (s >= 0) acc = fmaf(w[c * 4 + k], ring[(size_t)(s % CR) * C + c], acc); }
        out[(size_t)t * C + c] = bf16r((acc + ring[(size_t)(p % CR) * C + c]) * 0.4472135954999579f);
    }
}




// Router (5.1): top-8 of sigmoid(logit) + bias, weights = 8 * sigmoid / sum(selected sigmoids).
__global__ void k_route(const float* logit, const float* rbias, int* ids, float* gamma) {
    __shared__ float sel[NE]; __shared__ float sg[TOPK]; __shared__ int id[TOPK];
    int t = blockIdx.x, e = threadIdx.x; float l = logit[(size_t)t * NE + e];
    sel[e] = sigmoidf(l) + rbias[e]; __syncthreads();
    int rank = 0; for (int j = 0; j < NE; j++) rank += (sel[j] > sel[e]) || (sel[j] == sel[e] && j < e);
    if (rank < TOPK) { sg[rank] = sigmoidf(l); id[rank] = e; }
    __syncthreads();
    if (e == 0) {
        float s = 0; for (int k = 0; k < TOPK; k++) s += sg[k];
        for (int k = 0; k < TOPK; k++) { ids[t * TOPK + k] = id[k]; gamma[t * TOPK + k] = sg[k] / s * 8.0f; }
    }
}







// stack_new = bf16((partial + attn_out) + y), or bf16(attn_out + y) when partial is null.
__global__ void k_resid(const float* partial, const float* attn_out, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    out[i] = bf16r(partial ? (partial[i] + attn_out[i]) + y[i] : attn_out[i] + y[i]);
}

// Final logit soft cap (in place) and argmax per token: (a) 1024-thread blocks each cover 4096 logits,
// (b) one small block per token picks the winner. Ties go to the lower token id.
const int HEAD_SPAN = 4096, HEAD_PARTS = (V0 + HEAD_SPAN - 1) / HEAD_SPAN;
__global__ void k_head_part(float* logits, float* pv, int* pi) {
    __shared__ float sv[32]; __shared__ int si[32];
    int t = blockIdx.y, part = blockIdx.x, tid = threadIdx.x; float* l = logits + (size_t)t * V0; float best = -FLT_MAX; int bi = 0x7fffffff;
    for (int k = 0; k < 4; k++) { int v = part * HEAD_SPAN + tid + 1024 * k; if (v >= V0) break; float x = 30.0f * tanhf(l[v] * (1.0f / 30.0f)); l[v] = x; if (x > best) { best = x; bi = v; } }
    for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, best, o); int oi = __shfl_down_sync(0xffffffffu, bi, o); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
    if ((tid & 31) == 0) { sv[tid >> 5] = best; si[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) { for (int w = 1; w < 32; w++) if (sv[w] > best || (sv[w] == best && si[w] < bi)) { best = sv[w]; bi = si[w]; } pv[t * HEAD_PARTS + part] = best; pi[t * HEAD_PARTS + part] = bi; }
}
__global__ void k_head_final(const float* pv, const int* pi, int* arg, float* rowmax) {
    int t = blockIdx.x, lane = threadIdx.x; float best = -FLT_MAX; int bi = 0x7fffffff;
    for (int k = lane; k < HEAD_PARTS; k += 32) { float x = pv[t * HEAD_PARTS + k]; int i = pi[t * HEAD_PARTS + k]; if (x > best || (x == best && i < bi)) { best = x; bi = i; } }
    for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, best, o); int oi = __shfl_down_sync(0xffffffffu, bi, o); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
    if (lane == 0) { arg[t] = bi; rowmax[t] = best; }
}

// ---- v1 kernels: one warp (32 threads) per matrix row, 8 rows per block. A row is only a few KB, so
// giving it a whole block (v0) spent most of the time on block scheduling instead of reading memory.
__device__ __forceinline__ float wsum(float v) {         // sum across the warp; lane 0 gets the total
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}
// FP4 E2M1 code -> 2 * value as a small integer (0,1,2,3,4,6,8,12), via a 32-bit lookup word.
__device__ __forceinline__ float e2m1x2(uint32_t code) {
    float v = (float)((0xC8643210u >> ((code & 7u) * 4u)) & 15u);
    return (code & 8u) ? -v : v;
}

// ---- L2 prefetch. The 5090 has a 96 MB L2 cache, and memory is idle while the small "glue" kernels compute.
// A glue kernel can be launched with extra blocks that do nothing but touch (one load per 32-byte cache
// sector) part of a matrix that a later kernel will read; blocks of one kernel run side by side, so this
// overlaps the memory traffic with the glue computation. (tools/hwprobe/kprefetch.cu has the measurements.)
struct PF { const uint8_t* p; unsigned bytes; int nwork; };          // region to touch; number of real work blocks (x)
__device__ __forceinline__ bool pf_block(const PF& pf, size_t extra_index) {
    size_t i = (extra_index * blockDim.x + threadIdx.x) * 32;
    if (i < pf.bytes) { volatile unsigned v = *(const unsigned*)(pf.p + i); (void)v; }
    return true;
}
#define PF_X(pf) if ((int)blockIdx.x >= (pf).nwork) { pf_block(pf, blockIdx.x - (pf).nwork); return; }

// "Wide" small kernels: NT threads per token, each owning DPT of the 2,304 dimensions, so that no thread
// does more than a handful of memory loads in a row (a thread pays ~0.1 microsecond per load).
const int NT = 768, DPT = 3;

// Reduce NV (<= 32) values per thread over the whole block: warp shuffles, then warp 0 adds the warp totals.
// On return every thread holds the block totals in v[]. sh: [warps * NV], sh2: [32].
template <int NV> __device__ __forceinline__ void breduce(float* v, float* sh, float* sh2) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int k = 0; k < NV; k++) { float x = wsum(v[k]); if (lane == 0) sh[warp * NV + k] = x; }
    __syncthreads();
    if (warp == 0 && lane < NV) { float a = 0; for (int w = 0; w < nw; w++) a += sh[w * NV + lane]; sh2[lane] = a; }
    __syncthreads();
    for (int k = 0; k < NV; k++) v[k] = sh2[k];
    __syncthreads();
}
#define MIX_SH __shared__ float sh[(NT / 32) * 30]; __shared__ float sh2[32]







// Group the (token, slot) pairs of this step by expert. A task = one expert + up to 8 columns that use it,
// so the expert's weights are read once for all tokens that routed to it. tasks[i] = {expert, ncols, col[8]}.
const int TASKW = 10;
__global__ void k_tasks(const int* ids, int N, int* tasks, int* ntasks) {
    // Per expert: which tokens picked it (a bit each: an expert is picked at most once per token) and as which of
    // their 8 picks. Filled by the threads in parallel, 8 picks each at most, so the cost does not grow with the
    // batch the way a scan of all picks by every expert's thread did (20 us at 128 tokens against 5 now, 48 times per step).
    // (slot: 4 bits per token, 8 tokens to a word; a byte each would not fit a block's 48 KB with 512 tokens)
    // (dynamic shared memory, TASKS_SHARED bytes: more than a kernel gets without asking)
    extern __shared__ unsigned int tsh[]; unsigned int (*mask)[TMAX / 32] = (unsigned int (*)[TMAX / 32])tsh; unsigned int (*slot)[TMAX / 8] = (unsigned int (*)[TMAX / 8])(tsh + NE * (TMAX / 32));
    __shared__ int wsum_[NE / 32];
    int e = threadIdx.x; const int MW = (N / TOPK + 31) / 32, SW = (N / TOPK + 7) / 8;      // words in use: a small step should not pay for 512 rows
    for (int w = 0; w < MW; w++) mask[e][w] = 0;
    for (int w = 0; w < SW; w++) slot[e][w] = 0;
    __syncthreads();
    for (int n = e; n < N; n += NE) {
        int id = ids[n], t = n / TOPK; if ((unsigned)id >= (unsigned)NE) continue;        // (NaN router scores can leave a pick unset)
        atomicOr(&slot[id][t >> 3], (unsigned)(n % TOPK) << (4 * (t & 7))); atomicOr(&mask[id][t >> 5], 1u << (t & 31));
    }
    __syncthreads();
    int cnt = 0; for (int w = 0; w < MW; w++) cnt += __popc(mask[e][w]);
    int mine = (cnt + 7) / 8;                              // tasks of this expert: 8 tokens each
    // Tasks are numbered by expert: a running sum over the 128 threads, inside each warp by shuffles.
    int incl = mine;
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, o); if ((e & 31) >= o) incl += v; }
    if ((e & 31) == 31) wsum_[e >> 5] = incl;
    __syncthreads();
    int rank = incl - mine; for (int w = 0; w < (e >> 5); w++) rank += wsum_[w];
    int k = 0;                                           // tokens in ascending order, as before
    for (int w = 0; w < MW; w++) {
        unsigned int m = mask[e][w];
        while (m) {
            int t = w * 32 + __ffs(m) - 1; m &= m - 1;
            int* tk = tasks + (rank + k / 8) * TASKW;
            if ((k & 7) == 0) { tk[0] = e; tk[1] = min(8, cnt - k); }
            tk[2 + (k & 7)] = t * TOPK + ((slot[e][t >> 3] >> (4 * (t & 7))) & 15); k++;
        }
    }
    if (e == NE - 1) *ntasks = rank + mine;
}

// Most tasks N picks can make: each of at most NE experts rounds its tokens up to a multiple of 8. The expert
// kernels' grids are this tall, not N: at 128 tokens that is 240 task rows instead of 1,024, mostly empty, each
// costing a block start.
static inline int max_tasks(int N) { return (N + 7 * std::min(NE, N)) / 8; }

const size_t TASKS_SHARED = (size_t)NE * (TMAX / 32 + TMAX / 8) * 4;

// Expert GEMM on the tensor cores. One warp = 16 weight rows (operand A); columns (operand B) = the tokens
// of this task. Grid: (R/128, max tasks). x_per_slot: the activation row is per (token, slot), else per token.
__global__ void k_expert4_mma(const uint8_t* W, const uint8_t* S, const float* wamax, float inv_den, const int* tasks, const int* ntasks,
                              const uint8_t* xcode, const uint8_t* xscale, const float* xamax, int x_per_slot, int R, int C, float* out, int nsplit) {
    __shared__ float sh[8][32][4];
    if ((int)blockIdx.y >= *ntasks) return;
    const int* task = tasks + blockIdx.y * TASKW; int e = task[0], ncols = task[1];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, i = lane & 3, j = lane >> 2, rgi = blockIdx.x * (8 / nsplit) + warp / nsplit, r0 = rgi * 16;
    int nblk = C / 16, KB = C / 64, kseg = KB / nsplit, kbase = (warp % nsplit) * kseg;     // nsplit warps share the columns of a row group
    const uint4* wt = (const uint4*)(W + (((size_t)e * (R / 16) + rgi) * KB) * 512 + lane * 16);              // this thread's 16 bytes of tile 0
    const uint32_t* st = (const uint32_t*)(S + (((size_t)e * (R / 16) + rgi) * KB) * 64 + (j * 2 + (lane & 1)) * 4);
    int nj = j < ncols ? task[2 + j] : -1, xj = nj < 0 ? 0 : (x_per_slot ? nj : nj / TOPK);
    const uint8_t* xc = xcode + (size_t)xj * (C / 2) + 4 * i; const uint8_t* xs = xscale + (size_t)xj * nblk;
    float d[4] = {0, 0, 0, 0};
    for (int it = 0; it < kseg; it += 3) {
        uint4 w[3]; uint32_t b0[3], b1[3], sa[3], sb[3];
        for (int u = 0; u < 3; u++) {
            int kb = kbase + it + u;
            w[u] = wt[(size_t)kb * 32]; sa[u] = st[(size_t)kb * 16]; b0[u] = b1[u] = sb[u] = 0;
            if (nj >= 0) { b0[u] = *(const uint32_t*)(xc + kb * 32); b1[u] = *(const uint32_t*)(xc + kb * 32 + 16); sb[u] = *(const uint32_t*)(xs + kb * 4); }
        }
        for (int u = 0; u < 3; u++) mma_fp4(d, w[u].x, w[u].y, w[u].z, w[u].w, b0[u], b1[u], sa[u], sb[u]);
    }
    for (int q = 0; q < 4; q++) sh[warp][lane][q] = d[q];
    __syncthreads();
    if (warp % nsplit) return;
    float ws = wamax[e] * (inv_den * (1.0f / 1536.0f));
    for (int q = 0; q < 2; q++) {                        // this lane holds columns 2i, 2i+1 of rows j and j+8
        int c = 2 * i + q; if (c >= ncols) continue;
        int n = task[2 + c]; float sc = xamax[x_per_slot ? n : n / TOPK] * ws, lo = 0, hi = 0;
        for (int w2 = 0; w2 < nsplit; w2++) { lo += sh[warp + w2][lane][q]; hi += sh[warp + w2][lane][2 + q]; }
        out[(size_t)n * R + r0 + j] = bf16r(lo * sc); out[(size_t)n * R + r0 + j + 8] = bf16r(hi * sc);
    }
}

// =====================================================================================================
// Experts for prompt-sized batches. With 512 rows an expert is picked about 36 times per layer visit, and the task
// list above cuts that into five tasks of 8 columns, each reading the expert's weights again: the weights go by
// five times, and every operation costs five loads (weights, their scales, two halves of the activations, their
// scales). Here a task is an expert with up to 64 of its columns (k_tasks_w), the activations of a task are
// gathered first into the operand layout, 16 bytes per thread and operation (k_xgather4), and a block reads each
// slice of weights once for all its column groups. The chain of operations behind every output value is the one
// k_expert4_mma performs, so results are the same bit for bit.
// =====================================================================================================
const int WCOLS = 64, TASKWW = 3 + WCOLS;                 // wide task: expert, columns, first column group, then the picks
static inline int max_wtasks(int N) { return (N + (WCOLS - 1) * std::min(NE, N)) / WCOLS; }
__global__ void k_tasks_w(const int* ids, int N, int* tasks, int* ntasks) {
    extern __shared__ unsigned int tsh[]; unsigned int (*mask)[TMAX / 32] = (unsigned int (*)[TMAX / 32])tsh; unsigned int (*slot)[TMAX / 8] = (unsigned int (*)[TMAX / 8])(tsh + NE * (TMAX / 32));
    __shared__ int wsum_[NE / 32][2];
    int e = threadIdx.x; const int MW = (N / TOPK + 31) / 32, SW = (N / TOPK + 7) / 8;      // words in use: a small step should not pay for 512 rows
    for (int w = 0; w < MW; w++) mask[e][w] = 0;
    for (int w = 0; w < SW; w++) slot[e][w] = 0;
    __syncthreads();
    for (int n = e; n < N; n += NE) {
        int id = ids[n], t = n / TOPK; if ((unsigned)id >= (unsigned)NE) continue;
        atomicOr(&slot[id][t >> 3], (unsigned)(n % TOPK) << (4 * (t & 7))); atomicOr(&mask[id][t >> 5], 1u << (t & 31));
    }
    __syncthreads();
    int cnt = 0; for (int w = 0; w < MW; w++) cnt += __popc(mask[e][w]);
    int mine = (cnt + WCOLS - 1) / WCOLS, mcg = (cnt + 7) / 8;      // this expert's tasks, and its column groups of 8
    int incl = mine, icg = mcg;
    for (int o = 1; o < 32; o <<= 1) { int v = __shfl_up_sync(0xffffffffu, incl, o), c = __shfl_up_sync(0xffffffffu, icg, o); if ((e & 31) >= o) { incl += v; icg += c; } }
    if ((e & 31) == 31) { wsum_[e >> 5][0] = incl; wsum_[e >> 5][1] = icg; }
    __syncthreads();
    int rank = incl - mine, cg0 = icg - mcg; for (int w = 0; w < (e >> 5); w++) { rank += wsum_[w][0]; cg0 += wsum_[w][1]; }
    int k = 0;
    for (int w = 0; w < MW; w++) {
        unsigned int m = mask[e][w];
        while (m) {
            int t = w * 32 + __ffs(m) - 1; m &= m - 1;
            int* tk = tasks + (rank + k / WCOLS) * TASKWW;
            if (k % WCOLS == 0) { tk[0] = e; tk[1] = min(WCOLS, cnt - k); tk[2] = cg0 + k / 8; }
            tk[3 + k % WCOLS] = t * TOPK + ((slot[e][t >> 3] >> (4 * (t & 7))) & 15); k++;
        }
    }
    if (e == NE - 1) *ntasks = rank + mine;
}
// The activations of every wide task as tensor-core operands: out[((column group) * KB + slice) * 32 + lane] =
// { the two 4-byte halves of the thread's operand, its 4 scale bytes, 0 }. Grid (8, wide tasks), 256 threads.
__global__ void k_xgather4(const int* tasks, const int* ntasks, const uint8_t* xcode, const uint8_t* xscale, int x_per_slot, int C, uint4* out) {
    if ((int)blockIdx.y >= *ntasks) return;
    const int* task = tasks + blockIdx.y * TASKWW; int ncols = task[1], cg = blockIdx.x; if (cg * 8 >= ncols) return;
    int KB = C / 64, nblk = C / 16; uint4* o = out + ((size_t)(task[2] + cg) * KB) * 32;
    for (int e = threadIdx.x; e < KB * 32; e += 256) {
        int kb = e >> 5, lane = e & 31, i = lane & 3, c = cg * 8 + (lane >> 2); uint4 v = make_uint4(0, 0, 0, 0);
        if (c < ncols) {
            int n = task[3 + c], xj = x_per_slot ? n : n / TOPK; const uint8_t* xc = xcode + (size_t)xj * (C / 2) + 4 * i + kb * 32;
            v = make_uint4(*(const uint32_t*)xc, *(const uint32_t*)(xc + 16), *(const uint32_t*)(xscale + (size_t)xj * nblk + kb * 4), 0);
        }
        o[e] = v;
    }
}
// One warp per 16 weight rows: it goes through the nsplit column segments in turn and adds their partial sums in
// order, which is what k_expert4_mma's nsplit warps and its exchange through shared memory compute. No block-wide
// synchronisation, and a warp does hundreds of operations instead of nine: the tensor cores can do one FP4 operation
// in 1.5 ns per core, and with nine operations per warp the kernel spent 24 ns on each, nearly all of it on starting
// and finishing blocks.
__global__ void k_expert4_wide(const uint8_t* W, const uint8_t* S, const float* wamax, float inv_den, const int* tasks, const int* ntasks,
                               const uint4* XB, const float* xamax, int x_per_slot, int R, int C, float* out, int nsplit) {
    if ((int)blockIdx.y >= *ntasks) return;
    const int* task = tasks + blockIdx.y * TASKWW; const int e = task[0], ncols = task[1], ncg = (ncols + 7) / 8;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, i = lane & 3, j = lane >> 2, rgi = blockIdx.x * (blockDim.x >> 5) + warp, r0 = rgi * 16;
    if (rgi >= R / 16) return;
    int KB = C / 64, kseg = KB / nsplit;
    const uint4* wt = (const uint4*)(W + (((size_t)e * (R / 16) + rgi) * KB) * 512 + lane * 16);
    const uint32_t* st = (const uint32_t*)(S + (((size_t)e * (R / 16) + rgi) * KB) * 64 + (j * 2 + (lane & 1)) * 4);
    const uint4* xb = XB + ((size_t)task[2] * KB) * 32 + lane;
    float tot[WCOLS / 8][4]; for (int cg = 0; cg < WCOLS / 8; cg++) for (int q = 0; q < 4; q++) tot[cg][q] = 0;
    for (int sgm = 0; sgm < nsplit; sgm++) {
        float d[WCOLS / 8][4]; for (int cg = 0; cg < WCOLS / 8; cg++) for (int q = 0; q < 4; q++) d[cg][q] = 0;
        for (int it = 0; it < kseg; it++) {
            int kb = sgm * kseg + it; uint4 w = wt[(size_t)kb * 32]; uint32_t sa = st[(size_t)kb * 16];
            #pragma unroll
            for (int cg = 0; cg < WCOLS / 8; cg++) if (cg < ncg) { uint4 b = xb[((size_t)cg * KB + kb) * 32]; mma_fp4(d[cg], w.x, w.y, w.z, w.w, b.x, b.y, sa, b.z); }   // (a fixed trip count keeps d in registers)
        }
        for (int cg = 0; cg < WCOLS / 8; cg++) for (int q = 0; q < 4; q++) tot[cg][q] += d[cg][q];
    }
    float ws = wamax[e] * (inv_den * (1.0f / 1536.0f));
    #pragma unroll
    for (int cg = 0; cg < WCOLS / 8; cg++) {
        if (cg >= ncg) break;
        for (int q = 0; q < 2; q++) {                    // this lane holds columns 2i, 2i+1 of the group, rows j and j+8
            int c = cg * 8 + 2 * i + q; if (c >= ncols) continue;
            int n = task[3 + c]; float sc = xamax[x_per_slot ? n : n / TOPK] * ws;
            out[(size_t)n * R + r0 + j] = bf16r(tot[cg][q] * sc); out[(size_t)n * R + r0 + j + 8] = bf16r(tot[cg][2 + q] * sc);
        }
    }
}

// Loop message controller (2.4), v2, split so that no thread does a long serial chain of memory loads
// (a GPU thread is slow at that: ~0.3 ms for 2,300 dependent-latency loads in v1).
// (a) pair = bf16([dest/rms(dest), msg/rms(msg)])
__global__ void k_pair(const float* dest, int dstride, const float* msg, int mstride, float* pair) {
    __shared__ float sh[(NT / 32) * 2]; __shared__ float sh2[32];
    int t = blockIdx.x, tid = threadIdx.x; const float* d = dest + (size_t)t * dstride; const float* m = msg + (size_t)t * mstride;
    float dv[DPT], mv[DPT], r[2] = {0, 0};
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; dv[k] = d[c]; mv[k] = m[c]; r[0] += dv[k] * dv[k]; r[1] += mv[k] * mv[k]; }
    breduce<2>(r, sh, sh2);
    float id = rsqrtf(r[0] / D + EPS), im = rsqrtf(r[1] / D + EPS);
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; pair[(size_t)t * 2 * D + c] = bf16r(dv[k] * id); pair[(size_t)t * 2 * D + D + c] = bf16r(mv[k] * im); }
}
// (c) x = (1 - tanh(forget.h)) * dest + tanh(gain.h) * msg; one warp per output element, 8 elements per block.
// The loop controller's gate: x = (1 - tanh(h . forget_c)) * dest + tanh(h . gain_c) * msg for every token and channel,
// h being the token's 64 controller values. One warp per (token, channel): the fastest form for a few tokens.
__global__ void k_gate(const float* h, const float* gain, const float* forget, const float* dest, int dstride, const float* msg, int mstride, float* x) {
    int t = blockIdx.y, c = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const float* hh = h + (size_t)t * 64; const float* g = gain + (size_t)c * 64; const float* f = forget + (size_t)c * 64;
    float ga = fmaf(hh[lane], g[lane], hh[lane + 32] * g[lane + 32]), fa = fmaf(hh[lane], f[lane], hh[lane + 32] * f[lane + 32]);
    ga = wsum(ga); fa = wsum(fa);
    if (lane == 0) { float a = (1.0f - tanhf(fa)) * dest[(size_t)t * dstride + c]; float b = tanhf(ga) * msg[(size_t)t * mstride + c]; x[(size_t)t * D + c] = a + b; }
}
// The same for big batches (GATE_BIG tokens or more), where a warp per (token, channel) reads the 4.7 MB of tables
// again for every token (93 us at 128 tokens, 24 times per step). One block per 8 channels stages h for all tokens
// and its 8 rows of gain and forget in shared memory; each thread then works through (token, channel) pairs alone,
// summing in the order the warp reduction above takes (pairs 32 apart, then a halving tree), so results are the same
// bit for bit. Slower than the warp form below about 40 tokens: few threads have work.
const int GATE_BIG = 48;
__global__ void k_gate_big(const float* h, const float* gain, const float* forget, const float* dest, int dstride, const float* msg, int mstride, float* x, int T) {
    __shared__ float sh[HMAX * 64]; __shared__ float sg[8 * 64]; __shared__ float sf[8 * 64];
    int c0 = blockIdx.x * 8, tid = threadIdx.x;
    for (int i = tid; i < 8 * 64; i += 256) { sg[i] = gain[(size_t)c0 * 64 + i]; sf[i] = forget[(size_t)c0 * 64 + i]; }
    for (int t0 = 0; t0 < T; t0 += HMAX) {                 // HMAX tokens at a time: what fits in shared memory
        int n = min(HMAX, T - t0);
        __syncthreads();
        for (int i = tid; i < n * 64; i += 256) sh[i] = h[(size_t)t0 * 64 + i];
        __syncthreads();
        for (int p = tid; p < n * 8; p += 256) {
            int t = t0 + (p >> 3), w = p & 7, c = c0 + w; const float* hh = sh + (p >> 3) * 64; const float* g = sg + w * 64; const float* f = sf + w * 64;
            float va[32], vf[32];
            for (int l = 0; l < 32; l++) { va[l] = fmaf(hh[l], g[l], hh[l + 32] * g[l + 32]); vf[l] = fmaf(hh[l], f[l], hh[l + 32] * f[l + 32]); }
            for (int o = 16; o > 0; o >>= 1) for (int l = 0; l < o; l++) { va[l] += va[l + o]; vf[l] += vf[l + o]; }
            float a = (1.0f - tanhf(vf[0])) * dest[(size_t)t * dstride + c]; float b = tanhf(va[0]) * msg[(size_t)t * mstride + c];
            x[(size_t)t * D + c] = a + b;
        }
    }
}
// out[t*ostride ..] = bf16(rmsnorm(x[t]) * w)
__global__ void k_rms_s(const float* x, const float* w, float* out, int ostride) {
    __shared__ float sh[NT / 32]; __shared__ float sh2[32];
    int tid = threadIdx.x; const float* r = x + (size_t)blockIdx.x * D;
    float v[DPT], ss = 0; for (int k = 0; k < DPT; k++) { v[k] = r[tid + NT * k]; ss += v[k] * v[k]; }
    breduce<1>(&ss, sh, sh2);
    float inv = rsqrtf(ss / D + EPS);
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; out[(size_t)blockIdx.x * ostride + c] = bf16r(v[k] * inv * w[c]); }
}

// =====================================================================================================
// v2 fused kernels. Every kernel costs a few microseconds of fixed latency however little it does, and a
// token needs ~1,500 of them in v1, so here adjacent small steps are merged into one launch each.
// =====================================================================================================

// Block attention residual (4) on values already held by the threads: xs[n][k] is source n at this thread's
// k-th dimension (c = tid + NT*k). Writes out[qi*D + c] for Q queries. The source and query counts are
// template parameters on purpose: with run-time loop bounds the compiler keeps these small arrays in
// (slow) per-thread memory instead of registers, which made this routine 4x slower.
template <int NS, int Q> __device__ __forceinline__ void mix_core(float xs[NS][DPT], const float* q, const float* norm, float* out, float* sh, float* sh2) {
    int tid = threadIdx.x;
    // One block-wide reduction for both the squared norms (first NS values) and the query dot products.
    float rd[NS + Q * NS]; for (int i = 0; i < NS + Q * NS; i++) rd[i] = 0;
    for (int a = 0; a < NS; a++) for (int k = 0; k < DPT; k++) rd[a] += xs[a][k] * xs[a][k];
    for (int qi = 0; qi < Q; qi++) for (int k = 0; k < DPT; k++) { float qv = q[(size_t)qi * D + tid + NT * k]; for (int a = 0; a < NS; a++) rd[NS + qi * NS + a] = fmaf(qv, xs[a][k], rd[NS + qi * NS + a]); }
    breduce<NS + Q * NS>(rd, sh, sh2);
    float inv[NS]; for (int a = 0; a < NS; a++) inv[a] = rsqrtf(rd[a] / D + EPS);
    const float* dt = rd + NS;
    float m[Q][DPT], ss[Q];
    for (int qi = 0; qi < Q; qi++) {
        float lg[NS], mx = -FLT_MAX, pw[NS], ps = 0; ss[qi] = 0;
        for (int a = 0; a < NS; a++) { lg[a] = dt[qi * NS + a] * inv[a]; mx = fmaxf(mx, lg[a]); }
        for (int a = 0; a < NS; a++) { pw[a] = expf(lg[a] - mx); ps += pw[a]; pw[a] *= inv[a]; }
        float inv_sum = 1.0f / ps;
        for (int k = 0; k < DPT; k++) { float acc = 0; for (int a = 0; a < NS; a++) acc = fmaf(pw[a], xs[a][k], acc); m[qi][k] = acc * inv_sum; ss[qi] += m[qi][k] * m[qi][k]; }
    }
    if (!norm) { for (int qi = 0; qi < Q; qi++) for (int k = 0; k < DPT; k++) out[(size_t)qi * D + tid + NT * k] = m[qi][k]; return; }
    breduce<Q>(ss, sh, sh2);
    for (int qi = 0; qi < Q; qi++) { float invn = rsqrtf(ss[qi] / D + EPS); for (int k = 0; k < DPT; k++) { int c = tid + NT * k; out[(size_t)qi * D + c] = bf16r(m[qi][k] * invn * norm[(size_t)qi * D + c]); } }
}

// Stand-alone mix: sources from memory (the last one optionally + add). One block per token.
template <int NS, int Q> __global__ void k_mixq(MixArgs a) {
    MIX_SH;
    int t = blockIdx.x, tid = threadIdx.x; float xs[NS][DPT];
    for (int k = 0; k < DPT; k++) { size_t o = (size_t)t * D + tid + NT * k; for (int n = 0; n < NS; n++) xs[n][k] = a.src[n][o] + ((n == NS - 1 && a.add) ? a.add[o] : 0.0f); }
    mix_core<NS, Q>(xs, a.q, a.norm, a.out + (size_t)t * Q * D, sh, sh2);
}

// After the attention output projection: conv over tokens -> attn_out, then the MoE-input mix.
struct PostAttn { PF pf; const float* attn_raw; float* ring; const float* w; const Row* rows; float* attn_out; const float* src[5]; int ns;
                  const float* partial; const float* q; const float* norm; float* out; };
template <int NS> __global__ void k_post_attn(PostAttn a) {
    MIX_SH;
    PF_X(a.pf)
    int t = blockIdx.x, tid = threadIdx.x; const Row rw = a.rows[t]; const int p = rw.pos; float* ring = a.ring + (size_t)rw.seq * CR * D; float xs[NS][DPT];
    for (int k = 0; k < DPT; k++) {
        int c = tid + NT * k; float cur = a.attn_raw[(size_t)t * D + c], acc = 0; const float4 w = *(const float4*)(a.w + c * 4);
        const float wk[3] = {w.x, w.y, w.z};
        for (int kk = 0; kk < 3; kk++) {                 // lags 3, 2, 1: from this batch if present, else from the ring
            int lag = 3 - kk; if (p - lag < 0) continue;
            float v = t - lag >= rw.run0 ? a.attn_raw[(size_t)(t - lag) * D + c] : ring[(size_t)((p - lag) % CR) * D + c];
            acc = fmaf(wk[kk], v, acc);
        }
        acc = fmaf(w.w, cur, acc);
        if (rw.runlen <= CR - 3) ring[(size_t)(p % CR) * D + c] = cur;
        float ao = bf16r((acc + cur) * 0.4472135954999579f);
        a.attn_out[(size_t)t * D + c] = ao;
        for (int n = 0; n < NS - 1; n++) xs[n][k] = a.src[n][(size_t)t * D + c];
        xs[NS - 1][k] = a.partial ? a.partial[(size_t)t * D + c] + ao : ao;
    }
    mix_core<NS, 1>(xs, a.q, a.norm, a.out + (size_t)t * D, sh, sh2);
}

// After the expert down projection: combine the 8 experts, conv over tokens, residual update, and the
// mix that produces the next visit's attention input (or the final hidden state).
struct PostMoe { PF pf; const float* down; float* ring; const float* w; const Row* rows; const float* partial; const float* attn_out; float* stack_out;
                 float* moe_out; const float* src[5]; int ns; const float* q; int Q; const float* norm; float* out; };
template <int NS, int Q> __global__ void k_post_moe(PostMoe a) {
    MIX_SH;
    PF_X(a.pf)
    int t = blockIdx.x, tid = threadIdx.x; const Row rw = a.rows[t]; const int p = rw.pos; float* ring = a.ring + (size_t)rw.seq * CR * D; float xs[NS][DPT];
    for (int k = 0; k < DPT; k++) {
        int c = tid + NT * k; float v[4]; const float4 w = *(const float4*)(a.w + c * 4); const float wk[4] = {w.x, w.y, w.z, w.w};
        for (int kk = 0; kk < 4; kk++) {                 // moe_out at lags 3, 2, 1, 0
            int lag = 3 - kk; v[kk] = 0;
            if (p - lag < 0) continue;
            if (t - lag >= rw.run0) { float s2 = 0; for (int j = 0; j < TOPK; j++) s2 += a.down[((size_t)(t - lag) * TOPK + j) * D + c]; v[kk] = bf16r(s2); }
            else v[kk] = ring[(size_t)((p - lag) % CR) * D + c];
        }
        float acc = 0; for (int kk = 0; kk < 4; kk++) if (p - 3 + kk >= 0) acc = fmaf(wk[kk], v[kk], acc);
        if (rw.runlen <= CR - 3) ring[(size_t)(p % CR) * D + c] = v[3];
        a.moe_out[(size_t)t * D + c] = v[3];
        float y = bf16r((acc + v[3]) * 0.4472135954999579f), ao = a.attn_out[(size_t)t * D + c];
        float nw = bf16r(a.partial ? (a.partial[(size_t)t * D + c] + ao) + y : ao + y);
        a.stack_out[(size_t)t * D + c] = nw;
        for (int n = 0; n < NS - 1; n++) xs[n][k] = a.src[n][(size_t)t * D + c];
        xs[NS - 1][k] = nw;
    }
    if (a.Q > 0) mix_core<NS, Q>(xs, a.q, a.norm, a.out + (size_t)t * Q * D, sh, sh2);
}

// q/r clip + bias table (blocks y < 24) and K/V conv + cache write (blocks y >= 24) in one launch.
// 1024 threads per block: the 128 head dimensions use the first 4 warps, the 1024 bias distances use all.
__global__ void k_attn_prep(const float* proj, const float* qscale, const float* rscale, const float* rel, int is_global, const Row* rows,
                            float* qc, float* bias, float* kring, float* vring, const float* wk, const float* wv, const float* dk,
                            const float* dv, uint16_t* Kc, uint16_t* Vc, int kv8, size_t kvstride, int t0) {
    __shared__ float sh[32 * 2]; __shared__ float sh2[32]; __shared__ float rc[DR];
    int t = t0 + blockIdx.x, d = threadIdx.x; const Row rw = rows[t]; const int p = rw.pos;
    const float* pr = proj + (size_t)t * QKVR;
    if (blockIdx.y < H) {
        int h = blockIdx.y; float tau = 1.0f;
        if (is_global) tau = 1.0f + 0.1f * logf(fmaxf((float)(p + 1) / 8192.0f, 1.0f));
        float x = d < DH ? pr[R_Q + h * DH + d] * qscale[h * DH + d] : 0.0f, y = d < DR ? pr[R_R + h * DR + d] * rscale[h * DR + d] : 0.0f;
        float n2[2] = {x * x, y * y}; breduce<2>(n2, sh, sh2);
        if (d < DH) { float v = x * fminf(QK_CLIP / (sqrtf(n2[0]) + EPS), 1.0f); if (is_global) v = bf16r(v) * tau; qc[((size_t)t * H + h) * DH + d] = bf16r(v); }
        if (d < DR) { float u = y * fminf(REL_CLIP / (sqrtf(n2[1]) + EPS), 1.0f); if (is_global) u = bf16r(u) * tau; rc[d] = bf16r(u); }
        __syncthreads();
        // Bias per distance: a thread takes EXT / blockDim distances (a block of 1,024 threads for 16 multiply-adds each
        // cost more to start than to run).
        for (int dd = d; dd < EXT; dd += blockDim.x) {
            float acc = 0; for (int i = 0; i < DR; i++) acc = fmaf(rc[i], rel[i * EXT + dd], acc);
            bias[((size_t)t * H + h) * EXT + dd] = bf16r(acc);
        }
        return;
    }
    int j = blockIdx.y - H, c = j * DH + d; float out[2] = {0, 0};
    if (d < DH) for (int s = 0; s < 2; s++) {
        float* ring = (s ? vring : kring) + (size_t)rw.seq * CR * 1024; const float* w = s ? wv : wk; const float* dl = s ? dv : dk; int R0 = s ? R_V : R_K;
        float x[4];                                      // conv inputs at lags 0..3
        for (int lag = 0; lag < 4; lag++)
            x[lag] = p - lag < 0 ? 0.0f : (t - lag >= rw.run0 ? proj[(size_t)(t - lag) * QKVR + R0 + c] : ring[(size_t)((p - lag) % CR) * 1024 + c]);
        if (rw.runlen <= CR - 3) ring[(size_t)(p % CR) * 1024 + c] = x[0];
        float acc;
        if (is_global) {
            acc = (w[c] + dl[(size_t)t * 32 + j * 4]) * x[0];
            for (int lag = 1; lag < 4; lag++) if (p - lag >= 0) acc = fmaf(w[lag * 1024 + c] + dl[(size_t)t * 32 + j * 4 + lag], x[lag], acc);
            acc = (acc + x[0]) * 0.3333333432674408f;
        } else {
            const float4 w4 = *(const float4*)(w + c * 4); const float wq[4] = {w4.x, w4.y, w4.z, w4.w};
            acc = 0; for (int k = 0; k < 4; k++) if (p - 3 + k >= 0) acc = fmaf(wq[k], x[3 - k], acc);
            acc = (acc + x[0]) * 0.4472135954999579f;
        }
        out[s] = bf16r(acc);
    }
    float n2[2] = {out[0] * out[0], out[1] * out[1]}; breduce<2>(n2, sh, sh2);
    if (d >= DH) return;
    float kv = out[0] * fminf(QK_CLIP / (sqrtf(n2[0]) + EPS), 1.0f);
    float vv = out[1] * rsqrtf(n2[1] / DH + EPS);
    Kc = (uint16_t*)((uint8_t*)Kc + (size_t)rw.seq * kvstride); Vc = (uint16_t*)((uint8_t*)Vc + (size_t)rw.seq * kvstride);   // this conversation's cache
    size_t slot = is_global ? (size_t)p : (size_t)(p % RING);
    if (kv8) {                                              // native float -> E4M3 conversion (round to nearest, saturating)
        uint32_t r; asm volatile("{\n .reg .b16 t;\n cvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\n cvt.u32.u16 %0, t;\n}" : "=r"(r) : "f"(vv), "f"(kv));
        ((uint8_t*)Kc)[ktile_off8(slot, j, d)] = (uint8_t)r; ((uint8_t*)Vc)[vtile_off8(slot, j, d)] = (uint8_t)(r >> 8);
        return;
    }
    *(uint16_t*)((uint8_t*)Kc + ktile_off(slot, j, d)) = bf16bits(kv);
    *(uint16_t*)((uint8_t*)Vc + vtile_off(slot, j, d)) = bf16bits(vv);
}

// Router top-8 for all T tokens, then the expert task list, in one single-block launch.
__global__ void k_route_tasks(const float* logit, const float* rbias, int T, int* ids, float* gamma, int* tasks, int* ntasks) {
    __shared__ float sel[NE]; __shared__ float sg[TOPK]; __shared__ int sid[HMAX * TOPK]; __shared__ int nt[NE];   // (launched for one token only)
    int e = threadIdx.x;
    for (int t = 0; t < T; t++) {
        float l = logit[(size_t)t * NE + e];
        sel[e] = sigmoidf(l) + rbias[e]; __syncthreads();
        int rank = 0; for (int j = 0; j < NE; j++) rank += (sel[j] > sel[e]) || (sel[j] == sel[e] && j < e);
        if (rank < TOPK) { sg[rank] = sigmoidf(l); sid[t * TOPK + rank] = e; ids[t * TOPK + rank] = e; }
        __syncthreads();
        if (e < TOPK) { float s = 0; for (int k = 0; k < TOPK; k++) s += sg[k]; gamma[t * TOPK + e] = sg[e] / s * 8.0f; }
        __syncthreads();
    }
    int N = T * TOPK, cols[HMAX], cnt = 0;
    for (int n = 0; n < N; n++) if (sid[n] == e) cols[cnt++] = n;
    nt[e] = (cnt + 7) / 8; __syncthreads();
    int rank = 0; for (int j = 0; j < e; j++) rank += nt[j];
    for (int k = 0; k * 8 < cnt; k++) {
        int* tk = tasks + (rank + k) * TASKW; int c = min(8, cnt - k * 8);
        tk[0] = e; tk[1] = c; for (int q = 0; q < c; q++) tk[2 + q] = cols[k * 8 + q];
    }
    if (e == NE - 1) *ntasks = rank + nt[e];
}

// One 16-element NVFP4 block: choose the scale candidate, return packed codes and the scale byte (6.2).
__device__ __forceinline__ void q4_block(const float* x, float bm, float amax, uint64_t* packed, uint8_t* sc) {
    float enc = fminf(1536.0f / amax, FLT_MAX), dec = 1.0f / enc, outer = amax * (1.0f / 1536.0f);
    float val[2][16], err[2], s2[2];
    for (int cnd = 0; cnd < 2; cnd++) {
        float s = e4m3_rn(fminf(bm * (enc * (cnd ? 0.25f : (1.0f / 6.0f))), 448.0f));
        float inv = fminf(1.0f / (s * dec), FLT_MAX), e[16];
        for (int i = 0; i < 16; i++) { val[cnd][i] = e2m1_rn(x[i] * inv); e[i] = fabsf(val[cnd][i] * s * outer - x[i]); }
        for (int w = 8; w > 0; w >>= 1) for (int i = 0; i < w; i++) e[i] = e[2 * i] + e[2 * i + 1];
        err[cnd] = e[0]; s2[cnd] = s;
    }
    int pick = err[1] < err[0]; uint64_t pk = 0;
    if (amax > 0) for (int i = 0; i < 16; i++) pk |= (uint64_t)e2m1_code(val[pick][i]) << (4 * i);
    *packed = pk; *sc = amax > 0 ? e4m3_code(s2[pick]) : 0;
}
// Quantise rows of n values to NVFP4 codes (no float copy). One block per row.
__global__ void k_quant4c(const float* x, int n, float* amax_out, uint8_t* code_out, uint8_t* scale_out, PF pf) {
    __shared__ float sh[256];
    PF_X(pf)
    int row = blockIdx.x, b = threadIdx.x, nb = n / 16; float xv[16], bm = 0;
    if (b < nb) for (int i = 0; i < 16; i++) { xv[i] = x[(size_t)row * n + b * 16 + i]; bm = fmaxf(bm, fabsf(xv[i])); }
    float amax = bmax(bm, sh);
    if (b == 0) amax_out[row] = amax;
    if (b >= nb) return;
    uint64_t pk; uint8_t sc; q4_block(xv, bm, amax, &pk, &sc);
    *(uint64_t*)(code_out + (size_t)row * (n / 2) + b * 8) = pk; scale_out[(size_t)row * nb + b] = sc;
}
// SwiGLU (5.2) fused with the quantisation of its output. One block per (token, slot) row of 1152 values.
__global__ void k_swiglu_quant4(const float* gu, const float* gamma, float* amax_out, uint8_t* code_out, uint8_t* scale_out, PF pf) {
    __shared__ float sh[256];
    PF_X(pf)
    int n = blockIdx.x, b = threadIdx.x, nb = I2 / 16; float xv[16], bm = 0, gm = gamma[n];
    if (b < nb) for (int i = 0; i < 16; i++) {
        int j = b * 16 + i; float g = gu[(size_t)n * D + 2 * j], u = gu[(size_t)n * D + 2 * j + 1];
        xv[i] = bf16r(((4.0f * tanhf(g * 0.25f) * sigmoidf(g)) * (25.0f * tanhf(u * 0.04f))) * gm); bm = fmaxf(bm, fabsf(xv[i]));
    }
    float amax = bmax(bm, sh);
    if (b == 0) amax_out[n] = amax;
    if (b >= nb) return;
    uint64_t pk; uint8_t sc; q4_block(xv, bm, amax, &pk, &sc);
    *(uint64_t*)(code_out + (size_t)n * (I2 / 2) + b * 8) = pk; scale_out[(size_t)n * nb + b] = sc;
}

// Loop controller for the three attention streams at once (2.4): pair -> [gemv] -> gate -> norm.
__global__ void k_pair3(const float* dest, int dstride, int doff, const float* msg, float* pair) {
    __shared__ float sh[(NT / 32) * 2]; __shared__ float sh2[32];
    int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x; const float* d = dest + (size_t)t * dstride + s * doff; const float* m = msg + ((size_t)t * 4 + s) * D;
    float dv[DPT], mv[DPT], r[2] = {0, 0};
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; dv[k] = d[c]; mv[k] = m[c]; r[0] += dv[k] * dv[k]; r[1] += mv[k] * mv[k]; }
    breduce<2>(r, sh, sh2);
    float id = rsqrtf(r[0] / D + EPS), im = rsqrtf(r[1] / D + EPS); float* o = pair + ((size_t)t * 3 + s) * 2 * D;
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; o[c] = bf16r(dv[k] * id); o[D + c] = bf16r(mv[k] * im); }
}
// As k_gate / k_gate_big, for the three streams q, k, v. Grids (D/8, 3 T) and (D/8, 3).
__global__ void k_gate3(const float* h, const float* gain, const float* forget, const float* dest, int dstride, int doff, const float* msg, float* x) {
    int ts = blockIdx.y, t = ts / 3, s = ts % 3, c = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    const float* hh = h + (size_t)t * 192 + s * 64; const float* g = gain + ((size_t)s * D + c) * 64; const float* f = forget + ((size_t)s * D + c) * 64;
    float ga = fmaf(hh[lane], g[lane], hh[lane + 32] * g[lane + 32]), fa = fmaf(hh[lane], f[lane], hh[lane + 32] * f[lane + 32]);
    ga = wsum(ga); fa = wsum(fa);
    if (lane == 0) { float a = (1.0f - tanhf(fa)) * dest[(size_t)t * dstride + s * doff + c]; float b = tanhf(ga) * msg[((size_t)t * 4 + s) * D + c]; x[((size_t)t * 3 + s) * D + c] = a + b; }
}
__global__ void k_gate3_big(const float* h, const float* gain, const float* forget, const float* dest, int dstride, int doff, const float* msg, float* x, int T) {
    __shared__ float sh[HMAX * 64]; __shared__ float sg[8 * 64]; __shared__ float sf[8 * 64];
    int s = blockIdx.y, c0 = blockIdx.x * 8, tid = threadIdx.x;
    for (int i = tid; i < 8 * 64; i += 256) { sg[i] = gain[((size_t)s * D + c0) * 64 + i]; sf[i] = forget[((size_t)s * D + c0) * 64 + i]; }
    for (int t0 = 0; t0 < T; t0 += HMAX) {
        int n = min(HMAX, T - t0);
        __syncthreads();
        for (int i = tid; i < n * 64; i += 256) sh[i] = h[(size_t)(t0 + (i >> 6)) * 192 + s * 64 + (i & 63)];
        __syncthreads();
        for (int p = tid; p < n * 8; p += 256) {
            int t = t0 + (p >> 3), w = p & 7, c = c0 + w; const float* hh = sh + (p >> 3) * 64; const float* g = sg + w * 64; const float* f = sf + w * 64;
            float va[32], vf[32];
            for (int l = 0; l < 32; l++) { va[l] = fmaf(hh[l], g[l], hh[l + 32] * g[l + 32]); vf[l] = fmaf(hh[l], f[l], hh[l + 32] * f[l + 32]); }
            for (int o = 16; o > 0; o >>= 1) for (int l = 0; l < o; l++) { va[l] += va[l + o]; vf[l] += vf[l + o]; }   // the order a warp sum takes
            float a = (1.0f - tanhf(vf[0])) * dest[(size_t)t * dstride + s * doff + c]; float b = tanhf(va[0]) * msg[((size_t)t * 4 + s) * D + c];
            x[((size_t)t * 3 + s) * D + c] = a + b;
        }
    }
}
__global__ void k_rms3(const float* x, const float* w, int wstride, float* out) {
    __shared__ float sh[NT / 32]; __shared__ float sh2[32];
    int t = blockIdx.x, s = blockIdx.y, tid = threadIdx.x; const float* r = x + ((size_t)t * 3 + s) * D;
    float v[DPT], ss = 0; for (int k = 0; k < DPT; k++) { v[k] = r[tid + NT * k]; ss += v[k] * v[k]; }
    breduce<1>(&ss, sh, sh2);
    float inv = rsqrtf(ss / D + EPS);
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; out[((size_t)t * 3 + s) * D + c] = bf16r(v[k] * inv * w[(size_t)s * wstride + c]); }
}


// =====================================================================================================
// Tensor-core GEMMs for batches (T > 1). Layouts verified by tools/mmaprobe/probe2.cu.
// =====================================================================================================
__device__ __forceinline__ void mma_bf16(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]));
}
__device__ __forceinline__ void mma_fp8(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1, uint32_t sa, uint32_t sb) {
    uint16_t z = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(d[0]), "f"(d[1]), "f"(d[2]), "f"(d[3]), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) { return (__float_as_uint(lo) >> 16) | (__float_as_uint(hi) & 0xFFFF0000u); }

// =====================================================================================================
// Attention on the tensor cores (3.6, 3.7), three kernels:
//   scores:  logits[key][head] = K . q + relative bias      one warp per group of 16 keys, 8 multiplies
//   soft:    P = bf16(exp(logit - max)), log-sum-exp         one block per (token, query head)
//   pv:      out = (V^T . P) / sum, then sink and gate       one block per (token, KV head, 16 dims)
// The scratch buffer is indexed by cache slot. Slots outside the token's window get logit = -FLT_MAX, P = 0.
// =====================================================================================================
__device__ __forceinline__ bool slot_position(int slot, int p, int is_global, int* dist) {     // is this slot visible from position p?
    if (is_global) { *dist = p - slot; return slot <= p; }
    int delta = ((p % RING) - slot + RING) % RING;          // the newest position stored in this ring slot is p - delta
    *dist = delta; return delta < EXT && p - delta >= 0;
}
// =====================================================================================================
// Attention for groups of rows. A tensor-core operation takes 8 columns; one row's three query heads fill three of
// them, and every row used to read all the keys (and values) of its conversation for itself: 128 rows of a prompt
// step read the same 2.4 MB of ring 128 times per layer visit. Rows of one conversation see the same keys, so a
// block now takes up to 8 consecutive rows: their 24 (row, query head) pairs are the columns of three operations
// per key tile, and the keys are read once for all of them. Which row may see which slot is decided per element
// afterwards, as before. Each score and each output value is produced by the same chain of operations over the
// same operands as with one row per block, so results are the same bit for bit.
// A step's rows are cut into groups at every 8th row and wherever a conversation's run begins (Grp); gb[k] is the
// index of the first group at or after row 16 k, so that a launch over a sub-batch finds its groups.
// =====================================================================================================
struct Grp { int t0, n; };
// Score scratch of the sliding-window layers: fp32 scores of HMAX rows, then their probabilities as operands: at most
// 3/8 column sets per row plus one per group.
const size_t LOCAL_SC = (size_t)HMAX * H * RING * 4 + (size_t)(HMAX * 3 / 8 + HMAX) * HKV * (RING / 16) * 256;
// KGW: key groups per warp (a block covers 8 KGW of them: the queries are staged once per block).
template <int KGW> __global__ void k_attn_scores_g(const float* qc, const float* bias, const uint8_t* Kt, const Row* rows, const Grp* grp, const int* gb, int gi0, int gi1,
                                int is_global, int ngroups, float* scratch, int smax, int t0, int kv8, size_t kvstride) {
    __shared__ uint2 sq[3][8][32];                        // queries as tensor-core operands: [column set][slice][lane]
    int gy = blockIdx.y / HKV, j = blockIdx.y % HKV, tid = threadIdx.x, lane = tid & 31, qq = lane & 3, g = lane >> 2;
    int gbeg = gb[gi0]; if (gbeg + gy >= gb[gi1]) return;
    const Grp gr = grp[gbeg + gy]; const int tf = gr.t0, n = gr.n, p0 = rows[tf].pos;
    Kt += (size_t)rows[tf].seq * kvstride;
    // Full attention: the grid covers a power-of-two bucket of keys, the conversation perhaps half of it. Nothing past
    // the key group after the last row's own is ever read (k_attn_soft_g stops there), so nothing is computed there.
    if (is_global && (int)blockIdx.x * 8 * KGW > (p0 + n - 1) / 16 + 1) return;
    for (int e = tid; e < 3 * 8 * 32; e += 256) {
        int cs = e >> 8, ds = (e >> 5) & 7, ln = e & 31, c = cs * 8 + (ln >> 2), ti = c / 3, hh = c - ti * 3; uint2 v = make_uint2(0, 0);
        if (ti < n) { const float* x = qc + ((size_t)(tf + ti) * H + 3 * j + hh) * DH + ds * 16 + 2 * (ln & 3); v = make_uint2(pack_bf16(x[0], x[1]), pack_bf16(x[8], x[9])); }
        sq[cs][ds][ln] = v;
    }
    __syncthreads();
    for (int G = (blockIdx.x * 8 + (tid >> 5)) * KGW, Gend = G + KGW; G < Gend; G++) {
        if (G >= ngroups) break;
        if (is_global && G > (p0 + n - 1) / 16 + 1) break;
        const bool dark = is_global && G * 16 > p0 + n - 1;  // keys past every row of the group: masked, and their memory may not be attached
        uint4 w[8];
        if (!dark) {
            if (kv8) { const uint64_t* k8 = (const uint64_t*)(Kt + (((size_t)G * HKV + j) * 8) * 256 + lane * 8); for (int ds = 0; ds < 8; ds++) w[ds] = tile8_to_bf16(k8[(size_t)ds * 32]); }
            else { const uint4* kt = (const uint4*)(Kt + (((size_t)G * HKV + j) * 8) * 512 + lane * 16); for (int ds = 0; ds < 8; ds++) w[ds] = kt[(size_t)ds * 32]; }
        }
        for (int cs = 0; cs < 3; cs++) {
            if (cs * 8 >= n * 3) break;
            float d[4] = {0, 0, 0, 0};
            if (!dark) for (int ds = 0; ds < 8; ds++) { uint2 b = sq[cs][ds][lane]; mma_bf16(d, w[ds].x, w[ds].y, w[ds].z, w[ds].w, b.x, b.y); }
            for (int i = 0; i < 4; i++) {                         // this lane holds keys g, g+8 and columns 2qq, 2qq+1 of the set
                int c = cs * 8 + 2 * qq + (i & 1), ti = c / 3, hh = c - ti * 3; if (ti >= n) continue;
                int t = tf + ti, p = p0 + ti, slot = G * 16 + g + (i >> 1) * 8, h = 3 * j + hh, dist;
                bool ok = !(is_global && G * 16 > p) && slot_position(slot, p, is_global, &dist);
                scratch[((size_t)(t - t0) * H + h) * smax + slot] = ok ? d[i] * 0.08838834764831845f + (dist < EXT ? bias[((size_t)t * H + h) * EXT + dist] : 0.0f) : -FLT_MAX;
            }
        }
    }
}
// As k_attn_soft; also writes P as bf16 in the order the value kernel's operand takes it (pk: [row][slot'], 2 bytes
// each): within a group of 16 slots, slot r sits at (r % 8) / 2 * 4 + r / 8 * 2 + (r & 1), so that one 8-byte load
// is a thread's operand.
// strided (full attention): slots past the key group after the row's own hold nothing (masked scores, P = 0, which
// change neither the maximum nor the sum), so the row stops there; the value kernel reads no further either (rows of
// a group are at most 7 positions apart).
// oplay: write P as the operands of k_attn_pv_gl instead: [(group in the launch * 3 + column set) * HKV + KV head][key
// group][lane], 8 bytes each; the launch's rows must then be whole groups of 8 from its first row (step class 0).
__global__ void k_attn_soft_g(float* scratch, int smax, int nslots, float* lstat, int t0, uint16_t* pk, const Row* rows, int strided, int oplay) {
    __shared__ float sm[32];
    int t = t0 + blockIdx.x, h = blockIdx.y, tid = threadIdx.x; float* sc = scratch + ((size_t)blockIdx.x * H + h) * smax; uint16_t* pr = pk + ((size_t)blockIdx.x * H + h) * smax;
    const int oc = ((int)blockIdx.x & 7) * 3 + h % 3;       // this (row, head)'s column in its group
    if (oplay) pr = pk + ((size_t)(((int)blockIdx.x >> 3) * 3 + (oc >> 3)) * HKV + h / 3) * (smax / 16) * 128 + (oc & 7) * 16;
    if (strided) nslots = min(nslots, (rows[t].pos / 16 + 2) * 16);
    float mx = -FLT_MAX; for (int i = tid; i < nslots; i += 1024) mx = fmaxf(mx, sc[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    float m = sm[0]; for (int w = 1; w < 32; w++) m = fmaxf(m, sm[w]);
    __syncthreads();
    float l = 0;
    for (int i = tid; i < nslots; i += 1024) {
        float x = sc[i], P = x == -FLT_MAX ? 0.0f : expf(x - m); l += P;
        int r = i & 15; pr[(oplay ? (i >> 4) * 128 : (i & ~15)) + ((r & 7) >> 1) * 4 + (r >> 3) * 2 + (r & 1)] = (uint16_t)(__float_as_uint(bf16r(P)) >> 16);
    }
    l = wsum(l); if ((tid & 31) == 0) sm[tid >> 5] = l;
    __syncthreads();
    if (tid == 0) { float ls = 0; for (int w = 0; w < 32; w++) ls += sm[w]; lstat[((size_t)t * H + h) * 2] = ls; lstat[((size_t)t * H + h) * 2 + 1] = m + logf(ls); }
}
// As k_attn_pv, for a group of rows: one block per (group, KV head, 16 dims); the value tiles are read once for the
// group's rows. A row for which a key group lies past its own position has P = 0 there: the operation adds zeros.
template <int NW> __global__ void k_attn_pv_g(const uint16_t* PK, int smax, const uint8_t* Vt, int ngroups, const float* lstat, const float* proj, float* y, int strided, int t0, int kv8,
                                              const Row* rows, size_t kvstride, const Grp* grp, const int* gb, int gi0, int gi1) {
    __shared__ float sh[NW][32][4];
    int dg = blockIdx.x, gy = blockIdx.y / HKV, j = blockIdx.y % HKV, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, qq = lane & 3, g = lane >> 2;
    int gbeg = gb[gi0]; if (gbeg + gy >= gb[gi1]) return;
    const Grp gr = grp[gbeg + gy]; const int tf = gr.t0, n = gr.n;
    Vt += (size_t)rows[tf].seq * kvstride;
    const int gmax = strided ? (rows[tf].pos + n - 1) / 16 : ngroups, ncs = (n * 3 + 7) / 8;      // column sets in use
    int kseg = (ngroups + NW - 1) / NW, g0 = warp * kseg;
    const uint2* pr[3];                                   // this thread's column in each set: (row, query head) -> its packed P
    for (int cs = 0; cs < 3; cs++) { int c = cs * 8 + g, ti = c / 3, hh = c - ti * 3; pr[cs] = ti < n ? (const uint2*)(PK + ((size_t)(tf + ti - t0) * H + 3 * j + hh) * smax) : nullptr; }
    float d[3][4]; for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) d[cs][i] = 0;
    for (int it = 0; it < kseg; it++) {                   // (loading several key groups ahead of the operations was tried: 25% slower)
        int G = strided ? warp + NW * it : g0 + it; bool live = G < ngroups && G <= gmax;
        if (strided && G > gmax) break;                   // only zeros from here on: they would change nothing
        uint4 w = !live ? make_uint4(0, 0, 0, 0) : kv8 ? tile8_to_bf16(*(const uint64_t*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 256) + lane * 8))
                                                       : *(const uint4*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 512) + lane * 16);
        for (int cs = 0; cs < ncs; cs++) {
            uint2 b = live && pr[cs] ? pr[cs][G * 4 + qq] : make_uint2(0, 0);
            mma_bf16(d[cs], w.x, w.y, w.z, w.w, b.x, b.y);
        }
    }
    for (int cs = 0; cs < ncs; cs++) {
        for (int i = 0; i < 4; i++) sh[warp][lane][i] = d[cs][i];
        __syncthreads();
        if (warp == 0) for (int i = 0; i < 4; i++) {          // this lane holds dims g, g+8 of the group and columns 2qq, 2qq+1 of the set
            int c = cs * 8 + 2 * qq + (i & 1), ti = c / 3, hh = c - ti * 3; if (ti >= n) continue;
            int t = tf + ti, dim = dg * 16 + g + (i >> 1) * 8, h = 3 * j + hh; const float* pj = proj + (size_t)t * QKVR;
            float v = 0; for (int w2 = 0; w2 < NW; w2++) v += sh[w2][lane][i];
            float ls = lstat[((size_t)t * H + h) * 2], lse = lstat[((size_t)t * H + h) * 2 + 1];
            y[(size_t)t * (H * DH) + h * DH + dim] = bf16r((v / ls) * sigmoidf(pj[R_GATE + h * DH + dim]) * sigmoidf(lse - pj[R_SINK + h]));
        }
        __syncthreads();
    }
}
// k_attn_pv_g with P stored as operands (k_attn_soft_g, oplay): a warp's operand for a key group and column set is
// one coalesced 256-byte load, where the per-row layout made it eight pieces in eight cache lines.
template <int NW> __global__ void k_attn_pv_gl(const uint2* PL, int smax, const uint8_t* Vt, int ngroups, const float* lstat, const float* proj, float* y, int t0, int kv8,
                                               const Row* rows, size_t kvstride, const Grp* grp, const int* gb, int gi0, int gi1) {
    __shared__ float sh[NW][32][4];
    int dg = blockIdx.x, gy = blockIdx.y / HKV, j = blockIdx.y % HKV, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, qq = lane & 3, g = lane >> 2;
    int gbeg = gb[gi0]; if (gbeg + gy >= gb[gi1]) return;
    const Grp gr = grp[gbeg + gy]; const int tf = gr.t0, n = gr.n;
    Vt += (size_t)rows[tf].seq * kvstride;
    const int gmax = (rows[tf].pos + n - 1) / 16, ncs = (n * 3 + 7) / 8, KG = smax / 16;
    const uint2* pl = PL + ((size_t)(gy * 3) * HKV + j) * KG * 32 + lane;      // column set cs: + cs * HKV * KG * 32; key group G: + G * 32
    float d[3][4]; for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) d[cs][i] = 0;
    for (int G = warp; G < ngroups && G <= gmax; G += NW) {
        uint4 w = kv8 ? tile8_to_bf16(*(const uint64_t*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 256) + lane * 8))
                      : *(const uint4*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 512) + lane * 16);
        #pragma unroll
        for (int cs = 0; cs < 3; cs++) if (cs < ncs) { uint2 b = pl[((size_t)cs * HKV * KG + G) * 32]; mma_bf16(d[cs], w.x, w.y, w.z, w.w, b.x, b.y); }
    }
    #pragma unroll
    for (int cs = 0; cs < 3; cs++) {
        if (cs >= ncs) break;
        for (int i = 0; i < 4; i++) sh[warp][lane][i] = d[cs][i];
        __syncthreads();
        if (warp == 0) for (int i = 0; i < 4; i++) {
            int c = cs * 8 + 2 * qq + (i & 1), ti = c / 3, hh = c - ti * 3; if (ti >= n) continue;
            int t = tf + ti, dim = dg * 16 + g + (i >> 1) * 8, h = 3 * j + hh; const float* pj = proj + (size_t)t * QKVR;
            float v = 0; for (int w2 = 0; w2 < NW; w2++) v += sh[w2][lane][i];
            float ls = lstat[((size_t)t * H + h) * 2], lse = lstat[((size_t)t * H + h) * 2 + 1];
            y[(size_t)t * (H * DH) + h * DH + dim] = bf16r((v / ls) * sigmoidf(pj[R_GATE + h * DH + dim]) * sigmoidf(lse - pj[R_SINK + h]));
        }
        __syncthreads();
    }
}
// =====================================================================================================
// Sliding-window attention for groups of rows, in blocks that do enough work to be worth starting. The grouped
// kernels above keep the shape of the one-row kernels: a block per 8 key groups (scores) or per 16 dims with its
// 8 warps taking 9 key groups each (values), about 25 operations per warp, and then synchronisation, an exchange
// through shared memory and the stores. The tensor cores do an operation in 2.9 ns per core; these kernels spent
// 14 and 26. Here a block is a (group of rows, KV head): 8 warps that each do 216 operations with nothing shared.
//   k_attn_scores_l  the queries are staged once per block; warp w takes key groups 9 w .. 9 w + 8.
//   k_attn_soft_l    one warp per (row, head), which goes through the 1,024-thread kernel's work in its order:
//                    thread i of that kernel has slots i and i + 1024, its warp adds its 32 threads by wsum, and
//                    warp totals are added in order. Writes P as the value kernel's operands: PL[(column set) *
//                    72 + key group][lane], 8 bytes each, one coalesced load per operation.
//   k_attn_pv_l      warp = 16 dims; it runs the 8 chains of 9 key groups one after the other and adds the partial
//                    sums in chain order, which is what the 8 warps of k_attn_pv and their exchange compute.
// Same operations on the same operands in the same order as the kernels they replace: same results bit for bit.
// cso[g]: the first column set of group g (a group of n rows has ceil(3 n / 8)); rowg[t]: the group of row t.
// =====================================================================================================
template <int NWS> __global__ void k_attn_scores_l(const float* qc, const float* bias, const uint8_t* Kt, const Row* rows, const Grp* grp, const int* gb, int gi0, int gi1,
                                float* scratch, int t0, int kv8, size_t kvstride) {
    __shared__ uint2 sq[3][8][32];
    int gy = blockIdx.x / HKV, j = blockIdx.x % HKV, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, qq = lane & 3, g = lane >> 2;
    int gbeg = gb[gi0]; if (gbeg + gy >= gb[gi1]) return;
    const Grp gr = grp[gbeg + gy]; const int tf = gr.t0, n = gr.n, p0 = rows[tf].pos, ncs = (n * 3 + 7) / 8;
    Kt += (size_t)rows[tf].seq * kvstride;
    for (int e = tid; e < 3 * 8 * 32; e += NWS * 32) {
        int cs = e >> 8, ds = (e >> 5) & 7, ln = e & 31, c = cs * 8 + (ln >> 2), ti = c / 3, hh = c - ti * 3; uint2 v = make_uint2(0, 0);
        if (ti < n) { const float* x = qc + ((size_t)(tf + ti) * H + 3 * j + hh) * DH + ds * 16 + 2 * (ln & 3); v = make_uint2(pack_bf16(x[0], x[1]), pack_bf16(x[8], x[9])); }
        sq[cs][ds][ln] = v;
    }
    __syncthreads();
    const int KG = RING / 16 / NWS, pm0 = p0 % RING;      // key groups per warp
    for (int G = warp * KG; G < warp * KG + KG; G++) {
        uint4 w[8];
        if (kv8) { const uint64_t* k8 = (const uint64_t*)(Kt + (((size_t)G * HKV + j) * 8) * 256 + lane * 8); for (int ds = 0; ds < 8; ds++) w[ds] = tile8_to_bf16(k8[(size_t)ds * 32]); }
        else { const uint4* kt = (const uint4*)(Kt + (((size_t)G * HKV + j) * 8) * 512 + lane * 16); for (int ds = 0; ds < 8; ds++) w[ds] = kt[(size_t)ds * 32]; }
        #pragma unroll
        for (int cs = 0; cs < 3; cs++) {
            if (cs >= ncs) break;
            float d[4] = {0, 0, 0, 0};
            for (int ds = 0; ds < 8; ds++) { uint2 b = sq[cs][ds][lane]; mma_bf16(d, w[ds].x, w[ds].y, w[ds].z, w[ds].w, b.x, b.y); }
            for (int i = 0; i < 4; i++) {
                int c = cs * 8 + 2 * qq + (i & 1), ti = c / 3, hh = c - ti * 3; if (ti >= n) continue;
                int t = tf + ti, p = p0 + ti, slot = G * 16 + g + (i >> 1) * 8, h = 3 * j + hh;
                int pm = pm0 + ti; if (pm >= RING) pm -= RING;                   // p % RING (slot_position without its divisions)
                int dist = pm - slot; if (dist < 0) dist += RING;
                bool ok = dist < EXT && p - dist >= 0;
                scratch[((size_t)(t - t0) * H + h) * RING + slot] = ok ? d[i] * 0.08838834764831845f + bias[((size_t)t * H + h) * EXT + dist] : -FLT_MAX;
            }
        }
    }
}
template <int NP> __global__ void k_attn_soft_l(const float* scratch, float* lstat, int t0, uint2* PL, const int* rowg, const Grp* grp, const int* cso, const int* gb, int gi0) {
    __shared__ float sm[32];
    int t = t0 + blockIdx.x, h = blockIdx.y, lane = threadIdx.x & 31, pw = threadIdx.x >> 5; const float* sc = scratch + ((size_t)blockIdx.x * H + h) * RING;
    const int gi = rowg[t], c = (t - grp[gi].t0) * 3 + h % 3, j = h / 3;
    uint16_t* pl = (uint16_t*)(PL + ((size_t)((cso[gi] - cso[gb[gi0]]) + c / 8) * HKV + j) * (RING / 16) * 32) + (c % 8) * 16;   // + key group * 128 + qq * 4 + half * 2 + odd
    float mx = -FLT_MAX; for (int i = threadIdx.x; i < RING; i += NP * 32) mx = fmaxf(mx, sc[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (NP > 1) {
        if (lane == 0) sm[pw] = mx;
        __syncthreads();
        mx = sm[0]; for (int w = 1; w < NP; w++) mx = fmaxf(mx, sm[w]);
        __syncthreads();
    }
    float ls = 0;
    for (int w = pw; w < 32; w += NP) {                   // warp w of the 1,024-thread kernel: its thread 32 w + lane has slots i and i + 1024
        float l = 0;
        for (int i = 32 * w + lane; i < RING; i += 1024) {
            float x = sc[i], P = x == -FLT_MAX ? 0.0f : expf(x - mx); l += P;
            int r = i & 15; pl[(i >> 4) * 128 + ((r & 7) >> 1) * 4 + (r >> 3) * 2 + (r & 1)] = (uint16_t)(__float_as_uint(bf16r(P)) >> 16);
        }
        l = wsum(l);
        if (NP == 1) ls += l; else if (lane == 0) sm[w] = l;
    }
    if (NP > 1) { __syncthreads(); if (threadIdx.x == 0) for (int w = 0; w < 32; w++) ls += sm[w]; }
    if (threadIdx.x == 0) { lstat[((size_t)t * H + h) * 2] = ls; lstat[((size_t)t * H + h) * 2 + 1] = mx + logf(ls); }
}
// Block = (group, KV head, 16 dims); its W warps take 8 / W of the 8 chains each, one after the other; the chains'
// partial sums meet in shared memory and are added in chain order.
template <int W, int UNV> __global__ void k_attn_pv_l(const uint2* PL, const uint8_t* Vt, const float* lstat, const float* proj, float* y, int kv8, const Row* rows, size_t kvstride,
                            const Grp* grp, const int* cso, const int* gb, int gi0, int gi1) {
    __shared__ float sh[8][32][12];
    int dg = blockIdx.x, gy = blockIdx.y / HKV, j = blockIdx.y % HKV, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, qq = lane & 3, g = lane >> 2;
    int gbeg = gb[gi0]; if (gbeg + gy >= gb[gi1]) return;
    const Grp gr = grp[gbeg + gy]; const int tf = gr.t0, n = gr.n, ncs = (n * 3 + 7) / 8;
    Vt += (size_t)rows[tf].seq * kvstride;
    const uint2* pl = PL + ((size_t)(cso[gbeg + gy] - cso[gbeg]) * HKV + j) * (RING / 16) * 32 + lane;      // column set cs: + cs * HKV * 72 * 32
    float tot[3][4]; for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) tot[cs][i] = 0;
    for (int ch = warp * (8 / W); ch < (warp + 1) * (8 / W); ch++) {
        float d[3][4]; for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) d[cs][i] = 0;
        for (int G0 = ch * 9; G0 < ch * 9 + 9; G0 += UNV) {   // UNV key groups: their loads first (the waits overlap), then the operations
            uint4 w[UNV]; uint2 b[UNV][3];
            #pragma unroll
            for (int u = 0; u < UNV; u++) {
                int G = G0 + u;
                w[u] = kv8 ? tile8_to_bf16(*(const uint64_t*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 256) + lane * 8))
                           : *(const uint4*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 512) + lane * 16);
                #pragma unroll
                for (int cs = 0; cs < 3; cs++) b[u][cs] = cs < ncs ? pl[((size_t)cs * HKV * (RING / 16) + G) * 32] : make_uint2(0, 0);
            }
            #pragma unroll
            for (int u = 0; u < UNV; u++) {
                #pragma unroll
                for (int cs = 0; cs < 3; cs++) if (cs < ncs) mma_bf16(d[cs], w[u].x, w[u].y, w[u].z, w[u].w, b[u][cs].x, b[u][cs].y);
            }
        }
        if (W == 1) { for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) tot[cs][i] += d[cs][i]; }
        else for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) sh[ch][lane][cs * 4 + i] = d[cs][i];
    }
    if (W > 1) {
        __syncthreads();
        if (warp) return;
        for (int ch = 0; ch < 8; ch++) for (int cs = 0; cs < 3; cs++) for (int i = 0; i < 4; i++) tot[cs][i] += sh[ch][lane][cs * 4 + i];
    }
    #pragma unroll
    for (int cs = 0; cs < 3; cs++) {
        if (cs >= ncs) break;
        for (int i = 0; i < 4; i++) {                         // this lane holds dims g, g+8 of the group and columns 2qq, 2qq+1 of the set
            int c = cs * 8 + 2 * qq + (i & 1), ti = c / 3, hh = c - ti * 3; if (ti >= n) continue;
            int t = tf + ti, dim = dg * 16 + g + (i >> 1) * 8, h = 3 * j + hh; const float* pj = proj + (size_t)t * QKVR;
            float ls = lstat[((size_t)t * H + h) * 2], lse = lstat[((size_t)t * H + h) * 2 + 1];
            y[(size_t)t * (H * DH) + h * DH + dim] = bf16r((tot[cs][i] / ls) * sigmoidf(pj[R_GATE + h * DH + dim]) * sigmoidf(lse - pj[R_SINK + h]));
        }
    }
}
// t0: first token handled by this launch; scratch rows are relative to it (the scratch holds one sub-batch).
__global__ void k_attn_scores(const float* qc, const float* bias, const uint8_t* Kt, const Row* rows, int is_global, int ngroups, float* scratch, int smax, int t0, int kv8, size_t kvstride) {
    __shared__ float q[3][DH];
    int tj = blockIdx.y, t = t0 + tj / HKV, j = tj % HKV, tid = threadIdx.x, lane = tid & 31, qq = lane & 3, g = lane >> 2, p = rows[t].pos;
    Kt += (size_t)rows[t].seq * kvstride;
    int G = blockIdx.x * 8 + (tid >> 5);
    if (is_global && (int)blockIdx.x * 8 > p / 16 + 1) return;   // past the key group after the row's own: never read (see k_attn_soft)
    for (int i = tid; i < 3 * DH; i += 256) q[i / DH][i % DH] = qc[((size_t)t * H + 3 * j) * DH + i];
    __syncthreads();
    if (G >= ngroups || (is_global && G > p / 16 + 1)) return;
    if (is_global && G * 16 > p) {                           // keys past this row's own position: masked, and their memory may not be attached
        for (int i = 0; i < 4; i++) { int col = 2 * qq + (i & 1), slot = G * 16 + g + (i >> 1) * 8; if (col < 3) scratch[((size_t)(t - t0) * H + 3 * j + col) * smax + slot] = -FLT_MAX; }
        return;
    }
    uint4 w[8];                                              // all loads first, then the multiplies
    if (kv8) { const uint64_t* k8 = (const uint64_t*)(Kt + (((size_t)G * HKV + j) * 8) * 256 + lane * 8); for (int ds = 0; ds < 8; ds++) w[ds] = tile8_to_bf16(k8[(size_t)ds * 32]); }
    else { const uint4* kt = (const uint4*)(Kt + (((size_t)G * HKV + j) * 8) * 512 + lane * 16); for (int ds = 0; ds < 8; ds++) w[ds] = kt[(size_t)ds * 32]; }
    float d[4] = {0, 0, 0, 0};
    for (int ds = 0; ds < 8; ds++) {
        uint32_t b0 = 0, b1 = 0;
        if (g < 3) { const float* x = q[g] + ds * 16 + 2 * qq; b0 = pack_bf16(x[0], x[1]); b1 = pack_bf16(x[8], x[9]); }
        mma_bf16(d, w[ds].x, w[ds].y, w[ds].z, w[ds].w, b0, b1);
    }
    for (int i = 0; i < 4; i++) {                             // this lane holds keys g, g+8 and columns 2qq, 2qq+1
        int col = 2 * qq + (i & 1), slot = G * 16 + g + (i >> 1) * 8, h = 3 * j + col, dist;
        if (col >= 3) continue;
        bool ok = slot_position(slot, p, is_global, &dist);
        scratch[((size_t)(t - t0) * H + h) * smax + slot] = ok ? d[i] * 0.08838834764831845f + (dist < EXT ? bias[((size_t)t * H + h) * EXT + dist] : 0.0f) : -FLT_MAX;
    }
}
// lstat[(t*H + h)*2] = {sum of exp(logit - max), log-sum-exp}; scratch is overwritten with P rounded to bf16.
// strided (full attention): slots past the key group after the row's own are masked (they change neither the maximum
// nor the sum, and P = 0 there), so the row stops there.
__global__ void k_attn_soft(float* scratch, int smax, int nslots, float* lstat, int t0, const Row* rows, int strided) {
    __shared__ float sm[32];
    int t = t0 + blockIdx.x, h = blockIdx.y, tid = threadIdx.x; float* sc = scratch + ((size_t)blockIdx.x * H + h) * smax;
    if (strided) nslots = min(nslots, (rows[t].pos / 16 + 2) * 16);
    float mx = -FLT_MAX; for (int i = tid; i < nslots; i += 1024) mx = fmaxf(mx, sc[i]);
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if ((tid & 31) == 0) sm[tid >> 5] = mx;
    __syncthreads();
    float m = sm[0]; for (int w = 1; w < 32; w++) m = fmaxf(m, sm[w]);
    __syncthreads();
    float l = 0;
    for (int i = tid; i < nslots; i += 1024) { float x = sc[i], P = x == -FLT_MAX ? 0.0f : expf(x - m); l += P; sc[i] = bf16r(P); }
    l = wsum(l); if ((tid & 31) == 0) sm[tid >> 5] = l;
    __syncthreads();
    if (tid == 0) { float ls = 0; for (int w = 0; w < 32; w++) ls += sm[w]; lstat[((size_t)t * H + h) * 2] = ls; lstat[((size_t)t * H + h) * 2 + 1] = m + logf(ls); }
}
// One block per (token, KV head, group of 16 dims); its NW warps share the key groups.
// strided = 1 (full-attention layers): warp w takes groups w, w+NW, ... so the order of additions does not depend on
// how large the grid is; a longer context only appends groups whose probabilities are exactly zero.
template <int NW> __global__ void k_attn_pv(const float* P, int smax, const uint8_t* Vt, int ngroups, const float* lstat, const float* proj, float* y, int strided, int t0, int kv8, const Row* rows, size_t kvstride) {
    __shared__ float sh[NW][32][4];
    int dg = blockIdx.x, tj = blockIdx.y, t = t0 + tj / HKV, j = tj % HKV, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, qq = lane & 3, g = lane >> 2;
    Vt += (size_t)rows[t].seq * kvstride;
    const int gmax = strided ? rows[t].pos / 16 : ngroups;   // full attention: groups past the row's own position hold nothing (and may have no memory)
    int kseg = (ngroups + NW - 1) / NW, g0 = warp * kseg;
    const float* pr = g < 3 ? P + ((size_t)(t - t0) * H + 3 * j + g) * smax : nullptr;  // column g = query head 3j + g
    float d[4] = {0, 0, 0, 0};
    for (int it = 0; it < kseg; it += 3) {
        uint4 w[3]; uint32_t b0[3], b1[3];
        if (strided && warp + NW * it > gmax) break;      // only zeros from here on: they would change nothing
        for (int u = 0; u < 3; u++) {
            int G = strided ? warp + NW * (it + u) : g0 + it + u; bool live = it + u < kseg && G < ngroups && G <= gmax;
            w[u] = !live ? make_uint4(0, 0, 0, 0) : kv8 ? tile8_to_bf16(*(const uint64_t*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 256) + lane * 8))
                                                          : *(const uint4*)(Vt + ((((size_t)G * HKV + j) * 8 + dg) * 512) + lane * 16);
            b0[u] = b1[u] = 0;
            if (live && pr) { const float2* x = (const float2*)(pr + G * 16 + 2 * qq); float2 lo = x[0], hi = x[4]; b0[u] = pack_bf16(lo.x, lo.y); b1[u] = pack_bf16(hi.x, hi.y); }
        }
        for (int u = 0; u < 3; u++) mma_bf16(d, w[u].x, w[u].y, w[u].z, w[u].w, b0[u], b1[u]);
    }
    for (int i = 0; i < 4; i++) sh[warp][lane][i] = d[i];
    __syncthreads();
    if (warp) return;
    const float* pj = proj + (size_t)t * QKVR;
    for (int i = 0; i < 4; i++) {                             // this lane holds dims g, g+8 of the group and columns 2qq, 2qq+1
        int col = 2 * qq + (i & 1), dim = dg * 16 + g + (i >> 1) * 8, h = 3 * j + col;
        if (col >= 3) continue;
        float v = 0; for (int w2 = 0; w2 < NW; w2++) v += sh[w2][lane][i];
        float ls = lstat[((size_t)t * H + h) * 2], lse = lstat[((size_t)t * H + h) * 2 + 1];
        y[(size_t)t * (H * DH) + h * DH + dim] = bf16r((v / ls) * sigmoidf(pj[R_GATE + h * DH + dim]) * sigmoidf(lse - pj[R_SINK + h]));
    }
}

// out[t][r] = W[r] . x[t] for T <= 16 tokens, W bf16 [R][C]. One block = 16 weight rows; its 8 warps each take
// one eighth of the columns (short load chains), partial sums meet in shared memory. Rows < r1 use x0,
// rows < r2 use x1, the rest x2; blocks never straddle a range. Inputs x must already be bf16-valued.
template <int NCG> __global__ void k_gemm_bf16(const uint16_t* W, int C, int T, const float* x0, const float* x1, const float* x2, int xstride, int r1, int r2,
                            float* out, int R, int round) {
    __shared__ float sh[16][32][4];
    int n0 = (r1 + 15) / 16, n1 = (r2 - r1 + 15) / 16, blk = blockIdx.x, r0, rend; const float* xb;
    if (blk < n0) { r0 = blk * 16; rend = r1; xb = x0; }
    else if (blk < n0 + n1) { r0 = r1 + (blk - n0) * 16; rend = r2; xb = x1; }
    else { r0 = r2 + (blk - n0 - n1) * 16; rend = R; xb = x2; }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = lane & 3, g = lane >> 2;
    int KB = C / 16, kseg = KB / 16, kbase = warp * kseg;                   // 16 warps share the columns
    const uint4* wt = (const uint4*)((const uint8_t*)W + ((size_t)blk * KB) * 512 + lane * 16);
    float d[NCG][4]; for (int cg = 0; cg < NCG; cg++) for (int i = 0; i < 4; i++) d[cg][i] = 0;
    // A GPU thread pays roughly 0.1 microsecond per load instruction, cached or not, so the loop is built
    // around few, wide loads: 16 bytes of weights and two 8-byte activation pairs per slice.
    for (int it = 0; it < kseg; it += 3) {
        uint4 w[3]; uint32_t b0[NCG][3], b1[NCG][3];
        for (int u = 0; u < 3; u++) {
            int kb = kbase + it + u, k0 = kb * 16; bool live = it + u < kseg;
            w[u] = live ? wt[(size_t)kb * 32] : make_uint4(0, 0, 0, 0);
            for (int cg = 0; cg < NCG; cg++) {
                int t = cg * 8 + g; b0[cg][u] = b1[cg][u] = 0;
                if (t < T && live) { const float2* x = (const float2*)(xb + (size_t)t * xstride + k0 + 2 * q); float2 lo = x[0], hi = x[4]; b0[cg][u] = pack_bf16(lo.x, lo.y); b1[cg][u] = pack_bf16(hi.x, hi.y); }
            }
        }
        for (int u = 0; u < 3; u++) for (int cg = 0; cg < NCG; cg++) mma_bf16(d[cg], w[u].x, w[u].y, w[u].z, w[u].w, b0[cg][u], b1[cg][u]);
    }
    for (int cg = 0; cg < NCG; cg++) {                       // one column group (8 tokens) at a time: keeps the shared buffer small
        for (int i = 0; i < 4; i++) sh[warp][lane][i] = d[cg][i];
        __syncthreads();
        if (warp == 0) for (int i = 0; i < 4; i++) {
            int t = cg * 8 + 2 * q + (i & 1), row = r0 + g + (i >> 1) * 8;
            if (t >= T || row >= rend) continue;
            float v = 0; for (int w2 = 0; w2 < 16; w2++) v += sh[w2][lane][i];
            out[(size_t)t * R + row] = round ? bf16r(v) : v;
        }
        __syncthreads();
    }
}

// Activations for k_gemm_bf16p, packed once per GEMM instead of by every thread at every use: the bf16 bits of a
// row in the order the tensor-core operand takes them. For each 16-value slice and q = 0..3: values 2q, 2q+1, 8+2q,
// 9+2q, 8 bytes, which is what one thread loads for one slice.
__global__ void k_pack_bf16(const float* x, int xstride, int C, uint2* out) {
    int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= C / 4) return;
    const float* r = x + (size_t)t * xstride + (i >> 2) * 16 + 2 * (i & 3);
    float2 lo = *(const float2*)r, hi = *(const float2*)(r + 8);
    out[(size_t)t * (C / 4) + i] = make_uint2(pack_bf16(lo.x, lo.y), pack_bf16(hi.x, hi.y));
}
// k_gemm_bf16 for bigger batches. Same operations in the same order, so the same results bit for bit; the
// activations come packed (k_pack_bf16), one 8-byte load per slice where the plain kernel does two loads and a
// conversion. With several column groups the plain kernel spends most of its time on exactly that.
// U: slices per loop iteration (fewer for many column groups: a block of 512 threads has 65,536 registers in all).
template <int NCG, int U> __global__ void k_gemm_bf16p(const uint16_t* W, int C, int T, const uint2* x0, const uint2* x1, const uint2* x2, int r1, int r2,
                            float* out, int R, int round) {
    __shared__ float sh[16][32][4];
    int n0 = (r1 + 15) / 16, n1 = (r2 - r1 + 15) / 16, blk = blockIdx.x, r0, rend; const uint2* xb;
    if (blk < n0) { r0 = blk * 16; rend = r1; xb = x0; }
    else if (blk < n0 + n1) { r0 = r1 + (blk - n0) * 16; rend = r2; xb = x1; }
    else { r0 = r2 + (blk - n0 - n1) * 16; rend = R; xb = x2; }
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = lane & 3, g = lane >> 2;
    int KB = C / 16, kseg = KB / 16, kbase = warp * kseg, XS = C / 4;
    const uint4* wt = (const uint4*)((const uint8_t*)W + ((size_t)blk * KB) * 512 + lane * 16);
    float d[NCG][4]; for (int cg = 0; cg < NCG; cg++) for (int i = 0; i < 4; i++) d[cg][i] = 0;
    for (int it = 0; it < kseg; it += U) {
        uint4 w[U]; uint2 b[NCG][U];
        for (int u = 0; u < U; u++) {
            int kb = kbase + it + u; bool live = it + u < kseg;
            w[u] = live ? wt[(size_t)kb * 32] : make_uint4(0, 0, 0, 0);
            for (int cg = 0; cg < NCG; cg++) {
                int t = cg * 8 + g;
                b[cg][u] = (t < T && live) ? xb[(size_t)t * XS + kb * 4 + q] : make_uint2(0, 0);
            }
        }
        for (int u = 0; u < U && it + u < kseg; u++)     // (a slice past the end is skipped, not multiplied by zero: exactly the plain kernel's operations)
            for (int cg = 0; cg < NCG; cg++) mma_bf16(d[cg], w[u].x, w[u].y, w[u].z, w[u].w, b[cg][u].x, b[cg][u].y);
    }
    for (int cg = 0; cg < NCG; cg++) {
        for (int i = 0; i < 4; i++) sh[warp][lane][i] = d[cg][i];
        __syncthreads();
        if (warp == 0) for (int i = 0; i < 4; i++) {
            int t = cg * 8 + 2 * q + (i & 1), row = r0 + g + (i >> 1) * 8;
            if (t >= T || row >= rend) continue;
            float v = 0; for (int w2 = 0; w2 < 16; w2++) v += sh[w2][lane][i];
            out[(size_t)t * R + row] = round ? bf16r(v) : v;
        }
        __syncthreads();
    }
}

// Activations for k_gemm_bf16_bulk: slice-major, two column groups per 16 bytes. Entry (kb, pair, lane) holds, for
// the thread at `lane` (g = lane / 4, q = lane % 4), the tensor-core B operand of column groups 2 pair and 2 pair + 1
// for slice kb: what that thread needs for two operations, in one load. Tokens >= T are zeros.
__global__ void k_pack_bf16s(const float* x, int xstride, int C, int T, int NP, uint4* out) {
    int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= (C / 16) * NP * 32) return;
    int lane = e & 31, pair = (e >> 5) % NP, kb = (e >> 5) / NP, g = lane >> 2, q = lane & 3; uint32_t v[4] = {0, 0, 0, 0};
    for (int h = 0; h < 2; h++) {
        int t = (2 * pair + h) * 8 + g; if (t >= T) continue;
        const float* r = x + (size_t)t * xstride + kb * 16 + 2 * q; float2 lo = *(const float2*)r, hi = *(const float2*)(r + 8);
        v[2 * h] = pack_bf16(lo.x, lo.y); v[2 * h + 1] = pack_bf16(hi.x, hi.y);
    }
    out[e] = make_uint4(v[0], v[1], v[2], v[3]);
}
// k_gemm_bf16p for prompt-sized batches (33 to 64 tokens per launch). What limits these kernels is the number of
// memory instructions, not the arithmetic: the packed kernel loads 8 bytes of activations per tensor-core operation,
// and its 16 warps per tile read all tokens' activations at once (295 KB per block at 64 tokens, out of the
// second-level cache). Here one load serves two operations (k_pack_bf16s), and a warp walks the slices in order, so
// the warps of a core work in a window of a few KB of activations.
// A block is BT tiles of 16 weight rows; SP warps per tile each take every SP-th column segment.
// Same results bit for bit: each output is still the sum, in segment order, of sixteen partial sums, and each
// partial sum is the same chain of tensor-core operations over the same operands.
template <int NCG, int BT, int SP> __global__ void k_gemm_bf16_bulk(const uint16_t* W, int C, int T, const uint4* x0, const uint4* x1, const uint4* x2, int r1, int r2,
                            float* out, int R, int round) {
    __shared__ float ex[BT * SP * 32 * 4];                // one column group's partial sums of the round, to be added in segment order
    const int NP = (NCG + 1) / 2;
    int n0 = (r1 + 15) / 16, n1 = (r2 - r1 + 15) / 16, n2 = (R - r2 + 15) / 16;       // 16-row tiles per input range
    int b0n = (n0 + BT - 1) / BT, b1n = (n1 + BT - 1) / BT;                           // blocks per range: a block never straddles two
    int blk = blockIdx.x, tile0, ntile, r0, gt0, rend; const uint4* xb;
    if (blk < b0n) { tile0 = blk * BT; ntile = n0; r0 = 0; rend = r1; xb = x0; gt0 = 0; }
    else if (blk < b0n + b1n) { tile0 = (blk - b0n) * BT; ntile = n1; r0 = r1; rend = r2; xb = x1; gt0 = n0; }
    else { tile0 = (blk - b0n - b1n) * BT; ntile = n2; r0 = r2; rend = R; xb = x2; gt0 = n0 + n1; }
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, q = lane & 3, g = lane >> 2, bt = warp / SP, j = warp % SP;
    int KB = C / 16, kseg = KB / 16, tile = tile0 + bt;
    const bool work = tile < ntile;
    const uint4* wt = (const uint4*)((const uint8_t*)W + ((size_t)(gt0 + tile) * KB) * 512 + lane * 16);
    float tot[NCG][4]; for (int cg = 0; cg < NCG; cg++) for (int i = 0; i < 4; i++) tot[cg][i] = 0;
    for (int r = 0; r < 16 / SP; r++) {
        float d[2 * NP][4]; for (int cg = 0; cg < 2 * NP; cg++) for (int i = 0; i < 4; i++) d[cg][i] = 0;
        if (work) {
            const int kb0 = (r * SP + j) * kseg;
            for (int u = 0; u < kseg; u++) {
                uint4 w = wt[(size_t)(kb0 + u) * 32]; const uint4* xs = xb + (size_t)(kb0 + u) * NP * 32 + lane;
                for (int p = 0; p < NP; p++) {
                    uint4 b = xs[p * 32];
                    mma_bf16(d[2 * p], w.x, w.y, w.z, w.w, b.x, b.y);
                    if (2 * p + 1 < NCG) mma_bf16(d[2 * p + 1], w.x, w.y, w.z, w.w, b.z, b.w);
                }
            }
        }
        if (SP == 1) { for (int cg = 0; cg < NCG; cg++) for (int i = 0; i < 4; i++) tot[cg][i] += d[cg][i]; }
        else for (int cg = 0; cg < NCG; cg++) {          // the round's SP partial sums meet in warp j = 0 of their tile
            for (int i = 0; i < 4; i++) ex[(warp * 32 + lane) * 4 + i] = d[cg][i];
            __syncthreads();
            if (j == 0) for (int jj = 0; jj < SP; jj++) for (int i = 0; i < 4; i++) tot[cg][i] += ex[((warp + jj) * 32 + lane) * 4 + i];
            __syncthreads();
        }
    }
    if (work && j == 0) for (int cg = 0; cg < NCG; cg++) for (int i = 0; i < 4; i++) {
        int t = cg * 8 + 2 * q + (i & 1), row = r0 + tile * 16 + g + (i >> 1) * 8;
        if (t >= T || row >= rend) continue;
        out[(size_t)t * R + row] = round ? bf16r(tot[cg][i]) : tot[cg][i];
    }
}

// MXFP8 activation quantiser emitting E4M3 codes and one exponent byte per 32 values (6.3).
__device__ __forceinline__ void q8_block(const float* x, uint8_t* code, uint8_t* sc) {
    float bm = 0; for (int i = 0; i < 32; i++) bm = fmaxf(bm, fabsf(x[i]));
    float raw = bm / 448.0f; uint32_t bits = __float_as_uint(raw);
    int ex = raw <= 5.877471754111438e-39f ? 0 : (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
    if (ex > 254) ex = 254;
    float inv = scalbnf(1.0f, 127 - ex);
    for (int i = 0; i < 32; i++) { float v = e4m3_rn(fminf(fabsf(x[i]) * inv, 448.0f)); code[i] = e4m3_code(v) | (signbit(x[i]) ? 0x80 : 0); }
    *sc = (uint8_t)ex;
}
// Fast MXFP8 activation quantiser built on Blackwell's native conversion instruction. One thread per 4 values
// (one 16-byte load, one 4-byte store); the 8 threads of a 32-value block agree on the block maximum with three
// warp shuffles. Same result as k_quant8c. Grid (n/128, rows) with 32 threads: a 9,216-wide row would not fit one block.
__device__ __forceinline__ uint32_t cvt_e4m3x2(float hi, float lo) {       // two floats -> two E4M3 bytes (round to nearest, saturating)
    uint32_t r; asm volatile("{\n .reg .b16 t;\n cvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\n cvt.u32.u16 %0, t;\n}" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__global__ void k_quant8f(const float* x, int n, uint8_t* code_out, uint8_t* scale_out, int xstride) {
    int row = blockIdx.y, q = blockIdx.x * 32 + threadIdx.x;
    float4 v = *(const float4*)(x + (size_t)row * xstride + q * 4);
    float bm = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1)); bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2)); bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 4));
    float raw = bm / 448.0f; uint32_t bits = __float_as_uint(raw);
    int ex = raw <= 5.877471754111438e-39f ? 0 : (int)(((bits + 0x007FFFFFu) & 0x7F800000u) >> 23);
    if (ex > 254) ex = 254;
    float inv = scalbnf(1.0f, 127 - ex);
    uint32_t w = cvt_e4m3x2(v.y * inv, v.x * inv) | (cvt_e4m3x2(v.w * inv, v.z * inv) << 16);
    *(uint32_t*)(code_out + (size_t)row * n + q * 4) = w;
    if ((q & 7) == 0) scale_out[(size_t)row * (n / 32) + (q >> 3)] = (uint8_t)ex;
}
// ---- NVFP4 activation quantiser on Blackwell's native conversion instructions (same rule and same results
// as q4_block, 6.2). One thread per 4 values; the 4 threads of a 16-value block cooperate through shuffles.
__device__ __forceinline__ uint32_t cvt_e2m1x2(float hi, float lo) {       // two floats -> one byte holding two E2M1 codes
    uint32_t r; asm volatile("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u32.u8 %0, t;\n}" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
// v[4] = this thread's values, amax = row maximum. Returns the 16 packed code bits of this thread; *sc = block scale code.
__device__ __forceinline__ uint32_t q4_lane(const float* v, float amax, uint8_t* sc) {
    float bm = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
    bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 1)); bm = fmaxf(bm, __shfl_xor_sync(0xffffffffu, bm, 2));
    float enc = fminf(1536.0f / amax, FLT_MAX), dec = 1.0f / enc, outer = amax * (1.0f / 1536.0f);
    uint32_t code[2]; uint8_t scode[2]; float err[2];
    for (int cnd = 0; cnd < 2; cnd++) {
        uint32_t sb = cvt_e4m3x2(0.0f, fminf(bm * (enc * (cnd ? 0.25f : (1.0f / 6.0f))), 448.0f)) & 0xFF;      // block scale, rounded to E4M3
        float s = e4m3((uint8_t)sb), inv = fminf(1.0f / (s * dec), FLT_MAX);
        uint32_t c = cvt_e2m1x2(v[1] * inv, v[0] * inv) | (cvt_e2m1x2(v[3] * inv, v[2] * inv) << 8);
        // value * scale * outer, multiplied in this order: the choice between the two candidates hinges on last-bit differences
        float e0 = fabsf(e2m1x2(c) * 0.5f * s * outer - v[0]), e1 = fabsf(e2m1x2(c >> 4) * 0.5f * s * outer - v[1]);
        float e2 = fabsf(e2m1x2(c >> 8) * 0.5f * s * outer - v[2]), e3 = fabsf(e2m1x2(c >> 12) * 0.5f * s * outer - v[3]);
        float e = (e0 + e1) + (e2 + e3);                                    // pairwise, like the reference's 16 -> 8 -> 4 -> 2 -> 1
        e += __shfl_xor_sync(0xffffffffu, e, 1); e += __shfl_xor_sync(0xffffffffu, e, 2);
        code[cnd] = c; scode[cnd] = (uint8_t)sb; err[cnd] = e;
    }
    int pick = err[1] < err[0];
    *sc = amax > 0 ? scode[pick] : 0;
    return amax > 0 ? code[pick] : 0u;
}
// Row maximum over a block of n/4 threads (each holding the max of its 4 values).
__device__ __forceinline__ float row_amax(float m, float* sh) {
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) sh[threadIdx.x >> 5] = m;
    __syncthreads();
    float a = sh[0]; for (int w = 1; w < (int)(blockDim.x >> 5); w++) a = fmaxf(a, sh[w]);
    return a;
}
__global__ void k_quant4f(const float* x, int n, float* amax_out, uint8_t* code_out, uint8_t* scale_out, PF pf) {
    __shared__ float sh[32];
    PF_X(pf)
    int row = blockIdx.x, q = threadIdx.x; float4 f = *(const float4*)(x + (size_t)row * n + q * 4); float v[4] = {f.x, f.y, f.z, f.w};
    float amax = row_amax(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))), sh);
    if (q == 0) amax_out[row] = amax;
    uint8_t sc; uint32_t c = q4_lane(v, amax, &sc);
    *(uint16_t*)(code_out + (size_t)row * (n / 2) + q * 2) = (uint16_t)c;
    if ((q & 3) == 0) scale_out[(size_t)row * (n / 16) + (q >> 2)] = sc;
}
__global__ void k_swiglu_quant4f(const float* gu, const float* gamma, float* amax_out, uint8_t* code_out, uint8_t* scale_out, PF pf) {
    __shared__ float sh[32];
    PF_X(pf)
    int n = blockIdx.x, q = threadIdx.x; float gm = gamma[n], v[4];
    const float4* g4 = (const float4*)(gu + (size_t)n * D + q * 8); float4 a = g4[0], b = g4[1];      // (gate, up) pairs, interleaved
    const float gs[4] = {a.x, a.z, b.x, b.z}, us[4] = {a.y, a.w, b.y, b.w};
    for (int i = 0; i < 4; i++) v[i] = bf16r(((4.0f * tanhf(gs[i] * 0.25f) * sigmoidf(gs[i])) * (25.0f * tanhf(us[i] * 0.04f))) * gm);
    float amax = row_amax(fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3]))), sh);
    if (q == 0) amax_out[n] = amax;
    uint8_t sc; uint32_t c = q4_lane(v, amax, &sc);
    *(uint16_t*)(code_out + (size_t)n * (I2 / 2) + q * 2) = (uint16_t)c;
    if ((q & 3) == 0) scale_out[(size_t)n * (I2 / 16) + (q >> 2)] = sc;
}
__global__ void k_quant8c(const float* x, int n, uint8_t* code_out, uint8_t* scale_out, int xstride) {
    int row = blockIdx.x, b = threadIdx.x; if (b >= n / 32) return;
    float xv[32]; for (int i = 0; i < 32; i++) xv[i] = x[(size_t)row * xstride + b * 32 + i];
    q8_block(xv, code_out + (size_t)row * n + b * 32, scale_out + (size_t)row * (n / 32) + b);
}
__global__ void k_swiglu_quant8(const float* gu, const float* gamma, uint8_t* code_out, uint8_t* scale_out) {
    int n = blockIdx.x, b = threadIdx.x; if (b >= I2 / 32) return;
    float xv[32], gm = gamma[n];
    for (int i = 0; i < 32; i++) {
        int j = b * 32 + i; float g = gu[(size_t)n * D + 2 * j], u = gu[(size_t)n * D + 2 * j + 1];
        xv[i] = bf16r(((4.0f * tanhf(g * 0.25f) * sigmoidf(g)) * (25.0f * tanhf(u * 0.04f))) * gm);
    }
    q8_block(xv, code_out + (size_t)n * I2 + b * 32, scale_out + (size_t)n * (I2 / 32) + b);
}
// Expert GEMM for the MXFP8 layers on the tensor cores; same structure as k_expert4_mma.
__global__ void k_expert8_mma(const uint8_t* W, const uint8_t* S, const int* tasks, const int* ntasks, const uint8_t* xcode, const uint8_t* xscale,
                              int x_per_slot, int R, int C, float* out) {
    if ((int)blockIdx.y >= *ntasks) return;
    const int* task = tasks + blockIdx.y * TASKW; int e = task[0], ncols = task[1];
    int lane = threadIdx.x & 31, i = lane & 3, j = lane >> 2, rgi = blockIdx.x * 8 + (threadIdx.x >> 5), r0 = rgi * 16, nblk = C / 32;
    const uint64_t* wt = (const uint64_t*)(W + (((size_t)e * (R / 16) + rgi) * nblk) * 512 + lane * 16);
    const uint8_t* st = S + (((size_t)e * (R / 16) + rgi) * nblk) * 16 + j * 2 + (lane & 1);
    int nj = j < ncols ? task[2 + j] : -1, xj = nj < 0 ? 0 : (x_per_slot ? nj : nj / TOPK);
    const uint8_t* xc = xcode + (size_t)xj * C + 4 * i; const uint8_t* xs = xscale + (size_t)xj * nblk;
    float d[4] = {0, 0, 0, 0};
    for (int kb0 = 0; kb0 < nblk; kb0 += 6) {
        uint64_t w0[6], w1[6]; uint32_t b0[6], b1[6], sa[6], sb[6];
        for (int u = 0; u < 6; u++) {
            int kb = kb0 + u;
            w0[u] = wt[(size_t)kb * 64]; w1[u] = wt[(size_t)kb * 64 + 1]; sa[u] = st[(size_t)kb * 16]; b0[u] = b1[u] = 0; sb[u] = 127;
            if (nj >= 0) { b0[u] = *(const uint32_t*)(xc + kb * 32); b1[u] = *(const uint32_t*)(xc + kb * 32 + 16); sb[u] = xs[kb]; }
        }
        for (int u = 0; u < 6; u++) mma_fp8(d, (uint32_t)w0[u], (uint32_t)(w0[u] >> 32), (uint32_t)w1[u], (uint32_t)(w1[u] >> 32), b0[u], b1[u], sa[u], sb[u]);
    }
    for (int q = 0; q < 2; q++) {
        int c = 2 * i + q; if (c >= ncols) continue;
        int n = task[2 + c];
        out[(size_t)n * R + r0 + j] = bf16r(d[q]); out[(size_t)n * R + r0 + j + 8] = bf16r(d[2 + q]);
    }
}

// =====================================================================================================
// DSpark drafter (docs/ARCHITECTURE.md section 7; reference: ref/dspark_ref.py).
// A block of 8 slots (the anchor token + 7 mask slots) runs through 3 dense layers. Its attention reads a
// separate key/value cache built from the target model's hidden-state taps at every earlier position.
// =====================================================================================================
const int DB = 8, DQ = 3072, DKV = 1024, DI = 9216, MR = 256;      // block size, q width, k/v width, MLP width, Markov rank
__device__ __forceinline__ void rope_pair(float a, float b, int d, int pos, float* out) {   // a = x[d], b = partner
    int m = d & 63; float ang = (float)pos * expf(-(float)m * 0.21586735246819397f);     // theta^(-2m/128), theta = 1e6
    float sn, cs; sincosf(ang, &sn, &cs);
    *out = bf16r(d < 64 ? a * cs - b * sn : a * cs + b * sn);
}
// Mix the 4 taps with learned weights and normalise: one row per consumer (3 draft layers + Markov head).
__global__ void k_dctx(const float* feats, const float* alpha, const float* norm, float* out) {
    __shared__ float sh[NT / 32]; __shared__ float sh2[32];
    int t = blockIdx.x, r = blockIdx.y, tid = threadIdx.x; float v[DPT], ss = 0;
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; float a = 0; for (int tap = 0; tap < 4; tap++) a = fmaf(alpha[r * 4 + tap], feats[((size_t)tap * TMAX + t) * D + c], a); v[k] = a; ss += a * a; }
    breduce<1>(&ss, sh, sh2);
    float inv = rsqrtf(ss / D + EPS);
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; out[((size_t)t * 4 + r) * D + c] = bf16r(v[k] * inv * norm[(size_t)r * D + c]); }
}
// Per-head RMS norm (+ RoPE) of one 128-wide head held in the first 4 warps. Returns this thread's value.
__device__ __forceinline__ float head_norm_rope(const float* x, const float* w, int d, int pos, float* sh, float* sh2) {
    float a = x[d], ss = a * a; breduce<1>(&ss, sh, sh2);
    float inv = rsqrtf(ss / DH + EPS), b = x[d ^ 64], out;
    rope_pair(bf16r(a * inv * w[d]), bf16r(b * inv * w[d ^ 64]), d, pos, &out);
    return out;
}
// Context keys/values of the 3 draft layers for the T positions just processed by the target.
__global__ void k_dctx_write(const float* kv, const float* kcn, const Row* rows, uint16_t* K0, uint16_t* V0_, uint16_t* K1, uint16_t* V1, uint16_t* K2, uint16_t* V2, size_t lstride, size_t gstride) {
    __shared__ float sh[4]; __shared__ float sh2[32];
    int t = blockIdx.x, layer = blockIdx.y / HKV, j = blockIdx.y % HKV, d = threadIdx.x, p = rows[t].pos;
    const float* x = kv + (size_t)t * 6144 + layer * 2048 + j * DH;
    float k = head_norm_rope(x, kcn + layer * DH, d, p, sh, sh2), v = x[DKV + d];
    uint16_t* K = layer == 0 ? K0 : (layer == 1 ? K1 : K2); uint16_t* V = layer == 0 ? V0_ : (layer == 1 ? V1 : V2);
    size_t so = (size_t)rows[t].seq * (layer == 1 ? gstride : lstride);
    K = (uint16_t*)((uint8_t*)K + so); V = (uint16_t*)((uint8_t*)V + so);
    size_t slot = layer == 1 ? (size_t)p : (size_t)(p % RING);
    K[(slot * HKV + j) * DH + d] = bf16bits(k); V[(slot * HKV + j) * DH + d] = bf16bits(v);
}
// Slot 0 = embedding of the anchor token, slots 1..7 = the mask vector; also the first layer's input norm.
__global__ void k_dinit(const uint16_t* embed, const float* embed_norm, const int* anchor, const float* mask, const float* in_norm, float* resid, float* h) {
    __shared__ float sh[(NT / 32) * 2]; __shared__ float sh2[32];
    int bs = blockIdx.x, slot = bs % DB, tid = threadIdx.x; float v[DPT], r[2] = {0, 0};      // bs = conversation * DB + slot
    int a = anchor[bs / DB];
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; v[k] = slot ? 0.0f : bf16w(embed[(size_t)a * D + c]); r[0] += v[k] * v[k]; }
    if (slot == 0) { breduce<1>(r, sh, sh2); float inv = rsqrtf(r[0] / D + EPS); for (int k = 0; k < DPT; k++) v[k] = bf16r(v[k] * inv * embed_norm[tid + NT * k]); }
    else for (int k = 0; k < DPT; k++) v[k] = mask[tid + NT * k];
    for (int k = 0; k < DPT; k++) r[1] += v[k] * v[k];
    breduce<1>(r + 1, sh, sh2);
    float inv = rsqrtf(r[1] / D + EPS);
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; resid[(size_t)bs * D + c] = v[k]; h[(size_t)bs * D + c] = bf16r(v[k] * inv * in_norm[c]); }
}
// q and k of the block: per-head norm + RoPE; block keys/values go into the draft cache at positions L..L+7.
// Lp / seqs: per conversation of the batch, the anchor's position and the conversation's slot; stride: bytes of one slot's cache.
__global__ void k_dqk(const float* qkv, const float* qn, const float* kn, const int* Lp, const int* seqs, size_t stride, float* qd, uint16_t* K, uint16_t* V, int is_global) {
    __shared__ float sh[4]; __shared__ float sh2[32];
    int bs = blockIdx.x, b = bs / DB, slot = bs % DB, y = blockIdx.y, d = threadIdx.x, p = Lp[b] + slot; const float* row = qkv + (size_t)bs * (DQ + 2 * DKV);
    if (y < H) { qd[((size_t)bs * H + y) * DH + d] = head_norm_rope(row + y * DH, qn, d, p, sh, sh2); return; }
    size_t so = (size_t)seqs[b] * stride; K = (uint16_t*)((uint8_t*)K + so); V = (uint16_t*)((uint8_t*)V + so);
    int j = y - H; float k = head_norm_rope(row + DQ + j * DH, kn, d, p, sh, sh2);
    size_t s2 = is_global ? (size_t)p : (size_t)(p % RING);
    K[(s2 * HKV + j) * DH + d] = bf16bits(k); V[(s2 * HKV + j) * DH + d] = bf16bits(row[DQ + DKV + j * DH + d]);
}
// Draft attention scores: every slot sees the same keys, lo .. L+7 (the whole block, both directions).
__global__ void k_dattn_logits(const float* qd, const uint16_t* Kc, const int* Lp, const int* seqs, size_t stride, int is_global, float* scratch, int smax) {
    __shared__ float q[3][DH];
    int tj = blockIdx.y, t = tj / HKV, j = tj % HKV, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, pe = Lp[t / DB] + DB - 1;
    Kc = (const uint16_t*)((const uint8_t*)Kc + (size_t)seqs[t / DB] * stride);
    int lo = is_global ? 0 : max(0, pe - (EXT + DB - 1)), S = pe - lo + 1;
    if ((int)blockIdx.x * 32 >= S) return;
    for (int i = tid; i < 3 * DH; i += 256) q[i / DH][i % DH] = qd[((size_t)t * H + 3 * j) * DH + i];
    __syncthreads();
    for (int it = 0; it < 4; it++) {
        int idx = blockIdx.x * 32 + it * 8 + warp; if (idx >= S) continue;
        int s = lo + idx; size_t slot = is_global ? (size_t)s : (size_t)(s % RING);
        uint64_t u = *(const uint64_t*)(Kc + (slot * HKV + j) * DH + 4 * lane);
        float k0 = bf16w((uint16_t)u), k1 = bf16w((uint16_t)(u >> 16)), k2 = bf16w((uint16_t)(u >> 32)), k3 = bf16w((uint16_t)(u >> 48));
        for (int g = 0; g < 3; g++) {
            const float* qq = q[g] + 4 * lane;
            float dd = wsum(fmaf(qq[3], k3, fmaf(qq[2], k2, fmaf(qq[1], k1, qq[0] * k0))));
            if (lane == 0) scratch[((size_t)t * H + 3 * j + g) * smax + idx] = dd * 0.08838834764831845f;
        }
    }
}
__global__ void k_dattn_out(const float* scratch, int smax, const uint16_t* Vc, const int* Lp, const int* seqs, size_t stride, int is_global, float* o) {
    __shared__ float po[32][DH]; __shared__ float pl[32]; __shared__ float pm[32];
    int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, g = tid & 31, cls = tid >> 5, j = h / 3, pe = Lp[t / DB] + DB - 1;
    Vc = (const uint16_t*)((const uint8_t*)Vc + (size_t)seqs[t / DB] * stride);
    int lo = is_global ? 0 : max(0, pe - (EXT + DB - 1)), S = pe - lo + 1;
    const float* sc = scratch + ((size_t)t * H + h) * smax;
    float mx = -FLT_MAX; for (int i = tid; i < S; i += 1024) mx = fmaxf(mx, sc[i]);
    for (int of = 16; of > 0; of >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, of));
    if (g == 0) pm[cls] = mx;
    __syncthreads();
    float m = pm[0]; for (int w = 1; w < 32; w++) m = fmaxf(m, pm[w]);
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, l = 0;
    for (int i = cls; i < S; i += 32) {
        int s = lo + i; size_t slot = is_global ? (size_t)s : (size_t)(s % RING);
        float P = expf(sc[i] - m); l += P;
        uint64_t u = *(const uint64_t*)(Vc + (slot * HKV + j) * DH + 4 * g);
        a0 = fmaf(P, bf16w((uint16_t)u), a0); a1 = fmaf(P, bf16w((uint16_t)(u >> 16)), a1);
        a2 = fmaf(P, bf16w((uint16_t)(u >> 32)), a2); a3 = fmaf(P, bf16w((uint16_t)(u >> 48)), a3);
    }
    po[cls][4 * g] = a0; po[cls][4 * g + 1] = a1; po[cls][4 * g + 2] = a2; po[cls][4 * g + 3] = a3;
    if (g == 0) pl[cls] = l;
    __syncthreads();
    if (tid >= DH) return;
    float ov = 0, ls = 0; for (int w = 0; w < 32; w++) { ov += po[w][tid]; ls += pl[w]; }
    o[(size_t)t * DQ + h * DH + tid] = bf16r(ov / ls);
}
// resid += add; h = bf16(rmsnorm(resid_unrounded) * w); resid is stored rounded to bf16.
__global__ void k_dres(float* resid, const float* add, const float* w, float* h) {
    __shared__ float sh[NT / 32]; __shared__ float sh2[32];
    int slot = blockIdx.x, tid = threadIdx.x; float v[DPT], ss = 0;
    for (int k = 0; k < DPT; k++) { size_t o = (size_t)slot * D + tid + NT * k; v[k] = resid[o] + add[o]; ss += v[k] * v[k]; resid[o] = bf16r(v[k]); }
    breduce<1>(&ss, sh, sh2);
    float inv = rsqrtf(ss / D + EPS);
    for (int k = 0; k < DPT; k++) { int c = tid + NT * k; h[(size_t)slot * D + c] = bf16r(v[k] * inv * w[c]); }
}
__global__ void k_dsilu(const float* gu, float* act, int rows) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (size_t)rows * DI) return;
    size_t slot = i / DI, j = i % DI; float g = gu[slot * 2 * DI + j], u = gu[slot * 2 * DI + DI + j];
    act[i] = bf16r((g / (1.0f + expf(-g))) * u);
}
// Rows of the head are ordered draft-major: row = draft index * NB + conversation, so each draft step works on NB
// consecutive rows. Gathers the mask slots' hidden states into `hsel` in that order, and builds the input of the Markov
// head's constant part: [hidden of the mask slot | f_rnn of the conversation's last valid target position].
__global__ void k_dcin(const float* h, const float* frnn, const int* seqs, float* cin, float* hsel, int NB, int nd) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (size_t)nd * NB * 2 * D) return;
    size_t r = i / (2 * D), c = i % (2 * D); int di = (int)(r / NB), b = (int)(r % NB);
    if (c < D) { float x = h[((size_t)b * DB + 1 + di) * D + c]; cin[i] = x; hsel[r * D + c] = x; }
    else cin[i] = frnn[(size_t)seqs[b] * D + (c - D)];
}
__global__ void k_mk_reset(float* state, const int* anchor, int* prev) { int b = blockIdx.x; state[b * MR + threadIdx.x] = 0.0f; if (threadIdx.x == 0) prev[b] = anchor[b]; }
__global__ void k_mk_in(const float* state, const uint16_t* w1, const int* prev, float* x) {
    int b = blockIdx.x, d = threadIdx.x; x[b * 512 + d] = d < MR ? state[b * MR + d] : bf16w(w1[(size_t)prev[b] * MR + (d - MR)]);
}
// One step of the recurrent head: gates -> new state and the latent that biases this slot's logits.
__global__ void k_mk_gate(const float* sw, const float* C, const float* bias, float* state, float* lat) {
    int b = blockIdx.x, d = threadIdx.x; sw += b * 768; C += b * 768; state += b * MR; lat += b * MR;
    float gate = bf16r(C[d] + bias[d] + sw[d]), cand = bf16r(C[MR + d] + bias[MR + d] + sw[MR + d]), outp = bf16r(C[2 * MR + d] + bias[2 * MR + d] + sw[2 * MR + d]);
    float g = bf16r(sigmoidf(gate)), c = bf16r(tanhf(cand));
    state[d] = bf16r(bf16r(g * state[d]) + bf16r(bf16r(1.0f - g) * c));
    lat[d] = bf16r(tanhf(outp));
}
// Draft logits of one slot = softcap(base + Markov bias).
// A greedy conversation (1/T = 0 in dsamp) drafts the most likely token; its confidence is that token's probability.
// A sampling conversation drafts a token drawn from q = softmax(logits / T) (Gumbel-max), which the target then accepts
// with probability min(1, p/q): far more drafts survive than with the most likely token when several are plausible.
// The verifier needs the whole of q for a rejected position, so the logits are kept in qlog; sums are in double
// precision because exp((x - 30) / T) underflows a float at low temperature.
__device__ __forceinline__ float gumbel(uint32_t seed, uint32_t pos, uint32_t v);
const float DRAFT_SAMPLE_MIN_TEMP = 0.3f;                 // below this a sampled draft is the most likely token almost always
__global__ void k_dhead_part(const float* base, const float* mb, const float* bvec, float* pv, int* pi, float* psum, float* px, double* psumd, float* pmx,
                             const float* dsamp, const int* Lp, const int* seqs, float* qlog, int draft) {
    __shared__ float sv[32]; __shared__ int si[32]; __shared__ float ss[32]; __shared__ float sx[32]; __shared__ double sd[32]; __shared__ float sm[32];
    int part = blockIdx.x, b = blockIdx.y, tid = threadIdx.x; float best = -FLT_MAX, sum = 0, bx = 0, mx = -FLT_MAX; double dsum = 0; int bi = 0x7fffffff;      // b: conversation
    const float invT = dsamp[b * 2]; const uint32_t seed = __float_as_uint(dsamp[b * 2 + 1]) ^ 0xD1B54A35u, pos = (uint32_t)(Lp[b] + 1 + draft);
    float* ql = qlog + ((size_t)seqs[b] * (DB - 1) + draft) * V0;
    base += (size_t)b * V0; mb += (size_t)b * V0; pv += b * HEAD_PARTS; pi += b * HEAD_PARTS; psum += b * HEAD_PARTS; px += b * HEAD_PARTS; psumd += b * HEAD_PARTS; pmx += b * HEAD_PARTS;
    for (int k = 0; k < 4; k++) {
        int v = part * HEAD_SPAN + tid + 1024 * k; if (v >= V0) break;
        float x = bf16r(30.0f * tanhf(bf16r(base[v] + bf16r(mb[v] + bvec[v])) * (1.0f / 30.0f))), sc = x;
        mx = fmaxf(mx, x);
        if (invT > 0.0f) { ql[v] = x; sc = x * invT + gumbel(seed, pos, (uint32_t)v); dsum += exp((double)(x - 30.0f) * (double)invT); }
        else sum += expf(x - 30.0f);
        if (sc > best) { best = sc; bi = v; bx = x; }
    }
    sum = wsum(sum);
    for (int o = 16; o > 0; o >>= 1) { dsum += __shfl_down_sync(0xffffffffu, dsum, o); mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o)); }
    for (int o = 16; o > 0; o >>= 1) {
        float ob = __shfl_down_sync(0xffffffffu, best, o), ox = __shfl_down_sync(0xffffffffu, bx, o); int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; bx = ox; }
    }
    if ((tid & 31) == 0) { sv[tid >> 5] = best; si[tid >> 5] = bi; ss[tid >> 5] = sum; sx[tid >> 5] = bx; sd[tid >> 5] = dsum; sm[tid >> 5] = mx; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 32; w++) { sum += ss[w]; dsum += sd[w]; mx = fmaxf(mx, sm[w]); if (sv[w] > best || (sv[w] == best && si[w] < bi)) { best = sv[w]; bi = si[w]; bx = sx[w]; } }
        pv[part] = best; pi[part] = bi; psum[part] = sum; px[part] = bx; psumd[part] = dsum; pmx[part] = mx;
    }
}
// Per (conversation, draft), for sampling conversations: qz = the sum that normalises q, qit = the 1/T it was built with.
// conf_out: probability the drafter gives the token it drafted. top_out: the largest probability in its distribution,
// which says how sure it is without depending on which token was drawn - the only kind of number that may decide
// whether a sampled draft gets verified (deciding by the drawn token's own probability would bias the output).
__global__ void k_dhead_final(const float* pv, const int* pi, const float* psum, const float* px, const double* psumd, const float* pmx, const float* dsamp, const int* seqs,
                              double* qz, float* qit, int* tok_out, float* conf_out, float* top_out, int* prev, int draft) {
    int b = blockIdx.x, lane = threadIdx.x; float best = -FLT_MAX, sum = 0, bx = 0, mx = -FLT_MAX; double dsum = 0; int bi = 0x7fffffff;
    pv += b * HEAD_PARTS; pi += b * HEAD_PARTS; psum += b * HEAD_PARTS; px += b * HEAD_PARTS; psumd += b * HEAD_PARTS; pmx += b * HEAD_PARTS;
    for (int k = lane; k < HEAD_PARTS; k += 32) { float x = pv[k]; int i = pi[k]; sum += psum[k]; dsum += psumd[k]; mx = fmaxf(mx, pmx[k]); if (x > best || (x == best && i < bi)) { best = x; bi = i; bx = px[k]; } }
    sum = wsum(sum);
    for (int o = 16; o > 0; o >>= 1) { dsum += __shfl_down_sync(0xffffffffu, dsum, o); mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o)); }
    for (int o = 16; o > 0; o >>= 1) {
        float ob = __shfl_down_sync(0xffffffffu, best, o), ox = __shfl_down_sync(0xffffffffu, bx, o); int oi = __shfl_down_sync(0xffffffffu, bi, o);
        if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; bx = ox; }
    }
    if (lane == 0) {
        const float invT = dsamp[b * 2]; float conf, top; int q = seqs[b] * (DB - 1) + draft;
        if (invT > 0.0f) { qz[q] = dsum; qit[q] = invT; conf = (float)(exp((double)(bx - 30.0f) * (double)invT) / dsum); top = (float)(exp((double)(mx - 30.0f) * (double)invT) / dsum); }
        else { conf = expf(best - 30.0f) / sum; top = conf; }
        tok_out[b * (DB - 1) + draft] = bi; prev[b] = bi; conf_out[b * (DB - 1) + draft] = conf; top_out[b * (DB - 1) + draft] = top;
    }
}
// Fold step for the Markov bias matrix: M[v][c0 + t] = bf16(tmp[t][v]).
__global__ void k_fold(const float* tmp, uint16_t* M, int c0, int n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (size_t)V0 * n) return;
    size_t v = i / n, t = i % n; M[v * MR + c0 + t] = bf16bits(tmp[t * V0 + v]);
}

// =====================================================================================================
// Sampling on the GPU (Gumbel-max) plus what speculative rejection sampling needs, per row:
//   res[t] = { greedy argmax, a sample from p, a sample from p with token x_t excluded, log p(x_t) }
// p = softmax(logits / temperature). x_t = the draft token proposed for the position this row predicts
// (-1: none). The host accepts the draft with probability p(x_t); if it rejects, the "excluded" sample is
// exactly a draw from the leftover distribution, so the output distribution equals plain sampling.
// =====================================================================================================
__device__ __forceinline__ float gumbel(uint32_t seed, uint32_t pos, uint32_t v) {
    uint32_t h = seed ^ (pos * 0x9E3779B9u) ^ (v * 0x85EBCA6Bu);
    h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
    float u = fminf(((float)h + 0.5f) * 2.3283064e-10f, 0.99999994f);
    return -logf(-logf(u));
}
struct SampPart { float sum, b1, b2, lx; int i1, i2; };
__global__ void k_samp_part(const float* logits, const float* sparam, const int* excl, const Row* rows, const float* rowmax, SampPart* parts) {
    __shared__ SampPart sp[32];
    float invT = sparam[blockIdx.y * 2]; if (invT <= 0.0f) return;       // greedy row: nothing to do
    uint32_t seed = __float_as_uint(sparam[blockIdx.y * 2 + 1]);
    int t = blockIdx.y, part = blockIdx.x, tid = threadIdx.x, x = excl[t]; const float* l = logits + (size_t)t * V0;
    SampPart a; a.sum = 0; a.b1 = a.b2 = -FLT_MAX; a.i1 = a.i2 = 0x7fffffff; a.lx = -FLT_MAX; const float zmax = rowmax[t] * invT;
    for (int k = 0; k < 4; k++) {
        int v = part * HEAD_SPAN + tid + 1024 * k; if (v >= V0) break;
        float z = l[v] * invT; a.sum += expf(z - zmax);                       // relative to the row maximum: the sum is >= 1, never underflows
        float gz = z + gumbel(seed, (uint32_t)rows[t].pos, (uint32_t)v);
        if (gz > a.b1 || (gz == a.b1 && v < a.i1)) { a.b1 = gz; a.i1 = v; }      // equal scores: the lower token id, as for the argmax
        if (v == x) a.lx = z; else if (gz > a.b2) { a.b2 = gz; a.i2 = v; }
    }
    a.sum = wsum(a.sum);
    for (int o = 16; o > 0; o >>= 1) {
        float ob1 = __shfl_down_sync(0xffffffffu, a.b1, o), ob2 = __shfl_down_sync(0xffffffffu, a.b2, o), olx = __shfl_down_sync(0xffffffffu, a.lx, o);
        int oi1 = __shfl_down_sync(0xffffffffu, a.i1, o), oi2 = __shfl_down_sync(0xffffffffu, a.i2, o);
        if (ob1 > a.b1 || (ob1 == a.b1 && oi1 < a.i1)) { a.b1 = ob1; a.i1 = oi1; } if (ob2 > a.b2) { a.b2 = ob2; a.i2 = oi2; } a.lx = fmaxf(a.lx, olx);
    }
    if ((tid & 31) == 0) sp[tid >> 5] = a;
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 32; w++) { a.sum += sp[w].sum; if (sp[w].b1 > a.b1 || (sp[w].b1 == a.b1 && sp[w].i1 < a.i1)) { a.b1 = sp[w].b1; a.i1 = sp[w].i1; } if (sp[w].b2 > a.b2) { a.b2 = sp[w].b2; a.i2 = sp[w].i2; } a.lx = fmaxf(a.lx, sp[w].lx); }
        parts[t * HEAD_PARTS + part] = a;
    }
}
__global__ void k_samp_final(const SampPart* parts, const float* sparam, const int* arg, const float* rowmax, int* res, float* rowz) {
    int t = blockIdx.x, lane = threadIdx.x; float invT = sparam[t * 2];
    if (invT <= 0.0f) { if (lane == 0) { res[t * 4] = arg[t]; res[t * 4 + 1] = arg[t]; res[t * 4 + 2] = arg[t]; res[t * 4 + 3] = 0; } return; }
    SampPart a; a.sum = 0; a.b1 = a.b2 = -FLT_MAX; a.i1 = a.i2 = 0x7fffffff; a.lx = -FLT_MAX;
    for (int k = lane; k < HEAD_PARTS; k += 32) { SampPart q = parts[t * HEAD_PARTS + k]; a.sum += q.sum; if (q.b1 > a.b1 || (q.b1 == a.b1 && q.i1 < a.i1)) { a.b1 = q.b1; a.i1 = q.i1; } if (q.b2 > a.b2) { a.b2 = q.b2; a.i2 = q.i2; } a.lx = fmaxf(a.lx, q.lx); }
    a.sum = wsum(a.sum);
    for (int o = 16; o > 0; o >>= 1) {
        float ob1 = __shfl_down_sync(0xffffffffu, a.b1, o), ob2 = __shfl_down_sync(0xffffffffu, a.b2, o), olx = __shfl_down_sync(0xffffffffu, a.lx, o);
        int oi1 = __shfl_down_sync(0xffffffffu, a.i1, o), oi2 = __shfl_down_sync(0xffffffffu, a.i2, o);
        if (ob1 > a.b1 || (ob1 == a.b1 && oi1 < a.i1)) { a.b1 = ob1; a.i1 = oi1; } if (ob2 > a.b2) { a.b2 = ob2; a.i2 = oi2; } a.lx = fmaxf(a.lx, olx);
    }
    if (lane == 0) {
        float logp = a.lx == -FLT_MAX ? -FLT_MAX : fminf(a.lx - (rowmax[t] * invT + logf(a.sum)), 0.0f);
        res[t * 4] = arg[t]; res[t * 4 + 1] = a.i1; res[t * 4 + 2] = a.i2; res[t * 4 + 3] = (int)__float_as_uint(logp);
        rowz[t] = a.sum;
    }
}
// =====================================================================================================
// Two-stage output head. The full head reads the 922 MB unembedding matrix for every step (0.57 ms); only the
// winner of each row is wanted: the largest logit, or with sampling the largest of logit / T + Gumbel noise.
//   1. Approximate logits from the 8-bit copy of the matrix (the one the drafter uses) and 8-bit activations: 0.29 ms.
//   2. For every token a bound on |approximate - exact|: with u the exact row, u~ the 8-bit row, h the exact and h~
//      the 8-bit activation, |u.h - u~.h~| <= |u - u~| |h~| + |u| |h - h~| (Cauchy-Schwarz twice), plus an
//      allowance for rounding in the two sums. |u - u~| and |u| are stored per token at load; the rest is per row.
//   3. The exact logit of the approximate winner. No token whose approximate logit plus its bound stays below
//      that can win; the others are candidates, a handful as a rule.
//   4. Exact logits of the candidates' 16-row blocks, by the same operations in the same order as the full
//      product (k_gemm_rows), and the winner among them. The result is the full head's, bit for bit.
// More candidates than CAND_CAP blocks: the step reports it and the full head runs after all.
// Not for rows that need a probability (a sampled draft's acceptance, the nll command): those take the full head.
// =====================================================================================================
const int FAST_T = 32, CAND_CAP = 4096;                   // most rows of a step using this head; most candidate blocks
const float HEAD_ROUNDING = 5e-4f, HEAD_MARGIN = 1e-4f;   // rounding allowance (x |u| |h|); slack on the thresholds, in capped logits
// Per row of an MXFP8 matrix: |row - quantised row| and |row|, rounded up. One warp per row.
__global__ void k_fp8_rowerr(const uint16_t* src, const uint8_t* code, const uint8_t* scale, size_t R, int C, float* err, float* nrm) {
    size_t row = (size_t)blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5); int lane = threadIdx.x & 31; if (row >= R) return;
    double se = 0, sn = 0;
    for (int c = lane; c < C; c += 32) {
        float u = bf16w(src[row * C + c]), q = e4m3(code[row * C + c]) * scalbnf(1.0f, (int)scale[row * (C / 32) + c / 32] - 127);
        double dl = (double)u - (double)q; se += dl * dl; sn += (double)u * (double)u;
    }
    for (int o = 16; o > 0; o >>= 1) { se += __shfl_down_sync(0xffffffffu, se, o); sn += __shfl_down_sync(0xffffffffu, sn, o); }
    if (lane == 0) { err[row] = (float)(sqrt(se) * 1.0001) + 1e-20f; nrm[row] = (float)(sqrt(sn) * 1.0001); }
}
// Per row of the step: hb[t] = { |h~|, |h - h~| + HEAD_ROUNDING |h| } from the activations and their 8-bit codes.
__global__ void k_hbound(const float* h, const uint8_t* code, const uint8_t* scale, float* hb) {
    __shared__ double sd[8][3];
    int t = blockIdx.x, tid = threadIdx.x; double a[3] = {0, 0, 0};                          // sums of h~^2, (h - h~)^2, h^2
    for (int c = tid; c < D; c += 256) {
        float x = h[(size_t)t * D + c], q = e4m3(code[(size_t)t * D + c]) * scalbnf(1.0f, (int)scale[(size_t)t * (D / 32) + c / 32] - 127);
        double dl = (double)x - (double)q; a[0] += (double)q * q; a[1] += dl * dl; a[2] += (double)x * x;
    }
    for (int k = 0; k < 3; k++) for (int o = 16; o > 0; o >>= 1) a[k] += __shfl_down_sync(0xffffffffu, a[k], o);
    if ((tid & 31) == 0) for (int k = 0; k < 3; k++) sd[tid >> 5][k] = a[k];
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 8; w++) for (int k = 0; k < 3; k++) a[k] += sd[w][k];
        hb[t * 2] = (float)(sqrt(a[0]) * 1.0001); hb[t * 2 + 1] = (float)((sqrt(a[1]) + (double)HEAD_ROUNDING * sqrt(a[2])) * 1.0001);
    }
}
// The approximate winners of each row: by logit, and for a sampling row by capped logit / T + noise. Any token
// would do as the source of the threshold; the approximate winner gives the tightest one.
__global__ void k_apx_part(const float* apx, const float* sparam, const Row* rows, float* pv, int* pi) {
    __shared__ float sv[32][2]; __shared__ int si[32][2];
    int t = blockIdx.y, part = blockIdx.x, tid = threadIdx.x; const float* l = apx + (size_t)t * V0;
    const float invT = sparam[t * 2]; const uint32_t seed = __float_as_uint(sparam[t * 2 + 1]), pos = (uint32_t)rows[t].pos;
    float b[2] = {-FLT_MAX, -FLT_MAX}; int bi[2] = {0x7fffffff, 0x7fffffff};
    for (int k = 0; k < 4; k++) {
        int v = part * HEAD_SPAN + tid + 1024 * k; if (v >= V0) break;
        float a = l[v]; if (a > b[0]) { b[0] = a; bi[0] = v; }
        if (invT > 0.0f) { float z = 30.0f * tanhf(a * (1.0f / 30.0f)) * invT + gumbel(seed, pos, (uint32_t)v); if (z > b[1]) { b[1] = z; bi[1] = v; } }
    }
    for (int k = 0; k < 2; k++) for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, b[k], o); int oi = __shfl_down_sync(0xffffffffu, bi[k], o); if (ob > b[k] || (ob == b[k] && oi < bi[k])) { b[k] = ob; bi[k] = oi; } }
    if ((tid & 31) == 0) for (int k = 0; k < 2; k++) { sv[tid >> 5][k] = b[k]; si[tid >> 5][k] = bi[k]; }
    __syncthreads();
    if (tid == 0) for (int k = 0; k < 2; k++) {
        for (int w = 1; w < 32; w++) if (sv[w][k] > b[k] || (sv[w][k] == b[k] && si[w][k] < bi[k])) { b[k] = sv[w][k]; bi[k] = si[w][k]; }
        pv[(t * HEAD_PARTS + part) * 2 + k] = b[k]; pi[(t * HEAD_PARTS + part) * 2 + k] = bi[k];
    }
}
// win[2t], win[2t+1]: the two approximate winners of row t (the same token twice for a greedy row); list1: their blocks.
__global__ void k_apx_final(const float* pv, const int* pi, const float* sparam, int* win, int* list1) {
    int t = blockIdx.x, lane = threadIdx.x; float b[2] = {-FLT_MAX, -FLT_MAX}; int bi[2] = {0x7fffffff, 0x7fffffff};
    for (int p = lane; p < HEAD_PARTS; p += 32) for (int k = 0; k < 2; k++) { float x = pv[(t * HEAD_PARTS + p) * 2 + k]; int i = pi[(t * HEAD_PARTS + p) * 2 + k]; if (x > b[k] || (x == b[k] && i < bi[k])) { b[k] = x; bi[k] = i; } }
    for (int k = 0; k < 2; k++) for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, b[k], o); int oi = __shfl_down_sync(0xffffffffu, bi[k], o); if (ob > b[k] || (ob == b[k] && oi < bi[k])) { b[k] = ob; bi[k] = oi; } }
    if (lane == 0) {
        int g = bi[0], sm = sparam[t * 2] > 0.0f ? bi[1] : bi[0];
        if ((unsigned)g >= (unsigned)V0) g = -1;        // no winner: every logit NaN. k_head_thr then lets nothing through, as the full head finds nothing
        if ((unsigned)sm >= (unsigned)V0) sm = -1;
        win[t * 2] = g; win[t * 2 + 1] = sm; list1[t * 2] = max(g, 0) >> 4; list1[t * 2 + 1] = max(sm, 0) >> 4;
    }
}
// Thresholds from the exact logits of the approximate winners (ex1: [2 T blocks][16 rows][FAST_T]), in the units the
// full head compares in: thr[2t] a capped logit, thr[2t+1] a sampling score (FLT_MAX for a greedy row). Both sit a
// little below the winner's own value, for the last-digit wobble of tanh. Also empties the candidate list.
__global__ void k_head_thr(const float* ex1, const int* win, const float* sparam, const Row* rows, float* thr, int* count) {
    int t = blockIdx.x; if (threadIdx.x) return;
    if (t == 0) *count = 0;
    const float invT = sparam[t * 2]; const int g = win[t * 2], sm = win[t * 2 + 1];
    thr[t * 2] = g < 0 ? -FLT_MAX : 30.0f * tanhf(ex1[((size_t)(t * 2) * 16 + (g & 15)) * FAST_T + t] * (1.0f / 30.0f)) - HEAD_MARGIN;
    if (invT > 0.0f && sm >= 0) {
        float e = ex1[((size_t)(t * 2 + 1) * 16 + (sm & 15)) * FAST_T + t];
        thr[t * 2 + 1] = 30.0f * tanhf(e * (1.0f / 30.0f)) * invT + gumbel(__float_as_uint(sparam[t * 2 + 1]), (uint32_t)rows[t].pos, (uint32_t)sm) - HEAD_MARGIN * invT - 1e-4f;
    } else thr[t * 2 + 1] = invT > 0.0f ? -FLT_MAX : FLT_MAX;
}
// Candidates: tokens whose approximate logit plus bound reaches a threshold. Their 16-row blocks go on the list
// (a block can be listed more than once: harmless).
__global__ void k_head_cand(const float* apx, const float* eu, const float* un, const float* hb, const float* thr, const float* sparam, const Row* rows, int* list, int* count) {
    int t = blockIdx.y, part = blockIdx.x, tid = threadIdx.x; const float* l = apx + (size_t)t * V0;
    const float invT = sparam[t * 2], nq = hb[t * 2], ne = hb[t * 2 + 1], tg = thr[t * 2], ts = thr[t * 2 + 1];
    const uint32_t seed = __float_as_uint(sparam[t * 2 + 1]), pos = (uint32_t)rows[t].pos;
    for (int k = 0; k < 4; k++) {
        int v = part * HEAD_SPAN + tid + 1024 * k; if (v >= V0) break;
        float up = 30.0f * tanhf((l[v] + (eu[v] * nq + un[v] * ne + 1e-3f)) * (1.0f / 30.0f));   // the most its capped logit can be
        bool c = up >= tg;
        if (!c && invT > 0.0f) c = up * invT + gumbel(seed, pos, (uint32_t)v) >= ts;
        if (c) { int i = atomicAdd(count, 1); if (i < CAND_CAP) list[i] = v >> 4; }
    }
}
// The rows of the 16-row blocks list[0..n) of a dense bf16 matrix times T activation rows: k_gemm_bf16's
// operations in k_gemm_bf16's order, so each value is the full product's bit for bit. out: [n][16][ostride].
template <int NCG> __global__ void k_gemm_rows(const uint16_t* W, int C, int T, const float* xb, int xstride, const int* list, const int* count, int nmax, float* out, int ostride) {
    __shared__ float sh[16][32][4];
    int n = count ? min(*count, nmax) : nmax; if ((int)blockIdx.x >= n) return;
    int blk = list[blockIdx.x], lane = threadIdx.x & 31, warp = threadIdx.x >> 5, q = lane & 3, g = lane >> 2;
    int KB = C / 16, kseg = KB / 16, kbase = warp * kseg;
    const uint4* wt = (const uint4*)((const uint8_t*)W + ((size_t)blk * KB) * 512 + lane * 16);
    float d[NCG][4]; for (int cg = 0; cg < NCG; cg++) for (int i = 0; i < 4; i++) d[cg][i] = 0;
    for (int it = 0; it < kseg; it += 3) {
        uint4 w[3]; uint32_t b0[NCG][3], b1[NCG][3];
        for (int u = 0; u < 3; u++) {
            int kb = kbase + it + u, k0 = kb * 16; bool live = it + u < kseg;
            w[u] = live ? wt[(size_t)kb * 32] : make_uint4(0, 0, 0, 0);
            for (int cg = 0; cg < NCG; cg++) {
                int t = cg * 8 + g; b0[cg][u] = b1[cg][u] = 0;
                if (t < T && live) { const float2* x = (const float2*)(xb + (size_t)t * xstride + k0 + 2 * q); float2 lo = x[0], hi = x[4]; b0[cg][u] = pack_bf16(lo.x, lo.y); b1[cg][u] = pack_bf16(hi.x, hi.y); }
            }
        }
        for (int u = 0; u < 3; u++) for (int cg = 0; cg < NCG; cg++) mma_bf16(d[cg], w[u].x, w[u].y, w[u].z, w[u].w, b0[cg][u], b1[cg][u]);
    }
    for (int cg = 0; cg < NCG; cg++) {
        for (int i = 0; i < 4; i++) sh[warp][lane][i] = d[cg][i];
        __syncthreads();
        if (warp == 0) for (int i = 0; i < 4; i++) {
            int t = cg * 8 + 2 * q + (i & 1), r = g + (i >> 1) * 8;
            if (t >= T) continue;
            float v = 0; for (int w2 = 0; w2 < 16; w2++) v += sh[w2][lane][i];
            out[((size_t)blockIdx.x * 16 + r) * ostride + t] = v;
        }
        __syncthreads();
    }
}
// The winners among the candidates, from their exact logits (ex2: [blocks][16][FAST_T]), by the full head's rules:
// soft cap, ties to the lower token id. res[4t..] = { argmax, sample, sample, 0 }; res[4T] = 1 if the list
// overflowed (then nothing here is valid), res[4T+1] = candidate blocks listed.
__global__ void k_head_pick(const float* ex2, const int* list, const int* count, const float* sparam, const Row* rows, int* res, int T) {
    __shared__ float sv[32][2]; __shared__ int si[32][2];
    int t = blockIdx.x, tid = threadIdx.x, n = min(*count, CAND_CAP) * 16;
    const float invT = sparam[t * 2]; const uint32_t seed = __float_as_uint(sparam[t * 2 + 1]), pos = (uint32_t)rows[t].pos;
    float b[2] = {-FLT_MAX, -FLT_MAX}; int bi[2] = {0x7fffffff, 0x7fffffff};
    for (int e = tid; e < n; e += 1024) {
        int v = list[e >> 4] * 16 + (e & 15); if (v >= V0) continue;
        float x = 30.0f * tanhf(ex2[(size_t)e * FAST_T + t] * (1.0f / 30.0f));
        if (x > b[0] || (x == b[0] && v < bi[0])) { b[0] = x; bi[0] = v; }
        if (invT > 0.0f) { float z = x * invT + gumbel(seed, pos, (uint32_t)v); if (z > b[1] || (z == b[1] && v < bi[1])) { b[1] = z; bi[1] = v; } }
    }
    for (int k = 0; k < 2; k++) for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, b[k], o); int oi = __shfl_down_sync(0xffffffffu, bi[k], o); if (ob > b[k] || (ob == b[k] && oi < bi[k])) { b[k] = ob; bi[k] = oi; } }
    if ((tid & 31) == 0) for (int k = 0; k < 2; k++) { sv[tid >> 5][k] = b[k]; si[tid >> 5][k] = bi[k]; }
    __syncthreads();
    if (tid == 0) {
        for (int k = 0; k < 2; k++) for (int w = 1; w < 32; w++) if (sv[w][k] > b[k] || (sv[w][k] == b[k] && si[w][k] < bi[k])) { b[k] = sv[w][k]; bi[k] = si[w][k]; }
        int sm = invT > 0.0f ? bi[1] : bi[0];
        res[t * 4] = bi[0]; res[t * 4 + 1] = sm; res[t * 4 + 2] = sm; res[t * 4 + 3] = 0;
        if (t == 0) { res[T * 4] = *count > CAND_CAP; res[T * 4 + 1] = *count; }
    }
}

// A row that verifies a *sampled* draft (qrow[t] >= 0: index of the draft's logits in qlog) needs, in case the draft is
// rejected, a token drawn from the leftover distribution max(0, p - q) / Z. Gumbel-max over log(p - q) does that.
// The result replaces the row's "sample excluding the draft", which is the same thing for a most-likely-token draft.
struct ResPart { float b; int i; };
__global__ void k_resid_part(const float* logits, const float* sparam, const int* qrow, const Row* rows, const float* rowmax, const float* rowz,
                             const float* qlog, const double* qz, const float* qit, ResPart* parts) {
    __shared__ float sv[32]; __shared__ int si[32];
    int t = blockIdx.y, part = blockIdx.x, tid = threadIdx.x, q = qrow[t]; const float invT = sparam[t * 2];
    if (q < 0 || invT <= 0.0f) return;
    const uint32_t seed = __float_as_uint(sparam[t * 2 + 1]) ^ 0x5851F42Du, pos = (uint32_t)rows[t].pos;
    const float* l = logits + (size_t)t * V0; const float* x = qlog + (size_t)q * V0;
    const float zmax = rowmax[t] * invT, izp = 1.0f / rowz[t]; const double izq = 1.0 / qz[q], qT = (double)qit[q];   // q exactly as the draft was drawn
    float best = -FLT_MAX; int bi = 0x7fffffff;
    for (int k = 0; k < 4; k++) {
        int v = part * HEAD_SPAN + tid + 1024 * k; if (v >= V0) break;
        float w = expf(l[v] * invT - zmax) * izp - (float)(exp((double)(x[v] - 30.0f) * qT) * izq);
        if (w > 0.0f) { float sc = logf(w) + gumbel(seed, pos, (uint32_t)v); if (sc > best) { best = sc; bi = v; } }
    }
    for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, best, o); int oi = __shfl_down_sync(0xffffffffu, bi, o); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
    if ((tid & 31) == 0) { sv[tid >> 5] = best; si[tid >> 5] = bi; }
    __syncthreads();
    if (tid == 0) { for (int w = 1; w < 32; w++) if (sv[w] > best || (sv[w] == best && si[w] < bi)) { best = sv[w]; bi = si[w]; } parts[t * HEAD_PARTS + part] = ResPart{best, bi}; }
}
__global__ void k_resid_final(const ResPart* parts, const int* qrow, const float* sparam, int* res) {
    int t = blockIdx.x, lane = threadIdx.x;
    if (qrow[t] < 0 || sparam[t * 2] <= 0.0f) return;
    float best = -FLT_MAX; int bi = 0x7fffffff;
    for (int k = lane; k < HEAD_PARTS; k += 32) { ResPart r = parts[t * HEAD_PARTS + k]; if (r.b > best || (r.b == best && r.i < bi)) { best = r.b; bi = r.i; } }
    for (int o = 16; o > 0; o >>= 1) { float ob = __shfl_down_sync(0xffffffffu, best, o); int oi = __shfl_down_sync(0xffffffffu, bi, o); if (ob > best || (ob == best && oi < bi)) { best = ob; bi = oi; } }
    if (lane == 0 && bi != 0x7fffffff) res[t * 4 + 2] = bi;   // (p <= q everywhere cannot be rejected; the plain value stays)
}

// =====================================================================================================
// Dense matrices in MXFP8 on the FP8 tensor cores (Blackwell only). Used for the DSpark drafter: its
// matrices are converted from bf16 to 8-bit at load, halving the bytes it reads. Quantising the drafter
// cannot change the model's output - every draft is still verified by the full-precision target.
// =====================================================================================================
// bf16 row-major [R][C] -> E4M3 codes [R16][C] + one power-of-two exponent byte per 32 columns. Rows >= R are zero.
__global__ void k_to_mxfp8(const uint16_t* src, size_t R, int C, uint8_t* code, uint8_t* scale) {
    size_t blk = (size_t)blockIdx.x * blockDim.x + threadIdx.x, nb = C / 32, row = blk / nb; int b = (int)(blk % nb);
    if (row >= ((R + 15) / 16) * 16) return;
    float x[32]; for (int i = 0; i < 32; i++) x[i] = row < R ? bf16w(src[row * C + b * 32 + i]) : 0.0f;
    q8_block(x, code + row * C + b * 32, scale + row * nb + b);
}
// out[t][r] = W[r] . x[t], W in MXFP8 tiles, x already quantised (codes [T][C], exponents [T][C/32]). T <= 8 per NCG.
template <int NCG> __global__ void k_gemm_fp8(const uint8_t* W, const uint8_t* S, int C, int T, const uint8_t* xcode, const uint8_t* xscale, float* out, int R, int round, int ostride) {
    __shared__ float sh[8][32][4];
    int blk = blockIdx.x, r0 = blk * 16, lane = threadIdx.x & 31, warp = threadIdx.x >> 5, i = lane & 3, j = lane >> 2;
    int KB = C / 32, kseg = KB / 8, kbase = warp * kseg;
    const uint4* wt = (const uint4*)(W + ((size_t)blk * KB) * 512 + lane * 16);
    const uint8_t* st = S + ((size_t)blk * KB) * 16 + j * 2 + (lane & 1);
    float d[NCG][4]; for (int cg = 0; cg < NCG; cg++) for (int q = 0; q < 4; q++) d[cg][q] = 0;
    for (int it = 0; it < kseg; it += 3) {
        uint4 w[3]; uint32_t sa[3], b0[NCG][3], b1[NCG][3], sb[NCG][3];
        for (int u = 0; u < 3; u++) {
            int kb = kbase + it + u; bool live = it + u < kseg;
            w[u] = live ? wt[(size_t)kb * 32] : make_uint4(0, 0, 0, 0); sa[u] = live ? st[(size_t)kb * 16] : 127;
            for (int cg = 0; cg < NCG; cg++) {
                int t = cg * 8 + j; b0[cg][u] = b1[cg][u] = 0; sb[cg][u] = 127;
                if (t < T && live) { const uint8_t* xc = xcode + (size_t)t * C + kb * 32 + 4 * i; b0[cg][u] = *(const uint32_t*)xc; b1[cg][u] = *(const uint32_t*)(xc + 16); sb[cg][u] = xscale[(size_t)t * KB + kb]; }
            }
        }
        for (int u = 0; u < 3; u++) for (int cg = 0; cg < NCG; cg++) mma_fp8(d[cg], w[u].x, w[u].y, w[u].z, w[u].w, b0[cg][u], b1[cg][u], sa[u], sb[cg][u]);
    }
    for (int cg = 0; cg < NCG; cg++) {
        for (int q = 0; q < 4; q++) sh[warp][lane][q] = d[cg][q];
        __syncthreads();
        if (warp == 0) for (int q = 0; q < 4; q++) {
            int t = cg * 8 + 2 * i + (q & 1), row = r0 + j + (q >> 1) * 8;
            if (t >= T || row >= R) continue;
            float v = 0; for (int w2 = 0; w2 < 8; w2++) v += sh[w2][lane][q];
            out[(size_t)t * ostride + row] = round ? bf16r(v) : v;
        }
        __syncthreads();
    }
}
// Launch the right instantiation for T tokens.
void gemm_fp8(const Fp8Mat& f, int C, int T, const uint8_t* xcode, const uint8_t* xscale, float* out, int round, int ostride) {
    if (T > 64) {
        gemm_fp8(f, C, 64, xcode, xscale, out, round, ostride);
        gemm_fp8(f, C, T - 64, xcode + (size_t)64 * C, xscale + (size_t)64 * (C / 32), out + (size_t)64 * ostride, round, ostride);
        return;
    }
    int nb = (f.R + 15) / 16;
    // One kernel per number of 8-token column groups: a step of 17 tokens should not pay for 32.
    switch ((T + 7) / 8) {
        #define GEMM8(n) case n: k_gemm_fp8<n><<<nb, 256, 0, g_stream>>>(f.w, f.s, C, T, xcode, xscale, out, f.R, round, ostride); break;
        GEMM8(1) GEMM8(2) GEMM8(3) GEMM8(4) GEMM8(5) GEMM8(6) GEMM8(7) default: GEMM8(8)
        #undef GEMM8
    }
}

// Scratch for packed activations (three inputs of up to 64 tokens x XPK_C values), and the batch size from which
// packing pays (INK_PACK_MIN to experiment; a large value turns it off).
const int XPK_C = 4608;
static uint2* g_xpk[3];
static int g_attn_group = getenv("INK_ATTN_GROUP") ? atoi(getenv("INK_ATTN_GROUP")) : 17;   // rows from which attention runs in groups of 8 (at most GSUB)
// Threads per block of k_attn_prep for n rows, as measured (us per launch with 1024 / 512 / 256 / 128 threads: 1 row
// 2.6 / 2.9 / 3.1 / 5.9; 8 rows 6.1 / 4.6 / 5.3 / 6.1; 32 rows 15.9 / 7.5 / 5.7 / 6.7; 128 rows 44.5 / 19.8 / 14.6 / 12.9).
static inline int prep_threads(int n) { return n <= 4 ? 1024 : n <= 16 ? 512 : n <= 48 ? 256 : 128; }
static int g_xwide = getenv("INK_XWIDE") ? atoi(getenv("INK_XWIDE")) : 48;          // rows from which experts run as wide tasks
static int g_pack_min = getenv("INK_PACK_MIN") ? atoi(getenv("INK_PACK_MIN")) : 9;
static int g_bulk_min = getenv("INK_BULK_MIN") ? atoi(getenv("INK_BULK_MIN")) : 9;    // from here on: k_gemm_bf16_bulk (a large value: the older packed kernel)
void gemv_scratch() {
    if (g_xpk[0]) return;
    for (int i = 0; i < 3; i++) { CK(cudaMalloc(&g_xpk[i], (size_t)64 * (XPK_C / 4) * sizeof(uint2))); CK(cudaMemset(g_xpk[i], 0, (size_t)64 * (XPK_C / 4) * sizeof(uint2))); }
}
// Dense bf16 matrix (stored as tiles) x T activation rows, on the tensor cores for every batch size.
void gemv(int R, int T, const uint16_t* W, int C, int T2, const float* x0, const float* x1, const float* x2, int xstride, int r1, int r2, float* out, int R2, int round) {
    gemv_scratch();
    if (T > 64) {                                        // two launches of at most 64 tokens each
        gemv(R, 64, W, C, 64, x0, x1, x2, xstride, r1, r2, out, R2, round);
        gemv(R, T - 64, W, C, T - 64, x0 + (size_t)64 * xstride, x1 + (size_t)64 * xstride, x2 + (size_t)64 * xstride, xstride, r1, r2, out + (size_t)64 * R, R2, round);
        return;
    }
    int nb = (r1 + 15) / 16 + (r2 - r1 + 15) / 16 + (R - r2 + 15) / 16;
    // One kernel per number of 8-token column groups: a step of 17 tokens should not pay for 32.
    if (T >= g_bulk_min && C <= XPK_C) {                  // prompt-sized batches: k_gemm_bf16_bulk
        const float* xs[3] = {x0, x1, x2}; const uint4* xp[3]; bool packed[3] = {false, false, false}; const int NP = ((T + 7) / 8 + 1) / 2;
        for (int i = 0; i < 3; i++) {
            bool used = i == 0 ? r1 > 0 : i == 1 ? r2 > r1 : R > r2;
            int same = -1; for (int j = 0; j < i; j++) if (packed[j] && xs[j] == xs[i]) same = j;
            if (!used) xp[i] = (const uint4*)g_xpk[i];    // never read
            else if (same >= 0) xp[i] = xp[same];
            else { k_pack_bf16s<<<((C / 16) * NP * 32 + 255) / 256, 256, 0, g_stream>>>(xs[i], xstride, C, T, NP, (uint4*)g_xpk[i]); xp[i] = (const uint4*)g_xpk[i]; packed[i] = true; }
        }
        int nt0 = (r1 + 15) / 16, nt1 = (r2 - r1 + 15) / 16, nt2 = (R - r2 + 15) / 16;
        // One warp per (tile, column segment in turn). Big matrices: 4 warps per tile (more would only queue); small
        // ones: all 16 segments at once, or there are too few warps to keep the GPU busy. (Measured at 64 tokens:
        // 8,600 rows 30 us with 4, 43 with 16; 2,304 rows 31 and 17; the 200,058-row head 625 and 796.)
        #define BULKS(SP) { \
            int nbb = nt0 + nt1 + nt2; \
            switch ((T + 7) / 8) { \
                case 1: k_gemm_bf16_bulk<1, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                case 2: k_gemm_bf16_bulk<2, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                case 3: k_gemm_bf16_bulk<3, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                case 4: k_gemm_bf16_bulk<4, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                case 5: k_gemm_bf16_bulk<5, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                case 6: k_gemm_bf16_bulk<6, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                case 7: k_gemm_bf16_bulk<7, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
                default: k_gemm_bf16_bulk<8, 1, SP><<<nbb, SP * 32, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break; \
            } }
        if (nt0 + nt1 + nt2 > 300) BULKS(4) else BULKS(16)
        #undef BULKS
        return;
    }
    if (T >= g_pack_min && C <= XPK_C) {                  // bigger batches: pack each distinct input once
        const float* xs[3] = {x0, x1, x2}; const uint2* xp[3]; bool packed[3] = {false, false, false};
        for (int i = 0; i < 3; i++) {
            bool used = i == 0 ? r1 > 0 : i == 1 ? r2 > r1 : R > r2;
            int same = -1; for (int j = 0; j < i; j++) if (packed[j] && xs[j] == xs[i]) same = j;
            if (!used) xp[i] = g_xpk[i];                  // never read
            else if (same >= 0) xp[i] = xp[same];
            else { k_pack_bf16<<<dim3((C / 4 + 255) / 256, T), 256, 0, g_stream>>>(xs[i], xstride, C, g_xpk[i]); xp[i] = g_xpk[i]; packed[i] = true; }
        }
        switch ((T + 7) / 8) {
            #define GEMM16P(n, u) case n: k_gemm_bf16p<n, u><<<nb, 512, 0, g_stream>>>(W, C, T, xp[0], xp[1], xp[2], r1, r2, out, R, round); break;
            GEMM16P(1, 3) GEMM16P(2, 3) GEMM16P(3, 3) GEMM16P(4, 3) GEMM16P(5, 3) GEMM16P(6, 3) GEMM16P(7, 2) default: GEMM16P(8, 2)
            #undef GEMM16P
        }
        return;
    }
    switch ((T + 7) / 8) {
        #define GEMM16(n) case n: k_gemm_bf16<n><<<nb, 512, 0, g_stream>>>(W, C, T, x0, x1, x2, xstride, r1, r2, out, R, round); break;
        GEMM16(1) GEMM16(2) GEMM16(3) GEMM16(4) GEMM16(5) GEMM16(6) GEMM16(7) default: GEMM16(8)
        #undef GEMM16
    }
}


// =====================================================================================================
// Tile layout ("swizzle"). A tensor-core operation consumes a tile of 16 weight rows x 32 bytes, 16 bytes
// per thread of the warp. In row-major storage those 512 bytes are 16 separate pieces, 1-5 KB apart. At
// load time we rewrite every matrix so that tile (row group rg, slice kb) is one contiguous 512-byte
// chunk in exactly the per-thread order {a0, a1, a2, a3}; a warp then reads it as one coalesced access.
//   thread l (i = l & 3, j = l >> 2), register r: row = rg*16 + j + (r & 1)*8, bytes kb*32 + (r >> 1)*16 + 4i ..+3
// =====================================================================================================
__global__ void k_swizzle(const uint8_t* src, uint32_t* dst, size_t n_u32, size_t row0, size_t rows_valid, int RB) {
    size_t d = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (d >= n_u32) return;
    size_t tile = d / 128; int KB = RB / 32, lane = (int)(d % 128) / 4, r = (int)(d % 4), i = lane & 3, j = lane >> 2;
    size_t rg = tile / KB, row = rg * 16 + j + (r & 1) * 8; int kb = (int)(tile % KB);
    dst[d] = row < rows_valid ? *(const uint32_t*)(src + (row0 + row) * RB + kb * 32 + (r >> 1) * 16 + 4 * i) : 0u;
}
// Scales per tile: 16 entries ordered (j, parity) -> row rg*16 + j + parity*8. `w` bytes per entry (4: NVFP4, 1: MXFP8).
__global__ void k_swizzle_scale(const uint8_t* src, uint8_t* dst, size_t n_entries, int nblk, int w) {
    size_t d = (size_t)blockIdx.x * blockDim.x + threadIdx.x; if (d >= n_entries) return;
    size_t tile = d / 16; int KB = nblk / w, jj = (int)(d % 16) / 2, par = (int)(d % 2);
    size_t rg = tile / KB, row = rg * 16 + jj + par * 8; int kb = (int)(tile % KB);
    for (int b = 0; b < w; b++) dst[d * w + b] = src[row * nblk + kb * w + b];
}

// ------------------------------------------------------------------ host side
struct Raw { void* dev; size_t bytes; std::vector<uint8_t> host; };


struct Layer {
    bool global, fp8, core;
    uint16_t *qkvr, *wo, *router, *kproj, *vproj;        // bf16 matrices
    float *qscale, *rscale, *rel, *in_norm, *mlp_norm, *attn_sconv, *mlp_sconv, *kconv, *vconv, *rbias;
    float *ctl[4][3];                                    // [stream][unused, gain, forget] as fp32
    uint16_t* ctl_down[4];
    uint16_t* ctl3_down; float *ctl3_gain, *ctl3_forget;   // attn|k|v concatenated                               // [stream] bf16 [64][4608]
    uint8_t *w13, *w13s, *w2, *w2s; float *w13amax, *w2amax; float inv_den13, inv_den2;
    Fp8Mat f_q, f_k, f_v, f_wo;                          // optional MXFP8 copies of the attention matrices (INK_FP8_ATTN)
};

// ---- Memory that grows with the conversation. The caches indexed by position (full-attention keys/values, the
// drafter's full-attention layer, the score scratch) are address ranges reserved at start-up for the largest context
// allowed, with physical memory attached only as far as each conversation has got. Addresses never change, so the
// kernels and recorded graphs are unaware of it; the GPU memory in use follows what is actually being served.
#define CU(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { fprintf(stderr, "CUDA driver error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); abort(); } } while (0)
const size_t VGRAN = 2u << 20;                            // mappings are multiples of 2 MiB
const int CTX_STEP = 2048;                                // context memory is attached in steps of this many tokens
static size_t vround(size_t b) { return (b + VGRAN - 1) / VGRAN * VGRAN; }
extern cudaStream_t g_stream;
struct VRange {
    CUdeviceptr base = 0; size_t reserved = 0, mapped = 0;
    std::vector<std::pair<CUmemGenericAllocationHandle, size_t>> chunks;
    // Attach memory so that the first `need` bytes are usable (zero-filled). False if the GPU has none to give.
    bool grow(size_t need) {
        need = std::min(vround(need), reserved);
        if (mapped >= need) return true;
        size_t sz = need - mapped;
        CUmemAllocationProp prop = {}; prop.type = CU_MEM_ALLOCATION_TYPE_PINNED; prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE; prop.location.id = 0;
        CUmemGenericAllocationHandle h;
        if (cuMemCreate(&h, sz, &prop, 0) != CUDA_SUCCESS) return false;
        if (cuMemMap(base + mapped, sz, 0, h, 0) != CUDA_SUCCESS) { cuMemRelease(h); return false; }
        CUmemAccessDesc acc = {}; acc.location = prop.location; acc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        CU(cuMemSetAccess(base + mapped, sz, &acc, 1));
        if (cudaMemsetAsync((void*)(base + mapped), 0, sz, g_stream) != cudaSuccess) { fprintf(stderr, "memset of new context memory failed\n"); abort(); }
        chunks.push_back({h, sz}); mapped += sz;
        return true;
    }
    // Give back whole attachments that lie entirely above `keep` bytes.
    void shrink(size_t keep) {
        while (!chunks.empty() && mapped - chunks.back().second >= keep) {
            size_t sz = chunks.back().second; mapped -= sz;
            CU(cuMemUnmap(base + mapped, sz)); CU(cuMemRelease(chunks.back().first)); chunks.pop_back();
        }
    }
};
static CUdeviceptr vreserve(size_t bytes) { CUdeviceptr b = 0; CU(cuMemAddressReserve(&b, vround(bytes), 0, 0, 0)); return b; }

struct Engine {
    int max_ctx;
    std::map<std::string, Raw> raw;
    Layer L[24];
    uint16_t *embed, *unembed, *unembed_rm = nullptr;
    float *embed_norm, *norm, *attn_q[23], *mlp_q[24], *out_q, *msg_q[12];
    bool kv8 = false;                                    // 8-bit key/value cache (INK_KV_FP8)
    bool fp8_attn = false, fp8_head = false; Fp8Mat f_head; uint8_t *tq, *ts;   // lossy 8-bit options for the target model
    // per-visit state
    uint16_t *Kc[NVIS], *Vc[NVIS]; float *ring[NVIS][4];  // rings: K, V, ATTN, MLP conv inputs
    // scratch
    float *stack[5], *prelude_partial, *bank[2], *prelude_dests, *inputs, *msg, *xin, *proj, *dk, *dv, *qc, *bias, *att_sc, *att_l, *att_y,
          *attn_raw, *attn_out, *mixtmp, *pair, *hbuf, *gx, *mlp_in, *rlogit, *gamma, *xq, *xamax, *gu, *act, *aq, *aamax, *down, *moe_out, *y, *out, *logits;
    int *tok, *ids, *arg, *tasks, *ntasks, *tasksw = nullptr; uint4* xb = nullptr; Row* rows;   // tasksw, xb: wide expert tasks and their gathered activations      // rows: per-row conversation and position of the current step
    Grp* grp = nullptr; int *gb = nullptr, *rowg = nullptr, *cso = nullptr;   // the step's row groups for attention (see k_attn_scores_g, k_attn_scores_l)
    uint8_t* snap = nullptr; size_t snap_bytes = 0; bool vis_global[NVIS] = {}, snap_ok[MAXSEQ] = {}; int last_T = 0;   // prefix-cache snapshots: [conversations][snap_bytes]
    int* qrow = nullptr; float* rowz = nullptr; ResPart* rparts = nullptr;   // per row: index of the sampled draft it verifies (-1: none); normaliser of p; scratch
    VRange vK[NVIS][MAXSEQ], vV[NVIS][MAXSEQ], vdK[MAXSEQ], vdV[MAXSEQ], vsc;   // on-demand ranges: per global visit and conversation; drafter; score scratch
    int ctx_mapped[MAXSEQ] = {}; size_t reserve_bytes = 0;
    int seq_tokens[MAXSEQ] = {}, snap_tokens[MAXSEQ] = {};   // positions each conversation has processed; and had when its snapshot was taken   // tokens of context each conversation has memory for; GPU memory to leave free
    int nseq = 1; size_t kv_local = 0, kv_global = 0;        // conversations; bytes of one conversation's KV cache per visit (sliding-window / full attention)
    float* head_v; int* head_i; float* rowmax;
    SampPart* sparts; float* sparam; int *excl, *sres;       // GPU sampling state: per-row {1/T, seed}, per-row excluded token, results
    float def_temp = 0.0f; unsigned def_seed = 1;            // sampling used by rows that do not specify their own
    // DSpark drafter (present when mtp.* tensors were uploaded)
    struct Draft {
        bool on = false;
        uint16_t *kvctx, *qkv[3], *o[3], *gu[3], *down[3], *rnn_hf, *rnn_sw, *M, *w1, *Kd[3], *Vd[3];
        float *alpha, *ctx_norm, *in_norm[3], *post_norm[3], *norm, *qn[3], *kn[3], *kcn, *mask, *rnn_b, *bvec;
        float *dctx, *dkv, *resid, *h, *qkvo, *qd, *att, *hp, *guo, *act, *base, *cin, *hsel, *C, *mkin, *sw, *state, *lat, *mb, *pv, *psum, *conf;
        float *px, *pmx, *top, *qit, *dsamp, *qlog; double *psumd, *qz;        // sampled drafts: per-conversation {1/T, seed}; kept draft logits and their normalisers, [conversations][7]
        int *anchor, *L, *dseq, *prev, *out, *pi;            // per conversation of the batch: anchor token, its position, its slot
        cudaGraphExec_t graph[24][DB][DBATCH + 1] = {};      // per context bucket, number of drafts, conversations in the batch
        float* frnn;                                         // [conversations][D]: Markov-head feature of each conversation's last valid position
        size_t lstride = 0, gstride = 0;                     // bytes of one conversation's draft cache (sliding-window / full layer)
        bool fp8 = false; Fp8Mat f_qkv[3], f_o[3], f_gu[3], f_down[3], f_un, f_M;   // 8-bit copies of the drafter's matrices
        uint8_t *qc, *qs;                                    // quantised activations of the current GEMM input
    } dr;
    float* feats;                                        // DSpark taps of the last forward: [4 taps][TMAX][D]
    uint8_t *xcode, *xscale, *acode, *ascale, *xcode8, *acode8;
    cudaGraphExec_t graph[TMAX + 1][24][3][2] = {};             // per batch size, context bucket (1024 << b), and output head: none / full / two-stage
    // two-stage head: per token |u - u~| and |u|; per row bounds, approximate winners, thresholds; candidate blocks and their exact logits
    float *h_eu = nullptr, *h_un = nullptr, *h_hb = nullptr, *h_pv = nullptr, *h_thr = nullptr, *h_ex1 = nullptr, *h_ex2 = nullptr;
    int *h_pi = nullptr, *h_win = nullptr, *h_l1 = nullptr, *h_l2 = nullptr, *h_cnt = nullptr;
    bool head2 = false; long head2_fallbacks = 0;
    float* trace = nullptr;                              // host [48][4][D], filled for the last token of a call
};

// Optional profiler (INK_PROF=1): synchronise after each stage and accumulate wall time. Slows things down.
bool g_prof = getenv("INK_PROF") != nullptr;
std::map<std::string, double> g_time; double g_last = 0;
double now() { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
void prof(const char* stage) { if (!g_prof) return; CK(cudaDeviceSynchronize()); double t = now(); g_time[stage] += t - g_last; g_last = t; }

float* dalloc(size_t n) { float* p; CK(cudaMalloc(&p, n * sizeof(float))); CK(cudaMemset(p, 0, n * sizeof(float))); return p; }

const Raw& get(Engine* e, const std::string& name) {
    auto it = e->raw.find(name);
    if (it == e->raw.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); abort(); }
    return it->second;
}
// Small tensor -> fp32 on the device (from bf16 or f32 host copy), optionally scaled.
float* small_f32(Engine* e, const std::string& name, bool is_bf16, float scale = 1.0f) {
    const Raw& r = get(e, name); std::vector<float> v;
    if (r.host.empty()) { fprintf(stderr, "tensor %s has no host copy\n", name.c_str()); abort(); }
    if (is_bf16) { size_t n = r.bytes / 2; v.resize(n); const uint16_t* s = (const uint16_t*)r.host.data(); for (size_t i = 0; i < n; i++) { uint32_t u = (uint32_t)s[i] << 16; memcpy(&v[i], &u, 4); v[i] *= scale; } }
    else { size_t n = r.bytes / 4; v.resize(n); memcpy(v.data(), r.host.data(), r.bytes); for (auto& x : v) x *= scale; }
    float* d; CK(cudaMalloc(&d, v.size() * 4)); CK(cudaMemcpy(d, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return d;
}
template <class T> T* big(Engine* e, const std::string& name) { return (T*)get(e, name).dev; }


// Rewrite a row-major matrix (rows x RB bytes) into tile order; frees the source. Ranges (r1, r2) get their own
// row groups so a tile never mixes rows that read different inputs.
uint8_t* swizzle(const void* src, size_t rows, int RB, size_t r1 = 0, size_t r2 = 0, bool free_src = true) {
    if (r1 == 0) r1 = r2 = rows;
    size_t KB = RB / 32, g0 = (r1 + 15) / 16, g1 = (r2 - r1 + 15) / 16, g2 = (rows - r2 + 15) / 16;
    uint8_t* dst; CK(cudaMalloc(&dst, (g0 + g1 + g2) * KB * 512));
    size_t gs[3] = {g0, g1, g2}, row0[3] = {0, r1, r2}, valid[3] = {r1, r2 - r1, rows - r2}, goff = 0;
    for (int k = 0; k < 3; k++) {
        size_t n = gs[k] * KB * 128;
        if (n) k_swizzle<<<(unsigned)((n + 255) / 256), 256>>>((const uint8_t*)src, (uint32_t*)(dst + goff * KB * 512), n, row0[k], valid[k], RB);
        goff += gs[k];
    }
    CK(cudaDeviceSynchronize());
    if (free_src) CK(cudaFree((void*)src));
    return dst;
}
uint8_t* swizzle_scale(const void* src, size_t rows, int nblk, int w) {
    size_t n = rows * nblk / w; uint8_t* dst; CK(cudaMalloc(&dst, n * w));
    k_swizzle_scale<<<(unsigned)((n + 255) / 256), 256>>>((const uint8_t*)src, dst, n, nblk, w);
    CK(cudaDeviceSynchronize()); CK(cudaFree((void*)src));
    return dst;
}

static bool g_prefetch = getenv("INK_PREFETCH") == nullptr || atoi(getenv("INK_PREFETCH")) != 0;
int pf_blocks(const PF& pf, int threads) { return (int)(((size_t)pf.bytes / 32 + threads - 1) / threads); }
// A slice [a, b) (fractions) of a matrix of `bytes` bytes, to be touched by a kernel whose real work uses `nwork` blocks.
PF pf_slice(const void* p, size_t bytes, double a, double b, int nwork) {
    PF pf; pf.nwork = nwork; pf.p = nullptr; pf.bytes = 0;
    if (!g_prefetch || !p) return pf;
    size_t lo = (size_t)(bytes * a) & ~(size_t)31, hi = (size_t)(bytes * b) & ~(size_t)31;
    pf.p = (const uint8_t*)p + lo; pf.bytes = (unsigned)(hi - lo); return pf;
}

// Pick the instantiation for this source / query count (2..6 sources; 1, 3 or 4 queries).
#define NS_SWITCH(n, CALL) switch (n) { case 2: CALL(2); break; case 3: CALL(3); break; case 4: CALL(4); break; case 5: CALL(5); break; case 6: CALL(6); break; \
                                        default: fprintf(stderr, "bad source count %d\n", n); abort(); }
void launch_post_attn(int T, const PostAttn& pa) {
#define C_(NS) k_post_attn<NS><<<T + pf_blocks(pa.pf, NT), NT, 0, g_stream>>>(pa)
    NS_SWITCH(pa.ns + 1, C_)
#undef C_
}
void launch_post_moe(int T, const PostMoe& pm) {
#define C1_(NS) k_post_moe<NS, 1><<<T + pf_blocks(pm.pf, NT), NT, 0, g_stream>>>(pm)
#define C3_(NS) k_post_moe<NS, 3><<<T + pf_blocks(pm.pf, NT), NT, 0, g_stream>>>(pm)
    if (pm.Q == 3) { NS_SWITCH(pm.ns + 1, C3_) } else { NS_SWITCH(pm.ns + 1, C1_) }      // Q == 0 skips the mix inside
#undef C1_
#undef C3_
}
void launch_mix(int T, const MixArgs& a) {
#define C1_(NS) k_mixq<NS, 1><<<T, NT, 0, g_stream>>>(a)
#define C3_(NS) k_mixq<NS, 3><<<T, NT, 0, g_stream>>>(a)
#define C4_(NS) k_mixq<NS, 4><<<T, NT, 0, g_stream>>>(a)
    if (a.Q == 4) { NS_SWITCH(a.n, C4_) } else if (a.Q == 3) { NS_SWITCH(a.n, C3_) } else { NS_SWITCH(a.n, C1_) }
#undef C1_
#undef C3_
#undef C4_
}

// err / nrm (optional, [R]): per row, the size of the quantisation error and of the row (see the two-stage head).
Fp8Mat to_fp8(const uint16_t* src, size_t R, int C, float* err = nullptr, float* nrm = nullptr) {
    size_t R16 = ((R + 15) / 16) * 16, nb = C / 32; uint8_t *code, *scale;
    CK(cudaMalloc(&code, R16 * C)); CK(cudaMalloc(&scale, R16 * nb));
    k_to_mxfp8<<<(unsigned)((R16 * nb + 255) / 256), 256>>>(src, R, C, code, scale);
    if (err) k_fp8_rowerr<<<(unsigned)((R + 7) / 8), 256>>>(src, code, scale, R, C, err, nrm);
    CK(cudaDeviceSynchronize());
    Fp8Mat m; m.R = (int)R; m.C = C; m.w = swizzle(code, R16, C); m.s = swizzle_scale(scale, R16, (int)nb, 1);
    return m;
}

void mix(Engine* e, int T, std::vector<const float*> src, const float* add, const float* q, int Q, const float* norm, float* out) {
    MixArgs a{}; a.n = (int)src.size(); for (int i = 0; i < a.n; i++) a.src[i] = src[i];
    a.add = add; a.q = q; a.Q = Q; a.norm = norm; a.out = out;
    launch_mix(T, a);
}
void blend(Engine* e, int T, Layer& L, int s, const float* dest, int dstride, const float* msg, const float* w, float* out, int ostride) {
    k_pair<<<T, NT, 0, g_stream>>>(dest, dstride, msg, 4 * D, e->pair);
    gemv(64, T, L.ctl_down[s], 2 * D, T, e->pair, e->pair, e->pair, 2 * D, 64, 64, e->hbuf, 64, 1);
    if (T >= GATE_BIG) k_gate_big<<<D / 8, 256, 0, g_stream>>>(e->hbuf, L.ctl[s][1], L.ctl[s][2], dest, dstride, msg, 4 * D, e->gx, T);
    else k_gate<<<dim3(D / 8, T), 256, 0, g_stream>>>(e->hbuf, L.ctl[s][1], L.ctl[s][2], dest, dstride, msg, 4 * D, e->gx);
    k_rms_s<<<T, NT, 0, g_stream>>>(e->gx, w, out, ostride);
}
void trace_copy(Engine* e, int visit, int slot, const float* dev, int T) {
    if (e->trace) CK(cudaMemcpy(e->trace + ((size_t)visit * 4 + slot) * D, dev + (size_t)(T - 1) * D, D * 4, cudaMemcpyDeviceToHost));
}
}  // namespace

namespace {
// ---- micro-benchmark: each kernel type alone, 240 launches replayed from one graph (no per-kernel syncs),
// cycling over layers so the weights are cold like in a real token. Call after at least one forward.
template <class F> void mbench(const char* name, double mb, F launch) {
    const int N = 240; cudaGraph_t g; cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal));
    for (int r = 0; r < N; r++) launch(r);
    CK(cudaStreamEndCapture(g_stream, &g)); CK(cudaGraphInstantiate(&ge, g, 0));
    double best[7];
    for (int r = 0; r < 7; r++) { double t0 = now(); CK(cudaGraphLaunch(ge, g_stream)); CK(cudaStreamSynchronize(g_stream)); best[r] = now() - t0; }
    std::sort(best, best + 7);
    double us = best[3] / N * 1e6;
    if (mb > 0) fprintf(stderr, "  %-34s %7.2f us   (%5.1f MB -> %5.0f GB/s)\n", name, us, mb, mb / 1e3 / (us * 1e-6));
    else fprintf(stderr, "  %-34s %7.2f us\n", name, us);
    CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g));
}

}  // namespace

// ---- Recorded graphs. A step is recorded once per shape (batch size, context bucket, with/without head) and
// replayed afterwards. Each recorded graph holds 3-5 MiB of GPU memory, and a long session meets hundreds of shapes,
// so only the most recently used GRAPH_CAP are kept (at most about 260 MiB). If the GPU has no memory left for a new one even after dropping
// the others, the step runs its kernels directly: slower (about 20% for one token, a few % for big batches), never fatal.
// Load every kernel onto the GPU when CUDA starts instead of at its first launch. A first launch needs GPU memory, and
// the fallback below must work exactly when there is none left. (Runs before main; CUDA reads it at its first call.)
static const int g_eager_loading = setenv("CUDA_MODULE_LOADING", "EAGER", 0);
const int GRAPH_CAP = 48;
struct GraphCache {
    struct Ent { cudaGraphExec_t* slot; uint64_t used; };
    std::vector<Ent> ents; uint64_t tick = 0; bool warned = false;
    void touch(cudaGraphExec_t* slot) { for (auto& x : ents) if (x.slot == slot) { x.used = ++tick; return; } }
    void evict(size_t keep) {
        while (ents.size() > keep) {
            size_t lru = 0; for (size_t i = 1; i < ents.size(); i++) if (ents[i].used < ents[lru].used) lru = i;
            cudaGraphExecDestroy(*ents[lru].slot); *ents[lru].slot = nullptr; ents.erase(ents.begin() + lru);
        }
    }
    bool add(cudaGraphExec_t* slot, cudaGraph_t g) {
        static const int cap = getenv("INK_GRAPH_CAP") ? std::max(1, atoi(getenv("INK_GRAPH_CAP"))) : GRAPH_CAP;
        evict(cap - 1);
        cudaError_t err = cudaGraphInstantiate(slot, g, 0);
        if (err != cudaSuccess) { cudaGetLastError(); evict(0); err = cudaGraphInstantiate(slot, g, 0); }
        if (err != cudaSuccess) {
            cudaGetLastError(); *slot = nullptr;
            if (!warned) { fprintf(stderr, "warning: no GPU memory left to record a step graph (%s); running without it. Lower --ctx to leave some room.\n", cudaGetErrorString(err)); warned = true; }
            return false;
        }
        static const bool log = getenv("INK_GRAPH_LOG") != nullptr;   // debugging: how often are step graphs recorded?
        if (log) fprintf(stderr, "graph: recorded (%zu held)\n", ents.size() + 1);
        ents.push_back({slot, ++tick}); return true;
    }
} g_graphs;
// Run `body` (a sequence of kernel launches on g_stream) through its recorded graph, recording it on first use.
template <class F> static void run_graphed(cudaGraphExec_t* slot, F body) {
    if (*slot) g_graphs.touch(slot);
    else {
        cudaGraph_t g;
        CK(cudaStreamBeginCapture(g_stream, cudaStreamCaptureModeGlobal));
        body();
        CK(cudaStreamEndCapture(g_stream, &g));
        bool ok = g_graphs.add(slot, g);
        CK(cudaGraphDestroy(g));
        if (!ok) { body(); return; }
    }
    CK(cudaGraphLaunch(*slot, g_stream));
}

extern "C" {

void ink_trim(void* ep, int seq, int tokens);
// Make sure conversation `seq` has memory for positions below `tokens` (rounded up to CTX_STEP, at most max_ctx).
// Returns 0, or 1 if that would leave less than the reserve free or the GPU has no memory: the conversation then
// cannot grow, everything already attached keeps working.
int ink_ensure(void* ep, int seq, int tokens) {
    auto* e = (Engine*)ep; auto& d = e->dr;
    if (seq < 0 || seq >= e->nseq || tokens < 0) return 1;
    tokens = (std::min(tokens, e->max_ctx) + CTX_STEP - 1) / CTX_STEP * CTX_STEP;      // max_ctx is a multiple of CTX_STEP
    if (tokens <= e->ctx_mapped[seq]) return 0;
    size_t kvt = (size_t)HKV * DH * (e->kv8 ? 1 : 2), dkt = (size_t)HKV * DH * 2;   // bytes per token: one cache of a full-attention visit; drafter
    int bucket = 1024; while (bucket < tokens) bucket *= 2;
    size_t sc = vround(std::max(LOCAL_SC, (size_t)GSUB * H * std::min(bucket, e->max_ctx) * 6));
    size_t add = sc > e->vsc.mapped ? sc - e->vsc.mapped : 0;
    for (int v = 0; v < NVIS; v++) if (e->vis_global[v]) add += 2 * ((size_t)tokens * kvt - std::min(e->vK[v][seq].mapped, (size_t)tokens * kvt));
    if (d.on) add += 2 * ((size_t)tokens * dkt - std::min(e->vdK[seq].mapped, (size_t)tokens * dkt));
    size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot));
    if (fr < add + e->reserve_bytes) return 1;
    bool ok = e->vsc.grow(sc);
    for (int v = 0; v < NVIS && ok; v++) if (e->vis_global[v]) ok = e->vK[v][seq].grow((size_t)tokens * kvt) && e->vV[v][seq].grow((size_t)tokens * kvt);
    if (ok && d.on) ok = e->vdK[seq].grow((size_t)tokens * dkt) && e->vdV[seq].grow((size_t)tokens * dkt);
    if (!ok) {                                            // someone else took the memory meanwhile: give back the part we got
        int had = e->ctx_mapped[seq]; e->ctx_mapped[seq] = tokens; ink_trim(ep, seq, had); e->ctx_mapped[seq] = std::min(e->ctx_mapped[seq], had);
        return 1;
    }
    e->ctx_mapped[seq] = tokens;
    return 0;
}
int ink_ctx_mapped(void* ep, int seq) { auto* e = (Engine*)ep; return seq < 0 || seq >= e->nseq ? 0 : e->ctx_mapped[seq]; }
// Give back conversation `seq`'s memory above `tokens` positions (whole attachments only, so it may keep a little more).
void ink_trim(void* ep, int seq, int tokens) {
    auto* e = (Engine*)ep; auto& d = e->dr;
    if (seq < 0 || seq >= e->nseq || tokens < 0 || tokens >= e->ctx_mapped[seq]) return;
    CK(cudaStreamSynchronize(g_stream));
    size_t kvt = (size_t)HKV * DH * (e->kv8 ? 1 : 2), dkt = (size_t)HKV * DH * 2, left = (size_t)e->max_ctx;
    for (int v = 0; v < NVIS; v++) if (e->vis_global[v]) {
        e->vK[v][seq].shrink((size_t)tokens * kvt); e->vV[v][seq].shrink((size_t)tokens * kvt);
        left = std::min(left, std::min(e->vK[v][seq].mapped, e->vV[v][seq].mapped) / kvt);
    }
    if (d.on) { e->vdK[seq].shrink((size_t)tokens * dkt); e->vdV[seq].shrink((size_t)tokens * dkt); left = std::min(left, std::min(e->vdK[seq].mapped, e->vdV[seq].mapped) / dkt); }
    e->ctx_mapped[seq] = (int)left;
    if ((int)left < e->snap_tokens[seq]) e->snap_ok[seq] = false;          // the snapshot's prefix is gone
    int most = 0; for (int q = 0; q < e->nseq; q++) most = std::max(most, e->ctx_mapped[q]);
    int bucket = 1024; while (bucket < most) bucket *= 2;                  // the score scratch follows the longest conversation left
    e->vsc.shrink(std::max(LOCAL_SC, (most ? (size_t)GSUB * H * std::min(bucket, e->max_ctx) : 0) * 6));
}

// Free memory on the GPU right now, in bytes (used to size the context before loading).
size_t ink_vram_free() { size_t fr = 0, tot = 0; CK(cudaMemGetInfo(&fr, &tot)); return fr; }

// max_ctx: the longest conversation allowed (memory for it is attached as conversations grow, see ink_ensure).
// reserve_mib: GPU memory that must stay free for other programs when attaching more.
void* ink_new(int max_ctx, int nseq, int reserve_mib) {
    auto* e = new Engine(); CK(cudaStreamCreate(&g_stream));
    gemv_scratch();
    CK(cudaFuncSetAttribute(k_tasks, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TASKS_SHARED));
    CK(cudaFuncSetAttribute(k_tasks_w, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TASKS_SHARED));
    e->max_ctx = std::max(CTX_STEP, (max_ctx + CTX_STEP - 1) / CTX_STEP * CTX_STEP); e->nseq = std::min(MAXSEQ, std::max(1, nseq));
    e->reserve_bytes = (size_t)std::max(0, reserve_mib) << 20;
    return e;
}

// Copy one checkpoint tensor to the GPU. Tensors up to 2 MB also keep a host copy for conversion in ink_finalize.
void ink_upload(void* ep, const char* name, const void* data, size_t bytes) {
    auto* e = (Engine*)ep; Raw r; r.bytes = bytes;
    CK(cudaMalloc(&r.dev, bytes)); CK(cudaMemcpy(r.dev, data, bytes, cudaMemcpyHostToDevice));
    if (bytes <= (2u << 20)) r.host.assign((const uint8_t*)data, (const uint8_t*)data + bytes);
    e->raw[name] = std::move(r);
}

void ink_finalize(void* ep) {
    auto* e = (Engine*)ep; const std::string M = "model.llm.";
    const float qs = 1.0f / 48.0f;                       // AttnRes and loop queries are pre-scaled by D^-0.5
    e->embed = big<uint16_t>(e, M + "embed.weight"); e->unembed = big<uint16_t>(e, M + "unembed.weight");
    e->embed_norm = small_f32(e, M + "embed_norm.weight", true); e->norm = small_f32(e, M + "norm.weight", true);
    for (int m = 0; m < 23; m++) e->attn_q[m] = small_f32(e, M + "attn_res.attn_queries." + std::to_string(m), false, qs);
    for (int n = 0; n < 24; n++) e->mlp_q[n] = small_f32(e, M + "attn_res.mlp_queries." + std::to_string(n), false, qs);
    e->out_q = small_f32(e, M + "attn_res.output_query", false, qs);
    {   // message queries per core layer: rows (attn, k, v, mlp)
        const Raw& a = get(e, M + "middle_cycle_cross_pass_queries"); const Raw& kv = get(e, M + "middle_cycle_cross_pass_kv_queries");
        const float* af = (const float*)a.host.data(); const float* kf = (const float*)kv.host.data();
        for (int j = 0; j < 12; j++) {
            std::vector<float> v(4 * D);
            for (int c = 0; c < D; c++) { v[c] = af[(2 * j) * D + c] * qs; v[D + c] = kf[(2 * j) * D + c] * qs; v[2 * D + c] = kf[(2 * j + 1) * D + c] * qs; v[3 * D + c] = af[(2 * j + 1) * D + c] * qs; }
            CK(cudaMalloc(&e->msg_q[j], v.size() * 4)); CK(cudaMemcpy(e->msg_q[j], v.data(), v.size() * 4, cudaMemcpyHostToDevice));
        }
    }
    for (int i = 0; i < 24; i++) {
        Layer& L = e->L[i]; const std::string P = M + "layers." + std::to_string(i) + ".";
        L.global = (i % 6 == 5); L.core = (i >= 6 && i < 18);
        CK(cudaMalloc(&L.qkvr, (size_t)QKVR * D * 2));
        const char* parts[6] = {"attn.wq_du.weight", "attn.wr_du.weight", "attn.gate_attn.weight", "attn.attn_sink_proj.weight", "attn.wk_dv.weight", "attn.wv_dv.weight"};
        size_t off = 0;
        for (auto p : parts) { const Raw& r = get(e, P + p); CK(cudaMemcpy((uint8_t*)L.qkvr + off, r.dev, r.bytes, cudaMemcpyDeviceToDevice)); off += r.bytes; CK(cudaFree(r.dev)); }
        if (off != (size_t)QKVR * D * 2) { fprintf(stderr, "qkvr size mismatch\n"); abort(); }
        L.wo = big<uint16_t>(e, P + "attn.wo_ud.weight"); L.router = big<uint16_t>(e, P + "mlp.gate.weight");
        L.qscale = small_f32(e, P + "attn.q_l2_clip.scale", true); L.rscale = small_f32(e, P + "attn.rel_l2_clip.scale", true);
        L.rel = small_f32(e, P + "attn.rel_logits_proj.proj", true); L.mlp_norm = small_f32(e, P + "mlp_norm.weight", true);
        L.attn_sconv = small_f32(e, P + "attn_sconv.weight", true); L.mlp_sconv = small_f32(e, P + "mlp_sconv.weight", true);
        L.rbias = small_f32(e, P + "mlp.gate.bias", false);
        if (L.global) {
            std::vector<float> v(3 * D);
            const char* nm[3] = {"q_input_norm.weight", "k_input_norm.weight", "v_input_norm.weight"};
            for (int s = 0; s < 3; s++) { const uint16_t* h = (const uint16_t*)get(e, P + nm[s]).host.data(); for (int c = 0; c < D; c++) { uint32_t u = (uint32_t)h[c] << 16; memcpy(&v[s * D + c], &u, 4); } }
            CK(cudaMalloc(&L.in_norm, v.size() * 4)); CK(cudaMemcpy(L.in_norm, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
            L.kconv = small_f32(e, P + "attn.k_sconv.static_w", true); L.vconv = small_f32(e, P + "attn.v_sconv.static_w", true);
            L.kproj = big<uint16_t>(e, P + "attn.k_sconv.weight_proj"); L.vproj = big<uint16_t>(e, P + "attn.v_sconv.weight_proj");
        } else {
            L.in_norm = small_f32(e, P + "attn_norm.weight", true);
            L.kconv = small_f32(e, P + "attn.k_sconv.weight", true); L.vconv = small_f32(e, P + "attn.v_sconv.weight", true);
        }
        if (L.core) {
            const char* st[4] = {"attn", "k", "v", "mlp"}; const char* wn[3] = {"down_weight", "gain_weight", "forget_weight"};
            for (int s = 0; s < 4; s++) { L.ctl_down[s] = big<uint16_t>(e, P + "controllers." + st[s] + "." + wn[0]); for (int w = 1; w < 3; w++) L.ctl[s][w] = small_f32(e, P + "controllers." + st[s] + "." + wn[w], true); }
            const size_t db = (size_t)64 * 2 * D * 2, gb = (size_t)D * 64 * 4;
            CK(cudaMalloc(&L.ctl3_down, 3 * db)); CK(cudaMalloc(&L.ctl3_gain, 3 * gb)); CK(cudaMalloc(&L.ctl3_forget, 3 * gb));
            for (int s = 0; s < 3; s++) {
                CK(cudaMemcpy((uint8_t*)L.ctl3_down + s * db, L.ctl_down[s], db, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy((uint8_t*)L.ctl3_gain + s * gb, L.ctl[s][1], gb, cudaMemcpyDeviceToDevice));
                CK(cudaMemcpy((uint8_t*)L.ctl3_forget + s * gb, L.ctl[s][2], gb, cudaMemcpyDeviceToDevice));
            }
        }
        L.w13 = big<uint8_t>(e, P + "mlp.experts.w13_weight"); L.w13s = big<uint8_t>(e, P + "mlp.experts.w13_weight_scale");
        L.w2 = big<uint8_t>(e, P + "mlp.experts.w2_weight"); L.w2s = big<uint8_t>(e, P + "mlp.experts.w2_weight_scale");
        L.fp8 = e->raw.find(P + "mlp.experts.w13_weight_amax") == e->raw.end();
        if (!L.fp8) {
            L.w13amax = small_f32(e, P + "mlp.experts.w13_weight_amax", false); L.w2amax = small_f32(e, P + "mlp.experts.w2_weight_amax", false);
            L.inv_den13 = 1.0f / *(const float*)get(e, P + "mlp.experts.w13_weight_outer_scale_denominator").host.data();
            L.inv_den2 = 1.0f / *(const float*)get(e, P + "mlp.experts.w2_weight_outer_scale_denominator").host.data();
        }
    }
    // ---- tile layout for everything the tensor cores read
    // INK_FP8=1: the lossy "8-bit dense" mode (attention matrices + output head in MXFP8); either part can be chosen alone.
    bool fp8_all = getenv("INK_FP8") != nullptr && atoi(getenv("INK_FP8")) != 0;
    e->kv8 = getenv("INK_KV_FP8") != nullptr && atoi(getenv("INK_KV_FP8")) != 0;
    e->fp8_attn = fp8_all || getenv("INK_FP8_ATTN") != nullptr; e->fp8_head = fp8_all || getenv("INK_FP8_HEAD") != nullptr;
    if (e->fp8_head) e->f_head = to_fp8(e->unembed, V0, D);
    CK(cudaMalloc(&e->tq, (size_t)TMAX * 3 * H * DH)); CK(cudaMalloc(&e->ts, (size_t)TMAX * 3 * H * DH / 32));
    e->unembed_rm = e->unembed;                                                  // row-major copy, needed once more for the drafter's 8-bit head
    e->unembed = (uint16_t*)swizzle(e->unembed, 201024, D * 2, V0, V0, false);   // only the 200,058 real rows
    for (int i = 0; i < 24; i++) {
        Layer& L = e->L[i];
        if (e->fp8_attn) {                                 // 8-bit copies replace the 16-bit matrices
            L.f_q = to_fp8(L.qkvr, R_K, D); L.f_k = to_fp8(L.qkvr + (size_t)R_K * D, 1024, D); L.f_v = to_fp8(L.qkvr + (size_t)R_V * D, 1024, D); L.f_wo = to_fp8(L.wo, D, H * DH);
            CK(cudaFree(L.qkvr)); CK(cudaFree(L.wo)); L.qkvr = nullptr; L.wo = nullptr;
        } else {
        L.qkvr = (uint16_t*)swizzle(L.qkvr, QKVR, D * 2, R_K, R_V);
        L.wo = (uint16_t*)swizzle(L.wo, D, H * DH * 2);
        }
        L.router = (uint16_t*)swizzle(L.router, NE, D * 2);
        if (L.global) { L.kproj = (uint16_t*)swizzle(L.kproj, 32, D * 2); L.vproj = (uint16_t*)swizzle(L.vproj, 32, D * 2); }
        if (L.core) { L.ctl3_down = (uint16_t*)swizzle(L.ctl3_down, 192, 4 * D); L.ctl_down[3] = (uint16_t*)swizzle(L.ctl_down[3], 64, 4 * D); }
        if (L.fp8) {
            L.w13 = swizzle(L.w13, (size_t)NE * D, D); L.w13s = swizzle_scale(L.w13s, (size_t)NE * D, D / 32, 1);
            L.w2 = swizzle(L.w2, (size_t)NE * D, I2); L.w2s = swizzle_scale(L.w2s, (size_t)NE * D, I2 / 32, 1);
        } else {
            L.w13 = swizzle(L.w13, (size_t)NE * D, D / 2); L.w13s = swizzle_scale(L.w13s, (size_t)NE * D, D / 16, 4);
            L.w2 = swizzle(L.w2, (size_t)NE * D, I2 / 2); L.w2s = swizzle_scale(L.w2s, (size_t)NE * D, I2 / 16, 4);
        }
    }
    // The drafter is built first so that its temporary buffers (and the row-major output head it needs once) are
    // gone before the context-sized caches are allocated: that peak decides how much context fits.
    // ---- DSpark drafter
    if (e->raw.count("mtp.fuse_weight")) {
        auto& d = e->dr; d.on = true; const std::string P = "mtp.";
        auto cat = [&](std::vector<std::string> names, size_t rows, int RB, size_t r1 = 0, size_t r2 = 0) {   // concatenate matrices by rows, then tile
            size_t tot = 0; for (auto& n : names) tot += get(e, P + n).bytes;
            uint8_t* buf; CK(cudaMalloc(&buf, tot)); size_t off = 0;
            for (auto& n : names) { const Raw& r = get(e, P + n); CK(cudaMemcpy(buf + off, r.dev, r.bytes, cudaMemcpyDeviceToDevice)); off += r.bytes; CK(cudaFree(r.dev)); }
            return (uint16_t*)swizzle(buf, rows, RB, r1, r2);
        };
        auto L_ = [](int i, const char* n) { return "layers." + std::to_string(i) + "." + n; };
        d.fp8 = getenv("INK_DRAFT_BF16") == nullptr;      // default: 8-bit drafter
        auto cat_fp8 = [&](std::vector<std::string> names, size_t rows, int C) {     // row-major concatenation -> MXFP8 tiles (sources freed)
            size_t tot = 0; for (auto& n : names) tot += get(e, P + n).bytes;
            uint8_t* buf; CK(cudaMalloc(&buf, tot)); size_t off = 0;
            for (auto& n : names) { const Raw& r = get(e, P + n); CK(cudaMemcpy(buf + off, r.dev, r.bytes, cudaMemcpyDeviceToDevice)); off += r.bytes; CK(cudaFree(r.dev)); }
            Fp8Mat m = to_fp8((const uint16_t*)buf, rows, C); CK(cudaFree(buf)); return m;
        };
        d.kvctx = cat({L_(0, "self_attn.k_proj_ctx.weight"), L_(0, "self_attn.v_proj_ctx.weight"), L_(1, "self_attn.k_proj_ctx.weight"), L_(1, "self_attn.v_proj_ctx.weight"),
                       L_(2, "self_attn.k_proj_ctx.weight"), L_(2, "self_attn.v_proj_ctx.weight")}, 6144, D * 2, 2048, 4096);
        std::vector<float> kcn(3 * DH), cn(4 * D);
        auto host_bf16 = [&](const std::string& n, float* dst, size_t cnt) { const uint16_t* h = (const uint16_t*)get(e, P + n).host.data(); for (size_t i = 0; i < cnt; i++) { uint32_t u = (uint32_t)h[i] << 16; memcpy(dst + i, &u, 4); } };
        for (int i = 0; i < 3; i++) {
            if (d.fp8) {
                d.f_qkv[i] = cat_fp8({L_(i, "self_attn.q_proj.weight"), L_(i, "self_attn.k_proj.weight"), L_(i, "self_attn.v_proj.weight")}, DQ + 2 * DKV, D);
                d.f_o[i] = cat_fp8({L_(i, "self_attn.o_proj.weight")}, D, DQ);
                d.f_gu[i] = cat_fp8({L_(i, "mlp.gate_proj.weight"), L_(i, "mlp.up_proj.weight")}, 2 * DI, D);
                d.f_down[i] = cat_fp8({L_(i, "mlp.down_proj.weight")}, D, DI);
            } else {
                d.qkv[i] = cat({L_(i, "self_attn.q_proj.weight"), L_(i, "self_attn.k_proj.weight"), L_(i, "self_attn.v_proj.weight")}, DQ + 2 * DKV, D * 2);
                d.o[i] = cat({L_(i, "self_attn.o_proj.weight")}, D, DQ * 2);
                d.gu[i] = cat({L_(i, "mlp.gate_proj.weight"), L_(i, "mlp.up_proj.weight")}, 2 * DI, D * 2);
                d.down[i] = cat({L_(i, "mlp.down_proj.weight")}, D, DI * 2);
            }
            d.in_norm[i] = small_f32(e, P + L_(i, "input_layernorm.weight"), true); d.post_norm[i] = small_f32(e, P + L_(i, "post_attention_layernorm.weight"), true);
            d.qn[i] = small_f32(e, P + L_(i, "self_attn.q_norm.weight"), true); d.kn[i] = small_f32(e, P + L_(i, "self_attn.k_norm.weight"), true);
            host_bf16(L_(i, "self_attn.k_norm_ctx.weight"), kcn.data() + i * DH, DH); host_bf16(L_(i, "dflare_norm.weight"), cn.data() + i * D, D);
            size_t slots = i == 1 ? (size_t)e->max_ctx : RING, one = slots * HKV * DH * 2;
            (i == 1 ? d.gstride : d.lstride) = one;
            if (i == 1) {
                CUdeviceptr bk = vreserve(one * e->nseq), bv = vreserve(one * e->nseq);
                d.Kd[i] = (uint16_t*)bk; d.Vd[i] = (uint16_t*)bv;
                for (int q = 0; q < e->nseq; q++) { e->vdK[q].base = bk + q * one; e->vdV[q].base = bv + q * one; e->vdK[q].reserved = e->vdV[q].reserved = one; }
            } else { CK(cudaMalloc(&d.Kd[i], one * e->nseq)); CK(cudaMalloc(&d.Vd[i], one * e->nseq)); }
        }
        host_bf16("markov_head.dflare_norm.weight", cn.data() + 3 * D, D);
        CK(cudaMalloc(&d.kcn, kcn.size() * 4)); CK(cudaMemcpy(d.kcn, kcn.data(), kcn.size() * 4, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&d.ctx_norm, cn.size() * 4)); CK(cudaMemcpy(d.ctx_norm, cn.data(), cn.size() * 4, cudaMemcpyHostToDevice));
        {   // alpha = bf16(softmax(fuse_weight, dim = -1))
            float fw[16], al[16]; host_bf16("fuse_weight", fw, 16);
            for (int r = 0; r < 4; r++) { float mx = fw[r * 4], sm = 0; for (int k = 1; k < 4; k++) mx = std::max(mx, fw[r * 4 + k]); for (int k = 0; k < 4; k++) sm += std::exp(fw[r * 4 + k] - mx);
                for (int k = 0; k < 4; k++) { float v = std::exp(fw[r * 4 + k] - mx) / sm; uint32_t u; memcpy(&u, &v, 4); u += 0x7FFFu + ((u >> 16) & 1u); u &= 0xFFFF0000u; memcpy(&al[r * 4 + k], &u, 4); } }
            CK(cudaMalloc(&d.alpha, 64)); CK(cudaMemcpy(d.alpha, al, 64, cudaMemcpyHostToDevice));
        }
        d.norm = small_f32(e, P + "norm.weight", true); d.mask = small_f32(e, P + "mask_embed", true); d.rnn_b = small_f32(e, P + "markov_head.rnn_proj.bias", true);
        d.w1 = big<uint16_t>(e, P + "markov_head.markov_w1.weight");
        {   // rnn_proj.weight [768][5120]: columns 0..511 = [state | w1(prev)], columns 512.. = [hidden | f_rnn]
            const Raw& r = get(e, P + "markov_head.rnn_proj.weight"); std::vector<uint16_t> w(768 * 5120), a(768 * 512), b(768 * 4608);
            CK(cudaMemcpy(w.data(), r.dev, r.bytes, cudaMemcpyDeviceToHost));
            for (int row = 0; row < 768; row++) { memcpy(&a[row * 512], &w[row * 5120], 1024); memcpy(&b[(size_t)row * 4608], &w[row * 5120 + 512], 9216); }
            uint8_t *da, *db; CK(cudaMalloc(&da, a.size() * 2)); CK(cudaMalloc(&db, b.size() * 2));
            CK(cudaMemcpy(da, a.data(), a.size() * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 2, cudaMemcpyHostToDevice));
            d.rnn_sw = (uint16_t*)swizzle(da, 768, 1024); d.rnn_hf = (uint16_t*)swizzle(db, 768, 9216);
        }
        {   // Fold the head: M = bf16(unembed @ hidden_out.weight) [V0][256], bvec = unembed @ hidden_out.bias [V0]
            std::vector<float> how(D * MR), x((size_t)MR * D), hb(D); host_bf16("markov_head.hidden_out.weight", how.data(), how.size()); host_bf16("markov_head.hidden_out.bias", hb.data(), D);
            for (int c = 0; c < D; c++) for (int j = 0; j < MR; j++) x[(size_t)j * D + c] = how[(size_t)c * MR + j];
            float *dx, *tmp = dalloc((size_t)HMAX * V0); CK(cudaMalloc(&dx, x.size() * 4)); CK(cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice));
            uint16_t* M; CK(cudaMalloc(&M, (size_t)V0 * MR * 2));
            for (int c0 = 0; c0 < MR; c0 += HMAX) {
                gemv(V0, HMAX, e->unembed, D, HMAX, dx + (size_t)c0 * D, dx + (size_t)c0 * D, dx + (size_t)c0 * D, D, V0, V0, tmp, V0, 0);
                k_fold<<<(unsigned)(((size_t)V0 * HMAX + 255) / 256), 256, 0, g_stream>>>(tmp, M, c0, HMAX);
            }
            CK(cudaMemcpy(dx, hb.data(), D * 4, cudaMemcpyHostToDevice));
            d.bvec = dalloc(V0); gemv(V0, 1, e->unembed, D, 1, dx, dx, dx, D, V0, V0, d.bvec, V0, 0);
            CK(cudaStreamSynchronize(g_stream)); CK(cudaFree(dx)); CK(cudaFree(tmp));
            if (d.fp8) {
                d.f_M = to_fp8(M, V0, MR);
                e->h_eu = dalloc(V0); e->h_un = dalloc(V0);
                d.f_un = to_fp8(e->unembed_rm, V0, D, e->h_eu, e->h_un);
                e->head2 = !e->fp8_head && getenv("INK_HEAD_FULL") == nullptr;   // the two-stage head needs this 8-bit copy and the exact matrix
            }
            d.M = (uint16_t*)swizzle(M, V0, MR * 2);
        }
        const size_t T2 = TMAX;
        d.dctx = dalloc(T2 * 4 * D); d.dkv = dalloc(T2 * 6144); d.resid = dalloc(DBATCH * DB * D); d.h = dalloc(DBATCH * DB * D); d.qkvo = dalloc(DBATCH * DB * (DQ + 2 * DKV)); d.qd = dalloc(DBATCH * DB * H * DH);
        d.att = dalloc(DBATCH * DB * DQ); d.hp = dalloc(DBATCH * DB * D); d.guo = dalloc(DBATCH * DB * 2 * DI); d.act = dalloc(DBATCH * DB * DI); d.base = dalloc((size_t)DBATCH * (DB - 1) * V0);
        d.cin = dalloc(DBATCH * (DB - 1) * 2 * D); d.hsel = dalloc(DBATCH * (DB - 1) * D);
        d.C = dalloc(DBATCH * (DB - 1) * 768); d.mkin = dalloc(DBATCH * 512); d.sw = dalloc(DBATCH * 768); d.state = dalloc(DBATCH * MR); d.lat = dalloc(DBATCH * MR); d.mb = dalloc((size_t)DBATCH * V0);
        d.pv = dalloc(DBATCH * HEAD_PARTS); d.psum = dalloc(DBATCH * HEAD_PARTS); d.conf = dalloc(DBATCH * DB);
        d.px = dalloc(DBATCH * HEAD_PARTS); d.pmx = dalloc(DBATCH * HEAD_PARTS); d.top = dalloc(DBATCH * DB); d.qit = dalloc((size_t)e->nseq * (DB - 1)); d.dsamp = dalloc(DBATCH * 2); d.qlog = dalloc((size_t)e->nseq * (DB - 1) * V0);
        CK(cudaMalloc(&d.psumd, DBATCH * HEAD_PARTS * 8)); CK(cudaMalloc(&d.qz, (size_t)e->nseq * (DB - 1) * 8)); CK(cudaMemset(d.qz, 0, (size_t)e->nseq * (DB - 1) * 8));
        CK(cudaMalloc(&d.qc, (size_t)DBATCH * DB * 2 * DI)); CK(cudaMalloc(&d.qs, (size_t)DBATCH * DB * 2 * DI / 32));
        d.frnn = dalloc((size_t)e->nseq * D);
        CK(cudaMalloc(&d.anchor, DBATCH * 4)); CK(cudaMalloc(&d.L, DBATCH * 4)); CK(cudaMalloc(&d.dseq, DBATCH * 4)); CK(cudaMalloc(&d.prev, DBATCH * 4)); CK(cudaMalloc(&d.out, DBATCH * DB * 4)); CK(cudaMalloc(&d.pi, DBATCH * HEAD_PARTS * 4));
    }
    if (e->unembed_rm) { CK(cudaFree(e->unembed_rm)); e->unembed_rm = nullptr; }
    if (e->fp8_head && (!e->dr.on || e->dr.fp8)) { CK(cudaFree(e->unembed)); e->unembed = nullptr; }   // the tiled 16-bit head is no longer read
    // per-visit state
    int v = 0;
    auto add_visit = [&](int layer) {
        size_t slots = e->L[layer].global ? (size_t)e->max_ctx : RING;
        size_t kv1 = slots * HKV * DH * (e->kv8 ? 1 : 2), kvb = kv1 * e->nseq;     // one cache per conversation, back to back
        (e->L[layer].global ? e->kv_global : e->kv_local) = kv1;
        e->vis_global[v] = e->L[layer].global;
        if (e->L[layer].global) {                        // position-indexed: reserve addresses now, attach memory as conversations grow
            CUdeviceptr bk = vreserve(kvb), bv = vreserve(kvb);
            e->Kc[v] = (decltype(e->Kc[v]))bk; e->Vc[v] = (decltype(e->Vc[v]))bv;
            for (int q = 0; q < e->nseq; q++) { e->vK[v][q].base = bk + q * kv1; e->vV[v][q].base = bv + q * kv1; e->vK[v][q].reserved = e->vV[v][q].reserved = kv1; }
        } else {                                         // stale slots get P = 0, so they must hold finite numbers (attached memory is zero-filled too)
            CK(cudaMalloc(&e->Kc[v], kvb)); CK(cudaMalloc(&e->Vc[v], kvb)); CK(cudaMemset(e->Kc[v], 0, kvb)); CK(cudaMemset(e->Vc[v], 0, kvb));
        }
        e->ring[v][0] = dalloc((size_t)e->nseq * CR * 1024); e->ring[v][1] = dalloc((size_t)e->nseq * CR * 1024);
        e->ring[v][2] = dalloc((size_t)e->nseq * CR * D); e->ring[v][3] = dalloc((size_t)e->nseq * CR * D); v++;
    };
    for (int i = 0; i < 6; i++) add_visit(i);
    for (int p = 0; p < 3; p++) for (int i = 6; i < 18; i++) add_visit(i);
    for (int i = 18; i < 24; i++) add_visit(i);
    // scratch
    const size_t T = TMAX;
    for (auto& s : e->stack) s = dalloc(T * D);
    e->prelude_partial = dalloc(T * D); e->bank[0] = dalloc(T * D); e->bank[1] = dalloc(T * D); e->prelude_dests = dalloc(T * D);
    e->inputs = dalloc(T * 3 * D); e->msg = dalloc(T * 4 * D); e->xin = dalloc(T * 3 * D); e->proj = dalloc(T * QKVR);
    e->dk = dalloc(T * 32); e->dv = dalloc(T * 32); e->qc = dalloc(T * H * DH); e->bias = dalloc(T * H * EXT);
    // Score scratch: T x H x RING for the sliding-window layers; GSUB x H x (context bucket) for full attention, attached on demand.
    // scores as fp32 and, behind them, the probabilities again as bf16 operands: 4 + 2 bytes per (row, head, slot)
    e->vsc.reserved = vround(std::max(LOCAL_SC, (size_t)GSUB * H * (size_t)e->max_ctx * 6)); e->vsc.base = vreserve(e->vsc.reserved);
    if (!e->vsc.grow(LOCAL_SC)) { fprintf(stderr, "out of GPU memory for the score scratch\n"); abort(); }
    CK(cudaMalloc(&e->grp, TMAX * sizeof(Grp))); CK(cudaMalloc(&e->gb, (TMAX / 16 + 2) * 4)); CK(cudaMalloc(&e->rowg, TMAX * 4)); CK(cudaMalloc(&e->cso, (TMAX + 1) * 4));
    e->att_sc = (float*)e->vsc.base; e->att_l = dalloc(T * H * 2); e->att_y = dalloc(T * H * DH); e->attn_raw = dalloc(T * D); e->attn_out = dalloc(T * D);
    e->mixtmp = dalloc(T * D); e->pair = dalloc(T * 3 * 2 * D); e->hbuf = dalloc(T * 192); e->gx = dalloc(T * 3 * D); e->mlp_in = dalloc(T * D); e->rlogit = dalloc(T * NE); e->gamma = dalloc(T * TOPK); e->xq = dalloc(T * D);
    e->xamax = dalloc(T); e->gu = dalloc(T * TOPK * D); e->act = dalloc(T * TOPK * I2); e->aq = dalloc(T * TOPK * I2); e->aamax = dalloc(T * TOPK);
    e->down = dalloc(T * TOPK * D); e->moe_out = dalloc(T * D); e->y = dalloc(T * D); e->out = dalloc(T * D); e->logits = dalloc(HMAX * (size_t)V0);
    CK(cudaMalloc(&e->tok, T * 4)); CK(cudaMalloc(&e->ids, T * TOPK * 4)); CK(cudaMalloc(&e->arg, T * 4)); CK(cudaMalloc(&e->rows, T * sizeof(Row))); e->feats = dalloc(4 * T * D); e->head_v = dalloc(T * HEAD_PARTS); e->rowmax = dalloc(T); CK(cudaMalloc(&e->head_i, T * HEAD_PARTS * 4));
    CK(cudaMalloc(&e->sparts, T * HEAD_PARTS * sizeof(SampPart))); e->sparam = dalloc(2 * T); CK(cudaMalloc(&e->excl, 2 * TMAX * 4)); e->qrow = e->excl + TMAX; e->rowz = dalloc(T); CK(cudaMalloc(&e->rparts, (size_t)T * HEAD_PARTS * sizeof(ResPart))); CK(cudaMalloc(&e->sres, T * 16 + 8));
    e->h_hb = dalloc(2 * FAST_T); e->h_pv = dalloc((size_t)FAST_T * HEAD_PARTS * 2); e->h_thr = dalloc(2 * FAST_T);
    e->h_ex1 = dalloc((size_t)2 * FAST_T * 16 * FAST_T); e->h_ex2 = dalloc((size_t)CAND_CAP * 16 * FAST_T);
    CK(cudaMalloc(&e->h_pi, (size_t)FAST_T * HEAD_PARTS * 2 * 4)); CK(cudaMalloc(&e->h_win, 2 * FAST_T * 4)); CK(cudaMalloc(&e->h_l1, 2 * FAST_T * 4));
    CK(cudaMalloc(&e->h_l2, CAND_CAP * 4)); CK(cudaMalloc(&e->h_cnt, 4)); CK(cudaMemset(e->h_cnt, 0, 4)); CK(cudaMemset(e->h_l2, 0, CAND_CAP * 4));
    CK(cudaMalloc(&e->tasks, T * TOPK * TASKW * 4)); CK(cudaMalloc(&e->ntasks, 4));
    CK(cudaMalloc(&e->tasksw, (size_t)max_wtasks(TMAX * TOPK) * TASKWW * 4)); CK(cudaMalloc(&e->xb, (size_t)max_tasks(TMAX * TOPK) * (D / 64) * 32 * sizeof(uint4)));
    CK(cudaMalloc(&e->xcode, T * (D / 2))); CK(cudaMalloc(&e->xscale, T * (D / 16)));
    CK(cudaMalloc(&e->xcode8, T * D)); CK(cudaMalloc(&e->acode8, T * TOPK * I2));
    CK(cudaMalloc(&e->acode, T * TOPK * (I2 / 2))); CK(cudaMalloc(&e->ascale, T * TOPK * (I2 / 16)));

    if (getenv("INK_NO_PREFIX_CACHE") == nullptr) {       // ring snapshots for the prefix cache
        size_t b = 0; for (int v2 = 0; v2 < NVIS; v2++) b += (size_t)CR * (1024 + 1024 + D + D) * 4 + (e->vis_global[v2] ? 0 : 2 * e->kv_local);
        if (e->dr.on) b += 4 * e->dr.lstride + (size_t)D * 4;
        e->snap_bytes = b; CK(cudaMalloc(&e->snap, b * e->nseq));
    }
    CK(cudaDeviceSynchronize());
    size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
    fprintf(stderr, "engine ready: %.1f GiB VRAM in use on the card, %.1f GiB free%s%s%s\n", (tot - fr) / 1073741824.0, fr / 1073741824.0,
            e->fp8_attn ? "; 8-bit attention matrices (lossy)" : "", e->fp8_head ? "; 8-bit output head (lossy)" : "", e->kv8 ? "; 8-bit key/value cache (lossy)" : "");
}

void ink_set_trace(void* ep, float* host_buf) { ((Engine*)ep)->trace = host_buf; }

// The kernel sequence for T tokens. It depends only on T (not on the position or the tokens, which the
// kernels read from device memory), so it can be captured once as a CUDA graph and replayed.
// v2: fused kernels (about half as many launches per token as v1).
// The output head for the last step's rows (e->out), in full: every logit, the soft cap, argmax, sampling.
static void head_full(Engine* e, int T) {
    if (e->fp8_head) {                                   // lossy option: 8-bit output head
        k_quant8f<<<dim3((D) / 128, T), 32, 0, g_stream>>>(e->out, D, e->tq, e->ts, D);
        gemm_fp8(e->f_head, D, T, e->tq, e->ts, e->logits, 0, V0);
    } else gemv(V0, T, e->unembed, D, T, e->out, e->out, e->out, D, V0, V0, e->logits, V0, 0);
    prof("unembed");
    k_head_part<<<dim3(HEAD_PARTS, T), 1024, 0, g_stream>>>(e->logits, e->head_v, e->head_i);
    k_head_final<<<T, 32, 0, g_stream>>>(e->head_v, e->head_i, e->arg, e->rowmax);
    k_samp_part<<<dim3(HEAD_PARTS, T), 1024, 0, g_stream>>>(e->logits, e->sparam, e->excl, e->rows, e->rowmax, e->sparts);
    k_samp_final<<<T, 32, 0, g_stream>>>(e->sparts, e->sparam, e->arg, e->rowmax, e->sres, e->rowz);
    if (e->dr.on) {
        k_resid_part<<<dim3(HEAD_PARTS, T), 1024, 0, g_stream>>>(e->logits, e->sparam, e->qrow, e->rows, e->rowmax, e->rowz, e->dr.qlog, e->dr.qz, e->dr.qit, e->rparts);
        k_resid_final<<<T, 32, 0, g_stream>>>(e->rparts, e->qrow, e->sparam, e->sres);
    }
}
static void gemm_rows(const uint16_t* W, int C, int T, const float* x, int xstride, const int* list, const int* count, int nmax, float* out, int ostride) {
    switch ((T + 7) / 8) {
        #define ROWS(n) case n: k_gemm_rows<n><<<nmax, 512, 0, g_stream>>>(W, C, T, x, xstride, list, count, nmax, out, ostride); break;
        ROWS(1) ROWS(2) ROWS(3) default: ROWS(4)
        #undef ROWS
    }
}
// The two-stage head (see k_hbound and the kernels after it): the winners only, exactly, for T <= FAST_T rows.
static void head_two_stage(Engine* e, int T) {
    auto& d = e->dr;
    k_quant8f<<<dim3((D) / 128, T), 32, 0, g_stream>>>(e->out, D, e->tq, e->ts, D);
    gemm_fp8(d.f_un, D, T, e->tq, e->ts, e->logits, 0, V0);                              // approximate logits
    prof("unembed");
    k_hbound<<<T, 256, 0, g_stream>>>(e->out, e->tq, e->ts, e->h_hb);
    k_apx_part<<<dim3(HEAD_PARTS, T), 1024, 0, g_stream>>>(e->logits, e->sparam, e->rows, e->h_pv, e->h_pi);
    k_apx_final<<<T, 32, 0, g_stream>>>(e->h_pv, e->h_pi, e->sparam, e->h_win, e->h_l1);
    gemm_rows(e->unembed, D, T, e->out, D, e->h_l1, nullptr, 2 * T, e->h_ex1, FAST_T);     // exact logits of the approximate winners
    k_head_thr<<<T, 32, 0, g_stream>>>(e->h_ex1, e->h_win, e->sparam, e->rows, e->h_thr, e->h_cnt);
    k_head_cand<<<dim3(HEAD_PARTS, T), 1024, 0, g_stream>>>(e->logits, e->h_eu, e->h_un, e->h_hb, e->h_thr, e->sparam, e->rows, e->h_l2, e->h_cnt);
    gemm_rows(e->unembed, D, T, e->out, D, e->h_l2, e->h_cnt, CAND_CAP, e->h_ex2, FAST_T); // exact logits of the candidates
    k_head_pick<<<T, 1024, 0, g_stream>>>(e->h_ex2, e->h_l2, e->h_cnt, e->sparam, e->rows, e->sres, T);
}

// hm: the output head to run - 0 none, 1 full, 2 two-stage.
#ifndef SKG
#define SKG 1                                             // key groups per warp of the full-attention score kernel (4 / 8 / 16 measured slower: 73 / 86 / 131 us against 57 at 30K)
#endif
// Sliding-window attention of rows [h0, h0 + hn): scores, softmax, value sum (k_attn_*_l). ng: the grid's groups.
// Shapes as measured at 128 rows: scores 28.8 us with 8 warps per block, 22.2 with 12, 17.9 with 18 (24 do not fit a
// block's registers); softmax 12.7 us with 1 warp per (row, head), 10.7 with 2, no better with more; values 23 us
// with one warp per 16 dims and three key groups loaded ahead (33 loading one at a time; more warps are slower).
static void local_attention(Engine* e, int visit, size_t kvs, int h0, int hn, int ng, uint2* pl, int g0i, int g1i) {
    static const char* only = getenv("INK_LA_ONLY");     // timing: run one of the three (s, p, v)
    if (!only || *only == 's') k_attn_scores_l<18><<<ng * HKV, 18 * 32, 0, g_stream>>>(e->qc, e->bias, (const uint8_t*)e->Kc[visit], e->rows, e->grp, e->gb, g0i, g1i, e->att_sc, h0, e->kv8, kvs);
    if (!only || *only == 'p') k_attn_soft_l<2><<<dim3(hn, H), 2 * 32, 0, g_stream>>>(e->att_sc, e->att_l, h0, pl, e->rowg, e->grp, e->cso, e->gb, g0i);
    if (!only || *only == 'v') k_attn_pv_l<1, 3><<<dim3(8, ng * HKV), 32, 0, g_stream>>>(pl, (const uint8_t*)e->Vc[visit], e->att_l, e->proj, e->att_y, e->kv8, e->rows, kvs, e->grp, e->cso, e->gb, g0i, g1i);
}

// cls: 0 if every run of the step begins at a row that is a multiple of 8 (then a sub-batch of n rows has exactly
// ceil(n / 8) attention groups), 1 otherwise (up to n).
static void forward_body(Engine* e, int T, int gchunks, int hm, int cls) {
    // Prefetch plan (fractions of each matrix), tunable for experiments: INK_PF_WO=f, INK_PF_Q=a,b,c,d (cumulative).
    static const size_t WO_BYTES = (size_t)(D / 16) * (H * DH / 16) * 512, QKVR_BYTES_BF16 = (size_t)(410 + 64 + 64) * (D / 16) * 512;
    static double pf_wo = getenv("INK_PF_WO") ? atof(getenv("INK_PF_WO")) : 0.0, pf_q[4] = {0.1, 0.2, 0.3, 0.4};
    static const bool q4_native = getenv("INK_Q4_ALU") == nullptr;      // NVFP4 quantisers on native conversion instructions (default)
    static bool pf_init = false;
    if (!pf_init) { pf_init = true; if (const char* c = getenv("INK_PF_Q")) sscanf(c, "%lf,%lf,%lf,%lf", &pf_q[0], &pf_q[1], &pf_q[2], &pf_q[3]); }
    k_embed<<<T, 256, 0, g_stream>>>(e->tok, e->embed, e->embed_norm, e->stack[0]);
    if (g_prof) { CK(cudaStreamSynchronize(g_stream)); g_last = now(); }
    int nb = 0, Q = 0, visit = 0, N = T * TOPK, NTK = max_tasks(N); bool have_inputs = false, have_dests = false;
    for (int seg = 0; seg < 5; seg++) {
        int lo = seg == 0 ? 0 : (seg == 4 ? 18 : 6), hi = seg == 0 ? 6 : (seg == 4 ? 24 : 18), p = seg == 0 || seg == 4 ? 0 : seg - 1;
        for (int i = lo; i < hi; i++, visit++) {
            Layer& L = e->L[i]; bool looped = p > 0, first = (i % 6 == 0);
            const size_t kvs = L.global ? e->kv_global : e->kv_local;      // bytes between two conversations' caches
            if (i == 6 && p == 0) {
                CK(cudaMemcpyAsync(e->prelude_partial, e->stack[nb], (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream));
            } else if (i == 6) {
                CK(cudaMemcpyAsync(e->bank[0], e->stack[2], (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream));
                CK(cudaMemcpyAsync(e->bank[1], e->stack[3], (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream));
                nb = 1;
                CK(cudaMemcpyAsync(e->stack[1], e->prelude_partial, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream));
                if (!have_dests) { mix(e, T, {e->stack[0], e->prelude_partial}, nullptr, e->attn_q[5], 1, nullptr, e->prelude_dests); have_dests = true; }
                CK(cudaMemcpyAsync(e->inputs, e->prelude_dests, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream)); Q = 1;
            }
            float* partial = e->stack[nb];
            // DSpark taps (7.2): the running block sum at the start of visits 5, 23 and 41.
            if (visit == 5 || visit == 23 || visit == 41)
                CK(cudaMemcpyAsync(e->feats + (size_t)(visit / 18) * TMAX * D, partial, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream));
            const float *xq_, *xk_, *xv_; int xs;
            if (looped) {
                mix(e, T, {e->bank[0], e->bank[1]}, nullptr, e->msg_q[i - 6], 4, nullptr, e->msg);
                int ds = Q * D, doff = Q == 3 ? D : 0;
                k_pair3<<<dim3(T, 3), NT, 0, g_stream>>>(e->inputs, ds, doff, e->msg, e->pair);
                gemv(192, T, L.ctl3_down, 2 * D, T, e->pair, e->pair + 2 * D, e->pair + 4 * D, 6 * D, 64, 128, e->hbuf, 192, 1);
                if (T >= GATE_BIG) k_gate3_big<<<dim3(D / 8, 3), 256, 0, g_stream>>>(e->hbuf, L.ctl3_gain, L.ctl3_forget, e->inputs, ds, doff, e->msg, e->gx, T);
                else k_gate3<<<dim3(D / 8, T * 3), 256, 0, g_stream>>>(e->hbuf, L.ctl3_gain, L.ctl3_forget, e->inputs, ds, doff, e->msg, e->gx);
                k_rms3<<<dim3(T, 3), NT, 0, g_stream>>>(e->gx, L.in_norm, L.global ? D : 0, e->xin);
                xq_ = e->xin; xk_ = e->xin + D; xv_ = e->xin + 2 * D; xs = 3 * D;
            } else {
                if (!have_inputs) { k_rms<<<T, 256, 0, g_stream>>>(partial, L.in_norm, e->inputs); Q = 1; have_inputs = true; }
                xq_ = e->inputs; xk_ = e->inputs + (Q == 3 ? D : 0); xv_ = e->inputs + (Q == 3 ? 2 * D : 0); xs = Q * D;
            }
            prof(looped ? "blend" : "mix");
            // ---- attention (3)
            if (e->fp8_attn) {                           // lossy option: 8-bit attention projections (three inputs -> three GEMMs)
                const float* xin3[3] = {xq_, xk_, xv_}; const Fp8Mat* m3[3] = {&L.f_q, &L.f_k, &L.f_v}; const int r0[3] = {0, R_K, R_V};
                for (int s3 = 0; s3 < 3; s3++) {
                    if (s3 == 0 || xin3[s3] != xin3[0]) k_quant8f<<<dim3((D) / 128, T), 32, 0, g_stream>>>(xin3[s3], D, e->tq, e->ts, xs);   // identical inputs are quantised once
                    gemm_fp8(*m3[s3], D, T, e->tq, e->ts, e->proj + r0[s3], 1, QKVR);
                }
            } else gemv(QKVR, T, L.qkvr, D, T, xq_, xk_, xv_, xs, R_K, R_V, e->proj, QKVR, 1);
            if (L.global) {
                gemv(32, T, L.kproj, D, T, xk_, xk_, xk_, xs, 32, 32, e->dk, 32, 1);
                gemv(32, T, L.vproj, D, T, xv_, xv_, xv_, xs, 32, 32, e->dv, 32, 1);
            }
            prof("attn proj gemv");
            // Attention, at most HMAX rows at a time. A sliding-window layer keeps its keys and values in a ring with
            // room for the window plus HMAX new positions, and every row's key and value are written before any score
            // is read: with more rows in one go, late rows would overwrite slots that early rows still need. Each
            // half therefore writes and reads before the next one writes - exactly what two smaller steps would do.
            // (Everything outside this block runs once for the whole step, which is the point: the experts are read once.)
            const bool big = T > CR - 3;                    // only then can a run be too long to store its conv history in-kernel
            for (int h0 = 0; h0 < T; h0 += HMAX) {
                int hn = std::min(HMAX, T - h0);
                k_attn_prep<<<dim3(hn, H + HKV), prep_threads(hn), 0, g_stream>>>(e->proj, L.qscale, L.rscale, L.rel, L.global, e->rows, e->qc, e->bias, e->ring[visit][0],
                                                                   e->ring[visit][1], L.kconv, L.vconv, e->dk, e->dv, e->Kc[visit], e->Vc[visit], e->kv8, kvs, h0);
                prof("attn q/r/kv prep");
                int ng = L.global ? gchunks * 2 : RING / 16, ns = ng * 16;      // scores, softmax, value sum on the tensor cores
                if (T < g_attn_group && T <= GSUB) {     // few rows: one block per row keeps more of the GPU busy
                    if (!L.global) {
                        k_attn_scores<<<dim3((ng + 7) / 8, hn * HKV), 256, 0, g_stream>>>(e->qc, e->bias, (const uint8_t*)e->Kc[visit], e->rows, 0, ng, e->att_sc, RING, h0, e->kv8, kvs);
                        k_attn_soft<<<dim3(hn, H), 1024, 0, g_stream>>>(e->att_sc, RING, ns, e->att_l, h0, e->rows, 0);
                        k_attn_pv<8><<<dim3(8, hn * HKV), 256, 0, g_stream>>>(e->att_sc, RING, (const uint8_t*)e->Vc[visit], ng, e->att_l, e->proj, e->att_y, 0, h0, e->kv8, e->rows, kvs);
                    } else {
                        k_attn_scores<<<dim3((ng + 7) / 8, hn * HKV), 256, 0, g_stream>>>(e->qc, e->bias, (const uint8_t*)e->Kc[visit], e->rows, 1, ng, e->att_sc, gchunks * 32, h0, e->kv8, kvs);
                        k_attn_soft<<<dim3(hn, H), 1024, 0, g_stream>>>(e->att_sc, gchunks * 32, ns, e->att_l, h0, e->rows, 1);
                        k_attn_pv<32><<<dim3(8, hn * HKV), 1024, 0, g_stream>>>(e->att_sc, gchunks * 32, (const uint8_t*)e->Vc[visit], ng, e->att_l, e->proj, e->att_y, 1, h0, e->kv8, e->rows, kvs);
                    }
                } else {
                    auto ngr = [&](int n) { return cls ? n : (n + 7) / 8; };          // most groups among n rows: the grid's height
                    if (!L.global) {
                        uint2* pl = (uint2*)(e->att_sc + (size_t)HMAX * H * RING); const int g0i = h0 / 16, g1i = (h0 + hn + 15) / 16;
                        local_attention(e, visit, kvs, h0, hn, ngr(hn), pl, g0i, g1i);
                    } else for (int t0 = h0; t0 < h0 + hn; t0 += GSUB) {   // full-attention layers: GSUB tokens at a time, so the scratch stays small
                        int tn = std::min(GSUB, h0 + hn - t0); uint16_t* pk = (uint16_t*)(e->att_sc + (size_t)GSUB * H * gchunks * 32);
                        k_attn_scores_g<SKG><<<dim3((ng + 8 * SKG - 1) / (8 * SKG), ngr(tn) * HKV), 256, 0, g_stream>>>(e->qc, e->bias, (const uint8_t*)e->Kc[visit], e->rows, e->grp, e->gb, t0 / 16, t0 / 16 + 1,
                                                                                             1, ng, e->att_sc, gchunks * 32, t0, e->kv8, kvs);
                        prof("full attn: scores");
                        k_attn_soft_g<<<dim3(tn, H), 1024, 0, g_stream>>>(e->att_sc, gchunks * 32, ns, e->att_l, t0, pk, e->rows, 1, cls == 0);
                        prof("full attn: softmax");
                        if (cls == 0) k_attn_pv_gl<32><<<dim3(8, ngr(tn) * HKV), 1024, 0, g_stream>>>((const uint2*)pk, gchunks * 32, (const uint8_t*)e->Vc[visit], ng, e->att_l, e->proj, e->att_y, t0, e->kv8, e->rows, kvs,
                                                                                                    e->grp, e->gb, t0 / 16, t0 / 16 + 1);
                        else k_attn_pv_g<32><<<dim3(8, ngr(tn) * HKV), 1024, 0, g_stream>>>(pk, gchunks * 32, (const uint8_t*)e->Vc[visit], ng, e->att_l, e->proj, e->att_y, 1, t0, e->kv8, e->rows, kvs,
                                                                                           e->grp, e->gb, t0 / 16, t0 / 16 + 1);
                        prof("full attn: values");
                    }
                }
                prof("attn core");
            }
            if (big) {
                k_ring_put<<<T, 256, 0, g_stream>>>(e->proj, QKVR, R_K, e->ring[visit][0], 1024, e->rows);
                k_ring_put<<<T, 256, 0, g_stream>>>(e->proj, QKVR, R_V, e->ring[visit][1], 1024, e->rows);
            }
            prof("attn q/r/kv prep");
            if (e->fp8_attn) {
                k_quant8f<<<dim3((H * DH) / 128, T), 32, 0, g_stream>>>(e->att_y, H * DH, e->tq, e->ts, H * DH);
                gemm_fp8(L.f_wo, H * DH, T, e->tq, e->ts, e->attn_raw, 1, D);
            } else gemv(D, T, L.wo, H * DH, T, e->att_y, e->att_y, e->att_y, H * DH, D, D, e->attn_raw, D, 1);
            prof("attn out gemv");
            // ---- attention-output conv + MoE input mix (4)
            // The next visit's fused q|r|k|v projection (39.6 MB) is touched in four slices by this visit's glue kernels.
            Layer* nl = visit == NVIS - 1 ? nullptr : &e->L[(i == 17) ? (p < 2 ? 6 : 18) : i + 1];
            const void* nq = !nl ? nullptr : (e->fp8_attn ? (const void*)nl->f_q.w : (const void*)nl->qkvr);
            const size_t QKVR_BYTES = e->fp8_attn ? (size_t)410 * (D / 32) * 512 : QKVR_BYTES_BF16;
            PostAttn pa{}; pa.pf = pf_slice(nq, QKVR_BYTES, 0.0, pf_q[0], T); pa.attn_raw = e->attn_raw; pa.ring = e->ring[visit][2]; pa.w = L.attn_sconv; pa.rows = e->rows; pa.attn_out = e->attn_out;
            pa.ns = first ? nb + 1 : nb; for (int n = 0; n < pa.ns; n++) pa.src[n] = e->stack[n];
            pa.partial = first ? nullptr : partial; pa.q = e->mlp_q[i]; pa.norm = looped ? nullptr : L.mlp_norm; pa.out = looped ? e->mixtmp : e->mlp_in;
            launch_post_attn(T, pa);
            if (big) k_ring_put<<<T, 256, 0, g_stream>>>(e->attn_raw, D, 0, e->ring[visit][2], D, e->rows);
            if (looped) blend(e, T, L, 3, e->mixtmp, D, e->msg + 3 * D, L.mlp_norm, e->mlp_in, D);
            prof(looped ? "post-attn+blend" : "post-attn");
            // ---- MoE (5, 6)
            gemv(NE, T, L.router, D, T, e->mlp_in, e->mlp_in, e->mlp_in, D, NE, NE, e->rlogit, NE, 0);
            prof("router gemv");
            const bool wide = T >= g_xwide && !L.fp8;   // big batches: tasks of up to 64 columns per expert (k_expert4_wide)
            const int NTW = max_wtasks(N); static const int XWB = getenv("INK_XWB") ? atoi(getenv("INK_XWB")) : 8;   // warps (16-row tiles) per block of the wide kernel
            if (T == 1) k_route_tasks<<<1, NE, 0, g_stream>>>(e->rlogit, L.rbias, T, e->ids, e->gamma, e->tasks, e->ntasks);
            else {                                       // one block per token routes in parallel, then one block groups by expert
                k_route<<<T, NE, 0, g_stream>>>(e->rlogit, L.rbias, e->ids, e->gamma);
                if (wide) k_tasks_w<<<1, NE, TASKS_SHARED, g_stream>>>(e->ids, T * TOPK, e->tasksw, e->ntasks);
                else k_tasks<<<1, NE, TASKS_SHARED, g_stream>>>(e->ids, T * TOPK, e->tasks, e->ntasks);
            }
            prof("router");
            if (L.fp8) {
                k_quant8c<<<T, 128, 0, g_stream>>>(e->mlp_in, D, e->xcode8, e->xscale, D);
                k_expert8_mma<<<dim3(D / 128, NTK), 256, 0, g_stream>>>(L.w13, L.w13s, e->tasks, e->ntasks, e->xcode8, e->xscale, 0, D, D, e->gu);
                k_swiglu_quant8<<<N, 64, 0, g_stream>>>(e->gu, e->gamma, e->acode8, e->ascale);
                k_expert8_mma<<<dim3(D / 128, NTK), 256, 0, g_stream>>>(L.w2, L.w2s, e->tasks, e->ntasks, e->acode8, e->ascale, 1, D, I2, e->down);
            } else {
                PF pq = pf_slice(nq, QKVR_BYTES, pf_q[0], pf_q[1], T), ps = pf_slice(nq, QKVR_BYTES, pf_q[1], pf_q[2], N);
                if (q4_native) k_quant4f<<<T + pf_blocks(pq, D / 4), D / 4, 0, g_stream>>>(e->mlp_in, D, e->xamax, e->xcode, e->xscale, pq);
                else k_quant4c<<<T + pf_blocks(pq, 256), 256, 0, g_stream>>>(e->mlp_in, D, e->xamax, e->xcode, e->xscale, pq);
                prof("x quant4");
                if (wide) {
                    k_xgather4<<<dim3(WCOLS / 8, NTW), 256, 0, g_stream>>>(e->tasksw, e->ntasks, e->xcode, e->xscale, 0, D, e->xb);
                    k_expert4_wide<<<dim3(D / 16 / XWB, NTW), XWB * 32, 0, g_stream>>>(L.w13, L.w13s, L.w13amax, L.inv_den13, e->tasksw, e->ntasks, e->xb, e->xamax, 0, D, D, e->gu, 4);
                } else k_expert4_mma<<<dim3(D / 32, NTK), 256, 0, g_stream>>>(L.w13, L.w13s, L.w13amax, L.inv_den13, e->tasks, e->ntasks, e->xcode, e->xscale, e->xamax, 0, D, D, e->gu, 4);
                prof("x mma w13");
                if (q4_native) k_swiglu_quant4f<<<N + pf_blocks(ps, I2 / 4), I2 / 4, 0, g_stream>>>(e->gu, e->gamma, e->aamax, e->acode, e->ascale, ps);
                else k_swiglu_quant4<<<N + pf_blocks(ps, 256), 256, 0, g_stream>>>(e->gu, e->gamma, e->aamax, e->acode, e->ascale, ps);
                prof("x swiglu+quant");
                if (wide) {
                    k_xgather4<<<dim3(WCOLS / 8, NTW), 256, 0, g_stream>>>(e->tasksw, e->ntasks, e->acode, e->ascale, 1, I2, e->xb);
                    k_expert4_wide<<<dim3(D / 16 / XWB, NTW), XWB * 32, 0, g_stream>>>(L.w2, L.w2s, L.w2amax, L.inv_den2, e->tasksw, e->ntasks, e->xb, e->aamax, 1, D, I2, e->down, 2);
                } else k_expert4_mma<<<dim3(D / 64, NTK), 256, 0, g_stream>>>(L.w2, L.w2s, L.w2amax, L.inv_den2, e->tasks, e->ntasks, e->acode, e->ascale, e->aamax, 1, D, I2, e->down, 2);
                prof("x mma w2");
            }
            prof("experts fp8");
            // ---- combine + conv + residual update + mix for the next visit
            PostMoe pm{}; pm.pf = pf_slice(nq, QKVR_BYTES, L.fp8 ? 0.0 : pf_q[2], pf_q[3], T); pm.down = e->down; pm.ring = e->ring[visit][3]; pm.w = L.mlp_sconv; pm.rows = e->rows; pm.attn_out = e->attn_out; pm.moe_out = e->moe_out;
            pm.partial = first ? nullptr : partial;
            if (first) nb++;
            pm.stack_out = e->stack[nb]; pm.ns = nb; for (int n = 0; n < nb; n++) pm.src[n] = e->stack[n];
            if (visit == NVIS - 1) { pm.q = e->out_q; pm.Q = 1; pm.norm = e->norm; pm.out = e->out; }
            else {
                int ni = (i == 17) ? (p < 2 ? 6 : 18) : i + 1, np = (i == 17) ? (p < 2 ? p + 1 : 0) : p;
                Q = e->L[ni].global ? 3 : 1;
                pm.q = e->attn_q[ni - 1]; pm.Q = (ni == 6 && np > 0) ? 0 : Q; pm.norm = np > 0 ? nullptr : e->L[ni].in_norm; pm.out = e->inputs;
            }
            launch_post_moe(T, pm);
            if (big) k_ring_put<<<T, 256, 0, g_stream>>>(e->moe_out, D, 0, e->ring[visit][3], D, e->rows);
            trace_copy(e, visit, 0, e->attn_raw, T); trace_copy(e, visit, 1, e->mlp_in, T);
            trace_copy(e, visit, 2, e->moe_out, T); trace_copy(e, visit, 3, e->stack[nb], T);
            prof("post-moe");
        }
    }
    CK(cudaMemcpyAsync(e->feats + (size_t)3 * TMAX * D, e->out, (size_t)T * D * 4, cudaMemcpyDeviceToDevice, g_stream));   // 4th tap
    if (hm == 2) head_two_stage(e, T);                   // prompt chunks whose predictions nobody reads skip the output head
    else if (hm) head_full(e, T);
    if (e->dr.on) {                                      // draft-side context for these positions (7.3)
        auto& d = e->dr;
        k_dctx<<<dim3(T, 4), NT, 0, g_stream>>>(e->feats, d.alpha, d.ctx_norm, d.dctx);
        gemv(6144, T, d.kvctx, D, T, d.dctx, d.dctx + D, d.dctx + 2 * D, 4 * D, 2048, 4096, d.dkv, 6144, 1);
        k_dctx_write<<<dim3(T, 3 * HKV), 128, 0, g_stream>>>(d.dkv, d.kcn, e->rows, d.Kd[0], d.Vd[0], d.Kd[1], d.Vd[1], d.Kd[2], d.Vd[2], d.lstride, d.gstride);
    }
}

// Sampling parameters for the following forward calls: temperature <= 0 means greedy.
void ink_set_sampling(void* ep, float temperature, unsigned seed) { auto* e = (Engine*)ep; e->def_temp = temperature; e->def_seed = seed; }

// One engine step over T rows. Row t is token tokens[t] of conversation seqs[t] at absolute position pos[t]; rows of the
// same conversation must be adjacent and in position order. res_out[T][4] = {greedy argmax, sample, sample excluding
// excl[t], bits of log p(excl[t])} for the token after each row; excl (optional) holds one token per row or -1.
// logits_out (optional) gets [T][V0] fp32. want_head = 0 skips the output head (prompt chunks nobody reads).
// temps / seeds (optional, per row): sampling temperature (<= 0: greedy) and seed; default = ink_set_sampling's values.
// Returns 0, or 1 without doing anything if the rows are not valid: each conversation's rows must be adjacent, at
// consecutive positions (one run per conversation per step), with tokens, slots and positions in range.
// qrows (optional, per row): for a row that verifies a sampled draft, conversation * 7 + draft index; -1 otherwise.
int ink_step(void* ep, int T, const int* tokens, const int* seqs, const int* pos, const int* excl, int* res_out, float* logits_out, int want_head,
             const float* temps, const unsigned* seeds, const int* qrows) {
    auto* e = (Engine*)ep;
    if (T < 1 || T > TMAX || ((want_head != 0 || logits_out != nullptr) && T > HMAX)) return 1;
    std::vector<Row> rows(T); int maxpos = 0; bool seen[MAXSEQ] = {};
    for (int t = 0, r0 = 0; t < T; t++) {
        if (seqs[t] < 0 || seqs[t] >= e->nseq || pos[t] < 0 || pos[t] >= e->ctx_mapped[seqs[t]] || tokens[t] < 0 || tokens[t] >= V0) return 1;
        if (t == 0 || seqs[t] != seqs[t - 1]) { if (seen[seqs[t]]) return 1; seen[seqs[t]] = true; r0 = t; }   // a new run starts here
        else if (pos[t] != pos[t - 1] + 1) return 1;
        rows[t] = Row{pos[t], seqs[t], r0, 0}; maxpos = std::max(maxpos, pos[t]); e->seq_tokens[seqs[t]] = pos[t] + 1;
    }
    for (int t = T - 1, end = T; t >= 0; t--) { rows[t].runlen = end - rows[t].run0; if (rows[t].run0 == t) end = t; }
    int xq[2 * TMAX];                                     // [excluded token per row | sampled-draft index per row], one upload
    for (int t = 0; t < T; t++) {
        xq[t] = excl ? excl[t] : -1; xq[TMAX + t] = qrows ? qrows[t] : -1;
        if (xq[TMAX + t] >= e->nseq * (DB - 1)) return 1;
    }
    CK(cudaMemcpyAsync(e->tok, tokens, T * 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaMemcpyAsync(e->rows, rows.data(), T * sizeof(Row), cudaMemcpyHostToDevice, g_stream));
    // Attention groups: up to 8 consecutive rows of one conversation, cut at every 8th row and at every run start.
    Grp grp[TMAX]; int gb[TMAX / 16 + 2], ngrp = 0, cls = 0;
    int rowg[TMAX], cso[TMAX + 1];
    for (int t = 0; t < T; t++) {
        if (t % 16 == 0) gb[t / 16] = ngrp;
        if (t % 8 == 0 || rows[t].run0 == t) { grp[ngrp++] = Grp{t, 0}; if (t % 8) cls = 1; }
        grp[ngrp - 1].n++; rowg[t] = ngrp - 1;
    }
    for (int k = (T + 15) / 16; k < TMAX / 16 + 2; k++) gb[k] = ngrp;
    cso[0] = 0; for (int k = 0; k < ngrp; k++) cso[k + 1] = cso[k] + (grp[k].n * 3 + 7) / 8;
    if (T >= g_attn_group) {                              // (smaller steps use the one-row attention kernels: no groups, no uploads)
        CK(cudaMemcpyAsync(e->grp, grp, ngrp * sizeof(Grp), cudaMemcpyHostToDevice, g_stream));
        CK(cudaMemcpyAsync(e->gb, gb, sizeof(gb), cudaMemcpyHostToDevice, g_stream));
        CK(cudaMemcpyAsync(e->rowg, rowg, T * 4, cudaMemcpyHostToDevice, g_stream));
        CK(cudaMemcpyAsync(e->cso, cso, (ngrp + 1) * 4, cudaMemcpyHostToDevice, g_stream));
    } else cls = 0;
    CK(cudaMemcpyAsync(e->excl, xq, (TMAX + T) * 4, cudaMemcpyHostToDevice, g_stream));
    float sp[2 * TMAX];
    for (int t = 0; t < T; t++) { float tp = temps ? temps[t] : e->def_temp; unsigned sd = seeds ? seeds[t] : e->def_seed; sp[2 * t] = tp > 0 ? 1.0f / tp : 0.0f; memcpy(&sp[2 * t + 1], &sd, 4); }
    CK(cudaMemcpyAsync(e->sparam, sp, T * 8, cudaMemcpyHostToDevice, g_stream));
    static const bool no_graph = getenv("INK_NOGRAPH") != nullptr;
    // Launch grids are frozen into a captured graph, so the full-attention layers get one graph per
    // power-of-two context bucket: their grid covers `bucket` keys and extra blocks exit immediately.
    int bkt = 0; while ((1024 << bkt) < maxpos + 1) bkt++;
    int gchunks = std::min(1024 << bkt, e->max_ctx) / 32;
    bool head = want_head != 0 || logits_out != nullptr;
    // want_head 2: the caller only needs each row's argmax and sample, so the two-stage head will do.
    int hm = !head ? 0 : (want_head == 2 && !logits_out && e->head2 && T <= FAST_T) ? 2 : 1;
    if (g_prof || e->trace || no_graph) forward_body(e, T, gchunks, hm, cls);
    else run_graphed(&e->graph[T][bkt][hm][cls], [&] { forward_body(e, T, gchunks, hm, cls); });
    CK(cudaStreamSynchronize(g_stream));
    CK(cudaGetLastError());                              // a kernel that failed to launch must stop us, not hand back garbage
    prof("head+copy");
    if (hm == 2) {
        int tmp[FAST_T * 4 + 2]; CK(cudaMemcpy(tmp, e->sres, T * 16 + 8, cudaMemcpyDeviceToHost));
        static const bool always = getenv("INK_HEAD_FALLBACK") != nullptr;   // testing: take the fall-back path every time
        if (tmp[T * 4] || always) {                      // too many candidates: the full head after all (not part of the recorded graph)
            if (e->head2_fallbacks++ == 0 && getenv("INK_HEAD_FALLBACK") == nullptr) fprintf(stderr, "note: a step had more output-head candidates than the two-stage head keeps (%d blocks); such steps use the full head\n", tmp[T * 4 + 1]);
            head_full(e, T); CK(cudaStreamSynchronize(g_stream)); CK(cudaGetLastError());
            CK(cudaMemcpy(res_out, e->sres, T * 16, cudaMemcpyDeviceToHost));
        } else memcpy(res_out, tmp, T * 16);
    } else if (head) CK(cudaMemcpy(res_out, e->sres, T * 16, cudaMemcpyDeviceToHost)); else memset(res_out, 0, T * 16);
    if (logits_out) CK(cudaMemcpy(logits_out, e->logits, (size_t)T * V0 * 4, cudaMemcpyDeviceToHost));
    e->last_T = T;
    return 0;
}

// Developer check of the two-stage head on the rows of the last step (which must have used a head): runs it
// again, then the full head, and compares. out = { rows whose argmax or sample differ, the worst (|exact - approximate|
// - bound) over all tokens (must be negative), the worst |error| / bound, candidate blocks, 1 if the list overflowed }.
int ink_headcheck(void* ep, double* out) {
    auto* e = (Engine*)ep; int T = e->last_T; if (!e->head2 || T < 1 || T > FAST_T) return 1;
    std::vector<int> fast(T * 4 + 2), full(T * 4), step(T * 4); std::vector<float> apx((size_t)T * V0), ex((size_t)T * V0), eu(V0), un(V0), hb(2 * FAST_T);
    CK(cudaMemcpy(step.data(), e->sres, T * 16, cudaMemcpyDeviceToHost));    // what the step itself returned (from its recorded graph)
    head_two_stage(e, T); CK(cudaStreamSynchronize(g_stream)); CK(cudaGetLastError());
    CK(cudaMemcpy(fast.data(), e->sres, T * 16 + 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(apx.data(), e->logits, (size_t)T * V0 * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hb.data(), e->h_hb, 2 * FAST_T * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(eu.data(), e->h_eu, V0 * 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(un.data(), e->h_un, V0 * 4, cudaMemcpyDeviceToHost));
    gemv(V0, T, e->unembed, D, T, e->out, e->out, e->out, D, V0, V0, e->logits, V0, 0); CK(cudaStreamSynchronize(g_stream));
    CK(cudaMemcpy(ex.data(), e->logits, (size_t)T * V0 * 4, cudaMemcpyDeviceToHost));
    head_full(e, T); CK(cudaStreamSynchronize(g_stream)); CK(cudaGetLastError());
    CK(cudaMemcpy(full.data(), e->sres, T * 16, cudaMemcpyDeviceToHost));
    double worst = -1e30, ratio = 0; long diff = 0;
    for (int t = 0; t < T; t++) {
        for (int v = 0; v < V0; v++) {
            double err = fabs((double)ex[(size_t)t * V0 + v] - (double)apx[(size_t)t * V0 + v]), b = (double)eu[v] * hb[t * 2] + (double)un[v] * hb[t * 2 + 1] + 1e-3;
            worst = std::max(worst, err - b); ratio = std::max(ratio, err / b);
        }
        if (!fast[T * 4]) diff += (fast[t * 4] != full[t * 4]) + (fast[t * 4 + 1] != full[t * 4 + 1]);
        diff += (step[t * 4] != full[t * 4]) + (step[t * 4 + 1] != full[t * 4 + 1]);
    }
    out[0] = (double)diff; out[1] = worst; out[2] = ratio; out[3] = fast[T * 4 + 1]; out[4] = fast[T * 4];
    return 0;
}
// Single conversation (0), T consecutive positions from pos0.
void ink_forward(void* ep, const int* tokens, int T, int pos0, const int* excl, int* res_out, float* logits_out, int want_head, const int* qrows) {
    int seqs[TMAX], pos[TMAX]; for (int t = 0; t < T && t < TMAX; t++) { seqs[t] = 0; pos[t] = pos0 + t; }
    if (ink_step(ep, T, tokens, seqs, pos, excl, res_out, logits_out, want_head, nullptr, nullptr, qrows)) { fprintf(stderr, "ink_forward: invalid input (T=%d, pos0=%d)\n", T, pos0); abort(); }
}
int ink_nseq(void* ep) { return ((Engine*)ep)->nseq; }
// Longest run of rows (next token + drafts) of one conversation whose rejected tail can be discarded afterwards.
int ink_max_rewind_run() { return CR - 3; }

// The drafter's kernel sequence: block forward over 8 slots, then 7 steps of the recurrent head.
// One pass of the drafter for NB conversations (their anchors, positions and slots are in d.anchor / d.L / d.dseq).
// Every weight matrix is read once whatever NB is, which is why drafting together is much cheaper than in turn.
static void draft_body(Engine* e, int gchunks, int ndraft, int NB) {
    auto& d = e->dr; int TB = NB * DB;
    // Dense GEMM of the drafter: 8-bit weights + 8-bit activations on the FP8 tensor cores, or the bf16 path.
    auto dgemm = [&](const Fp8Mat& f, const uint16_t* wbf, int R, int C, int T, const float* x, float* out, int round) {
        if (!d.fp8) { gemv(R, T, wbf, C, T, x, x, x, C, R, R, out, R, round); return; }
        k_quant8f<<<dim3((C) / 128, T), 32, 0, g_stream>>>(x, C, d.qc, d.qs, C);
        gemm_fp8(f, C, T, d.qc, d.qs, out, round, R);
    };
    k_dinit<<<TB, NT, 0, g_stream>>>(e->embed, e->embed_norm, d.anchor, d.mask, d.in_norm[0], d.resid, d.h);
    for (int i = 0; i < 3; i++) {
        int glob = i == 1, W3 = DQ + 2 * DKV;
        size_t st = glob ? d.gstride : d.lstride;                                      // bytes of one conversation's draft cache
        dgemm(d.f_qkv[i], d.qkv[i], W3, D, TB, d.h, d.qkvo, 1);
        k_dqk<<<dim3(TB, H + HKV), 128, 0, g_stream>>>(d.qkvo, d.qn[i], d.kn[i], d.L, d.dseq, st, d.qd, d.Kd[i], d.Vd[i], glob);
        // The score scratch holds GSUB rows of full-context scores, so the full-attention layer goes GSUB / DB
        // conversations at a time; the sliding-window layers need only RING scores per row and go all at once.
        int grp = glob ? GSUB / DB : NB, smax = glob ? gchunks * 32 : RING;
        for (int c0 = 0; c0 < NB; c0 += grp) {
            int nb = std::min(grp, NB - c0);
            k_dattn_logits<<<dim3(glob ? gchunks : (EXT + DB + 31) / 32, nb * DB * HKV), 256, 0, g_stream>>>(d.qd + (size_t)c0 * DB * H * DH, d.Kd[i], d.L + c0, d.dseq + c0, st, glob, e->att_sc, smax);
            k_dattn_out<<<dim3(nb * DB, H), 1024, 0, g_stream>>>(e->att_sc, smax, d.Vd[i], d.L + c0, d.dseq + c0, st, glob, d.att + (size_t)c0 * DB * DQ);
        }
        dgemm(d.f_o[i], d.o[i], D, DQ, TB, d.att, d.hp, 1);
        k_dres<<<TB, NT, 0, g_stream>>>(d.resid, d.hp, d.post_norm[i], d.h);
        dgemm(d.f_gu[i], d.gu[i], 2 * DI, D, TB, d.h, d.guo, 1);
        k_dsilu<<<(TB * DI + 255) / 256, 256, 0, g_stream>>>(d.guo, d.act, TB);
        dgemm(d.f_down[i], d.down[i], D, DI, TB, d.act, d.hp, 1);
        k_dres<<<TB, NT, 0, g_stream>>>(d.resid, d.hp, i < 2 ? d.in_norm[i + 1] : d.norm, d.h);
    }
    int TR = ndraft * NB;                                                              // head rows: draft-major (see k_dcin)
    k_dcin<<<(TR * 2 * D + 255) / 256, 256, 0, g_stream>>>(d.h, d.frnn, d.dseq, d.cin, d.hsel, NB, ndraft);
    dgemm(d.f_un, e->unembed, V0, D, TR, d.hsel, d.base, 1);                           // base logits of the mask slots that will be used
    gemv(768, TR, d.rnn_hf, 2 * D, TR, d.cin, d.cin, d.cin, 2 * D, 768, 768, d.C, 768, 0);
    k_mk_reset<<<NB, MR, 0, g_stream>>>(d.state, d.anchor, d.prev);
    for (int i = 0; i < ndraft; i++) {
        k_mk_in<<<NB, 512, 0, g_stream>>>(d.state, d.w1, d.prev, d.mkin);
        gemv(768, NB, d.rnn_sw, 512, NB, d.mkin, d.mkin, d.mkin, 512, 768, 768, d.sw, 768, 0);
        k_mk_gate<<<NB, MR, 0, g_stream>>>(d.sw, d.C + (size_t)i * NB * 768, d.rnn_b, d.state, d.lat);
        dgemm(d.f_M, d.M, V0, MR, NB, d.lat, d.mb, 0);
        k_dhead_part<<<dim3(HEAD_PARTS, NB), 1024, 0, g_stream>>>(d.base + (size_t)i * NB * V0, d.mb, d.bvec, d.pv, d.pi, d.psum, d.px, d.psumd, d.pmx, d.dsamp, d.L, d.dseq, d.qlog, i);
        k_dhead_final<<<NB, 32, 0, g_stream>>>(d.pv, d.pi, d.psum, d.px, d.psumd, d.pmx, d.dsamp, d.dseq, d.qz, d.qit, d.out, d.conf, d.top, d.prev, i);
    }
}

// After a step: row `row` of the last batch is conversation `seq`'s last valid position (rows after it were
// rejected drafts). Keeps what the drafter needs from that position.
void ink_commit(void* ep, int seq, int row) {
    auto* e = (Engine*)ep; auto& d = e->dr;
    if (!d.on || seq < 0 || seq >= e->nseq || row < 0 || row >= e->last_T) return;
    CK(cudaMemcpy(d.frnn + (size_t)seq * D, d.dctx + ((size_t)row * 4 + 3) * D, D * 4, cudaMemcpyDeviceToDevice));
}

// Propose `ndraft` (1..7) tokens for conversation `seq` after `anchor`, which sits at position L.
// Returns the tokens and the drafter's probability for each. Requires ink_commit for that conversation.
int ink_has_draft(void* ep) { return ((Engine*)ep)->dr.on; }
int ink_max_ctx(void* ep) { return ((Engine*)ep)->max_ctx; }
// Drafts for n (<= DBATCH) conversations in one pass. seqs / anchors / Ls: each conversation's slot, anchor token and
// the anchor's position. tokens_out / conf_out: [n][7]. Returns 1 if drafting is not possible (caller decodes plainly).
// temps / seeds (optional, per conversation): a temperature of at least DRAFT_SAMPLE_MIN_TEMP makes that conversation's
// drafts samples from the drafter's distribution (see k_dhead_part); otherwise they are its most likely tokens.
// top_out (optional, [n][7]): the drafter's largest probability at each position (see k_dhead_final).
int ink_draft_batch(void* ep, int n, const int* seqs, const int* anchors, const int* Ls, int ndraft, int* tokens_out, float* conf_out, const float* temps, const unsigned* seeds, float* top_out) {
    auto* e = (Engine*)ep; auto& d = e->dr;
    if (!d.on || n < 1 || n > DBATCH) return 1;
    int Lmax = 0;
    for (int b = 0; b < n; b++) {
        if (seqs[b] < 0 || seqs[b] >= e->nseq || Ls[b] < 1 || Ls[b] + DB > e->ctx_mapped[seqs[b]] || anchors[b] < 0 || anchors[b] >= V0) return 1;
        for (int c = 0; c < b; c++) if (seqs[c] == seqs[b]) return 1;                  // one draft per conversation
        Lmax = std::max(Lmax, Ls[b]);
    }
    CK(cudaMemcpyAsync(d.anchor, anchors, n * 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaMemcpyAsync(d.L, Ls, n * 4, cudaMemcpyHostToDevice, g_stream));
    CK(cudaMemcpyAsync(d.dseq, seqs, n * 4, cudaMemcpyHostToDevice, g_stream));
    float ds[2 * DBATCH];
    for (int b = 0; b < n; b++) { ds[2 * b] = temps && temps[b] >= DRAFT_SAMPLE_MIN_TEMP ? 1.0f / temps[b] : 0.0f; unsigned sd = seeds ? seeds[b] : 0; memcpy(&ds[2 * b + 1], &sd, 4); }
    CK(cudaMemcpyAsync(d.dsamp, ds, n * 8, cudaMemcpyHostToDevice, g_stream));
    int bkt = 0; while ((1024 << bkt) < Lmax + DB) bkt++;
    int gchunks = std::min(1024 << bkt, e->max_ctx) / 32;
    static const bool no_graph = getenv("INK_NOGRAPH") != nullptr;
    if (ndraft < 1 || ndraft > DB - 1) ndraft = DB - 1;
    if (no_graph) draft_body(e, gchunks, ndraft, n);
    else run_graphed(&d.graph[bkt][ndraft][n], [&] { draft_body(e, gchunks, ndraft, n); });
    CK(cudaStreamSynchronize(g_stream));
    CK(cudaGetLastError());
    int tk[DBATCH * (DB - 1)]; float cf[DBATCH * (DB - 1)];
    CK(cudaMemcpy(tk, d.out, n * (DB - 1) * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(cf, d.conf, n * (DB - 1) * 4, cudaMemcpyDeviceToHost));
    memcpy(tokens_out, tk, n * (DB - 1) * 4);
    if (conf_out) memcpy(conf_out, cf, n * (DB - 1) * 4);
    if (top_out) CK(cudaMemcpy(top_out, d.top, n * (DB - 1) * 4, cudaMemcpyDeviceToHost));
    return 0;
}
// One conversation; tokens_out / conf_out: [7]. temp / seed as in ink_draft_batch.
int ink_draft_seq(void* ep, int seq, int anchor, int L, int ndraft, int* tokens_out, float* conf_out, float temp, unsigned seed, float* top_out) {
    int tk[DB - 1]; float cf[DB - 1], tp[DB - 1];
    if (ink_draft_batch(ep, 1, &seq, &anchor, &L, ndraft, tk, cf, &temp, &seed, tp)) return 1;
    memcpy(tokens_out, tk, sizeof tk); if (conf_out) memcpy(conf_out, cf, sizeof cf); if (top_out) memcpy(top_out, tp, sizeof tp);
    return 0;
}
// Single-conversation form: fidx = index, within the last step, of conversation 0's last valid position.
int ink_draft(void* ep, int anchor, int L, int fidx, int ndraft, int* tokens_out, float* conf_out, float temp, unsigned seed, float* top_out) {
    if (fidx < 0 || fidx >= TMAX) return 1;
    ink_commit(ep, 0, fidx);
    return ink_draft_seq(ep, 0, anchor, L, ndraft, tokens_out, conf_out, temp, seed, top_out);
}

// ---- Prefix cache support. The full-attention key/value caches are indexed by position, so a conversation can be
// resumed at an earlier position by simply overwriting from there. The ring buffers cannot: the sliding-window
// caches, the convolution history and the drafter's window have been overwritten by whatever came after. A snapshot
// keeps a copy of exactly those rings (about 0.2 GiB per conversation, 0.1 GiB with the 8-bit cache).
static void snap_copy(Engine* e, int seq, bool save) {
    auto& d = e->dr; uint8_t* sn = e->snap + (size_t)seq * e->snap_bytes; size_t off = 0;
    auto cp = [&](void* live, size_t bytes) {
        if (save) CK(cudaMemcpyAsync(sn + off, live, bytes, cudaMemcpyDeviceToDevice, g_stream));
        else CK(cudaMemcpyAsync(live, sn + off, bytes, cudaMemcpyDeviceToDevice, g_stream));
        off += bytes;
    };
    for (int v = 0; v < NVIS; v++) {
        const int C[4] = {1024, 1024, D, D};
        for (int k = 0; k < 4; k++) cp(e->ring[v][k] + (size_t)seq * CR * C[k], (size_t)CR * C[k] * 4);
        if (!e->vis_global[v]) { cp((uint8_t*)e->Kc[v] + (size_t)seq * e->kv_local, e->kv_local); cp((uint8_t*)e->Vc[v] + (size_t)seq * e->kv_local, e->kv_local); }
    }
    if (d.on) {
        for (int i = 0; i < 3; i += 2) { cp((uint8_t*)d.Kd[i] + (size_t)seq * d.lstride, d.lstride); cp((uint8_t*)d.Vd[i] + (size_t)seq * d.lstride, d.lstride); }
        cp(d.frnn + (size_t)seq * D, (size_t)D * 4);
    }
    CK(cudaStreamSynchronize(g_stream));
    if (off != e->snap_bytes) { fprintf(stderr, "snapshot size mismatch\n"); abort(); }
}
// Save / restore conversation `seq`'s ring state. Returns 1 if snapshots are not available.
int ink_snapshot(void* ep, int seq) { auto* e = (Engine*)ep; if (!e->snap || seq < 0 || seq >= e->nseq) return 1; snap_copy(e, seq, true); e->snap_ok[seq] = true; e->snap_tokens[seq] = e->seq_tokens[seq]; return 0; }
int ink_restore(void* ep, int seq) { auto* e = (Engine*)ep; if (!e->snap || seq < 0 || seq >= e->nseq || !e->snap_ok[seq]) return 1; snap_copy(e, seq, false); return 0; }

// Copy the DSpark taps of the last forward call to the host as [T][4][D].
void ink_get_feats(void* ep, int T, float* out) {
    auto* e = (Engine*)ep; std::vector<float> tmp((size_t)4 * TMAX * D);
    CK(cudaMemcpy(tmp.data(), e->feats, tmp.size() * 4, cudaMemcpyDeviceToHost));
    for (int t = 0; t < T; t++) for (int k = 0; k < 4; k++) memcpy(out + ((size_t)t * 4 + k) * D, tmp.data() + ((size_t)k * TMAX + t) * D, D * 4);
}

void ink_microbench(void* ep, int T) {
    auto* e = (Engine*)ep; int N = T * TOPK, NTK = max_tasks(N);
    auto vis = [](int i) { return i < 18 ? i : i + 24; };
    if (T < 1 || T > HMAX) { fprintf(stderr, "micro-benchmark: batch sizes 1..%d only\n", HMAX); return; }   // it runs the head and unsplit attention
    fprintf(stderr, "micro-benchmark, T=%d (time per launch)\n", T);
    mbench("empty kernel", 0, [&](int) { k_resid<<<1, 32, 0, g_stream>>>(nullptr, e->y, e->y, e->mixtmp, 1); });
    if (!e->fp8_attn) mbench("qkvr gemm (8600x2304 bf16)", 39.6, [&](int r) { Layer& L = e->L[r % 24]; gemv(QKVR, T, L.qkvr, D, T, e->inputs, e->inputs, e->inputs, D, R_K, R_V, e->proj, QKVR, 1); });
    if (e->fp8_attn) {
        mbench("  fp8: quantise input (2304)", 0, [&](int) { k_quant8f<<<dim3((D) / 128, T), 32, 0, g_stream>>>(e->inputs, D, e->tq, e->ts, D); });
        mbench("  fp8: q|r|gate|sink gemm (6552 rows)", 15.1, [&](int r) { Layer& L = e->L[r % 24]; gemm_fp8(L.f_q, D, T, e->tq, e->ts, e->proj, 1, QKVR); });
        mbench("  fp8: k gemm (1024 rows)", 2.4, [&](int r) { Layer& L = e->L[r % 24]; gemm_fp8(L.f_k, D, T, e->tq, e->ts, e->proj + R_K, 1, QKVR); });
        mbench("  fp8: wo gemm (2304x3072)", 7.1, [&](int r) { Layer& L = e->L[r % 24]; gemm_fp8(L.f_wo, H * DH, T, e->tq, e->ts, e->attn_raw, 1, D); });
    }
    if (e->fp8_head) mbench("  fp8: output head gemm", 461.0, [&](int) { gemm_fp8(e->f_head, D, T, e->tq, e->ts, e->logits, 0, V0); });
    mbench("attn prep (q/r/bias + kv conv)", 0, [&](int r) { int i = r % 5; Layer& L = e->L[i];
        k_attn_prep<<<dim3(T, H + HKV), prep_threads(T), 0, g_stream>>>(e->proj, L.qscale, L.rscale, L.rel, 0, e->rows, e->qc, e->bias, e->ring[i][0], e->ring[i][1], L.kconv, L.vconv, e->dk, e->dv, e->Kc[i], e->Vc[i], e->kv8, e->kv_local, 0); });
    {   // the rows as one run: groups of 8
        Grp grp[TMAX]; int gb[TMAX / 16 + 2], rowg[TMAX], cso[TMAX + 1], n = 0;
        for (int t = 0; t < T; t++) { if (t % 16 == 0) gb[t / 16] = n; if (t % 8 == 0) grp[n++] = Grp{t, 0}; grp[n - 1].n++; rowg[t] = n - 1; }
        for (int k = (T + 15) / 16; k < TMAX / 16 + 2; k++) gb[k] = n;
        cso[0] = 0; for (int k = 0; k < n; k++) cso[k + 1] = cso[k] + (grp[k].n * 3 + 7) / 8;
        CK(cudaMemcpy(e->grp, grp, n * sizeof(Grp), cudaMemcpyHostToDevice)); CK(cudaMemcpy(e->gb, gb, sizeof(gb), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(e->rowg, rowg, T * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(e->cso, cso, (n + 1) * 4, cudaMemcpyHostToDevice));
    }
    uint2* mpl = (uint2*)(e->att_sc + (size_t)HMAX * H * RING);
    mbench("attention, sliding window (scores + softmax + values)", 0, [&](int r) { int i = r % 5; local_attention(e, i, e->kv_local, 0, T, (T + 7) / 8, mpl, 0, (T + 15) / 16); });
    if (T >= 16) {                                       // full attention for one sub-batch of 16 rows at the current context
        int gv = 0; while (gv < NVIS && !e->vis_global[gv]) gv++;
        int pmax = 0; { std::vector<Row> rr(T); CK(cudaMemcpy(rr.data(), e->rows, T * sizeof(Row), cudaMemcpyDeviceToHost)); for (int t = 0; t < 16; t++) pmax = std::max(pmax, rr[t].pos); }
        int bkt = 0; while ((1024 << bkt) < pmax + 1) bkt++;
        int gch = std::min(1024 << bkt, e->max_ctx) / 32, ng = gch * 2; uint16_t* pk = (uint16_t*)(e->att_sc + (size_t)GSUB * H * gch * 32);
        fprintf(stderr, "  (full attention: 16 rows at position %d)\n", pmax);
        mbench("full attention: scores", 0, [&](int) { k_attn_scores_g<SKG><<<dim3((ng + 8 * SKG - 1) / (8 * SKG), 2 * HKV), 256, 0, g_stream>>>(e->qc, e->bias, (const uint8_t*)e->Kc[gv], e->rows, e->grp, e->gb, 0, 1, 1, ng, e->att_sc, gch * 32, 0, e->kv8, e->kv_global); });
        mbench("full attention: softmax", 0, [&](int) { k_attn_soft_g<<<dim3(16, H), 1024, 0, g_stream>>>(e->att_sc, gch * 32, ng * 16, e->att_l, 0, pk, e->rows, 1, 1); });
        mbench("full attention: values", 0, [&](int) { k_attn_pv_gl<32><<<dim3(8, 2 * HKV), 1024, 0, g_stream>>>((const uint2*)pk, gch * 32, (const uint8_t*)e->Vc[gv], ng, e->att_l, e->proj, e->att_y, 0, e->kv8, e->rows, e->kv_global, e->grp, e->gb, 0, 1); });
    }
    if (!e->fp8_attn) mbench("wo gemm (2304x3072 bf16)", 14.2, [&](int r) { Layer& L = e->L[r % 24]; gemv(D, T, L.wo, H * DH, T, e->att_y, e->att_y, e->att_y, H * DH, D, D, e->attn_raw, D, 1); });
    mbench("post-attn (conv + mix)", 0, [&](int r) { int i = r % 24; Layer& L = e->L[i]; PostAttn pa{}; pa.attn_raw = e->attn_raw; pa.ring = e->ring[vis(i)][2]; pa.w = L.attn_sconv; pa.rows = e->rows;
        pa.attn_out = e->attn_out; pa.ns = 2; pa.src[0] = e->stack[0]; pa.src[1] = e->stack[1]; pa.partial = e->stack[2]; pa.q = e->mlp_q[i]; pa.norm = L.mlp_norm; pa.out = e->mlp_in; pa.pf = PF{nullptr, 0, T}; launch_post_attn(T, pa); });
    mbench("router gemm (128x2304 bf16)", 0.59, [&](int r) { Layer& L = e->L[r % 24]; gemv(NE, T, L.router, D, T, e->mlp_in, e->mlp_in, e->mlp_in, D, NE, NE, e->rlogit, NE, 0); });
    // as the step does it: one fused kernel for a single token, otherwise one block per token and one that groups by expert
    if (T == 1) mbench("route top-8 + task list", 0, [&](int r) { Layer& L = e->L[r % 24]; k_route_tasks<<<1, NE, 0, g_stream>>>(e->rlogit, L.rbias, T, e->ids, e->gamma, e->tasks, e->ntasks); });
    else {
        mbench("route top-8", 0, [&](int r) { Layer& L = e->L[r % 24]; k_route<<<T, NE, 0, g_stream>>>(e->rlogit, L.rbias, e->ids, e->gamma); });
        mbench("task list", 0, [&](int) { k_tasks<<<1, NE, TASKS_SHARED, g_stream>>>(e->ids, T * TOPK, e->tasks, e->ntasks); });
    }
    mbench("quantise x to NVFP4", 0, [&](int) { k_quant4c<<<T, 256, 0, g_stream>>>(e->mlp_in, D, e->xamax, e->xcode, e->xscale, PF{nullptr, 0, T}); });
    mbench("quantise x to NVFP4 (native cvt)", 0, [&](int) { k_quant4f<<<T, D / 4, 0, g_stream>>>(e->mlp_in, D, e->xamax, e->xcode, e->xscale, PF{nullptr, 0, T}); });
    mbench("swiglu + quantise (native cvt)", 0, [&](int) { k_swiglu_quant4f<<<N, I2 / 4, 0, g_stream>>>(e->gu, e->gamma, e->aamax, e->acode, e->ascale, PF{nullptr, 0, N}); });
    mbench("experts w13 (NVFP4 mma)", T == 1 ? 23.9 : 0, [&](int r) { Layer& L = e->L[r % 20]; k_expert4_mma<<<dim3(D / 32, NTK), 256, 0, g_stream>>>(L.w13, L.w13s, L.w13amax, L.inv_den13, e->tasks, e->ntasks, e->xcode, e->xscale, e->xamax, 0, D, D, e->gu, 4); });
    mbench("swiglu + quantise", 0, [&](int) { k_swiglu_quant4<<<N, 256, 0, g_stream>>>(e->gu, e->gamma, e->aamax, e->acode, e->ascale, PF{nullptr, 0, N}); });
    mbench("experts w2 (NVFP4 mma)", T == 1 ? 11.9 : 0, [&](int r) { Layer& L = e->L[r % 20]; k_expert4_mma<<<dim3(D / 64, NTK), 256, 0, g_stream>>>(L.w2, L.w2s, L.w2amax, L.inv_den2, e->tasks, e->ntasks, e->acode, e->ascale, e->aamax, 1, D, I2, e->down, 2); });
    mbench("post-moe (combine+conv+resid+mix)", 0, [&](int r) { int i = r % 24; Layer& L = e->L[i]; PostMoe pm{}; pm.down = e->down; pm.ring = e->ring[vis(i)][3]; pm.w = L.mlp_sconv; pm.rows = e->rows;
        pm.attn_out = e->attn_out; pm.moe_out = e->moe_out; pm.partial = e->stack[2]; pm.stack_out = e->gx; pm.ns = 2; pm.src[0] = e->stack[0]; pm.src[1] = e->stack[1];
        pm.q = e->attn_q[i % 23]; pm.Q = 1; pm.norm = e->L[i].mlp_norm; pm.out = e->mixtmp; pm.pf = PF{nullptr, 0, T}; launch_post_moe(T, pm); });
    mbench("loop: message mix (4 queries)", 0, [&](int r) { mix(e, T, {e->bank[0], e->bank[1]}, nullptr, e->msg_q[r % 12], 4, nullptr, e->msg); });
    mbench("loop: pair x3", 0, [&](int) { k_pair3<<<dim3(T, 3), NT, 0, g_stream>>>(e->xin, 3 * D, D, e->msg, e->pair); });
    mbench("loop: controller gemm (192x4608)", 1.77, [&](int r) { Layer& L = e->L[6 + r % 12]; gemv(192, T, L.ctl3_down, 2 * D, T, e->pair, e->pair + 2 * D, e->pair + 4 * D, 6 * D, 64, 128, e->hbuf, 192, 1); });
    mbench("loop: gate x3", 3.5, [&](int r) { Layer& L = e->L[6 + r % 12]; if (T >= GATE_BIG) k_gate3_big<<<dim3(D / 8, 3), 256, 0, g_stream>>>(e->hbuf, L.ctl3_gain, L.ctl3_forget, e->xin, 3 * D, D, e->msg, e->gx, T); else k_gate3<<<dim3(D / 8, T * 3), 256, 0, g_stream>>>(e->hbuf, L.ctl3_gain, L.ctl3_forget, e->xin, 3 * D, D, e->msg, e->gx); });
    mbench("loop: norm x3", 0, [&](int r) { Layer& L = e->L[6 + r % 12]; k_rms3<<<dim3(T, 3), NT, 0, g_stream>>>(e->gx, L.in_norm, 0, e->xin); });
    if (e->unembed) mbench("unembed gemm (200058x2304 bf16)", 921.9, [&](int) { gemv(V0, T, e->unembed, D, T, e->out, e->out, e->out, D, V0, V0, e->logits, V0, 0); });
    mbench("softcap + argmax", 0, [&](int) { k_head_part<<<dim3(HEAD_PARTS, T), 1024, 0, g_stream>>>(e->logits, e->head_v, e->head_i); k_head_final<<<T, 32, 0, g_stream>>>(e->head_v, e->head_i, e->arg, e->rowmax); });
}

// Print and reset the per-stage timings collected with INK_PROF=1.
void ink_prof_report(int tokens) {
    if (!g_prof) return;
    double tot = 0; for (auto& kv : g_time) tot += kv.second;
    fprintf(stderr, "profile over %d tokens: %.2f ms/token (synchronised)\n", tokens, tot / tokens * 1e3);
    for (auto& kv : g_time) fprintf(stderr, "  %-22s %7.3f ms/token  %5.1f%%\n", kv.first.c_str(), kv.second / tokens * 1e3, 100 * kv.second / tot);
    g_time.clear();
}

}  // extern "C"
