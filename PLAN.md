# PLAN — Inkling 2 Mini engine on RTX 5090

Timestamped log of experiments and decisions (see the inference-optimisation skill). Numbers are medians unless stated.

## Targets and constraints
- Model: thinkingmachines/Inkling-2-Mini @ ed40e7da (NVFP4 checkpoint, 19.5 GB) + bundled DSpark draft (mtp.safetensors).
- Hardware: RTX 5090 32 GB (driver 580.178), Ryzen 7 9800X3D, 29 GB RAM. Desktop session uses ~1.2 GB VRAM.
- Primary metric: single-stream (c1) decode tok/s. Secondary: c4–c16 aggregate, prefill tok/s, TTFT.
- Quality: lossless versus the reference by default. Lossy trades (e.g. FP8 attention weights) only as labelled options.
- Memory cap: 30 GiB VRAM. Context target: start at 32K, extend later (model max 1M).

## Milestones
- M0 toolchain + hardware probe — done 2026-11-23
- M1 architecture spec + slow Python reference validated against vLLM logprobs
- M2 correctness-first CUDA forward (c1 decode) matching the reference layer by layer
- M3 Rust host: tokenizer, chat template, OpenAI-compatible streaming server
- M4 baselines (vLLM, Strata, LM Studio) with scripts/bench.py from the skill
- M5 hill-climb: CUDA graph, tuned GEMV kernels, DSpark, concurrency

## Log
- 2026-11-23 M0: hwprobe — read-only 1,692 GB/s (4M threads), copy 698 GB/s, FP16 GEMM 404/451/483 TFLOPS at 4096³/8192³/16384³.
- 2026-11-23 M4 (early) baselines, skill bench.py, temp 0.6, reasoning_effort none, median of 6:
  - vLLM no speculation, BF16 KV: c1 162.6 tok/s (161.9–163.1), c2 aggregate 166.0, cold prefill 490 tok/s @3.3K tokens.
  - vLLM DSpark k=7 probabilistic, FP8 KV: c1 186.9 and 196.1 in two runs (samples 150–225, noisy: acceptance depends on text), c2 aggregate 310 and 205, cold prefill 502 tok/s. TTFT short 38–141 ms.
  - First-run TTFT/prefill figures include warm-up; the 15,859 tok/s prefill in run 2 was a prefix-cache hit and is not a prefill number.
- 2026-11-23 golden outputs recorded: tests/oracle/golden.json (4 prompts, greedy, top-20 logprobs, from vLLM no-spec BF16 KV).
- 2026-11-23 M1 done: ref/inkling_ref.py vs vLLM golden: 43/44, 62/64 greedy tokens equal; tail logprobs differ up to ~2 nats. Sensitivity test (ref/sens.py): removing one bf16 rounding in our own reference moves outputs by the same amount → model is numerically chaotic; bit-exactness with vLLM is not a usable criterion. Criterion: per-layer agreement on one token + share of identical greedy tokens.
- 2026-11-23 M2 done: engine v0 (one kernel per op). 227/236 greedy tokens equal to vLLM (mismatches 0/2/2/5). Token-0 trace vs Python ref: visit 0 attention and MoE input exact, MoE output 1.7e-3 rel. Decode 13.7 tok/s c1 (baseline for the hill climb). Load 1.3–3.4 s warm page cache.
- 2026-11-23 M5 hill-climb (c1 decode, greedy, "sky is blue" prompt, 120–150 tokens; interleaved A/B):
  - v0 one kernel per op: 13.7 tok/s. Profile: blend 41 ms, experts 24 ms of 75 ms.
  - warp-per-row GEMV/expert kernels + blend split into short kernels: 65–69 tok/s. ACCEPT.
  - CUDA graph replay per batch size: 67.8 → 78.4 (twice). ACCEPT. GPU-bound at ~8.7 µs × ~1,460 kernels.
  - FP4 tensor-core experts (mma mxf4nvf4, sm_120a) + grouping by expert: 78.7 → 85.5 (twice); T=8 step 45.9 → 29.8 ms. ACCEPT. vs vLLM golden: 9/236 greedy mismatches.
  - Next: kernel fusion (count ↓), bf16/MXFP8 tensor cores for T>1, DSpark, HTTP server.
  - Kernel fusion v2 (post-attn, post-moe, attn-prep, route+tasks, swiglu+quant, 3-stream blend): 85.1 → 91.6 (twice). ACCEPT, but smaller than the −45% kernel count suggested: the fused single-block kernels cost 25–30 µs each (long per-thread load chains) while well-shaped small kernels cost ~3 µs. Shape > count.
  - Attention v2 (warp-per-key logits + 1024-thread softmax/PV, graph per context bucket): at 1,500 tokens of context T=1 18.4 ms (v1) → 10.2 ms; speed now flat from 64 to 4,000 tokens of context. ACCEPT. Checks: batch-16 vs batch-1 argmax identical over 2,400 tokens (covers window wrap); v2 vs v1 agreement 93.5–98.6% (noise band); next-token accuracy on text 75.7% vs 75.4%.
  - MMA load batching (6 k-slices of loads before the MMAs): w13 kernel 41 → 28 µs; T=1 at 1,500 ctx 10.21 → 9.54 ms (105 tok/s), T=8 28.5 → 27.0 ms. ACCEPT. Same golden mismatch counts (arithmetic unchanged).
- 2026-11-23 M3 done: OpenAI-compatible server (`inkling serve [addr] [max_ctx]`), streaming, reasoning/content split, CPU sampling, one request at a time.
- 2026-11-23 skill bench.py, same settings as the vLLM baselines (temp 0.6, effort none, median of 6): ink-v2 c1 94.7 tok/s (93.4–95.3), TTFT short 87 ms, cold prefill 356 tok/s @3.3K, c2 aggregate 92.9 (requests are served one after another). vLLM: 162.6 c1 no-spec, ~190 with DSpark, prefill 490.

## STATUS 2026-11-23
| metric | vLLM | this engine | ceiling (est.) |
|---|---|---|---|
| c1 decode, no speculation | 162.6 | 94.7 served (105 raw greedy step rate) | ~310 |
| c1 decode, DSpark | ~190 (150–268 per sample) | not implemented | ~500 (rough) |
| cold prefill @3.3K | ~490 | 356 | not computed |
| 8-token step (DSpark verify shape) | n/a | 27.0 ms | — |

What limits it now (T=1, ~9.5 ms/token, sync-profiled shares): expert stage ~3.6 ms (MMA kernels at ~1/2 of memory speed + quantisers), attention projections 1.7 ms (near memory speed), post-attn/post-moe/blend ~4 ms (badly shaped fused kernels), attention core ~1 ms, router ~0.9 ms, unembed 0.56 ms. CPU sampling costs ~10% in the server.

Ranked next levers:
1. DSpark drafter + verify (reuses the T=8 path; largest expected gain).
2. Re-shape post-attn / post-moe / quantiser kernels to short per-thread load chains (est. −1.5 to −2.5 ms).
3. GPU-side sampling; pinned buffers for the per-step token/position/argmax copies.
4. MXFP8 expert layers and bf16 GEMMs on tensor cores for T>1 (prefill, verify).
5. Split K across warps in the FP4 MMA kernel to reach memory speed.
6. Long-context global attention (>8K): multi-block softmax merge. Prefill batch > 16. Prefix cache, concurrent requests, tool-call parsing in the server.
Not verified: outputs with images (vision tower not loaded), contexts beyond 4K tokens, tool calls, behaviour at tau>1 positions (>8,191).
- 2026-11-23 (cont.) DSpark groundwork + batched-step work. Raw step times at 1,500 tokens of context unless noted.
  - Python drafter (ref/dspark_ref.py) on engine taps: 3.03 tokens/step (code prompt), 2.31 (prose). vLLM's own counters on the same prompts: 2.99 and 2.07. Drafter validated; acceptance on Mini is low (per-position accept 60/34/23/14/7/4/3 % prose, 68/45/34/23/16/9/8 % code).
  - MXFP8 experts on tensor cores (probe2 verified): fp8 stage at T=8 4.54 → 1.45 ms. ACCEPT.
  - bf16 tensor-core GEMM for T>1, row-major weights: attn proj at T=8 3.67 → 3.54 ms. No gain by itself (layout-bound).
  - Tile layout for all matrices + scales (swizzle at load): T=8 23.7 → 17.6 ms; T=1 9.2 → 10.9 (regression, fixed below). ACCEPT.
  - 16-byte loads, 16-warp K-split (dense), 4/2-warp K-split (FP4 experts): T=8 17.6 → 16.1; T=1 10.9 → 10.8. ACCEPT. (Bug: w2 has 18 slices, not divisible by 4 → 27/236 golden mismatches; fixed with per-matrix split.)
  - Wide glue kernels (768 threads × 3 dims): T=1 10.8 → 9.9, T=8 16.1 → 17.1 (noise-level change at T=8). ACCEPT.
  - Floors measured: empty graph kernel 0.67 µs; cold 39.7 MB read 27 µs (1,460 GB/s); cold 0.6 MB 2.2 µs. INK_PROF adds ~6 µs per stage — use `inkling microbench` for per-kernel numbers.
  - Compile-time source/query counts in mix kernels and NCG in the dense GEMM (arrays in registers), multi-block argmax: T=1 9.94 → 6.56 ms (152 tok/s raw), T=8 17.1 → 12.8 ms. Bit-identical outputs. ACCEPT.
  - compute-sanitizer memcheck (pip wheel nvidia-cuda-sanitizer-api): 0 errors over load + T=15 prefill + decode.
- 2026-11-23 (cont.) DSpark in the engine.
  - CUDA drafter (3 dense layers on the tile GEMM, own KV caches, folded Markov head [V0][256]) + greedy verify. Output byte-identical with/without speculation (short prompt and 2,847-token prompt). ACCEPT.
  - Sweep of verified drafts k (greedy, tok/s code / prose prompt): k=0 137/137, k=1 181/174, k=2 209/196, k=3 222/200, k=4 222/205, k=5 217/194, k=7 205/181. Confidence cutoff tau=0.5 with k<=7: 221/202. Default: k=4, tau=0.
  - GPU sampling (Gumbel-max) + rejection rule for speculative sampling (greedy drafts: accept with p(d), else draw from p without d). samptest (3,000 draws, T=1.0): plain vs speculative 64.8/65.0 %, 33.1/33.5 %, 1.6/1.3 %; TV 0.006 (same-distribution noise ~0.025). ACCEPT.
  - Server uses step() (speculation + GPU sampling); streaming decode made incremental.
  - skill bench.py, temp 0.6, effort none, median of 6, interleaved twice:
    - no speculation: c1 134.0, 133.9 (samples 132.7–134.8); TTFT short 42 ms; cold prefill 1,022 / 1,029 tok/s @3.3K.
    - DSpark k=4: c1 189.3, 188.2 (samples 145–260); c2 aggregate 214 / 200 (requests served one after another).
    - vLLM for reference: 162.6 no-spec; 186.9 / 196.1 DSpark (samples 150–268); prefill 490–502.
  - Step-time distribution, T=1: min/base 6.5 ms, median 6.6–6.8, p90 8.9, max ~10; mean 7.3. About half the steps are delayed 1.2–3 ms: gnome-shell (44% GPU "sm" in nvidia-smi pmon), LM Studio, Firefox and Xwayland share the card. Served no-spec speed is mean-bound (134–138) while the engine's own step is ~154 tok/s. Not fixable without root (compute timeslice) or moving the display off the 5090.
- 2026-11-23 code review (10 findings) fixed: sampler probability underflow at low temperature (accepted every draft), tojson kwargs for tools, draft bounds, server read/write timeouts, body size cap, SSE error path, wasted last step, seed overflow, --fmad=false, dumpfeats count. Added tool-call output parsing (OpenAI tool_calls) and argument canonicalisation; tool round trip verified.
- 2026-11-23 re-baseline, same session, skill bench.py, temp 0.6 (desktop quiet: gnome-shell 2% GPU):
  - this engine: DSpark k=4 c1 202.2 / 208.7 (samples 151–270); no-spec 147.9 / 148.1 (147.5–148.6); prefill 1,107–1,121; TTFT 35–39 ms; c2 served sequentially (234/248 aggregate with DSpark).
  - vLLM: DSpark 195.8 (155–270); no-spec 163.5 (162.8–164.2); c2 aggregate 227 (no-spec, true batching).
- L2 prefetch from extra blocks of glue kernels (5090 L2 = 96 MB): works in a probe (33 µs cold read → 13–16 µs after a ≥17 µs compute kernel), but in the engine every plan tried was within noise (6.65–6.81 ms vs 6.78 off). Glue kernels are too short individually. Left at a small default (4 × 10% of the next projection); effectively REJECTED as a lever. A separate parallel graph branch does not run concurrently on this driver (62 µs = sum, not max).
- 2026-11-23 Blackwell-specific pass (user request: use what vLLM cannot assume).
  - Device facts (tools/hwprobe/devprops.cu): 170 SMs, 96 MB L2, 1,536 threads/SM; native conversions cvt.rn.satfinite.e2m1x2.f32 / e4m3x2.f32 work (first operand lands in the HIGH nibble/byte).
  - 8-bit drafter (MXFP8 weights + activations on the FP8 tensor cores; lossless for the output): step 10.44 → 10.0 ms; acceptance 1.82 → 1.80 and 1.28 → 1.27 per step; +3% tok/s. Output identical to plain greedy; samptest TV 0.004. ACCEPT (default; INK_DRAFT_BF16=1 reverts).
  - Native-instruction MXFP8 activation quantiser: 7.6 → 0.95 µs per call. ACCEPT.
  - Native-instruction NVFP4 quantisers (x: 5.2 → 1.55 µs; SwiGLU+quantise: 6.0 → 1.75 µs): T=1 6.66 → 6.38 ms, bit-identical results (100% of 2,400 greedy tokens) once the error term is multiplied in the reference order (value·scale·outer); with the other order 4.25% of tokens differed. ACCEPT (INK_Q4_ALU=1 reverts).
  - LOSSY option INK_FP8=1 (attention matrices + output head in MXFP8, activations quantised to 8 bits): T=1 6.37 → 5.67 ms, T=4 8.90 → 8.12, T=8 11.59 → 10.70; +1.2 GiB VRAM. Quality, teacher-forced NLL (nats/token) / top-1: ARCHITECTURE.md 3,968 tokens 0.8948 / 79.71% → 0.8985 / 79.81% (+0.41%); IMPLEMENTATION_LOG.md 7,968 tokens 1.2133 / 68.69% → 1.2140 / 68.66% (+0.06%). Off by default.
  - Bug found by re-running the speculation identity test: the fast 8-bit quantiser was launched with 2,304 threads per block for the drafter's 9,216-wide rows (limit 1,024) → graph capture error. Fixed (32-thread blocks). scripts/verify.sh added: golden, batch invariance, speculation identity in both modes (k=4, 7), NLL, samptest, memcheck ×3 — all pass.
- 2026-11-23 harness (skill bench.py, temp 0.6, median of 6; desktop quiet):
  | config | c1 decode | samples | prefill @3.3K | TTFT short |
  |---|---|---|---|---|
  | lossless, DSpark k=4 | 220.3 / 227.2 | 176–294 | 1,136 | 34 ms |
  | lossless, no speculation | 155.1 | 154.7–155.6 | 1,144 | 36 ms |
  | INK_FP8=1, DSpark k=4 | 236.0 / 247.7 | 191–374 | 1,312 | 31 ms |
  | INK_FP8=1, no speculation | 175.7 | 174.9–176.4 | 1,310 | 34 ms |
  | vLLM DSpark (same session, earlier) | 195.8 | 155–270 | ~495 | 40 ms |
  | vLLM no speculation | 163.5 | 162.8–164.2 | ~495 | 40 ms |

## STATUS 2026-11-23 (evening)
Lossless engine: +13–16% over vLLM with DSpark, 95% of vLLM without, 2.3× its prefill. 8-bit dense mode (lossy, +0.06–0.4% NLL): +20–26% with DSpark, +7.5% without.
Base step (T=1, lossless) 6.37 ms: matrix kernels ≈3.1 ms (at memory speed, ~1.4–1.5 TB/s), output head 0.58 ms, glue ≈2.2 ms, host ≈0.1 ms.
Next levers, ranked:
1. Attention on tensor cores / wider attention kernels (logits + softmax·V are 10.7 µs of glue per visit at T=1, 29 µs at T=8).
2. Remaining glue: post-attn, post-moe (5 µs each), attn prep 3.5, router+route 4.4 → target ~2 each.
3. Drafter head in NVFP4 (its two output-head-sized passes are most of the ~0.6 ms draft); prune the draft vocabulary.
4. Concurrency: batch several sequences per step (vLLM gets 227 tok/s aggregate at 2 streams without speculation; this engine serves requests one after another).
5. Prefill batch > 16; long-context (>8K) global attention; prefix cache.
- 2026-11-23 (late) attention on tensor cores + adaptive verification.
  - KV caches re-laid as 16×16 tiles (K: keys × dims, V: dims × keys), zero-initialised; three kernels: scores (bf16 mma, one warp per 16 keys), softmax (P rounded to bf16 into the scratch, indexed by cache slot), value sum + sink + gate (bf16 mma, K-split over 8 or 32 warps). Attention per visit: T=1 10.7 → 7.7 µs, T=8 29.2 → 14.0 µs. Steps: T=1 6.37 → 6.17 ms, T=4 8.90 → 8.64, T=8 11.59 → 10.88. ACCEPT.
    Checks: verify.sh full passes; trace at position 60 vs Python reference: visit-0 attention output within 1.8e-3; agreement with previous attention 96.0% of greedy tokens (noise band).
    Side finding: teacher-forced NLL moved 0.8948 → 0.8984 (+0.4%) from this mathematically equivalent change, so ±0.4% is the noise floor of that metric on 4K tokens; the 8-bit mode's measured +0.06–0.5% is not distinguishable from it.
  - mix_core: squared norms and query dots in one block-wide reduction: post-attn 4.92 → 4.72 µs, post-moe 5.26 → 4.79; bit-identical. T=1 6.12 ms. ACCEPT. Glue is now ~29 µs per visit, of which ~12 µs is the 0.69 µs launch floor × 17 kernels.
  - Verification policy re-swept (greedy, tok/s, 3 prompts code/prose/table): k=0 155.7 mean; fixed k=4 245.5; k≤7 with tau=0.5 256.1; tau=0.6 256.7; k≤5 tau=0.4 254.9. Server default now INK_SPEC=7, INK_TAU=0.5. ACCEPT.
- 2026-11-23 harness (temp 0.6, median of 6, desktop quiet), new defaults:
  | config | c1 decode | samples | prefill @3.3K |
  |---|---|---|---|
  | lossless, DSpark (k≤7, tau 0.5) | 231.1 / 234.4 | 191–297 | 1,212 |
  | lossless, no speculation | 154.9 | 154.3–155.4 | 1,211 |
  | INK_FP8=1, DSpark | 253.7 / 249.2 | 208–374 | 1,405 |
  | INK_FP8=1, no speculation | 175.5 | 174.9–176.0 | 1,404 |
  vLLM (same session, earlier): DSpark 195.8, no speculation 163.5, prefill ~495.
- 2026-11-23 long context: raw T=1 step 6.12 ms @1.5K, 6.25 @4K, 6.43 @8K, 6.45 @16K, 6.81 @30K (T=8: 10.8 → 12.9). Needle gate (bench.py --long): 3/3 at 10,968 prompt tokens and 3/3 at 21,925; prefill 1,168 / 1,101 tok/s; decode 350 / 329 tok/s on the retrieval answer.
- Temperature vs speculation (k≤7, tau 0.5; code / prose prompt): T=0 290 / 240 tok/s (1.81 / 1.18 accepted per step); T=0.6 284–325 / 222–229; T=1.0 274–299 / 211–216 (0.87–0.92 accepted on prose). Greedy drafts lose ≤12% at T=1.
- Step-time series shows a ~30 Hz disturbance (+1.2 ms on every ~6th step) even with gnome-shell at 2%: mean 6.42 vs median 6.12 ms.
- 2026-11-23 (night) prefill batch 16 → 64 → 128.
  - TMAX 64: dense GEMM column groups up to 8 (per-group epilogue keeps shared memory fixed), conv ring 128; k_tasks copy fixed for N > 128. Prefill 1,212 → 1,750 tok/s (8,037-token prompt). ACCEPT.
  - Output head skipped for prompt chunks whose predictions are not read: 1,750 → 1,868. ACCEPT.
  - TMAX 128 (KV ring 1,152, conv ring 256; dense GEMMs run as two 64-token launches, expert kernels share reads across all 128): 1,868 → 2,177 (8-bit mode 2,666). +0.7 GiB VRAM. ACCEPT.
  - Found by testing odd batch sizes: results depended on batch size when a batch straddled the 1,024-token bucket boundary (attention value-sum split its key groups across warps differently per bucket → different order of additions). Fixed with a bucket-independent strided assignment for full-attention layers. Batches of 1 / 16 / 48 / 100 / 128 now give identical greedy tokens over 2,400 tokens; verify.sh checks 1, 48, 100, 128 against 16.
  - verify.sh full passes (memcheck 0 errors ×3; samptest TV 0.004).
- 2026-11-23 harness after prefill work (temp 0.6, median of 6):
  | config | c1 decode | prefill @3.3K | long prompt (21,925 tokens) |
  |---|---|---|---|
  | lossless, DSpark | 218.3 (earlier runs 231 / 234) | 2,415 tok/s | needles 3/3, TTFT 11.4 s (1,921 tok/s), decode 332 |
  | lossless, no speculation | 154.9 | 2,419 | — |
  | INK_FP8=1, DSpark | 262.3 | 3,027 | needles 3/3, TTFT 9.6 s (2,279 tok/s), decode 364 |
  | INK_FP8=1, no speculation | 175.4 | 3,014 | — |
  vLLM: DSpark 195.8, no speculation 163.5, prefill ~495.
- 2026-11-24 VRAM budget (measured, lossless, drafter loaded; card 31.84 GiB, desktop ~1.0): in use 21.4 GiB @2K ctx, 21.7 @8K, 22.8 @32K, 24.3 @64K, 27.3 @128K → fixed ≈20.3 GiB engine, 49.1 KB per token of context capacity (global KV 32.8 KB = 8 visits × 4,096 B; attention scratch 12.3 KB = TMAX 128 × 24 heads × 4 B; drafter global layer 4.1 KB). No drafter: −1.1 GiB. INK_FP8: +1.2 GiB (keeps bf16 copies). Cap today 128K (power-of-two sizes only). User priority: large context per agent matters more than agent count ("8 agents × 20K is useless").
- 2026-11-24 memory work (user priority: context per agent).
  - Full-attention scratch sub-batched (16 tokens per launch): per-token context cost 49.1 → 38.5 KB. Conv history ring 256 → 16 slots (0.33 GiB → 20 MB): first attempt raced (late tokens of a batch overwrote slots early tokens read; caught by batch-invariance check: ~4% tokens changed) → batches ≤13 store in-kernel, larger batches store their last 13 tokens with a separate kernel. Any context size (multiple of 32). 8-bit dense mode frees the 16-bit copies (24.0 → 21.1 GiB at 32K). Drafter built before context-sized caches (lower load peak). Results bit-identical to before; verify.sh full passes.
  - VRAM in use (incl. ~1.0 GiB desktop), 16-bit KV: 21.1 GiB @8K, 22.0 @32K, 25.5 @128K, 29.8 @250K, 30.5 @270K. Cap ≈ 270K tokens.
  - Optional 8-bit KV cache (INK_KV_FP8=1; E4M3 unit scale as in the reference, dequantised to bf16 in the attention kernels): 20.7 KB per token; 21.3 GiB @32K, 25.8 @250K, 31.0 @500K. Cap ≈ 480K (≈530K with INK_FP8=1). Speed: T=1 6.12 → 6.31 ms (+3%), T=8 10.95 → 11.21. Quality: NLL 0.8944 → 0.8894 (text A, 3,968 tok) and 1.2112 → 1.2144 (text B, 7,968 tok); top-1 79.81 → 79.96 % and 68.62 → 68.37 %; golden 13 vs 8 of 236 — all inside the ±0.4% noise band. Off by default.
  - All 8-bit (INK_FP8=1 INK_KV_FP8=1): 20.5 GiB @32K; T=1 5.60 ms, T=4 7.97, T=8 10.26; NLL 0.8951 / 1.2142.

## Concurrency plan (2026-11-24)
Constraints from the user: do not sacrifice single-stream speed; the card is VRAM-limited, and context per agent matters more than agent count.
- Rows of one engine step may belong to different conversations ("slots"). Each row carries {slot, position, run start, run length}; kernels address ring buffers and KV caches by slot. One slot with one run must stay bit-identical and equally fast.
- Per-slot memory: local KV rings 189 MB (95 MB with 8-bit KV), conv history 20 MB, drafter rings 9 MB, plus context × 36.9 KB (16-bit) or 18.5 KB (8-bit KV) for the full-attention layers and the drafter. Shared: weights, scratch.
- Phases: (1) per-row addressing in all kernels, single slot, verify identical + same speed; (2) several slots in one step, verify that interleaved conversations give the same tokens as separate runs; (3) batched drafter; (4) scheduler + multi-connection server, load-adaptive speculation (verify a draft only if its cumulative confidence exceeds the current tokens-per-ms rate × cost of a row); (5) shared page pool for full-attention KV so slots can use uneven context.
- 2026-11-24 concurrency, phases 1–4.
  - Phase 1: every kernel addresses position / conv history / KV cache per row (Row {pos, seq, run0, runlen}); caches and histories are allocated per conversation. One conversation: 0 of 2,400 greedy tokens differ; T=1 6.11 → 6.14 ms (noise), T=8 unchanged. verify.sh passes.
  - Phase 2: ink_step takes rows from several conversations. `inkling interleave`: 3 conversations × 1,300 tokens and 6 × 600, irregular 1–19 rows each per step, fresh slots → identical to separate runs. Added to verify.sh.
  - Phase 3: per-conversation drafter caches + persisted Markov feature (ink_commit), per-row temperature/seed in the sampler.
  - Phase 4: scheduler on the engine thread + thread per connection (HTTP, detokenising, SSE off the engine thread). Decoding rows first; prompt chunks share the rest (≤48 rows per step while anyone is generating). `inkling serve addr ctx slots`.
  - Aggregate tok/s, 8 slots, temp 0.6 (bench.py cN, median of 2): no speculation under load c2 256 / c4 394 / c8 571; k≤2 τ0.6: 347 / 415 / 552; k≤4 τ0.5: 372 / 436 / 535. Policy: up to 4 drafts for 2–4 streams, 2 for 5–6, none for 7+. (vLLM c2: 227 without speculation, 205–310 with.) Drafts are still computed one conversation at a time (0.8 ms each).
  - Single stream through the new server: DSpark 238.5, no speculation 157.9, prefill 2,370 — no regression (was 232.5 / 154.9).
  - Memory per extra slot: 1.3 GiB at 32K measured (25.9 GiB in use for 4 × 32K) = 0.21 GiB + ctx × 36.9 KB; with INK_KV_FP8 about 0.11 GiB + ctx × 20.5 KB. With ≈10.5 GiB for slots: 16-bit KV 2 × 146K, 4 × 70K, 8 × 32K; 8-bit KV 2 × 269K, 4 × 132K, 8 × 63K.
  - Not done: batched drafter across conversations; shared page pool for uneven context; prefix cache.
- 2026-11-24 prefix cache, tool-call opening, batched drafter, review, packaging (binary renamed `kuroshio`).
  - Prefix cache (ring snapshots at prompt end; slot chosen by longest shared prefix): 16K-token conversation follow-ups 8.2 s → ~40 ms TTFT, same answers; 4 agents × growing 6K conversation 23.3 s → 12.5 s; identical-prompt retry 2.18 s → 0.05 s. Costs 0.2 GiB per slot (0.1 with 8-bit KV).
  - Tool calls: model puts 0.82 on skipping `,"` after `{"name":"X"` (kuroshio topk). Well-formed first tool call: vLLM 9/24, ours 7/24 (temp 0) → 24/24 with the fixed opening supplied by the server (temp 0 and 0.6). Also fixed non-unique tool-call ids (garbled the next prompt).
  - Batched drafter (≤4 conversations per pass): 2.91 → 1.01 ms for 4; drafts identical (draftcheck). Aggregate c2/c4/c8: 349/415/569 → 366–382/462–470/622–625. Policy: k≤7 for 1–2 streams, ≤4 for 3–4, ≤3 beyond.
  - Review fixes: unsafe live-resume margin (now a per-slot written high-water mark), slot-dependent sampling noise, ink_step input validation, request validation, bounded header reads, connection cap, cancellation of abandoned requests, tool-call flush at end of stream, JSON key order in tool arguments. 349 lines of dead kernels removed. verify.sh full passes.
  - Final, same harness/session (temp 0.6, median of 3): c1 DSpark 227–248 (5 runs; text-dependent ±7%), no speculation 157, all-8-bit 247; c2 356, c4 473, c8 633; prefill 4K 2,200–2,350 (8-bit 2,740). vLLM (DSpark, fp8 KV): c1 187–200, c2 310, c4 431, c8 431, prefill 580; without speculation c1 163.5, c4 289, c8 425.
  - `--ctx auto` (free VRAM − 0.5 GiB): 1×270K, 2×133K, 4×60K, 8×24K; 8-bit KV 1×480K, 2×245K, 4×119K, 8×54K. Estimate vs actual within 0.2 GiB.
  - Not done: shared page pool for uneven context per slot; sampled drafts at temperature > 0; `stop` / `top_p`.
- 2026-11-24 run-time out-of-memory crash (reported by the user after a 60K-token pi session on `kuroshio serve`, 1 slot, auto context): `cudaGraphInstantiate` failed.
  - Cause: one recorded CUDA graph per step shape (batch size × context bucket × head), never freed. Measured 2.9–4.9 MiB of GPU memory per graph: 582 MiB for T = 1..128 in one bucket. Auto context left ~0.5 GiB.
  - Fix: LRU cache of 48 graphs (plateau ≈ 260 MiB, was unbounded); 320 MiB reserved in the context estimate; if instantiation still fails, evict all and retry, then run the step ungraphed (T=1 7.2 vs 5.9 ms, T=128 62 vs 57 ms). First attempt at the fallback returned garbage: kernels are loaded lazily at first launch, which itself needs GPU memory → CUDA_MODULE_LOADING=EAGER, plus a launch-error check after every step.
  - Test: helper takes all free GPU memory while the server runs; 8 requests with unseen shapes are served with answers identical to normal; verify.sh passes (new bounded-graph check); T=1 step 6.12 ms, drafter 0.74 ms (unchanged).
  - Auto context now ≈ 260K / 128K / 58K / 23K for 1 / 2 / 4 / 8 slots (8-bit KV: 465K / 235K / 115K / 52K).
- 2026-11-24 on-demand context memory, sampled drafts, expected-gain verification.
  - Memory: full-attention K/V, the drafter's full-attention layer and the score scratch are reserved address ranges (CUDA driver VMM: cuMemAddressReserve / cuMemCreate / cuMemMap) with memory attached in steps of 2,048 tokens as a conversation grows (`ink_ensure`), released by `ink_trim`. Kernels unchanged apart from never reading K/V groups past the row's own position (they used to read the whole bucket and mask). Idle server 22.1 GiB on the card (was 30.4); 51K-token conversation 24.1 GiB; back to 22.4 after a short one takes the slot. T=1 6.10 ms, T=8 10.72, 24K context T=1 6.66 (unchanged). verify.sh passes.
  - Full card (helper takes all free memory): prompt that does not fit → HTTP 503; running generation stops at its attached memory with finish=length (2,042 tokens with 2,048 attached); works again when memory returns. `--reserve`, `--idle-release`.
  - Slots now share the pool: 1 slot 240K, 4 slots 205K, 8 slots 165K tokens in total (8-bit KV 415K / 385K / 350K), desktop at 1.4 GiB.
  - Sampled drafts (temperature ≥ 0.3): draft ~ q = softmax(drafter logits / T), accept with min(1, p/q), rejected position resampled from max(0, p − q). samptest 10,000 samples at T=1: TV 0.003 between plain and speculative (same-distribution noise 0.014); accepted drafts/step 0.94 vs 0.46 with top-token drafts (T=0.6: 0.46 vs 0.29).
  - Expected-gain verification (`--tau auto`): draft j is worth P(drafts 1..j accepted) tokens for one row; rows are added best-first across conversations while tokens/ms rises; acceptance calibrated online per confidence bucket, step time from a measured table × online scale.
  - A/B, fixed 8 prompts × 400 tokens × fixed seeds, tok/s (old = tau 0.5 + top-token drafts; repeat of old within 1.5%): T=0 245–249 → 255; T=0.6 231–234 → 248–249; T=1.0 221 → 234–244. Sampled drafts alone (tau 0.5): 250 / 236 / 227; expected-gain alone: 255 / 242 / 226.
  - Harness (random text, ±7%): c1 217–244, c2 369, c4 465, c8 639 (old policy same session: 355 / 462 / 634); no speculation 157.6; prefill 2,330–2,390; all-8-bit c1 261, prefill 2,805.
- 2026-11-24 review of the memory / speculation work, and what it changed.
  - Review finding (real): with sampled drafts, the number of drafts verified was decided from each sampled token's own probability q(x_j), which biases the output (example: p=(.9,.1), q=(.6,.4), tau .5 → P(a)=0.96). samptest had not caught it because it verified every draft. Fix: the drafter also returns its top probability per position (independent of the token drawn); decisions and calibration use that. samptest with tau 0.5, 10,000 samples, T=1: TV 0.002 (noise 0.014).
  - With the unbiased rule, sampled drafts no longer pay. Fixed prompts/seeds, tok/s at T=0 / 0.6 / 1.0: old (tau .5, top-token) 249 / 233 / 221; expected-gain + top-token drafts 254 / 241–242 / 232–236; expected-gain + sampled drafts 254 / 238 / 232. Part of the earlier "sampled" gain (249 at 0.6) was the bias. → REJECTED as default; kept behind `--sampled-drafts` (exact, tested, no faster). Expected-gain verification KEPT: +2% / +4% / +5–7%.
  - Other fixes: prompts reserve all their context memory up front (refused before any work if it cannot fit; nothing left attached); a full card reclaims idle slots (longest idle first) before refusing; ink_ensure rolls back a partial failure; the score scratch shrinks with trims; snapshots invalidated when their prefix loses its memory; residual q uses the temperature the draft was drawn with; --reserve validation.
  - `--slots` limited to 1–4 (design decision; more requests queue).
  - Final for this push (harness, temp 0.6, median of 3): c1 233–253 (four runs, median 242), c2 381, c4 474, prefill 2,361, TTFT 28–32 ms; all-8-bit c1 241–261, prefill 2,796. verify.sh full + fullcard.sh pass.
- 2026-11-24 seeded sampling made independent of speculation: a draft is accepted when it equals the sampler's own (Gumbel-max) draw at that position, probability p(draft) as before. Same seed → same text for INK_SPEC 0 / 3 / 7 (new check in verify.sh); samptest 6,000 samples T=1 with tau 0.5: TV 0.004 (noise 0.016). The expected-gain policy had made the number of verified drafts depend on timing, and each verification consumed a host random number, so a fixed seed was no longer reproducible.
- 2026-11-23 gauntlet (`gauntlet/`, black-box suite over HTTP: API, determinism, long context, robustness, speed, memory, quality, a pi agent) and the hardening it prompted. First full run on the pushed build: ~280 checks passed, 2 failed, 19 warnings; nothing crashed, hung or corrupted output; README speed claims within −4% to +6%.
  - Failures: (1) a client that disconnects from a NON-streaming request was never noticed while tokens kept arriving (the check only ran after 500 ms of silence): 4 abandoned requests ran to 3,400 tokens and the next request waited 24 s → checked on a clock every 250 ms, wait now 0.03 s. (2) the model called a tool that was not declared → model behaviour, passed on as other servers do; reclassified as a warning.
  - Control tokens spelt out in client text (`<|end_message|><|message_system|>…` in a user message or tool result) were tokenised as real control tokens, so content could forge a system message → every client string is cut inside such spellings (and inside a beginning of one at its end, because the template joins strings: the independent review assembled a token from four tool-call names) and the pieces are tokenised separately. Ordinary prompts tokenise exactly as before. The forced tool-call opening re-encodes the model's tool name the same way.
  - Also: `stop` sequences (answer text only; ≤16, ≤256 bytes each); SSE header sent with the first event so an early failure is a real 503/400; first delta carries `role`; one socket write per batch of events (was ~49 reads per event at the client); `Expect: 100-continue`; query strings; a 413 the client can read; `--cors ORIGIN` (off by default); `messages` must be a non-empty array, `tools` a list of ≤512 objects, ≤8192 messages; `seed` must be a non-negative integer or −1; `logprobs`, `response_format` ≠ text, `n` > 1 refused; `top_p` accepted and ignored; cancelled requests are logged with `finish=cancelled`; `fullcard.sh` exit code.
  - Known and documented, not changed: a half-closed connection is treated as a disconnect; the prefix cache needs the new prompt to extend the old one (same document + different question re-reads everything).
  - No speed change (same session, harness): c1 238–247, c4 471–482, c2 372–387 vs pushed build 364–373, prefill 2,320–2,360. verify.sh, fullcard.sh, gauntlet quick: no failures.
- 2026-11-23 lever 1: drafts copied from the conversation (`--copy N`, default 31). When the last tokens match an earlier passage (trigram index over prompt + output, match length measured backwards), the tokens that followed it are verified as drafts, without a drafter pass; a repeating pattern continues into its own drafts. → KEPT.
  - Exactness bug found by the A/B's identity check (2 of 8 greedy outputs changed): a run of rows longer than CR − 3 = 13 keeps only its last 13 conv-history positions, so a rejected draft early in a 25-row run restarted from stale history. Conv ring 16 → 40 positions (+31 MiB per slot, twice with the snapshot); runs that may be rejected are capped at CR − 3 (`ink_max_rewind_run`); live-resume margin generalised to CR − 6. Greedy and seeded outputs now identical with `--copy 0`, 1 and 4 streams (also in the gauntlet: default vs `--spec 0`, 4 slots alone/busy).
  - Calibration: acceptance probability by match length (4–5 / 6–9 / 10–19 / 20+, priors .55/.7/.85/.92), kept PER REQUEST. A shared estimate let two verbatim copiers inflate it for two streams that only quote a phrase (5.8 rows verified per step for 3.6 accepted, vs 4.6 with the drafter). Copying is used when its expected yield Σ p^j beats the drafter's recent accepted drafts per step in that request; an unused match slowly forgets its record, so copying is retried.
  - Step cost was wrong: the table assumed 20.8 / 31.8 / 57 ms for 32 / 64 / 128 rows; with the head on every row it is 23.0 / 40.0 / 72.3 ms, and step time jumps at the 9th, 17th, 33rd and 65th row (tiles of rows in the matrix kernels: 14.97 → 16.27 ms at 16 → 17, 23.03 → 26.07 at 32 → 33). Measured table for 1..128 rows in `rows_ms` (`INK_STEP_T=… kuroshio steptime`); `choose` takes the best prefix of rows instead of stopping at the first that does not pay (it stopped at 16 rows with 32 clearly better); big steps are rounded down to multiples of 4 (8 above 64 rows), which bounds the number of recorded graphs (graph recordings 4 streams: ~105 → ~150 per minute, < 0.5% of time) and lands on full tiles. Copy default 31 = 32 rows with the next token.
  - Tried and rejected: a proportionally fair row rule (gain weighed against each conversation's own expected tokens): 4-stream wall throughput −5% to −11% even with copying off. Kept: total tokens per ms. Consequence: a conversation copying a long passage lengthens the step for the others while it lasts.
  - Review fixes: forced tool-call text has first claim on a step's rows (copied rows of earlier slots could crowd out a 45+-token forced stretch); `--tau X` mode verifies copied tokens while p^j ≥ tau; no index is built when speculation is off.
  - A/B, fixed prompts and seeds, 900 tokens, tok/s per stream, `--copy 0` → 31. 1 stream: rewrite code T=0 420 → 483, T=0.6 496 → 612; quote/repeat a document 484 → 591, 495 → 625; repeating patterns 379 → 376, 418 → 417 (6–7% copied); prose 223 → 222, 225 → 226. 4 streams, whole batch, tok/s wall (two runs each): rewrite 558 → 604, 1000 → 1130; quote 764 → 782, 975 → 1035; patterns 688 → 679, 788 → 769 (noise ±2%); prose 435 → 431, 482 → 484.
  - Harness (random prose, 1.2% of tokens copied): c1 228 / 246 / 263, c2 376–384, c4 475–494, prefill 2,394–2,430. verify.sh, fullcard.sh, gauntlet determinism pass.
  - Seen on the way, not yet used: a decode step that also carries prompt rows computes the 0.9 GB output head for those rows too (about 0.12 ms per row).
- 2026-11-23 lever 2: time between the big matrix reads. All changes bit-identical (verify.sh reference numbers and NLL unchanged to the last digit; outputs of unchanged prompts hash the same as the previous build). → KEPT.
  - Where the step-time jumps came from: the dense bf16 GEMM handles tokens in column groups of 8 and existed only for 1 / 2 / 4 / 8 groups, so 17 rows paid for 32 and 33 for 64. Instantiated for every group count 1..8: 17–24 rows −3.5–4%, 33–48 rows −7–10% (16.27 → 15.70 ms at 17 rows, 26.07 → 23.57 at 33).
  - That GEMM is compute-bound, not memory-bound, once it has several column groups (39.6 MB matrix: 26 µs at 1–8 rows, 51 µs at 32, 115 at 64): every thread loaded each activation pair as two floats and converted them to bf16 at every use. New path for batches of 9+: `k_pack_bf16` packs each distinct input once into the tensor-core operand layout, `k_gemm_bf16p` loads 8 bytes per slice. Attention projection 32.4 → 28.3 µs at 16 rows, 51.3 → 38.3 at 32, 114.8 → 80.5 at 64; output head 972 → 776 µs at 32 rows. Break-even measured at 9 rows (below: no gain). 133 registers × 512 threads did not fit a block for 8 groups → 2 slices per loop iteration for 7–8 groups.
  - Loop controller gate (k_gate / k_gate3): one warp per (token, channel) re-read 4.7 MB of tables per token (93 µs at 128 rows, 24× per step). Three rewrites measured: warp per channel looping over tokens 84 µs (the loop is serial behind lane 0's global loads); the same with the controller vector in shared memory 79 µs; one thread per (token, channel) pair summing in the warp reduction's order 34 µs at 128 rows but 5.7 µs (vs 2.0) at 1 row → used from 48 rows up, the old form below.
  - Task list (k_tasks): per-expert token bitmasks (atomicOr in shared memory, popcount, shuffle prefix sum) instead of every expert's thread scanning every pick: 19.8 → 5.1 µs at 128 rows, 3.6 → 1.5 at 16.
  - A mistake on the way: the micro-benchmark's routing line timed the single-token fused kernel at every batch size (37.8 µs at 32 rows, "6–8% of the step"); a real multi-row step never runs it, and the true cost was 3.8 µs. Fixed the benchmark; the rewrite above is worth ~1%, not 7%.
  - Step-time table re-measured (rows: ms, head included): 1: 6.10, 8: 10.60, 9: 11.55, 16: 14.63, 17: 15.29, 24: 17.81, 32: 20.84, 33: 21.83, 48: 27.62, 64: 35.68, 128: 62.65 (was 6.12 / 10.69 / 11.57 / 14.97 / 16.27 / 19.30 / 23.03 / 26.07 / 31.96 / 39.98 / 72.32).
  - Results (harness, temp 0.6): prefill 4K 2,394–2,430 → 2,897–2,954 tok/s (+21%); c4 475–494 → 485–501; c2 376–384 → 369–389; c1 unchanged (228–263 → 225–244; steps of ≤ 8 rows are untouched). Gauntlet: prompt processing 2,712 / 2,501 / 1,699 / 863 tok/s at 4K / 16K / 48K / 100K (was 2,220 / 2,060 / 1,486 / 815), 4 streams 500, all-8-bit prefill 3,021.
  - Remaining at 128 rows (per step, no head, ~44 ms): attention projection 7.9 ms, experts 14.8, attention kernels 9.2, attention output 2.7, router 1.4. At 8 rows the step is 51% expert reads (bandwidth-bound) and no single glue kernel is above 0.34 ms.
  - Review: no exactness defect. Fixed: pack de-duplication could reuse a buffer that was never packed (unreachable from today's call sites); a dead slice is skipped rather than multiplied by zero; k_tasks ignores an out-of-range expert id. verify.sh now checks batch sizes 128/100/56/48/38/37/30/24/9/8 and plain against packed kernels; interleave uses runs of 1–45 rows.
- 2026-11-23 lever 4: prompt-only steps of 256 rows (and the expert grid bound). Bit-identical. → KEPT.
  - Why bigger prompt batches help: the experts are the memory-bound part of a prompt step (at 128 rows a visit reads 93 distinct experts, at 256 rows 105: 13% more bytes for twice the tokens). The dense kernels are compute-bound at these sizes and gain nothing.
  - Why not simply raise the batch limit: a sliding-window layer's key/value ring holds the window (1,024) plus 128 new positions, and all rows' keys and values are written before any score is read, so 256 rows of one conversation would overwrite slots the first rows still need. A bigger ring (1,280) would make every local-attention launch 11% bigger, decode included (about +0.6% on a one-row step). Instead the attention part of each visit runs in two halves of ≤ 128 rows (the kernels already took a row offset; `k_attn_prep` gained one) and everything else runs once for the step. Ring, attention scratch and decode steps are unchanged; other per-row buffers doubled (+~0.1 GiB).
  - The output head still takes 128 rows at most (its logits buffer is 102 MB per 128 rows): the scheduler uses 256-row steps only when no prompt can end in the step.
  - Expert kernel grids were T × 8 task rows tall; at most (N + 7·min(128, N)) / 8 tasks can exist (240, not 1,024, at 128 rows; from the loop-architecture study): w13 205 → 183 µs, w2 104 → 95 µs at 128 rows (3.4% of a prompt step).
  - Measured, prompt-only steps over a 4K prompt (per 128 tokens): 128-row steps 46.6 ms (2,745 tok/s), 192-row 44.7 ms, 256-row 41.9 ms (3,057 tok/s, +11%). Harness prefill 4K: 2,897–2,954 → 3,105 / 3,150 / 3,289 tok/s. Gauntlet (request to first token): 3,124 / 2,783 / 1,841 tok/s at 4K / 16K / 48K (was 2,712 / 2,501 / 1,699). Decode unchanged: c2 394–396, c4 485–490.
  - verify.sh: prompt steps of 256 / 200 / 129 rows over 2,304 / 2,400 / 1,408 tokens, followed by 16-row steps, give the same tokens as 16-row steps throughout. (The first version of this test stopped at 1,100 tokens and could not fail: the ring hazard needs a big step past position 1,152. The review's simulation showed it passing with the split removed.) All earlier checks unchanged (NLL 0.9007).
  - Not done: 384- or 512-row steps would shorten a prompt step by an estimated further 6–9% (111 and ~113 distinct experts: 30–46% fewer expert bytes per token), but the task-list kernel's per-expert slot map no longer fits a block's 48 KB of shared memory, and scratch grows by ~0.25 GiB.
- 2026-11-23 lever 3: exact two-stage output head. Output identical to the full head. → KEPT.
  - The full head reads the 922 MB unembedding matrix every step (0.57 ms, whatever the number of rows) to find one winner per row. Feasibility first, on CPU with real hidden states (2,300 teacher-forced positions, 456 from the engine, 9,200 more for robustness): raw top logits are ~50 and unembedding rows have norm 1.000, so a Cauchy–Schwarz bound on an 8-bit pass (≈10 logits with 8-bit weights and activations) leaves few tokens that could still win. With the threshold taken from the EXACT logit of the approximate winner: mean 3–17 candidates per row, p99 ≤ 350, max ~4,600 (legal text). Without that refinement (threshold = best lower bound): mean 1,650, p99 47K: useless. Plain 4-bit weights: mean 1,800 on legal text; 4-bit weights and activations: dead; low-rank projection: bound 120–150 logits, everything is a candidate.
  - Built: approximate logits from the drafter's MXFP8 copy of the matrix (no extra memory) and 8-bit activations (289–307 µs for 1–32 rows, against 563–776); per token |u − ũ| and |u| stored at load (computed from the stored codes, rounded up), per row |h̃| and |h − h̃| + 5e-4·|h|; exact logits of each row's approximate winners; candidates = tokens whose approximate logit + bound reaches the threshold (greedy criterion, and for sampling rows capped logit / T + Gumbel noise); exact logits of the candidates' 16-row blocks by a kernel that repeats the full product's operations in order (bit-identical values); winner by the full head's rules. More than 4,096 candidate blocks → the step reports it and the full head runs (outside the recorded graph). Used for steps of ≤ 32 rows whose caller reads only argmax and sample (the server's default path); rows that need probabilities (sampled drafts, nll) and the 8-bit head mode take the full head. `INK_HEAD_FULL=1` turns it off.
  - `kuroshio headcheck`: 302 steps of 1–32 rows at temperature 0 / 0.6 / 1 over three texts, both heads on the same rows: 0 of 3,254 rows differ; worst approximation error 12.4% of its bound; 7.1 candidate blocks per row on average, 3,265 at most in one step (32 rows of a licence text), no overflow. In verify.sh, with a run that forces the fall-back path every step.
  - A/B, fixed prompts and seeds, one stream, tok/s, full head → two-stage (two runs each, within 0.5%): prose 222 → 228 and 226 → 232; rewrite code 484 → 496 and 623 → 638; quote 443 → 455 and 412 → 422; patterns 379 → 389 and 421 → 432: +2.3% to +3.2% everywhere. Greedy output hashes identical.
  - Harness on a quiet machine (temp 0.6): c1 241 / 251 / 253 / 257, c2 386 / 395 / 405, c4 495 / 501 / 516, prefill 3,207 / 3,290 / 3,302, no speculation 166.2 (was 158).
  - Review: no defect on realistic inputs. Changed after it: samples break exact ties towards the lower token id in BOTH heads (the full head used scan order; a tie is ~1e-6 of rows); thresholds are compared as capped logits, the unit the full head compares in (a raw-logit threshold could miss a capped tie above ~150 logits); a row with no valid approximate winner lets nothing through; headcheck also compares the step's own (graph-replayed) result and mixes temperatures 0 / 0.05 / 0.3 / 0.6 / 1 / 2 within a step: 482 steps, 5,254 rows, 0 differ.
  - Soak follow-up: GPU memory plateaus (23.0 → 23.6 GiB within 3 minutes, then both ways over 15 minutes); recording graphs does not leak host memory (1,536 re-recordings: +35 MiB, then flat). Host memory crept 778 → 894 MiB in the 15-minute soak: glibc's per-thread arenas (a thread per connection). `mallopt(M_ARENA_MAX, 2)` at server start: flat at 958 MiB over the last 4 minutes of a 7-minute repeating load (was still rising).
  - A scare, resolved: after the review fixes and the arena limit the A/B read 224 / 228 (prose) where it had read 228 / 232. Four runs back to back: full head 218 / 223 with the arena limit and 219 / 223 without; two-stage 226 / 229 with and 226 / 229 without. So the head is worth +2.5–3.5% as before, the arena limit costs nothing, and the whole machine was ~1.5% slower that hour. (Compare A and B in the same session, never against yesterday.)
