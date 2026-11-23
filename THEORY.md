# THEORY — ceilings

Measured on this card (tools/hwprobe): 1,692 GB/s read-only bandwidth (94% of the 1,792 spec),
404 / 451 / 483 TFLOPS FP16 GEMM at 4096³ / 8192³ / 16384³.

## Decode, one stream, no speculation (memory bound)

Bytes read per token, from the checkpoint's tensor sizes (48 layer visits; see docs/ARCHITECTURE.md §1):

| component | per visit | visits | GB / token |
|---|---:|---:|---:|
| attention weights, bf16 (q, r, gate, sink, k, v, o) | 53.8 MB | 48 | 2.58 |
| NVFP4 experts, 8 of 128 (layers 0–19) | 8 × 4.48 MB | 44 | 1.58 |
| MXFP8 experts, 8 of 128 (layers 20–23) | 8 × 8.21 MB | 4 | 0.26 |
| loop controllers (passes 2–3) | 4 × 1.18 MB | 24 | 0.11 |
| output head, bf16 201,024 × 2,304 | — | 1 | 0.93 |
| **total** | | | **5.46** |

Ceiling = 1,692 / 5.46 ≈ **310 tok/s**. KV reads are negligible at short context (4 KB per token per visit).

Observed: per-kernel latency, not bandwidth, is what binds today (≈800 kernels per token; see PLAN.md).
At full bandwidth the bytes above take 3.2 ms; the engine currently takes ≈9.5 ms.

Measured floors on this card (tools/hwprobe/kfloor.cu, kcold.cu): an empty graph-replayed kernel costs 0.67 µs;
a cold 39.7 MB read takes 27 µs (≈1,460 GB/s, the practical bandwidth for this access pattern); a cold 0.6 MB read 2.2 µs.
At 1,460 GB/s the 5.46 GB take 3.7 ms → practical ceiling ≈ 265 tok/s without speculation.
Engine today: base step 6.5 ms (≈154 tok/s), of which ≈3.8 ms is the memory-bound matrix kernels and the rest glue.

## Decode with DSpark (measured)

Acceptance on Mini is modest. vLLM's own counters (two prompts, greedy): per-position acceptance of the 7 drafts
≈ 60/34/23/14/7/4/3 % (prose) and 68/45/34/23/16/9/8 % (code). Expected tokens per step when verifying k drafts:
1 + sum of the first k rates → prose 1.60 / 1.94 / 2.17 / 2.31 / 2.46 for k = 1 / 2 / 3 / 4 / 7.

Cost of a step = verify pass over k+1 positions + draft (≈1 ms). Raw verify times at 1,500 tokens of context:
6.8 / 7.6 / 9.0 / 9.5 / 10.5 / 11.1 / 12.5 ms for 1 / 2 / 3 / 4 / 5 / 6 / 8 positions. Each extra position adds ≈1.3 ms,
which is the extra experts it routes to (8 per layer visit, 1.84 GB) at memory speed — so the marginal cost of a
verified draft is memory-bound and cannot shrink much.

Tokens per millisecond therefore peaks at k = 3–4: measured 222 / 200–205 tok/s (code / prose, greedy) against
137 without speculation; k = 7 gives 205 / 181.

Remaining levers on this ceiling: a cheaper draft (its two passes over output-head-sized tables are ≈1 ms of the
≈11 ms step), better drafts (only via the model), and a cheaper base step.

## Prefill

Measured, 128 tokens per batch: ≈2,180–2,420 tok/s lossless, ≈2,670–3,030 in 8-bit mode (vLLM ≈495).
Per token at 64 per batch (profiled): experts ≈160 µs (memory-bound: at this batch size every one of the 128 experts
of a layer is used, so all 15.7 GB of expert weights are read once per batch), attention projections ≈100 µs,
attention itself ≈94 µs, the rest ≈200 µs. Doubling the batch halves the expert and dense shares per token.
Remaining levers: pre-packed activations for the dense GEMMs at large batch, batches above 128, and a single pass of
the dense matrices for batches above 64 (they are currently read twice per 128 tokens).
