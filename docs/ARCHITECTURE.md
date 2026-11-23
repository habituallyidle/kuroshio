# Inkling 2 Mini — forward-pass specification

Derived from the vLLM reference implementation. Purpose: implement a from-scratch
engine (CUDA C++ kernels + Rust host) on one RTX 5090 (SM120) without re-reading the
reference.

Citation convention: paths are relative to `reference/`, a local copy of the model implementation shipped in the official vLLM image (third-party code, not part of this repository).
`i2/` = `vllm-inkling2/`, `i1/` = `vllm-inkling/`. `IMG:` = a file copied out of the
docker image `vllm/vllm-openai:inkling-2-mini` (path under
`/usr/local/lib/python3.14/dist-packages/vllm/`); the image's `models/inkling2` tree is
byte-identical to `i2/`. Anything not verified is marked **UNSURE** and collected in §10.

Notation: `D = 2304` hidden, `H = 24` query heads, `HKV = 8` KV heads, `G = H/HKV = 3`,
`dh = 128` head dim, `DR = 16` (`d_rel`), `E = 1024` rel extent, `W = 4` conv taps,
`NE = 128` experts, `K8 = 8` experts per token, `I = 1152` expert intermediate,
`V = 201024` padded vocab, `V0 = 200058` real vocab. `bf16(x)` = round fp32 to bf16
(round-to-nearest-even). `rms(x) = sqrt(mean(x^2))`.

---

## 0. One-page summary

* 24 stored decoder layers; layers 6..17 (the "core") run 3 times, so one token costs
  **48 layer visits**. Every visit has its **own KV cache and its own conv state**.
* There is no ordinary residual stream. The model keeps a small **stack of block sums**
  (embedding, then one running sum per 6 layers). The input of every sublayer is a
  learned softmax mix over those blocks ("block attention residual", AttnRes) followed by
  an RMSNorm.
* Attention: GQA 24/8 heads, no RoPE. Position enters only through a learned relative
  bias `r · rel_proj[:, distance]` for distance < 1024. q/k/r are L2-clipped, v is
  RMS-normalised, K and V go through a 4-tap causal convolution over the token axis
  before caching. Output is multiplied by a per-element sigmoid gate and by
  `sigmoid(lse − sink)` (attention sink).
* 20 of 24 layers are local (window 1024); layers 5, 11, 17, 23 are global and use
  separate q/k/v inputs and per-token dynamic conv taps.
* MoE: sigmoid router + bias for selection, top-8 of 128, weights = 8 × renormalised
  sigmoids. Experts are softcapped SwiGLU. Layers 0–19 are NVFP4 (W4A4: activations are
  quantised to NVFP4 at run time too), layers 20–23 are MXFP8 (W8A8).
* The attention output and the MoE output each pass through another 4-tap causal conv
  before they are added to the block sum.
* DSpark: a 3-layer dense drafter reads 4 hidden-state taps of the target, proposes 7
  tokens in one pass plus a tiny recurrent head, and the target verifies 8 positions in
  one forward.

---

## 1. Tensor inventory

Obtained from the safetensors JSON headers only (payload never read).
Snapshot `ed40e7da625f594fc9c4990f5f66be24873c22d7`.

### 1.1 `model-00001-of-00001.safetensors` (1244 tensors, 19,525,870,752 payload bytes, header 148,408 B)

Layer index sets: `ALL` = 0..23; `LOCAL` = {0,1,2,3,4, 6,7,8,9,10, 12,13,14,15,16, 18,19,20,21,22};
`GLOBAL` = {5, 11, 17, 23}; `CORE` = 6..17; `FP4` = 0..19; `FP8` = 20..23.
Prefix `L.` = `model.llm.layers.{N}.`.

| Name pattern | dtype | shape | bytes each | count | Used by |
|---|---|---|---|---|---|
| `model.llm.embed.weight` | BF16 | [201024, 2304] | 926,318,592 | 1 | token embedding |
| `model.llm.embed_norm.weight` | BF16 | [2304] | 4,608 | 1 | RMSNorm on embeddings |
| `model.llm.norm.weight` | BF16 | [2304] | 4,608 | 1 | final RMSNorm |
| `model.llm.unembed.weight` | BF16 | [201024, 2304] | 926,318,592 | 1 | LM head (fp32 output) |
| `model.llm.attn_res.attn_queries.{M}`, M ∈ 0..22, M∉{4,10,16,22} | F32 | [2304] | 9,216 | 19 | AttnRes query for the attention input of layer M+1 (local) |
| `model.llm.attn_res.attn_queries.{M}`, M ∈ {4,10,16,22} | F32 | [3, 2304] | 27,648 | 4 | same, for global layer M+1: rows = (q, k, v) inputs |
| `model.llm.attn_res.mlp_queries.{N}`, N ∈ ALL | F32 | [2304] | 9,216 | 24 | AttnRes query for the MoE input of layer N |
| `model.llm.attn_res.output_query` | F32 | [2304] | 9,216 | 1 | AttnRes query for the final output |
| `model.llm.middle_cycle_cross_pass_queries` | F32 | [24, 2304] | 221,184 | 1 | loop message queries, viewed [12, 2, D]: per core layer (attn-stream, mlp-stream) |
| `model.llm.middle_cycle_cross_pass_kv_queries` | F32 | [24, 2304] | 221,184 | 1 | loop message queries, viewed [12, 2, D]: per core layer (k-stream, v-stream) |
| `L.attn.wq_du.weight` | BF16 | [3072, 2304] | 14,155,776 | 24 | query projection (24 heads × 128) |
| `L.attn.wr_du.weight` | BF16 | [384, 2304] | 1,769,472 | 24 | relative-query projection (24 heads × 16) |
| `L.attn.gate_attn.weight` | BF16 | [3072, 2304] | 14,155,776 | 24 | output gate logits (24 heads × 128) |
| `L.attn.attn_sink_proj.weight` | BF16 | [24, 2304] | 110,592 | 24 | sink logit per query head |
| `L.attn.wk_dv.weight` | BF16 | [1024, 2304] | 4,718,592 | 24 | key projection (8 heads × 128) |
| `L.attn.wv_dv.weight` | BF16 | [1024, 2304] | 4,718,592 | 24 | value projection (8 heads × 128) |
| `L.attn.wo_ud.weight` | BF16 | [2304, 3072] | 14,155,776 | 24 | attention output projection |
| `L.attn.q_l2_clip.scale` | BF16 | [24, 128] | 6,144 | 24 | per-head, per-dim scale applied to q before L2 clip |
| `L.attn.rel_l2_clip.scale` | BF16 | [24, 16] | 768 | 24 | per-head scale applied to r before L2 clip |
| `L.attn.rel_logits_proj.proj` | BF16 | [16, 1024] | 32,768 | 24 | relative logits table: column = distance |
| `L.attn.k_sconv.weight`, N ∈ LOCAL | BF16 | [1024, 1, 4] | 8,192 | 20 | static K conv taps [channel, 1, tap] |
| `L.attn.v_sconv.weight`, N ∈ LOCAL | BF16 | [1024, 1, 4] | 8,192 | 20 | static V conv taps |
| `L.attn.k_sconv.static_w`, N ∈ GLOBAL | BF16 | [4, 1024] | 8,192 | 4 | static part of dynamic K taps [lag, channel] |
| `L.attn.k_sconv.weight_proj`, N ∈ GLOBAL | BF16 | [32, 2304] | 147,456 | 4 | dynamic K tap deltas: 8 heads × 4 lags from the K input |
| `L.attn.v_sconv.static_w`, N ∈ GLOBAL | BF16 | [4, 1024] | 8,192 | 4 | as above for V |
| `L.attn.v_sconv.weight_proj`, N ∈ GLOBAL | BF16 | [32, 2304] | 147,456 | 4 | as above for V |
| `L.attn_norm.weight`, N ∈ LOCAL | BF16 | [2304] | 4,608 | 20 | RMSNorm gain of the (single) attention input |
| `L.q_input_norm.weight`, `L.k_input_norm.weight`, `L.v_input_norm.weight`, N ∈ GLOBAL | BF16 | [2304] | 4,608 | 4 each | RMSNorm gains of the three attention inputs |
| `L.attn_sconv.weight` | BF16 | [2304, 1, 4] | 18,432 | 24 | static conv on the attention output |
| `L.mlp_sconv.weight` | BF16 | [2304, 1, 4] | 18,432 | 24 | static conv on the MoE output |
| `L.mlp_norm.weight` | BF16 | [2304] | 4,608 | 24 | RMSNorm gain of the MoE input |
| `L.mlp.gate.weight` | BF16 | [128, 2304] | 589,824 | 24 | router |
| `L.mlp.gate.bias` | F32 | [128] | 512 | 24 | router selection bias |
| `L.mlp.experts.w13_weight`, N ∈ FP4 | U8 | [128, 2304, 1152] | 339,738,624 | 20 | NVFP4 gate/up weights: [expert, out row, in/2] |
| `L.mlp.experts.w13_weight_scale`, N ∈ FP4 | F8_E4M3 | [128, 2304, 144] | 42,467,328 | 20 | block scales, 1 per 16 inputs |
| `L.mlp.experts.w13_weight_amax`, N ∈ FP4 | F32 | [128] | 512 | 20 | per-expert outer amax |
| `L.mlp.experts.w13_weight_outer_scale_denominator`, N ∈ FP4 | F32 | [1] | 4 | 20 | outer scale denominator |
| `L.mlp.experts.w2_weight`, N ∈ FP4 | U8 | [128, 2304, 576] | 169,869,312 | 20 | NVFP4 down weights: [expert, out row, in/2] |
| `L.mlp.experts.w2_weight_scale`, N ∈ FP4 | F8_E4M3 | [128, 2304, 72] | 21,233,664 | 20 | block scales |
| `L.mlp.experts.w2_weight_amax`, N ∈ FP4 | F32 | [128] | 512 | 20 | per-expert outer amax |
| `L.mlp.experts.w2_weight_outer_scale_denominator`, N ∈ FP4 | F32 | [1] | 4 | 20 | outer scale denominator |
| `L.mlp.experts.w13_weight`, N ∈ FP8 | F8_E4M3 | [128, 2304, 2304] | 679,477,248 | 4 | MXFP8 gate/up weights |
| `L.mlp.experts.w13_weight_scale`, N ∈ FP8 | U8 | [128, 2304, 72] | 21,233,664 | 4 | UE8M0 exponents, 1 per 32 inputs |
| `L.mlp.experts.w2_weight`, N ∈ FP8 | F8_E4M3 | [128, 2304, 1152] | 339,738,624 | 4 | MXFP8 down weights |
| `L.mlp.experts.w2_weight_scale`, N ∈ FP8 | U8 | [128, 2304, 36] | 10,616,832 | 4 | UE8M0 exponents |
| `L.controllers.{attn,k,v,mlp}.down_weight`, N ∈ CORE | BF16 | [64, 4608] | 589,824 | 12×4 | loop message controller, rank-64 down projection of [rms(dest), rms(message)] |
| `L.controllers.{attn,k,v,mlp}.gain_weight`, N ∈ CORE | BF16 | [2304, 64] | 294,912 | 12×4 | controller gain |
| `L.controllers.{attn,k,v,mlp}.forget_weight`, N ∈ CORE | BF16 | [2304, 64] | 294,912 | 12×4 | controller forget |
| `model.visual.*` (vision tower, 23 layers, `proj_in` [4096,4096], `proj_out` [2304,4096]) | BF16 | — | ≈ 0.63 GB total | 283 | images only; **not needed for text** |

Size roll-up (text path): experts NVFP4 20 × 573,309,960 ≈ 11.47 GB; experts MXFP8
4 × 1,051,066,368 ≈ 4.20 GB; embed + unembed 1.85 GB; attention 24 × 53,784,576 ≈ 1.29 GB;
controllers 12 × 4 × 1,179,648 ≈ 56.6 MB; everything else < 5 MB.

`hf_quant_config.json` groups: fp32 "cast" = AttnRes queries, router bias, loop queries;
bf16 "cast" = everything dense; `nvfp4` group = `w13_weight`/`w2_weight` of layers 0–19 with
`weight_block_shape [1,16]`, `weight_scale_format amax_denominator`,
`required_activation_quantization {amax_type per_token, block_size 16, format nvfp4_e2m1,
rounding nearest, scale_dtype float8_e4m3fn, scale_selection_rule fouroversix_mae}`;
`mxfp8` group = layers 20–23 with `weight_block_shape [32,32]`,
`required_activation_quantization {block_size 32, format mxfp8_e4m3, scale_dtype ue8m0}`;
`unembed` is bf16 with `fwd_output_dtype float32`. The reference selects MXFP8 per layer
from this file (`i2/quant_config.py:76-78`, `i2/nvidia/model.py:208-212`).

Name remapping done by the reference loader (`i2/nvidia/model.py:530-548`): the six
`wq_du, wr_du, gate_attn, attn_sink_proj, wk_dv, wv_dv` weights are concatenated in that
order into one `qkvr` matrix (rows 0..3071 q, 3072..3455 r, 3456..6527 gate, 6528..6551
sink, 6552..7575 k, 7576..8599 v); `q/k/v_input_norm` become rows 0/1/2 of a stacked
`attn_norm`. All AttnRes queries and loop message queries are **multiplied by
`D^-0.5 = 1/48` at load** (`i2/nvidia/model.py:158-161`, `i2/nvidia/loop.py:167-176`).

### 1.2 `mtp.safetensors` (54 tensors, 635,795,744 payload bytes, header 5,664 B) — DSpark drafter

| Name pattern | dtype | shape | bytes each | count | Used by |
|---|---|---|---|---|---|
| `fuse_weight` | BF16 | [4, 4] | 32 | 1 | tap mixing logits: rows 0..2 = draft layers, row 3 = Markov head; columns = 4 target taps |
| `mask_embed` | BF16 | [2304] | 4,608 | 1 | embedding of the 7 mask positions |
| `norm.weight` | BF16 | [2304] | 4,608 | 1 | final RMSNorm of the drafter |
| `layers.{0,1,2}.input_layernorm.weight` | BF16 | [2304] | 4,608 | 3 | pre-attention RMSNorm |
| `layers.{0,1,2}.post_attention_layernorm.weight` | BF16 | [2304] | 4,608 | 3 | pre-MLP RMSNorm |
| `layers.{0,1,2}.dflare_norm.weight` | BF16 | [2304] | 4,608 | 3 | RMSNorm of the mixed target features (context K/V input) |
| `layers.N.self_attn.q_proj.weight` | BF16 | [3072, 2304] | 14,155,776 | 3 | block queries |
| `layers.N.self_attn.k_proj.weight` | BF16 | [1024, 2304] | 4,718,592 | 3 | block keys |
| `layers.N.self_attn.v_proj.weight` | BF16 | [1024, 2304] | 4,718,592 | 3 | block values |
| `layers.N.self_attn.k_proj_ctx.weight` | BF16 | [1024, 2304] | 4,718,592 | 3 | context keys (from target features) |
| `layers.N.self_attn.v_proj_ctx.weight` | BF16 | [1024, 2304] | 4,718,592 | 3 | context values |
| `layers.N.self_attn.o_proj.weight` | BF16 | [2304, 3072] | 14,155,776 | 3 | attention output |
| `layers.N.self_attn.q_norm.weight`, `k_norm.weight`, `k_norm_ctx.weight` | BF16 | [128] | 256 | 3 each | per-head RMSNorm gains |
| `layers.N.mlp.gate_proj.weight`, `up_proj.weight` | BF16 | [9216, 2304] | 42,467,328 | 3 each | dense SwiGLU |
| `layers.N.mlp.down_proj.weight` | BF16 | [2304, 9216] | 42,467,328 | 3 | dense SwiGLU |
| `markov_head.markov_w1.weight` | BF16 | [201024, 256] | 102,924,288 | 1 | previous-token embedding of the Markov head |
| `markov_head.rnn_proj.weight` | BF16 | [768, 5120] | 7,864,320 | 1 | recurrence: input [state 256, w1 256, hidden 2304, f_rnn 2304] → [gate, cand, out] × 256 |
| `markov_head.rnn_proj.bias` | BF16 | [768] | 1,536 | 1 | |
| `markov_head.hidden_out.weight` | BF16 | [2304, 256] | 1,179,648 | 1 | latent → hidden, then shared LM head |
| `markov_head.hidden_out.bias` | BF16 | [2304] | 4,608 | 1 | |
| `markov_head.dflare_norm.weight` | BF16 | [2304] | 4,608 | 1 | RMSNorm of the mixed features for `f_rnn` |

The drafter shares the target's `embed.weight`, `embed_norm.weight` and `unembed.weight`
(`i2/nvidia/dspark.py:447-449, 486-494`).

---

## 2. Token-level flow

### 2.1 Visit schedule (the loop)

`i2/nvidia/loop.py:339-342`, `i2/configs.py:166-181`.

```
visits = [(i, 0) for i in 0..5]                      # prelude        visit ids  0..5
       + [(i, p) for p in 0..2 for i in 6..17]       # core, 3 passes visit ids  6..41
       + [(i, 0) for i in 18..23]                    # coda           visit ids 42..47
```

Visit id of layer `i`: prelude `i`; core pass `p`: `i + 12*p`; coda `i + 24`.
`num_layer_visits = 24 + 2*12 = 48`.

What is per visit vs per stored layer:

* **Shared across the 3 passes** (same weights): all attention weights, norms, conv taps,
  router, experts, AttnRes queries of that layer.
* **Separate per visit**: the KV cache (`Inkling2AttentionPassCache`,
  `i2/nvidia/attention.py:132-163, 268-277, 323-328`) and the conv state for all four
  streams (`i2/nvidia/model.py:182-192`, `i2/nvidia/loop.py:220`). `loop_share_cache` is a
  var-loop-only option and is rejected for this checkpoint (`i2/configs.py:146-151`).
  So: **48 KV caches and 48 conv states per sequence.**
* **Different in passes 1 and 2**: every core sublayer input (attention q/k/v inputs and
  the MoE input) is blended with a "message" read from the previous pass through a gated
  rank-64 controller (§2.4). Local core layers then receive three distinct q/k/v inputs
  instead of one. Pass 0 is an ordinary run of the layers.

### 2.2 The block stack (AttnRes state)

`n_layers_per_attn_res_block = 6`. Per token the model keeps `stack[0..4]`, each bf16 `[D]`:

* `stack[0]` = normalised embedding. It is the first "completed block".
* `stack[1]` = running sum of the outputs of layers 0..5; `stack[2]` = layers 6..11;
  `stack[3]` = layers 12..17; `stack[4]` = layers 18..23.
* `num_blocks` = number of completed blocks; `stack[num_blocks]` is the running partial
  block. A layer with `i % 6 == 0` ("first") **completes** the current partial and starts a
  new one: its attention input still treats `stack[num_blocks]` as the partial, but its
  MoE input mix sees it as a completed block, and the layer's output starts a fresh
  block (no carry-over).

Maximum 4 completed blocks + 1 partial = 5 slots (`i2/nvidia/loop.py:189-197`).

### 2.3 Pseudocode for one forward over `T` new tokens of one sequence

Shapes are per token; everything is row-wise over `T`. `mix(...)` is §4.
`blend(...)` is §2.4. `attention(...)` is §3. `moe(...)` is §5. `sconv_static(...)` is §3.4.
Source: `i2/nvidia/loop.py:199-398`.

```
# --- embedding (i2/nvidia/model.py:303-304, 104-128)
e      = embed.weight[token_id]                               # bf16 [D]
stack[0] = rmsnorm(e, embed_norm.weight, eps=1e-6) -> bf16    # [D]
tau    = 1 + 0.1 * ln(max((pos + 1) / 8192, 1))               # fp32 scalar per token (§3.3)
num_blocks = 0
inputs = None

for visit, (i, p) in enumerate(visits):
    looped = p > 0
    layer  = layers[i];  is_local = i not in {5, 11, 17, 23};  first = (i % 6 == 0)

    # --- pass bookkeeping (loop.py:349-371)
    if i == 6 and p == 0:
        prelude_blocks  = num_blocks                 # == 1
        prelude_partial = copy(stack[num_blocks])    # stack[1], complete sum of layers 0..5
    elif i == 6:                                     # start of pass 1 or 2
        bank = copy(stack[prelude_blocks+1 : num_blocks+1])   # = [stack[2], stack[3]] of the
                                                              #   previous pass, bf16 [2, D]
        num_blocks = prelude_blocks                  # back to 1
        stack[num_blocks] = prelude_partial
        if prelude_dests is None:
            prelude_dests = mix(sources=[stack[0], prelude_partial],
                                queries=attn_queries[5] ([1, D]),
                                norm=None)           # fp32 [1, D], no output norm
        inputs = prelude_dests

    if visit in {5, 23, 41}:                         # DSpark taps (§7.2)
        aux.append(copy(stack[num_blocks]))          # bf16 [D]

    partial = stack[num_blocks]

    # --- attention inputs
    if looped:
        # inputs is fp32 [Q, D] with Q = 1 (local) or 3 (global), un-normalised
        msg = mix(sources=[bank[0], bank[1]], queries=message_queries[i-6] ([4, D]),
                  norm=None)                         # fp32 [4, D]; streams (attn, k, v, mlp)
        dests = inputs if Q == 3 else repeat(inputs[0], 3)
        nw    = [q_input_norm, k_input_norm, v_input_norm] if not is_local
                else [attn_norm, attn_norm, attn_norm]
        x_q, x_k, x_v = blend(streams 0..2, dests, msg[0..2], nw, eps=1e-6)   # bf16 [D] each
    elif inputs is None:                             # only visit 0
        x_q = x_k = x_v = rmsnorm(partial, attn_norm.weight, 1e-6)            # layer 0 is local
    else:
        # inputs is bf16 [Q, D], already normalised by the previous visit's fused mix
        (x_q, x_k, x_v) = inputs if Q == 3 else (inputs[0], inputs[0], inputs[0])

    attn_raw = attention(layer, visit, pos, x_q, x_k, x_v,
                         tau if not is_local else None)       # bf16 [D]  (§3)

    # --- attention-output conv, then MoE input mix (loop.py:254-282)
    attn_out = sconv_static(attn_raw, layer.attn_sconv, state=(visit, ATTN)) -> bf16 [D]
    if first:
        srcs = stack[0 .. num_blocks] (inclusive)  +  [attn_out]
    else:
        srcs = stack[0 .. num_blocks-1]            +  [partial + attn_out]   # sum in fp32
    if looped:
        m      = mix(srcs, mlp_queries[i] ([1, D]), norm=None)               # fp32 [1, D]
        mlp_in = blend(stream 3, m, msg[3], [mlp_norm], 1e-6)[0]             # bf16 [D]
    else:
        mlp_in = mix(srcs, mlp_queries[i], norm=mlp_norm, eps=1e-6)[0]       # bf16 [D]

    moe_out = moe(layer, mlp_in)                                             # bf16 [D]  (§5)

    # --- MoE-output conv + residual update (loop.py:284-316; ops/sconv.py:379-385)
    y = sconv_static(moe_out, layer.mlp_sconv, state=(visit, MLP))           # bf16 [D]
    if first:
        num_blocks += 1
        stack[num_blocks] = bf16( fp32(attn_out) + fp32(y) )
    else:
        stack[num_blocks] = bf16( (fp32(partial) + fp32(attn_out)) + fp32(y) )

    # --- mix for the next visit, fused here (loop.py:374-380)
    if visit == 47:
        out = mix(stack[0..num_blocks-1] + [stack[num_blocks]],
                  output_query ([1, D]), norm=norm.weight, 1e-6)[0]          # bf16 [D]
    else:
        (ni, np_) = visits[visit+1]
        q = attn_queries[ni-1]                       # [1, D] local, [3, D] global
        if np_ > 0:   inputs = mix(same sources, q, norm=None)               # fp32 [Q, D]
        else:         inputs = mix(same sources, q, norm=layers[ni].attn_norm rows)  # bf16
        # At a pass restart (ni == 6, np_ > 0) this result is discarded and replaced by
        # prelude_dests.

# --- head (i2/nvidia/model.py:84-101, 575-587; IMG: model_executor/layers/logits_processor.py:98-120)
logits = fp32( out(bf16) @ unembed.weight(bf16).T )          # fp32 accumulate, fp32 result [V]
logits = logits[:200058]                                     # drop padded vocab rows
logits = 30.0 * tanh(logits / 30.0)                          # fp32
aux.append(out)                                              # 4th DSpark tap
```

Consequences worth stating explicitly:

* During pass `p > 0` the stack slots 2 and 3 are **overwritten** as the pass proceeds; the
  previous pass's two blocks survive only in `bank`. The coda sees only the last pass.
* `bank` is the *complete* previous pass (blocks of layers 6..11 and 12..17) for *every*
  core layer of the current pass, including layer 6.
* In pass 0, the output mix after layer 17 is computed but thrown away.
* `msg` is recomputed per core layer with that layer's 4 message queries.
* The `mix` for the next visit uses the *next* layer's query and norm weights.

### 2.4 Loop message controller ("gates", `gate_rank = 64`)

`i2/nvidia/loop.py:37-63, 112-130`, `i2/nvidia/ops/loop.py:18-132`.

Message queries (per core layer `j = i − 6`), all pre-scaled by 1/48:

```
message_queries[j][0] (attn / q-side) = middle_cycle_cross_pass_queries   [2j + 0]
message_queries[j][1] (k)             = middle_cycle_cross_pass_kv_queries[2j + 0]
message_queries[j][2] (v)             = middle_cycle_cross_pass_kv_queries[2j + 1]
message_queries[j][3] (mlp)           = middle_cycle_cross_pass_queries   [2j + 1]
```

`blend` for one stream `s` (weights `controllers.{attn,k,v,mlp}` in stream order 0..3),
`dest` and `message` fp32 `[D]`:

```
pd   = dest    * rsqrt(mean(dest^2)    + 1e-6)          # fp32, weightless
pm   = message * rsqrt(mean(message^2) + 1e-6)
pair = bf16(concat(pd, pm))                              # [2D] = [4608]
h    = bf16( pair @ down_weight.T )                      # [64]   bf16 GEMM, fp32 accumulate
gain   = fp32( h @ gain_weight.T )                       # [D]    fp32 output
forget = fp32( h @ forget_weight.T )                     # [D]
x    = (1 - tanh(forget)) * dest + tanh(gain) * message  # fp32, NOT fused into FMAs
y    = bf16( x * rsqrt(mean(x^2) + norm_eps) * norm_weight )   # the sublayer's RMSNorm
```

`norm_weight`: for streams 0..2 the layer's q/k/v input norm gains (a local layer uses
`attn_norm.weight` for all three), for stream 3 `mlp_norm.weight`. `norm_eps = 1e-6`.

---

## 3. Attention

Module: `i2/nvidia/attention.py:166-400`. Kernels: `i2/nvidia/ops/sconv.py` (prologue),
`i2/nvidia/ops/norm.py` (clips), `i2/nvidia/ops/attention_sm90.py` (the softmax kernel that
actually runs on SM120 — `use_sm100_kernels()` is false there, `i2/nvidia/ops/__init__.py:6-13`,
`i2/nvidia/attention.py:610-629`, `i2/nvidia/ops/attention.py:1233-1271`),
`i2/nvidia/ops/attention.py:252-337` (gate/sink).

### 3.1 Inputs and their norms

* Local layer, pass 0: one input `x = rmsnorm(mix, attn_norm.weight)`; `x_q = x_k = x_v = x`.
* Global layer (5, 11, 17, 23): three inputs, each from its own AttnRes query
  (`attn_queries[i-1]` rows 0,1,2) and its own RMSNorm gain (`q_input_norm`,
  `k_input_norm`, `v_input_norm`) (`i2/nvidia/model.py:202-206, 545-547`).
* Any core layer in passes 1–2: three inputs produced by `blend` (§2.4)
  (`i2/nvidia/attention.py:169-173`, `i2/nvidia/model.py:231-241`).

### 3.2 Projections

All are bias-free bf16 linears with fp32 accumulation and bf16 output
(`i2/nvidia/attention.py:330-348`):

```
q     = wq_du          @ x_q    # [3072]  -> view [24, 128]   (head h = columns h*128 .. h*128+127)
r     = wr_du          @ x_q    # [384]   -> view [24, 16]
gate  = gate_attn      @ x_q    # [3072]  -> view [24, 128]   output gate logits
sink  = attn_sink_proj @ x_q    # [24]    one sink logit per query head
k_raw = wk_dv          @ x_k    # [1024]  -> view [8, 128]
v_raw = wv_dv          @ x_v    # [1024]  -> view [8, 128]
out   = wo_ud @ gated_attention_output([3072])    # [2304]
```

Name decoding: `d` = model width (2304), `u` = query-side width (3072), `v` = KV width
(1024); `wq_du` maps d→u, `wo_ud` maps u→d.

Global layers additionally compute dynamic conv tap deltas
(`i2/nvidia/sconv.py:96-107`; plain bf16 linear because 2304 < 2560):

```
dk = k_sconv.weight_proj @ x_k   # [32] bf16 -> index [kv_head (8), lag (4)] = head*4 + lag
dv = v_sconv.weight_proj @ x_v   # [32]
```

### 3.3 q and r: scale, L2 clip, log scaling

`i2/nvidia/ops/norm.py:73-82` (`_l2_clip`), called per token from
`i2/nvidia/ops/norm.py:85-185`. Per query head `h`:

```
x      = fp32(q[h, :]) * fp32(q_l2_clip.scale[h, :])                  # [128]
ss     = sum(x^2);  n = ss * rsqrt(ss) if ss > 0 else 0                # = sqrt(ss), approx rsqrt
qc     = x * min(22.627416997969522 / (n + 1e-6), 1.0)
if global layer:  qc = fp32(bf16(qc)) * tau                            # round, then scale
qc     = bf16(qc)

x      = fp32(r[h, :]) * fp32(rel_l2_clip.scale[h, :])                 # [16]
rc     = same with threshold 8.0, eps 1e-6, and the same tau rule       -> bf16 [16]
```

`22.627… = sqrt(512) = 16·√2`. `tau = 1 + alpha * ln(max((pos+1)/n_floor, 1))` with
`n_floor = 8192`, `alpha = 0.1`, fp32, per token; it is exactly 1.0 for positions ≤ 8191
(`i2/nvidia/ops/attention.py:340-365`, `i2/nvidia/model.py:491-496`). `tau` is passed only
to global layers (`i2/nvidia/model.py:238`); local layers never use it, even in loop passes.

### 3.4 Short convolutions

Kernel: `i2/nvidia/ops/sconv.py:12-22, 95-183, 187-298`. All are causal depthwise
convolutions over the **token axis**, `W = 4`, with an identity skip and a variance
rescale. Taps at positions before 0 are zero.

**Static** (local-layer K and V; `attn_sconv`; `mlp_sconv`). Weight `w[c, k]` =
`weight[c, 0, k]`, `k = 0` is the oldest tap (lag 3), `k = 3` the current token:

```
y[t, c] = ( sum_{k=0..3} w[c, k] * x[t - 3 + k, c]  +  x[t, c] ) * (W + 1)^-0.5      # * 5^-0.5
```

**Dynamic** (global-layer K and V). `static_w[l, c]`, lag `l = 0` is the current token;
`d[t, head, l]` is the per-token delta from §3.2, shared by the 128 channels of a KV head:

```
y[t, c] = ( sum_{l=0..3} ( static_w[l, c] + d[t, c // 128, l] ) * x[t - l, c]  +  x[t, c] )
          * (2W + 1)^-0.5                                                              # * 1/3
```

The delta `d` is taken from the **current** token `t` for all four lags.

Numerics: the conv inputs are rounded to bf16 before use (they are what is stored in the
state); accumulation is fp32 with FMA in tap order (static: oldest first; dynamic: lag 0
first, the first product is a plain multiply); the tap weight for the dynamic case is the
fp32 sum of two bf16 values; `y` is rounded to bf16 (`ops/sconv.py:137-183, 298, 376, 795`).
The static scale for hidden streams is the literal `5**-0.5` (`i2/nvidia/ops/conv_mix.py:79`).

**State to keep across tokens** (`i1/nvidia/sconv_swa_attn.py:3-15, 162-173`,
`i2/nvidia/sconv.py:30`): for each of the 48 visits, the *inputs* of the four convs at the
previous `W − 1 = 3` positions:

| stream | what is stored | width | bytes/position (bf16) |
|---|---|---|---|
| K | `k_raw` (projection output, before conv/clip) | 1024 | 2,048 |
| V | `v_raw` | 1024 | 2,048 |
| ATTN | `attn_raw` = `wo_ud` output | 2304 | 4,608 |
| MLP | `moe_out` | 2304 | 4,608 |

= 6,656 values = 13,312 B per position per visit; × 3 positions × 48 visits = 1.92 MB per
sequence. (The reference pads each of 8 "heads" to 1024 values and pages it; that is a
vLLM artefact.) The state is indexed by absolute position, which makes speculative
rollback free: a rejected position is simply overwritten (§7.6). Keep at least
`3 + 7` positions when DSpark is on.

### 3.5 K clip, V norm, and what goes into the KV cache

`i2/nvidia/ops/sconv.py:706-813`. Per KV head `j`, after the conv:

```
yk = fp32(bf16(conv_k[j, :]))                                   # [128]
ss = sum(yk^2);  n = ss * rsqrt(ss) if ss > 0 else 0
K[j] = yk * min(22.627416997969522 / (n + 1e-6), 1.0)           # L2 clip, NO learned scale for k

yv = fp32(bf16(conv_v[j, :]))
V[j] = yv * rsqrt(mean(yv^2) + 1e-6)                            # RMS norm, no gain ("value norm no gamma")
```

Both are cast to the cache dtype and written at the token's position **before** the
attention of that token runs, so a token attends to itself.

**KV cache contents, per token, per visit:** `K` `[8, 128]` and `V` `[8, 128]` = 2,048
values.

| cache dtype | bytes / token / visit | notes |
|---|---|---|
| bf16 (`auto`) | 4,096 | default |
| fp8 E4M3 | 2,048 | SM120 only; values are stored **unit-scale** (a direct fp32→E4M3 cast, no scale factor) because K is clipped to ‖k‖ ≤ 22.6 and V is RMS-normalised (`i2/nvidia/attention.py:293-298, 353-355`, `i2/speculator_config.py:15-24`). The reference run used `--kv-cache-dtype fp8`. |

Retention: a local visit needs only the last 1024 positions (plus speculative headroom);
a global visit needs the whole sequence. Visits: local = 40 (layers 0–4: 5, core local
layers 6–10 and 12–16: 10 × 3, coda 18–22: 5); global = 8 (layer 5: 1, layers 11 and 17:
3 each, layer 23: 1). So per sequence:

* growing part: 8 × 4,096 = **32,768 B/token** bf16 (16,384 B fp8);
* fixed part: 40 × 1024 × 4,096 B = **167.8 MB** bf16 (83.9 MB fp8).

A local ring buffer must hold at least `1024 + 7` positions when DSpark is on, otherwise
a speculative write at `L+7` evicts a key that the post-rollback query at `L+1` still needs.

### 3.6 Scores, mask, relative bias

Per request, per KV head `j`, the 3 query heads `h = 3j, 3j+1, 3j+2` attend the same K/V
(GQA: query head `h` ↔ KV head `h // 3`; `i2/nvidia/ops/attention_sm90.py:117`,
`i2/nvidia/ops/attention.py:749`). For a query at absolute position `p` and a key at
position `s`, `dist = p − s`:

```
visible = (dist >= 0)                                   # causal, self included
if local:  visible &= (dist < 1024)                     # sliding window = keys p-1023 .. p
bias[h, dist] = sum_{i<16} rc[h, i] * rel_logits_proj[i, dist]     if dist < 1024
              = 0                                                   if dist >= 1024 (global only)
logit[h, s] = (qc[h] · K[h//3, s]) * 128^-0.5  +  bias[h, dist]     if visible else -inf
```

* `softmax_scale = head_dim^-0.5` (`i2/nvidia/attention.py:206`). The bias is added
  **unscaled** (the FA4 form is `(qk + bias/scale) * scale`,
  `i2/nvidia/ops/attention.py:79-83`; the SM120 kernel does `s = qk*scale*log2e;
  s += bias*log2e`, `attention_sm90.py:169-177`).
* Rel extent is 1024 for both layer kinds (`rel_extent = sliding_window_size` for local,
  `config.rel_extent` for global, both 1024; `i2/nvidia/attention.py:208`). For a global
  layer, keys further than 1023 back simply get zero bias
  (`i2/nvidia/ops/sconv.py:470-489`, `attention_sm90.py:175-176`).
* The bias table is computed per token as `rc (bf16 [24,16]) @ rel_proj (bf16 [16,1024])`
  with fp32 accumulation and **rounded to bf16** before use
  (`i2/nvidia/ops/norm.py:162-185`). The "sheared" storage layout in the reference is only
  an addressing trick for its kernels; an engine can compute `bias` in any layout.
* Local window tuple is `(sliding_window − 1, 0)` (`i2/nvidia/attention.py:209`).

### 3.7 Softmax, value sum, sink, gate, output

`i2/nvidia/ops/attention_sm90.py:147-227`:

```
m    = max_s logit[h, s];   l = sum_s exp(logit[h, s] - m)
P    = exp(logit - m)                        # fp32, then cast to bf16 before the P·V product
o[h] = ( sum_s bf16(P[s]) * V[h//3, s] ) / l # fp32 accumulate, [128]
lse[h] = m + ln(l)                           # natural log; -inf if no visible key (then o = 0)

y[h, :] = o[h, :] * sigmoid(gate[h, :])                 # per element
y[h, :] = y[h, :] * sigmoid(lse[h] - sink[h])           # per head scalar
y = bf16(y)                                             # [24*128] = [3072]
attn_raw = bf16( wo_ud @ y )                            # [2304]
```

`sigmoid(lse − sink) = Σexp(logits) / (Σexp(logits) + exp(sink))`: the sink is an extra
key with logit `sink[h]` and value 0 in the softmax. `gate` and `sink` are bf16
projections read as fp32. There is no attention bias term and no output bias
(`q_bias = o_bias = false`).

### 3.8 Which layers are which

| stored layer | kind | inputs (pass 0) | K/V conv | uses tau |
|---|---|---|---|---|
| 0–4, 6–10, 12–16, 18–22 | local, window 1024 | one (`attn_norm`) | static `[1024,1,4]` | no |
| 5, 11, 17, 23 | global | three (`q/k/v_input_norm`) | dynamic `static_w` + `weight_proj` | yes |

Each 6-layer AttnRes block is 5 local layers followed by one global layer.
Head counts and head dim are identical for both kinds (`swa_*` config values equal the
global ones).

---

## 4. Block attention residuals (`mix`)

`i2/nvidia/ops/attn_res.py:3-11, 170-306`; queries in `i2/nvidia/model.py:131-161`.

Inputs: an ordered list of `N` source vectors `x_n` (`[D]`, bf16 blocks widened to fp32;
the last source may be the fp32 sum `partial + addend` of two bf16 vectors), `Q` query
vectors `q` (fp32 `[D]`, **already multiplied by 1/48**), optional norm gain `w` (bf16
`[D]` per query), `eps = 1e-6` for source RMS, `norm_eps = 1e-6`.

For each query independently:

```
inv_rms_n = rsqrt( mean(x_n^2) + 1e-6 )                 # fp32 (fma(sum_sq, 1/D, eps))
logit_n   = (q · x_n) * inv_rms_n                       # dot of the query with the RMS-normalised source
p_n       = exp(logit_n - max_m logit_m)
inv_sum   = 1 / sum_n p_n                               # correctly rounded divide
mixed     = ( sum_n (p_n * inv_rms_n) * x_n ) * inv_sum # values are RMS-normalised too
                                                        # (attn_res_normalize_values = true)
if w is None:   return mixed                            # fp32 [D]
return bf16( mixed * rsqrt(mean(mixed^2) + norm_eps) * w )
```

The accumulation `sum_n` is FMA in source order (`tl.fma(p, x, acc)`), other operations
are unfused (`enable_fp_fusion=False`).

Where it is used, and with which sources (this table is the whole algorithm):

| mix | sources (in order) | query | norm |
|---|---|---|---|
| attention input of layer `i ≥ 1` | completed blocks `stack[0..nb-1]`, then partial `stack[nb]` | `attn_queries[i-1]` (1 or 3 rows) | next layer's `attn_norm` / `q,k,v_input_norm`; **none** (fp32) if that visit is in pass 1–2 |
| attention input of layer 0 | — (no mix) | — | `attn_norm(stack[0])` directly |
| MoE input, layer `i`, not first | completed blocks, then `partial + attn_out` | `mlp_queries[i]` | `mlp_norm`; none in pass 1–2 |
| MoE input, first layer of a block | completed blocks **including the old partial**, then `attn_out` alone | `mlp_queries[i]` | same |
| final output | completed blocks, then partial | `output_query` | `norm` |
| loop message | `bank[0]`, `bank[1]` | 4 message queries of the layer | none |
| `prelude_dests` | `stack[0]`, `prelude_partial` | `attn_queries[5]` | none |

What is stored and when: `stack[nb]` is rewritten at the end of every layer as
`bf16((partial + attn_out) + bf16(conv(moe_out)))` (or `bf16(attn_out + …)` for a first
layer). Blocks are stored in bf16; every mix is fp32 (`i2/nvidia/model.py:5-8`).
Nothing of the stack persists across tokens — it is per-token scratch. Number of sources
is 2 to 5.

---

## 5. MoE

`i2/nvidia/moe.py:69-410`, `i2/nvidia/ops/moe.py`, `i2/nvidia/ops/activation.py`,
`i2/nvidia/ops/moe_sm120.py`. Input `x` = `mlp_in`, bf16 `[D]`. Output bf16 `[D]`.

### 5.1 Router

`i2/nvidia/ops/moe.py:31-64, 110-145`.

```
logit[e]  = fp32( x(bf16) @ gate.weight[e](bf16) )        # e in 0..127; fp32 accumulate in
                                                          # K-groups of 1024 inputs summed in order
                                                          # ([0,1024) + [1024,2048) + [2048,2304))
sel[e]    = 1 / (1 + exp(-logit[e]))  +  gate.bias[e]     # sigmoid + fp32 bias  (selection score)
ids       = the 8 experts with the largest sel, sorted descending; ties -> lower expert index
ls[k]     = min(l, 0) - log1p(exp(-|l|))  with l = logit[ids[k]]     # log-sigmoid, WITHOUT bias
gamma[k]  = exp( ls[k] - logsumexp_k(ls) ) * route_scale  # route_scale = 8.0
```

So: selection uses `sigmoid(logit) + bias`; the mixing weights are
`8 · sigmoid(logit_k) / Σ_{selected} sigmoid(logit_j)` (sigmoid, renormalised over the
top-8; not a softmax over logits; the bias does not enter the weights). The 8 weights sum
to 8. The `logsumexp` uses the pairwise order
`((s0+s2)+(s1+s3)) + ((s4+s6)+(s5+s7))` over the descending-sorted slots
(`ops/moe.py:20-28`). There are no shared experts (`n_shared_experts = 0`).

### 5.2 Expert MLP

For each selected expert `e = ids[k]` with weight `gamma = gamma[k]`
(`i2/nvidia/moe.py:261-291`, `i2/nvidia/ops/activation.py:61-78`):

```
gu      = bf16( W13[e] @ x~ )                   # [2304]; x~ = x after activation quantisation (§6)
gate[j] = gu[2j],  up[j] = gu[2j + 1]           # j in 0..1151  -- INTERLEAVED, not two halves
sg      = sigmoid(gate[j])                      # of the un-capped gate
g       = 4.0  * tanh(gate[j] / 4.0)            # mlp_gate_softcap
u       = 25.0 * tanh(up[j]   / 25.0)           # mlp_up_softcap
a[j]    = bf16( ((g * sg) * u) * gamma )        # routing weight applied here, before w2
down    = bf16( W2[e] @ a~ )                    # [2304]; a~ = a after activation quantisation
```

`W13[e]` is `[2304 (= 2·1152 interleaved gate/up rows), 2304]`; `W2[e]` is `[2304, 1152]`.
Evidence for interleaving: the SwiGLU kernel reads each `(gate, up)` pair as one 32-bit
word, low 16 bits = gate (`ops/activation.py:61-78`), and on SM120 the checkpoint rows are
loaded unmodified (`i2/nvidia/moe.py:187-195`: the row shuffle is SM100-only).

### 5.3 Combine

`i2/nvidia/ops/moe.py:1057-1092`:

```
moe_out = bf16( sum_{k=0..7} fp32(down_k) )     # fp32 sum in slot order (descending selection score)
```

The routing weights are already inside `down_k`; the combine is an unweighted sum.

---

## 6. Quantised expert formats

### 6.1 NVFP4 weights (layers 0–19)

Storage: `w13_weight` U8 `[128, 2304, 1152]`, `w2_weight` U8 `[128, 2304, 576]`. Row-major
`[expert, output row n, input column k / 2]`.

* **Nibble order**: byte `b` of a row holds input column `2b` in the **low nibble** (bits
  0–3) and column `2b + 1` in the **high nibble** (bits 4–7)
  (`i2/nvidia/ops/nvfp4.py:66-84, 132-150`: `low_code | (high_code << 4)`, "`a` in the low
  nibble"; the weights are consumed by the same native `e2m1` MMA operand format as these
  activations, `i2/nvidia/ops/moe_sm120.py:57-67`).
* **FP4 E2M1 code** (`i2/nvidia/ops/nvfp4.py:27-63`): bit 3 = sign, bits 2–1 = exponent,
  bit 0 = mantissa.

  | code (low 3 bits) | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | magnitude | 0 | 0.5 | 1 | 1.5 | 2 | 3 | 4 | 6 |

  Codes 8–15 are the negatives (code 8 = −0).
* **Blocks**: 16 consecutive **input columns** of one output row share one scale
  (`weight_block_shape [1, 16]`). `w13_weight_scale` F8_E4M3 `[128, 2304, 144]`,
  `w2_weight_scale` `[128, 2304, 72]`: scale of `(e, n, k)` is `scale[e, n, k // 16]`
  (`moe_sm120.py:62-66`).
* **Outer scale**: per expert `amax[e]` (F32 `[128]`) divided by the single scalar
  `outer_scale_denominator` (F32 `[1]`) of that projection.
* **Dequantisation**:

  ```
  W[e, n, k] = e2m1( nibble(e, n, k) ) * e4m3( scale[e, n, k // 16] ) * amax[e] / denominator
  ```

  Derivation: the GEMM computes `Σ_k (a_code·a_blockscale)(w_code·w_blockscale)` natively,
  then multiplies by `row_scale = (x_amax · w_amax[e]) · rcp(1536 · denominator)`
  (`moe_sm120.py:67-69`, `ops/moe.py:346-354`, `ops/activation.py:170`); the activation's
  own outer scale is `x_amax / 1536`, leaving `w_amax[e] / denominator` for the weight. The
  Hopper path states the same thing directly: global scale `= amax / denominator`
  (`i2/nvidia/moe.py:209-214`).

### 6.2 NVFP4 activations (W4A4) — exact rule

The reference quantises both expert inputs at run time on Blackwell, and the checkpoint
metadata calls this *required* (`required_activation_quantization`). Rule
"fouroversix_mae" (`i2/nvidia/ops/nvfp4.py:1-8, 97-129`; callers `ops/moe.py:306-343`,
`ops/activation.py:162-206`). Per row `x` (fp32 widened from bf16; row = the `[2304]` MoE
input, once per token and shared by its 8 experts; or the `[1152]` SwiGLU output, per
(token, expert)):

```
amax  = max |x| over the whole row
enc   = min(1536 / amax, FLT_MAX)          # correctly rounded divide
dec   = 1 / enc
outer = amax * (1 / 1536)
for each block of 16:
    bmax = max |x_block|
    s6 = e4m3( min(bmax * (enc * (1/6)), 448) )       # candidate block scales, RN to E4M3
    s4 = e4m3( min(bmax * (enc * 0.25),  448) )
    inv6 = min(1 / (s6 * dec), FLT_MAX);  inv4 = min(1 / (s4 * dec), FLT_MAX)
    c6 = e2m1_rn_sat(x_block * inv6);     c4 = e2m1_rn_sat(x_block * inv4)   # RN-even, saturating
    err6 = sum | value(c6) * s6 * outer - x_block |    # pairwise sum, 16 -> 8 -> 4 -> 2 -> 1
    err4 = sum | value(c4) * s4 * outer - x_block |
    use (c4, s4) if err4 < err6 else (c6, s6)
real value of an element = value(code) * blockscale * (amax / 1536)
```

E2M1 rounding thresholds on `|v|` (ties to even code): ≤0.25→0, <0.75→0.5, ≤1.25→1,
<1.75→1.5, ≤2.5→2, <3.5→3, ≤5→4, else 6 (`nvfp4.py:27-45`); on Blackwell the PTX
instruction `cvt.rn.satfinite.e2m1x2.f32` does it (`nvfp4.py:69-84`).

Expert GEMM result (`moe_sm120.py:43-74`): `acc = Σ_k` block-scaled products in fp32,
`acc *= 1.0` (the "expert scale" buffer is all ones), `acc *= row_scale`, stored bf16.
`row_scale` for the gate/up GEMM = `(x_amax · w13_amax[e]) · rcp(1536 · den13)`; for the
down GEMM = `(a_amax · w2_amax[e]) · rcp(1536 · den2)`.

### 6.3 MXFP8 (layers 20–23)

* Weights: `w13_weight` F8_E4M3 `[128, 2304, 2304]`, `w2_weight` F8_E4M3 `[128, 2304, 1152]`,
  one byte per element, row-major `[expert, out row, in column]`.
* Scales: U8 exponent (UE8M0) per 32 input columns of a row: `w13_weight_scale`
  `[128, 2304, 72]`, `w2_weight_scale` `[128, 2304, 36]`.
* Dequantisation (`i2/nvidia/ops/moe_sm120.py:77-82`):

  ```
  W[e, n, k] = e4m3(byte) * 2^(scale_byte[e, n, k // 32] - 127)      # scale_byte 0 -> 2^-127
  ```

  No outer scale, no amax. On SM120 the reference dequantises both operands to bf16
  (exact: E4M3 has 3 mantissa bits) and runs a bf16 GEMM with fp32 accumulation, output
  bf16 (`moe_sm120.py:94-156`).
* Activations (W8A8; `i2/nvidia/ops/mxfp8.py:14-30`): per block of 32,
  `raw = amax_block / 448`; exponent = `raw` rounded **up** to a power of two
  (`((bits + 0x007FFFFF) & 0x7F800000) >> 23`, clamped to 254, 0 if `raw ≤ 2^-127`);
  `value = e4m3_rn( clamp(x / 2^(exp-127), ±448) )`. No per-row amax.
* The SwiGLU and routing are identical to the NVFP4 layers.

### 6.4 W4A4 vs W4A16 in the reference

| platform | path | activations |
|---|---|---|
| SM100 (B200/GB200) | TRT-LLM-gen cubins (`ops/trtllm_bmm.py`) | quantised (W4A4 / W8A8) |
| **SM120 (RTX 50)** | Triton `tl.dot_scaled` native NVFP4 MMA (`ops/moe_sm120.py`); optional Humming NVFP4 backend that also keeps quantised activations (`ops/moe_humming.py:44-49`) | **quantised (W4A4 / W8A8)** |
| SM90 (Hopper) | Humming GEMMs (`ops/moe_sm90.py:3-4`) | **bf16** (W4A16 / W8A16) |

So both exist in the reference, selected by hardware, not by a flag
(`i2/nvidia/moe.py:277-303, 385-410`). There is no bf16-expert path
(`i2/nvidia/model.py:789-794`). Everything outside the experts (router, attention,
norms, heads) is unquantised bf16/fp32 (`i2/quant_config.py:3-5`).

---

## 7. DSpark speculative decoding

Sources: `i2/nvidia/dspark.py`, `i2/nvidia/ops/dspark.py`, `i2/speculator.py`,
`i2/configs.py:238-306`; base classes `IMG: v1/worker/gpu/spec_decode/dspark/speculator.py`,
`IMG: …/dflash/speculator.py`, `IMG: …/rejection_sampler.py`,
`IMG: …/rejection_sampler_utils.py`.

### 7.1 Plain-language summary

After the target model has processed some tokens and sampled the next token `a` (the
"anchor", not yet run through the target), the drafter builds a block of 8 slots:
slot 0 holds the embedding of `a`, slots 1–7 hold one learned "mask" vector. The
3-layer drafter lets these 8 slots attend to each other in both directions and to a
key/value memory computed from the *target's* hidden states at every earlier position.
Each mask slot then yields a guess for the token at its own position — all 7 at once.
Because 7 independent guesses would not be mutually consistent, a tiny recurrent head
walks left to right, adding a correction to each slot's logits that depends on the tokens
chosen so far. The target then runs once over `[a, d1, …, d7]` (8 positions). Its logits
at position `k` say whether `d(k+1)` is acceptable. The longest accepted prefix is kept,
one more token comes for free from the target's own logits at the first disagreement (or
after the last draft), and everything computed for the rejected tail is discarded by
simply overwriting it on the next step.

### 7.2 Target hidden states consumed

`target_layer_ids = [5, 23, 41, 48]` are **visit ids** (loop passes counted), the last
being `num_layer_visits` = "final-norm output" (`i2/configs.py:258-261, 283-292`,
`i2/nvidia/loop.py:372-373, 396-397`). A tap at visit `v` is a copy of the running partial
block `stack[num_blocks]` **at the start** of that visit (bf16 `[D]`):

| tap | visit | pass, stored layer | value |
|---|---|---|---|
| 0 | 5 | prelude, layer 5 | `stack[1]` = sum of outputs of layers 0..4 |
| 1 | 23 | pass 1, layer 11 | `stack[2]` = sum of outputs of layers 6..10 in pass 1 |
| 2 | 41 | pass 2, layer 17 | `stack[3]` = sum of outputs of layers 12..16 in pass 2 |
| 3 | 48 | — | `out`, the final-normed hidden state that feeds the LM head |

Features for a token are the concatenation `[4, 2304]` bf16, stored raw
(`i2/speculator.py:22-29`). They are needed for **every** target position, prompt included.
The drafter requires the checkpoint's 3 passes (`i2/configs.py:272-277`).

Mixing (`i2/nvidia/dspark.py:420-422, 506-508`, `i2/nvidia/ops/dspark.py:10-37`):

```
alpha = softmax(fp32(fuse_weight), dim=-1)            # [4, 4]; stored back in fuse_weight's dtype
mixed(row) = sum_tap alpha[row, tap] * feature[tap]   # fp32
ctx(row)   = mixed * rsqrt(mean(mixed^2) + 1e-6) * norm_weight   -> bf16 [D]
```

Row `i` ∈ 0..2 with `layers.i.dflare_norm.weight` feeds draft layer `i`'s context K/V;
row 3 with `markov_head.dflare_norm.weight` gives `f_rnn`.

### 7.3 Drafter network

`mtp_config`: 3 layers, hidden 2304, 24 heads / 8 KV heads × 128, intermediate 9216,
`local_layer_ids [0, 2]`, window 1024, RoPE theta 1e6, eps 1e-6, block size 8.
Standard pre-norm dense transformer (`i2/nvidia/dspark.py:289-318, 433-439`).

**Context K/V** (written for every valid target position `p`, per draft layer `i`;
`dspark.py:249-265, 517-530`):

```
c       = ctx(i) at position p                               # bf16 [D]
k       = k_proj_ctx @ c   -> [8, 128];   v = v_proj_ctx @ c -> [8, 128]
k       = rmsnorm_per_head(k, k_norm_ctx.weight, 1e-6)       # bf16
k       = rope(k, position p)                                # fp32 math, bf16 result
cache_i[p] = (k, v)                                          # v is used raw
```

**Block forward** (8 slots at positions `L .. L+7`, `L` = position of the anchor;
`dspark.py:424-439, 194-247`):

```
x[0]    = rmsnorm(embed.weight[a], embed_norm.weight)        # the target's embedding + norm
x[1..7] = mask_embed
residual = x;  h = rmsnorm(x, input_layernorm_0)
for i in 0..2:
    qkv = [q_proj; k_proj; v_proj] @ h                       # [3072 | 1024 | 1024]
    q   = rope( bf16(rmsnorm_per_head(q, q_norm.weight)), pos )      # 24 heads
    k   = rope( bf16(rmsnorm_per_head(k, k_norm.weight)), pos )      # 8 heads
    cache_i[L + slot] = (k, v)                               # block K/V go into the same cache
    keys visible to EVERY slot of the block (bidirectional inside the block):
        global layer (i = 1):      all positions 0 .. L+7
        local layers (i = 0, 2):   positions (L+7) - 1031 .. L+7     # window = 1024 + 8
    o   = softmax( (q · K) * 128^-0.5 ) · V                  # plain attention, GQA h -> h//3,
                                                             # no relative bias, gate or sink
    h   = o_proj @ o
    residual = residual + h;  h = rmsnorm(residual, post_attention_layernorm_i)
    g, u = split( [gate_proj; up_proj] @ h )                 # two contiguous halves of 9216
    h   = down_proj @ bf16( (g / (1 + exp(-g))) * u )
    residual = residual + h
    h   = rmsnorm(residual, input_layernorm_{i+1}  or  norm (after the last layer))
hidden[slot] = h                                             # bf16 [8, D]
```

* Per-head RMSNorm: `x * (1 / sqrt(mean(x^2) + 1e-6)) * w`, rounded to bf16 **before** RoPE
  (`ops/dspark.py:94-106`).
* RoPE ("neox" layout, `ops/dspark.py:85-113`): `x1 = x[0:64]`, `x2 = x[64:128]`,
  `o1 = x1·cos − x2·sin`, `o2 = x2·cos + x1·sin`, with
  `angle[m] = pos · theta^(−2m/128)`, `m = 0..63`, `theta = 1e6`, fp32 table
  (`IMG: model_executor/layers/rotary_embedding/base.py:81-104`, `dspark.py:410-415`).
* Residual add + norm: the sum is formed in fp32, stored to the residual as bf16, and the
  norm is taken from the **unrounded** sum (`dspark.py:59-68`).
* Draft attention semantics: "every query of a request sits at its last position"
  (`ops/attention.py:903-905, 916-918`; `attention_sm90.py:119-122`), window
  `sliding_window_size + block` (`dspark.py:172-173`), draft `seq_len = L + 8`
  (`IMG: dflash/speculator.py:956-962`).
* The context K/V later written for accepted positions overwrite the block K/V stored at
  the same positions, so the cache always holds context K/V for `< L` and block K/V for
  the current block.
* Draft KV cache dtype: forced to bf16 on SM120 (`i2/speculator_config.py:15-24`).
  Bytes: 4,096 per position per draft layer; layer 1 keeps the whole sequence, layers 0
  and 2 need 1032 positions.

### 7.4 Proposing 7 tokens: base logits + recurrent Markov head

`i2/speculator.py:49-77`, `i2/nvidia/dspark.py:321-385, 544-567`,
`i2/nvidia/ops/dspark.py:188-254`.

```
base[i]  = bf16( hidden[i + 1] @ unembed.weight.T )          # i = 0..6, mask slots 1..7; bf16 GEMM
f_rnn    = ctx(row 3) of the LAST VALID target position      # = position L - 1 (bf16 [D])
C[i]     = fp32( rnn_proj.bias + rnn_proj.weight[:, 512:] @ concat(hidden[i + 1], f_rnn) )   # [768]
state    = zeros[256] (bf16);   prev = a                      # reset for every proposal
for i in 0..6:
    raw   = C[i] + fp32( rnn_proj.weight[:, :512] @ concat(state, markov_w1.weight[prev]) )
    gate, cand, outp = bf16(raw[0:256]), bf16(raw[256:512]), bf16(raw[512:768])
    g     = bf16( 1 / (1 + exp(-gate)) )
    c     = bf16( tanh(cand) )
    state = bf16( bf16(g * state) + bf16( bf16(1 - g) * c ) )
    lat   = bf16( tanh(outp) )                               # [256]
    bias  = bf16( unembed.weight @ (hidden_out.weight @ lat + hidden_out.bias) )   # [V]
    logit = bf16( 30 * tanh( bf16(base[i] + bias) / 30 ) )[:200058]
    d[i+1] = argmax(logit)                       if draft_sample_method == "greedy"
           = sample from softmax(logit / T)      if "probabilistic" (and keep `logit` for verify)
    prev  = d[i+1]
```

The reference pre-folds `unembed @ hidden_out` into a `[V, 256]` bf16 matrix and a `[V]`
bf16 bias computed in fp64 (`dspark.py:347-361`); `rnn_proj` input column order is
`[state 256 | w1(prev) 256 | hidden 2304 | f_rnn 2304]` (`dspark.py:363-385`). Draft token
`d[i+1]` is the token at position `L + 1 + i`. `markov_head_type = rnn_hidden`,
`markov_rank = 256`. The block is `anchor + 7 masks`, each mask predicting the token at
its own position (`i2/configs.py:250-256`; `sample_from_anchor = False`), hence
`num_speculative_tokens = 7` is mandatory (`dspark.py:467-472`).

### 7.5 Verify pass and acceptance

The target runs **one ordinary forward with `T = 8`** on `[a, d1, …, d7]` at positions
`L .. L+7`. Nothing special: K/V of all 8 tokens are convolved (in-batch taps come from the
batch, older taps from the conv state), written, and each query attends causally. It
yields 8 logit rows; row `k` is the target distribution for position `L + k + 1`.

Target logits are first processed by the normal sampler (temperature, then min-p,
top-k/top-p, penalties: `IMG: sample/sampler.py`); call the result `p_k`.

**Greedy** (`temperature == 0`; `IMG: rejection_sampler_utils.py:619-646, 757-764`):

```
n = 0
for k in 0..6:
    t = argmax(p_k)
    if t == d[k+1]: n += 1                # accept
    else:           emit t; break         # the target's own token replaces the draft
if n == 7: emit argmax(p_7)               # bonus token
```

**Sampled, `rejection_sample_method = "standard"`** (Leviathan et al.;
`rejection_sampler_utils.py:683-734, 906-980`):

```
for k in 0..6:
    u ~ Uniform(0, 1]     (deterministic in (seed, position))
    logp = log_softmax(p_k)[d[k+1]]
    logq = log_softmax(draft_logit_k / T)[d[k+1]]     if draft_sample_method == "probabilistic"
         = 0                                          if "greedy" (one-hot draft, q = 1)
    accept iff  logp > ln(u) + logq                   # i.e. u < p/q
    if rejected:
        resample position L+k+1 from the residual  r(x) ∝ max(p(x) − q(x), 0)
            (probabilistic)  or from p with d[k+1] masked out (one-hot draft)
        break
if all 7 accepted: sample the bonus token from p_7
```

With `"draft_sample_method": "probabilistic"` (the user's setting) the draft logits are
cached **pre-temperature** (bf16, `[7, V0]` per request) and divided by the request
temperature inside the test (`rejection_sampler_utils.py:176-190`). Note the asymmetry:
`p` has top-k/top-p/penalties applied, `q` only temperature. Draft sampling noise uses a
separate stream from the target's (`IMG: sample/gumbel.py:15-19`). Output of a step is
`n + 1` tokens (`n` accepted drafts + 1 target token), `n ∈ 0..7`.

### 7.6 Rollback

`num_rejected = 7 − n` (`IMG: v1/worker/gpu/input_batch.py:516-542`). The new anchor is
the emitted target token, at position `L' = L + n + 1`. State touched by the rejected
positions `L+n+1 .. L+7`:

| state | what happens |
|---|---|
| Target KV caches (48) | Not rolled back explicitly: sequence length is reduced, and the next verify overwrites those positions. Needs position-indexed storage; local rings need ≥ 1031 slots. |
| Target conv states (48 × 4 streams) | Same: slot written before read, lag taps read only positions ≤ `L+n`. |
| Block stack | Per-token scratch, nothing to undo. |
| Draft context K/V | Written only for the valid positions `L .. L+n` (rejected rows get a pad slot; `IMG: dflash/speculator.py:846-848, 876-901`). |
| Draft block K/V | Rewritten every proposal at `L' .. L'+7`. |
| `f_rnn` | Recomputed from the features of the last valid position `L+n` (`i2/speculator.py:49-58`). |
| Markov state | Starts from zeros each proposal. |
| Draft logits cache | Overwritten each proposal. |

### 7.7 In-tree vs PR 69096 DSpark

`reference/pr69096-dspark.py` differs from `i2/nvidia/dspark.py` only in `_write_kv`
(lines 194-203): on SM120 with a bf16 cache it scatters K/V with its own Triton kernel
`write_kv_paged` (`reference/pr69096-ops-dspark.py:11-106`) instead of FlashAttention's
`do_kv_cache_update`. Same values, same slots — a compatibility fix for RTX 50 cards, with
no change to the mathematics. Everything else in both files is identical.

---

## 8. Chat template, special tokens, sampling defaults

Tokenizer: byte-level BPE (`tokenizer.json`: 199,998 base tokens, 424,413 merges, no
normaliser, o200k-style split regex, no BOS/EOS post-processing). Special tokens:

| token | id | | token | id |
|---|---|---|---|---|
| `<|endoftext|>` (pad) | 199999 | | `<|content_thinking|>` | 200008 |
| `<|message_user|>` | 200000 | | `<|end_message|>` | 200010 |
| `<|message_model|>` | 200001 | | `<|content_tool_error|>` | 200022 |
| `<|message_system|>` | 200002 | | `<|content_xml|>` | 200024 |
| `<|message_tool|>` | 200003 | | `<|begin_of_text|>` | 200028 |
| `<|content_text|>` | 200004 | | `<|content_invoke_tool_json|>` | 200049 |
| `<|content_image|>` | 200005 | | `<|unused_200054|>` (image placeholder) | 200054 |
| `<|content_model_end_sampling|>` (**EOS**) | 200006 | | `<|content_invoke_tool_text|>` | 200057 |

Added tokens occupy ids 199998..200057; `unpadded_vocab_size = 200058`; rows
200058..201023 of embed/unembed are padding and are never sampled.

Text-only chat rendering (`chat_template.jinja`):

```
[<|message_system|><|content_text|>{system text}<|end_message|>]      # leading system messages, if any
<|message_system|><|content_text|>Thinking effort level: {E}<|end_message|>
<|message_user|><|content_text|>{user text}<|end_message|>
... (assistant turns in history: [<|message_model|><|content_thinking|>{reasoning}<|end_message|>]
     <|message_model|><|content_text|>{content}<|end_message|><|content_model_end_sampling|>)
```

* The effort line is inserted once, immediately before the first non-system message (or
  at the very end if there are only system messages). `E` = `reasoning_effort` mapped
  `none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh 0.99, max 0.99`; default
  **0.9**; printed as `0` for 0.0, else the number as Jinja prints it (`0.9`).
* The template emits **no generation prompt and no `<|begin_of_text|>`**. Generation starts
  right after the last `<|end_message|>`; the model itself emits
  `<|message_model|><|content_thinking|>…<|end_message|><|message_model|><|content_text|>…<|end_message|><|content_model_end_sampling|>`.
* Stop on id 200006. Output parsing (`tokenizer_config.json` `response_template`): reasoning
  = spans opened by `(<|message_model|>[^<]*)?<|content_thinking|>`, content = spans opened
  by `…<|content_(text|tool_error|invoke_tool_text)|>`, tool calls = `…<|content_invoke_tool_json|>`
  with JSON `{"name":…,"args":{…}}`; all closed by `<|end_message|>`.

Sampling defaults (`generation_config.json`): `do_sample: true`, `temperature: 1.0`,
`eos_token_id: 200006`, `pad_token_id: 199999`. No top-p / top-k / repetition penalty.
The user's serving script used `{"method":"dspark","num_speculative_tokens":7,
"draft_sample_method":"probabilistic"}` and `--kv-cache-dtype fp8`.

---

## 9. Numerics traps

### 9.1 dtype of every intermediate

| quantity | dtype | note |
|---|---|---|
| embedding row, `stack[*]` blocks | bf16 | blocks are rounded to bf16 after every layer |
| any AttnRes mix before its norm | fp32 | including `partial + attn_out` |
| sublayer inputs `x_q, x_k, x_v, mlp_in` | bf16 | rounded once, after RMSNorm gain |
| loop `prelude_dests`, messages, pass-1/2 mixes | fp32 | not normalised until `blend` |
| controller `pair`, `h` | bf16 | `gain`, `forget` are fp32 |
| q, r, gate, sink, k_raw, v_raw, dk, dv | bf16 | bf16 GEMM, fp32 accumulate |
| `tau` | fp32 | applied to a bf16-rounded value, result rounded again |
| conv accumulation | fp32 | inputs and output rounded to bf16 |
| relative bias | bf16 | fp32 dot, rounded |
| cached K, V | bf16 or E4M3 | |
| attention scores, softmax, LSE | fp32 | `P` is rounded to bf16 before `P·V` |
| attention output before `wo_ud` | bf16 | gate and sink applied in fp32 first |
| router logits, selection, weights | fp32 | |
| expert GEMM outputs `gu`, `down` | bf16 | fp32 accumulate, fp32 row scale |
| SwiGLU | fp32 → bf16 | then NVFP4/MXFP8 quantised |
| MoE combine | fp32 sum → bf16 | |
| final hidden `out` | bf16 | |
| target logits | **fp32** | bf16 operands, fp32 accumulate and output (`head_dtype float32`, `i2/configs.py:219-220`) |
| draft base logits, Markov bias, draft logits | **bf16** | "unlike the target's fp32 head" (`i2/speculator.py:38-40`) |

### 9.2 eps values

All `1e-6`: every RMSNorm (`rms_norm_eps`), AttnRes source RMS, AttnRes output norm,
controller pair norm, q/k/r L2-clip denominators (`threshold / (norm + 1e-6)`), V norm,
drafter norms. RMSNorm form everywhere: `x * rsqrt(mean(x^2) + eps) * w` (eps inside the
root). L2 clip form: eps added to the norm, outside the root.

### 9.3 Order-of-operation traps

1. AttnRes queries and loop message queries must be scaled by 1/48 (once, at load).
2. AttnRes logits divide the dot product by the source RMS, and the mixed values are
   RMS-normalised too; a second RMSNorm with a learned gain follows.
3. First layer of a block: the old partial becomes a completed source for the MoE mix, and
   the new block starts from `attn_out + conv(moe)` with no carry.
4. Residual update order is `(partial + attn_out) + bf16(conv)`; the conv result is
   rounded to bf16 before the add.
5. Static conv tap index 0 is the **oldest** token; dynamic conv lag index 0 is the
   **current** token. Static scale `5^-0.5`, dynamic scale `1/3`.
6. Conv state stores the conv *inputs* (pre-conv), not outputs; for K/V that is the raw
   projection, not what goes into the KV cache.
7. K has no learned scale and is clipped *after* the conv; q has a learned scale and is
   clipped right after projection. V is RMS-normalised without gain.
8. Global layers: clip → round to bf16 → × tau → round. Local layers never apply tau.
9. Relative bias is added unscaled to `q·k / sqrt(128)`. Global layers get zero bias
   (not −inf) beyond distance 1023; local layers mask there.
10. The token's own K/V must be in the cache before its attention.
11. Sink and gate: `o · sigmoid(gate) · sigmoid(lse − sink)`, `lse` in natural log.
12. Router: selection by `sigmoid + bias`, weights by renormalised `sigmoid` only, × 8.
13. Expert gate/up rows are interleaved (even = gate, odd = up). `sigmoid` takes the
    uncapped gate; the product order is `((cap(g) · sigmoid(g)) · cap(u)) · gamma`; `gamma`
    is applied before the down projection and before quantisation.
14. NVFP4 activation scale choice (6-based vs 4-based per block, by absolute error) changes
    results; a plain "amax/6" quantiser does not reproduce the reference.
15. Logits: truncate to 200058 first, then `30·tanh(x/30)` in fp32. Draft: softcap applied
    to the bf16 sum `base + bias`, result bf16.
16. Drafter: per-head q/k norm is rounded to bf16 before RoPE; residual norm uses the
    unrounded fp32 sum; `fuse_alpha` and the Markov gate arithmetic round to bf16 at every
    step.

### 9.4 Approximate device math

The reference uses PTX approximations in several kernels: `rsqrt.approx` for all RMS and
L2 norms (`i2/nvidia/ops/norm.py:14-24`), `ex2.approx`/`rcp.approx` sigmoids for the
attention gate/sink (`ops/attention.py:252-269`) and the expert SwiGLU, and `tanh.approx`
for the softcaps (`ops/activation.py:19-58`). Bit-exact agreement with vLLM therefore
needs the same instructions; otherwise expect last-bit differences that occasionally flip
a bf16 rounding or an expert selection.

### 9.5 Decode (M = 1) vs prefill

The mathematics is identical; only kernel dispatch and therefore summation order differ:

* bf16 linears: ≤ 16 rows on SM120 use a Triton kernel accumulating 256-wide K chunks
  (`i2/nvidia/ops/gemv.py:10-54, 92-113`); larger batches use cuBLAS (`ops/linear.py:20-39`).
* Router: ≤ 16 tokens use one fused route+quantise launch; logits are always the in-order
  sum of 1024-wide K-group partials (`ops/moe.py:565-566, 932-976`, `ops/linear.py:42-98`).
* AttnRes: ≤ 256 (token, query) pairs use a register-resident row kernel
  (`ops/attn_res.py:298-305`).
* Attention: small batches (≤ 64/3 = 21 query tokens per request) use the split-KV decode
  kernel with the gate/sink fused or applied in a combine kernel; larger ones produce
  fp32 `out` + LSE and a separate gate/sink kernel (`i2/nvidia/attention.py:478-525`,
  `ops/attention.py:375-403`). Split count changes the order of the online-softmax merge.
* Conv: taps inside the current batch are read from the batch (rounded to bf16 first),
  older ones from the state; M = 1 reads all three lags from the state. Same values.
* Expert GEMMs: rows are grouped per expert in tiles; tile sizes depend on batch size
  (`ops/trtllm_bmm.py:61-63`), which does not change per-row results.

---

## 10. UNSURE / could not determine

1. **Value of `w13/w2_weight_outer_scale_denominator`** — it is a 4-byte payload tensor,
   and payloads were not read. The dequantisation formula uses the tensor, so the engine
   just reads it; the common ModelOpt value would be 2688 (= 6 × 448) but that is a guess.
2. **Gate/up interleaving of `w13`** is inferred from the SM120 code path (kernel reads
   interleaved pairs; loader does not permute rows on SM120). No comment in the reference
   states the checkpoint layout outright. Confidence high; verify once against vLLM
   logits, since the alternative (two contiguous halves) gives a silently wrong model.
3. **NVFP4 weight nibble order** (even column in the low nibble) is inferred from the
   activation packer feeding the same native MMA operand format, not from a statement
   about the weight file. Confidence high; same verification applies.
4. **MXFP8 `weight_block_shape [32, 32]`** in `hf_quant_config.json`: the stored scale
   tensors are per row per 32 columns, which is all the reference reads. Whether scales
   are additionally constant across 32 rows is not determinable from headers and does not
   matter for dequantisation.
5. **Controller down-projection rounding**: the batched path uses `torch.bmm` (cuBLAS,
   unspecified accumulation order) while the per-stream reference in the same file uses
   1024-wide K-group fp32 accumulation; both round `h` to bf16. Bit-level behaviour of the
   production path is not determinable from source.
6. **`fuse_alpha` dtype**: the buffer is created `empty_like(fuse_weight)`; the parameter
   is bf16 after loading, so alpha is most likely the fp32 softmax rounded to bf16. Not
   confirmed at run time.
7. **flashinfer `rmsnorm` / `fused_add_rmsnorm`** internals (drafter norms) are outside
   the reference tree; the "norm from the unrounded sum" behaviour is taken from the
   comment at `i2/nvidia/dspark.py:66`.
8. **Exact RNG** of draft sampling, acceptance `u`, and residual resampling (Triton
   `tl.rand` Philox + Gumbel-max keyed by seed and position) was not transcribed. An
   engine that draws its own uniform numbers is distributionally equivalent but will not
   reproduce vLLM's sampled outputs token for token.
9. **Sampler processing order beyond temperature → min-p → top-k/top-p** (penalties,
   logit bias) was only skimmed in `IMG: sample/sampler.py`.
10. **BOS**: neither the chat template nor the tokenizer post-processor adds
    `<|begin_of_text|>`; whether the OpenAI server layer injects one for chat requests was
    not traced (vLLM's default for chat is not to add special tokens).
11. **Summation order of the 8 expert outputs** in the combine is "slot order", which is
    the descending order from `tl.topk` for ≤ 128 tokens and the k-max-pass order for
    larger batches; both are descending by selection key, assumed identical.
12. **Vision path** (image tokens, vision tower) was not specified; text-only flow never
    touches it.
