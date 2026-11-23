"""Slow, readable reference forward pass for Inkling 2 Mini (text only, one token at a time).

Purpose: a correctness oracle for the CUDA engine. It follows docs/ARCHITECTURE.md section by
section and favours clarity over speed. Arithmetic is done in fp32 and rounded to bf16 exactly
where the spec says so.

  python ref/inkling_ref.py check tests/oracle/golden.json     # compare against vLLM
"""
import json, math, os, struct, sys
from collections import OrderedDict

import numpy as np
import torch

DEV = "cuda"
D, H, HKV, DH, DR, E, W = 2304, 24, 8, 128, 16, 1024, 4
NE, TOPK, V0 = 128, 8, 200058
EPS = 1e-6
QK_CLIP, REL_CLIP = 22.627416997969522, 8.0
GLOBAL = {5, 11, 17, 23}
VISITS = [(i, 0) for i in range(6)] + [(i, p) for p in range(3) for i in range(6, 18)] + [(i, 0) for i in range(18, 24)]
TAPS = {5, 23, 41}
F32 = torch.float32


def bf16(x):
    """Round fp32 to bfloat16 (round-to-nearest-even) and widen back."""
    return x.to(torch.bfloat16).to(F32)


# ---------------------------------------------------------------------------- weights
class Checkpoint:
    """mmap a .safetensors file; tensors are copied to the GPU only when asked for."""
    NP = {"BF16": np.uint16, "F32": np.float32, "U8": np.uint8, "F8_E4M3": np.uint8}

    def __init__(self, path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.meta = json.loads(f.read(n))
        self.meta.pop("__metadata__", None)
        self.mm = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)

    def raw(self, name, index=None):
        m = self.meta[name]
        a = np.frombuffer(self.mm, dtype=self.NP[m["dtype"]], count=int(np.prod(m["shape"])), offset=m["data_offsets"][0]).reshape(m["shape"])
        return a if index is None else a[index]

    def f32(self, name):
        """A BF16 or F32 tensor as fp32 on the GPU."""
        a = self.raw(name)
        if self.meta[name]["dtype"] == "BF16":
            return torch.from_numpy(a.view(np.int16).copy()).to(DEV).view(torch.bfloat16).to(F32)
        return torch.from_numpy(a.copy()).to(DEV)


def e4m3_table():
    """Value of each of the 256 FP8 E4M3 codes (sign, 4-bit exponent, 3-bit mantissa, bias 7)."""
    v = np.zeros(256, dtype=np.float64)
    for c in range(256):
        s, e, m = c >> 7, (c >> 3) & 15, c & 7
        mag = (m / 8) * 2.0 ** -6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7)
        v[c] = -mag if s else mag
    v[0x7F] = v[0xFF] = 0.0  # NaN codes; never produced
    return v


E4M3 = torch.tensor(e4m3_table(), dtype=F32, device=DEV)
E4M3_POS = E4M3[:127].clone()  # codes 0..126 are the non-negative values in ascending order (max 448)
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=F32, device=DEV)


def e4m3_round(x):
    """Round non-negative fp32 to the nearest E4M3 value (ties to the even code)."""
    hi = torch.bucketize(x, E4M3_POS).clamp(1, 126)
    lo = hi - 1
    dl, dh = x - E4M3_POS[lo], E4M3_POS[hi] - x
    pick_hi = (dh < dl) | ((dh == dl) & (hi % 2 == 0))
    return torch.where(pick_hi, E4M3_POS[hi], E4M3_POS[lo])


def e2m1_round(v):
    """Round fp32 to the nearest FP4 E2M1 value, saturating at 6 (thresholds from the spec, 6.2)."""
    a = v.abs()
    m = torch.full_like(a, 6.0)
    for limit, val, inclusive in [(5.0, 4.0, True), (3.5, 3.0, False), (2.5, 2.0, True), (1.75, 1.5, False),
                                  (1.25, 1.0, True), (0.75, 0.5, False), (0.25, 0.0, True)]:
        m = torch.where((a <= limit) if inclusive else (a < limit), torch.full_like(a, val), m)
    return torch.sign(v) * m


def quant_nvfp4(x):
    """NVFP4 activation quantisation, rule "fouroversix_mae". x: fp32 [..., n], n % 16 == 0.
    Returns the de-quantised values (what the 4-bit GEMM effectively multiplies by)."""
    shp = x.shape
    amax = x.abs().amax(-1, keepdim=True)
    FMAX = torch.finfo(F32).max
    enc = torch.clamp(1536.0 / amax, max=FMAX)
    dec = 1.0 / enc
    outer = amax * (1.0 / 1536.0)
    xb = x.reshape(*shp[:-1], -1, 16)
    bmax = xb.abs().amax(-1, keepdim=True)
    enc_, dec_, outer_ = enc.unsqueeze(-1), dec.unsqueeze(-1), outer.unsqueeze(-1)
    best = None
    for div in (1.0 / 6.0, 0.25):
        s = e4m3_round(torch.clamp(bmax * (enc_ * div), max=448.0))
        inv = torch.clamp(1.0 / (s * dec_), max=FMAX)
        val = e2m1_round(xb * inv) * s * outer_
        err = (val - xb).abs().sum(-1, keepdim=True)
        if best is None:
            best = (val, err)
        else:
            best = (torch.where(err < best[1], val, best[0]), None)  # 4-based wins only if strictly better
    out = best[0].reshape(shp)
    return torch.where(amax > 0, out, torch.zeros_like(out))


def quant_mxfp8(x):
    """MXFP8 activation quantisation: per block of 32, power-of-two scale rounded up, E4M3 values."""
    shp = x.shape
    xb = x.reshape(*shp[:-1], -1, 32)
    raw = xb.abs().amax(-1, keepdim=True) / 448.0
    exp = torch.ceil(torch.log2(raw.clamp(min=2.0 ** -126)))
    scale = torch.pow(2.0, exp)
    q = torch.clamp(xb / scale, -448.0, 448.0)
    return (torch.sign(q) * e4m3_round(q.abs()) * scale).reshape(shp)


class Model:
    def __init__(self, model_dir, expert_cache=600):
        self.ck = ck = Checkpoint(os.path.join(model_dir, "model-00001-of-00001.safetensors"))
        g = lambda n: ck.f32("model.llm." + n)
        self.embed, self.unembed = g("embed.weight"), g("unembed.weight")
        self.embed_norm, self.norm = g("embed_norm.weight"), g("norm.weight")
        s = D ** -0.5  # all AttnRes and loop queries are pre-scaled by 1/48
        self.attn_q = {m: g(f"attn_res.attn_queries.{m}").reshape(-1, D) * s for m in range(23)}
        self.mlp_q = [g(f"attn_res.mlp_queries.{n}").reshape(1, D) * s for n in range(24)]
        self.out_q = g("attn_res.output_query").reshape(1, D) * s
        a = (g("middle_cycle_cross_pass_queries") * s).reshape(12, 2, D)
        kv = (g("middle_cycle_cross_pass_kv_queries") * s).reshape(12, 2, D)
        self.msg_q = [torch.stack([a[j, 0], kv[j, 0], kv[j, 1], a[j, 1]]) for j in range(12)]  # attn, k, v, mlp
        self.L = []
        for i in range(24):
            p = f"layers.{i}."
            L = {k: g(p + n) for k, n in [("wq", "attn.wq_du.weight"), ("wr", "attn.wr_du.weight"), ("gate", "attn.gate_attn.weight"),
                                          ("sink", "attn.attn_sink_proj.weight"), ("wk", "attn.wk_dv.weight"), ("wv", "attn.wv_dv.weight"),
                                          ("wo", "attn.wo_ud.weight"), ("q_scale", "attn.q_l2_clip.scale"), ("r_scale", "attn.rel_l2_clip.scale"),
                                          ("rel", "attn.rel_logits_proj.proj"), ("mlp_norm", "mlp_norm.weight"), ("router", "mlp.gate.weight"),
                                          ("router_b", "mlp.gate.bias")]}
            L["attn_sconv"] = g(p + "attn_sconv.weight")[:, 0, :]  # [C, 4], tap 0 = oldest
            L["mlp_sconv"] = g(p + "mlp_sconv.weight")[:, 0, :]
            if i in GLOBAL:
                L["in_norm"] = torch.stack([g(p + f"{s_}_input_norm.weight") for s_ in "qkv"])
                for s_ in "kv":
                    L[s_ + "_static"] = g(p + f"attn.{s_}_sconv.static_w")  # [4 lags, 1024], lag 0 = current
                    L[s_ + "_proj"] = g(p + f"attn.{s_}_sconv.weight_proj")  # [32, D]
            else:
                L["in_norm"] = g(p + "attn_norm.weight").reshape(1, D)
                for s_ in "kv":
                    L[s_ + "_conv"] = g(p + f"attn.{s_}_sconv.weight")[:, 0, :]
            if 6 <= i < 18:
                L["ctl"] = [{k: g(p + f"controllers.{s_}.{k}_weight") for k in ("down", "gain", "forget")} for s_ in ("attn", "k", "v", "mlp")]
            L["fp8"] = ck.meta["model.llm." + p + "mlp.experts.w13_weight"]["dtype"] == "F8_E4M3"
            if not L["fp8"]:
                for w in ("w13", "w2"):
                    L[w + "_amax"] = g(p + f"mlp.experts.{w}_weight_amax")
                    L[w + "_den"] = g(p + f"mlp.experts.{w}_weight_outer_scale_denominator")
            self.L.append(L)
        self.cache, self.cache_cap = OrderedDict(), expert_cache
        self.route_log = []

    # ------------------------------------------------------------------ expert weights
    def expert(self, i, e):
        """De-quantised (W13 [2304, 2304], W2 [2304, 1152]) of expert e in layer i, fp32."""
        key = (i, e)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        L, out = self.L[i], []
        for w in ("w13", "w2"):
            name = f"model.llm.layers.{i}.mlp.experts.{w}_weight"
            codes = torch.from_numpy(self.ck.raw(name, e).copy()).to(DEV)
            scale = torch.from_numpy(self.ck.raw(name + "_scale", e).copy()).to(DEV)
            if L["fp8"]:  # MXFP8: one E4M3 byte per weight, power-of-two scale per 32 columns
                wgt = E4M3[codes.long()] * torch.pow(2.0, scale.to(F32) - 127.0).repeat_interleave(32, dim=1)
            else:  # NVFP4: two 4-bit codes per byte (low nibble = even column), E4M3 scale per 16 columns
                c = codes.long()
                nib = torch.stack([c & 15, c >> 4], dim=-1).reshape(c.shape[0], -1)
                wgt = E2M1[nib] * E4M3[scale.long()].repeat_interleave(16, dim=1) * (L[w + "_amax"][e] / L[w + "_den"][0])
            out.append(wgt)
        self.cache[key] = tuple(out)
        if len(self.cache) > self.cache_cap:
            self.cache.popitem(last=False)
        return self.cache[key]

    # ------------------------------------------------------------------ building blocks
    @staticmethod
    def rms(x, w):
        return bf16(x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w)

    @staticmethod
    def mix(sources, queries, norm=None):
        """Block attention residual (spec 4). sources: list of fp32 [D]; queries [Q, D]; -> [Q, D]."""
        X = torch.stack(sources)
        inv = torch.rsqrt(X.pow(2).mean(-1) + EPS)
        logit = (queries @ X.T) * inv
        p = torch.exp(logit - logit.amax(-1, keepdim=True))
        mixed = ((p * inv) @ X) / p.sum(-1, keepdim=True)
        return mixed if norm is None else Model.rms(mixed, norm)

    @staticmethod
    def blend(ctl, dest, message, norm_w):
        """Loop message controller (spec 2.4)."""
        pd = dest * torch.rsqrt(dest.pow(2).mean() + EPS)
        pm = message * torch.rsqrt(message.pow(2).mean() + EPS)
        h = bf16(bf16(torch.cat([pd, pm])) @ ctl["down"].T)
        gain, forget = h @ ctl["gain"].T, h @ ctl["forget"].T
        x = (1 - torch.tanh(forget)) * dest + torch.tanh(gain) * message
        return Model.rms(x, norm_w)

    @staticmethod
    def l2_clip(x, thr):
        n = x.pow(2).sum(-1, keepdim=True).sqrt()
        return x * torch.clamp(thr / (n + EPS), max=1.0)

    @staticmethod
    def sconv_static(hist, w):
        """hist: the last <=4 conv inputs [n, C], newest last. w [C, 4], tap 3 = current token."""
        acc = hist[-1].clone()
        for back in range(min(4, hist.shape[0])):
            acc = acc + w[:, 3 - back] * hist[-1 - back]
        return bf16(acc * 5 ** -0.5)

    @staticmethod
    def sconv_dynamic(hist, static_w, delta):
        """Global-layer K/V conv. static_w [4, 1024] (lag 0 = current), delta [8 heads, 4 lags]."""
        acc = hist[-1].clone()
        for lag in range(min(4, hist.shape[0])):
            acc = acc + (static_w[lag] + delta[:, lag].repeat_interleave(DH)) * hist[-1 - lag]
        return bf16(acc / 3.0)

    def attention(self, st, i, pos, xq, xk, xv, tau):
        L, is_global = self.L[i], i in GLOBAL
        q, r = bf16(L["wq"] @ xq).view(H, DH), bf16(L["wr"] @ xq).view(H, DR)
        gate, sink = bf16(L["gate"] @ xq).view(H, DH), bf16(L["sink"] @ xq)
        st["k_raw"].append(bf16(L["wk"] @ xk)); st["v_raw"].append(bf16(L["wv"] @ xv))
        kh, vh = torch.stack(st["k_raw"][-4:]), torch.stack(st["v_raw"][-4:])
        qc, rc = self.l2_clip(q * L["q_scale"], QK_CLIP), self.l2_clip(r * L["r_scale"], REL_CLIP)
        if is_global:
            qc, rc = bf16(qc) * tau, bf16(rc) * tau
            ck_ = self.sconv_dynamic(kh, L["k_static"], bf16(L["k_proj"] @ xk).view(HKV, W))
            cv_ = self.sconv_dynamic(vh, L["v_static"], bf16(L["v_proj"] @ xv).view(HKV, W))
        else:
            ck_, cv_ = self.sconv_static(kh, L["k_conv"]), self.sconv_static(vh, L["v_conv"])
        qc, rc = bf16(qc), bf16(rc)
        K = bf16(self.l2_clip(ck_.view(HKV, DH), QK_CLIP))
        Vv = cv_.view(HKV, DH)
        Vv = bf16(Vv * torch.rsqrt(Vv.pow(2).mean(-1, keepdim=True) + EPS))
        st["K"].append(K); st["V"].append(Vv)  # the token attends to itself, so write first
        lo = 0 if is_global else max(0, pos - (E - 1))
        Ks, Vs = torch.stack(st["K"][lo:]), torch.stack(st["V"][lo:])  # [S, 8, 128]
        dist = pos - torch.arange(lo, pos + 1, device=DEV)
        bias_tab = bf16(rc @ L["rel"])  # [24, 1024], column = distance
        bias = torch.where(dist < E, bias_tab[:, dist.clamp(max=E - 1)], torch.zeros((), device=DEV))  # [24, S]
        Kq = Ks.repeat_interleave(H // HKV, dim=1)  # query head h uses KV head h // 3
        logit = torch.einsum("hd,shd->hs", qc, Kq) * DH ** -0.5 + bias
        m = logit.amax(-1, keepdim=True)
        P = torch.exp(logit - m)
        l = P.sum(-1, keepdim=True)
        o = torch.einsum("hs,shd->hd", bf16(P), Vs.repeat_interleave(H // HKV, dim=1)) / l
        lse = m + torch.log(l)
        y = bf16(o * torch.sigmoid(gate) * torch.sigmoid(lse - sink.view(H, 1)))
        return bf16(L["wo"] @ y.reshape(-1))

    def moe(self, i, x):
        L = self.L[i]
        logit = L["router"] @ x
        sel = torch.sigmoid(logit) + L["router_b"]
        ids = torch.topk(sel, TOPK).indices
        sg = torch.sigmoid(logit[ids])
        gamma = sg / sg.sum() * 8.0
        self.route_log.append(ids.tolist())
        quant = quant_mxfp8 if L["fp8"] else quant_nvfp4
        xq = quant(x)
        acc = torch.zeros(D, device=DEV)
        for k in range(TOPK):
            w13, w2 = self.expert(i, int(ids[k]))
            gu = bf16(w13 @ xq)
            g, u = gu[0::2], gu[1::2]  # gate and up rows are interleaved
            a = bf16(((4.0 * torch.tanh(g / 4.0) * torch.sigmoid(g)) * (25.0 * torch.tanh(u / 25.0))) * gamma[k])
            acc = acc + bf16(w2 @ quant(a))
        return bf16(acc)

    # ------------------------------------------------------------------ one token
    def new_state(self):
        return [{"K": [], "V": [], "k_raw": [], "v_raw": [], "attn": [], "mlp": []} for _ in VISITS]

    @torch.no_grad()
    def step(self, state, token, pos):
        """Run one token through all 48 visits. Returns (logits [V0] fp32, taps [4, D])."""
        stack = [None] * 5
        stack[0] = self.rms(self.embed[token], self.embed_norm)
        tau = 1.0 + 0.1 * math.log(max((pos + 1) / 8192.0, 1.0))
        nb, inputs, taps = 0, None, []
        prelude_partial = prelude_dests = bank = None
        for visit, (i, p) in enumerate(VISITS):
            L, st, looped, first = self.L[i], state[visit], p > 0, i % 6 == 0
            if i == 6 and p == 0:
                prelude_partial = stack[nb].clone()
            elif i == 6:
                bank = [stack[2].clone(), stack[3].clone()]
                nb = 1
                stack[1] = prelude_partial
                if prelude_dests is None:
                    prelude_dests = self.mix([stack[0], prelude_partial], self.attn_q[5])
                inputs = prelude_dests
            if visit in TAPS:
                taps.append(stack[nb].clone())
            partial = stack[nb]
            if looped:
                msg = self.mix(bank, self.msg_q[i - 6])
                dests = inputs if inputs.shape[0] == 3 else inputs.expand(3, D)
                nw = L["in_norm"] if L["in_norm"].shape[0] == 3 else L["in_norm"].expand(3, D)
                xq, xk, xv = (self.blend(L["ctl"][s], dests[s], msg[s], nw[s]) for s in range(3))
            elif inputs is None:
                xq = xk = xv = self.rms(partial, L["in_norm"][0])
            else:
                xq, xk, xv = (inputs[0], inputs[1], inputs[2]) if inputs.shape[0] == 3 else (inputs[0],) * 3
            st["attn"].append(self.attention(st, i, pos, xq, xk, xv, tau))
            attn_out = self.sconv_static(torch.stack(st["attn"][-4:]), L["attn_sconv"])
            srcs = stack[:nb + 1] + [attn_out] if first else stack[:nb] + [partial + attn_out]
            if looped:
                mlp_in = self.blend(L["ctl"][3], self.mix(srcs, self.mlp_q[i])[0], msg[3], L["mlp_norm"])
            else:
                mlp_in = self.mix(srcs, self.mlp_q[i], L["mlp_norm"])[0]
            st["mlp"].append(self.moe(i, mlp_in))
            y = self.sconv_static(torch.stack(st["mlp"][-4:]), L["mlp_sconv"])
            if first:
                nb += 1
                stack[nb] = bf16(attn_out + y)
            else:
                stack[nb] = bf16((partial + attn_out) + y)
            if getattr(self, "trace", None) is not None:
                self.trace.append(torch.stack([st["attn"][-1], mlp_in, st["mlp"][-1], stack[nb]]))
            srcs = stack[:nb + 1]
            if visit == 47:
                out = self.mix(srcs, self.out_q, self.norm)[0]
            else:
                ni, np_ = VISITS[visit + 1]
                inputs = self.mix(srcs, self.attn_q[ni - 1], None if np_ > 0 else self.L[ni]["in_norm"])
        logits = (self.unembed[:V0] @ out)
        taps.append(out)
        return 30.0 * torch.tanh(logits / 30.0), torch.stack(taps)


def check(golden_path, model_dir, only=None):
    g = json.load(open(golden_path))
    m = Model(model_dir)
    for case in g["cases"]:
        if only and case["name"] != only:
            continue
        state, pos, worst, mism, n = m.new_state(), 0, 0.0, 0, 0
        for t in case["prompt_ids"][:-1]:
            m.step(state, t, pos); pos += 1
        cur = case["prompt_ids"][-1]
        for s in case["steps"]:
            logits, _ = m.step(state, cur, pos); pos += 1
            lp = torch.log_softmax(logits, -1)
            ids = torch.tensor([t for t, _ in s["top"]], device=DEV)
            ref = torch.tensor([v for _, v in s["top"]], device=DEV)
            keep = ref > -20  # far-tail logprobs are numerically meaningless
            worst = max(worst, float((lp[ids] - ref)[keep].abs().max()))
            mism += int(lp.argmax()) != s["token"]; n += 1
            cur = s["token"]  # teacher forcing: always continue with vLLM's token
        print(f"{case['name']:6s} {n} steps: argmax mismatches {mism}, max |dlogprob| over top-20 = {worst:.4f}", flush=True)


def trace_compare(golden_path, model_dir, engine_bin, pos=0):
    """Layer-by-layer comparison with the CUDA engine on the token at position `pos` of the first golden case."""
    g = json.load(open(golden_path)); m = Model(model_dir, expert_cache=120)
    case = g["cases"][0]; toks = case["prompt_ids"] + [s["token"] for s in case["steps"]]
    st = m.new_state()
    for i in range(pos): m.step(st, toks[i], i)
    m.trace = []
    m.step(st, toks[pos], pos)
    ref = torch.stack(m.trace).cpu().numpy()                       # [48, 4, D]
    eng = np.fromfile(engine_bin, dtype=np.float32).reshape(48, 4, D)
    names = ["attn_raw", "mlp_in", "moe_out", "stack"]
    for v in range(48):
        rel = [np.abs(eng[v, k] - ref[v, k]).max() / (np.abs(ref[v, k]).max() + 1e-12) for k in range(4)]
        if v < 8 or v % 8 == 7 or max(rel) > 0.2:
            print(f"visit {v:2d} layer {VISITS[v][0]:2d} pass {VISITS[v][1]}: " + "  ".join(f"{n} {r:.1e}" for n, r in zip(names, rel)))


if __name__ == "__main__":
    md = os.environ["INKLING_WEIGHTS"]
    if sys.argv[1] == "trace":
        trace_compare(sys.argv[2], md, sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 0)
    if sys.argv[1] == "check":
        check(sys.argv[2], md, sys.argv[3] if len(sys.argv) > 3 else None)
