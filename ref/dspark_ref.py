"""Reference DSpark drafter (docs/ARCHITECTURE.md section 7), in plain PyTorch.

Input: a token sequence and the target model's four hidden-state taps per position, both dumped by
`inkling dumpfeats`. For every anchor position L it proposes 7 tokens the way the drafter would and
counts how many the target would accept under greedy decoding (the target's own greedy tokens are
simply the next tokens of the dumped sequence).

  python ref/dspark_ref.py <prefix> [first_anchor]
"""
import json, math, os, sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from inkling_ref import Checkpoint, bf16, DEV, D, V0, EPS

H, HKV, DH, BLOCK, WIN = 24, 8, 128, 8, 1024
F32 = torch.float32


def rms(x, w):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * w


def rope(x, pos, theta=1e6):
    """x [..., heads, 128] at integer positions pos [...]; 'neox' layout: halves are rotated against each other."""
    m = torch.arange(DH // 2, device=DEV, dtype=F32)
    ang = pos.to(F32)[..., None, None] * theta ** (-2 * m / DH)
    x1, x2 = x[..., : DH // 2], x[..., DH // 2:]
    return bf16(torch.cat([x1 * ang.cos() - x2 * ang.sin(), x2 * ang.cos() + x1 * ang.sin()], -1))


class Drafter:
    def __init__(self, model_dir):
        mt = Checkpoint(os.path.join(model_dir, "mtp.safetensors"))
        main = Checkpoint(os.path.join(model_dir, "model-00001-of-00001.safetensors"))
        g = mt.f32
        self.embed, self.unembed = main.f32("model.llm.embed.weight"), main.f32("model.llm.unembed.weight")[:V0]
        self.embed_norm = main.f32("model.llm.embed_norm.weight")
        self.alpha = bf16(torch.softmax(g("fuse_weight"), -1))      # [4 consumers, 4 taps]
        self.mask, self.norm = g("mask_embed"), g("norm.weight")
        self.L = []
        for i in range(3):
            p = f"layers.{i}."
            self.L.append({k: g(p + n) for k, n in [
                ("in_norm", "input_layernorm.weight"), ("post_norm", "post_attention_layernorm.weight"), ("ctx_norm", "dflare_norm.weight"),
                ("q", "self_attn.q_proj.weight"), ("k", "self_attn.k_proj.weight"), ("v", "self_attn.v_proj.weight"),
                ("kc", "self_attn.k_proj_ctx.weight"), ("vc", "self_attn.v_proj_ctx.weight"), ("o", "self_attn.o_proj.weight"),
                ("qn", "self_attn.q_norm.weight"), ("kn", "self_attn.k_norm.weight"), ("kcn", "self_attn.k_norm_ctx.weight"),
                ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight")]})
        self.rnn_w, self.rnn_b = g("markov_head.rnn_proj.weight"), g("markov_head.rnn_proj.bias")
        self.w1 = g("markov_head.markov_w1.weight")
        self.ho_w, self.ho_b = g("markov_head.hidden_out.weight"), g("markov_head.hidden_out.bias")
        self.rnn_norm = g("markov_head.dflare_norm.weight")
        self.local = [True, False, True]                            # layers 0 and 2 use a sliding window

    @torch.no_grad()
    def context(self, feats):
        """feats [P, 4, D] -> per-layer context keys/values [P, 8, 128] and the Markov feature f_rnn [P, D]."""
        P = feats.shape[0]
        pos = torch.arange(P, device=DEV)
        mixed = torch.einsum("rt,ptd->prd", self.alpha, feats)      # [P, 4, D]
        self.ctx = []
        for i, L in enumerate(self.L):
            c = bf16(rms(mixed[:, i], L["ctx_norm"]))
            k = bf16(c @ L["kc"].T).view(P, HKV, DH)
            v = bf16(c @ L["vc"].T).view(P, HKV, DH)
            self.ctx.append((rope(bf16(rms(k, L["kcn"])), pos), v))
        self.f_rnn = bf16(rms(mixed[:, 3], self.rnn_norm))

    @torch.no_grad()
    def propose(self, anchor, Lpos):
        """Anchor token at position Lpos (context = positions < Lpos). Returns 7 greedy draft tokens."""
        x = torch.cat([bf16(rms(self.embed[anchor], self.embed_norm))[None], self.mask[None].expand(BLOCK - 1, D)])
        pos = Lpos + torch.arange(BLOCK, device=DEV)
        resid, h = x, bf16(rms(x, self.L[0]["in_norm"]))
        for i, L in enumerate(self.L):
            q = rope(bf16(rms(bf16(h @ L["q"].T).view(BLOCK, H, DH), L["qn"])), pos)
            k = rope(bf16(rms(bf16(h @ L["k"].T).view(BLOCK, HKV, DH), L["kn"])), pos)
            v = bf16(h @ L["v"].T).view(BLOCK, HKV, DH)
            lo = max(0, Lpos + BLOCK - 1 - (WIN + BLOCK - 1)) if self.local[i] else 0
            K = torch.cat([self.ctx[i][0][lo:Lpos], k]).repeat_interleave(H // HKV, 1)   # [S, 24, 128]
            Vv = torch.cat([self.ctx[i][1][lo:Lpos], v]).repeat_interleave(H // HKV, 1)
            att = torch.softmax(torch.einsum("bhd,shd->bhs", q, K) * DH ** -0.5, -1)       # every slot sees the whole block
            o = bf16(torch.einsum("bhs,shd->bhd", att, Vv)).reshape(BLOCK, H * DH)
            s = resid + bf16(o @ L["o"].T)
            resid, h = bf16(s), bf16(rms(s, L["post_norm"]))
            a = bf16(torch.nn.functional.silu(bf16(h @ L["gate"].T)) * bf16(h @ L["up"].T))
            s = resid + bf16(a @ L["down"].T)
            resid, h = bf16(s), bf16(rms(s, self.L[i + 1]["in_norm"] if i < 2 else self.norm))
        hidden = h[1:]                                               # the 7 mask slots
        base = bf16(hidden @ self.unembed.T)
        f = self.f_rnn[Lpos - 1].expand(BLOCK - 1, D)
        C = self.rnn_b + torch.cat([hidden, f], -1) @ self.rnn_w[:, 512:].T          # [7, 768]
        state, prev, out = torch.zeros(256, device=DEV), anchor, []
        for i in range(BLOCK - 1):
            raw = C[i] + torch.cat([state, self.w1[prev]]) @ self.rnn_w[:, :512].T
            gate, cand, outp = bf16(raw[:256]), bf16(raw[256:512]), bf16(raw[512:])
            g = bf16(torch.sigmoid(gate))
            state = bf16(bf16(g * state) + bf16(bf16(1 - g) * bf16(torch.tanh(cand))))
            lat = bf16(torch.tanh(outp))
            bias = bf16(self.unembed @ (self.ho_w @ lat + self.ho_b))
            logit = bf16(30 * torch.tanh(bf16(base[i] + bias) / 30))
            prev = int(logit.argmax())
            out.append(prev)
        return out, hidden


def main(prefix, first=None, dump=None):
    toks = json.load(open(prefix + ".tokens.json"))
    feats = torch.from_numpy(np.fromfile(prefix + ".feats.f32", dtype=np.float32).reshape(len(toks), 4, D)).to(DEV)
    d = Drafter(os.environ["INKLING_WEIGHTS"])
    d.context(feats)
    first = first or 30
    hist, emitted, steps, L = [0] * 8, 0, 0, first
    per_anchor = []
    for a in range(first, len(toks) - 8):                           # acceptance at every possible anchor
        drafts, _ = d.propose(toks[a], a)
        n = 0
        while n < 7 and drafts[n] == toks[a + 1 + n]:
            n += 1
        per_anchor.append(n)
    while L < len(toks) - 8:                                         # the walk a real decoder would take
        n = per_anchor[L - first]
        hist[n] += 1; emitted += n + 1; steps += 1; L += n + 1
    print(f"{os.path.basename(prefix)}: mean accepted drafts over all anchors {np.mean(per_anchor):.2f}; "
          f"decode walk: {steps} steps, {emitted / steps:.2f} tokens/step; accepted-count histogram {hist}")
    if dump:                                                         # oracle for the CUDA drafter
        drafts, hidden = d.propose(toks[first], first)
        np.save(dump, {"anchor_pos": first, "drafts": drafts, "hidden": hidden.cpu().numpy(), "per_anchor": per_anchor}, allow_pickle=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None, sys.argv[3] if len(sys.argv) > 3 else None)
