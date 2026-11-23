"""What would 8-bit (MXFP8) attention weights / output head cost in quality?

Uses the slow reference model. For each variant, teacher-forces the 4 golden prompts and reports the mean
log-probability the model assigns to vLLM's greedy tokens and how often its own argmax agrees with them.
'control' removes one harmless rounding step: it shows how much these numbers move from numerical noise alone.
  python ref/fp8_dense_exp.py tests/oracle/golden.json
"""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import inkling_ref as R

ATT = ["wq", "wr", "gate", "sink", "wk", "wv", "wo"]

def pad32(w):  # quantise rows in blocks of 32 columns (pad the last block)
    n = w.shape[-1]; p = (-n) % 32
    q = R.quant_mxfp8(torch.nn.functional.pad(w, (0, p)))
    return q[..., :n]

def build(variant):
    m = R.Model(os.environ["INKLING_WEIGHTS"])
    if "attn" in variant:
        for L in m.L:
            for k in ATT: L[k] = pad32(L[k])
        orig = m.attention
        m.attention = lambda st, i, pos, xq, xk, xv, tau: orig(st, i, pos, pad32(xq), pad32(xk), pad32(xv), tau)   # 8-bit inputs too
    if "head" in variant:
        m.unembed = pad32(m.unembed)
    return m

def run(m, cases, head_act=False):
    nll, hit, n = 0.0, 0, 0
    for case in cases:
        st, pos = m.new_state(), 0
        for t in case["prompt_ids"][:-1]: m.step(st, t, pos); pos += 1
        cur = case["prompt_ids"][-1]
        for s in case["steps"]:
            lg, _ = m.step(st, cur, pos); pos += 1
            lp = torch.log_softmax(lg, -1)
            nll -= float(lp[s["token"]]); hit += int(lp.argmax()) == s["token"]; n += 1; cur = s["token"]
    return nll / n, hit, n

cases = json.load(open(sys.argv[1]))["cases"]
variants = sys.argv[2:] or ["base", "control", "head", "attn", "attn+head"]
for v in variants:
    if v == "control":
        saved = R.Model.rms
        R.Model.rms = staticmethod(lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + R.EPS) * w)
        m = build("base")
    else:
        m = build(v)
    nll, hit, n = run(m, cases)
    if v == "control": R.Model.rms = saved
    print(f"{v:10s} mean NLL of vLLM's greedy tokens {nll:.4f}   argmax agreement {hit}/{n}", flush=True)
    del m; torch.cuda.empty_cache()
