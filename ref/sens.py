"""How chaotic is the model? Compare our reference with a copy that differs only in harmless rounding."""
import json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import inkling_ref as R
g = json.load(open("tests/oracle/golden.json")); m = R.Model(os.environ["INKLING_WEIGHTS"])
case = g["cases"][0]
def run(n=30):
    st, pos, out = m.new_state(), 0, []
    for t in case["prompt_ids"][:-1]: m.step(st, t, pos); pos += 1
    cur = case["prompt_ids"][-1]
    for s in case["steps"][:n]:
        lg, _ = m.step(st, cur, pos); pos += 1
        out.append(torch.log_softmax(lg, -1)); cur = s["token"]
    return out
base = run()
orig = R.Model.rms
R.Model.rms = staticmethod(lambda x, w: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + R.EPS) * w)  # skip one bf16 rounding
pert = run()
d = []
for s, a, b in zip(case["steps"], base, pert):
    ids = torch.tensor([t for t, v in s["top"] if v > -12], device="cuda")
    d.append(float((a[ids] - b[ids]).abs().max()))
print("max |dlogprob| per step, base vs rounding-perturbed:", " ".join(f"{x:.2f}" for x in d))
