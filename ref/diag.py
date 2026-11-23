import json, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import inkling_ref as R
g = json.load(open("tests/oracle/golden.json")); m = R.Model(os.environ["INKLING_WEIGHTS"])
for case in g["cases"][:int(sys.argv[1]) if len(sys.argv) > 1 else 1]:
    st, pos = m.new_state(), 0
    for t in case["prompt_ids"][:-1]: m.step(st, t, pos); pos += 1
    cur, rows = case["prompt_ids"][-1], []
    for s in case["steps"]:
        lg, _ = m.step(st, cur, pos); pos += 1
        lp = torch.log_softmax(lg, -1)
        top = [(t, v) for t, v in s["top"] if v > -12]
        d = [float(lp[t]) - v for t, v in top]
        rows.append((max(abs(x) for x in d), s["logprob"], float(lp[s["token"]]), int(lp.argmax()) == s["token"], len(top)))
        cur = s["token"]
    print(case["name"], "median maxdiff %.4f" % sorted(r[0] for r in rows)[len(rows)//2])
    print(" ".join(f"{r[0]:.2f}{'' if r[3] else '!'}" for r in rows))
    print("ref  chosen lp:", " ".join(f"{r[1]:.2f}" for r in rows[:24]))
    print("ours chosen lp:", " ".join(f"{r[2]:.2f}" for r in rows[:24]))
