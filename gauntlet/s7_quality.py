"""Stage 7 - quality regression guard: a fixed set of prompts with objectively checkable answers, scored
automatically, at default settings and with the lossy 8-bit modes."""
from core import chat, http, repro
from data import QUALITY, req, score_quality

KEY, TITLE = "quality", "Quality (regression guard)"

# The questions are deliberately easy (see data.QUALITY and the scoring rules there). On the build this suite was
# written against, the default configuration scores 56 of 56 with reasoning_effort "low" at temperature 0.
# Two wrong answers are tolerated silently (96%); below 95% is a WARN; below 85% something is broken in the
# engine or the chat template rather than in the model's knowledge (FAIL).
SCORE_WARN_BELOW, SCORE_FAIL_BELOW = 0.95, 0.85
# 8-bit cache and matrices are documented as lossy; more than 5 points below the default score is a WARN.
EIGHT_BIT_MAX_DROP_POINTS = 5.0
EFFORT = "low"
MAX_TOKENS = 700


def _run(st, ctx, label):
    results = {}
    for qid, prompt, kind, expected in QUALITY:
        p = req(prompt, MAX_TOKENS, effort=EFFORT)
        c = chat(p, timeout=180)
        good = bool(c.ok and c.finish_reason == "stop" and score_quality(kind, expected, c.content))
        results[qid] = {"ok": good, "answer": c.content if c.ok else c.brief()[:200], "expected": expected, "finish": c.finish_reason, "tokens": c.completion_tokens}
    n = sum(r["ok"] for r in results.values())
    score = n / len(results)
    ctx.shared.setdefault("quality", {})[label] = results
    cats = {}
    for qid, r in results.items():
        cat = qid.split("-")[0]
        a, b = cats.get(cat, (0, 0))
        cats[cat] = (a + r["ok"], b + 1)
    st.num(f"{label}: score", f"{n}/{len(results)} ({score:.1%})", head=True)
    st.num(f"{label}: by category", ", ".join(f"{k} {a}/{b}" for k, (a, b) in cats.items()))
    wrong = [(qid, r) for qid, r in results.items() if not r["ok"]]
    for qid, r in wrong:
        st.info(f"{label}: wrong answer to {qid}", f"expected {r['expected']!r}, got {r['answer'][:120]!r} (finish={r['finish']})")
    detail = f"{n}/{len(results)}; wrong: {[q for q, _ in wrong]}"
    prompts = {q[0]: q[1] for q in QUALITY}
    rp = repro(req(prompts[wrong[0][0]], MAX_TOKENS, effort=EFFORT), f"{wrong[0][1]['expected']!r} (scoring rules in gauntlet/README.md)", wrong[0][1]["answer"][:300]) if wrong else None
    if score < SCORE_FAIL_BELOW:
        st.fail(f"{label}: quality score at least {SCORE_WARN_BELOW:.0%}", detail, rp)
    elif score < SCORE_WARN_BELOW:
        st.warn(f"{label}: quality score at least {SCORE_WARN_BELOW:.0%}", detail, rp)
    else:
        st.passed(f"{label}: quality score at least {SCORE_WARN_BELOW:.0%}", detail)
    return score


def default(st, ctx):
    _run(st, ctx, "default (16-bit)")


def eightbit(st, ctx):
    cw = ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    ctx.shared["context_window_8bit"] = cw
    _run(st, ctx, "--kv 8 --dense 8")


def compare(st, ctx):
    q = ctx.shared.get("quality", {})
    a, b = q.get("default (16-bit)"), q.get("--kv 8 --dense 8")
    if not a or not b:
        st.skip("8-bit score against default", "one of the two runs is missing")
        return
    sa, sb = 100.0 * sum(r["ok"] for r in a.values()) / len(a), 100.0 * sum(r["ok"] for r in b.values()) / len(b)
    changed = [qid for qid in a if a[qid]["ok"] != b[qid]["ok"]]
    same_text = sum(1 for qid in a if a[qid]["answer"] == b[qid]["answer"])
    st.num("8-bit minus default", f"{sb - sa:+.1f} points")
    st.num("answers word-for-word identical in both modes", f"{same_text}/{len(a)}")
    st.check(f"--kv 8 --dense 8 scores no more than {EIGHT_BIT_MAX_DROP_POINTS:.0f} points below the default", sa - sb <= EIGHT_BIT_MAX_DROP_POINTS,
             f"default {sa:.1f}%, 8-bit {sb:.1f}% ({sb - sa:+.1f} points); questions whose verdict changed: {changed}",
             repro(None, f"at most {EIGHT_BIT_MAX_DROP_POINTS:.0f} points lower", f"{sb - sa:+.1f} points; changed: {changed}"), warn_only=True)


def parts(suite):
    return [("default", "default", default), ("8bit", "8bit", eightbit), ("compare", None, compare)]
