"""Stage 5 - throughput and latency, measured from the client, compared with the figures in README.md.

How things are measured (all with stream=true; the throughput requests set reasoning_effort "high" explicitly, as
the README's benchmarks do: left unset or at "none" every engine falls back to a default of its own):
  decode tok/s   = (usage.completion_tokens - 1) / (arrival of the last chunk - arrival of the first chunk that
                   carries text). The first token is excluded: its latency is prompt processing.
  TTFT           = arrival of the first text chunk - moment the request was sent.
  prompt tok/s   = usage.prompt_tokens / time until the single generated token's chunk arrives (max_tokens 1, a
                   document the server has never seen, so nothing is cached).
  aggregate      = sum of completion_tokens of N simultaneous requests / wall time from the first send to the last
                   final chunk (so it includes prompt processing and any queueing, as a user would see it).
This is the method of bench.py (inference-optimisation skill), which produced the README's numbers; the prompt
panel is the same.
"""
import json

from core import FAIL, PASS, WARN, chat, fmt, http, median, now, repro, run_parallel
from data import PERF_PROMPTS, T, haystack, req, sentences, user

KEY, TITLE = "perf", "Throughput and latency"

# A measurement more than 15% worse than the README's figure is a WARN, more than 35% worse a FAIL (task spec:
# single runs vary by about +-5% with the text, so 15% is outside noise; 35% means the claim is simply not met).
WARN_FRAC, FAIL_FRAC = 0.15, 0.35

# Figures claimed in README.md ("Headline" table and the bullets under it).
CLAIM_DECODE_1 = 250.0        # "Decode, 1 stream (tok/s)", temperature 0.6, 512-token completions, effort high
CLAIM_DECODE_2_TOTAL = 419.0  # "Decode, 2 streams, total"
CLAIM_DECODE_4_TOTAL = 559.0  # "Decode, 4 streams, total"
CLAIM_PREFILL_4K = 8440.0     # "Prompt processing (tok/s, 4.7K prompt)"
CLAIM_SPEC0 = 161.0           # "Without speculative decoding: 161 tok/s (--spec 0)"
CLAIM_8BIT_DECODE = 265.0     # "With --kv 8 --dense 8: ... about 265 tok/s"
CLAIM_8BIT_PREFILL = 6580.0   # "... 6,600 tok/s prompt processing"
CLAIM_FOLLOWUP_TTFT_MS = 50.0  # "typically 30-50 ms to first token in an agent loop": the upper end is the claim checked

PERF_EFFORT = "high"          # reasoning effort of every throughput request (the README states it)
GEN_TOKENS = 512              # as in the README's benchmark; every run must reach at least MIN_TOKENS
MIN_TOKENS = 300
SEEDS_QUICK, SEEDS_FULL = [11, 12], [11, 12, 13]
# Time to first token for a short prompt. No README figure; the engine's own log shows 15-30 ms. Anything above
# 100 ms would be noticeable in interactive use (WARN), above 500 ms something is broken (FAIL).
TTFT_SHORT_WARN_MS, TTFT_SHORT_FAIL_MS = 100.0, 500.0
# Fairness between simultaneous streams = slowest / fastest decode rate. Streams legitimately differ (speed
# depends on how predictable each text is; the README's own 4-stream run has 122 vs 190 tok/s = 0.64), so only a
# ratio below 0.4 is flagged: that would mean one conversation is being starved.
FAIRNESS_WARN_BELOW = 0.4
# 8 clients on 4 slots: queueing must not cost throughput. Compared with the 4-stream aggregate measured in the
# same run; below 70% of it is a WARN.
QUEUED_THROUGHPUT_WARN_BELOW = 0.7


def claim(st, ctx, name, measured, claimed, unit, higher_is_better=True, how=""):
    status = st.threshold(f"README claim: {name}", measured, claimed, WARN_FRAC, FAIL_FRAC, unit, higher_is_better, how)
    delta = "n/a" if measured is None else f"{(measured - claimed) / claimed * 100:+.1f}%"
    ctx.shared.setdefault("claims", []).append({"name": name, "claim": f"{claimed:g} {unit}", "measured": "n/a" if measured is None else f"{measured:.4g} {unit}",
                                                "delta": delta, "status": status, "how": how})
    return status


def warm_up(ctx):
    chat(req("Say hello.", 200, stream=True))
    chat(req(PERF_PROMPTS[5], 200, temperature=0.6, seed=1, stream=True, effort=PERF_EFFORT))
    chat(req(" ".join(sentences(120, 424242)) + "\n\nSay ok.", 8, stream=True))


def decode_runs(st, ctx, label, temperature, prompts, seeds):
    """Returns the list of runs: dicts with prompt index, seed, tok/s, tokens, text."""
    runs = []
    for pi in prompts:
        for seed in seeds:
            p = req(PERF_PROMPTS[pi], GEN_TOKENS, temperature=temperature, seed=seed, stream=True, effort=PERF_EFFORT)
            c = chat(p, timeout=300)
            if not (c.ok and c.decode_tps):
                st.fail(f"{label}: benchmark request failed", c.brief()[:300], repro(p, "HTTP 200 stream", c.brief()))
                continue
            runs.append({"prompt": pi, "seed": seed, "tps": c.decode_tps, "tokens": c.completion_tokens, "text": c.text(), "ttft": c.ttft})
    short = [r for r in runs if r["tokens"] < MIN_TOKENS]
    if short:
        st.warn(f"{label}: some runs generated fewer than {MIN_TOKENS} tokens (their rate is less reliable)", str([(r["prompt"], r["seed"], r["tokens"]) for r in short]))
    return runs


def prefill_rate(st, ctx, label, tokens, reps, seed0):
    from s3_longctx import tokens_per_sentence
    tps = tokens_per_sentence(ctx)
    rates, n_tok = [], None
    for r in range(reps):
        doc = " ".join(sentences(int(tokens / tps), seed0 + r)) + "\n\nSay ok."
        p = req(doc, 1, stream=True)
        c = chat(p, timeout=600)
        t = (c.t_first or c.t_last)
        if not (c.ok and t):
            st.fail(f"{label}: prompt-processing request failed", c.brief()[:300])
            continue
        if c.cached_tokens:
            st.warn(f"{label}: prompt was partly cached, rate not comparable", str(c.usage))
        n_tok = c.prompt_tokens
        rates.append(c.prompt_tokens / (t - c.t_start))
    return median(rates), n_tok, rates


def single(st, ctx):
    warm_up(ctx)
    seeds = SEEDS_FULL if ctx.full else SEEDS_QUICK
    panel = [0, 1, 2]  # the README's three prompts
    extra = [3, 4, 5] if ctx.full else []

    # ---- TTFT, short prompt
    tt = []
    for i in range(9):
        c = chat(req(f"Reply with the single word: ready ({i})", 8, stream=True))
        if c.ok and (c.t_first or c.t_last):
            tt.append(((c.t_first or c.t_last) - c.t_start) * 1000)
    ms = median(tt)
    st.num("time to first token, short prompt (median of 9)", round(ms, 1) if ms else None, "ms", head=True)
    if ms is None or ms > TTFT_SHORT_FAIL_MS:
        st.fail("time to first token for a short prompt", f"{ms} ms (FAIL above {TTFT_SHORT_FAIL_MS:.0f} ms)")
    elif ms > TTFT_SHORT_WARN_MS:
        st.warn("time to first token for a short prompt", f"{ms:.0f} ms (WARN above {TTFT_SHORT_WARN_MS:.0f} ms)")
    else:
        st.passed("time to first token for a short prompt", f"{ms:.0f} ms, range {min(tt):.0f}-{max(tt):.0f} ms")

    # ---- decode, one stream
    by_temp = {}
    for temp, sds in ((0.6, seeds), (0.0, seeds[:1]), (1.0, seeds[:1])):
        runs = decode_runs(st, ctx, f"1 stream, temperature {temp}", temp, panel + extra, sds)
        by_temp[temp] = runs
        vals = [r["tps"] for r in runs]
        vals_panel = [r["tps"] for r in runs if r["prompt"] in panel]
        st.num(f"decode, 1 stream, temperature {temp}: median (min-max) over {len(vals)} runs", f"{fmt(median(vals))} ({fmt(min(vals)) if vals else 'n/a'}-{fmt(max(vals)) if vals else 'n/a'})", "tok/s")
        if temp == 0.6:
            st.num("1 stream at temperature 0.6 (README panel)", round(median(vals_panel), 1) if vals_panel else None, "tok/s", head=True)
        if temp == 0.6:
            ctx.shared["perf_default_06"] = runs
            claim(st, ctx, "decode, 1 stream, temperature 0.6", median(vals_panel), CLAIM_DECODE_1, "tok/s",
                  how=f"median of {len(vals_panel)} runs: README prompt panel (3 prompts) x seeds {sds}, {GEN_TOKENS} tokens each")
            if extra:
                st.num("decode, 1 stream, temperature 0.6: median over all six panel prompts", round(median(vals), 1), "tok/s")
            per_prompt = {pi: round(median([r["tps"] for r in runs if r["prompt"] == pi]), 1) for pi in panel + extra}
            st.num("decode, 1 stream, temperature 0.6: median per prompt", json.dumps(per_prompt), "tok/s")

    # ---- prompt processing
    r4, n4, all4 = prefill_rate(st, ctx, "4K", 4000, 3, 880000)
    st.num(f"prompt processing, {n4}-token prompt (median of {len(all4)})", round(r4) if r4 else None, "tok/s", head=True)
    claim(st, ctx, "prompt processing, 4K prompt", r4, CLAIM_PREFILL_4K, "tok/s", how=f"{n4} uncached prompt tokens / time to the first chunk, median of {len(all4)}")
    r16, n16, all16 = prefill_rate(st, ctx, "16K", 16000, 3 if ctx.full else 2, 881000)
    st.num(f"prompt processing, {n16}-token prompt (median of {len(all16)})", round(r16) if r16 else None, "tok/s")
    st.info("prompt processing at 16K (no README figure)", f"{r16 and round(r16)} tok/s = {((r16 or 0) / r4 * 100 if r4 else 0):.0f}% of the 4K rate")

    # ---- follow-up turns of an agent loop: only the new part of the prompt is read
    tools = [T["bash"], T["read_file"], T["edit_file"]]
    msgs = [{"role": "system", "content": "You are a coding agent working in a repository. Use the bash tool for every step; run exactly one command per step. " + " ".join(sentences(60, 55))},
            {"role": "user", "content": "Explore the repository step by step: list the files, then show the README, then show the tests, then run them. One bash call per step."}]
    canned = ["README.md\nsrc\ntests\nsetup.py", "# Demo project\nA small library with two modules.", "tests/test_core.py\ntests/test_io.py", "collected 12 items\n12 passed in 0.31s",
              "src/core.py\nsrc/io.py", "def add(a, b):\n    return a + b", "nothing to commit, working tree clean", "ok"]
    ttfts, cached_share = [], []
    for step in range(8):
        p = req(list(msgs), 300, tools=tools, stream=True)
        c = chat(p)
        if not c.ok:
            break
        if step > 0:
            ttfts.append((c.t_first - c.t_start) * 1000 if c.t_first else None)
            cached_share.append((c.cached_tokens or 0) / c.prompt_tokens)
        if not c.tool_calls:
            break
        msgs.append({"role": "assistant", "content": c.content or None, "tool_calls": c.tool_calls[:1]})
        msgs.append({"role": "tool", "tool_call_id": c.tool_calls[0]["id"], "content": canned[step]})
    ttfts = [t for t in ttfts if t is not None]
    if len(ttfts) >= 3:
        st.num("agent loop: follow-up time to first token, median (min-max)", f"{median(ttfts):.0f} ({min(ttfts):.0f}-{max(ttfts):.0f})", "ms")
        st.num("agent loop: share of prompt tokens served from the cache", f"{100 * median(cached_share):.0f}", "%")
        claim(st, ctx, "follow-up turn, time to first token in an agent loop (upper end of 30-50 ms)", median(ttfts), CLAIM_FOLLOWUP_TTFT_MS, "ms", higher_is_better=False,
              how=f"median over {len(ttfts)} follow-up requests of a synthetic tool loop (about 1.3K-token system prompt, one bash call and a short tool result per step)")
    else:
        st.warn("agent loop: the model stopped calling tools before 3 follow-ups could be timed", f"{len(ttfts)} follow-ups")


def spec0(st, ctx):
    warm_up(ctx)
    seeds = SEEDS_FULL if ctx.full else SEEDS_QUICK
    runs = decode_runs(st, ctx, "--spec 0, temperature 0.6", 0.6, [0, 1, 2], seeds)
    vals = [r["tps"] for r in runs]
    st.num(f"decode, 1 stream, --spec 0, temperature 0.6: median (min-max) over {len(vals)} runs", f"{fmt(median(vals))} ({fmt(min(vals)) if vals else 'n/a'}-{fmt(max(vals)) if vals else 'n/a'})", "tok/s")
    st.num("--spec 0", round(median(vals), 1) if vals else None, "tok/s", head=True)
    claim(st, ctx, "decode without speculation (--spec 0), temperature 0.6", median(vals), CLAIM_SPEC0, "tok/s", how=f"median of {len(vals)} runs, same prompts and seeds as the 1-stream figure")
    base = {(r["prompt"], r["seed"]): r for r in ctx.shared.get("perf_default_06", [])}
    ratios, same = [], 0
    for r in runs:
        b = base.get((r["prompt"], r["seed"]))
        if b:
            ratios.append(b["tps"] / r["tps"])
            same += b["text"] == r["text"]
    if ratios:
        st.num("speed-up from speculative decoding (same prompt, same seed)", f"x{median(ratios):.2f} (min x{min(ratios):.2f}, max x{max(ratios):.2f})")
        st.check("seeded sampling produced the same text with and without speculation in every benchmark run", same == len(ratios), f"{same}/{len(ratios)} identical 512-token texts",
                 repro(req(PERF_PROMPTS[0], GEN_TOKENS, temperature=0.6, seed=seeds[0], effort=PERF_EFFORT), "identical text on servers started with and without --spec 0", f"{len(ratios) - same} of {len(ratios)} differ"))


def eightbit(st, ctx):
    warm_up(ctx)
    seeds = SEEDS_FULL if ctx.full else SEEDS_QUICK
    runs = decode_runs(st, ctx, "--kv 8 --dense 8, temperature 0.6", 0.6, [0, 1, 2], seeds)
    vals = [r["tps"] for r in runs]
    st.num(f"decode, 1 stream, --kv 8 --dense 8, temperature 0.6: median (min-max) over {len(vals)} runs", f"{fmt(median(vals))} ({fmt(min(vals)) if vals else 'n/a'}-{fmt(max(vals)) if vals else 'n/a'})", "tok/s")
    claim(st, ctx, "decode with --kv 8 --dense 8, temperature 0.6", median(vals), CLAIM_8BIT_DECODE, "tok/s", how=f"median of {len(vals)} runs, README prompt panel")
    r4, n4, all4 = prefill_rate(st, ctx, "4K (8-bit)", 4000, 3, 882000)
    st.num(f"prompt processing with --kv 8 --dense 8, {n4}-token prompt", round(r4) if r4 else None, "tok/s")
    claim(st, ctx, "prompt processing with --kv 8 --dense 8", r4, CLAIM_8BIT_PREFILL, "tok/s", how=f"{n4} uncached prompt tokens, median of {len(all4)}")


def _concurrent(n, seed0, tokens=GEN_TOKENS, clients=None):
    clients = clients or n
    res = run_parallel([lambda i=i: chat(req(PERF_PROMPTS[i % len(PERF_PROMPTS)], tokens, temperature=0.6, seed=seed0 + i, stream=True, effort=PERF_EFFORT), timeout=600) for i in range(clients)])
    good = [c for c in res if getattr(c, "ok", False) and c.completion_tokens]
    if len(good) != clients:
        return None, res
    wall = max(c.t_end for c in good) - min(c.t_start for c in good)
    return {"aggregate": sum(c.completion_tokens for c in good) / wall, "per_stream": [c.decode_tps for c in good],
            "ttft": [c.ttft for c in good], "wall": wall}, res


def multi(st, ctx):
    warm_up(ctx)
    reps = 3 if ctx.full else 2
    # one stream on the 4-slot server, as the reference for scaling
    runs = decode_runs(st, ctx, "4 slots, 1 stream", 0.6, [0, 1, 2], [SEEDS_QUICK[0]])
    one = median([r["tps"] for r in runs])
    st.num("decode, 1 stream on the 4-slot server", round(one, 1) if one else None, "tok/s")
    agg = {}
    for n, claimed in ((2, CLAIM_DECODE_2_TOTAL), (4, CLAIM_DECODE_4_TOTAL)):
        results = []
        for r in range(reps):
            m, raw = _concurrent(n, 300 + 10 * r)
            if m is None:
                st.fail(f"{n} simultaneous streams: a request failed", "; ".join(getattr(c, "brief", lambda: repr(c))()[:120] for c in raw))
            else:
                results.append(m)
        if not results:
            continue
        a = median([m["aggregate"] for m in results])
        fair = [min(m["per_stream"]) / max(m["per_stream"]) for m in results]
        agg[n] = a
        st.num(f"{n} streams: aggregate (median of {len(results)})", round(a, 1), "tok/s", head=True)
        st.num(f"{n} streams: per-stream decode rates (last run)", "/".join(f"{x:.0f}" for x in results[-1]["per_stream"]), "tok/s")
        st.num(f"{n} streams: fairness, slowest/fastest (worst run)", round(min(fair), 2))
        claim(st, ctx, f"decode, {n} streams, total", a, claimed, "tok/s", how=f"total completion tokens / wall time, {n} requests x {GEN_TOKENS} tokens on --slots 4, median of {len(results)} rounds")
        st.check(f"{n} streams: no stream is starved (slowest/fastest >= {FAIRNESS_WARN_BELOW})", min(fair) >= FAIRNESS_WARN_BELOW, f"ratios per round: {[round(f, 2) for f in fair]}", warn_only=True)
    # 8 clients on 4 slots
    m, raw = _concurrent(4, 500, tokens=384, clients=8)
    if m is None:
        st.fail("8 clients on 4 slots: a request failed", "; ".join(getattr(c, "brief", lambda: repr(c))()[:120] for c in raw))
    else:
        st.num("8 clients on 4 slots: total throughput", round(m["aggregate"], 1), "tok/s")
        st.num("8 clients on 4 slots: worst wait for a first token", round(max(m["ttft"]), 2), "s")
        st.num("8 clients on 4 slots: all done after", round(m["wall"], 1), "s")
        if agg.get(4):
            st.check("8 clients on 4 slots: queueing does not cost throughput", m["aggregate"] >= QUEUED_THROUGHPUT_WARN_BELOW * agg[4],
                     f"{m['aggregate']:.0f} tok/s against {agg[4]:.0f} tok/s for 4 unqueued streams", warn_only=True)
    cw = ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    ctx.shared["context_window_4slots"] = cw


def parts(suite):
    return [("single", "default", single), ("spec0", "spec0", spec0), ("8bit", "8bit", eightbit), ("multi", "slots4", multi)]
