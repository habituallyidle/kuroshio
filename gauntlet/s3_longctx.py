"""Stage 3 - long context: needle retrieval at increasing prompt sizes, prefix-cache reuse on the same document,
and behaviour at and beyond the context limit."""
from core import chat, healthy, http, repro, median
from data import haystack, needle_question, req, sentences

KEY, TITLE = "longctx", "Long context"

SIZES_QUICK = [4_000, 16_000, 48_000]
SIZES_FULL = [4_000, 16_000, 48_000, 100_000]
# Needles are six-digit codes stated in a plain sentence; a healthy model finds all of them. One miss in a run is
# tolerated as a warning (model quality), more than 20% missed means long-context attention is broken.
RETRIEVAL_WARN_BELOW, RETRIEVAL_FAIL_BELOW = 1.0, 0.8
# A second question on the same document must be served from the prefix cache: nearly all prompt tokens cached ...
CACHED_SHARE_MIN = 0.98
# ... and the first token must come at least 10x sooner than for the uncached prompt, and within 1.5 s in absolute
# terms (README: "follow-up turns answer in milliseconds"; restoring the snapshot of a 100K-token document and
# reading the new question is allowed some time).
CACHED_TTFT_MAX_S = 1.5
CACHED_TTFT_MAX_RATIO = 0.1
# Room for the answer: a six-digit code, plus the thinking the model does even at reasoning_effort "none".
ANSWER_MAX_TOKENS = 800
# The boundary walk starts this many tokens below the context limit and adds about 6 tokens per step.
WALK_START_BELOW = 200
# Rejecting an oversized prompt only needs tokenising it: must not take long or touch the GPU.
REJECT_MAX_S = 20.0


def tokens_per_sentence(ctx):
    """Calibrate the filler: tokens per sentence, measured by the server's own usage report."""
    if "tok_per_sentence" not in ctx.shared:
        n = 300
        base = chat(req("Say ok.", 1))
        c = chat(req(" ".join(sentences(n, 99991)) + "\n\nSay ok.", 1))
        if not (c.ok and base.ok):
            raise RuntimeError(f"calibration request failed: {c.brief()}")
        ctx.shared["tok_per_sentence"] = (c.prompt_tokens - base.prompt_tokens) / n
    return ctx.shared["tok_per_sentence"]


def needles(st, ctx):
    tps = tokens_per_sentence(ctx)
    st.info("filler calibration", f"{tps:.2f} tokens per sentence")
    sizes = SIZES_FULL if ctx.full else SIZES_QUICK
    hits = total = 0
    for si, size in enumerate(sizes):
        n_sent = int(size / tps)
        doc, nd = haystack(n_sent, seed=1000 + si)
        order = nd[si % 3:] + nd[:si % 3]  # rotate which position is asked first (uncached)
        pos_name = {nd[0][0]: "start", nd[1][0]: "middle", nd[2][0]: "end"}
        first = None
        label = f"{size // 1000}K"
        how = {"note": f"haystack({n_sent} sentences, seed {1000 + si}) from gauntlet/data.py, then one question per turn in the same conversation"}
        msgs = []
        for qi, (colour, code) in enumerate(order):
            # the first question comes with the document; the others are follow-up turns of the same conversation
            if qi == 0:
                msgs.append({"role": "user", "content": needle_question(doc, colour)})
            else:
                msgs.append({"role": "user", "content": f"And what is the access code for the {colour} vault? Reply with the six-digit code only."})
            p = req(list(msgs), ANSWER_MAX_TOKENS, stream=True)
            c = chat(p, timeout=900)
            total += 1
            if not c.ok:
                st.fail(f"{label}: question {qi + 1} (needle at {pos_name[colour]}) failed", c.brief()[:300], repro(how, "HTTP 200", c.brief()))
                break
            msgs.append({"role": "assistant", "content": c.content})
            hit = code in c.content
            hits += hit
            rate = (c.prompt_tokens - (c.cached_tokens or 0)) / c.ttft if c.ttft else None
            if qi == 0:
                first = c
                st.num(f"{label}: prompt tokens", c.prompt_tokens, "tokens")
                st.num(f"{label}: time to first token, uncached", round(c.ttft, 2), "s")
                st.num(f"{label}: prompt processing", round(rate) if rate else None, "tok/s", head=True)
                st.check(f"{label}: first question is processed uncached (unique document)", (c.cached_tokens or 0) < 0.05 * c.prompt_tokens,
                         f"cached {c.cached_tokens} of {c.prompt_tokens}", repro(how, "cached_tokens near 0 for a new document", str(c.usage)))
            else:
                share = (c.cached_tokens or 0) / c.prompt_tokens
                lim = max(0.05, min(CACHED_TTFT_MAX_S, CACHED_TTFT_MAX_RATIO * (first.ttft if first and first.ttft else CACHED_TTFT_MAX_S)))
                if qi == 1:
                    st.num(f"{label}: time to first token, cached follow-up", round(c.ttft * 1000), "ms", head=(si == len(sizes) - 1))
                st.check(f"{label}: follow-up question {qi + 1} is answered from the cache",
                         share >= CACHED_SHARE_MIN and c.ttft <= lim,
                         f"cached {c.cached_tokens}/{c.prompt_tokens} ({share:.1%}), TTFT {c.ttft * 1000:.0f} ms (limit {lim * 1000:.0f} ms)",
                         repro(how, f"cached share >= {CACHED_SHARE_MIN:.0%} and TTFT <= {lim * 1000:.0f} ms", f"cached {c.cached_tokens}/{c.prompt_tokens}, TTFT {c.ttft * 1000:.0f} ms"))
            st.check(f"{label}: needle at {pos_name[colour]} retrieved", hit, f"expected {code}, answer {c.content.strip()[:60]!r} (finish={c.finish_reason}, {c.completion_tokens} tokens, {len(c.reasoning)} characters of thinking)",
                     repro(how, code, c.content.strip()[:200]), warn_only=True)
        if si == 0 and first is not None:
            # The same document with a different question as a NEW conversation (document QA, "edit my last message",
            # regenerate with a changed prompt): the prompts share everything but their last few tokens.
            other = order[1]
            p = req(needle_question(doc, other[0]), ANSWER_MAX_TOKENS, stream=True)
            c = chat(p, timeout=900)
            share = (c.cached_tokens or 0) / c.prompt_tokens if c.ok else 0
            st.num(f"{label}: cached share when only the end of the prompt changes", f"{share:.0%}")
            st.check(f"{label}: a new conversation that shares the document but ends with a different question reuses the cached document",
                     c.ok and share >= 0.9,
                     f"cached {c.cached_tokens}/{c.prompt_tokens}, TTFT {c.ttft * 1000:.0f} ms (the first question took {first.ttft * 1000:.0f} ms)" if c.ok else c.brief()[:200],
                     repro({"note": f"request A: one user message '<{label} document> ... access code for the {order[0][0]} vault?'; request B: the same document with '... for the {other[0]} vault?'"},
                           "request B reuses the shared prefix (cached_tokens close to the document size)",
                           f"cached_tokens {c.cached_tokens} of {c.prompt_tokens}: the whole document is processed again",
                           note="the prefix cache is all-or-nothing: it only helps when the new prompt contains the ENTIRE previous prompt (a follow-up turn); a prompt that diverges before the "
                                "previous prompt's last token is re-read from token 0 (server.rs resume_point)"),
                     warn_only=True)
    acc = hits / total if total else 0
    st.num("needle retrieval accuracy", f"{hits}/{total}", head=True)
    if acc < RETRIEVAL_FAIL_BELOW:
        st.fail("retrieval accuracy", f"{hits}/{total} ({acc:.0%}) is below {RETRIEVAL_FAIL_BELOW:.0%}")
    elif acc < RETRIEVAL_WARN_BELOW:
        st.warn("retrieval accuracy", f"{hits}/{total} ({acc:.0%})")
    else:
        st.passed("retrieval accuracy", f"{hits}/{total}")


def limits(st, ctx):
    tps = tokens_per_sentence(ctx)
    cw = ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    if not st.check("context limit is reported by /v1/models", isinstance(cw, int), str(cw)):
        return
    doc = " ".join(sentences(int(cw * 1.06 / tps), 31337)) + "\n\nSay ok."
    for stream in (False, True):
        p = req(doc, 16, stream=stream)
        c = chat(p, timeout=300)
        msg = (c.error or "")
        st.check(f"a prompt longer than the context limit ({cw} tokens) is refused with a clear 400" + (" (stream=true)" if stream else ""),
                 c.status == 400 and "token" in msg and str(cw) in msg and c.wall < REJECT_MAX_S and healthy(),
                 f"HTTP {c.status} in {c.wall:.1f}s: {msg[:160]}",
                 repro({"note": f"one user message of about {int(cw * 1.06)} tokens ({len(doc)} characters of filler)", "stream": stream},
                       "HTTP 400 naming the prompt size and the limit", f"HTTP {c.status}: {msg[:300]} ({c.wall:.1f}s)"))
    st.num("time to reject an oversized prompt", round(c.wall, 2), "s")
    for mt in (10 ** 9, cw, 2 ** 53):
        p = req("What is the capital of France? Answer in one word.", mt)
        c = chat(p)
        st.check(f"max_tokens={mt} (more than the room left) is clamped, not an error", c.ok and c.finish_reason == "stop" and "paris" in c.content.lower(), c.brief()[:200],
                 repro(p, "HTTP 200, normal answer", c.brief()))


def boundary(st, ctx):
    """With --ctx 4096 the edge of the context is cheap to reach: walk prompt sizes across it, and let generations
    run into it."""
    tps = tokens_per_sentence(ctx)
    cw = ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    if not st.check("--ctx 4096: /v1/models reports a context of about 4096 tokens", isinstance(cw, int) and 4096 <= cw <= 4096 + 1024, str(cw),
                    repro(raw="GET /v1/models on a server started with --ctx 4096", expected="context_window 4096 (or slightly above)", actual=str(cw))):
        return
    st.num("--ctx 4096: context_window reported", cw, "tokens")
    essay = "\n\nNow write a very long essay about the history of bridges. Do not stop early."
    # a generation that runs into the end of the context (no max_tokens: the default is all the room there is)
    doc = " ".join(sentences(int((cw - 500) / tps), 776))
    p = req(doc + essay, None)
    del p["max_tokens"]
    c = chat(p, timeout=300)
    st.check("a generation without max_tokens stops with finish_reason 'length' at the context limit",
             c.ok and c.finish_reason == "length" and cw - 64 <= c.usage["total_tokens"] <= cw,
             f"finish={c.finish_reason}, prompt {c.prompt_tokens} + completion {c.completion_tokens} = {c.usage and c.usage.get('total_tokens')} of {cw}",
             repro({"note": f"a {cw - 500}-token document + 'write a very long essay', no max_tokens, server --ctx 4096"}, f"finish_reason=length with total_tokens just under {cw}", c.brief()[:300]))
    # Conversations of increasing size walked across the limit, a few tokens at a time. Each step appends a turn,
    # so the previous prompt is cached and a step costs only its generation.
    msgs = [{"role": "user", "content": " ".join(sentences(int((cw - WALK_START_BELOW) / tps), 777)) + essay}]
    seen, bad, first_refused, last_ok, at_edge = [], [], None, None, 0
    for k in range(120):
        p = req(list(msgs), 100_000, stream=(k % 2 == 0))
        c = chat(p, timeout=120)
        if c.status == 200 and c.error is None:
            tot = c.usage["total_tokens"] if c.usage else None
            seen.append((c.prompt_tokens, c.completion_tokens, c.finish_reason))
            last_ok = c.prompt_tokens
            if first_refused is not None:
                bad.append(f"step {k}: accepted ({c.prompt_tokens} prompt tokens) after a shorter prompt was refused")
            if tot is None or tot > cw or not c.completion_tokens or c.finish_reason not in ("length", "stop"):
                bad.append(f"step {k}: prompt {c.prompt_tokens}, completion {c.completion_tokens}, total {tot}, finish {c.finish_reason}")
            if c.finish_reason == "length" and tot and tot >= cw - 64:
                at_edge += 1
        elif c.status == 400 and "token" in (c.error or ""):
            if first_refused is None:
                first_refused = k
        else:
            bad.append(f"step {k}: {c.brief()[:160]}")
        if first_refused is not None and k >= first_refused + 3:
            break
        msgs.append({"role": "assistant", "content": "ok"})
        msgs.append({"role": "user", "content": "Go on" + " x" * (k % 7)})
    ok = not bad and first_refused is not None and last_ok is not None and at_edge >= 3 and healthy()
    st.check("conversation sizes walked across the context limit: accepted ones are clamped and end with 'length', then a clean 400, nothing else",
             ok, f"{len(seen)} accepted ({at_edge} generated right up to the limit), largest accepted prompt {last_ok} tokens, first refusal at step {first_refused}, context {cw}; " + "; ".join(bad[:4]),
             repro({"note": f"a {cw - WALK_START_BELOW}-token document + 'write a very long essay'; then repeatedly append an assistant turn 'ok' and a user turn 'Go on x x ..' (0-6 x), max_tokens 100000 (server --ctx 4096)"},
                   "HTTP 200 with total_tokens <= context while the prompt fits, HTTP 400 once it does not; server healthy", "; ".join(bad[:6]) or f"refusal seen: {first_refused is not None}; reached the edge {at_edge} times; healthy={healthy()}"))
    if last_ok:
        st.num("--ctx 4096: largest prompt accepted", last_ok, "tokens")
    c = chat(req("What is the capital of France? Answer in one word.", 300))
    st.check("after the boundary walk the server still answers correctly", c.ok and "paris" in c.content.lower(), c.brief()[:160], repro(req("What is the capital of France? Answer in one word.", 300), "Paris", c.brief()))


def parts(suite):
    return [("needles", "default", needles), ("limits", "default", limits), ("boundary", "ctx4k", boundary)]
