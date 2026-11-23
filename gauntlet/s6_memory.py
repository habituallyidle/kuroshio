"""Stage 6 - memory behaviour: GPU memory at idle, growth with a long conversation, release when a short one
replaces it, uneven conversations on 4 slots, and --reserve."""
import threading

from core import chat, gpu_proc_mib, gpu_total_mib, gpu_used_mib, healthy, http, now, pysleep, repro, run_parallel
from data import haystack, needle_question, req, sentences
from s3_longctx import ANSWER_MAX_TOKENS, tokens_per_sentence
from s5_perf import FAIL_FRAC, WARN_FRAC, claim

KEY, TITLE = "memory", "Memory behaviour"

MIB_PER_GIB = 1024
# README: "It starts at about 20 GiB of GPU memory" / "takes about 20 GiB at start-up (plus 0.4 GiB per slot)".
CLAIM_IDLE_GIB = 20.0
# README: "about 40 KB per token of conversation" (lossless cache).
CLAIM_KB_PER_TOKEN = 40.0
# README: tokens in total: 240K with 1 slot, 205K with 4 slots; "about 450K" with --kv 8 --dense 8.
CLAIM_CTX_1SLOT, CLAIM_CTX_4SLOTS, CLAIM_CTX_8BIT = 240_000, 205_000, 450_000
MEDIUM_TOKENS, LONG_TOKENS = 10_000, 30_000
# After a short conversation takes over the slot, memory must come back down. The server keeps a hysteresis of
# 8,192 tokens (about 320 MiB at 40 KB/token) before trimming, plus allocation granularity: allow 640 MiB above
# the level before the long conversation, and require that at least 60% of the growth is returned.
RELEASE_SLACK_MIB = 640
RELEASE_MIN_SHARE = 0.6
RESERVE_GIB = 6
# nvidia-smi reports in MiB and other programs (the desktop) move by tens of MiB: tolerate 150 MiB under 6 GiB.
RESERVE_TOLERANCE_MIB = 150


def _mem(srv):
    """(GPU MiB of the server process, GPU MiB in use on the whole card)."""
    pysleep(0.4)
    return gpu_proc_mib(srv.pid), gpu_used_mib()


def one_slot(st, ctx):
    srv = ctx.server
    tps = tokens_per_sentence(ctx)
    idle_proc, idle_total = getattr(srv, "gpu_idle_proc", None), getattr(srv, "gpu_idle_total", None)
    if idle_proc is None:
        st.skip("GPU memory at idle", "nvidia-smi does not list the server process")
        return
    st.num("GPU memory of the server right after start (1 slot)", idle_proc, "MiB", head=True)
    st.num("GPU memory in use on the card right after start", idle_total, "MiB")
    claim(st, ctx, "GPU memory at start-up (1 slot)", idle_proc / MIB_PER_GIB, CLAIM_IDLE_GIB, "GiB", higher_is_better=False, how="nvidia-smi per-process memory right after 'listening'")
    cw = ctx.shared.get("context_window") or ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    if cw:
        claim(st, ctx, "context for one conversation (1 slot)", cw / 1000.0, CLAIM_CTX_1SLOT / 1000.0, "K tokens", how="context_window from GET /v1/models; depends on what else uses the card")

    # a short conversation first, so the slot holds as little as it ever does
    c = chat(req("What is the capital of France? Answer in one word.", 300))
    m0, _ = _mem(srv)
    st.num("after a short conversation", m0, "MiB")
    # a medium one, then a long one: the per-token cost is the slope between the two (both are beyond the
    # 8,192-token hysteresis the slot keeps anyway)
    doc_m, nd_m = haystack(int(MEDIUM_TOKENS / tps), seed=2023)
    cm = chat(req(needle_question(doc_m, nd_m[0][0]), ANSWER_MAX_TOKENS), timeout=600)
    mm, _ = _mem(srv)
    doc, nd = haystack(int(LONG_TOKENS / tps), seed=2024)
    p = req(needle_question(doc, nd[1][0]), ANSWER_MAX_TOKENS)
    c = chat(p, timeout=600)
    m1, _ = _mem(srv)
    ok = c.ok and nd[1][1] in c.content and cm.ok and nd_m[0][1] in cm.content
    st.check(f"a {MEDIUM_TOKENS // 1000}K-token and a {LONG_TOKENS // 1000}K-token conversation are answered correctly", ok, c.brief()[:200],
             repro({"note": f"haystacks of about {MEDIUM_TOKENS} and {LONG_TOKENS} tokens + needle question"}, nd[1][1], c.brief()[:200]))
    if c.ok and cm.ok and None not in (m0, mm, m1):
        grown = m1 - m0
        kb = (m1 - mm) * 1024.0 / (c.prompt_tokens - cm.prompt_tokens)
        st.num(f"after a {cm.prompt_tokens}-token conversation", mm, "MiB")
        st.num(f"after a {c.prompt_tokens}-token conversation", m1, "MiB", head=True)
        st.num("memory per token of conversation", round(kb, 1), "KB/token", head=True)
        st.check("GPU memory grows with the conversation", m0 < mm < m1, f"{m0} MiB (short) -> {mm} MiB ({cm.prompt_tokens} tokens) -> {m1} MiB ({c.prompt_tokens} tokens)",
                 repro({"note": "nvidia-smi per-process memory after each of three conversations of growing size"}, "memory grows with the conversation", f"{m0} -> {mm} -> {m1} MiB"))
        claim(st, ctx, "GPU memory per token of conversation", kb, CLAIM_KB_PER_TOKEN, "KB/token", higher_is_better=False,
              how=f"(memory with a {c.prompt_tokens}-token conversation - memory with a {cm.prompt_tokens}-token one) / difference in tokens")
        c2 = chat(req("What is the capital of Japan? Answer in one word.", 300))
        m2, _ = _mem(srv)
        st.num("after a short conversation replaced the long one", m2, "MiB", head=True)
        back = (m1 - m2) / grown if grown > 0 else 0
        st.check("memory falls back when a short conversation replaces the long one",
                 c2.ok and m2 <= m0 + RELEASE_SLACK_MIB and back >= RELEASE_MIN_SHARE,
                 f"{m0} MiB -> {m1} MiB (long) -> {m2} MiB (short again): {back:.0%} of the growth returned",
                 repro({"note": f"1) short request 2) {LONG_TOKENS // 1000}K-token prompt 3) short request; nvidia-smi after each"},
                       f"after step 3 at most {m0 + RELEASE_SLACK_MIB} MiB and >= {RELEASE_MIN_SHARE:.0%} of the growth returned", f"{m2} MiB, {back:.0%} returned"))


def four_slots(st, ctx):
    srv = ctx.server
    tps = tokens_per_sentence(ctx)
    long_tokens, short_tokens = (100_000, 15_000) if ctx.full else (24_000, 3_000)
    cw = ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    st.num("context shared by 4 slots (context_window)", cw, "tokens")
    if cw:
        claim(st, ctx, "context shared by 4 slots", cw / 1000.0, CLAIM_CTX_4SLOTS / 1000.0, "K tokens", how="context_window from GET /v1/models on --slots 4")
    m_before, _ = _mem(srv)
    jobs = []
    for i, size in enumerate([long_tokens, short_tokens, short_tokens, short_tokens]):
        doc, nd = haystack(int(size / tps), seed=3000 + i)
        jobs.append((size, nd[i % 3], req(needle_question(doc, nd[i % 3][0]), ANSWER_MAX_TOKENS)))
    t0 = now()
    res = run_parallel([lambda p=p: chat(p, timeout=900) for _s, _n, p in jobs])
    wall = now() - t0
    m_after, _ = _mem(srv)
    bad = []
    for (size, (colour, code), _p), c in zip(jobs, res):
        if not (getattr(c, "ok", False) and code in c.content):
            bad.append(f"{size}-token conversation: {getattr(c, 'brief', lambda: repr(c))()[:160]}")
    total_tok = sum(c.prompt_tokens for c in res if getattr(c, "ok", False))
    st.check(f"uneven conversations on 4 slots at once (one of {long_tokens // 1000}K, three of {short_tokens // 1000}K tokens) all answer correctly", not bad,
             f"{total_tok} prompt tokens in {wall:.1f}s; " + ("; ".join(bad) or "4/4 needles retrieved"),
             repro({"note": f"4 concurrent needle questions over unique documents of {long_tokens}/{short_tokens}/{short_tokens}/{short_tokens} tokens, --slots 4"}, "4 correct answers", "; ".join(bad)))
    if m_before is not None and m_after is not None:
        st.num("4 slots: GPU memory before / after the uneven conversations", f"{m_before} / {m_after}", "MiB")
    st.num("4 slots: uneven conversations, combined prompt processing", round(total_tok / wall) if wall else None, "tok/s")
    # afterwards four short conversations take the slots over: memory comes back
    res = run_parallel([lambda i=i: chat(req(f"Reply with only the number {i + 51}.", 120), timeout=120) for i in range(4)])
    pysleep(0.5)
    res = run_parallel([lambda i=i: chat(req(f"Reply with only the number {i + 61}.", 120), timeout=120) for i in range(4)])
    m_end, _ = _mem(srv)
    if None not in (m_before, m_after, m_end) and m_after > m_before:
        st.num("4 slots: GPU memory after short conversations took over", m_end, "MiB")
        st.check("4 slots: memory falls back when short conversations take the slots over", m_end <= m_before + RELEASE_SLACK_MIB * 2,
                 f"{m_before} -> {m_after} -> {m_end} MiB", repro({"note": "after the uneven conversations, 2 rounds of 4 concurrent short requests"}, f"at most {m_before + 2 * RELEASE_SLACK_MIB} MiB", f"{m_end} MiB"),
                 warn_only=True)


def reserve(st, ctx):
    srv = ctx.server
    tps = tokens_per_sentence(ctx)
    total = gpu_total_mib()
    need = RESERVE_GIB * MIB_PER_GIB - RESERVE_TOLERANCE_MIB
    used = getattr(srv, "gpu_idle_total", None) or gpu_used_mib()
    free_idle = total - used
    st.num("--reserve 6g: free GPU memory at idle", free_idle, "MiB", head=True)
    st.check(f"--reserve 6g leaves at least {RESERVE_GIB} GiB of the card free at idle", free_idle >= need, f"{free_idle} MiB free of {total} MiB",
             repro(raw=srv.cmdline() + "; nvidia-smi --query-gpu=memory.used", expected=f">= {need} MiB free", actual=f"{free_idle} MiB free"))
    cw = ((http("GET", "/v1/models").json() or {}).get("data") or [{}])[0].get("context_window")
    st.num("--reserve 6g --slots 2: context_window", cw, "tokens")
    if not cw:
        st.fail("--reserve 6g: /v1/models reports a context size", str(cw))
        return
    # Two conversations that together need more than the pool: the second one cannot grow.
    size = int(cw * 0.62)
    min_free = [free_idle]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            u = gpu_used_mib()
            if u is not None:
                min_free[0] = min(min_free[0], total - u)
            stop.wait(0.4)
    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    jobs = []
    for i in range(2):
        doc, nd = haystack(int(size / tps), seed=4000 + i)
        jobs.append((nd[1], req(needle_question(doc, nd[1][0]), ANSWER_MAX_TOKENS)))
    res = run_parallel([lambda p=p: chat(p, timeout=900) for _n, p in jobs], stagger=0.5)
    pysleep(0.6)
    stop.set()
    th.join(timeout=5)
    statuses = sorted(getattr(c, "status", -1) for c in res)
    served = [((colour, code), c) for ((colour, code), _p), c in zip(jobs, res) if getattr(c, "status", 0) == 200]
    refused = [c for c in res if getattr(c, "status", 0) == 503]
    st.num("--reserve 6g: least free GPU memory while two conversations tried to grow past the pool", min_free[0], "MiB")
    st.check("growth beyond the reserve is refused with a 503 and the other conversation is served",
             statuses == [200, 503] and all(code in c.content for (_c, code), c in served) and "memory" in (refused[0].error or "") and srv.alive(),
             f"statuses {statuses}; 503 message: {(refused[0].error or '')[:140] if refused else '-'}",
             repro({"note": f"two concurrent non-streaming prompts of about {size} tokens each (0.62 x context_window {cw}) on --slots 2 --reserve 6g"},
                   "one HTTP 200 with the right answer and one HTTP 503 'out of GPU memory'; server alive",
                   "; ".join(getattr(c, "brief", lambda: repr(c))()[:200] for c in res)))
    st.check(f"the reserve is respected while conversations grow (free memory never below {RESERVE_GIB} GiB)", min_free[0] >= need, f"minimum free {min_free[0]} MiB",
             repro(raw="nvidia-smi sampled every 0.4 s during the two prompts", expected=f">= {need} MiB free at all times", actual=f"{min_free[0]} MiB"))
    c = chat(req("What is the capital of France? Answer in one word.", 300))
    st.check("after the refusal the server keeps serving", c.ok and "paris" in c.content.lower() and healthy(), c.brief()[:160], repro(req("What is the capital of France? Answer in one word.", 300), "Paris", c.brief()))
    if not ctx.full:
        st.check("the server survived the memory-pressure test", srv.alive() and healthy(), f"alive={srv.alive()}")
        return
    # the same refusal for a streaming client arrives after the 200 has been sent
    doc, nd = haystack(int(size / tps), seed=4010)
    hold = threading.Thread(target=lambda: chat(req(needle_question(doc, nd[0][0]), ANSWER_MAX_TOKENS), timeout=900), daemon=True)
    hold.start()
    pysleep(1.5)
    doc2, nd2 = haystack(int(size / tps), seed=4011)
    cs = chat(req(needle_question(doc2, nd2[0][0]), ANSWER_MAX_TOKENS, stream=True), timeout=900)
    hold.join(timeout=900)
    if cs.stream_error:
        st.warn("a streaming request that cannot get memory is told so inside a 200 stream, not with HTTP 503",
                f"HTTP {cs.status}, then an error event: {cs.stream_error[:140]}",
                repro({"note": f"as above, but the second request has stream=true"}, "HTTP 503 (README: 'a request that no longer fits gets a 503')",
                      f"HTTP 200 text/event-stream followed by data: {{\"error\": ...}} and [DONE]; clients that only look at the status see an empty successful reply"))
    elif cs.status == 503:
        st.passed("a streaming request that cannot get memory gets HTTP 503")
    else:
        st.info("streaming request under memory pressure", cs.brief()[:200])
    st.check("the server survived the memory-pressure tests", srv.alive() and healthy(), f"alive={srv.alive()}")


IDLE_RELEASE_S = 5
# After --idle-release seconds without use a slot gives its context memory back: within 250 MiB of the level
# right after start (allocation granularity), checked a few seconds after the deadline.
IDLE_RELEASE_SLACK_MIB = 250


def idle_release(st, ctx):
    srv = ctx.server
    tps = tokens_per_sentence(ctx)
    m_start = getattr(srv, "gpu_idle_proc", None)
    doc, nd = haystack(int(20_000 / tps), seed=2099)
    msgs = [{"role": "user", "content": needle_question(doc, nd[0][0])}]
    c = chat(req(msgs, ANSWER_MAX_TOKENS), timeout=600)
    m_long, _ = _mem(srv)
    if not st.check("--idle-release: a 20K-token conversation is served", c.ok and nd[0][1] in c.content and m_start is not None and m_long is not None, c.brief()[:160]):
        return
    pysleep(IDLE_RELEASE_S + 4)
    m_idle, _ = _mem(srv)
    st.num(f"--idle-release {IDLE_RELEASE_S}: GPU memory at start / with a 20K conversation / after {IDLE_RELEASE_S + 4} s idle", f"{m_start} / {m_long} / {m_idle}", "MiB")
    st.check(f"--idle-release {IDLE_RELEASE_S}: an idle slot gives its context memory back", m_idle <= m_start + IDLE_RELEASE_SLACK_MIB and m_long > m_idle,
             f"{m_start} MiB at start, {m_long} MiB with the conversation, {m_idle} MiB after {IDLE_RELEASE_S + 4} s idle; log: " + "; ".join(l for l in srv.log_text().splitlines() if "idle" in l)[:200],
             repro({"note": f"server --idle-release {IDLE_RELEASE_S}; one 20K-token prompt; wait {IDLE_RELEASE_S + 4} s; nvidia-smi"}, f"at most {m_start + IDLE_RELEASE_SLACK_MIB} MiB", f"{m_idle} MiB"))
    msgs += [{"role": "assistant", "content": c.content}, {"role": "user", "content": f"And what is the access code for the {nd[2][0]} vault? Reply with the six-digit code only."}]
    c2 = chat(req(msgs, ANSWER_MAX_TOKENS), timeout=600)
    st.check("--idle-release: the conversation continues correctly after its memory was released (prompt re-read, nothing cached)", c2.ok and nd[2][1] in c2.content and c2.cached_tokens == 0,
             f"cached {c2.cached_tokens} of {c2.prompt_tokens}; answer {c2.content.strip()[:40]!r}", repro({"note": "follow-up turn after the idle release"}, f"{nd[2][1]} with cached_tokens 0", c2.brief()[:200]))


def claims(st, ctx):
    cw8 = ctx.shared.get("context_window_8bit")
    if cw8:
        claim(st, ctx, "context with --kv 8 --dense 8 (1 slot)", cw8 / 1000.0, CLAIM_CTX_8BIT / 1000.0, "K tokens", how="context_window from GET /v1/models")
    else:
        st.skip("context with --kv 8 --dense 8", "the 8-bit server was not started in this run (select the quality or perf stage)")


def parts(suite):
    extra = [("idle-release", "idle", idle_release)] if suite == "full" else []
    return [("one-slot", "default", one_slot), ("reserve", "reserve", reserve)] + extra + [("four-slots", "slots4", four_slots), ("claims", None, claims)]
