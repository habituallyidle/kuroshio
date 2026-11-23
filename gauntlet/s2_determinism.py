"""Stage 2 - determinism and correctness: repeatability, invariance to speculation / prefix cache / concurrency,
seeded sampling, multi-turn caching, Unicode, tool calling."""
import json
import threading
import unicodedata

from core import chat, repro, run_parallel, now, pysleep
from data import (DETERMINISM_PROMPTS, LONG_OUTPUT_PROMPT, MULTI_TURN, PERF_PROMPTS, T, TOOL_CASES, UNICODE_CASES, req,
                  sentences, tool_case_request, user)

KEY, TITLE = "determinism", "Determinism and correctness"

# Tokens generated per determinism prompt. Long enough that a single flipped token is very likely to show
# (the model is numerically chaotic: one different token changes everything after it).
GEN_TOKENS = 200
SEED_A, SEED_B = 7, 8
SEEDED_TEMPERATURE = 0.8
# A follow-up turn must reuse at least the previous prompt (the server snapshots the state at the end of each
# prompt, one token earlier for prompts of 512+ tokens), so allow 2 tokens of slack.
CACHE_SLACK_TOKENS = 2
# README: "Well-formed tool calls: 24 / 24". Below 95% of the cases producing a valid call the claim is in doubt
# (WARN); below 85% tool use is unreliable (FAIL). A call that is emitted but malformed always fails.
TOOL_VALID_WARN, TOOL_VALID_FAIL = 0.95, 0.85
# Argument understanding (right tool, right values) is model quality, not engine correctness: WARN only.
TOOL_ARGS_WARN = 0.85


def first_diff(a, b):
    n = min(len(a), len(b))
    i = next((k for k in range(n) if a[k] != b[k]), n)
    return i, f"first difference at character {i} of {len(a)}/{len(b)}: {a[max(0, i - 30):i + 40]!r} vs {b[max(0, i - 30):i + 40]!r}"


def _greedy_set(ctx):
    ps = [(f"greedy-{i}", req(p, GEN_TOKENS)) for i, p in enumerate(DETERMINISM_PROMPTS)]
    return ps if ctx.full else ps[:2]


def _long_prompt():
    doc = " ".join(sentences(45, 4242))  # roughly 900 tokens: long enough for the repeat-prompt snapshot path (512+)
    return req(doc + "\n\nSummarise the entries above in three sentences.", GEN_TOKENS)


def _seeded_set(ctx):
    ps = [(f"seeded-{i}", req(p, GEN_TOKENS, temperature=SEEDED_TEMPERATURE, seed=SEED_A)) for i, p in enumerate(DETERMINISM_PROMPTS[2:4])]
    return ps if ctx.full else ps[:1]


def _run_store(st, ctx, label, items):
    out = ctx.shared.setdefault("det", {}).setdefault(label, {})
    for pid, p in items:
        c = chat(p)
        if not st.check(f"[{label}] {pid}: request succeeds", c.ok and c.completion_tokens, c.brief(), repro(p, "HTTP 200", c.brief())):
            continue
        out[pid] = {"text": c.text(), "tokens": c.completion_tokens, "payload": p, "cached": c.cached_tokens}
    return out


def _run_multi_turn(st, ctx, label, expect_cache):
    turns = MULTI_TURN if ctx.full else MULTI_TURN[:3]
    msgs, answers, cached, prompts = [], [], [], []
    for i, u in enumerate(turns):
        msgs.append({"role": "user", "content": u})
        p = req(list(msgs), 400)
        c = chat(p)
        if not st.check(f"[{label}] multi-turn: turn {i + 1} succeeds with a non-empty answer", c.ok and c.content.strip() and c.finish_reason == "stop", c.brief(), repro(p, "200, finish=stop", c.brief())):
            break
        answers.append(c.content)
        cached.append(c.cached_tokens)
        prompts.append(c.prompt_tokens)
        if i > 0 and expect_cache is True:
            st.check(f"[{label}] multi-turn: turn {i + 1} reuses the previous prompt from the cache",
                     c.cached_tokens >= prompts[i - 1] - CACHE_SLACK_TOKENS,
                     f"cached_tokens {c.cached_tokens}, previous prompt {prompts[i - 1]} tokens, this prompt {c.prompt_tokens}",
                     repro(p, f"cached_tokens >= {prompts[i - 1] - CACHE_SLACK_TOKENS}", f"cached_tokens {c.cached_tokens}"))
        if expect_cache is False:
            st.check(f"[{label}] multi-turn: turn {i + 1} reports no cached tokens", c.cached_tokens == 0, f"cached_tokens {c.cached_tokens}", repro(p, "cached_tokens 0 with --no-prefix-cache", str(c.usage)))
        msgs.append({"role": "assistant", "content": c.content})
    ctx.shared.setdefault("det", {}).setdefault(label, {})["multi-turn"] = {"answers": answers, "cached": cached, "prompts": prompts}
    return answers, cached, prompts


def base(st, ctx):
    # ---- greedy repeatability
    a = _run_store(st, ctx, "default", _greedy_set(ctx))
    for pid, p in _greedy_set(ctx):
        if pid not in a:
            continue
        c = chat(p)
        same = c.text() == a[pid]["text"]
        st.check(f"temperature 0: {pid} repeated gives identical output", same, f"{a[pid]['tokens']} tokens" if same else first_diff(a[pid]["text"], c.text())[1],
                 repro(p, "the same text on every run", first_diff(a[pid]["text"], c.text())[1]))
    pid, p = _greedy_set(ctx)[0]
    if pid in a:
        c = chat(dict(p, stream=True))
        st.check("temperature 0: streaming gives the same text as non-streaming", c.text() == a[pid]["text"] and not c.framing,
                 "identical" if c.text() == a[pid]["text"] else first_diff(a[pid]["text"], c.text())[1], repro(dict(p, stream=True), "same text as stream=false", first_diff(a[pid]["text"], c.text())[1]))

    # ---- the same long prompt twice: the second run resumes from the snapshot taken at the end of the prompt
    lp = _long_prompt()
    c1, c2 = chat(lp), chat(lp)
    if st.check("long prompt (about 900 tokens) answered", c1.ok and c2.ok, c1.brief()[:150], repro(lp, "200", c1.brief())):
        ctx.shared["det"]["default"]["long-prompt"] = {"text": c1.text(), "tokens": c1.completion_tokens, "payload": lp, "cached": c1.cached_tokens}
        st.check("the same long prompt sent again is served from the prefix cache", (c2.cached_tokens or 0) >= c2.prompt_tokens - CACHE_SLACK_TOKENS,
                 f"cached {c2.cached_tokens} of {c2.prompt_tokens} prompt tokens", repro(lp, "cached_tokens = prompt_tokens - 1 on the second send", str(c2.usage)))
        st.check("temperature 0: answer from the cached prompt is identical to the uncached one", c1.text() == c2.text(),
                 "identical" if c1.text() == c2.text() else first_diff(c1.text(), c2.text())[1],
                 repro(lp, "identical text whether the prompt was cached or not (send it twice)", first_diff(c1.text(), c2.text())[1]))

    # ---- seeded sampling
    s = _run_store(st, ctx, "default", _seeded_set(ctx))
    for pid, p in _seeded_set(ctx):
        if pid not in s:
            continue
        c = chat(p)
        st.check(f"temperature {SEEDED_TEMPERATURE}, seed {SEED_A}: {pid} repeated gives identical output", c.text() == s[pid]["text"],
                 f"{s[pid]['tokens']} tokens" if c.text() == s[pid]["text"] else first_diff(s[pid]["text"], c.text())[1], repro(p, "identical text for the same seed", first_diff(s[pid]["text"], c.text())[1]))
        p2 = dict(p, seed=SEED_B)
        c = chat(p2)
        st.check(f"temperature {SEEDED_TEMPERATURE}: seed {SEED_B} gives a different text than seed {SEED_A} ({pid})", c.ok and c.text() != s[pid]["text"], c.brief()[:120], repro(p2, "a different text", "identical text"))
    pid, p = _seeded_set(ctx)[0]
    pu = {k: v for k, v in p.items() if k != "seed"}
    c1, c2 = chat(pu), chat(pu)
    st.check(f"temperature {SEEDED_TEMPERATURE} without a seed: two runs differ (sampling is live)", c1.ok and c2.ok and c1.text() != c2.text(), c1.brief()[:100], repro(pu, "two different texts", "identical"), warn_only=True)
    # a seed the server cannot use (negative, fractional, string) should be rejected rather than silently replaced by a random one
    for bad_seed in (-5, 7.5, "7"):
        pb = dict(pu, seed=bad_seed)
        c1, c2 = chat(pb), chat(pb)
        if c1.status == 400:
            st.passed(f"seed={bad_seed!r} is rejected with 400")
        elif c1.ok and c2.ok and c1.text() == c2.text():
            st.passed(f"seed={bad_seed!r} is honoured (repeatable)")
        else:
            st.warn(f"seed={bad_seed!r} is silently ignored: the request succeeds but is not repeatable", "two runs gave different texts" if c1.ok and c2.ok else c1.brief()[:120],
                    repro(pb, "HTTP 400, or the same text on every run", "HTTP 200 and a different text each time (a random seed is used)"))
    # greedy must not depend on the seed
    pid, p = _greedy_set(ctx)[0]
    if pid in a:
        c = chat(dict(p, seed=12345))
        st.check("temperature 0 ignores the seed", c.text() == a[pid]["text"], "identical" if c.text() == a[pid]["text"] else first_diff(a[pid]["text"], c.text())[1], repro(dict(p, seed=12345), "same text as without seed", "differs"))

    # ---- multi-turn
    _run_multi_turn(st, ctx, "default", True)


# Two worked examples put the model in "echo" mode; without them it sometimes rewrites the text.
ECHO_SHOTS = [{"role": "user", "content": "Repeat exactly: abc def"}, {"role": "assistant", "content": "abc def"},
              {"role": "user", "content": "Repeat exactly: 12 → 34 ✓"}, {"role": "assistant", "content": "12 → 34 ✓"}]
# Even with reasoning_effort "none" the model thinks for several hundred tokens about unusual characters.
ECHO_MAX_TOKENS = 2500


def unicode_part(st, ctx):
    for name, text in UNICODE_CASES:
        p = req(ECHO_SHOTS + [{"role": "user", "content": "Repeat exactly: " + text}], ECHO_MAX_TOKENS)
        c = chat(p)
        cs = chat(dict(p, stream=True))
        if not st.check(f"unicode/{name}: request succeeds", c.ok and cs.ok, c.brief(), repro(p, "200", c.brief())):
            continue
        got = c.content.strip()
        exact = text in got
        nfc = unicodedata.normalize("NFC", text) in unicodedata.normalize("NFC", got)
        if exact or nfc:
            st.passed(f"unicode/{name}: text round-trips through prompt and reply", "exact" if exact else "equal after NFC normalisation (the model composed the combining marks)")
        else:
            # the server carried the characters both ways; what the model chose to write is model quality
            st.warn(f"unicode/{name}: text round-trips through prompt and reply", f"sent {text!r}, got {got[:120]!r} (finish={c.finish_reason})", repro(p, f"content == {text!r}", f"content {got[:200]!r}"))
        bad = [f for f in cs.framing] + (["U+FFFD in a delta"] if "�" in cs.content + cs.reasoning else [])
        st.check(f"unicode/{name}: streaming emits only complete UTF-8 characters", not bad, "; ".join(bad) or f"{len(cs.events)} chunks",
                 repro(dict(p, stream=True), "every SSE event valid UTF-8, no U+FFFD", "; ".join(bad)))
        st.check(f"unicode/{name}: streamed text equals the non-streamed text", cs.content == c.content and cs.reasoning == c.reasoning,
                 "identical" if cs.content == c.content else first_diff(c.content, cs.content)[1],
                 repro(p, "same content with stream true/false", first_diff(c.content, cs.content)[1]))
        st.check(f"unicode/{name}: no U+FFFD in the non-streamed reply", "�" not in c.content + c.reasoning, c.content[:80], repro(p, "no replacement characters", c.content))
    # A reply cut by max_tokens in the middle of a multi-byte character (emoji are several tokens of raw bytes)
    # must not produce broken text: cut the emoji echo at each of its last tokens.
    p = req(ECHO_SHOTS + [{"role": "user", "content": "Repeat exactly: " + UNICODE_CASES[0][1]}], ECHO_MAX_TOKENS)
    full = chat(p)
    bad, n_cuts = [], (30 if ctx.full else 12)
    if full.ok:
        for cut in range(max(1, full.completion_tokens - n_cuts), full.completion_tokens):
            for stream in (False, True):
                c = chat(dict(p, max_tokens=cut, stream=stream))
                if "�" in c.content + c.reasoning or c.framing or not c.ok:
                    bad.append(f"max_tokens={cut} stream={stream}: {c.framing or c.content[-20:]!r}")
                elif not UNICODE_CASES[0][1].startswith(c.content):
                    bad.append(f"max_tokens={cut} stream={stream}: content {c.content!r} is not a prefix of the full reply")
    st.check("a reply cut by max_tokens never ends in a broken character", full.ok and not bad, "; ".join(bad)[:300] or f"{n_cuts} cut points x stream/non-stream over an emoji reply",
             repro(p, "for every max_tokens: valid UTF-8, no U+FFFD, content is a prefix of the full reply", "; ".join(bad)[:400]))


def _valid_call(tc, declared):
    """Returns (ok, parsed args or None, problem)."""
    if not isinstance(tc, dict):
        return False, None, "tool call is not an object"
    f = tc.get("function") or {}
    if not isinstance(tc.get("id"), str) or not tc["id"]:
        return False, None, "missing id"
    if tc.get("type") != "function":
        return False, None, f"type {tc.get('type')!r}"
    if f.get("name") not in declared:
        return False, None, f"name {f.get('name')!r} is not a declared tool {declared}"
    if not isinstance(f.get("arguments"), str):
        return False, None, "arguments is not a string"
    try:
        args = json.loads(f["arguments"])
    except ValueError as e:
        return False, None, f"arguments do not parse as JSON ({e}): {f['arguments'][:120]!r}"
    if not isinstance(args, dict):
        return False, None, f"arguments are not a JSON object: {f['arguments'][:120]!r}"
    return True, args, ""


def tools_part(st, ctx):
    ids, valid, understood, total = [], 0, 0, 0
    cases = TOOL_CASES
    for case in cases:
        cid, names, _msgs, expected, checks = case
        total += 1
        p = tool_case_request(case, stream=(total % 3 == 0))  # every third case through the streaming path
        c = chat(p)
        if not c.ok:
            st.fail(f"tool/{cid}: request failed", c.brief(), repro(p, "HTTP 200 with tool_calls", c.brief()))
            continue
        leaked = (not c.tool_calls) and ('"args"' in c.content or '{"name"' in c.content.replace(" ", ""))
        if leaked:
            st.fail(f"tool/{cid}: a malformed tool call leaked into content", c.content[:300], repro(p, "tool_calls with JSON arguments", c.brief()))
            continue
        if not c.tool_calls:
            st.warn(f"tool/{cid}: the model answered in text instead of calling a tool", c.brief()[:240], repro(p, "a tool call", c.brief()))
            continue
        problems = []
        parsed = []
        for tc in c.tool_calls:
            ok, args, why = _valid_call(tc, names)
            if not ok:
                problems.append(why)
            else:
                parsed.append((tc["function"]["name"], args))
            if isinstance(tc, dict) and tc.get("id"):
                ids.append(tc["id"])
        if c.finish_reason != "tool_calls":
            problems.append(f"finish_reason {c.finish_reason!r}")
        if problems:
            undeclared = all("is not a declared tool" in x for x in problems)
            # A well-formed call to a tool the model made up is the model's mistake, passed on as any OpenAI-compatible
            # server passes it on (the client answers "unknown tool"). A malformed call is the server's problem.
            (st.warn if undeclared else st.fail)(f"tool/{cid}: " + ("the model called a tool that was not declared; the server passes the call on" if undeclared else "malformed tool call"),
                    "; ".join(problems), repro(p, "well-formed tool_calls naming a declared tool with JSON-object arguments", c.brief()))
            continue
        valid += 1
        good = any(n == expected and all(chk(a) for chk in checks) for n, a in parsed)
        if cid == "two-calls":
            cities = sorted(str(a.get("city", "")).lower() for _n, a in parsed)
            good = len(parsed) == 2 and "paris" in cities[0] + cities[1] and "tokyo" in cities[0] + cities[1]
            st.info("tool/two-calls: calls in one reply", f"{len(parsed)} call(s): {parsed}")
        understood += good
        if not good:
            st.info(f"tool/{cid}: valid call, but not the expected tool/arguments", json.dumps(parsed, ensure_ascii=False)[:300])
    rate = valid / total
    st.num("tool cases with a well-formed call", f"{valid}/{total}", head=True)
    st.num("tool cases with the expected tool and arguments", f"{understood}/{total}")
    if rate < TOOL_VALID_FAIL:
        st.fail("tool calling: share of cases answered with a well-formed tool call", f"{valid}/{total} ({rate:.0%}), FAIL below {TOOL_VALID_FAIL:.0%}; README claims 24/24")
    elif rate < TOOL_VALID_WARN:
        st.warn("tool calling: share of cases answered with a well-formed tool call", f"{valid}/{total} ({rate:.0%}), WARN below {TOOL_VALID_WARN:.0%}; README claims 24/24")
    else:
        st.passed("tool calling: share of cases answered with a well-formed tool call", f"{valid}/{total} ({rate:.0%})")
    st.check("tool calling: the expected tool with the expected arguments (model quality)", understood / total >= TOOL_ARGS_WARN,
             f"{understood}/{total}", warn_only=True)

    # tool_choice "none": the tools must not be used
    p = dict(tool_case_request(TOOL_CASES[0]), tool_choice="none")
    c = chat(p)
    st.check("tool_choice 'none' suppresses tool calls", c.ok and not c.tool_calls and c.finish_reason == "stop", c.brief()[:200], repro(p, "a text reply", c.brief()))

    # ---- full loops
    for stream in (False, True):
        tag = "streaming" if stream else "non-streaming"
        msgs = user("What is the weather in Paris right now, in celsius? Use the tool, then tell me the temperature.")
        p = req(msgs, 500, tools=[T["get_weather"]], stream=stream)
        c = chat(p)
        okc = c.ok and len(c.tool_calls) == 1 and _valid_call(c.tool_calls[0], ["get_weather"])[0]
        if not st.check(f"tool loop ({tag}): first reply is a get_weather call", okc, c.brief()[:240], repro(p, "one get_weather call", c.brief())):
            continue
        ids.append(c.tool_calls[0]["id"])
        msgs = msgs + [{"role": "assistant", "content": c.content or None, "tool_calls": c.tool_calls},
                       {"role": "tool", "tool_call_id": c.tool_calls[0]["id"], "content": json.dumps({"temperature_c": 17, "sky": "light rain"})}]
        p = req(msgs, 500, tools=[T["get_weather"]], stream=stream)
        c2 = chat(p)
        st.check(f"tool loop ({tag}): the final answer uses the tool result", c2.ok and not c2.tool_calls and "17" in c2.content and c2.finish_reason == "stop",
                 c2.brief()[:240], repro(p, "a text answer containing 17", c2.brief()))
        st.check(f"tool loop ({tag}): the follow-up request reuses the cached prompt", (c2.cached_tokens or 0) >= c.prompt_tokens - CACHE_SLACK_TOKENS,
                 f"cached {c2.cached_tokens}, first prompt {c.prompt_tokens}", repro(p, f"cached_tokens >= {c.prompt_tokens - CACHE_SLACK_TOKENS}", str(c2.usage)))

        # two calls in a row
        tools = [T["get_user_id"], T["get_orders"]]
        msgs = user("How many orders does the user named alice have? First look up her numeric id, then list her orders, then answer with the number.")
        p = req(msgs, 500, tools=tools, stream=stream)
        c = chat(p)
        ok1 = c.ok and c.tool_calls and _valid_call(c.tool_calls[0], ["get_user_id", "get_orders"])[0] and c.tool_calls[0]["function"]["name"] == "get_user_id"
        if not st.check(f"two-step tool loop ({tag}): step 1 calls get_user_id", ok1, c.brief()[:240], repro(p, "get_user_id call", c.brief())):
            continue
        ids.append(c.tool_calls[0]["id"])
        msgs = msgs + [{"role": "assistant", "content": c.content or None, "tool_calls": c.tool_calls[:1]},
                       {"role": "tool", "tool_call_id": c.tool_calls[0]["id"], "content": json.dumps({"user_id": 4471})}]
        p = req(msgs, 500, tools=tools, stream=stream)
        c = chat(p)
        ok2 = c.ok and c.tool_calls and _valid_call(c.tool_calls[0], ["get_orders"])[0]
        args = json.loads(c.tool_calls[0]["function"]["arguments"]) if ok2 else {}
        if not st.check(f"two-step tool loop ({tag}): step 2 calls get_orders with the id from step 1", ok2 and args.get("user_id") == 4471, c.brief()[:240], repro(p, "get_orders {user_id: 4471}", c.brief())):
            continue
        ids.append(c.tool_calls[0]["id"])
        msgs = msgs + [{"role": "assistant", "content": c.content or None, "tool_calls": c.tool_calls[:1]},
                       {"role": "tool", "tool_call_id": c.tool_calls[0]["id"], "content": json.dumps({"orders": [{"id": "A1"}, {"id": "A2"}, {"id": "A3"}]})}]
        p = req(msgs, 500, tools=tools, stream=stream)
        c = chat(p)
        st.check(f"two-step tool loop ({tag}): final answer is 3", c.ok and not c.tool_calls and ("3" in c.content or "three" in c.content.lower()), c.brief()[:240], repro(p, "an answer containing 3", c.brief()))

    # ---- ids: unique, also when 4 requests finish in the same instant
    burst = run_parallel([lambda: chat(tool_case_request(TOOL_CASES[0])) for _ in range(4)])
    for c in burst:
        for tc in getattr(c, "tool_calls", []) or []:
            ids.append(tc.get("id"))
    st.check("tool-call ids are unique across all calls of this stage", len(ids) == len(set(ids)) and len(ids) > 10, f"{len(ids)} ids, {len(set(ids))} distinct",
             repro(None, "all ids distinct", f"{len(ids) - len(set(ids))} duplicates"))


def spec0(st, ctx):
    _run_store(st, ctx, "spec0", _greedy_set(ctx) + _seeded_set(ctx))
    lp = _long_prompt()
    _run_store(st, ctx, "spec0", [("long-prompt", lp)])


def nocache(st, ctx):
    _run_store(st, ctx, "nocache", _greedy_set(ctx))
    lp = _long_prompt()
    _run_store(st, ctx, "nocache", [("long-prompt", lp)])
    c = chat(lp)
    st.check("[nocache] the same long prompt sent again reports no cached tokens", c.ok and c.cached_tokens == 0, str(c.usage), repro(lp, "cached_tokens 0", str(c.usage)))
    _run_multi_turn(st, ctx, "nocache", False)


def concurrent(st, ctx):
    items = _greedy_set(ctx) + _seeded_set(ctx)
    _run_store(st, ctx, "slots4-alone", items)
    busy = ctx.shared.setdefault("det", {}).setdefault("slots4-busy", {})
    for k, (pid, p) in enumerate(items):
        # three other conversations generating for the whole time the measured request runs
        started = threading.Barrier(4, timeout=60)
        stop = threading.Event()

        def background(i):
            first = [True]

            def on_event(c, obj):
                if first[0] and c.t_first is not None:
                    first[0] = False
                    try:
                        started.wait()
                    except threading.BrokenBarrierError:
                        pass
            bp = req(PERF_PROMPTS[(k + i) % len(PERF_PROMPTS)] if i else LONG_OUTPUT_PROMPT, 1500, temperature=0.7 if i == 1 else 0, seed=100 + i, stream=True)
            return chat(bp, on_event=on_event, cancel_event=stop)

        res = {}
        ths = [threading.Thread(target=lambda i=i: res.__setitem__(i, background(i)), daemon=True) for i in range(3)]
        for th in ths:
            th.start()
        try:
            started.wait()
        except threading.BrokenBarrierError:
            st.fail(f"[slots4-busy] {pid}: background streams did not start", "barrier timeout")
            continue
        c = chat(p)
        bg_active = sum(1 for th in ths if th.is_alive())
        stop.set()  # the background clients hang up; their slots are free for the next round
        for th in ths:
            th.join(timeout=120)
        if st.check(f"[slots4-busy] {pid}: request succeeds while 3 other streams generate", c.ok and bg_active >= 2,
                    f"{bg_active} background streams still active when it finished; {c.brief()[:100]}", repro(p, "200 while 3 other streams run", c.brief())):
            busy[pid] = {"text": c.text(), "tokens": c.completion_tokens, "payload": p}


def _cmp(st, ctx, name, a_label, b_label, pid, what):
    det = ctx.shared.get("det", {})
    a, b = det.get(a_label, {}).get(pid), det.get(b_label, {}).get(pid)
    if a is None or b is None:
        st.skip(name, f"no output recorded for {pid} under {a_label if a is None else b_label} (that part did not run or failed)")
        return
    same = a["text"] == b["text"]
    st.check(name, same, f"{a['tokens']} tokens identical" if same else first_diff(a["text"], b["text"])[1],
             repro(a["payload"], f"identical output {what}", first_diff(a["text"], b["text"])[1],
                   note=f"compare the reply from the server started as [{a_label}] with the one started as [{b_label}]"))


def compare(st, ctx):
    det = ctx.shared.get("det", {})
    greedy = [pid for pid, _ in _greedy_set(ctx)]
    seeded = [pid for pid, _ in _seeded_set(ctx)]
    for pid in greedy + ["long-prompt"]:
        _cmp(st, ctx, f"temperature 0: {pid} identical with --spec 0 and with default speculation", "default", "spec0", pid, "with and without speculation")
        _cmp(st, ctx, f"temperature 0: {pid} identical with --no-prefix-cache", "default", "nocache", pid, "with and without the prefix cache")
    for pid in greedy:
        _cmp(st, ctx, f"temperature 0: {pid} identical on a 4-slot server (alone) and on a 1-slot server", "default", "slots4-alone", pid, "with --slots 4 and --slots 1")
        _cmp(st, ctx, f"temperature 0: {pid} identical alone and alongside 3 active streams (4 slots)", "slots4-alone", "slots4-busy", pid, "alone and under concurrency")
    for pid in seeded:
        _cmp(st, ctx, f"seeded sampling: {pid} identical with --spec 0 and with default speculation", "default", "spec0", pid, "for the same seed with and without speculation")
        _cmp(st, ctx, f"seeded sampling: {pid} identical alone and alongside 3 active streams (4 slots)", "slots4-alone", "slots4-busy", pid, "for the same seed alone and under concurrency")
    a, b = det.get("default", {}).get("multi-turn"), det.get("nocache", {}).get("multi-turn")
    if a and b:
        n = min(len(a["answers"]), len(b["answers"]))
        diffs = [i + 1 for i in range(n) if a["answers"][i] != b["answers"][i]]
        st.check("multi-turn: every answer is the same with the prefix cache on and off", not diffs and n > 0 and len(a["answers"]) == len(b["answers"]),
                 f"{n} turns identical; cached tokens per turn {a['cached']} for prompts of {a['prompts']} tokens" if not diffs else
                 f"turns {diffs} differ; turn {diffs[0]}: " + first_diff(a["answers"][diffs[0] - 1], b["answers"][diffs[0] - 1])[1],
                 repro({"conversation": MULTI_TURN}, "identical answers", f"turns {diffs} differ",
                       note="replay the user turns, feeding each assistant answer back, once on a default server and once with --no-prefix-cache"))
        if a["prompts"]:
            st.num("multi-turn cached tokens per turn", "/".join(map(str, a["cached"])), "tokens")
    else:
        st.skip("multi-turn: cache on vs off", "one of the two runs is missing")


def parts(suite):
    return [("base", "default", base), ("unicode", "default", unicode_part), ("tools", "default", tools_part),
            ("spec0", "spec0", spec0), ("nocache", "nocache", nocache), ("concurrent", "slots4", concurrent),
            ("compare", None, compare)]
