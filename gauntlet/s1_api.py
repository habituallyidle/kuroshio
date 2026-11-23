"""Stage 1 - API conformance: response shapes, SSE framing, and error handling that must leave the server healthy."""
import json
from http.client import HTTPConnection
import time

from core import (HOST, PORT, build_request, chat, healthy, http, raw_request, repro, now)
from data import MODEL, T, req, user

KEY, TITLE = "api", "API conformance"

# The server documents a 32 MiB request-body limit (MAX_BODY in host/src/server.rs).
MAX_BODY = 32 << 20
# An error reply must arrive quickly: rejecting a request is not allowed to tie up a connection thread.
ERROR_REPLY_MAX_S = 5.0
FINISH_REASONS = {"stop", "length", "tool_calls"}
# One SSE event should arrive as one write. Reads on loopback follow the server's writes closely; allow 3 per
# event before calling it fragmented.
SEGMENTS_PER_EVENT_WARN = 3


def _usage_ok(u):
    if not isinstance(u, dict):
        return "usage is missing"
    for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(u.get(k), int) or u[k] < 0:
            return f"usage.{k} is not a non-negative integer: {u.get(k)!r}"
    if u["total_tokens"] != u["prompt_tokens"] + u["completion_tokens"]:
        return f"total_tokens {u['total_tokens']} != prompt + completion"
    d = u.get("prompt_tokens_details")
    if not isinstance(d, dict) or not isinstance(d.get("cached_tokens"), int):
        return f"usage.prompt_tokens_details.cached_tokens missing: {d!r}"
    if not 0 <= d["cached_tokens"] <= u["prompt_tokens"]:
        return f"cached_tokens {d['cached_tokens']} outside 0..prompt_tokens"
    return None


def shapes(st, ctx):
    # ---- /health and /v1/models
    r = http("GET", "/health")
    st.check("GET /health is 200 {status: ok} with a correct Content-Length",
             r.status == 200 and (r.json() or {}).get("status") == "ok" and r.headers.get("content-length") == str(len(r.body)),
             r.raw[:200], repro(raw="GET /health", expected='200 {"status":"ok"}', actual=r.raw[:200].decode("latin-1")))
    r = http("GET", "/v1/models")
    j = r.json() or {}
    m = (j.get("data") or [{}])[0]
    st.check("GET /v1/models lists the model with a context size",
             r.status == 200 and j.get("object") == "list" and m.get("id") == MODEL and m.get("object") == "model"
             and isinstance(m.get("context_window"), int) and m["context_window"] >= 2048,
             r.body[:300], repro(raw="GET /v1/models", expected="object=list, data[0].id=inkling-2-mini, context_window>=2048", actual=r.text()))
    ctx.shared["context_window"] = m.get("context_window")
    st.num("context_window (1 slot)", m.get("context_window"), "tokens", head=True)

    # ---- non-streaming
    p = req("What is the capital of France? Answer in one word.", 300)
    c = chat(p)
    j = c.json or {}
    probs = []
    if c.status != 200:
        probs.append(f"status {c.status}")
    if not str(j.get("id", "")).startswith("chatcmpl-"):
        probs.append(f"id {j.get('id')!r}")
    if j.get("object") != "chat.completion":
        probs.append(f"object {j.get('object')!r}")
    if not isinstance(j.get("created"), int) or abs(j.get("created", 0) - time.time()) > 120:
        probs.append(f"created {j.get('created')!r} is not the current unix time")
    if j.get("model") != MODEL:
        probs.append(f"model {j.get('model')!r}")
    ch = (j.get("choices") or [{}])
    if len(ch) != 1 or ch[0].get("index") != 0:
        probs.append("choices is not one entry with index 0")
    msg = ch[0].get("message") or {}
    if msg.get("role") != "assistant" or not isinstance(msg.get("content"), str):
        probs.append(f"message role/content: {msg!r}"[:200])
    u = _usage_ok(j.get("usage"))
    if u:
        probs.append(u)
    if c.headers.get("content-type") != "application/json":
        probs.append(f"content-type {c.headers.get('content-type')!r}")
    st.check("non-streaming reply is a well-formed chat.completion", not probs, "; ".join(probs) or c.brief()[:160],
             repro(p, "a chat.completion object with id, created, model, one choice, usage", "; ".join(probs)))
    st.check("finish_reason is 'stop' when the model ends its answer", c.finish_reason == "stop" and "paris" in c.content.lower(),
             c.brief(), repro(p, "finish_reason=stop and 'Paris' in content", c.brief()))
    st.check("no control tokens leak into content or reasoning_content", "<|" not in c.content and "<|" not in c.reasoning,
             c.brief(), repro(p, "no '<|' in text", c.brief()))

    p = req("Count from 1 to 200, one number per line.", 7)
    c = chat(p)
    st.check("finish_reason is 'length' and completion_tokens equals max_tokens when the limit cuts the reply",
             c.status == 200 and c.finish_reason == "length" and c.completion_tokens == 7, c.brief(),
             repro(p, "finish_reason=length, completion_tokens=7", c.brief()))
    p = {"model": MODEL, "messages": user("Count from 1 to 200, one number per line."), "max_completion_tokens": 9,
         "temperature": 0, "reasoning_effort": "none"}
    c = chat(p)
    st.check("max_completion_tokens is honoured", c.status == 200 and c.completion_tokens == 9 and c.finish_reason == "length",
             c.brief(), repro(p, "completion_tokens=9", c.brief()))
    p = req("What is the weather in Oslo?", 400, tools=[T["get_weather"]])
    c = chat(p)
    st.check("finish_reason is 'tool_calls' when the reply is a tool call", c.status == 200 and c.finish_reason == "tool_calls" and len(c.tool_calls) >= 1,
             c.brief(), repro(p, "finish_reason=tool_calls with one call", c.brief()))

    # ---- reasoning vs content
    p = req("What is 17 * 23? Think it through, then give the result.", 800, effort="medium")
    c = chat(p)
    st.check("thinking is returned as reasoning_content, the answer as content",
             c.status == 200 and len(c.reasoning) > 0 and "391" in c.content and c.finish_reason == "stop",
             c.brief() + f" reasoning={c.reasoning[:80]!r}", repro(p, "non-empty reasoning_content and '391' in content", c.brief()))
    cs = chat(dict(p, stream=True))
    order = []
    for _t, ev in cs.events:
        for chx in ev.get("choices") or []:
            d = chx.get("delta") or {}
            if d.get("reasoning_content"):
                order.append("r")
            if d.get("content"):
                order.append("c")
    seq = "".join(order)
    st.check("streaming: reasoning_content deltas all come before content deltas", "cr" not in seq and "r" in seq and "c" in seq,
             f"delta kinds in order: {seq[:40]}...", repro(dict(p, stream=True), "reasoning deltas then content deltas", seq[:200]))
    st.check("streaming and non-streaming give the same reasoning and content at temperature 0",
             cs.content == c.content and cs.reasoning == c.reasoning,
             f"non-stream content {c.content[:80]!r} vs stream {cs.content[:80]!r}",
             repro(p, "identical text with stream true/false", f"non-stream: {c.text()[:300]!r}; stream: {cs.text()[:300]!r}"))
    p2 = {"model": MODEL, "messages": user("What is 17 * 23?"), "max_tokens": 600, "temperature": 0,
          "chat_template_kwargs": {"reasoning_effort": "low"}}
    c = chat(p2)
    st.check("chat_template_kwargs.reasoning_effort is accepted", c.status == 200 and "391" in c.content, c.brief(), repro(p2, "200 with 391", c.brief()))
    for eff, want in (("minimal", 200), ("xhigh", 200), (0.5, 200), ("bogus", 400), (5, 400)):
        p3 = req("Say hi.", 24, effort=eff)
        c = chat(p3)
        st.check(f"reasoning_effort={eff!r} gives HTTP {want}", c.status == want, c.brief(), repro(p3, f"HTTP {want}", c.brief()))

    # ---- message shapes clients really send
    p = req([{"role": "system", "content": "Always answer in French."}, {"role": "user", "content": "Say good morning."}], 300)
    c = chat(p)
    st.check("system message is honoured", c.status == 200 and "bonjour" in c.content.lower(), c.brief(), repro(p, "'Bonjour' in content", c.brief()))
    p = req([{"role": "developer", "content": "Always answer in French."}, {"role": "user", "content": "Say good morning."}], 300)
    c = chat(p)
    st.check("'developer' role is accepted as system", c.status == 200 and "bonjour" in c.content.lower(), c.brief(), repro(p, "'Bonjour' in content", c.brief()))
    p = req([{"role": "user", "content": [{"type": "text", "text": "What is the capital of France? "}, {"type": "text", "text": "Answer in one word."}]}], 300)
    c = chat(p)
    st.check("content given as an array of text parts is accepted", c.status == 200 and "paris" in c.content.lower(), c.brief(), repro(p, "200 with Paris", c.brief()))

    p = req("What is the capital of France? Answer in one word.", 300, tools=[])
    c = chat(p)
    st.check("an empty tools array is accepted", c.status == 200 and "paris" in c.content.lower(), c.brief(), repro(p, "200 with Paris", c.brief()))
    call = {"id": "call_abc123", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}
    p = req([{"role": "user", "content": "What is the weather in Oslo?"}, {"role": "assistant", "content": None, "tool_calls": [call]},
             {"role": "tool", "tool_call_id": "call_abc123", "content": [{"type": "text", "text": "{\"temperature_c\": -3, \"sky\": \"snow\"}"}]}], 400, tools=[T["get_weather"]])
    c = chat(p)
    st.check("a tool result whose content is an array of text parts is accepted", c.status == 200 and ("-3" in c.content or "−3" in c.content or "snow" in c.content.lower()), c.brief(),
             repro(p, "200 and an answer using the tool result", c.brief()), warn_only=True)

    # ---- streaming shape
    p = req("What is the capital of France? Answer in one word.", 300, stream=True)
    c = chat(p)
    probs = list(c.framing)
    if c.status != 200:
        probs.append(f"status {c.status}")
    if not c.headers.get("content-type", "").startswith("text/event-stream"):
        probs.append(f"content-type {c.headers.get('content-type')!r}")
    ids = {ev.get("id") for _t, ev in c.events}
    if len(ids) != 1:
        probs.append(f"{len(ids)} different ids across chunks")
    if any(ev.get("object") != "chat.completion.chunk" for _t, ev in c.events):
        probs.append("a chunk's object is not chat.completion.chunk")
    fins = [i for i, (_t, ev) in enumerate(c.events) if (ev.get("choices") or [{}])[0].get("finish_reason") is not None]
    if fins != [len(c.events) - 1]:
        probs.append(f"finish_reason is set on chunks {fins}, expected only the last ({len(c.events) - 1})")
    if c.finish_reason not in FINISH_REASONS:
        probs.append(f"finish_reason {c.finish_reason!r}")
    u = _usage_ok(c.usage)
    if u:
        probs.append("final chunk: " + u)
    if not c.done:
        probs.append("no [DONE]")
    st.check("streaming reply is well-formed SSE (data: lines, one finish_reason, usage, [DONE])", not probs,
             "; ".join(probs) or f"{len(c.events)} chunks", repro(p, "well-formed SSE stream", "; ".join(probs)))
    first = (c.events[0][1].get("choices") or [{}])[0].get("delta", {}) if c.events else {}
    st.check("streaming: the first chunk's delta carries role=assistant (OpenAI does; some strict clients expect it)",
             first.get("role") == "assistant", f"first delta: {json.dumps(first)[:120]}",
             repro(p, 'first chunk delta contains "role":"assistant"', json.dumps(first)[:200]), warn_only=True)
    # how the stream is put on the wire: one write per event, or many small ones?
    import socket
    sp = req("Count from 1 to 40, separated by commas.", 300, stream=True)
    sk = socket.create_connection((HOST, PORT), timeout=60)
    sk.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    sk.sendall(build_request("POST", "/v1/chat/completions", json.dumps(sp).encode()))
    sizes, buf = [], b""
    while True:
        b = sk.recv(65536)
        if not b:
            break
        sizes.append(len(b))
        buf += b
    sk.close()
    n_events = buf.count(b"\n\ndata: ") + 1
    tiny = sum(1 for x in sizes if x <= 8)
    st.num("streaming: socket reads per SSE event", round(len(sizes) / max(1, n_events), 1))
    st.check("streaming: an event reaches the client in one piece (at most 3 socket reads per event on average)", len(sizes) <= SEGMENTS_PER_EVENT_WARN * n_events,
             f"{n_events} events arrived in {len(sizes)} separate reads ({len(sizes) / max(1, n_events):.0f} per event), {tiny} of them 8 bytes or less; first sizes: {sizes[:16]}",
             repro(sp, "about one TCP segment per event", f"{len(sizes)} reads for {n_events} events",
                   note="the server formats JSON straight into an unbuffered socket with TCP_NODELAY: every fragment of an event is its own write() and its own packet "
                        "(tens of syscalls and packets per token batch; over a real network, mostly header overhead). The same happens for non-streaming replies and error bodies."),
             warn_only=True)

    p = req("Count from 1 to 200, one number per line.", 12, stream=True)
    c = chat(p)
    st.check("streaming: finish_reason 'length' and usage reported", c.finish_reason == "length" and c.completion_tokens == 12 and c.done, c.brief(), repro(p, "length / 12 tokens / [DONE]", c.brief()))
    p = req("What is the weather in Oslo?", 400, tools=[T["get_weather"]], stream=True)
    c = chat(p)
    tc = c.tool_calls[0] if c.tool_calls else {}
    okc = (c.finish_reason == "tool_calls" and isinstance(tc.get("index"), int) and isinstance(tc.get("id"), str) and tc.get("type") == "function"
           and (tc.get("function") or {}).get("name") == "get_weather")
    try:
        okc = okc and isinstance(json.loads(tc["function"]["arguments"]), dict)
    except Exception:
        okc = False
    st.check("streaming: tool call delta is well-formed (index, id, type, function.name, JSON arguments)", okc and not c.framing, c.brief(), repro(p, "one well-formed tool_calls delta, finish_reason=tool_calls", c.brief()))


def _expect_4xx(st, name, send, want=None, describe=None, allow_no_response=False):
    """Run a request that must be rejected; then the server must still be healthy."""
    t0 = now()
    try:
        r = send()
    except OSError as e:
        r = None
        err = e
    dt = now() - t0
    ok_health = healthy()
    if r is None:
        st.fail(name, f"network error {err!r}; healthy afterwards: {ok_health}", repro(raw=describe, expected="a 4xx response", actual=repr(err)))
        return None
    status = r.status
    good = (status in want) if want else (400 <= status < 500)
    if allow_no_response and status == 0:
        good = True
    detail = f"HTTP {status} in {dt * 1000:.0f} ms: {r.body[:140].decode('utf-8', 'replace')}"
    exp = f"HTTP {'/'.join(map(str, want)) if want else '4xx'} and a healthy server"
    st.check(name, good and ok_health and dt < ERROR_REPLY_MAX_S, detail + ("" if ok_health else "; SERVER NOT HEALTHY AFTERWARDS"),
             repro(raw=describe, expected=exp, actual=detail + f"; healthy afterwards: {ok_health}"))
    if good and status and not (isinstance(r.json(), dict) and isinstance((r.json() or {}).get("error"), dict) and (r.json()["error"].get("message"))):
        st.warn(name + " - error body is not {error: {message}}", r.body[:200])
    return r


def errors(st, ctx):
    post = lambda body, **kw: (lambda: http("POST", "/v1/chat/completions", body, **kw))
    base = req("Say hi.", 8)
    cases = [
        ("malformed JSON", b'{"messages": [{"role": "user", "content": "hi"}', None),
        ("empty body", b"", None),
        ("body is a JSON array, not an object", b"[1, 2, 3]", None),
        ("invalid UTF-8 in the body", b'{"messages":[{"role":"user","content":"\xff\xfe"}],"max_tokens":4}', None),
        ("JSON nested 100,000 levels deep", b"[" * 100_000, None),
        ("missing messages", json.dumps({"model": MODEL, "max_tokens": 8}).encode(), None),
        ("messages is a string", json.dumps(dict(base, messages="hello")).encode(), None),
        ("message without a role", json.dumps(dict(base, messages=[{"content": "hi"}])).encode(), None),
        ("unknown role", json.dumps(dict(base, messages=[{"role": "wizard", "content": "hi"}])).encode(), None),
        ("temperature -1", json.dumps(dict(base, temperature=-1)).encode(), None),
        ("temperature 2.5", json.dumps(dict(base, temperature=2.5)).encode(), None),
        ("temperature as a string", json.dumps(dict(base, temperature="hot")).encode(), None),
        ("n = 2", json.dumps(dict(base, n=2)).encode(), None),
        ("max_tokens 0", json.dumps(dict(base, max_tokens=0)).encode(), None),
        ("max_tokens -1", json.dumps(dict(base, max_tokens=-1)).encode(), None),
        ("max_tokens as a string", json.dumps(dict(base, max_tokens="ten")).encode(), None),
        ("max_completion_tokens 0", json.dumps(dict({k: v for k, v in base.items() if k != "max_tokens"}, max_completion_tokens=0)).encode(), None),
    ]
    for name, body, want in cases:
        _expect_4xx(st, f"error: {name} -> 4xx", post(body), want, describe=f"POST /v1/chat/completions with body {body[:200]!r}")
        # the same request with stream:true must be rejected with a status code too, not inside a 200 stream
    sbody = json.dumps(dict(base, stream=True, temperature=9)).encode()
    _expect_4xx(st, "error: invalid request with stream=true is still an HTTP 4xx (not an error inside a 200 stream)", post(sbody), None,
                describe=f"POST /v1/chat/completions with body {sbody!r}")

    # Requests OpenAI rejects but this server accepts: worth knowing, not a failure of the documented API.
    c = chat(dict(base, messages=[]))
    if c.status == 200:
        st.warn("empty messages array is accepted (OpenAI answers 400); the model replies to a conversation that does not exist",
                c.brief()[:260] + f" reasoning={c.reasoning[:80]!r}", repro(dict(base, messages=[]), "HTTP 400", c.brief()))
    else:
        st.check("empty messages array -> 4xx", 400 <= c.status < 500, c.brief())

    # ---- HTTP-level
    get = lambda path, method="GET", body=None: (lambda: http(method, path, body))
    _expect_4xx(st, "wrong method: GET /v1/chat/completions -> 4xx", get("/v1/chat/completions"), describe="GET /v1/chat/completions")
    _expect_4xx(st, "wrong method: POST /health -> 4xx", get("/health", "POST", b"{}"), describe="POST /health")
    _expect_4xx(st, "wrong method: DELETE /v1/models -> 4xx", get("/v1/models", "DELETE"), describe="DELETE /v1/models")
    _expect_4xx(st, "unknown path -> 404", get("/v1/nope"), want=(404,), describe="GET /v1/nope")
    body = json.dumps(base).encode()
    _expect_4xx(st, "POST without Content-Length -> 4xx", lambda: raw_request(build_request("POST", "/v1/chat/completions", body, content_length=False), timeout=20),
                describe="POST /v1/chat/completions with a JSON body but no Content-Length header")
    _expect_4xx(st, "non-numeric Content-Length -> 4xx", lambda: raw_request(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: abc\r\n\r\n{}", timeout=20),
                describe="Content-Length: abc")
    _expect_4xx(st, "negative Content-Length -> 4xx", lambda: raw_request(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: -5\r\n\r\n{}", timeout=20),
                describe="Content-Length: -5")
    _expect_4xx(st, "Transfer-Encoding: chunked -> 4xx (not supported, must say so)",
                lambda: raw_request(b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n", timeout=20),
                describe="chunked request body")
    for n, label in ((16_000, "16 KB"), (1_000_000, "1 MB")):
        _expect_4xx(st, f"very long header line ({label}) -> 4xx", lambda n=n: raw_request(b"GET /health HTTP/1.1\r\nX-Long: " + b"a" * n + b"\r\n\r\n", timeout=20),
                    describe=f"GET /health with one {label} header line")
    _expect_4xx(st, "very long request line (100 KB) -> 4xx", lambda: raw_request(b"GET /" + b"a" * 100_000 + b" HTTP/1.1\r\n\r\n", timeout=20),
                describe="GET /aaaa...(100 KB) HTTP/1.1")
    _expect_4xx(st, "500 header lines -> 4xx", lambda: raw_request(b"GET /health HTTP/1.1\r\n" + b"".join(b"X-H%d: 1\r\n" % i for i in range(500)) + b"\r\n", timeout=20),
                describe="GET /health with 500 header lines")
    _expect_4xx(st, "binary garbage instead of a request -> 4xx", lambda: raw_request(b"\x00\x01\x02\xff\xfe\r\n\r\n", timeout=20), describe="bytes 00 01 02 ff fe CRLF CRLF")
    r = raw_request(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 500\r\n\r\n{", timeout=20, shutdown_write=True)
    st.check("body shorter than Content-Length, then the client closes: connection dropped, server healthy", r.status in (0, 400) and healthy(),
             f"status {r.status}", repro(raw="Content-Length: 500, one byte of body, then FIN", expected="no reply or 400; server healthy", actual=f"status {r.status}, healthy {healthy()}"))

    # ---- body size limit
    small = json.dumps(req("Say hi.", 8)).encode()
    exact = small + b" " * (MAX_BODY - len(small))
    r = raw_request(build_request("POST", "/v1/chat/completions", exact), timeout=120)
    st.check("a body of exactly 32 MiB (valid JSON padded with spaces) is accepted", r.status == 200 and healthy(), f"HTTP {r.status} {r.body[:120]!r}",
             repro(raw="valid request padded with spaces to exactly 33554432 bytes", expected="HTTP 200", actual=f"HTTP {r.status} {r.body[:200]!r}"))
    head = f"POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nContent-Length: {MAX_BODY + 1}\r\n\r\n".encode()
    _expect_4xx(st, "body of 32 MiB + 1 byte announced (headers only) -> 413", lambda: raw_request(head, timeout=20), want=(413,),
                describe=f"POST with Content-Length: {MAX_BODY + 1}, no body sent")
    _expect_4xx(st, "body of 32 MiB + 1 byte, fully sent -> 413 and the server survives", lambda: raw_request(head + small + b" " * (MAX_BODY + 1 - len(small)), timeout=60), want=(413,),
                describe=f"POST with Content-Length: {MAX_BODY + 1} and that many bytes of body")
    # What a stock client sees: the server answers and closes while the client is still uploading, so the kernel
    # resets the connection. Python's http.client then raises instead of returning the 413.
    seen = None
    try:
        conn = HTTPConnection(HOST, PORT, timeout=60)
        conn.request("POST", "/v1/chat/completions", body=small + b" " * (MAX_BODY + 1 - len(small)), headers={"Content-Type": "application/json"})
        seen = f"HTTP {conn.getresponse().status}"
        conn.close()
    except Exception as e:  # noqa: BLE001
        seen = f"{type(e).__name__}: {e}"
    st.check("a stock HTTP client (Python http.client) that uploads 32 MiB + 1 gets to read the 413", seen == "HTTP 413", seen,
             repro(raw="http.client POST of 33554433 bytes", expected="HTTP 413 visible to the client",
                   actual=f"{seen} (the server closes without reading the body; unread data makes the kernel send RST, which can destroy the 413 in flight)"),
             warn_only=True)

    # ---- things a real client may do that are not covered by the README
    t0 = now()
    r = raw_request(b"POST /v1/chat/completions HTTP/1.1\r\nContent-Type: application/json\r\nExpect: 100-continue\r\nContent-Length: " + str(len(small)).encode() + b"\r\n\r\n",
                    timeout=2.5)
    st.check("Expect: 100-continue gets '100 Continue' (curl sends it for bodies over 1 MiB and otherwise stalls for 1 s)",
             r.raw.startswith(b"HTTP/1.1 100"), f"reply within 2.5 s: {r.raw[:60]!r}",
             repro(raw="POST headers with 'Expect: 100-continue', body withheld", expected="HTTP/1.1 100 Continue", actual=f"{r.raw[:60]!r} after {now() - t0:.1f}s"),
             warn_only=True)
    r = http("OPTIONS", "/v1/chat/completions", headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "POST"})
    # Documented: web pages may call the server only when it is started with --cors ORIGIN. This server was not.
    st.check("without --cors, a CORS preflight is refused and no Access-Control-Allow-Origin is sent", r.status == 404 and "access-control-allow-origin" not in r.headers,
             f"HTTP {r.status}, headers {sorted(r.headers)}", repro(raw="OPTIONS /v1/chat/completions with Origin", expected="404 without Access-Control-Allow-Origin", actual=f"HTTP {r.status}"))
    r = http("GET", "/v1/models?limit=20")
    st.check("a query string on a known path is tolerated (GET /v1/models?limit=20)", r.status == 200, f"HTTP {r.status}",
             repro(raw="GET /v1/models?limit=20", expected="HTTP 200", actual=f"HTTP {r.status}"), warn_only=True)
    for field, val, probe in (("stop", ["3"], lambda c: "4" not in c.content), ("response_format", {"type": "json_object"}, None), ("logprobs", True, None)):
        p = req("Count from 1 to 6 separated by commas, nothing else.", 300, **{field: val})
        c = chat(p)
        if c.status == 200 and (probe is None or not probe(c)):
            st.warn(f"unsupported field '{field}' is silently ignored (HTTP 200) instead of being rejected",
                    f"README lists it as not supported; reply: {c.content[:60]!r}", repro(p, "HTTP 400 naming the unsupported field (or the field honoured)", c.brief()))
        elif c.status == 200:
            st.passed(f"field '{field}' is honoured")
        else:
            st.check(f"unsupported field '{field}' is rejected with 4xx", 400 <= c.status < 500, c.brief())
    # Documented: top_p is accepted and ignored (sampling is temperature only), because many clients always send it.
    c = chat(req("Count from 1 to 6 separated by commas, nothing else.", 300, top_p=0.1))
    st.check("top_p is accepted (documented as ignored)", c.status == 200, c.brief())
    p = req("Say hi.", 300, model="gpt-4o")
    c = chat(p)
    st.info("unknown model name", f"model='gpt-4o' -> HTTP {c.status}, reply names model {((c.json or {}).get('model'))!r}")

    # ---- still fine?
    c = chat(req("What is the capital of France? Answer in one word.", 300))
    st.check("after all the bad requests the server still answers correctly", c.status == 200 and "paris" in c.content.lower(), c.brief(), repro(req("What is the capital of France? Answer in one word.", 300), "Paris", c.brief()))


def cli_flags(st, ctx):
    """Invalid command lines must be refused at once with a message, not start a server (or crash later)."""
    import subprocess
    cases = [(["--slots", "5"], "more slots than supported"), (["--slots", "0"], "zero slots"), (["--ctx", "100"], "context below the minimum"),
             (["--spec", "9"], "speculation depth out of range"), (["--kv", "4"], "unsupported cache precision"), (["--reserve", "lots"], "unparsable size"),
             (["--port", "notaport"], "bad port"), (["--frobnicate"], "unknown option")]
    for flags, what in cases:
        cmd = [ctx.binary, "serve", "--port", str(PORT), *flags]
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            out = proc.communicate(timeout=6)[0].decode("utf-8", "replace")
            code = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()  # by PID: our own child
            out = proc.communicate()[0].decode("utf-8", "replace")
            code = "still running after 6 s (it was starting a server)"
        st.check(f"command line: {' '.join(flags)} ({what}) is refused with an error message", code not in (0,) and isinstance(code, int) and out.strip() != "" and "panicked" not in out,
                 f"exit {code}: {out.strip().splitlines()[-1][:160] if out.strip() else ''}",
                 repro(raw=" ".join(cmd), expected="non-zero exit and an error message, at once", actual=f"exit {code}: {out.strip()[-300:]}"))


def parts(suite):
    return [("shapes", "default", shapes), ("errors", "default", errors), ("cli", None, cli_flags)]
