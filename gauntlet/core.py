"""Shared plumbing for the gauntlet: server lifecycle, a raw-socket HTTP/SSE client, result bookkeeping.

Python standard library only. Everything that talks to the server goes through raw sockets on purpose: the
suite has to observe exact bytes (SSE framing, UTF-8 boundaries), arrival times, and has to be able to misbehave
(disconnect mid-stream, read slowly, send broken requests), none of which urllib lets you do.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time

HOST = "127.0.0.1"
PORT = 8020  # the pi provider "kuroshio" is configured for this port

# How long to wait for the server to print "listening" (it loads 19.5 GB of weights; ~8 s when the page cache is
# warm, a minute or more from a cold disk).
START_TIMEOUT_S = 180
# The card must be (nearly) free before another server is started: two servers do not fit in 32 GiB.
GPU_FREE_BELOW_MIB = 3000
GPU_FREE_TIMEOUT_S = 90

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"
_RANK = {INFO: 0, PASS: 1, SKIP: 1, WARN: 2, FAIL: 3}


def now():
    return time.monotonic()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pysleep(seconds):
    time.sleep(seconds)


# --------------------------------------------------------------------------------------------- GPU / process
def gpu_used_mib():
    """Total memory in use on the card (all programs), MiB. None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def gpu_total_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def gpu_proc_mib(pid):
    """GPU memory attributed to one process, MiB (None if it is not listed)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            p, m = [x.strip() for x in line.split(",")[:2]]
            if int(p) == pid:
                return int(m)
    except Exception:
        pass
    return None


def proc_stats(pid):
    """RSS (MiB), thread count and open file descriptors of a process, from /proc."""
    d = {"rss_mib": None, "threads": None, "fds": None}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    d["rss_mib"] = int(line.split()[1]) / 1024.0
                elif line.startswith("Threads:"):
                    d["threads"] = int(line.split()[1])
        d["fds"] = len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        pass
    return d


def port_in_use(port=PORT):
    s = socket.socket()
    s.settimeout(1)
    try:
        s.connect((HOST, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# --------------------------------------------------------------------------------------------- server
class ServerError(Exception):
    pass


class Server:
    """One running `kuroshio serve` process started by the suite."""

    def __init__(self, binary, args, log_path, port=PORT):
        self.binary, self.args, self.log_path, self.port = binary, tuple(args), log_path, port
        self.proc = None
        self.started_at = None
        self.load_s = None
        self.gpu_before = None

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def cmdline(self):
        return " ".join([self.binary, "serve", "--port", str(self.port), *self.args])

    def start(self):
        t0 = now()
        while True:
            used = gpu_used_mib()
            if used is None or used < GPU_FREE_BELOW_MIB:
                break
            if now() - t0 > GPU_FREE_TIMEOUT_S:
                raise ServerError(f"GPU still has {used} MiB in use after {GPU_FREE_TIMEOUT_S}s; refusing to start a "
                                  f"second 20 GiB server (need < {GPU_FREE_BELOW_MIB} MiB). Is another server running?")
            pysleep(0.5)
        t0 = now()
        while port_in_use(self.port):  # a server that was just stopped may hold the port for a moment
            if now() - t0 > 10:
                raise ServerError(f"port {self.port} is already in use by another program; the gauntlet will not kill "
                                  f"a server it did not start. Stop it and run again.")
            pysleep(0.5)
        self.gpu_before = gpu_used_mib()
        logf = open(self.log_path, "wb")
        self.proc = subprocess.Popen([self.binary, "serve", "--port", str(self.port), *self.args],
                                     stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT)
        logf.close()
        self.started_at = now()
        while True:
            if self.proc.poll() is not None:
                raise ServerError(f"server exited with code {self.proc.returncode} during start-up: {self.cmdline()}\n"
                                  + self.log_tail(15))
            if "listening" in self.log_text():
                try:
                    r = http("GET", "/health", timeout=3, port=self.port)
                    if r.status == 200:
                        break
                except OSError:
                    pass
            if now() - self.started_at > START_TIMEOUT_S:
                self.stop()
                raise ServerError(f"server did not become ready within {START_TIMEOUT_S}s: {self.cmdline()}\n" + self.log_tail(15))
            pysleep(0.2)
        self.load_s = now() - self.started_at
        pysleep(0.5)
        self.gpu_idle_total, self.gpu_idle_proc = gpu_used_mib(), gpu_proc_mib(self.pid)
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        """Stop by PID (never by name pattern) and wait until it is gone."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
            except Exception:
                pass
        t0 = now()
        while now() - t0 < GPU_FREE_TIMEOUT_S:
            used = gpu_used_mib()
            if used is None or used < GPU_FREE_BELOW_MIB:
                break
            pysleep(0.5)

    def log_text(self):
        try:
            with open(self.log_path, "rb") as f:
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def log_size(self):
        try:
            return os.path.getsize(self.log_path)
        except OSError:
            return 0

    def log_since(self, offset):
        try:
            with open(self.log_path, "rb") as f:
                f.seek(offset)
                return f.read().decode("utf-8", "replace")
        except OSError:
            return ""

    def log_tail(self, n=10):
        return "\n".join(self.log_text().splitlines()[-n:])

    def request_lines(self, offset=0):
        """Parsed 'request:' lines of the server log written after `offset` bytes."""
        import re
        out = []
        for line in self.log_since(offset).splitlines():
            m = re.match(r"request: (\d+) prompt tokens \((\d+) cached\), ttft (\d+) ms; (\d+) tokens at ([\d.]+) tok/s;.*finish=(\w+)", line)
            if m:
                out.append({"prompt": int(m[1]), "cached": int(m[2]), "ttft_ms": int(m[3]), "tokens": int(m[4]),
                            "tok_s": float(m[5]), "finish": m[6]})
        return out

    def bad_log_lines(self):
        bad = ("panicked", "CUDA error", "Aborted")
        return [ln for ln in self.log_text().splitlines() if any(b in ln for b in bad)]


class ServerManager:
    """Keeps at most one server alive; restarts it only when a part needs a different command line."""

    def __init__(self, binary, log_dir):
        self.binary, self.log_dir = binary, log_dir
        self.server = None
        self.n = 0
        self.history = []  # (args, log file, load seconds)

    def ensure(self, args):
        args = tuple(args)
        if self.server and self.server.alive() and self.server.args == args:
            return self.server
        self.stop()
        self.n += 1
        tag = "default" if not args else "".join(c if c.isalnum() else "_" for c in "_".join(args)).strip("_")
        path = os.path.join(self.log_dir, f"server-{self.n:02d}-{tag}.log")
        srv = Server(self.binary, args, path)
        log(f"starting server: {srv.cmdline()}")
        srv.start()
        self.server = srv
        self.history.append({"args": list(args), "log": os.path.basename(path), "load_s": round(srv.load_s, 1), "pid": srv.pid})
        log(f"  ready in {srv.load_s:.1f}s (pid {srv.pid})")
        return srv

    def restart(self):
        args = self.server.args if self.server else ()
        self.stop()
        return self.ensure(args)

    def stop(self):
        if self.server:
            self.server.stop()
            self.server = None


# --------------------------------------------------------------------------------------------- HTTP client
class Resp:
    def __init__(self, status, headers, body, raw, t_start, t_first, t_end):
        self.status, self.headers, self.body, self.raw = status, headers, body, raw
        self.t_start, self.t_first, self.t_end = t_start, t_first, t_end

    def json(self):
        try:
            return json.loads(self.body.decode("utf-8"))
        except Exception:
            return None

    def error_message(self):
        j = self.json()
        if isinstance(j, dict) and isinstance(j.get("error"), dict):
            return str(j["error"].get("message"))
        return None

    def text(self, limit=300):
        return self.body.decode("utf-8", "replace")[:limit]


def _parse_head(raw):
    """Split raw response bytes into (status, headers, body). Status 0 if there is no parsable status line."""
    head, sep, body = raw.partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    status = 0
    try:
        parts = lines[0].split(b" ", 2)
        if parts[0].startswith(b"HTTP/"):
            status = int(parts[1])
    except Exception:
        status = 0
    headers = {}
    for ln in lines[1:]:
        k, _, v = ln.partition(b":")
        headers[k.decode("latin-1").strip().lower()] = v.decode("latin-1").strip()
    return status, headers, body if sep else b""


def raw_request(data, timeout=30, port=PORT, shutdown_write=False, chunks=None):
    """Send bytes, read until the server closes. Returns Resp (status 0 = no HTTP response came back).
    `chunks`: optional list of (bytes, pause_seconds) sent instead of `data`."""
    t0 = now()
    s = socket.create_connection((HOST, port), timeout=timeout)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    raw, t_first, send_error = b"", None, None
    try:
        try:
            if chunks is not None:
                for part, pause in chunks:
                    s.sendall(part)
                    if pause:
                        pysleep(pause)
            else:
                s.sendall(data)
            if shutdown_write:
                s.shutdown(socket.SHUT_WR)
        except OSError as e:  # the server may answer and close before the whole request is sent
            send_error = e
        while True:
            try:
                b = s.recv(65536)
            except socket.timeout:
                break
            except OSError as e:
                if send_error is None:
                    send_error = e
                break
            if not b:
                break
            if t_first is None:
                t_first = now()
            raw += b
    finally:
        s.close()
    status, headers, body = _parse_head(raw)
    r = Resp(status, headers, body, raw, t0, t_first, now())
    r.send_error = send_error
    return r


def build_request(method, path, body=None, headers=None, content_length=True):
    h = {"Host": f"{HOST}:{PORT}"}
    if body is not None:
        h["Content-Type"] = "application/json"
        if content_length:
            h["Content-Length"] = str(len(body))
    if headers:
        h.update(headers)
    head = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
    return head.encode("latin-1") + (body or b"")


def http(method, path, body=None, headers=None, timeout=30, port=PORT):
    if isinstance(body, (dict, list)):
        body = json.dumps(body).encode()
    elif isinstance(body, str):
        body = body.encode()
    return raw_request(build_request(method, path, body, headers), timeout=timeout, port=port)


def healthy(timeout=5):
    try:
        r = http("GET", "/health", timeout=timeout)
        return r.status == 200 and (r.json() or {}).get("status") == "ok"
    except OSError:
        return False


class Chat:
    """Result of one chat completion (streaming or not), as seen by the client."""

    def __init__(self):
        self.status = 0
        self.error = None  # error message from an HTTP error body or an in-stream error event
        self.stream_error = None  # error delivered inside a 200 SSE stream
        self.content = ""
        self.reasoning = ""
        self.tool_calls = []
        self.finish_reason = None
        self.usage = None
        self.headers = {}
        self.raw = b""
        self.events = []  # streaming: (arrival time, parsed JSON)
        self.framing = []  # SSE / UTF-8 violations found while parsing
        self.done = False  # saw data: [DONE]
        self.t_start = self.t_first = self.t_last = self.t_end = None
        self.exception = None
        self.json = None  # non-streaming: the response object
        self.cancelled = False

    @property
    def ok(self):
        return self.status == 200 and self.error is None and self.exception is None

    @property
    def prompt_tokens(self):
        return (self.usage or {}).get("prompt_tokens")

    @property
    def completion_tokens(self):
        return (self.usage or {}).get("completion_tokens")

    @property
    def cached_tokens(self):
        return ((self.usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")

    @property
    def ttft(self):
        """Seconds from sending the request to the first token-bearing chunk (streaming) or the response (not)."""
        if self.t_first is None or self.t_start is None:
            return None
        return self.t_first - self.t_start

    @property
    def wall(self):
        return None if self.t_end is None else self.t_end - self.t_start

    @property
    def decode_tps(self):
        """Tokens/s while generating, measured at the client: (completion_tokens - 1) / (arrival of the final chunk
        - arrival of the first token-bearing chunk). The first token is excluded because its time is prompt
        processing. Streaming only."""
        n = self.completion_tokens
        if not n or n < 2 or self.t_first is None or self.t_last is None or self.t_last <= self.t_first:
            return None
        return (n - 1) / (self.t_last - self.t_first)

    def text(self):
        return (self.reasoning or "") + "\x1f" + (self.content or "")

    def brief(self):
        if self.exception:
            return f"exception {self.exception!r}"
        if self.status != 200:
            return f"HTTP {self.status}: {self.error or self.raw[:200]!r}"
        return (f"HTTP 200 finish={self.finish_reason} usage={self.usage} content={self.content[:160]!r}"
                + (f" tool_calls={json.dumps(self.tool_calls)[:300]}" if self.tool_calls else "")
                + (f" stream_error={self.stream_error!r}" if self.stream_error else ""))


def _absorb_delta(c, obj, t):
    if isinstance(obj.get("error"), dict):
        c.stream_error = str(obj["error"].get("message"))
        c.error = c.stream_error
        return
    if obj.get("usage"):
        c.usage = obj["usage"]
    for ch in obj.get("choices") or []:
        d = ch.get("delta") or {}
        got = False
        if d.get("content"):
            c.content += d["content"]
            got = True
        if d.get("reasoning_content"):
            c.reasoning += d["reasoning_content"]
            got = True
        if d.get("tool_calls"):
            for tc in d["tool_calls"]:
                c.tool_calls.append(tc)
            got = True
        if got:
            if c.t_first is None:
                c.t_first = t
            c.t_last = t
        if ch.get("finish_reason") is not None:
            c.finish_reason = ch["finish_reason"]
            c.t_last = t


def chat(payload, timeout=300, port=PORT, on_event=None, read_size=65536, read_pause=0.0, rcvbuf=None,
         cancel_after_events=None, cancel_after_s=None, extra_headers=None, cancel_event=None):
    """POST /v1/chat/completions. Handles both streaming and non-streaming replies.

    cancel_after_events / cancel_after_s / cancel_event (a threading.Event, checked at each streamed chunk): close
    the connection early (a client that walks away).
    read_size / read_pause / rcvbuf: emulate a slow reader.
    Never raises for network errors: they are recorded in `.exception`.
    """
    c = Chat()
    body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    data = build_request("POST", "/v1/chat/completions", body, extra_headers)
    c.t_start = now()
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if rcvbuf:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        s.settimeout(timeout)
        s.connect((HOST, port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.sendall(data)
        buf, head_done, is_sse, pos = b"", False, False, 0
        deadline = c.t_start + timeout
        while True:
            if cancel_after_s is not None:
                left = c.t_start + cancel_after_s - now()
                if left <= 0:
                    c.cancelled = True
                    break
                s.settimeout(max(0.01, min(left, timeout)))
            try:
                b = s.recv(read_size)
            except socket.timeout:
                if cancel_after_s is not None and now() - c.t_start >= cancel_after_s:
                    c.cancelled = True
                    break
                if now() >= deadline:
                    raise
                continue
            t = now()
            if not b:
                break
            buf += b
            if not head_done and b"\r\n\r\n" in buf:
                head_done = True
                c.status, c.headers, _ = _parse_head(buf)
                is_sse = c.headers.get("content-type", "").startswith("text/event-stream")
                pos = buf.index(b"\r\n\r\n") + 4
            if head_done and is_sse:
                while True:
                    end = buf.find(b"\n\n", pos)
                    if end < 0:
                        break
                    ev, pos = buf[pos:end], end + 2
                    if c.done:
                        c.framing.append(f"event after [DONE]: {ev[:80]!r}")
                    try:
                        ev_text = ev.decode("utf-8")
                    except UnicodeDecodeError as e:
                        c.framing.append(f"event is not valid UTF-8 ({e}): {ev[:120]!r}")
                        continue
                    if not ev_text.startswith("data: "):
                        c.framing.append(f"event does not start with 'data: ': {ev_text[:80]!r}")
                        continue
                    if "\n" in ev_text:
                        c.framing.append(f"event has more than one line: {ev_text[:80]!r}")
                    d = ev_text[6:]
                    if d == "[DONE]":
                        c.done = True
                        continue
                    try:
                        obj = json.loads(d)
                    except ValueError:
                        c.framing.append(f"event data is not JSON: {d[:120]!r}")
                        continue
                    c.events.append((t, obj))
                    _absorb_delta(c, obj, t)
                    if on_event:
                        on_event(c, obj)
                    if (cancel_after_events is not None and len(c.events) >= cancel_after_events) or (cancel_event is not None and cancel_event.is_set()):
                        c.cancelled = True
                        break
                if c.cancelled:
                    break
            if read_pause:
                pysleep(read_pause)
        c.t_end = now()
        c.raw = buf
        if not c.cancelled:
            if head_done and is_sse:
                if buf[pos:].strip():
                    c.framing.append(f"trailing bytes after the last complete event: {buf[pos:][:80]!r}")
                if not c.done:
                    c.framing.append("stream ended without data: [DONE]")
            elif head_done:
                bodyb = buf[buf.index(b"\r\n\r\n") + 4:]
                c.t_first = c.t_last = c.t_end
                try:
                    j = json.loads(bodyb.decode("utf-8"))
                except Exception as e:
                    c.framing.append(f"body is not UTF-8 JSON: {e}")
                    j = None
                c.json = j
                if isinstance(j, dict):
                    if isinstance(j.get("error"), dict):
                        c.error = str(j["error"].get("message"))
                    c.usage = j.get("usage")
                    try:
                        ch = j["choices"][0]
                        m = ch.get("message") or {}
                        c.content = m.get("content") or ""
                        c.reasoning = m.get("reasoning_content") or ""
                        c.tool_calls = m.get("tool_calls") or []
                        c.finish_reason = ch.get("finish_reason")
                    except Exception:
                        pass
                if c.status != 200 and c.error is None:
                    c.error = bodyb[:300].decode("utf-8", "replace")
            else:
                c.exception = "connection closed without a response"
    except Exception as e:  # noqa: BLE001 - recorded, the caller decides
        c.exception = f"{type(e).__name__}: {e}"
        c.t_end = now()
    finally:
        if s is not None:
            try:
                s.close()
            except OSError:
                pass
    return c


def ask(prompt, max_tokens=64, temperature=0, effort="none", stream=False, timeout=300, **extra):
    """Convenience: one user message."""
    p = {"model": "inkling-2-mini", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
         "temperature": temperature, "stream": stream}
    if effort is not None:
        p["reasoning_effort"] = effort
    p.update(extra)
    return chat(p, timeout=timeout)


def run_parallel(fns, stagger=0.0):
    """Run callables in threads; returns their results in order (exceptions are returned, not raised)."""
    res = [None] * len(fns)

    def wrap(i, fn):
        try:
            res[i] = fn()
        except Exception as e:  # noqa: BLE001
            res[i] = e

    ts = []
    for i, fn in enumerate(fns):
        th = threading.Thread(target=wrap, args=(i, fn), daemon=True)
        th.start()
        ts.append(th)
        if stagger:
            pysleep(stagger)
    for th in ts:
        th.join()
    return res


# --------------------------------------------------------------------------------------------- results
def shorten(obj, limit=600):
    """JSON-ish rendering of a request/response for the report, with long strings elided."""
    def cut(o):
        if isinstance(o, str):
            return o if len(o) <= 240 else f"{o[:120]}...<{len(o)} chars>...{o[-60:]}"
        if isinstance(o, list):
            return [cut(x) for x in o[:12]] + ([f"...<{len(o) - 12} more>"] if len(o) > 12 else [])
        if isinstance(o, dict):
            return {k: cut(v) for k, v in o.items()}
        return o
    try:
        s = json.dumps(cut(obj), ensure_ascii=False)
    except TypeError:
        s = str(obj)
    return s if len(s) <= limit * 4 else s[:limit * 4] + "..."


class Stage:
    """Collects the checks and key numbers of one stage. A stage's status is the worst of its checks."""

    def __init__(self, key, title):
        self.key, self.title = key, title
        self.checks = []
        self.numbers = {}
        self.notes = []
        self.skip_reason = None
        self.seconds = 0.0
        self.server_args = None  # filled by the runner for reproduction lines
        self.part = None

    def _add(self, status, name, detail="", repro=None):
        rec = {"name": name, "status": status, "detail": str(detail) if detail is not None else "", "part": self.part,
               "server": " ".join(self.server_args) if self.server_args is not None else None}
        if repro is not None:
            rec["repro"] = repro
        self.checks.append(rec)
        if status in (FAIL, WARN):
            log(f"    {status}: {name}: {rec['detail'][:300]}")
        return status == PASS

    def check(self, name, cond, detail="", repro=None, warn_only=False):
        """Record PASS if cond else FAIL (or WARN with warn_only). Returns bool(cond)."""
        if cond:
            self._add(PASS, name, detail)
            return True
        self._add(WARN if warn_only else FAIL, name, detail, repro)
        return False

    def passed(self, name, detail=""):
        self._add(PASS, name, detail)

    def fail(self, name, detail="", repro=None):
        self._add(FAIL, name, detail, repro)

    def warn(self, name, detail="", repro=None):
        self._add(WARN, name, detail, repro)

    def info(self, name, detail=""):
        self._add(INFO, name, detail)

    def skip(self, name, reason):
        self._add(SKIP, name, reason)

    def num(self, key, value, unit="", head=False):
        """Record a measurement. head=True: also shown in the one-line summary of the stage."""
        self.numbers[key] = {"value": value, "unit": unit, "head": head}

    def threshold(self, name, value, claim, warn_frac, fail_frac, unit="", higher_is_better=True, source=""):
        """Compare a measurement with a claimed figure: WARN if more than warn_frac worse, FAIL if more than
        fail_frac worse. Returns the status."""
        if value is None:
            self._add(FAIL, name, f"no measurement (claim: {claim} {unit})")
            return FAIL
        ratio = value / claim if higher_is_better else claim / value
        delta = (value - claim) / claim * 100.0
        detail = f"measured {value:.4g} {unit} vs claimed {claim:g} {unit} ({delta:+.1f}%)" + (f"; {source}" if source else "")
        if ratio < 1 - fail_frac:
            st = FAIL
        elif ratio < 1 - warn_frac:
            st = WARN
        else:
            st = PASS
        self._add(st, name, detail, {"expected": f"within {warn_frac:.0%} of {claim:g} {unit}", "actual": f"{value:.4g} {unit}"} if st != PASS else None)
        return st

    @property
    def status(self):
        real = [c for c in self.checks if c["status"] not in (INFO, SKIP)]
        if not real:
            return SKIP  # nothing was verified: never report that as PASS
        return max((c["status"] for c in real), key=lambda s: _RANK[s])

    def counts(self):
        d = {PASS: 0, WARN: 0, FAIL: 0, SKIP: 0, INFO: 0}
        for c in self.checks:
            d[c["status"]] += 1
        return d


def curl_for(payload, path="/v1/chat/completions"):
    """A copy-pastable reproduction for small requests."""
    body = json.dumps(payload, ensure_ascii=False)
    if len(body) > 1500:
        return None
    return f"curl -sS http://{HOST}:{PORT}{path} -H 'Content-Type: application/json' -d '{body.replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'"


def repro(payload=None, expected=None, actual=None, note=None, raw=None):
    """Reproduction record for the report. `payload` is either a real request body (has "messages": rendered as
    JSON and, if small, as a curl command) or a {"note": ...} description of a request too large to print."""
    r = {}
    if isinstance(payload, dict) and "messages" in payload:
        r["request"] = shorten(payload)
        c = curl_for(payload)
        if c:
            r["curl"] = c
    elif isinstance(payload, dict) and payload:
        r["raw_request"] = "; ".join(f"{k}: {v}" if k != "note" else str(v) for k, v in payload.items())
    if raw is not None:
        r["raw_request"] = raw
    if expected is not None:
        r["expected"] = expected
    if actual is not None:
        r["actual"] = actual if isinstance(actual, str) else shorten(actual)
    if note:
        r["note"] = note
    return r


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def fmt(x, nd=1):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)
