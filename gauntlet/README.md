# Kuroshio gauntlet

A runnable test-and-benchmark suite that answers two questions about a Kuroshio build: **is it watertight**, and
**does it perform as the README claims**. It talks to the server only through its public interface (HTTP on port
8020, the command line, the server's log, `nvidia-smi`, `/proc`), needs Python 3 and nothing else, and never
changes anything outside its report directory and its work directory.

`scripts/verify.sh` and `scripts/fullcard.sh` cover the engine's internals and the "another program fills the card"
case; the gauntlet does not repeat them but can run them as an optional stage.

## Running it

```bash
python3 gauntlet/run.py --suite quick                      # about 10 minutes
python3 gauntlet/run.py --suite full                       # about 35 minutes
python3 gauntlet/run.py --suite quick --only api,perf      # some stages only
python3 gauntlet/run.py --suite full --with-scripts        # also scripts/verify.sh and scripts/fullcard.sh
```

| Option | Meaning |
|---|---|
| `--suite quick\|full` | `quick`: fewer prompts and repetitions, 2-minute soak, one agent task, contexts up to 48K. `full`: everything, 15-minute soak, six agent tasks, contexts up to 100K. |
| `--only a,b` | Stage names: `api`, `determinism`, `longctx`, `robustness`, `perf`, `memory`, `quality`, `agent`, `scripts`. |
| `--bin PATH` | Server binary. Default: `$KUROSHIO_BIN`, else `kuroshio` on `PATH`. |
| `--report DIR` | Where `report.md`, `report.json` and the server logs go. Default: `gauntlet/reports/<timestamp>/` (git-ignored). |
| `--work DIR` | Scratch directory for the agent tasks (`pi` runs model-written shell commands there). Default: a fresh temp directory. |
| `--with-scripts` | Run `scripts/verify.sh` and `scripts/fullcard.sh` (they test the repository's own build in `target/release`, not `--bin`). |
| `--soak-seconds N` | Override the soak duration. |

Exit code: `0` no stage failed (warnings allowed), `1` at least one stage FAILed, `2` the suite could not run
(binary missing, port 8020 taken by a server the suite did not start, GPU not free).

Requirements and behaviour:

- The suite starts and stops the server itself, **one at a time**, always on port 8020 (the `pi` agent is
  configured for that port). It stops servers by PID only and waits until the card is nearly empty (`nvidia-smi`
  below 3,000 MiB) before starting the next one. It refuses to run if something else is listening on 8020.
- Up to eight server configurations are used, each started once: default, `--ctx 4096`, `--spec 0`,
  `--no-prefix-cache`, `--kv 8 --dense 8`, `--slots 2 --reserve 6g`, `--idle-release 5` (`full` only),
  `--slots 4`. Parts of different stages that need the same server run back to back.
- Ctrl-C stops the server before exiting. A partial `report.md` is rewritten after every part.

## Reading the report

`report.md` starts with one table, a row per stage:

- **PASS**: every check passed. **WARN**: something a user should know (a README claim missed by more than 15%, an
  undocumented limitation, a model-quality miss) but nothing is broken. **FAIL**: a defect, or a claim missed by
  more than 35%. **SKIP**: the stage could not run, with the reason (never reported as PASS).
- A stage's status is the worst of its checks. `INFO` rows are observations and never affect the status.

Then: a table of README claims against measurements; every failure and warning with what was sent, what was
expected and what came back (a `curl` command when the request is small enough to print, otherwise a description
of how the request is built from `gauntlet/data.py`); every stage's numbers and checks; the servers that were
started and their log files (kept next to the report). `report.json` has the same content in machine-readable form,
plus the raw resource samples of the soak (`extra.soak_samples`) and per-task agent results (`extra.agent_tasks`).

A failing check named "the gauntlet itself raised an exception" is a bug in the suite or a reply so unexpected
that the suite could not interpret it; the traceback is in the detail.

## Stages

### 1. `api` - API conformance (`s1_api.py`)

- `/health`, `/v1/models`; non-streaming replies (id, object, created, model, choices, usage including
  `prompt_tokens_details.cached_tokens`); `finish_reason` `stop` / `length` / `tool_calls`; `max_tokens` and
  `max_completion_tokens`; `reasoning_content` against `content`, every `reasoning_effort` spelling; system and
  `developer` roles; content given as an array of parts.
- Streaming: every event is one `data:` line of valid UTF-8 JSON, one final chunk with `finish_reason` and usage,
  `[DONE]` last, nothing after it; reasoning deltas before content deltas; tool-call deltas; same text as
  non-streaming.
- Errors, each must give a 4xx within 5 s and leave `/health` green: malformed / empty / non-object JSON, JSON
  nested 100,000 levels deep, invalid UTF-8, missing or mistyped `messages`, unknown role, bad `temperature`, `n=2`, `max_tokens` 0 / -1 / string,
  wrong method, unknown path, no `Content-Length`, bad `Content-Length`, chunked body, 16 KB and 1 MB header
  lines, 100 KB request line, 500 headers, binary garbage, a body shorter than announced.
- Body limit: exactly 32 MiB is accepted; 32 MiB + 1 gives 413 whether or not the body is sent, and the server
  survives.
- Warn-only observations about things real clients do: whether an SSE event arrives in one piece or in many tiny
  packets, `Expect: 100-continue`, CORS preflight, query strings,
  empty `messages`, unsupported fields (`stop`, `top_p`, `response_format`, `logprobs`) being silently ignored,
  whether a stock client can read the 413.
- Command line: invalid options (`--slots 5`, `--ctx 100`, `--spec 9`, unknown flag, ...) must be refused at once.

### 2. `determinism` - determinism and correctness (`s2_determinism.py`)

- Temperature 0: the same request twice gives the same text; streaming equals non-streaming; the seed is
  ignored; a 900-token prompt sent twice is served from the cache and gives the same text.
- The same texts are produced by servers started with `--spec 0`, with `--no-prefix-cache`, with `--slots 4`, and
  with `--slots 4` while three other streams are generating.
- Seeded sampling (temperature 0.8): same seed, same text, also under `--spec 0` and under concurrency; another
  seed, another text; no seed, two different texts. Warn-only: a seed the server cannot use (negative,
  fractional, a string) should be rejected or honoured, not silently replaced by a random one.
- Multi-turn conversation: every follow-up reports cached tokens of at least the previous prompt, and all answers
  equal those of the `--no-prefix-cache` server.
- Unicode: emoji (ZWJ sequences, flags), CJK, combining marks, RTL, astral-plane characters round-trip; no SSE
  event contains broken UTF-8 or U+FFFD; a reply cut by `max_tokens` at each of its last tokens never ends in a
  broken character.
- Tool calling: 26 prompts over 25 tools (1-3 declared; nested objects, arrays of objects, arrays of arrays,
  numbers, booleans, enums, empty arguments, no `parameters` key, Unicode and RTL arguments, quotes, newlines,
  odd and long tool names, system prompt, prior history, two calls in one reply), every third through the
  streaming path. A reply counts as *well-formed* if it has `tool_calls`, each with an id, type `function`, a
  declared name and arguments that parse to a JSON object, and `finish_reason` is `tool_calls`. Separately
  (warn-only) whether the expected tool and argument values were chosen. Full loops (call, result, answer) and
  two calls in a row, streaming and not; `tool_choice: "none"`; ids unique across the stage including four
  simultaneous requests.

### 3. `longctx` - long context (`s3_longctx.py`)

- One document per size (about 4K, 16K, 48K tokens; 100K in `full`) with three needles (start, middle, end). The
  first question comes with the document and is timed uncached (prompt tokens/s, time to first token); the other
  two are follow-up turns of the same conversation and must be served from the cache (at least 98% of the prompt
  cached, first token at least 10x sooner and within 1.5 s). Which needle is asked first rotates with the size.
- Warn-only: the same document with a *different last question as a new conversation* - does the shared prefix
  help? (It does not: the cache needs the whole previous prompt to be a prefix of the new one.)
- A prompt 6% over the limit reported by `/v1/models` must get a 400 that names the sizes, quickly, also with
  `stream: true`; `max_tokens` far beyond the room is clamped.
- With `--ctx 4096`: a generation without `max_tokens` runs into the end of the context and ends with `length`;
  a conversation is grown a few tokens per step across the limit - every accepted step stays within the context,
  then a clean 400, and the server stays healthy.

### 4. `robustness` (`s4_robustness.py`, 4 slots)

- Clients that leave: mid-stream, while a 30K-token prompt is being read, during a non-streaming generation, and
  while waiting in the queue. After each, a new request must get its first token within 2 s.
- A slow reader (256 bytes every 10 ms through a 4 KB receive buffer) gets the complete identical stream and does
  not slow another stream down.
- Hostile input: user text that spells out each of the model's control tokens (`<|end_message|>`,
  `<|content_image|>`, ...) must give a well-formed 200 or 400; warn-only, whether such text is tokenised as the
  control token itself, and whether a tool result that spells out `<|end_message|><|message_system|>...` gets its
  forged system message obeyed. Sampling at temperature 2.0 (near-random
  tokens, with and without tools, streaming and not) must still give well-formed replies. Warn-only: a client
  that half-closes its socket after sending the request must still get its answer.
- 300 connections opened at once: about 44 get an immediate 503, the others stay usable, threads and file
  descriptors return to the idle level; `full` also checks that silent and half-sent connections are dropped
  after the server's 15 s read timeout.
- 14 simultaneous requests on 4 slots: each gets its own answer.
- A burst of 40 mixed requests (short, long, streaming or not, long prompts, cancelled, invalid, tool calls):
  each accounted for; afterwards the server gives the same answer to a reference prompt as before.
- Soak: 6 clients on 4 slots for 2 minutes (`quick`) or 15 minutes (`full`) with a mixed, seeded workload, while
  GPU memory (process and card), RSS, thread count and open file descriptors are sampled. Every request must get
  its expected outcome; trends are judged as described under Thresholds; the log must not contain `panicked`,
  `CUDA error` or `Aborted`.

### 5. `perf` - throughput and latency (`s5_perf.py`)

Everything is measured at the client from a streaming request with `reasoning_effort: "none"`:

- **decode tokens/s** = (`usage.completion_tokens` - 1) / (arrival of the last chunk - arrival of the first chunk
  that carries text). The first token is excluded because its latency is prompt processing.
- **time to first token** = arrival of the first text chunk - moment the request was sent.
- **prompt tokens/s** = `usage.prompt_tokens` / time to the first chunk, for a never-seen document with
  `max_tokens: 1` (nothing cached).
- **aggregate** = completion tokens of N simultaneous requests / wall time from the first send to the last final
  chunk.

This is the method of the benchmark the README's numbers come from (`bench.py`), with the same six-prompt panel.
The README's one-stream figure is a median over the first three prompts, so the comparison uses those three, each
with fixed seeds (2 in `quick`, 3 in `full`), 512 tokens per run. Measured: one stream at temperature 0, 0.6 and
1.0; `--spec 0`; `--kv 8 --dense 8`; short-prompt time to first token; prompt processing at 4K and 16K; a
synthetic agent loop's follow-up time to first token; 2 and 4 simultaneous streams on 4 slots with per-stream
fairness; 8 clients on 4 slots (throughput, worst wait). The `--spec 0` runs use the same seeds, so the benchmark
doubles as a check that seeded sampling gives the same 512-token texts with and without speculation.

### 6. `memory` (`s6_memory.py`)

- GPU memory of the server right after start (against "about 20 GiB"); after a short conversation; after a
  30K-token conversation (KB per token against "about 40 KB"); after a short conversation replaced it (must fall
  back). Context sizes reported by `/v1/models` against the README's 240K / 205K / 450K.
- 4 slots: one long and three short conversations at once (24K + 3x3K in `quick`, 100K + 3x15K in `full`), each a
  needle question that must be answered.
- `--slots 2 --reserve 6g`: at least 6 GiB free at idle; two simultaneous prompts of 62% of the context each -
  one must be served, the other refused with 503 "out of GPU memory", free memory never below 6 GiB (sampled
  every 0.4 s), server alive and serving afterwards. Warn-only (`full`): what a streaming client sees in that case.
- `--idle-release 5` (`full`): after a 20K-token conversation and 9 s of idleness the memory is back at the
  start-up level, and the conversation then continues correctly with nothing cached.

### 7. `quality` - regression guard (`s7_quality.py`, prompts in `data.py`)

56 prompts with one right answer: arithmetic (12), string manipulation (12), simple logic (10), format adherence
(10), facts (12). Temperature 0, `reasoning_effort: "low"`, run at default settings and with `--kv 8 --dense 8`.

Scoring rules: only `content` counts (thinking is ignored). The answer is trimmed, one trailing full stop and one
surrounding pair of quotes, backticks or `**` are removed, whitespace runs are collapsed, and it is lower-cased
(except for the two case-conversion questions and the chemical symbol). Then:

- text answers must **equal** the expected string - "Paris" passes, "The capital is Paris" fails, because every
  prompt asks for only the answer;
- numbers must be the number and nothing else (thousands separators are accepted, units are not);
- JSON answers must parse (one surrounding code fence is accepted) and equal the expected value;
- comma-separated lists are compared with all whitespace removed;
- a reply that hit the token limit or returned an error scores 0.

### 8. `agent` - agent runs with pi (`s8_agent.py`)

The `pi` coding agent (`pi -p --no-session --no-extensions --no-mcp --no-skills --no-context-files --provider
kuroshio --model inkling-2-mini --thinking low "<task>" < /dev/null`) is given tiny tasks, each in its own fresh
directory: fix a one-line bug, add a function for a provided test, rename a function across two files, write a
script that produces an exact output file, answer a question that needs two files, fix an off-by-one that a
unittest exposes (`quick` runs only the first). Afterwards the gauntlet runs each task's check itself and verifies
the files the task said not to touch. Reported: tasks solved, wall time, number of model requests and share of
prompt tokens served from the cache (both from the server log's `request:` lines), server-side time to first token
of cached requests. SKIP if `pi` is not installed.

### 9. `scripts` (optional, `s9_scripts.py`)

`scripts/verify.sh` (`full` in the full suite) and `scripts/fullcard.sh`, with no server of the gauntlet running;
each must exit 0. Their output is kept next to the report (`verify.out`, `fullcard.out`). Takes about 3 minutes.

### Always: server logs

Every log of every server started in the run is searched for `panicked`, `CUDA error` and `Aborted`, and a server
that exits during a part fails that part with its last log lines.

## Thresholds

Every threshold is a named constant with its reasoning at the top of the stage's module. The important ones:

| What | WARN | FAIL | Why |
|---|---|---|---|
| A measurement against a README figure (`s5_perf.WARN_FRAC`, `FAIL_FRAC`) | more than 15% worse | more than 35% worse | Single runs vary about 5% with the text; 15% is outside noise, 35% means the claim is not met. |
| Short-prompt time to first token | > 100 ms | > 500 ms | The engine's own log shows 15-30 ms. |
| Fairness, slowest / fastest of simultaneous streams | < 0.4 | - | Streams legitimately differ with their text (the README's own run: 0.64); only starvation is flagged. |
| Slot freed after clients leave (`SLOT_FREE_MAX_S`) | - | first token later than 2 s | A free slot answers in under 100 ms and the server polls for vanished clients every 500 ms; abandoned work that keeps running costs tens of seconds. |
| Soak: RSS growth after warm-up | > 40 MiB | > 100 MiB and still rising | The load is stationary; the first 25% of the soak is warm-up; growth is a fitted linear trend over the rest. |
| Soak: GPU memory growth after warm-up | > 256 MiB | > 768 MiB and still rising | Slots keep what they have grown to, so steps are fine and trends are not. |
| Threads / file descriptors after load | more than 2 above idle | more than 8 above idle | One thread and one socket per connection must go when the connection goes. |
| Cached follow-up (long context) | - | < 98% cached, or first token later than min(1.5 s, 10% of uncached) | "Follow-up turns answer in milliseconds". |
| Needle retrieval | any miss | < 80% | Needles are plain sentences; misses are model quality until they are frequent. |
| Tool cases answered with a well-formed call | < 95% | < 85%, or any malformed call | README: 24 of 24. |
| Memory returned when a short conversation replaces a long one | - | < 60% returned, or > 640 MiB above the starting level | The server keeps an 8,192-token hysteresis (about 320 MiB). |
| `--reserve 6g` | - | less than 6 GiB - 150 MiB free at any sample | 150 MiB covers `nvidia-smi` granularity and the desktop's own fluctuations. |
| Quality score, default settings | < 95% | < 85% | The baseline build scores 56/56. |
| Quality, 8-bit against default | more than 5 points lower | - | The 8-bit modes are documented as lossy. |
| Agent tasks solved | < 66% | <= 34% | The tasks are one-line fixes. |
| Agent prompt tokens from the cache | < 50% | - | The prefix cache is what makes an agent loop cheap. |

## Files

| File | Content |
|---|---|
| `run.py` | Command line, server configurations, stage scheduling, exit code. |
| `core.py` | Server start/stop by PID, raw-socket HTTP and SSE client, GPU and `/proc` sampling, check bookkeeping. |
| `data.py` | Prompts, document generator, tool definitions and cases, the quality set and its scoring. |
| `report.py` | `report.md` and `report.json`. |
| `s1_api.py` ... `s9_scripts.py` | One module per stage. |

## Safety notes

- Model output is only ever compared, parsed as JSON, or written to the report; the suite never executes it. The
  one exception is by design: the `agent` stage lets `pi` run model-written commands inside the task directory,
  and then runs the task's own test there with a timeout.
- Nothing is sent anywhere but 127.0.0.1:8020.
