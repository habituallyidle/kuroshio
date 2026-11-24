<p align="center">
  <img src="assets/logo.png" width="160" alt="Kuroshio logo: a dark river in pixel art">
</p>

<h1 align="center">Kuroshio 黒潮</h1>

<p align="center"><em>The "black current": the fast, warm stream off Japan's Pacific coast. Dark as ink, and full of squid.</em></p>

**Run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on your RTX 5090, at the highest
single-stream speed measured for it.** Kuroshio is an inference engine built for exactly one model and one GPU:
hand-written CUDA kernels for Blackwell's tensor cores behind an OpenAI-compatible server. One binary, nothing to
configure.

- **Faster than vLLM** on one stream (+13%), two (+20%) and four (+10%), and twice as fast as Strata
- **Reads prompts at 8,000 tok/s**: ahead of both up to about 20K tokens; follow-up turns answer in milliseconds
- **The longest conversations on the card**: 211K tokens lossless, 358K with an 8-bit cache, where vLLM holds
  132K (8-bit) and Strata 32K; all of them read back correctly at that length
- **Tool calls that parse**: 24 of 24, where vLLM manages 16 and Strata 2
- **Shares the card**: takes memory as conversations grow, and survives another program filling the GPU
- **Not the fastest at everything**: vLLM reads a 34K-token prompt 16% faster, and with its 8-bit cache it is ahead
  beyond 4 streams (see below)

## Headline

Same RTX 5090, same afternoon, same harness (`bench.py`, temperature 0.6, 512-token completions); every cell is
the median of 10 runs. **Every request sets `reasoning_effort: "high"`**: left unset, or at `"none"`, each engine
falls back to a default of its own, so the engines would not be doing the same work. vLLM is the official
`vllm/vllm-openai:inkling-2-mini` image with an 8-bit KV cache (how it is usually run on this card), with and
without its DSpark drafter; Strata is 0.2.19 with its `inkling-2-mini-sq-20` pack and DSpark drafting; Kuroshio runs
at its defaults (lossless 16-bit cache).

| Decode, tok/s in total | Kuroshio | vLLM + DSpark | vLLM, no drafter | Strata |
|---|---:|---:|---:|---:|
| 1 stream | **250** | 220 | 167 | 122 |
| 2 streams | **419** | 349 | 236 | 180 |
| 4 streams | **559** | 509 | 388 | 273 |
| 8 streams † | 761 | **791** | 661 | 390 |
| 16 streams † | 1,062 | **1,307** | 1,068 | did not finish ‡ |

| Prompt processing, tok/s (text never seen before) | Kuroshio | vLLM | Strata |
|---|---:|---:|---:|
| 4.7K tokens | **8,440** | 2,600 | 7,470 |
| 19K tokens | **8,290** | 7,050 | 6,680 |
| 34K tokens (Strata: 28K, its limit is 32K) | 7,950 | **9,260** | 5,940 |

| Tool calls | Kuroshio | vLLM + DSpark | vLLM, no drafter | Strata |
|---|---:|---:|---:|---:|
| Well-formed, of 24 (median of 5 passes; range) | **24** (24-24) | 16 (13-16) | 13 (13-15) | 2 (2-2) |

† Kuroshio supports 4 slots; these rows use a benchmark-only override. At 8 slots its conversations share 127K
tokens of context, at 16 slots 31K (under 2K each).
‡ With 16 parallel requests Strata stopped answering in three attempts (every request stayed at "reading the
prompt" until the harness gave up after 15 to 30 minutes). A day earlier, without the effort setting, it completed
this cell at 562 tok/s.

**The same comparison at equal cache precision.** vLLM's 8-bit cache is lossy; Kuroshio's default is not. Run both
ways:

| Decode, tok/s in total | Kuroshio, 16-bit | vLLM + DSpark, 16-bit § | Kuroshio, `--kv 8` | vLLM + DSpark, 8-bit |
|---|---:|---:|---:|---:|
| 1 stream | **250** | 219 | **267** | 220 |
| 2 streams | **419** | 358 | **426** | 349 |
| 4 streams | **559** | 505 | **546** | 509 |
| 8 streams † | **761** | 754 | 740 | **791** |
| 16 streams † | **1,062** | 1,054 | 1,067 | **1,307** |
| 34K-token prompt, tok/s | 7,950 | **9,420** | 6,530 | **9,260** |

§ With a 16-bit cache vLLM does not start at its usual 94K-token limit; this row ran with the limit lowered to 41K.

At 16 bits the two engines are level at 8 and 16 streams and Kuroshio leads below; vLLM's lead at many streams
comes from its 8-bit cache, which Kuroshio's `--kv 8` does not yet turn into speed (it stores 8 bits and
multiplies in 16).

**How long a conversation fits.** The model allows a million tokens; on a 24-32 GB card memory decides. Each row
is one real conversation: a document with a fact near its start, middle and end, then three questions about them.

| One stream | Context limit | Prompt run | First token after | Facts retrieved |
|---|---:|---:|---:|---:|
| Kuroshio, default (16-bit, lossless) | **210,944** | 208,854 | 41 s | 3 of 3 |
| Kuroshio, `--no-draft` | 266,240 | 258,311 | 55 s | 3 of 3 |
| Kuroshio, `--kv 8` | **358,400** | 351,301 | 155 s | 3 of 3 |
| Kuroshio, `--kv 8 --dense 8 --no-draft` | 524,288 | 518,832 | 314 s | 3 of 3 |
| vLLM + DSpark, 8-bit cache | 132,096 | 126,726 | 19 s | 3 of 3 |
| vLLM + DSpark, 16-bit cache | 45,056 | 36,302 | 9 s | 3 of 3 |
| Strata, as installed | 32,768 | 22,806 | 4 s | 3 of 3 |

vLLM's limits were found by trying: at 8 bits it refused 200K and started at the 132K it then reported; at 16 bits
it refused 70K and started at 45K.

- Follow-up questions took 0.2-0.5 s on Kuroshio and vLLM, which keep the conversation cached; Strata read its
  prompt again each time (4.3 s).
- These are for a card that also drives a desktop (1.4 GiB in use). The limits are what the servers accept; the
  prompts are a few thousand tokens shorter to leave room for the answers.
- Reading gets slower with length: 5,100 tok/s averaged over 209K tokens at 16 bits, and much slower with the 8-bit
  cache (2,300 tok/s over 351K), because every cached tile is converted back to 16 bits before it is used.

How to read these:

- **Single-stream speed is tokens over the whole request**, hidden work included. Strata generates its thinking
  without streaming it and counts those tokens; timed from the first visible token it would show 280 tok/s. Timed
  that way Kuroshio is 253 and vLLM 228.
- **Prompts are read by a separate, faster set of kernels**, 1,024 tokens per step. Their results differ from
  generation's in the last bits: after a prompt read this way about 97% of greedy tokens are the same as with the
  exact kernels, and quality measured on text is the same within noise (-0.014 to +0.007 nats per token over six
  texts). The price is that an answer can depend on what was cached and on what else was running.
  `--exact-prompts` reads prompts with generation's arithmetic instead: 7,550 / 6,740 / 5,870 tok/s at
  4.7K / 19K / 34K, and then the same request gives the same answer bit for bit, cached or not, alone or not.
- **vLLM is still faster on very long prompts**: it reads a whole prompt in one pass after about 1.7 s of fixed
  start-up, so its rate keeps rising with length, and it is fast on any prompt length it has seen before (its
  ranges above run from 1,900 to 18,000 tok/s; the medians are what a new prompt gets). Kuroshio also has the
  prefix cache: a follow-up turn skips what the server has already read (typically 30-50 ms to first token in an
  agent loop).
- **The tool-call failures are the model's, not the parsers'.** After `{"name":"read_file"` the model often skips
  the comma, and that one token is numerically unstable: for the same token ids, vLLM puts 0.95 on the comma while
  generating and 0.27 when it re-reads them as a prompt. vLLM's raw output, with no parser involved, has the same
  broken calls. The chat template allows only one spelling of that opening, so Kuroshio writes it itself. (The
  test allows 2,000 tokens per call so that thinking at effort high does not cut a call short.)
- Single-stream runs vary by about ±8% with the text being generated (230-269 over the ten runs).
- When the model repeats text it has already seen (rewriting a file with a small change, quoting a document,
  echoing tool output), tokens are verified up to 31 at a time: 480-620 tok/s on one stream.
- With `--kv 8 --dense 8` (8-bit cache and 8-bit attention matrices): about 400K tokens of context, about
  265 tok/s, 6,600 tok/s prompt processing at 4.7K.
- It starts at about 20 GiB of GPU memory and takes more only as conversations grow, so other programs can use the
  card; when the card is full a conversation stops growing instead of the server crashing.
- Without speculative decoding: 161 tok/s (`--spec 0`) against vLLM's 167.

## Install

Needs Linux, an RTX 5090 with a current NVIDIA driver, Rust, and Python 3 (only to fetch NVIDIA's compiler; no root).

```bash
git clone <this repo> kuroshio && cd kuroshio
./install.sh                                  # builds, links ~/.local/bin/kuroshio
hf download thinkingmachines/Inkling-2-Mini   # 19.5 GB, once
```

The model is found in the Hugging Face cache. If yours is elsewhere: `./install.sh --model /path/to/snapshot`.

## Run

```bash
kuroshio serve                          # 1 conversation, up to about 240K tokens
kuroshio serve --slots 4                # 4 conversations at once, sharing about 200K tokens
kuroshio serve --slots 4 --kv 8         # 4 conversations, sharing about 375K tokens
kuroshio serve --reserve 6g             # leave 6 GiB of the card for other programs
kuroshio serve --ctx 64k --port 9000    # cap conversations at 64K tokens, another port
kuroshio run "Why is the sky blue?" --effort low
```

```bash
curl http://127.0.0.1:8020/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "inkling-2-mini",
  "messages": [{"role": "user", "content": "Name three rivers in Romania."}],
  "reasoning_effort": "low", "stream": true
}'
```

Any OpenAI client works: base URL `http://127.0.0.1:8020/v1`, any API key, model `inkling-2-mini`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8020/v1", api_key="none")
r = client.chat.completions.create(model="inkling-2-mini", messages=[{"role": "user", "content": "Hello"}],
                                   extra_body={"reasoning_effort": "none"})
```

## Options

`kuroshio serve`:

| Option | Values | Default | What it does |
|---|---|---|---|
| `--host` | address | `127.0.0.1` | Address to listen on. |
| `--port` | 1-65535 | `8020` | Port. |
| `--slots` | 1-4 | `1` | Conversations served at the same time. More requests wait in a queue. |
| `--ctx` | `auto`, or tokens ≥ 2048 (`32768`, `64k`) | `auto` | Longest conversation allowed. `auto` is what the free GPU memory could hold. |
| `--reserve` | size (`512m`, `6g`) | `512m` | GPU memory to leave free for other programs. Conversations stop growing at that point. |
| `--idle-release` | seconds, 0 = never | `0` | Give a slot's context memory back after it has been idle this long. Its prefix cache goes with it. |
| `--kv` | `16`, `8` | `16` | Precision of the attention cache. `8` fits 1.7 times the context; decodes as fast or slightly faster, reads prompts 11-18% slower; lossy. |
| `--dense` | `16`, `8` | `16` | Precision of the attention matrices and output head. `8` saves 0.8 GiB (about 40K more tokens of context); decode speed is unchanged and prompts are read about 12% slower; lossy. |
| `--spec` | 0-7 | `7` | Speculative decoding: verify up to this many drafted tokens per step. `0` turns it off. Output is unchanged either way. |
| `--tau` | `auto`, or 0-1 | `auto` | Which drafts to verify. `auto` weighs each draft's chance of being accepted against the time a row costs, and adapts to load. A number verifies those the drafter gives at least that probability. |
| `--copy` | 0-32 | `31` | When the conversation starts repeating an earlier passage, verify up to this many copied tokens per step. `0` turns it off. Needs speculation on (`--spec` above 0, drafter loaded). Output is unchanged either way. |
| `--prefill-share` | 1-128 | `48` | Prompt tokens processed per step while another conversation is generating. Higher favours new requests, lower favours running streams. |
| `--cors` | origin, or `*` | none | Let web pages from this origin call the server. Off by default: no page can. |
| `--exact-prompts` | | | Read prompts with the same arithmetic as generation: 10-25% slower on prompts, and an answer then never depends on what was cached or what else was running (see the headline notes). |
| `--sampled-drafts` | | | Experimental: when sampling, draw drafts from the drafter's distribution instead of its top token. Exact, but measured no faster. |
| `--no-draft` | | | Do not load the drafter: saves 1.1 GiB; no speculation, and the output head runs in full (about 4% slower per step). |
| `--no-prefix-cache` | | | Always re-read the whole prompt: saves 0.2 GiB per slot. |
| `--no-tool-fix` | | | Let the model write the fixed opening of a tool call itself (see below). |
| `--model` | directory | auto | Model snapshot directory. Also `INKLING_WEIGHTS`. |
| `--name` | text | `inkling-2-mini` | Model id reported to clients. |

`kuroshio run "prompt"` takes `--effort none|minimal|low|medium|high|xhigh|max` (default `none`), `--max-tokens`
(512), `--temp` (0 = greedy), `--seed`, and `--spec`, `--tau`, `--kv`, `--dense`, `--no-draft`, `--exact-prompts`,
`--model` as above.

### Memory, slots and context

The card's memory is the limit, not its speed. The engine takes about 20.4 GiB at start-up (plus 0.5 GiB per slot)
and then about 40 KB per token of conversation (24 KB with `--kv 8`), **as conversations grow**. Slots share what
is left, in any proportion: with 4 slots, one agent can hold 130K tokens while three hold 15K each.

| | default | `--kv 8` |
|---|---:|---:|
| tokens in total, 1 slot | 211K | 358K |
| tokens in total, 4 slots | 176K | 326K |

- These are for a 5090 that also drives a desktop (about 1.4 GiB in use); the server prints the exact figure at
  start-up and reports the longest conversation it accepts in `GET /v1/models`.
- If another program takes the memory first, a request that no longer fits gets a 503 and a generation ends with
  `finish_reason: "length"`; everything else keeps running. Memory comes back when a shorter conversation takes
  over a slot, or after `--idle-release` seconds.
- These lengths have been run end to end: a document of that size with a fact near its start, middle and end,
  all three retrieved (see the headline). `--no-draft` adds about 55K tokens, `--dense 8` about 40K.

### API

`POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `GET /health`.

- Supported request fields: `messages`, `tools`, `tool_choice: "none"`, `max_tokens` / `max_completion_tokens`,
  `temperature` (0-2), `seed` (-1 = random), `stop`, `stream`, `reasoning_effort` (also as
  `chat_template_kwargs.reasoning_effort`).
- `stop` applies to the answer text, not to thinking or tool calls: up to 16 strings of at most 256 bytes.
- The model's thinking is returned as `reasoning_content`; `usage.prompt_tokens_details.cached_tokens` reports what
  the prefix cache saved.
- `top_p` is accepted and ignored. `logprobs`, `response_format` other than text and `n` > 1 are refused with a 400.
  Images are not supported.
- Closing the connection, or its sending half, cancels the request and frees its slot.
- Text in messages and tool results is only ever text: a control token such as `<|end_message|>` spelt out in it
  cannot end a message or start one.
- **Tool calls.** The model often garbles the fixed opening of a tool call (`{"name":"…","args":`), on any engine
  (see the headline notes). The chat template allows only one spelling of that opening, so the server writes it and
  lets the model continue with the arguments. `--no-tool-fix` turns that off.

### Thinking level

Set `reasoning_effort` per request, as a name or a number from 0 to 0.99:

| `none` | `minimal` | `low` | `medium` | `high` | `xhigh` / `max` |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.1 | 0.2 | 0.7 | 0.9 | 0.99 |

- **Default:** a request that does not set it gets 0.9 (`high`), the model's own default. `kuroshio run` defaults
  to `none`; change it with `--effort`.
- It is a line in the prompt, so the model is asked, not forced: `none` can still think a little.
- Thinking counts against `max_tokens`. With a small limit and a high level the answer can come back empty.

## Why it is fast

- **Blackwell tensor cores, directly.** The checkpoint's 4-bit (NVFP4) and 8-bit (MXFP8) expert weights are
  multiplied in their stored format by hand-written `mma.sync` kernels; nothing is expanded to 16 bits.
- **One CUDA graph per step.** A step is hundreds of small kernels replayed as a single graph, with weights laid
  out in the order the tensor cores read them.
- **Speculative decoding** with the model's bundled DSpark drafter. Drafts for all conversations are computed in
  one pass, and each step verifies only the drafts whose chance of acceptance is worth a row's time, so less is
  speculated as more streams share the card.
- **Copied drafts.** Agents repeat themselves: a file rewritten with one line changed, a quoted passage. When the
  last tokens match an earlier part of the conversation, what followed there is verified as a draft, up to 31 tokens
  in one step, with no drafter pass.
- **A two-stage output head.** Finding the next token means scoring all 200,058 of them against a 0.9 GB
  matrix. An 8-bit pass with a proven error bound narrows that to a handful of candidates, which are then scored
  exactly: the same token, at half the memory traffic.
- **1,024 prompt tokens per step**, so that each expert's weights are read once for many tokens, with attention
  fused into one kernel per step: scores, softmax and value sums stay in registers, laid out so that the
  probabilities feed the next tensor-core multiplication where they are. And a **prefix cache** that keeps each
  conversation's state between requests.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the model, [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md)
is a step-by-step account of how the engine was built and tuned, written for engineers new to GPU inference, and
[PLAN.md](PLAN.md) holds every experiment with its numbers, including the ones that failed.

## Checking it

```bash
source scripts/env.sh
scripts/verify.sh        # correctness: reference tokens, batch-size invariance, speculation, concurrency, both prompt modes
scripts/verify.sh full   # adds sampling statistics and a GPU memory checker
scripts/fullcard.sh      # another program takes the whole GPU mid-run: nothing may crash
python3 gauntlet/run.py --suite quick   # the server from outside: API, robustness, speed, memory, an agent (10 min)
```

Correctness is defined against vLLM's output and a slow PyTorch reference (`ref/`). The model is numerically
chaotic: two correct engines agree on about 95-97% of greedy tokens, so the tests compare layer by layer and
statistically, not bit for bit. Benchmarks use `bench.py` from the
[inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) with `configs/`.

## Limits

- Inkling 2 Mini only, text only, one GPU, RTX 5090 only (the kernels use instructions other cards lack).
- At most 4 conversations at once; more wait in a queue. For 8 or more streams vLLM with its 8-bit cache gives
  more in total.
- By default the answer to a request can differ in detail with what was cached and what else was running;
  `--exact-prompts` removes that at 10-25% of prompt speed.
- Prompt speed falls with length: about 8,000 tok/s up to 34K tokens, 5,100 averaged over a 209K-token prompt
  (41 s to the first token).
- Sampling is temperature only.

## License

[MIT](LICENSE): free for personal and commercial use. It covers this engine only; the model has its own license
on its [Hugging Face page](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
