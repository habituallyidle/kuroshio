<p align="center">
  <img src="assets/logo.png" width="160" alt="Kuroshio logo: a dark river in pixel art">
</p>

<h1 align="center">Kuroshio 黒潮</h1>

<p align="center"><em>The "black current": the fast, warm stream off Japan's Pacific coast. Dark as ink, and full of squid.</em></p>

**Run [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) on your RTX 5090, at the highest
single-stream speed measured for it.** Kuroshio is an inference engine built for exactly one model and one GPU:
hand-written CUDA kernels for Blackwell's tensor cores behind an OpenAI-compatible server. One binary, nothing to
configure.

- **27% faster than vLLM** on one stream and twice as fast as Strata; ahead of both up to 4 streams
- **240K tokens** of context, lossless, and follow-up turns answer in milliseconds
- **Tool calls that parse**: 24 of 24, where vLLM manages 15 and Strata 2
- **Shares the card**: takes memory as conversations grow, and survives another program filling the GPU
- **Not the fastest at everything**: vLLM and Strata read long prompts faster, and vLLM is ahead beyond 4 streams
  (see below)

## Headline

Same RTX 5090, same hour, same harness (`bench.py`, temperature 0.6, 512-token completions); every cell is the
median of 10 runs. vLLM is the official `vllm/vllm-openai:inkling-2-mini` image with an 8-bit KV cache, with and
without its DSpark drafter; Strata is 0.2.19 with its `inkling-2-mini-sq-20` pack and DSpark drafting; Kuroshio runs
at its defaults (lossless 16-bit cache).

| Decode, tok/s in total | Kuroshio | vLLM + DSpark | vLLM, no drafter | Strata |
|---|---:|---:|---:|---:|
| 1 stream | **248** | 196 | 170 | 119 |
| 2 streams | **389** | 336 | 239 | 181 |
| 4 streams | **502** | 458 | 397 | 273 |
| 8 streams † | 704 | **742** | 664 | 392 |
| 16 streams † | 949 | **1,220** | 1,068 | 562 |

| Prompt processing, tok/s (text never seen before) | Kuroshio | vLLM | Strata |
|---|---:|---:|---:|
| 4.7K tokens | 3,330 | 2,480 | **7,280** |
| 19K tokens | 2,850 | **6,800** | 6,520 |
| 34K tokens (Strata: 28K, its limit is 32K) | 2,470 | **8,710** | 5,840 |

| Tool calls | Kuroshio | vLLM + DSpark | vLLM, no drafter | Strata |
|---|---:|---:|---:|---:|
| Well-formed, of 24 (median of 10 passes; range) | **24** (24-24) | 15 (11-15) | 13 (8-13) | 2 (2-2) |

† Kuroshio supports 4 slots; these rows use a benchmark-only override. At 8 slots its conversations share 160K
tokens of context, at 16 slots 61K (under 4K each). Beyond 4 streams vLLM scales better: its step time grows more
slowly with the number of conversations.

How to read these:

- **Single-stream speed is tokens over the whole request**, hidden work included. Strata generates its thinking
  without streaming it and counts those tokens; timed from the first visible token it would show 274 tok/s. Timed
  that way Kuroshio is 250 and vLLM 199.
- **Prompt processing is where Kuroshio is behind.** It reads 256 tokens per step; vLLM reads a whole prompt in one
  pass after about 1.7 s of fixed start-up, and Strata is faster still on short prompts. On a 100K-token prompt
  Kuroshio falls to about 900 tok/s. What it does have is the prefix cache: a follow-up turn skips what the server
  has already read (typically 30-50 ms to first token in an agent loop).
- **The tool-call failures are the model's, not the parsers'.** After `{"name":"read_file"` the model often skips
  the comma, and that one token is numerically unstable: for the same token ids, vLLM puts 0.95 on the comma while
  generating and 0.27 when it re-reads them as a prompt. vLLM's raw output, with no parser involved, has the same
  broken calls. The chat template allows only one spelling of that opening, so Kuroshio writes it itself.
- Single-stream runs vary by about ±5% with the text being generated (236-259 over the ten runs).
- When the model repeats text it has already seen (rewriting a file with a small change, quoting a document,
  echoing tool output), tokens are verified up to 31 at a time: 480-620 tok/s on one stream.
- With `--kv 8 --dense 8` (the 8-bit cache vLLM was run with, plus 8-bit attention matrices): about 450K tokens of
  context, about 260 tok/s, 3,500 tok/s prompt processing at 4K.
- It starts at about 20 GiB of GPU memory and takes more only as conversations grow, so other programs can use the
  card; when the card is full a conversation stops growing instead of the server crashing.
- Without speculative decoding: 166 tok/s (`--spec 0`) against vLLM's 170.

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
| `--kv` | `16`, `8` | `16` | Precision of the attention cache. `8` nearly doubles the context that fits; about 3% slower; lossy. |
| `--dense` | `16`, `8` | `16` | Precision of the attention matrices and output head. `8` decodes about 7% faster, reads prompts about 10% faster and saves 0.8 GiB; lossy. |
| `--spec` | 0-7 | `7` | Speculative decoding: verify up to this many drafted tokens per step. `0` turns it off. Output is unchanged either way. |
| `--tau` | `auto`, or 0-1 | `auto` | Which drafts to verify. `auto` weighs each draft's chance of being accepted against the time a row costs, and adapts to load. A number verifies those the drafter gives at least that probability. |
| `--copy` | 0-32 | `31` | When the conversation starts repeating an earlier passage, verify up to this many copied tokens per step. `0` turns it off. Needs speculation on (`--spec` above 0, drafter loaded). Output is unchanged either way. |
| `--prefill-share` | 1-128 | `48` | Prompt tokens processed per step while another conversation is generating. Higher favours new requests, lower favours running streams. |
| `--cors` | origin, or `*` | none | Let web pages from this origin call the server. Off by default: no page can. |
| `--sampled-drafts` | | | Experimental: when sampling, draw drafts from the drafter's distribution instead of its top token. Exact, but measured no faster. |
| `--no-draft` | | | Do not load the drafter: saves 1.1 GiB; no speculation, and the output head runs in full (about 4% slower per step). |
| `--no-prefix-cache` | | | Always re-read the whole prompt: saves 0.2 GiB per slot. |
| `--no-tool-fix` | | | Let the model write the fixed opening of a tool call itself (see below). |
| `--model` | directory | auto | Model snapshot directory. Also `INKLING_WEIGHTS`. |
| `--name` | text | `inkling-2-mini` | Model id reported to clients. |

`kuroshio run "prompt"` takes `--effort none|minimal|low|medium|high|xhigh|max` (default `none`), `--max-tokens`
(512), `--temp` (0 = greedy), `--seed`, and `--spec`, `--tau`, `--kv`, `--dense`, `--no-draft`, `--model` as above.

### Memory, slots and context

The card's memory is the limit, not its speed. The engine takes about 20 GiB at start-up (plus 0.5 GiB per slot) and
then about 40 KB per token of conversation (24 KB with `--kv 8`), **as conversations grow**. Slots share what is
left, in any proportion: with 4 slots, one agent can hold 150K tokens while three hold 15K each.

| | default | `--kv 8` |
|---|---:|---:|
| tokens in total, 1 slot | 240K | 415K |
| tokens in total, 4 slots | 200K | 375K |

- These are for a 5090 that also drives a desktop (about 1.4 GiB in use); the server prints the exact figure at
  start-up and reports the longest conversation it accepts in `GET /v1/models`.
- If another program takes the memory first, a request that no longer fits gets a 503 and a generation ends with
  `finish_reason: "length"`; everything else keeps running. Memory comes back when a shorter conversation takes
  over a slot, or after `--idle-release` seconds.
- The longest conversation tested end to end is 60K tokens; the larger figures are what fits in memory.

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
- **256 prompt tokens per step**, so that each expert's weights are read once for many tokens, and a **prefix cache**
  that keeps each conversation's state between requests.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the model, [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md)
is a step-by-step account of how the engine was built and tuned, written for engineers new to GPU inference, and
[PLAN.md](PLAN.md) holds every experiment with its numbers, including the ones that failed.

## Checking it

```bash
source scripts/env.sh
scripts/verify.sh        # correctness: reference tokens, batch-size invariance, speculation, concurrency
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
- At most 4 conversations at once; more wait in a queue. For 8 or more streams vLLM gives more in total.
- Long prompts are read at 2,500-3,300 tok/s (about 900 at 100K tokens): a new 30K-token document takes about
  13 seconds, where vLLM takes 4.
- Sampling is temperature only.

## License

[MIT](LICENSE): free for personal and commercial use. It covers this engine only; the model has its own license
on its [Hugging Face page](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
