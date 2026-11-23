<img src="assets/logo.png" width="112" align="right" alt="Kuroshio logo: a dark river in pixel art">

# Kuroshio 黒潮

An inference engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) written for exactly
one GPU: the RTX 5090. One binary, an OpenAI-compatible server, nothing to configure.

*Kuroshio*, the "black current", is the fast warm stream off Japan's Pacific coast: dark as ink, and full of squid.

## Headline

Same RTX 5090, same harness, temperature 0.6, 512-token completions, median of 3 runs. vLLM is the official
`vllm/vllm-openai:inkling-2-mini` image with its DSpark drafter and an 8-bit KV cache; Kuroshio runs at its defaults
(lossless 16-bit cache).

| | Kuroshio | vLLM | |
|---|---:|---:|---:|
| Decode, 1 stream (tok/s) | **242** | 200 | +21% |
| Decode, 2 streams, total | **381** | 310 | +23% |
| Decode, 4 streams, total | **474** | 431 | +10% |
| Prompt processing (tok/s, 4K prompt) | **2,360** | 580 | 4.1x |
| Well-formed tool calls (24 variants of one agent prompt, temperature 0) | **24 / 24** | 9 / 24 | |

- Context: about 240K tokens for one conversation, lossless. With `--kv 8 --dense 8` (the 8-bit cache vLLM was run
  with, plus 8-bit attention matrices): about 450K tokens, about 250 tok/s, 2,800 tok/s prompt processing.
- Follow-up turns of a conversation skip the part of the prompt the server has already seen: 11-56 ms to first token
  in an agent loop, instead of re-reading everything.
- It starts at about 20 GiB of GPU memory and takes more only as conversations grow, so other programs can use the
  card; when the card is full a conversation stops growing instead of the server crashing.
- Single-stream runs vary by about ±5% with the text being generated (233-253 over four runs).
- Without speculative decoding the two are level: 158 tok/s (`--spec 0`) against vLLM's 163.

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
kuroshio serve --slots 4                # 4 conversations at once, sharing about 205K tokens
kuroshio serve --slots 4 --kv 8         # 4 conversations, sharing about 385K tokens
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
| `--dense` | `16`, `8` | `16` | Precision of the attention matrices and output head. `8` decodes about 7% faster, reads prompts about 20% faster and saves 0.8 GiB; lossy. |
| `--spec` | 0-7 | `7` | Speculative decoding: verify up to this many drafted tokens per step. `0` turns it off. Output is unchanged either way. |
| `--tau` | `auto`, or 0-1 | `auto` | Which drafts to verify. `auto` weighs each draft's chance of being accepted against the time a row costs, and adapts to load. A number verifies those the drafter gives at least that probability. |
| `--prefill-share` | 1-128 | `48` | Prompt tokens processed per step while another conversation is generating. Higher favours new requests, lower favours running streams. |
| `--sampled-drafts` | | | Experimental: when sampling, draw drafts from the drafter's distribution instead of its top token. Exact, but measured no faster. |
| `--no-draft` | | | Do not load the drafter: saves 1.1 GiB, no speculation. |
| `--no-prefix-cache` | | | Always re-read the whole prompt: saves 0.2 GiB per slot. |
| `--no-tool-fix` | | | Let the model write the fixed opening of a tool call itself (see below). |
| `--model` | directory | auto | Model snapshot directory. Also `INKLING_WEIGHTS`. |
| `--name` | text | `inkling-2-mini` | Model id reported to clients. |

`kuroshio run "prompt"` takes `--effort none|minimal|low|medium|high|xhigh|max` (default `none`), `--max-tokens`
(512), `--temp` (0 = greedy), `--seed`, and `--spec`, `--tau`, `--kv`, `--dense`, `--no-draft`, `--model` as above.

### Memory, slots and context

The card's memory is the limit, not its speed. The engine takes about 20 GiB at start-up (plus 0.4 GiB per slot) and
then about 40 KB per token of conversation (24 KB with `--kv 8`), **as conversations grow**. Slots share what is
left, in any proportion: with 4 slots, one agent can hold 150K tokens while three hold 15K each.

| | default | `--kv 8` |
|---|---:|---:|
| tokens in total, 1 slot | 240K | 415K |
| tokens in total, 4 slots | 205K | 385K |

- These are for a 5090 that also drives a desktop (about 1.4 GiB in use); the server prints the exact figure at
  start-up and reports the longest conversation it accepts in `GET /v1/models`.
- If another program takes the memory first, a request that no longer fits gets a 503 and a generation ends with
  `finish_reason: "length"`; everything else keeps running. Memory comes back when a shorter conversation takes
  over a slot, or after `--idle-release` seconds.
- The longest conversation tested end to end is 60K tokens; the larger figures are what fits in memory.

### API

`POST /v1/chat/completions` (streaming or not), `GET /v1/models`, `GET /health`.

- Supported request fields: `messages`, `tools`, `tool_choice: "none"`, `max_tokens` / `max_completion_tokens`,
  `temperature` (0-2), `seed`, `stream`, `reasoning_effort` (also as `chat_template_kwargs.reasoning_effort`).
- The model's thinking is returned as `reasoning_content`; `usage.prompt_tokens_details.cached_tokens` reports what
  the prefix cache saved.
- Not supported: `stop`, `top_p`, `n` > 1, `logprobs`, `response_format`, images.
- **Tool calls.** The model often garbles the fixed opening of a tool call (`{"name":"…","args":`); vLLM shows the
  same broken calls. The chat template allows only one spelling of that opening, so the server writes it and lets
  the model continue with the arguments.

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
- **128 prompt tokens per step**, and a **prefix cache** that keeps each conversation's state between requests.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) describes the model, [docs/IMPLEMENTATION_LOG.md](docs/IMPLEMENTATION_LOG.md)
is a step-by-step account of how the engine was built and tuned, written for engineers new to GPU inference, and
[PLAN.md](PLAN.md) holds every experiment with its numbers, including the ones that failed.

## Checking it

```bash
source scripts/env.sh
scripts/verify.sh        # correctness: reference tokens, batch-size invariance, speculation, concurrency
scripts/verify.sh full   # adds sampling statistics and a GPU memory checker
scripts/fullcard.sh      # another program takes the whole GPU mid-run: nothing may crash
```

Correctness is defined against vLLM's output and a slow PyTorch reference (`ref/`). The model is numerically
chaotic: two correct engines agree on about 95-97% of greedy tokens, so the tests compare layer by layer and
statistically, not bit for bit. Benchmarks use `bench.py` from the
[inference-optimisation skill](https://github.com/sybil-solutions/inference-optimisation-skill) with `configs/`.

## Limits

- Inkling 2 Mini only, text only, one GPU, RTX 5090 only (the kernels use instructions other cards lack).
- Sampling is temperature only.

## License

[MIT](LICENSE): free for personal and commercial use. It covers this engine only; the model has its own license
on its [Hugging Face page](https://huggingface.co/thinkingmachines/Inkling-2-Mini).
