"""Record golden outputs from the vLLM reference server (plain greedy decoding, BF16 KV cache).

For each prompt we store the exact prompt token ids and, for each generated position, the chosen
token id and the top-20 (token id, log-probability) pairs. The engine must reproduce these.
Usage: python collect.py http://127.0.0.1:8011 out.json
"""
import json, sys, time, urllib.request

BASE, OUT = sys.argv[1], sys.argv[2]
MODEL = "inkling-2-mini"
PROMPTS = [
    ("arith", "none", "What is 2+2? Answer with just the number."),
    ("code", "none", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("prose", "none", "Explain in three sentences why the sky is blue."),
    ("think", "low", "A train leaves at 9:40 and arrives at 11:05. How long is the trip?"),
]

def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)

cases = []
for name, effort, text in PROMPTS:
    tok = post("/tokenize", {"model": MODEL, "messages": [{"role": "user", "content": text}], "add_generation_prompt": True,
                             "chat_template_kwargs": {"reasoning_effort": effort}})
    ids = tok["tokens"]
    t0 = time.time()
    r = post("/v1/completions", {"model": MODEL, "prompt": ids, "max_tokens": 64, "temperature": 0, "logprobs": 20,
                                 "return_tokens_as_token_ids": True, "skip_special_tokens": False, "add_special_tokens": False})
    dt = time.time() - t0
    lp = r["choices"][0]["logprobs"]
    tid = lambda s: int(s.split(":", 1)[1])
    steps = [{"token": tid(t), "logprob": l, "top": sorted(((tid(k), v) for k, v in top.items()), key=lambda x: -x[1])}
             for t, l, top in zip(lp["tokens"], lp["token_logprobs"], lp["top_logprobs"])]
    cases.append({"name": name, "effort": effort, "text": text, "prompt_ids": ids, "steps": steps, "completion": r["choices"][0]["text"],
                  "finish": r["choices"][0]["finish_reason"]})
    print(f"{name}: {len(ids)} prompt tokens, {len(steps)} generated in {dt:.2f}s -> {r['choices'][0]['text'][:90]!r}")
json.dump({"source": "vllm/vllm-openai:inkling-2-mini, SPEC_DECODE=off, kv bf16, revision ed40e7da", "cases": cases}, open(OUT, "w"))
