#!/usr/bin/env bash
# What happens when another program takes the whole GPU while the server is running?
# Expected: nothing crashes. A generation stops at the memory its slot already has (finish_reason "length"),
# a prompt that no longer fits gets HTTP 503, and everything works again once the memory is back.
#   source scripts/env.sh && scripts/fullcard.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT"
BIN=target/release/kuroshio; PORT=8029; TMP="$(mktemp -d)"
nap() { timeout "$1" tail -f /dev/null || true; }
nvcc -O1 -o "$TMP/hog" tools/hog/hog.cu -L"$CUDA_HOME/lib" -l:libcudart.so.13 -Xlinker -rpath,"$CUDA_HOME/lib"
$BIN serve --port $PORT > "$TMP/serve.log" 2>&1 & SRV=$!
trap 'kill $SRV 2>/dev/null; pkill -x hog 2>/dev/null; rm -rf "$TMP"' EXIT
for i in $(seq 1 90); do curl -fsS -m 2 http://127.0.0.1:$PORT/health >/dev/null 2>&1 && break; nap 1; done
ask() {  # ask <words of prompt> <max_tokens> -> "status finish_reason total_tokens"
  python3 -I - "$1" "$2" <<PY
import json, sys, urllib.request
body = {"model": "x", "messages": [{"role": "user", "content": "word " * int(sys.argv[1]) + "\nWrite the integers from 1 to 3000, one per line."}], "reasoning_effort": "none", "temperature": 0, "max_tokens": int(sys.argv[2])}
try:
    d = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:$PORT/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=600))
    print(200, d["choices"][0]["finish_reason"], d["usage"]["total_tokens"])
except urllib.error.HTTPError as e:
    print(e.code, "-", 0)
PY
}
fail() { echo "FAIL: $1"; exit 1; }
r=$(ask 10 50);    echo "before:                         $r"; [[ $r == 200* ]] || fail "plain request"
"$TMP/hog" 40 > "$TMP/hog.log" & nap 5; echo "another program: $(cat "$TMP/hog.log")"
r=$(ask 10 6000);  echo "same conversation, card full:   $r"; [[ $r == "200 length"* ]] || fail "a generation must end with length at its memory"
r=$(ask 30000 20); echo "long prompt, card full:         $r"; [[ $r == 503* ]] || fail "a prompt that cannot fit must get 503"
wait %2 2>/dev/null || true; nap 1
r=$(ask 30000 20); echo "long prompt, memory back:       $r"; [[ $r == 200* ]] || fail "must work again once memory is back"
kill -0 $SRV 2>/dev/null || fail "the server died"
grep -E "panicked|CUDA error" "$TMP/serve.log" && fail "errors in the server log"
echo "ok: the server survived a full card"
