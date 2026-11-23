#!/usr/bin/env bash
# Regression checks to run after every engine change. Exits non-zero on the first failure.
#   scripts/verify.sh          quick set (about 1 minute)
#   scripts/verify.sh full     adds the sampling-distribution test and the memory checker
set -euo pipefail
cd "$(dirname "$0")/.."; source scripts/env.sh
BIN=./target/release/kuroshio; TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
echo "== golden outputs (vLLM greedy tokens; a few mismatches per prompt are numerical noise)"
$BIN check tests/oracle/golden.json 2>/dev/null | tee "$TMP/golden.txt"
bad=$(awk '{s+=$6} END{print s+0}' "$TMP/golden.txt"); [ "$bad" -le 20 ] || { echo "FAIL: $bad golden mismatches (expected about 10 of 236)"; exit 1; }
echo "== batch size must not change results"
$BIN argmaxes 1500 > "$TMP/b16.txt" 2>/dev/null; INK_CHUNK=1 $BIN argmaxes 1500 > "$TMP/b1.txt" 2>/dev/null; for c in 128 100 48 14 13; do INK_CHUNK=$c $BIN argmaxes 1500 > "$TMP/bx.txt" 2>/dev/null
  cmp -s "$TMP/b16.txt" "$TMP/bx.txt" || { echo "FAIL: batch-$c and batch-16 greedy tokens differ"; exit 1; }; done   # 100 and 48 straddle the 1,024-token bucket boundary
cmp -s "$TMP/b16.txt" "$TMP/b1.txt" || { echo "FAIL: batch-16 and batch-1 greedy tokens differ"; exit 1; }; echo "ok"
echo "== conversations advanced together must match separate runs"
$BIN interleave 3 1300 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: interleaved conversations differ from separate runs"; exit 1; }; echo "ok"
echo "== drafts computed for several conversations at once must match one at a time"
$BIN draftcheck 42 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: batched drafts differ"; exit 1; }; echo "ok"
echo "== recorded step graphs must not grow without bound (each holds GPU memory)"
INK_GRAPH_CAP=8 $BIN graphmem 2>/dev/null | tail -1 | awk '{ exit !($(NF-1) < 80) }' || { echo "FAIL: graph memory is not bounded"; exit 1; }; echo "ok"
echo "== speculation must not change greedy output (short and long prompt, lossless and 8-bit modes)"
LONG="$(head -c 7000 docs/ARCHITECTURE.md)

Summarise the document above in five bullet points."
for mode in 0 1; do for P in "Write a Python function that checks whether a string is a palindrome, with a short explanation." "$LONG"; do
  INK_FP8=$mode INK_SPEC=0 $BIN gen "$P" none 200 > "$TMP/plain.txt" 2>/dev/null
  for k in 4 7; do INK_FP8=$mode INK_SPEC=$k $BIN gen "$P" none 200 > "$TMP/spec.txt" 2>"$TMP/spec.err" || { echo "FAIL: speculative run crashed (INK_FP8=$mode k=$k)"; tail -2 "$TMP/spec.err"; exit 1; }
    cmp -s "$TMP/plain.txt" "$TMP/spec.txt" || { echo "FAIL: output differs with speculation (INK_FP8=$mode k=$k)"; exit 1; }; done
done; done; echo "ok"
echo "== quality on text"
$BIN nll 4000 2>/dev/null | tail -1; INK_FP8=1 $BIN nll 4000 2>/dev/null | tail -1 | sed 's/^/8-bit mode: /'
if [ "${1:-}" = full ]; then
  echo "== sampling distribution with and without speculation"; $BIN samptest 2000 2>/dev/null | tail -4
  echo "== memory checker"; compute-sanitizer --tool memcheck --error-exitcode 7 $BIN gen "Hi there" none 12 2>&1 | grep -E "ERROR SUMMARY"
  INK_SPEC=4 compute-sanitizer --tool memcheck --error-exitcode 7 $BIN gen "Hi there" none 12 2>&1 | grep -E "ERROR SUMMARY"
  INK_FP8=1 INK_SPEC=4 compute-sanitizer --tool memcheck --error-exitcode 7 $BIN gen "Hi there" none 12 2>&1 | grep -E "ERROR SUMMARY"
fi
echo "ALL CHECKS PASSED"
