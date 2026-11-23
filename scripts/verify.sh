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
$BIN argmaxes 1500 > "$TMP/b16.txt" 2>/dev/null; INK_CHUNK=1 $BIN argmaxes 1500 > "$TMP/b1.txt" 2>/dev/null; for c in 128 100 56 48 38 37 30 24 9 8; do INK_CHUNK=$c $BIN argmaxes 1500 > "$TMP/bx.txt" 2>/dev/null
  cmp -s "$TMP/b16.txt" "$TMP/bx.txt" || { echo "FAIL: batch-$c and batch-16 greedy tokens differ"; exit 1; }; done   # 100 and 48 straddle the 1,024-token bucket boundary; 37/38 the conv-history limit; 8/9, 24, 30, 56 the kernels' row groups
cmp -s "$TMP/b16.txt" "$TMP/b1.txt" || { echo "FAIL: batch-16 and batch-1 greedy tokens differ"; exit 1; }
# prompt-only steps of up to 256 rows (attention in two halves, experts once) must leave the same state as 16-row steps:
# what follows them must come out the same. The sliding-window caches are rings of 1,152 positions, so the steps have
# to go past position 1,152 to prove anything.
$BIN argmaxes 2600 > "$TMP/p16.txt" 2>/dev/null
for pre in 2304,256 2400,200 1408,129; do INK_PRE=$pre $BIN argmaxes 2600 > "$TMP/bx.txt" 2>/dev/null; n=${pre%%,*}
  cmp -s <(cut -d' ' -f$((n + 1))- "$TMP/p16.txt") <(cut -d' ' -f$((n + 1))- "$TMP/bx.txt") || { echo "FAIL: after prompt steps of ${pre##*,} rows the tokens differ"; exit 1; }; done
# the matrix kernel that packs its inputs first (batches of 9+) must agree with the plain one
INK_PACK_MIN=999 INK_CHUNK=100 $BIN argmaxes 1500 > "$TMP/bx.txt" 2>/dev/null; cmp -s "$TMP/b16.txt" "$TMP/bx.txt" || { echo "FAIL: plain and packed-input matrix kernels differ"; exit 1; }; echo "ok"
echo "== conversations advanced together must match separate runs"
$BIN interleave 3 1300 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: interleaved conversations differ from separate runs"; exit 1; }; echo "ok"
echo "== drafts computed for several conversations at once must match one at a time"
$BIN draftcheck 42 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: batched drafts differ"; exit 1; }; echo "ok"
echo "== with a seed, sampling must give the same text with and without speculation"
A=$(INK_TEMP=0.8 INK_SEED=7 INK_SPEC=0 $BIN gen "Tell a short story about a fox and a crow." none 160 2>/dev/null | md5sum)
B=$(INK_TEMP=0.8 INK_SEED=7 INK_SPEC=7 INK_TAU=0 $BIN gen "Tell a short story about a fox and a crow." none 160 2>/dev/null | md5sum)
C=$(INK_TEMP=0.8 INK_SEED=7 INK_SPEC=3 INK_TAU=0.5 $BIN gen "Tell a short story about a fox and a crow." none 160 2>/dev/null | md5sum)
[ "$A" = "$B" ] && [ "$A" = "$C" ] || { echo "FAIL: seeded sampling depends on speculation"; exit 1; }; echo "ok"
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
