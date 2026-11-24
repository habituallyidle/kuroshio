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
# prompt-only steps of up to 1,024 rows (attention in sub-batches of 128, experts once) must leave the same state as 16-row steps:
# what follows them must come out the same. The sliding-window caches are rings of 1,152 positions, so the steps have
# to go past position 1,152 to prove anything. (With --exact-prompts; the default prompt path has its own checks below.)
$BIN argmaxes 2600 > "$TMP/p16.txt" 2>/dev/null
for pre in 2048,1024 2560,512 2304,256 2400,200 1408,129; do INK_EXACT_PROMPTS=1 INK_PRE=$pre $BIN argmaxes 2600 > "$TMP/bx.txt" 2>/dev/null; n=${pre%%,*}
  cmp -s <(cut -d' ' -f$((n + 1))- "$TMP/p16.txt") <(cut -d' ' -f$((n + 1))- "$TMP/bx.txt") || { echo "FAIL: after prompt steps of ${pre##*,} rows the tokens differ"; exit 1; }; done
# the matrix kernel that packs its inputs first (batches of 9+) must agree with the plain one
for knobs in "INK_BULK_MIN=999" "INK_BULK_MIN=999 INK_PACK_MIN=999"; do    # the older packed-input kernel, then the plain one, against the bulk kernel
  env $knobs INK_CHUNK=100 $BIN argmaxes 1500 > "$TMP/bx.txt" 2>/dev/null; cmp -s "$TMP/b16.txt" "$TMP/bx.txt" || { echo "FAIL: the matrix kernels differ ($knobs)"; exit 1; }
done; echo "ok"
echo "== the two-stage output head must pick what the full head picks (argmax and samples; error bound never exceeded)"
$BIN headcheck 1200 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: two-stage head differs from the full head"; $BIN headcheck 1200 2>/dev/null | tail -3; exit 1; }; echo "ok"
echo "== conversations advanced together must match separate runs"
$BIN interleave 3 1300 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: interleaved conversations differ from separate runs"; exit 1; }; echo "ok"
# several prompts read together in steps of up to 256 rows, then the same comparison
INK_EXACT_PROMPTS=1 INK_IL_PRE=1250 $BIN interleave 3 1300 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: after prompt steps shared by several conversations the tokens differ"; exit 1; }; echo "ok"
# the same when the two-stage head always gives up and the full head runs after it (its overflow path)
INK_HEAD_FALLBACK=1 $BIN interleave 3 500 2>/dev/null | tail -1 | grep -q IDENTICAL || { echo "FAIL: the output head's fall-back path gives different tokens"; exit 1; }; echo "ok"
echo "== the fast prompt path (the default): the same result whatever the step size, and close to the exact path"
# What it computes for a row depends on that row alone, so prompt steps of any size must leave the same state.
for rows in 1024 512 200 129 37 5; do INK_PRE=2400,$rows $BIN argmaxes 2900 > "$TMP/f$rows.txt" 2>/dev/null
  cmp -s "$TMP/f1024.txt" "$TMP/f$rows.txt" || { echo "FAIL: fast prompt steps of $rows rows and of 1,024 rows leave different states"; exit 1; }; done
INK_EXACT_PROMPTS=1 INK_PRE=2400,1024 $BIN argmaxes 2900 > "$TMP/fe.txt" 2>/dev/null
same=$(paste -d' ' <(tr ' ' '\n' < "$TMP/fe.txt" | tail -n 500) <(tr ' ' '\n' < "$TMP/f1024.txt" | tail -n 500) | awk '$1 == $2' | wc -l)
[ "$same" -ge 450 ] || { echo "FAIL: after a fast-read prompt only $same of 500 greedy tokens agree with the exact path (about 485 expected)"; exit 1; }
# the fast path's attention and gate kernels against the exact ones on the same inputs: the difference must stay at rounding level
INK_MB_T=128 $BIN microbench 1536 2>/dev/null | grep -E "(fused|fast) against exact" > "$TMP/fused.txt"; cat "$TMP/fused.txt"
[ "$(wc -l < "$TMP/fused.txt")" -eq 3 ] && awk '{ lim = /gate/ ? 0.00001 : 0.002; if ($NF + 0 > lim || $NF + 0 <= 0) bad = 1 } END { exit bad }' "$TMP/fused.txt" || { echo "FAIL: a fast prompt kernel is further from the exact kernels than rounding explains"; exit 1; }
NE=$(INK_EXACT_PROMPTS=1 INK_PRE=3072,1024 $BIN nll 4000 2>/dev/null | tail -1 | sed -E 's/.*NLL ([0-9.]+).*/\1/'); NF_=$(INK_PRE=3072,1024 $BIN nll 4000 2>/dev/null | tail -1 | sed -E 's/.*NLL ([0-9.]+).*/\1/')
echo "NLL of the text after a 3,072-token prompt: $NE read exactly, $NF_ read fast ($same of 500 greedy tokens agree)"
awk -v a="$NE" -v b="$NF_" 'BEGIN { d = a - b; if (d < 0) d = -d; exit !(a > 0 && b > 0 && d < 0.03) }' || { echo "FAIL: quality after a fast-read prompt differs from the exact path by more than noise"; exit 1; }; echo "ok"
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
# The numbers are a fingerprint of the engine's arithmetic: every change so far that was meant to be bit-identical left
# them unchanged to the last digit. A different value is not necessarily wrong, but it must be intended: update it here.
N16=$($BIN nll 4000 2>/dev/null | tail -1); N8=$(INK_FP8=1 $BIN nll 4000 2>/dev/null | tail -1); echo "$N16"; echo "8-bit mode: $N8"
case "$N16" in *"NLL 0.9007 "*) ;; *) echo "FAIL: the lossless mode's NLL on docs/ARCHITECTURE.md was 0.9007 (the arithmetic changed, or that file did)"; exit 1;; esac
case "$N8" in *"NLL 0.9068 "*) ;; *) echo "FAIL: the 8-bit mode's NLL on docs/ARCHITECTURE.md was 0.9068 (the arithmetic changed, or that file did)"; exit 1;; esac
if [ "${1:-}" = full ]; then
  echo "== sampling distribution with and without speculation"; $BIN samptest 2000 2>/dev/null | tail -4
  echo "== memory checker"; compute-sanitizer --tool memcheck --error-exitcode 7 $BIN gen "Hi there" none 12 2>&1 | grep -E "ERROR SUMMARY"
  INK_SPEC=4 compute-sanitizer --tool memcheck --error-exitcode 7 $BIN gen "Hi there" none 12 2>&1 | grep -E "ERROR SUMMARY"
  INK_FP8=1 INK_SPEC=4 compute-sanitizer --tool memcheck --error-exitcode 7 $BIN gen "Hi there" none 12 2>&1 | grep -E "ERROR SUMMARY"
fi
echo "ALL CHECKS PASSED"
