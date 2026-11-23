"""Optional stage - the repository's own checks: scripts/verify.sh (engine-level correctness) and
scripts/fullcard.sh (another program fills the GPU). They test the repository's build (target/release/kuroshio),
not the binary given with --bin, and are therefore only run on request (--with-scripts or --only scripts)."""
import os
import subprocess

KEY, TITLE = "scripts", "Repository scripts (optional)"

TIMEOUT_S = 900  # verify.sh takes about a minute, fullcard.sh about two; anything near 15 minutes is a hang


def _run(st, ctx, script, extra=""):
    path = os.path.join(ctx.repo, "scripts", script)
    binary = os.path.join(ctx.repo, "target", "release", "kuroshio")
    if not os.path.isfile(path):
        st.skip(script, f"{path} does not exist")
        return
    if not os.path.isfile(binary):
        st.skip(script, f"{binary} does not exist (the script tests the repository's own build)")
        return
    out = os.path.join(ctx.mgr.log_dir, script.replace(".sh", "") + ".out")
    cmd = f"source scripts/env.sh && scripts/{script} {extra}"
    try:
        with open(out, "wb") as f:
            r = subprocess.run(["bash", "-c", cmd], cwd=ctx.repo, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, timeout=TIMEOUT_S)
        code = r.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    with open(out, "rb") as f:
        tail = "\n".join(f.read().decode("utf-8", "replace").splitlines()[-8:])
    st.check(f"scripts/{script} {extra}".strip() + " exits 0", code == 0, f"exit {code}; last lines:\n{tail}",
             {"raw_request": f"cd {ctx.repo} && {cmd}", "expected": "exit code 0", "actual": f"exit {code}", "note": f"full output: {out}"})


def verify(st, ctx):
    _run(st, ctx, "verify.sh", "full" if ctx.full else "")


def fullcard(st, ctx):
    _run(st, ctx, "fullcard.sh")


def parts(suite):
    return [("verify", None, verify), ("fullcard", None, fullcard)]
