"""Stage 8 - real agent runs: the `pi` coding agent works on tiny tasks against the server; the gauntlet then runs
each task's check itself.

pi executes shell commands written by the model, so every task runs in its own fresh directory under the work
directory (--work), with a timeout, and nothing outside that directory is referenced by the tasks."""
import hashlib
import os
import shutil
import signal
import subprocess
import sys

from core import healthy, median, now, repro

KEY, TITLE = "agent", "Agent runs with pi"

PI_ARGS = ["-p", "--no-session", "--no-extensions", "--no-mcp", "--no-skills", "--no-context-files",
           "--provider", "kuroshio", "--model", "inkling-2-mini", "--thinking", "low"]
TASK_TIMEOUT_S = 240   # each task is a handful of model turns; a run that takes longer is stuck in a loop
CHECK_TIMEOUT_S = 30
# Share of tasks the agent must complete. The tasks are tiny (one-line fixes), so a working stack solves nearly
# all of them; below 2/3 is a WARN, at or below 1/3 a FAIL. With a single task (quick suite) a miss is a WARN.
PASS_RATE, FAIL_RATE = 0.66, 0.34
# In an agent loop every request repeats the whole conversation so far; the prefix cache is what makes that
# cheap. Averaged over a task, less than half of all prompt tokens coming from the cache means it is not working.
CACHE_SHARE_WARN_BELOW = 0.5

TASKS = [
    {
        "id": "fix-bug",
        "files": {
            "calc.py": 'def average(values):\n    """Return the arithmetic mean of a non-empty list of numbers."""\n    return sum(values) / (len(values) - 1)\n',
            "test_calc.py": 'from calc import average\n\nassert average([2, 4, 6]) == 4, average([2, 4, 6])\nassert average([5]) == 5\nassert average([1, 2]) == 1.5\nprint("ok")\n',
        },
        "prompt": "The test in test_calc.py fails. Fix the bug in calc.py so that `python3 test_calc.py` passes. Do not modify test_calc.py.",
        "check": ["python3", "test_calc.py"], "frozen": ["test_calc.py"],
    },
    {
        "id": "add-function",
        "files": {
            "textutil.py": 'def shout(text):\n    """Upper-case the text and add an exclamation mark."""\n    return text.upper() + "!"\n',
            "test_textutil.py": ('from textutil import shout, slugify\n\nassert shout("hi") == "HI!"\nassert slugify("Hello, World!") == "hello-world"\n'
                                 'assert slugify("  Kuroshio   Engine ") == "kuroshio-engine"\nassert slugify("a_b c") == "a-b-c"\nassert slugify("2 Fast 2 Furious") == "2-fast-2-furious"\nprint("ok")\n'),
        },
        "prompt": "Add a function slugify(text) to textutil.py so that `python3 test_textutil.py` passes. Read the test to see what it must do. Do not modify the test file.",
        "check": ["python3", "test_textutil.py"], "frozen": ["test_textutil.py"],
    },
    {
        "id": "rename",
        "files": {
            "geometry.py": 'def calc_area(width, height):\n    """Area of a rectangle."""\n    return width * height\n\n\ndef perimeter(width, height):\n    return 2 * (width + height)\n',
            "main.py": 'from geometry import calc_area, perimeter\n\n\ndef describe(w, h):\n    return f"area={calc_area(w, h)} perimeter={perimeter(w, h)}"\n\n\nif __name__ == "__main__":\n    print(describe(3, 4))\n',
            "test_rename.py": ('import subprocess, sys\nimport geometry\n\nassert hasattr(geometry, "rectangle_area"), "geometry.rectangle_area is missing"\nassert not hasattr(geometry, "calc_area"), "calc_area still exists"\n'
                               'assert geometry.rectangle_area(3, 4) == 12\nout = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True)\nassert out.stdout.strip() == "area=12 perimeter=14", out.stdout + out.stderr\n'
                               'for name in ("geometry.py", "main.py"):\n    assert "calc_area" not in open(name).read(), name + " still mentions calc_area"\nprint("ok")\n'),
        },
        "prompt": "Rename the function calc_area to rectangle_area everywhere it is defined or used (geometry.py and main.py). Afterwards `python3 test_rename.py` must pass. Do not modify test_rename.py.",
        "check": ["python3", "test_rename.py"], "frozen": ["test_rename.py"],
    },
    {
        "id": "script-output",
        "files": {
            "data.csv": "name,score\nmira,71\njonas,88\nade,93\nlin,64\n",
            "SPEC.md": ("Write a Python script `make_report.py` that reads `data.csv` (columns: name, score) and writes `report.txt`.\n\n`report.txt` must contain one line per person, sorted by score from "
                        "highest to lowest, in the form `<rank>. <name>: <score>` (rank starts at 1), followed by a final line `average: <mean of the scores with exactly two decimals>`.\n"),
        },
        "prompt": "Read SPEC.md and do what it says: write make_report.py, then run it with python3 so that report.txt exists.",
        "check": ["python3", "make_report.py"], "frozen": ["data.csv"],
        "expect_file": ("report.txt", "1. ade: 93\n2. jonas: 88\n3. mira: 71\n4. lin: 64\naverage: 79.00\n"), "remove_before_check": ["report.txt"],
    },
    {
        "id": "two-files-question",
        "files": {
            "config.json": '{\n  "service": "billing",\n  "region": "eu-north-2",\n  "replicas": 3\n}\n',
            "regions.txt": "us-east-1 = datacenter Ashburn, code 1180\neu-north-2 = datacenter Kiruna, code 7421\neu-west-3 = datacenter Paris, code 3309\nap-south-1 = datacenter Mumbai, code 5562\n",
        },
        "prompt": "Read config.json and regions.txt. What is the datacenter code for the region this service is configured for? Reply with the code.",
        "stdout_contains": "7421", "stdout_excludes": ["1180", "3309", "5562"], "frozen": ["config.json", "regions.txt"],
    },
    {
        "id": "off-by-one",
        "files": {
            "pagination.py": ('def paginate(items, page_size):\n    """Split items into consecutive pages of at most page_size items."""\n    pages = []\n'
                              '    for start in range(0, len(items) - 1, page_size):\n        pages.append(items[start:start + page_size])\n    return pages\n'),
            "test_pagination.py": ('import unittest\nfrom pagination import paginate\n\n\nclass PaginateTest(unittest.TestCase):\n    def test_even(self):\n        self.assertEqual(paginate([1, 2, 3, 4], 2), [[1, 2], [3, 4]])\n\n'
                                   '    def test_last_page_of_one(self):\n        self.assertEqual(paginate([1, 2, 3, 4, 5], 2), [[1, 2], [3, 4], [5]])\n\n    def test_single(self):\n        self.assertEqual(paginate([7], 3), [[7]])\n\n'
                                   '    def test_empty(self):\n        self.assertEqual(paginate([], 3), [])\n\n\nif __name__ == "__main__":\n    unittest.main()\n'),
        },
        "prompt": "Run `python3 test_pagination.py`. Some tests fail. Find and fix the bug in pagination.py so that all tests pass. Do not modify the test file.",
        "check": ["python3", "test_pagination.py"], "frozen": ["test_pagination.py"],
    },
]


def _sha(path):
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def _run_check(cmd, cwd):
    try:
        r = subprocess.run(cmd, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=CHECK_TIMEOUT_S,
                           env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": cwd, "PYTHONDONTWRITEBYTECODE": "1"})
        return r.returncode, (r.stdout + r.stderr)[-600:]
    except subprocess.TimeoutExpired:
        return "timeout", ""


def run_task(ctx, task, root):
    srv = ctx.server
    d = os.path.join(root, task["id"])
    os.makedirs(d)
    for name, content in task["files"].items():
        with open(os.path.join(d, name), "w") as f:
            f.write(content)
    frozen = {n: _sha(os.path.join(d, n)) for n in task.get("frozen", [])}
    # make sure the starting state really fails, so a pass means the agent did something
    pre = _run_check(task["check"], d)[0] if "check" in task and "expect_file" not in task else None
    off = srv.log_size()
    out_path, err_path = os.path.join(root, task["id"] + ".stdout"), os.path.join(root, task["id"] + ".stderr")
    t0 = now()
    timed_out = False
    with open(out_path, "wb") as fo, open(err_path, "wb") as fe:
        proc = subprocess.Popen(["pi", *PI_ARGS, task["prompt"]], cwd=d, stdin=subprocess.DEVNULL, stdout=fo, stderr=fe, start_new_session=True)
        try:
            code = proc.wait(timeout=TASK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(proc.pid, signal.SIGKILL)  # the group we created for this pi run, nothing else
            except OSError:
                pass
            code = proc.wait()
    wall = now() - t0
    with open(out_path, "rb") as f:
        stdout = f.read().decode("utf-8", "replace")
    with open(err_path, "rb") as f:
        stderr = f.read().decode("utf-8", "replace")
    reqs = srv.request_lines(off)
    failed_reqs = srv.log_since(off).count("request failed:")
    why = []
    if timed_out:
        why.append(f"pi timed out after {TASK_TIMEOUT_S}s")
    elif code != 0:
        why.append(f"pi exited with code {code}: {stderr.strip()[-200:]}")
    for n, h in frozen.items():
        if _sha(os.path.join(d, n)) != h:
            why.append(f"{n} was modified (the task forbids it)")
    if "check" in task:
        for n in task.get("remove_before_check", []):
            try:
                os.remove(os.path.join(d, n))
            except OSError:
                pass
        rc, tail = _run_check(task["check"], d)
        if rc != 0:
            why.append(f"`{' '.join(task['check'])}` exited {rc}: {tail.strip()[-300:]}")
    if "expect_file" in task:
        name, want = task["expect_file"]
        try:
            with open(os.path.join(d, name)) as f:
                got = f.read()
        except OSError:
            got = None
        if got is None or got.rstrip() != want.rstrip():
            why.append(f"{name} is {got!r}, expected {want!r}")
    if "stdout_contains" in task:
        if task["stdout_contains"] not in stdout or any(x in stdout for x in task.get("stdout_excludes", [])):
            why.append(f"answer {stdout.strip()[-200:]!r} does not single out {task['stdout_contains']}")
    prompt_tok, cached_tok = sum(r["prompt"] for r in reqs), sum(r["cached"] for r in reqs)
    return {"id": task["id"], "ok": not why, "why": "; ".join(why), "wall": wall, "requests": len(reqs), "failed_requests": failed_reqs,
            "prompt_tokens": prompt_tok, "cached_tokens": cached_tok, "cache_share": (cached_tok / prompt_tok) if prompt_tok else None,
            "ttft_followup_ms": [r["ttft_ms"] for r in reqs if r["cached"] > 0], "gen_tokens": sum(r["tokens"] for r in reqs),
            "pre_check_failed": (pre != 0) if pre is not None else None, "dir": d, "stdout_tail": stdout.strip()[-300:]}


def agent(st, ctx):
    if not shutil.which("pi"):
        st.skip_reason = "the `pi` coding agent is not installed (not on PATH)"
        st.skip("agent runs", st.skip_reason)
        return
    root = os.path.join(ctx.work, f"agent-{os.getpid()}-{int(now())}")
    os.makedirs(root)
    tasks = TASKS if ctx.full else TASKS[:1]
    results = []
    for task in tasks:
        r = run_task(ctx, task, root)
        results.append(r)
        cs = f"{r['cache_share']:.0%}" if r["cache_share"] is not None else "n/a"
        st.num(f"task {r['id']}", f"{'solved' if r['ok'] else 'NOT solved'} in {r['wall']:.0f}s, {r['requests']} requests, cache share {cs}")
        if r["pre_check_failed"] is False:
            st.fail(f"task {r['id']}: the starting state already passes its check (broken task definition)", "")
        if not r["ok"]:
            st.info(f"task {r['id']} not solved", r["why"][:500] + f" | pi output: {r['stdout_tail'][:200]!r} | dir: {r['dir']}")
        if not healthy():
            st.fail(f"task {r['id']}: the server is not healthy after the agent run", "")
    ctx.shared.setdefault("report", {})["agent_tasks"] = results
    n_ok, n = sum(r["ok"] for r in results), len(results)
    total_req = sum(r["requests"] for r in results)
    pt, ct = sum(r["prompt_tokens"] for r in results), sum(r["cached_tokens"] for r in results)
    ttfts = [t for r in results for t in r["ttft_followup_ms"]]
    st.num("tasks solved", f"{n_ok}/{n}", head=True)
    st.num("wall time per task, median (max)", f"{median([r['wall'] for r in results]):.0f} ({max(r['wall'] for r in results):.0f})", "s")
    st.num("model requests in total", total_req, head=True)
    st.num("prompt tokens served from the cache", f"{ct}/{pt} ({(100.0 * ct / pt) if pt else 0:.0f}%)", head=True)
    if ttfts:
        st.num("server-side time to first token of cached follow-up requests, median (min-max)", f"{median(ttfts):.0f} ({min(ttfts)}-{max(ttfts)})", "ms")
    how = "pi " + " ".join(PI_ARGS) + " \"<task prompt>\" < /dev/null, in a fresh directory holding the task files (see s8_agent.TASKS)"
    if total_req == 0:
        st.fail("pi reached the server", "no 'request:' line appeared in the server log during the agent runs; " + results[0]["why"][:300],
                repro(raw=how, expected="pi sends chat completions to the server", actual=results[0]["why"][:300] + " | " + results[0]["stdout_tail"][:200]))
        return
    st.passed("pi reached the server", f"{total_req} requests")
    rate = n_ok / n
    failed = "; ".join(f"{r['id']}: {r['why'][:200]}" for r in results if not r["ok"])
    rp = repro(raw=how, expected="the task's check passes afterwards", actual=failed)
    if n == 1:
        st.check("the agent solves the task", n_ok == 1, failed or "solved", rp, warn_only=True)
    elif rate <= FAIL_RATE:
        st.fail(f"the agent solves at least {PASS_RATE:.0%} of the tasks", f"{n_ok}/{n}; {failed}", rp)
    elif rate < PASS_RATE:
        st.warn(f"the agent solves at least {PASS_RATE:.0%} of the tasks", f"{n_ok}/{n}; {failed}", rp)
    else:
        st.passed(f"the agent solves at least {PASS_RATE:.0%} of the tasks", f"{n_ok}/{n}" + (f"; not solved: {failed}" if failed else ""))
    if pt:
        st.check(f"at least {CACHE_SHARE_WARN_BELOW:.0%} of the agent's prompt tokens come from the prefix cache", ct / pt >= CACHE_SHARE_WARN_BELOW,
                 f"{ct}/{pt} ({100.0 * ct / pt:.0f}%)", repro(raw=how, expected="most of each follow-up prompt is cached", actual=f"{100.0 * ct / pt:.0f}%"), warn_only=True)
    bad_reqs = sum(r["failed_requests"] for r in results)
    st.check("no request of the agent failed on the server side", bad_reqs == 0, f"{bad_reqs} 'request failed:' lines in the server log during the agent runs", warn_only=True)


def parts(suite):
    return [("pi", "default", agent)]
