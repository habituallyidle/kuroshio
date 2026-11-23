#!/usr/bin/env python3
"""Kuroshio gauntlet: a test-and-benchmark suite for the Kuroshio inference server.

    python3 gauntlet/run.py --suite quick|full [--only NAME,...] [--bin PATH] [--report DIR]

Starts and stops the server itself (one at a time, on port 8020), runs the stages, prints a live summary and writes
report.md / report.json. Exit code: 0 = no stage failed, 1 = at least one FAIL, 2 = the suite could not run.
See gauntlet/README.md.
"""
import argparse
import atexit
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import core  # noqa: E402
from core import FAIL, INFO, PASS, SKIP, WARN, Stage, ServerError, ServerManager, log  # noqa: E402
import report  # noqa: E402
import s1_api, s2_determinism, s3_longctx, s4_robustness, s5_perf, s6_memory, s7_quality, s8_agent, s9_scripts  # noqa: E402,E401

STAGES = [s1_api, s2_determinism, s3_longctx, s4_robustness, s5_perf, s6_memory, s7_quality, s8_agent, s9_scripts]

# Server command lines used by the stages, in the order they are started. Parts of different stages that need the
# same server run back to back, so a whole run starts only one server per configuration.
CONFIGS = [
    ("default", ()),
    ("ctx4k", ("--ctx", "4096")),
    ("spec0", ("--spec", "0")),
    ("nocache", ("--no-prefix-cache",)),
    ("8bit", ("--kv", "8", "--dense", "8")),
    ("reserve", ("--slots", "2", "--reserve", "6g")),
    ("idle", ("--idle-release", "5")),
    ("slots4", ("--slots", "4")),
    (None, None),  # parts that need no server (comparisons across configurations, external scripts)
]


class Ctx:
    def __init__(self, suite, mgr, work, args):
        self.suite, self.full = suite, suite == "full"
        self.mgr, self.work, self.args = mgr, work, args
        self.server = None
        self.shared = {}
        self.binary = mgr.binary
        self.repo = os.path.dirname(HERE)


def resolve_binary(arg):
    b = arg or os.environ.get("KUROSHIO_BIN") or "kuroshio"
    path = shutil.which(b) if os.sep not in b else (b if os.path.isfile(b) and os.access(b, os.X_OK) else None)
    return os.path.abspath(path) if path else None


def binary_info(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    try:
        ver = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        ver = "?"
    return {"path": path, "sha256": h.hexdigest(), "version": ver, "size": os.path.getsize(path)}


def main():
    ap = argparse.ArgumentParser(description="Kuroshio gauntlet")
    ap.add_argument("--suite", choices=["quick", "full"], default="quick")
    ap.add_argument("--only", default="", help="comma-separated stage names: " + ",".join(m.KEY for m in STAGES))
    ap.add_argument("--bin", default=None, help="server binary (default: $KUROSHIO_BIN, else `kuroshio` on PATH)")
    ap.add_argument("--report", default=None, help="report directory (default: gauntlet/reports/<timestamp>/)")
    ap.add_argument("--work", default=None, help="scratch directory for the agent tasks (default: a fresh temp dir)")
    ap.add_argument("--with-scripts", action="store_true", help="also run scripts/verify.sh and scripts/fullcard.sh (they test the repository's own build)")
    ap.add_argument("--soak-seconds", type=int, default=None, help="override the soak duration")
    args = ap.parse_args()

    only = [x.strip() for x in args.only.split(",") if x.strip()]
    keys = [m.KEY for m in STAGES]
    for o in only:
        if o not in keys:
            ap.error(f"unknown stage {o!r}; choose from {', '.join(keys)}")
    binary = resolve_binary(args.bin)
    if not binary:
        print(f"error: server binary not found ({args.bin or os.environ.get('KUROSHIO_BIN') or 'kuroshio on PATH'})", file=sys.stderr)
        return 2
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rdir = os.path.abspath(args.report or os.path.join(HERE, "reports", stamp))
    os.makedirs(rdir, exist_ok=True)
    work = os.path.abspath(args.work) if args.work else tempfile.mkdtemp(prefix="gauntlet-work-")
    os.makedirs(work, exist_ok=True)

    mgr = ServerManager(binary, rdir)
    ctx = Ctx(args.suite, mgr, work, args)
    atexit.register(mgr.stop)

    def on_signal(signum, _frame):
        log(f"signal {signum}: stopping the server and exiting")
        mgr.stop()
        os._exit(130)
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    selected = [m for m in STAGES if (not only or m.KEY in only)]
    stages = {m.KEY: Stage(m.KEY, m.TITLE) for m in selected}
    logs_stage = Stage("logs", "Server logs")
    plan = []  # (config name, stage module, part name, fn)
    for m in selected:
        if m.KEY == "scripts" and not (args.with_scripts or "scripts" in only):
            stages[m.KEY].skip_reason = "optional stage: pass --with-scripts (or --only scripts) to run scripts/verify.sh and scripts/fullcard.sh"
            stages[m.KEY].skip("scripts/verify.sh, scripts/fullcard.sh", stages[m.KEY].skip_reason)
            continue
        for pname, cfg, fn in m.parts(args.suite):
            plan.append((cfg, m, pname, fn))

    meta = {"suite": args.suite, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "binary": binary_info(binary), "only": only,
            "gpu_total_mib": core.gpu_total_mib(), "gpu_used_before_mib": core.gpu_used_mib(), "report_dir": rdir, "work_dir": work,
            "argv": sys.argv[1:]}
    log(f"Kuroshio gauntlet, suite={args.suite}, binary={binary}")
    log(f"report directory: {rdir}")
    t_run = time.monotonic()
    fatal = None

    def write_reports(final=False):
        meta["seconds"] = round(time.monotonic() - t_run, 1)
        meta["servers"] = mgr.history
        meta["finished"] = final
        all_stages = [stages[m.KEY] for m in selected] + [logs_stage]
        report.write(rdir, meta, all_stages, ctx.shared)

    try:
        for cfg_name, cfg_args in CONFIGS:
            todo = [p for p in plan if p[0] == cfg_name]
            if not todo:
                continue
            for _cfg, m, pname, fn in todo:
                st = stages[m.KEY]
                st.part = pname
                srv = None
                if cfg_name is not None:
                    try:
                        srv = mgr.ensure(cfg_args)
                    except ServerError as e:
                        st.server_args = cfg_args
                        st.fail(f"{pname}: the server could not be started", str(e)[:1500], {"expected": "server starts", "actual": str(e)[:600], "raw_request": " ".join([binary, "serve", "--port", str(core.PORT), *cfg_args])})
                        if "already in use" in str(e):
                            raise
                        continue
                else:
                    mgr.stop()
                ctx.server = srv
                st.server_args = cfg_args
                n0 = len(st.checks)
                t0 = time.monotonic()
                log(f"{m.KEY}/{pname}" + (f"  [{' '.join(cfg_args) or 'default'}]" if cfg_name else ""))
                try:
                    fn(st, ctx)
                except Exception:  # noqa: BLE001 - a bug in the suite or an unexpected server reply
                    tb = traceback.format_exc()
                    st.fail(f"{pname}: the gauntlet itself raised an exception (suite error or unexpected server behaviour)", tb[-1500:])
                dt = time.monotonic() - t0
                st.seconds += dt
                if srv is not None and not srv.alive():
                    st.fail(f"{pname}: THE SERVER DIED during this part", f"exit code {srv.proc.returncode}; last log lines:\n{srv.log_tail(12)}",
                            {"expected": "the server keeps running", "actual": f"exit code {srv.proc.returncode}", "raw_request": srv.cmdline(), "note": f"log: {srv.log_path}"})
                new = st.checks[n0:]
                cnt = {s: sum(1 for c in new if c["status"] == s) for s in (PASS, WARN, FAIL, SKIP)}
                log(f"  -> {cnt[PASS]} pass, {cnt[WARN]} warn, {cnt[FAIL]} fail" + (f", {cnt[SKIP]} skip" if cnt[SKIP] else "") + f"  ({dt:.0f}s)")
                write_reports()
    except ServerError as e:
        fatal = str(e)
        log(f"FATAL: {e}")
    finally:
        mgr.stop()

    # every server log of this run must be clean
    bad_total = 0
    for h in mgr.history:
        path = os.path.join(rdir, h["log"])
        try:
            with open(path, "rb") as f:
                text = f.read().decode("utf-8", "replace")
        except OSError:
            continue
        bad = [ln for ln in text.splitlines() if any(b in ln for b in ("panicked", "CUDA error", "Aborted"))]
        bad_total += len(bad)
        logs_stage.server_args = tuple(h["args"])
        logs_stage.check(f"{h['log']}: no 'panicked', 'CUDA error' or 'Aborted'", not bad, "; ".join(bad[:3])[:600] or "clean",
                         {"expected": "no such lines", "actual": "\n".join(bad[:5]), "note": path})
    if not mgr.history:
        logs_stage.skip("server logs", "no server was started")
    if fatal:
        logs_stage.fail("the run was aborted", fatal)
    write_reports(final=True)

    all_stages = [stages[m.KEY] for m in selected] + [logs_stage]
    print()
    print(report.summary_table(all_stages, plain=True))
    print(f"\nreport: {os.path.join(rdir, 'report.md')}")
    left = core.gpu_used_mib()
    print(f"GPU memory in use after the run: {left} MiB; total time {time.monotonic() - t_run:.0f}s")
    if fatal:
        return 2
    return 1 if any(s.status == FAIL for s in all_stages) else 0


if __name__ == "__main__":
    sys.exit(main())
