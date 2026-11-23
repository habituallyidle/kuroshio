"""Writes report.md and report.json."""
import json
import os

from core import FAIL, INFO, PASS, SKIP, WARN


def _numbers_brief(st, limit=6):
    """The stage's headline numbers (those recorded with head=True; the first few otherwise)."""
    items = [(k, v) for k, v in st.numbers.items() if v.get("head")] or list(st.numbers.items())[:3]
    out = []
    for k, v in items[:limit]:
        val = v["value"]
        if isinstance(val, float):
            val = f"{val:.4g}"
        out.append(f"{k}: {val}{(' ' + v['unit']) if v['unit'] else ''}")
    return out


def summary_table(stages, plain=False):
    rows = []
    for st in stages:
        c = st.counts()
        rows.append((st.title, st.status, f"{c[PASS]}/{c[WARN]}/{c[FAIL]}/{c[SKIP]}", f"{st.seconds:.0f}s", "; ".join(_numbers_brief(st))))
    if plain:
        w0 = max(len(r[0]) for r in rows)
        lines = [f"{'STAGE'.ljust(w0)}  STATUS  pass/warn/fail/skip   time  key numbers"]
        for r in rows:
            lines.append(f"{r[0].ljust(w0)}  {r[1].ljust(6)}  {r[2].ljust(19)}  {r[3].rjust(5)}  {r[4][:260]}")
        return "\n".join(lines)
    lines = ["| Stage | Status | pass / warn / fail / skip | Time | Key numbers |", "|---|---|---|---:|---|"]
    for r in rows:
        lines.append(f"| {r[0]} | **{r[1]}** | {r[2].replace('/', ' / ')} | {r[3]} | {r[4].replace('|', '/')} |")
    return "\n".join(lines)


def _esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def _repro_md(c):
    r = c.get("repro") or {}
    lines = []
    if c.get("server") is not None:
        lines.append(("- server: `kuroshio serve --port 8020 " + c["server"]).rstrip() + "`")
    if r.get("curl"):
        lines.append("- request:\n\n  ```bash\n  " + r["curl"] + "\n  ```")
    elif r.get("request"):
        lines.append("- request (long strings elided): `" + r["request"][:1500].replace("`", "'") + "`")
    if r.get("raw_request"):
        lines.append("- how: " + str(r["raw_request"])[:600])
    if r.get("note"):
        lines.append("- note: " + str(r["note"])[:600])
    if r.get("expected"):
        lines.append("- expected: " + str(r["expected"])[:600])
    if r.get("actual"):
        lines.append("- actual: " + str(r["actual"])[:1200].replace("\n", "\n  "))
    return "\n".join(lines)


def write(rdir, meta, stages, shared):
    overall = FAIL if any(s.status == FAIL for s in stages) else (WARN if any(s.status == WARN for s in stages) else PASS)
    data = {
        "meta": meta,
        "overall": overall,
        "stages": [{"key": s.key, "title": s.title, "status": s.status, "seconds": round(s.seconds, 1), "counts": s.counts(),
                    "numbers": s.numbers, "checks": s.checks, "notes": s.notes, "skip_reason": s.skip_reason} for s in stages],
        "claims": shared.get("claims", []),
        "extra": shared.get("report", {}),
    }
    with open(os.path.join(rdir, "report.json"), "w") as f:
        json.dump(data, f, indent=1, ensure_ascii=False, default=str)

    b = meta["binary"]
    md = ["# Kuroshio gauntlet report", ""]
    md.append(f"- **Overall: {overall}**" + ("" if meta.get("finished") else " (run still in progress or interrupted: partial report)"))
    md.append(f"- suite `{meta['suite']}`" + (f", only: {', '.join(meta['only'])}" if meta.get("only") else "") + f", started {meta['started']}, took {meta.get('seconds', 0):.0f} s")
    md.append(f"- binary `{b['path']}` ({b['version'] or 'version unknown'}, sha256 `{b['sha256'][:16]}…`, {b['size']} bytes)")
    md.append(f"- GPU: {meta.get('gpu_total_mib')} MiB total, {meta.get('gpu_used_before_mib')} MiB in use by other programs before the run")
    md += ["", "Status of a stage is the worst of its checks: FAIL > WARN > PASS. SKIP means the stage (or check) could not run; INFO lines are observations.", ""]
    md.append(summary_table(stages))
    md.append("")

    claims = shared.get("claims") or []
    if claims:
        md += ["## README claims against measurements", "",
               "WARN: more than 15% worse than claimed; FAIL: more than 35% worse. How each number is measured is described in gauntlet/README.md.", "",
               "| Claim (README.md) | Claimed | Measured | Difference | Status |", "|---|---:|---:|---:|---|"]
        for c in claims:
            md.append(f"| {_esc(c['name'])} | {c['claim']} | {c['measured']} | {c['delta']} | {c['status']} |")
        md.append("")

    problems = [(s, c) for s in stages for c in s.checks if c["status"] == FAIL]
    warns = [(s, c) for s in stages for c in s.checks if c["status"] == WARN]
    for title, items in (("Failures", problems), ("Warnings", warns)):
        md += [f"## {title} ({len(items)})", ""]
        if not items:
            md += ["None.", ""]
        for s, c in items:
            md.append(f"### [{s.key}{'/' + c['part'] if c.get('part') else ''}] {c['name']}")
            md.append("")
            if c["detail"]:
                md.append("```\n" + c["detail"][:1500] + "\n```")
            rp = _repro_md(c)
            if rp:
                md.append(rp)
            md.append("")

    md += ["## Stages in detail", ""]
    for s in stages:
        md.append(f"### {s.title} — {s.status}")
        md.append("")
        if s.skip_reason:
            md += [f"Skipped: {s.skip_reason}", ""]
        if s.numbers:
            md += ["| Measurement | Value |", "|---|---:|"]
            for k, v in s.numbers.items():
                val = v["value"]
                if isinstance(val, float):
                    val = f"{val:.4g}"
                unit = "" if v["unit"] in ("", "detail") else " " + v["unit"]
                md.append(f"| {_esc(k)} | {_esc(val)}{unit} |")
            md.append("")
        for n in s.notes:
            md += [n, ""]
        if s.checks:
            md += ["| | Check | Detail |", "|---|---|---|"]
            for c in s.checks:
                d = c["detail"]
                d = d if len(d) <= 220 else d[:220] + "…"
                md.append(f"| {c['status']} | {_esc(c['name'])} | {_esc(d)} |")
            md.append("")

    md += ["## Servers started", "", "| # | Command line | Ready after | Log |", "|---:|---|---:|---|"]
    for i, h in enumerate(meta.get("servers") or [], 1):
        md.append(f"| {i} | `serve --port 8020 {' '.join(h['args'])}` | {h['load_s']} s | {h['log']} |")
    md.append("")
    with open(os.path.join(rdir, "report.md"), "w") as f:
        f.write("\n".join(md))
