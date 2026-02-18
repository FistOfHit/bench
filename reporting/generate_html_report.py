#!/usr/bin/env python3
"""
Generate a single-file HTML certification report from HPC Bench results.
Reads JSON files in the results directory; no MAAS or network required.
Usage:
  python3 reporting/generate_html_report.py -i /path/to/results -o /path/to/results/report.html
  HPC_RESULTS_DIR=/path/to/results python3 reporting/generate_html_report.py -o report.html
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(results_dir: Path, name: str) -> dict | None:
    p = results_dir / f"{name}.json"
    if not p.is_file():
        return None
    try:
        with p.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def esc(s: str) -> str:
    if s is None:
        return ""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def badge_class(status: str) -> str:
    s = (status or "").upper()
    if s in ("PASS", "OK"):
        return "badge-pass"
    if s in ("WARN", "WARNING"):
        return "badge-warn"
    if s in ("FAIL", "ERROR"):
        return "badge-fail"
    return "badge-na"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate HTML report from HPC Bench results")
    ap.add_argument("-i", "--input", default=os.environ.get("HPC_RESULTS_DIR", ""), help="Results directory")
    ap.add_argument("-o", "--output", default="", help="Output HTML path (default: <input>/report.html)")
    args = ap.parse_args()
    results_dir = Path(args.input).resolve() if args.input else None
    if not results_dir or not results_dir.is_dir():
        print("Error: need a valid results directory (-i or HPC_RESULTS_DIR)", file=sys.stderr)
        return 1
    out_path = Path(args.output).resolve() if args.output else results_dir / "report.html"

    report = load_json(results_dir, "report")
    run_all = load_json(results_dir, "run-all")
    bootstrap = load_json(results_dir, "bootstrap")
    gpu_inv = load_json(results_dir, "gpu-inventory")

    overall = (report or {}).get("overall", "N/A")
    pass_count = (report or {}).get("pass", 0)
    warn_count = (report or {}).get("warn", 0)
    fail_count = (report or {}).get("fail", 0)
    skip_count = (report or {}).get("skip", 0)
    verdict = (report or {}).get("verdict") or {}
    issues = verdict.get("issues") or []

    hostname = (run_all or {}).get("hostname") or (bootstrap or {}).get("hostname") or "unknown"
    duration = (run_all or {}).get("duration", "N/A")
    start_time = (run_all or {}).get("start_time", "")

    # Module list from report verdict or scan results dir
    modules_with_status = []
    for f in sorted(results_dir.glob("*.json")):
        name = f.stem
        if name in ("report", "run-all"):
            continue
        try:
            with f.open() as fp:
                data = json.load(fp)
            status = data.get("status", "missing")
            note = data.get("skip_reason") or data.get("note") or ""
            modules_with_status.append((name, status, note))
        except (json.JSONDecodeError, OSError):
            continue

    # Hardware
    cpu = (bootstrap or {}).get("cpu_model") or (bootstrap or {}).get("inventory", {}).get("cpu_model") or "N/A"
    ram_gb = "N/A"
    if bootstrap:
        inv = bootstrap.get("inventory") or bootstrap
        ram_gb = inv.get("ram_total_gb") or inv.get("memory_gb") or "N/A"
    gpu_model = "none"
    gpu_count = 0
    if gpu_inv:
        gpu_count = gpu_inv.get("gpu_count", 0) or len(gpu_inv.get("gpus") or [])
        if gpu_inv.get("gpus"):
            gpu_model = gpu_inv["gpus"][0].get("name", "N/A")

    html_parts = []
    html_parts.append(
        """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>HPC Bench Report — """
        + esc(hostname)
        + """</title>
<style>
:root {
  --bg: #0d0f14;
  --card: #12151c;
  --edge: #1f2433;
  --txt: #c8cdd8;
  --txt2: #6b7280;
  --bright: #eef0f6;
  --accent: #38bdf8;
  --green: #22c55e;
  --green-bg: rgba(34,197,94,.08);
  --amber: #eab308;
  --amber-bg: rgba(234,179,8,.08);
  --red: #ef4444;
  --red-bg: rgba(239,68,68,.08);
}
@media print {
  :root { --bg:#fff; --card:#f7f8fa; --edge:#dde0e6; --txt:#1a1c20; --txt2:#5c6070; --bright:#000; }
  body { font-size: 9pt; }
}
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family: system-ui, sans-serif; background: var(--bg); color: var(--txt); line-height: 1.5; padding: 1.5rem; }
.page { max-width: 960px; margin: 0 auto; }
h1 { font-size: 1.5rem; color: var(--bright); margin-bottom: 0.5rem; }
h2 { font-size: 1.1rem; color: var(--bright); margin: 1.2rem 0 0.5rem; border-bottom: 1px solid var(--edge); padding-bottom: 0.25rem; }
.section-label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--txt2); margin-bottom: 0.5rem; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
.badge-pass { background: var(--green-bg); color: var(--green); }
.badge-warn { background: var(--amber-bg); color: var(--amber); }
.badge-fail { background: var(--red-bg); color: var(--red); }
.badge-na { background: var(--card); color: var(--txt2); }
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; margin: 0.5rem 0; }
th, td { padding: 0.35rem 0.5rem; text-align: left; border-bottom: 1px solid var(--edge); }
th { color: var(--txt2); font-weight: 600; }
.kv-table td:first-child { color: var(--txt2); width: 140px; }
.issues-list { list-style: none; }
.issues-list li { padding: 0.35rem 0; border-bottom: 1px solid var(--edge); }
.footer { margin-top: 2rem; font-size: 0.75rem; color: var(--txt2); }
</style>
</head>
<body>
<div class="page">
<header>
  <h1>HPC Benchmarking Report</h1>
  <p class="section-label">"""
        + esc(hostname)
        + """ &middot; """
        + esc(start_time)
        + """ &middot; """
        + esc(str(duration))
        + """</p>
</header>

<div class="section-label">Device result</div>
<p><span class="badge """
        + badge_class(overall)
        + """">"""
        + esc(overall)
        + """</span> &nbsp; """
        + str(pass_count)
        + """ passed, """
        + str(warn_count)
        + """ warnings, """
        + str(fail_count)
        + """ failed, """
        + str(skip_count)
        + """ skipped</p>

<h2>Health Scorecard</h2>
<table>
<thead><tr><th>Module</th><th>Status</th><th>Notes</th></tr></thead>
<tbody>
"""
    )
    for name, status, note in modules_with_status:
        html_parts.append(
            f"<tr><td>{esc(name)}</td><td><span class=\"badge {badge_class(status)}\">{esc(status)}</span></td><td>{esc(note)}</td></tr>\n"
        )
    html_parts.append(
        """</tbody>
</table>

<h2>Hardware</h2>
<table class="kv-table">
<tr><td>Hostname</td><td>"""
        + esc(hostname)
        + """</td></tr>
<tr><td>CPU</td><td>"""
        + esc(str(cpu))
        + """</td></tr>
<tr><td>Memory</td><td>"""
        + esc(str(ram_gb))
        + """ GB</td></tr>
<tr><td>GPUs</td><td>"""
        + esc(str(gpu_count))
        + " &times; "
        + esc(str(gpu_model))
        + """</td></tr>
</table>
"""
    )
    if issues:
        html_parts.append("<h2>Issues</h2>\n<ul class=\"issues-list\">\n")
        for i in issues:
            mod = i.get("module", "")
            issue = i.get("issue", "")
            sev = i.get("severity", "")
            html_parts.append(f"<li><span class=\"badge {badge_class(sev)}\">{esc(sev)}</span> <strong>{esc(mod)}:</strong> {esc(issue)}</li>\n")
        html_parts.append("</ul>\n")

    version = (report or {}).get("suite_version") or "unknown"
    html_parts.append(
        f"""
<p class="footer">Report generated by HPC Bench Suite v{esc(version)}. Open <code>report.md</code> in this directory for the full markdown report.</p>
</div>
</body>
</html>
"""
    )

    try:
        out_path.write_text("".join(html_parts), encoding="utf-8")
    except OSError as e:
        print(f"Error writing {out_path}: {e}", file=sys.stderr)
        return 1
    print(f"Wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
