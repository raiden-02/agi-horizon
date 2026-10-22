"""CLI.  python -m horizon <command>

  run      evaluate models (default: roster.json) on all axes
  score    fit ability estimates -> results/summary.json
  report   render docs/index.html from the summary
  human    take the eval yourself (human baseline)
  show     print a sample task (public salt) for an axis and level
  lock     freeze the pre-registration hash (before the first scored run)
  verify   check the code still matches the pre-registration lock
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
from datetime import date

from . import runner, scoring
from .axes import AXES
from .client import BudgetExceeded, Client, Ledger, load_pricing

ROOT = runner.ROOT
LOCK = os.path.join(ROOT, "PREREGISTRATION.lock")
ROSTER = os.path.join(ROOT, "roster.json")
PRICING = os.path.join(ROOT, "results", "pricing.json")
LOCKED_FILES = ["DESIGN.md", "horizon/core.py", "horizon/scoring.py", "horizon/runner.py", "horizon/client.py",
                *sorted(f"horizon/axes/{f}" for f in os.listdir(os.path.join(ROOT, "horizon", "axes")) if f.endswith(".py"))]


def code_hash() -> str:
    h = hashlib.sha256()
    for rel in LOCKED_FILES:
        h.update(rel.encode())
        h.update(open(os.path.join(ROOT, rel), "rb").read().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def cmd_lock(a):
    with open(LOCK, "w") as f:
        json.dump({"sha256": code_hash(), "files": LOCKED_FILES, "locked_on": str(date.today())}, f, indent=2)
    print("locked", code_hash())


def cmd_verify(a=None) -> bool:
    if not os.path.exists(LOCK):
        print("no PREREGISTRATION.lock")
        return False
    ok = json.load(open(LOCK))["sha256"] == code_hash()
    print("pre-registration lock: " + ("MATCHES" if ok else "DOES NOT MATCH (scoring code changed after lock)"))
    return ok


def roster_ids(spec):
    if spec:
        return spec.split(",")
    return [m["id"] for m in json.load(open(ROSTER))["models"]]


def cmd_run(a):
    if not a.public and not cmd_verify() and not a.unlocked:
        sys.exit("refusing a scored run with unlocked code (use --unlocked for development)")
    pricing = load_pricing(PRICING)
    ledger = Ledger(os.path.join(ROOT, "results", "ledger.json"), a.budget)
    client = Client(a.backend, ledger, pricing)
    salt = runner.salt(a.public)
    axes = a.axes.split(",") if a.axes else list(AXES)
    models = roster_ids(a.models)
    print(f"{len(models)} models x {len(axes)} axes x {a.items} items, budget ${a.budget} (spent ${ledger.spent:.2f})")
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(runner.run_model, client, m, axes, a.items, salt): m for m in models}
        for f in cf.as_completed(futs):
            errs = f.result()
            print(f"== done {futs[f]} (total spent ${ledger.spent:.2f})" + (f" STOPPED: {errs[0]}" if errs else ""))
    print(f"total spent ${ledger.spent:.2f}")


def cmd_score(a):
    roster = {m["id"]: m for m in json.load(open(ROSTER))["models"]}
    out = {}
    for fn in sorted(os.listdir(runner.RUNS)):
        rows = [json.loads(l) for l in open(os.path.join(runner.RUNS, fn), encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r["salt_id"] != "public"]  # smoke tests never reach the leaderboard
        if not rows:
            continue
        model = rows[0]["model"]
        out[model] = scoring.summarize(rows, list(AXES)) | {"meta": roster.get(model, {})}
    with open(os.path.join(ROOT, "results", "summary.json"), "w") as f:
        json.dump({"generated": str(date.today()), "lock": json.load(open(LOCK))["sha256"] if os.path.exists(LOCK) else None,
                   "models": out}, f, indent=1)
    for m, s in sorted(out.items(), key=lambda kv: -(kv[1]["horizon_index"] or {"value": -99})["value"]):
        hi = s["horizon_index"]
        axes = " ".join(f"{k[:6]}={v['theta']}" for k, v in s["axes"].items())
        print(f"{m:42s} HI={hi['value'] if hi else '-':>6}  {axes}  cal={s['calibration'].get('skill')}")


def cmd_human(a):
    def solver(task, messages):
        print("\n" + "=" * 70 + "\n" + messages[-1]["content"] + "\n" + "=" * 70)
        print("Type your reply (include FINAL ANSWER: / CONFIDENCE: lines). End with a line containing only END.")
        lines = []
        for line in sys.stdin:
            if line.strip() == "END":
                break
            lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    runner.run_axis(None, f"human:{a.name}", a.axis, a.items, runner.salt(a.public), solver=solver)


def cmd_show(a):
    t = AXES[a.axis](a.level, runner.item_seed("public", a.axis, a.level, a.seed))
    print(t.full_prompt())
    if a.answer:
        print("\n--- reference answer ---\n" + t.oracle)


def main():
    p = argparse.ArgumentParser(prog="horizon")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--models", help="comma-separated ids (default: roster.json)")
    r.add_argument("--axes")
    r.add_argument("--items", type=int, default=8, help="scored items per model per axis")
    r.add_argument("--budget", type=float, default=50.0, help="hard USD cap across all runs")
    r.add_argument("--backend", default="openrouter")
    r.add_argument("--workers", type=int, default=6, help="models evaluated concurrently")
    r.add_argument("--public", action="store_true", help="use public seeds (smoke tests; not leaderboard)")
    r.add_argument("--unlocked", action="store_true")
    sub.add_parser("score")
    sub.add_parser("lock")
    sub.add_parser("verify")
    h = sub.add_parser("human")
    h.add_argument("--axis", required=True, choices=list(AXES))
    h.add_argument("--name", required=True)
    h.add_argument("--items", type=int, default=8)
    h.add_argument("--public", action="store_true")
    s = sub.add_parser("show")
    s.add_argument("axis", choices=list(AXES))
    s.add_argument("level", type=int)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--answer", action="store_true")
    sub.add_parser("report")
    a = p.parse_args()
    if a.cmd == "report":
        from .report import build
        build()
        return
    {"run": cmd_run, "score": cmd_score, "lock": cmd_lock, "verify": cmd_verify,
     "human": cmd_human, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    try:
        main()
    except BudgetExceeded as e:
        sys.exit(f"budget cap reached: {e}")
