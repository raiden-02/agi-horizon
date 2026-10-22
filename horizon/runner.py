"""Adaptive (staircase) evaluation of one model on one axis.

Climb fast (+2 levels) until the first failure, then step +1 on success and
-1 on failure. This spends items near each model's frontier, where they are
most informative, and has no ceiling: a future model just keeps climbing.
Results append to results/runs/<model>.jsonl; reruns resume where they left off.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from typing import Optional

from .axes import AXES
from .client import BudgetExceeded, Client
from .core import parse_reply

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "runs")
RAW = os.path.join(ROOT, "results", "raw")      # full transcripts: gitignored (they reveal items)
SALT_FILE = os.path.join(ROOT, ".secret_salt")  # gitignored; makes scored items unpredictable
START_LEVEL = 2
SUCCESS = 0.5
_io_lock = threading.Lock()


def salt(public: bool) -> str:
    if public:
        return "public"
    if not os.path.exists(SALT_FILE):
        with open(SALT_FILE, "w") as f:
            f.write(secrets.token_hex(32))
    return open(SALT_FILE).read().strip()


def item_seed(s: str, axis: str, level: int, visit: int) -> int:
    h = hashlib.sha256(f"{s}|{axis}|{level}|{visit}".encode()).digest()
    return int.from_bytes(h[:6], "big")


def _safe(model: str) -> str:
    return model.replace("/", "__").replace(":", "_").replace("~", "")


def load_rows(model: str) -> list[dict]:
    p = os.path.join(RUNS, _safe(model) + ".jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _append(model: str, row: dict, transcript: list[dict]):
    with _io_lock:
        os.makedirs(RUNS, exist_ok=True)
        os.makedirs(RAW, exist_ok=True)
        with open(os.path.join(RUNS, _safe(model) + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        with open(os.path.join(RAW, _safe(model) + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": row["item"], "transcript": transcript}) + "\n")


def next_level(history: list[tuple[int, float]]) -> int:
    level, failed = START_LEVEL, False
    for lv, score in history:
        if score >= SUCCESS:
            level = lv + (1 if failed else 2)
        else:
            failed, level = True, max(1, lv - 1)
    return level


def run_axis(client: Optional[Client], model: str, axis: str, n_items: int, salt_value: str,
             log=print, solver=None) -> list[dict]:
    """Run until the model has n_items scored items on this axis.

    `solver(task, messages) -> str` replaces the API (human mode, baselines)."""
    rows = [r for r in load_rows(model) if r["axis"] == axis and r["salt_id"] == salt_value[:8]]
    scored = [r for r in rows if r["status"] == "ok"]
    infra_streak = 0
    while len(scored) < n_items:
        history = [(r["level"], r["score"]) for r in scored]
        level = next_level(history)
        visit = sum(1 for r in rows if r["level"] == level)
        seed = item_seed(salt_value, axis, level, visit)
        task = AXES[axis](level, seed)
        messages = [{"role": "user", "content": task.full_prompt()}]
        score, conf, toks, cost, status, turns, err = 0.0, None, 0, 0.0, "ok", 0, None
        for turns in range(1, task.max_turns + 1):
            if solver:
                text, err = solver(task, messages), None
            else:
                rep = client.chat(model, messages)
                toks += rep.completion_tokens
                cost += rep.cost
                text, err = rep.text, rep.error
            if err and err.startswith("infra"):
                status = "infra_error"
                break
            if err == "context_overflow":
                score, status = 0.0, "ok"  # a context limit is a real capability limit
                break
            ans, conf = parse_reply(text)
            score, feedback = task.grade(ans) if ans is not None else (0.0, "No FINAL ANSWER block found.")
            messages.append({"role": "assistant", "content": text})
            if score >= 1.0 or feedback is None or turns == task.max_turns:
                break
            messages.append({"role": "user", "content": feedback + "\n\nSubmit again in the same format."})
        row = {"model": model, "axis": axis, "level": level, "item": f"{axis}:{level}:{visit}",
               "salt_id": salt_value[:8], "score": round(score, 4), "confidence": conf, "turns": turns,
               "completion_tokens": toks, "cost": round(cost, 6), "status": status,
               "error": err if status != "ok" else None}
        _append(model, row, messages)
        rows.append(row)
        if status == "ok":
            scored.append(row)
            infra_streak = 0
            log(f"{model:40s} {axis:15s} L{level:<3d} score={score:.2f} conf={conf} ${cost:.4f}")
        else:
            infra_streak += 1
            log(f"{model:40s} {axis:15s} L{level:<3d} INFRA ERROR ({infra_streak}) {err[:120]}")
            if infra_streak >= 3:
                log(f"{model} {axis}: giving up after 3 consecutive infra errors")
                break
    return scored


def run_model(client, model, axes, n_items, salt_value, log=print):
    threads, errors = [], []

    def work(axis):
        try:
            run_axis(client, model, axis, n_items, salt_value, log)
        except BudgetExceeded as e:
            errors.append(str(e))

    for axis in axes:
        t = threading.Thread(target=work, args=(axis,), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    return errors
