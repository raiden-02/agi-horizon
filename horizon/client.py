"""Minimal OpenAI-compatible chat client (stdlib only) with cost tracking.

Identical request settings for every model; see DESIGN.md "Run protocol".
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

BACKENDS = {
    "openrouter": ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}
MAX_TOKENS = 16000          # same output cap (reasoning + answer) for every model
REASONING_EFFORT = "medium"  # sent to every model; ignored by non-reasoning models
TIMEOUT_S = 900


class BudgetExceeded(Exception):
    pass


@dataclass
class Reply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    error: Optional[str] = None      # None, "context_overflow", or "infra: ..."
    latency_s: float = 0.0


class Ledger:
    """Thread-safe running spend with a hard cap, persisted across runs."""

    def __init__(self, path: str, cap: float):
        self.path, self.cap, self.lock = path, cap, threading.Lock()
        self.spent = json.load(open(path))["spent"] if os.path.exists(path) else 0.0

    def check(self):
        if self.spent >= self.cap:
            raise BudgetExceeded(f"spent ${self.spent:.2f} of ${self.cap:.2f} cap")

    def add(self, cost: float):
        with self.lock:
            self.spent += cost
            with open(self.path, "w") as f:
                json.dump({"spent": round(self.spent, 6)}, f)


class Client:
    def __init__(self, backend: str, ledger: Ledger, pricing: dict):
        self.url, key_var = BACKENDS[backend]
        self.key = os.environ.get(key_var) or _dotenv().get(key_var)
        if not self.key or self.key.startswith("REPLACE_WITH"):
            raise SystemExit(f"{key_var} is not set (environment or .env in the repo root)")
        self.backend, self.ledger, self.pricing = backend, ledger, pricing

    def chat(self, model: str, messages: list[dict]) -> Reply:
        self.ledger.check()
        body = {"model": model, "messages": messages}
        if self.backend == "openrouter":
            body |= {"max_tokens": MAX_TOKENS, "reasoning": {"effort": REASONING_EFFORT},
                     "usage": {"include": True}}
        else:
            body |= {"max_completion_tokens": MAX_TOKENS}
        t0, last = time.time(), None
        for attempt in range(6):
            try:
                data = self._post(body)
                return self._reply(model, data, time.time() - t0)
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:500]
                last = f"HTTP {e.code}: {msg}"
                if e.code == 400 and any(s in msg.lower() for s in ("context length", "context_length", "context window", "maximum context", "too long")):
                    return Reply("", error="context_overflow", latency_s=time.time() - t0)
                if e.code == 400 and "reasoning" in body:
                    body.pop("reasoning")  # model rejects the param: retry without it
                    continue
                if e.code in (401, 402, 403):
                    raise SystemExit(f"{model}: {last}")
            except Exception as e:  # timeouts, resets, bad JSON
                last = f"{type(e).__name__}: {e}"[:300]
            time.sleep(min(60, 4 * 2 ** attempt))
        return Reply("", error=f"infra: {last}", latency_s=time.time() - t0)

    def _post(self, body):
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
            "X-Title": "agi-horizon"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = json.loads(r.read().decode("utf-8"))
        if "error" in data and not data.get("choices"):
            raise RuntimeError(str(data["error"])[:300])
        return data

    def _reply(self, model, data, latency):
        msg = data["choices"][0]["message"]
        u = data.get("usage") or {}
        pt, ct = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        cost = u.get("cost")
        if cost is None:
            p = self.pricing.get(model) or self.pricing.get(f"openai/{model}", {})
            cost = pt * float(p.get("prompt", 0)) + ct * float(p.get("completion", 0))
        self.ledger.add(cost)
        return Reply(msg.get("content") or "", pt, ct, float(cost), latency_s=latency)


def _dotenv() -> dict:
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(p):
        return {}
    pairs = (ln.split("=", 1) for ln in open(p, encoding="utf-8") if "=" in ln and not ln.startswith("#"))
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


def load_pricing(path: str) -> dict:
    """OpenRouter /models pricing (USD per token), cached to `path`."""
    if not os.path.exists(path):
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))["data"]
        slim = {m["id"]: {"prompt": m["pricing"].get("prompt", "0"), "completion": m["pricing"].get("completion", "0"),
                          "created": m.get("created"), "context_length": m.get("context_length"), "name": m.get("name")}
                for m in data}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=0)
    return json.load(open(path, encoding="utf-8"))
