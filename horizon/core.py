"""Shared task interface, prompt protocol and answer parsing.

Every axis exposes `generate(level: int, seed: int) -> Task`. Levels are
unbounded integers >= 1; difficulty keeps rising with level so no model can
saturate an axis. A Task grades a model reply into a score in [0, 1].
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

# One neutral protocol for every model. Plain-text markers rather than any
# vendor's preferred markup (XML tags, JSON mode, tool calls) so no model
# family gets a formatting advantage.
PROTOCOL = """\
Solve the task below. Think as much as you need. When you are done, end your reply with exactly this format:

FINAL ANSWER:
<your answer>
CONFIDENCE: <integer 0-100, your probability that the answer is fully correct>"""


@dataclass
class Task:
    axis: str
    level: int
    seed: int
    prompt: str
    # grade(answer_text) -> (score in [0,1], feedback or None). Feedback is
    # only returned by multi-turn tasks and is shown to the model next turn.
    grade: Callable[[str], tuple[float, Optional[str]]]
    oracle: str  # a reference answer that must score 1.0 (self-test)
    max_turns: int = 1
    meta: dict = field(default_factory=dict)

    def full_prompt(self) -> str:
        return f"{PROTOCOL}\n\n=== TASK ===\n{self.prompt}"


def rng_for(*parts) -> random.Random:
    """Deterministic RNG from any parts (axis, level, seed, salt...)."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


_FINAL = re.compile(r"final\s+answer[\s*_]*:?[\s*_]*", re.I)
_CONF = re.compile(r"[*_]*confidence[\s*_]*:?[\s*_]*(\d{1,3}(?:\.\d+)?)\s*%?", re.I)


def parse_reply(text: str) -> tuple[Optional[str], Optional[float]]:
    """Return (answer, confidence in [0,1]) from a model reply.

    Uses the LAST 'FINAL ANSWER' marker so models may restate drafts while
    thinking. Missing marker -> answer None (scores 0; no lenient guessing,
    the same strictness for every model).
    """
    if not text:
        return None, None
    marks = list(_FINAL.finditer(text))
    if not marks:
        return None, _last_conf(text)
    body = text[marks[-1].end():]
    conf = None
    m = _CONF.search(body)
    if m:
        conf = min(max(float(m.group(1)) / 100.0, 0.0), 1.0)
        body = body[: m.start()]
    return strip_fences(body).strip(), conf


def _last_conf(text: str) -> Optional[float]:
    ms = list(_CONF.finditer(text))
    return min(max(float(ms[-1].group(1)) / 100.0, 0.0), 1.0) if ms else None


def strip_fences(s: str) -> str:
    """Drop markdown decoration around an answer: a wrapping code fence,
    horizontal rules, and stray bold/italic markers on their own lines."""
    s = s.strip()
    m = re.match(r"^```[^\n]*\n(.*?)\n?```$", s, re.S)
    if m:
        return m.group(1)
    if "```" in s and "def " not in s:  # fenced block amid decoration (code answers keep fences)
        m = re.search(r"```[^\n]*\n(.*?)\n?```", s, re.S)
        if m:
            return m.group(1)
    lines = [ln for ln in s.splitlines() if not re.fullmatch(r"\s*([*_\-=]{2,}|\*)\s*", ln)]
    return "\n".join(ln.strip().strip("*").strip() if ln.strip().startswith("*") else ln for ln in lines).strip()


def norm_ws(s: str) -> str:
    return " ".join(s.split()).lower()


def parse_int(s: Optional[str]) -> Optional[int]:
    """An integer answer; tolerant of $, \\boxed{}, commas, trailing period.
    Ambiguous replies containing several numbers are rejected."""
    if s is None:
        return None
    s = re.sub(r"\\boxed\{([^}]*)\}", r"\1", s)
    s = s.replace(",", "").replace("_", "").replace("$", "").strip().rstrip(".")
    nums = re.findall(r"-?\d+", s)
    return int(nums[0]) if len(nums) == 1 else None
