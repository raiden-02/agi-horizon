"""Agentic coding: implement an interpreter for a freshly invented language.

Each task invents a stack language (random op names, randomized semantics,
level-gated control flow and edge-case rules). The model writes Python,
we run it against hidden tests, and it gets up to three turns with
feedback from the public tests only. Score = fraction of hidden tests
passed by the final submission, so partial implementations earn partial
credit. The language is new every time, so it cannot be memorized.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile

from ..core import Task, rng_for

AXIS = "coding"
STEP_LIMIT = 5000
N_HIDDEN, N_PUBLIC = 25, 5


def _pop(st, rules):
    if st:
        return st.pop()
    if rules.get("underflow_zero"):
        return 0
    raise IndexError


# Each op: (description template, implementation(st, rules, out)).
def _binary(fn, desc):
    def impl(st, rules, out):
        a = _pop(st, rules)
        b = _pop(st, rules)
        st.append(_wrap(fn(a, b), rules))
    return desc, impl


def _wrap(v, rules):
    m = rules.get("modulus")
    return v % m if m else v


def _mod_impl(st, rules, out):
    a = _pop(st, rules)
    b = _pop(st, rules)
    st.append(_wrap(b % a if a else 0, rules))


def _dup(st, rules, out):
    a = _pop(st, rules); st += [a, a]


def _swap(st, rules, out):
    a = _pop(st, rules); b = _pop(st, rules); st += [a, b]


def _drop(st, rules, out):
    _pop(st, rules)


def _over(st, rules, out):
    a = _pop(st, rules); b = _pop(st, rules); st += [b, a, b]


def _rot(st, rules, out):
    a = _pop(st, rules); b = _pop(st, rules); c = _pop(st, rules); st += [b, a, c]


def _print(st, rules, out):
    out.append(_pop(st, rules))


def _neg(st, rules, out):
    st.append(_wrap(-_pop(st, rules), rules))


OPS = {
    "add": _binary(lambda a, b: a + b, "pops a (top) then b, pushes b + a"),
    "sub_ba": _binary(lambda a, b: b - a, "pops a (top) then b, pushes b - a"),
    "sub_ab": _binary(lambda a, b: a - b, "pops a (top) then b, pushes a - b"),
    "mul": _binary(lambda a, b: a * b, "pops a (top) then b, pushes b * a"),
    "max": _binary(max, "pops two values, pushes the larger"),
    "lt": _binary(lambda a, b: int(b < a), "pops a (top) then b, pushes 1 if b < a else 0"),
    "mod": ("pops a (top) then b, pushes b mod a (Python semantics; pushes 0 if a is 0)", _mod_impl),
    "dup": ("duplicates the top value", _dup),
    "swap": ("swaps the top two values", _swap),
    "drop": ("discards the top value", _drop),
    "over": ("copies the second value to the top (a b -> a b a, where b is top)", _over),
    "rot": ("moves the third value to the top (x y z -> y z x, where z is top)", _rot),
    "neg": ("negates the top value", _neg),
    "print": ("pops the top value and appends it to the output list", _print),
}
CORE = ["add", "print", "dup"]


def _syllable_name(rng, used):
    while True:
        n = "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(rng.randint(2, 3)))
        if n not in used:
            used.add(n)
            return n


class Lang:
    def __init__(self, rng, level):
        # sub has two variants; offer only one of them
        skip = rng.choice(["sub_ab", "sub_ba"])
        pool = [k for k in OPS if k not in CORE and k != skip]
        rng.shuffle(pool)
        chosen = (CORE + pool)[: 3 + level]
        used: set[str] = set()
        self.names = {_syllable_name(rng, used): op for op in chosen}
        self.rules = {}
        if level >= 3:
            self.rules["modulus"] = rng.choice([97, 256, 1000])
        if level >= 4:
            self.loop = (_syllable_name(rng, used), _syllable_name(rng, used))
        else:
            self.loop = None
        if level >= 6:
            self.rules["underflow_zero"] = True
        if level >= 8:
            self.cond = (_syllable_name(rng, used), _syllable_name(rng, used))
        else:
            self.cond = None
        self.prog_len = 6 + 2 * level

    def spec(self) -> str:
        lines = ["A program is a string of whitespace-separated tokens, executed left to right on an "
                 "initially empty stack of integers. An integer literal token (e.g. 7 or -3) pushes itself."]
        for name, op in self.names.items():
            lines.append(f"- `{name}`: {OPS[op][0]}")
        if self.loop:
            o, c = self.loop
            lines.append(f"- `{o}` ... `{c}` form a loop (they nest). At `{o}`: pop the top; if it is 0, jump "
                         f"past the matching `{c}`. At `{c}`: pop the top; if it is nonzero, jump back to just after "
                         f"the matching `{o}`.")
        if self.cond:
            o, c = self.cond
            lines.append(f"- `{o}` ... `{c}` form a conditional (they nest). At `{o}`: pop the top; if it is 0, "
                         f"skip to just past the matching `{c}`. `{c}` itself does nothing.")
        if "modulus" in self.rules:
            lines.append(f"- Every value pushed by an arithmetic op (any op that computes a new number, including "
                         f"negation) is reduced modulo {self.rules['modulus']} (Python % semantics, result in "
                         f"[0, {self.rules['modulus'] - 1}]). Literals are not reduced.")
        if self.rules.get("underflow_zero"):
            lines.append("- Popping an empty stack yields 0 instead of an error (including in loops/conditionals).")
        else:
            lines.append("- Test programs never pop an empty stack.")
        lines.append("Output: the list of integers produced by the print op, in order. "
                     "Test programs always terminate.")
        return "\n".join(lines)

    def run(self, program: str):
        toks = program.split()
        jump = _match(toks, [p for p in (self.loop, self.cond) if p])
        st, out, pc, steps = [], [], 0, 0
        while pc < len(toks):
            steps += 1
            if steps > STEP_LIMIT:
                raise TimeoutError
            t = toks[pc]
            if self.loop and t == self.loop[0]:
                if _pop(st, self.rules) == 0:
                    pc = jump[pc] + 1; continue
            elif self.loop and t == self.loop[1]:
                if _pop(st, self.rules) != 0:
                    pc = jump[pc] + 1; continue
            elif self.cond and t == self.cond[0]:
                if _pop(st, self.rules) == 0:
                    pc = jump[pc] + 1; continue
            elif self.cond and t == self.cond[1]:
                pass
            elif t in self.names:
                OPS[self.names[t]][1](st, self.rules, out)
            else:
                st.append(int(t))
            pc += 1
        return out

    def random_program(self, rng) -> str:
        names = list(self.names)
        toks: list[str] = []
        opens = []
        for _ in range(self.prog_len):
            r = rng.random()
            if r < 0.3:
                toks.append(str(rng.randint(-5, 12)))
            elif self.loop and r < 0.38:
                toks.append(self.loop[0]); opens.append(self.loop[1])
            elif self.cond and r < 0.44:
                toks.append(self.cond[0]); opens.append(self.cond[1])
            elif opens and r < 0.55:
                toks.append(opens.pop())
            else:
                toks.append(rng.choice(names))
        toks += opens[::-1]
        pname = next(n for n, op in self.names.items() if op == "print")
        toks.append(pname)
        return " ".join(toks)


def _match(toks, pairs):
    jump, stacks = {}, {}
    closers = {c: o for o, c in pairs}
    for i, t in enumerate(toks):
        if any(t == o for o, _ in pairs):
            stacks.setdefault(t, []).append(i)
        elif t in closers:
            j = stacks[closers[t]].pop()
            jump[i], jump[j] = j, i
    return jump


HARNESS = r"""
import json, sys
sys.setrecursionlimit(10000)
ns = {}
res = []
try:
    exec(open("solution.py", encoding="utf-8").read(), ns)
    run = ns["run"]
except Exception as e:
    print(json.dumps({"load_error": repr(e)[:300]})); sys.exit()
for prog in json.load(open("progs.json")):
    try:
        res.append(list(run(prog)))
    except Exception as e:
        res.append("ERR " + repr(e)[:120])
print(json.dumps({"results": res}))
"""


def execute(code: str, progs: list[str], timeout: float = 20.0):
    """Run model code in an isolated interpreter process; returns list or error string."""
    with tempfile.TemporaryDirectory() as d:
        for fn, body in (("solution.py", code), ("harness.py", HARNESS), ("progs.json", json.dumps(progs))):
            with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                f.write(body)
        try:
            p = subprocess.run([sys.executable, "-I", "harness.py"], cwd=d, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return "timeout (your code ran longer than 20s on the test batch; check loop handling)"
        try:
            data = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return "crashed: " + (p.stderr or p.stdout)[-400:]
        return data.get("results") or ("load error: " + data.get("load_error", "?"))


def extract_code(ans: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", ans or "", re.S)
    return max(blocks, key=len) if blocks else (ans or "")


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    lang = Lang(rng, level)
    tests = []
    while len(tests) < N_HIDDEN + N_PUBLIC:
        p = lang.random_program(rng)
        try:
            o = lang.run(p)
        except (TimeoutError, IndexError):
            continue
        tests.append((p, o))
    public, hidden = tests[:N_PUBLIC], tests[N_PUBLIC:]
    ex = "\n".join(f"run({p!r}) == {o!r}" for p, o in public)
    prompt = (f"Implement an interpreter for this invented language.\n\n{lang.spec()}\n\n"
              f"Write a Python 3 function `run(program: str) -> list[int]` (standard library only) "
              f"that returns the output list. Your code is tested on hidden programs. Public examples:\n{ex}\n\n"
              f"Put the complete code in a ```python block inside the FINAL ANSWER. After each submission "
              f"you will see which public examples fail, and may resubmit (up to 3 submissions total).")

    def grade(ans: str):
        code = extract_code(ans)
        res = execute(code, [p for p, _ in tests])
        if isinstance(res, str):
            return 0.0, f"Your submission failed: {res}"
        hid = res[N_PUBLIC:]
        score = sum(r == o for r, (_, o) in zip(hid, hidden)) / len(hidden)
        fails = [f"run({p!r}) expected {o!r}, got {r!r}" for r, (p, o) in zip(res[:N_PUBLIC], public) if r != o]
        fb = None
        if score < 1.0:
            fb = ("Public examples failing:\n" + "\n".join(fails)) if fails else \
                 "All public examples pass, but some hidden tests fail. Re-read the spec for edge cases."
        return score, fb

    oracle = "```python\n" + _oracle_source(lang) + "\n```"
    return Task(AXIS, level, seed, prompt, grade, oracle, max_turns=3, meta={"n_ops": len(lang.names)})


def _oracle_source(lang: Lang) -> str:
    """Self-contained reference solution (used only by the self-test)."""
    import inspect
    from . import coding as mod
    state = {"names": lang.names, "rules": lang.rules, "loop": lang.loop, "cond": lang.cond}
    src = inspect.getsource(mod)
    return (src.split("HARNESS = ")[0].replace("from ..core import Task, rng_for", "")
            + f"\n_L = Lang.__new__(Lang)\n_L.__dict__.update({state!r})\n"
            + "def run(program):\n    return _L.run(program)\n")
