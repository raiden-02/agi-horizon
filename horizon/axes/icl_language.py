"""Learning a new skill from examples: translate into an invented language.

Each task invents a vocabulary and grammar (word order, morphology,
agreement, negation, subordinate clauses). Levels switch on more grammar
features, grow the vocabulary and chain more clauses. Every word and
feature needed for the test sentences appears in the examples, so the task
is fully learnable from context alone. Score = fraction of test sentences
translated exactly.
"""
from __future__ import annotations

import itertools
import re

from ..core import Task, rng_for

AXIS = "icl_language"
NOUNS = ["dog", "cat", "bird", "farmer", "wolf", "king", "child", "horse", "fish", "sailor",
         "witch", "baker", "goat", "queen", "monk", "fox", "teacher", "robot", "owl", "giant"]
VERBS = [("see", "sees", "saw"), ("chase", "chases", "chased"), ("help", "helps", "helped"),
         ("love", "loves", "loved"), ("find", "finds", "found"), ("call", "calls", "called"),
         ("follow", "follows", "followed"), ("feed", "feeds", "fed"), ("teach", "teaches", "taught"),
         ("fear", "fears", "feared")]
ADJS = ["big", "small", "red", "old", "happy", "tall", "quiet", "brave", "green", "angry"]
N_TEST = 3


def _word(rng, used, syl=(2, 3)):
    while True:
        w = "".join(rng.choice("bdfgklmnprstvz") + rng.choice("aeiou") for _ in range(rng.randint(*syl)))
        if w not in used:
            used.add(w)
            return w


class Grammar:
    def __init__(self, rng, level):
        used: set[str] = set()
        self.level = level
        self.nouns = {n: _word(rng, used) for n in rng.sample(NOUNS, min(len(NOUNS), 3 + level))}
        self.verbs = {v: _word(rng, used) for v in rng.sample(VERBS, min(len(VERBS), 2 + level // 2))}
        self.adjs = {a: _word(rng, used) for a in rng.sample(ADJS, min(len(ADJS), 1 + level // 2))} if level >= 2 else {}
        self.order = rng.choice(list(itertools.permutations("SVO")))
        self.adj_after = rng.random() < 0.5
        aff = lambda: _word(rng, used, (1, 1))
        self.plural = aff() if level >= 3 else None
        self.past = (aff(), rng.random() < 0.5) if level >= 4 else None  # (affix, is_prefix)
        self.obj_case = aff() if level >= 5 else None
        self.neg = (_word(rng, used, (1, 2)), rng.choice(["start", "before_verb", "end"])) if level >= 6 else None
        self.agree = aff() if level >= 7 else None
        self.adj_agree = aff() if level >= 8 else None
        self.conj = _word(rng, used) if level >= 9 else None
        self.max_sub = 0 if level < 9 else 1 + (level - 9) // 3
        self.sub_order = rng.choice([o for o in itertools.permutations("SVO") if o != self.order]) if level >= 9 else None

    # -- random sentence: nested dict --
    def sample(self, rng, depth=0):
        def np():
            return {"n": rng.choice(list(self.nouns)), "pl": bool(self.plural) and rng.random() < 0.4,
                    "adj": rng.choice(list(self.adjs)) if self.adjs and rng.random() < 0.5 else None}
        s = {"S": np(), "O": np(), "v": rng.choice(list(self.verbs)),
             "past": bool(self.past) and rng.random() < 0.5, "neg": bool(self.neg) and rng.random() < 0.35}
        s["sub"] = self.sample(rng, depth + 1) if depth < self.max_sub and rng.random() < 0.7 else None
        return s

    def english(self, s) -> str:
        def np(p):
            n = p["n"] + ("s" if p["pl"] and not p["n"].endswith("s") else "es" if p["pl"] else "")
            n = {"childs": "children", "wolfs": "wolves", "foxs": "foxes", "fishs": "fish", "witchs": "witches"}.get(n, n)
            return "the " + (p["adj"] + " " if p["adj"] else "") + n
        base, third, past = s["v"]
        if s["neg"]:
            aux = "did not" if s["past"] else ("do not" if s["S"]["pl"] else "does not")
            vp = f"{aux} {base}"
        else:
            vp = past if s["past"] else (base if s["S"]["pl"] else third)
        out = f"{np(s['S'])} {vp} {np(s['O'])}"
        if s["sub"]:
            out += " because " + self.english(s["sub"])
        return out

    def target(self, s, sub=False) -> str:
        def np(p, obj):
            w = self.nouns[p["n"]] + (self.plural if p["pl"] else "") + (self.obj_case if obj and self.obj_case else "")
            if not p["adj"]:
                return w
            a = self.adjs[p["adj"]] + (self.adj_agree if self.adj_agree and p["pl"] else "")
            return f"{w} {a}" if self.adj_after else f"{a} {w}"
        v = self.verbs[s["v"]]
        if s["past"]:
            aff, pre = self.past
            v = aff + v if pre else v + aff
        if self.agree and s["S"]["pl"]:
            v += self.agree
        parts = {"S": np(s["S"], False), "O": np(s["O"], True), "V": v}
        order = self.sub_order if sub else self.order
        words = [parts[k] for k in order]
        if s["neg"]:
            neg, pos = self.neg
            if pos == "start":
                words.insert(0, neg)
            elif pos == "end":
                words.append(neg)
            else:
                words.insert(order.index("V"), neg)
        out = " ".join(words)
        if s["sub"]:
            out += f" {self.conj} " + self.target(s["sub"], sub=True)
        return out


def _features(s, acc=None):
    acc = set() if acc is None else acc
    acc |= {("n", s["S"]["n"]), ("n", s["O"]["n"]), ("v", s["v"][0])}
    for p, role in ((s["S"], "S"), (s["O"], "O")):
        if p["adj"]: acc |= {("a", p["adj"]), ("adj_pl", p["pl"])}
        if p["pl"]: acc.add(("pl", role))
    acc |= {("past", s["past"]), ("neg", s["neg"])}
    if s["neg"]: acc.add(("negpast", s["past"]))
    if s["sub"]:
        acc.add(("sub",))
        _features(s["sub"], acc)
    return acc


def _norm(line: str) -> str:
    line = re.sub(r"^\s*(\d+[.):]|-|\*)\s*", "", line)
    return " ".join(re.sub(r"[^\w\s]", " ", line.lower()).split())


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    g = Grammar(rng, level)
    tests = [g.sample(rng) for _ in range(N_TEST)]
    need = set().union(*(_features(t) for t in tests))
    seen_en = {g.english(t) for t in tests}
    examples, have = [], {}
    for _ in range(4000):
        s = g.sample(rng)
        e = g.english(s)
        if e in seen_en:
            continue
        f = _features(s)
        # keep sentences that add evidence for a needed feature still seen < 2 times
        if any(have.get(x, 0) < 2 for x in f & need) or len(examples) < 4:
            examples.append(s); seen_en.add(e)
            for x in f: have[x] = have.get(x, 0) + 1
        if all(have.get(x, 0) >= 2 for x in need) and len(examples) >= 4 + level:
            break
    rng.shuffle(examples)
    ex = "\n".join(f"{g.english(s)} => {g.target(s)}" for s in examples)
    qs = "\n".join(f"{i + 1}. {g.english(t)}" for i, t in enumerate(tests))
    answers = [g.target(t) for t in tests]
    prompt = (f"Below are English sentences with their translations into an invented language. Learn its "
              f"vocabulary and grammar from the examples, then translate the test sentences.\n\n{ex}\n\n"
              f"Test sentences:\n{qs}\n\nGive the {N_TEST} translations in order, one per line, nothing else on those lines.")

    def grade(ans: str):
        lines = [_norm(ln) for ln in (ans or "").splitlines() if _norm(ln)]
        got = lines[-N_TEST:] if len(lines) >= N_TEST else lines + [""] * (N_TEST - len(lines))
        return sum(a == _norm(b) for a, b in zip(got, answers)) / N_TEST, None

    return Task(AXIS, level, seed, prompt, grade, "\n".join(answers), meta={"n_examples": len(examples)})
