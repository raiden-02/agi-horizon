"""Abstract rule induction (fluid reasoning, ARC-style).

A hidden program composed of grid primitives maps input grids to outputs.
The model sees a few input/output pairs and must produce the output for a
new input. Level raises the number of composed primitives and grid size.
Score is exact match (1/0); ability comes from the level ladder, not from
partial credit, because near-miss cell accuracy rewards copying the input.
"""
from __future__ import annotations

from ..core import Task, rng_for

AXIS = "rule_induction"
Grid = list[list[int]]


def rot90(g): return [list(r) for r in zip(*g[::-1])]
def flip_h(g): return [r[::-1] for r in g]
def flip_v(g): return [list(r) for r in g[::-1]]
def transpose(g): return [list(r) for r in zip(*g)]


def gravity(g):
    h, w = len(g), len(g[0])
    out = [[0] * w for _ in range(h)]
    for c in range(w):
        col = [g[r][c] for r in range(h) if g[r][c]]
        for i, v in enumerate(reversed(col)):
            out[h - 1 - i][c] = v
    return out


def mirror_concat(g): return [r + r[::-1] for r in g]


def make_recolor(rng):
    a, b = rng.sample(range(1, 10), 2)
    def f(g): return [[b if v == a else a if v == b else v for v in r] for r in g]
    f.desc = f"swap {a}<->{b}"
    return f


def make_shift(rng):
    dx = rng.randint(1, 2)
    def f(g): return [r[-dx:] + r[:-dx] for r in g]
    return f


def make_row_fill(rng):
    c = rng.randint(1, 9)
    def f(g): return [[c] * len(r) if c in r else list(r) for r in g]
    return f


def make_outline(rng):
    c = rng.randint(1, 9)
    def f(g):
        h, w = len(g), len(g[0])
        out = [list(r) for r in g]
        for y in range(h):
            for x in range(w):
                if g[y][x] == 0 and any(
                    0 <= y + dy < h and 0 <= x + dx < w and g[y + dy][x + dx] not in (0, c)
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1))
                ):
                    out[y][x] = c
        return out
    return f


def make_count_bar(rng):
    # Append a row whose length-prefix encodes the count of nonzero cells mod width.
    c = rng.randint(1, 9)
    def f(g):
        w = len(g[0])
        n = sum(1 for r in g for v in r if v) % (w + 1)
        return [list(r) for r in g] + [[c] * n + [0] * (w - n)]
    return f


FIXED = [rot90, flip_h, flip_v, transpose, gravity, mirror_concat]
MAKERS = [make_recolor, make_shift, make_row_fill, make_outline, make_count_bar]


def _random_grid(rng, h, w, colors) -> Grid:
    density = rng.uniform(0.25, 0.55)
    return [[rng.choice(colors) if rng.random() < density else 0 for _ in range(w)] for _ in range(h)]


def fmt(g: Grid) -> str:
    return "\n".join("".join(map(str, r)) for r in g)


def parse_grid(s: str):
    rows = [ln.strip().replace(" ", "") for ln in (s or "").strip().splitlines() if ln.strip()]
    if not rows or not all(r.isdigit() for r in rows):
        return None
    return [[int(c) for c in r] for r in rows]


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    n_ops = 1 + (level - 1) // 2
    size_cap = 3 + level // 2
    for _ in range(200):
        ops = []
        for _ in range(n_ops):
            if rng.random() < 0.45:
                ops.append(rng.choice(FIXED))
            else:
                ops.append(rng.choice(MAKERS)(rng))
        if sum(op is mirror_concat for op in ops) > 1:
            continue
        def prog(g, ops=ops):
            for op in ops:
                g = op(g)
            return g
        colors = rng.sample(range(1, 10), min(9, 2 + level // 2))
        dims = [(rng.randint(3, size_cap + 2), rng.randint(3, size_cap + 2)) for _ in range(5)]
        ins = [_random_grid(rng, h, w, colors) for h, w in dims]
        outs = [prog(g) for g in ins]
        # Reject degenerate programs: any pair where output == input, or
        # identical outputs (rule not visible).
        if any(o == i for i, o in zip(ins, outs)):
            continue
        break
    n_train = 3 if level < 6 else 4
    train, test_in, test_out = list(zip(ins[:n_train], outs[:n_train])), ins[n_train], outs[n_train]
    parts = ["Each example maps an input grid to an output grid by the same hidden rule. "
             "Digits are colors; 0 is background. Infer the rule and give the output grid "
             "for the test input, one row per line, digits only.\n"]
    for k, (i, o) in enumerate(train, 1):
        parts.append(f"Example {k} input:\n{fmt(i)}\nExample {k} output:\n{fmt(o)}\n")
    parts.append(f"Test input:\n{fmt(test_in)}")

    def grade(ans: str):
        return (1.0 if parse_grid(ans) == test_out else 0.0), None

    return Task(AXIS, level, seed, "\n".join(parts), grade, fmt(test_out), meta={"n_ops": n_ops})
