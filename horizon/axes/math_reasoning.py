"""Quantitative reasoning with machine-checked integer answers.

Four generated families, each with a level-driven size knob that grows
without bound. Answers are computed by exact algorithms (inclusion-exclusion
over a period, DP, fast matrix power), so huge levels stay cheap to grade
while being infeasible to brute-force by hand.
"""
from __future__ import annotations

from math import gcd

from ..core import Task, parse_int, rng_for

AXIS = "math"


def _lcm(a, b): return a * b // gcd(a, b)


def counting(rng, level):
    # Count n in [1, N] meeting k modular conditions.
    k = 1 + (level + 1) // 2
    N = rng.randint(10 ** (1 + level // 2), 10 ** (2 + level // 2))
    while True:  # keep the period small enough to grade exactly; avoid empty answers
        conds, period = [], 1
        for _ in range(k):
            m = rng.randint(2, 6 + level)
            kind = rng.choice(["div", "notdiv", "res"])
            r = rng.randint(1, m - 1) if kind == "res" else 0
            conds.append((kind, m, r))
            period = _lcm(period, m)
        if period <= 200_000 and _count(conds, period):
            break
    return _counting_question(conds, period, N)


def _count(conds, upto):
    return sum(_ok(conds, n) for n in range(1, upto + 1))


def _ok(conds, n):
    for kind, m, r in conds:
        if kind == "div" and n % m: return False
        if kind == "notdiv" and n % m == 0: return False
        if kind == "res" and n % m != r: return False
    return True


def _counting_question(conds, period, N):
    full, rem = divmod(N, period)
    ans = full * _count(conds, period) + _count(conds, rem)
    words = {"div": "is divisible by {m}", "notdiv": "is not divisible by {m}",
             "res": "leaves remainder {r} when divided by {m}"}
    desc = "; ".join(words[c].format(m=m, r=r) for c, m, r in conds)
    return f"How many integers n with 1 <= n <= {N} satisfy all of the following: n {desc}?", ans


def lattice(rng, level):
    # Monotone lattice paths (right/down) on an H x W grid avoiding blocked cells.
    h = w = 3 + level
    blocked = set()
    for _ in range(level * 2):
        y, x = rng.randrange(h), rng.randrange(w)
        if (y, x) not in ((0, 0), (h - 1, w - 1)):
            blocked.add((y, x))
    dp = [[0] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if (y, x) in blocked: continue
            dp[y][x] = 1 if (y, x) == (0, 0) else (dp[y - 1][x] if y else 0) + (dp[y][x - 1] if x else 0)
    cells = ", ".join(f"({y},{x})" for y, x in sorted(blocked))
    q = (f"A grid has {h} rows and {w} columns, cells indexed (row, col) from (0,0) top-left. "
         f"Count the paths from (0,0) to ({h - 1},{w - 1}) moving only right or down that never "
         f"enter these blocked cells: {cells}.")
    return q, dp[h - 1][w - 1]


def recurrence(rng, level):
    # a_n = c1*a_{n-1} + ... + ck*a_{n-k} mod M; ask a_N for large N.
    k = 2 + level // 3
    coefs = [rng.randint(0, 5) for _ in range(k)]
    coefs[-1] = coefs[-1] or 1
    init = [rng.randint(0, 9) for _ in range(k)]
    M = rng.choice([97, 101, 997, 1009, 10007])
    N = rng.randint(10 ** (level // 2 + 1), 10 ** (level // 2 + 2))

    def mat_mul(A, B):
        return [[sum(A[i][t] * B[t][j] for t in range(k)) % M for j in range(k)] for i in range(k)]

    T = [coefs] + [[1 if j == i else 0 for j in range(k)] for i in range(k - 1)]
    R = [[1 if i == j else 0 for j in range(k)] for i in range(k)]
    e = N - (k - 1)
    while e > 0:
        if e & 1: R = mat_mul(R, T)
        T = mat_mul(T, T)
        e >>= 1
    state = init[::-1]  # [a_{k-1}, ..., a_0]
    ans = sum(R[0][j] * state[j] for j in range(k)) % M
    terms = " + ".join(f"{c}*a(n-{i + 1})" for i, c in enumerate(coefs))
    inits = ", ".join(f"a({i})={v}" for i, v in enumerate(init))
    return (f"Define a(n) = ({terms}) mod {M} for n >= {k}, with {inits}. "
            f"What is a({N})?"), ans


def linear_system(rng, level):
    n = 2 + level // 2
    span = 3 + level
    for _ in range(100):
        x = [rng.randint(-span, span) for _ in range(n)]
        A = [[rng.randint(-span, span) for _ in range(n)] for _ in range(n)]
        if _det(A) != 0:
            break
    b = [sum(A[i][j] * x[j] for j in range(n)) for i in range(n)]
    names = [f"x{j + 1}" for j in range(n)]
    eqs = "\n".join(" + ".join(f"({A[i][j]})*{names[j]}" for j in range(n)) + f" = {b[i]}" for i in range(n))
    return (f"This linear system has a unique integer solution:\n{eqs}\n"
            "Report " + " + ".join(f"{v}^2" for v in names) + "."), sum(v * v for v in x)


def _det(A):
    from fractions import Fraction
    M = [[Fraction(v) for v in r] for r in A]
    n, d = len(M), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if M[r][c] != 0), None)
        if p is None: return 0
        if p != c: M[c], M[p] = M[p], M[c]; d = -d
        d *= M[c][c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for j in range(c, n): M[r][j] -= f * M[c][j]
    return d


FAMILIES = [counting, lattice, recurrence, linear_system]


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    fam = FAMILIES[seed % len(FAMILIES)]
    q, ans = fam(rng, level)
    q += "\nGive the final answer as a single integer."

    def grade(a: str):
        return (1.0 if parse_int(a) == ans else 0.0), None

    return Task(AXIS, level, seed, q, grade, str(ans), meta={"family": fam.__name__})
