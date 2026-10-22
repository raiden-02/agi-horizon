"""World modeling: infer hidden dynamics, then simulate them forward.

A hidden cellular automaton (k colors, neighborhood radius r, wrap-around)
evolves a row of cells. The model sees an observed history and must roll
the system forward h more steps. Every neighborhood the rollout needs is
guaranteed to appear in the observed history, so the dynamics are fully
identifiable. Level raises colors, radius and rollout horizon. Score =
fraction of predicted rows that are exactly right.
"""
from __future__ import annotations

from itertools import product

from ..core import Task, rng_for

AXIS = "world_model"


def _step(row, rule, r):
    n = len(row)
    return [rule[tuple(row[(i + d) % n] for d in range(-r, r + 1))] for i in range(n)]


def _hoods(row, r):
    n = len(row)
    return {tuple(row[(i + d) % n] for d in range(-r, r + 1)) for i in range(n)}


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    # Colors and radius saturate (identifiability needs the observed history
    # to cover the rule table); the rollout horizon keeps growing.
    k = 2 if level < 5 else 3
    r = 1 if level < 8 else 2
    h = 1 + level // 2
    width = 16 + 3 * level
    min_obs, max_obs = 6 + level, 40 + 2 * level
    for _ in range(500):
        rule = {nb: rng.randrange(k) for nb in product(range(k), repeat=2 * r + 1)}
        rows = [[rng.randrange(k) for _ in range(width)]]
        for _ in range(max_obs + h):
            rows.append(_step(rows[-1], rule, r))
        # Reject rules that die out or freeze: they make rollouts trivial.
        if len({tuple(x) for x in rows[-4:]}) < 4:
            continue
        # Shortest history whose observed transitions (rows[0..n-2] -> next) cover
        # every neighborhood the rollout steps from (rows[n-1..n+h-2]).
        seen, n_obs = set(), None
        for n in range(2, max_obs + 1):
            seen |= _hoods(rows[n - 2], r)
            if n >= min_obs and set().union(*(_hoods(rows[j], r) for j in range(n - 1, n - 1 + h))) <= seen:
                n_obs = n
                break
        if n_obs:
            break
    obs, fut = rows[:n_obs], rows[n_obs:n_obs + h]
    fmt = lambda row: "".join(map(str, row))
    prompt = (f"A row of {width} cells evolves in discrete time steps by a fixed hidden local rule: each cell's next "
              f"value (0-{k - 1}) depends only on its own value and the values of the {r} cell(s) on each side, "
              f"with the row wrapping around at the edges. Observed history, one time step per line:\n\n"
              + "\n".join(fmt(x) for x in obs) +
              f"\n\nPredict the next {h} row(s), one per line, digits only, in time order.")

    def grade(ans: str):
        lines = [ln.strip() for ln in (ans or "").splitlines() if ln.strip().isdigit()]
        got = lines[-h:]
        return sum(g == fmt(f) for g, f in zip(got, fut)) / h, None

    return Task(AXIS, level, seed, prompt, grade, "\n".join(fmt(x) for x in fut),
                meta={"k": k, "radius": r, "horizon": h})
