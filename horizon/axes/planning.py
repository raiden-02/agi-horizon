"""Planning and search: keys-and-doors mazes with a verified optimum.

The model outputs a move string; a simulator checks it. Reaching the goal
scores optimal_length / submitted_length (1.0 for an optimal plan), so a
valid but wasteful plan earns partial credit and failing to arrive scores 0.
Level grows the maze and the number of key/door dependencies, which forces
longer chains of subgoals.
"""
from __future__ import annotations

import re
from collections import deque

from ..core import Task, rng_for

AXIS = "planning"
MOVES = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}


def _bfs(grid, start, goal):
    h, w = len(grid), len(grid[0])
    s0 = (start, frozenset())
    prev = {s0: None}
    q = deque([s0])
    while q:
        (pos, keys) = state = q.popleft()
        if pos == goal:
            path = []
            while prev[state]:
                state, m = prev[state]
                path.append(m)
            return "".join(reversed(path))
        for m, (dy, dx) in MOVES.items():
            y, x = pos[0] + dy, pos[1] + dx
            if not (0 <= y < h and 0 <= x < w):
                continue
            c = grid[y][x]
            if c == "#" or (c.isupper() and c not in "SG" and c.lower() not in keys):
                continue
            nk = keys | {c} if c.islower() else keys
            ns = ((y, x), nk)
            if ns not in prev:
                prev[ns] = (state, m)
                q.append(ns)
    return None


def simulate(grid, start, goal, plan):
    """Number of legal moves taken to first reach the goal, or None.
    Moves after arriving are ignored (the episode is over)."""
    h, w = len(grid), len(grid[0])
    pos, keys = start, set()
    for i, m in enumerate(plan, 1):
        if m not in MOVES:
            return None
        y, x = pos[0] + MOVES[m][0], pos[1] + MOVES[m][1]
        if not (0 <= y < h and 0 <= x < w):
            return None
        c = grid[y][x]
        if c == "#" or (c.isupper() and c not in "SG" and c.lower() not in keys):
            return None
        if c.islower():
            keys.add(c)
        pos = (y, x)
        if pos == goal:
            return i
    return None


def _extract_plan(ans):
    s = (ans or "").upper()
    if re.fullmatch(r"[UDLR\s,]+", s):
        return re.sub(r"[\s,]", "", s)
    runs = re.findall(r"\b[UDLR]+\b", s)
    return max(runs, key=len) if runs else ""


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    size = 5 + level
    n_doors = min(level // 2, 8)
    for attempt in range(2000):
        g = [["#" if rng.random() < 0.28 else "." for _ in range(size)] for _ in range(size)]
        cells = [(y, x) for y in range(size) for x in range(size)]
        rng.shuffle(cells)
        start, goal, *rest = cells
        for (y, x), c in ((start, "S"), (goal, "G")):
            g[y][x] = c
        letters = "abcdefhj"[:n_doors]  # skip g/s: G and S are taken
        for i, ch in enumerate(letters):
            ky, kx = rest[2 * i]
            dy, dx = rest[2 * i + 1]
            g[ky][kx], g[dy][dx] = ch, ch.upper()
        grid = ["".join(r) for r in g]
        opt = _bfs(grid, start, goal)
        if not opt or len(opt) < size:
            continue
        if n_doors:
            # Every door must matter: the goal is unreachable if all doors were walls.
            walled = ["".join("#" if c.isupper() and c not in "SG" else c for c in r) for r in grid]
            if _bfs(walled, start, goal):
                continue
        break
    prompt = (f"Grid maze ({size}x{size}). Legend: S start, G goal, # wall, . floor, lowercase letter = key, "
              f"uppercase letter = door that can only be entered after picking up the matching key (keys are "
              f"picked up by stepping on them and are never consumed). Moves: U D L R (one cell each; you "
              f"cannot leave the grid or enter walls).\n\n" + "\n".join(grid) +
              "\n\nGive a shortest move sequence from S to G as one string of letters, e.g. RRDDL. "
              "Shorter valid plans score higher; invalid plans score 0.")

    def grade(ans: str):
        plan = _extract_plan(ans)
        used = simulate(grid, start, goal, plan) if plan else None
        return (min(1.0, len(opt) / used) if used else 0.0), None

    return Task(AXIS, level, seed, prompt, grade, opt, meta={"opt_len": len(opt), "doors": n_doors})
