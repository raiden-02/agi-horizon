"""Turn item results into ability estimates.

Per axis we fit a one-parameter logistic (Rasch-style) model with a fixed,
pre-registered slope: P(success at level L) = sigmoid(SLOPE * (theta - L)).
theta is the level at which the model succeeds half the time - its
"frontier level" on that axis. Partial credit enters as a fractional
outcome. A weak prior keeps theta finite when a model never fails (or
never succeeds); such estimates are flagged as censored.
"""
from __future__ import annotations

import math
import random
import zlib

SLOPE = 1.0
PRIOR_MU, PRIOR_SD = 3.0, 8.0
GRID = [x / 50 for x in range(-5 * 50, 80 * 50 + 1)]  # theta in [-5, 80]
N_BOOT = 300


def _loglik(theta, obs):
    ll = -((theta - PRIOR_MU) ** 2) / (2 * PRIOR_SD ** 2)
    for level, s in obs:
        z = SLOPE * (theta - level)
        # log sigmoid(z) and log(1 - sigmoid(z)), numerically stable
        lp = -math.log1p(math.exp(-z)) if z > -30 else z
        lq = -math.log1p(math.exp(z)) if z < 30 else -z
        ll += s * lp + (1 - s) * lq
    return ll


def fit_theta(obs: list[tuple[int, float]]) -> float:
    # Coarse-to-fine search; the posterior is log-concave so this is exact enough.
    best = max(GRID[::25], key=lambda t: _loglik(t, obs))
    fine = [best + d / 50 for d in range(-25, 26)]
    return max(fine, key=lambda t: _loglik(t, obs))


def axis_estimate(obs: list[tuple[int, float]], seed: int = 0) -> dict:
    if not obs:
        return {"theta": None}
    theta = fit_theta(obs)
    rng = random.Random(seed)
    raw = [fit_theta([rng.choice(obs) for _ in obs]) for _ in range(N_BOOT)]
    boots = sorted(raw)
    return {
        "theta": round(theta, 2),
        "ci90": [round(boots[int(0.05 * N_BOOT)], 2), round(boots[int(0.95 * N_BOOT) - 1], 2)],
        "n": len(obs),
        "max_level_passed": max((l for l, s in obs if s >= 0.5), default=0),
        "censored": "low" if all(s < 0.5 for _, s in obs) else "high" if all(s >= 0.5 for _, s in obs) else None,
        "boot": raw,  # unsorted, so axes pair as independent draws for the index CI
    }


def calibration(rows: list[dict]) -> dict:
    """Brier score of stated confidence vs actual score over all items.
    Missing confidence counts as 0.5 (uninformative)."""
    pairs = [((r["confidence"] if r["confidence"] is not None else 0.5), r["score"]) for r in rows]
    if not pairs:
        return {}
    brier = sum((c - s) ** 2 for c, s in pairs) / len(pairs)
    return {"brier": round(brier, 4), "skill": round(1 - brier, 4),
            "overconfidence": round(sum(c for c, _ in pairs) / len(pairs) - sum(s for _, s in pairs) / len(pairs), 4),
            "missing_conf": sum(r["confidence"] is None for r in rows), "n": len(pairs)}


def summarize(rows: list[dict], axes: list[str]) -> dict:
    ok = [r for r in rows if r["status"] == "ok"]
    per_axis = {}
    for a in axes:
        obs = [(r["level"], r["score"]) for r in ok if r["axis"] == a]
        per_axis[a] = axis_estimate(obs, seed=zlib.crc32(a.encode()))
    complete = [a for a in axes if per_axis[a].get("theta") is not None]
    index = None
    if len(complete) == len(axes):
        # Horizon Index: unweighted mean of axis thetas (pre-registered equal weights);
        # CI from pairing bootstrap draws across axes.
        index = {"value": round(sum(per_axis[a]["theta"] for a in axes) / len(axes), 2)}
        draws = sorted(sum(per_axis[a]["boot"][i] for a in axes) / len(axes) for i in range(N_BOOT))
        index["ci90"] = [round(draws[int(0.05 * N_BOOT)], 2), round(draws[int(0.95 * N_BOOT) - 1], 2)]
    for a in axes:
        per_axis[a].pop("boot", None)
    return {"axes": per_axis, "horizon_index": index, "calibration": calibration(ok),
            "cost": round(sum(r["cost"] for r in rows), 4), "items": len(ok),
            "infra_errors": sum(r["status"] != "ok" for r in rows)}
