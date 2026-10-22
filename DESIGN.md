# agi-horizon: design and pre-registration (v1)

This file and the scoring code are hashed into `PREREGISTRATION.lock` **before the first scored run**.
`python -m horizon run` refuses to produce leaderboard results if the code no longer matches the lock.
Any change to tasks, grading or scoring is a new version with a new lock. Results from different locks
are never mixed.

## 1. Question

> Is each new model release closer to general (AGI) or beyond-human (superintelligent) capability, and by how much?

"General" is measured as **breadth × depth**: how far up an open-ended difficulty ladder a model gets on each of
several distinct cognitive abilities. A model counts as more general when it climbs higher on *all* of them,
not when it spikes on one.

## 2. Scoring: a frontier level, not pass/fail

```
P(success)
 1.0 ┤■■■■■■■■■■■■■■■■■■■■■
     │                     ■■■
 0.5 ┤· · · · · · · · · · · · ·■· · · ·   ← θ: the level where the model succeeds half the time
     │                           ■■■
 0.0 ┤                              ■■■■■■■■■
     └┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→ level (unbounded)
      1  2  3  4  5  6  7  8  9  10 11 12 13
```

* Every axis has an integer **level** ≥ 1 with no maximum. Each level is a fresh, procedurally generated task that
  is strictly harder in a stated way (bigger grid, more rules, longer document, longer rollout...).
* Each item scores in [0, 1]. Some items are exact match (1/0); others give principled partial credit, e.g. the
  share of hidden tests passed, or optimal path length divided by the model's path length.
* **Ability θ** per axis is fitted with a one-parameter logistic (Rasch) model,
  `P(success at L) = sigmoid(1.0 · (θ − L))`, plus a weak N(3, 8²) prior. Partial credit counts as a
  fractional outcome. A 90% interval comes from 300 bootstrap resamples.
* A yes/no task still yields a continuous score, because *which* levels a model passes is the measurement. A task
  that is easy today stays informative tomorrow: future models just climb higher. There is no saturation.
* **Adaptive testing (staircase):** start at level 2, climb +2 per success until the first failure, then step
  +1/−1. Items concentrate near each model's frontier, which is the most informative and cheapest place to test.
* **Horizon Index** = unweighted mean θ over the 7 laddered axes (equal weights fixed here; no tuning after results).
  Levels on different axes are not in the same units, so the index tracks change over time. It is not a
  cross-axis comparison. Human anchors (section 6) make the axes comparable.

## 3. Axes

| # | Axis | Ability (CHC / literature) | What the level grows | Item score |
|---|------|----------------------------|----------------------|------------|
| 1 | `rule_induction` | Fluid reasoning, abstraction (ARC, Chollet 2019) | number of composed grid transforms, grid size, colors | exact grid |
| 2 | `math` | Quantitative reasoning | problem size: modular counting to 10^L, DP lattices, recurrences to huge N, linear systems | exact integer |
| 3 | `coding` | Agentic software work (METR task horizon) | invented language with more ops, then modulus, loops, underflow rules, conditionals, longer programs; 3 turns with test feedback | share of 25 hidden tests passed |
| 4 | `planning` | Planning / search | maze size, number of key→door dependencies | optimal / used moves (0 if invalid) |
| 5 | `icl_language` | Learning a new skill from examples | invented grammar features (order, morphology, case, negation, agreement, subordinate clauses), vocabulary | share of 3 sentences exact |
| 6 | `long_context` | Working memory over long inputs, state tracking | document length doubles every 2 levels; later overrides; more join conditions | exact integer |
| 7 | `world_model` | Inferring hidden dynamics, mental simulation | colors, neighborhood radius, rollout horizon | share of predicted rows exact |
| 8 | `calibration` | Metacognition: knowing what you don't know | derived from every item: the model states P(correct) | Brier skill = 1 − mean (conf − score)² |

Design constraints on every generator, checked by `tests/test_generators.py` for levels 1–20:
deterministic per seed; the reference answer scores 1.0; empty or "I don't know" scores 0; the task is fully
determined by the prompt. For example, every grammar feature and word in a language test appears ≥ 2× in its
examples, and every neighborhood a world-model rollout needs appears in the observed history.

Deliberately **not** measured in v1: factual knowledge and trivia (contamination-prone and not tangible progress
toward generality), vision, real-world tool use, and multi-hour autonomy. These are candidates for v2.

## 4. Bias controls

The author of v1 (Claude Opus 5.5, an Anthropic model) **is one of the models being graded**. That is a real
conflict of interest. These controls are designed to make the author's identity irrelevant to the results:

1. **No model grades anything.** All scoring is deterministic code: exact match, unit tests, simulators. There is no
   LLM-as-judge anywhere.
2. **Fresh, unmemorizable items.** Every scored item is generated from a secret salt (`.secret_salt`, never
   committed). The generator code is public, but the scored instances are not, so no model can have trained on
   them. Raw transcripts are kept out of git for the same reason.
3. **Frozen before results.** Tasks, levels, grading, the scoring model, weights and the roster rule are locked by
   hash before the first scored run. They cannot be tuned after seeing who wins.
4. **Neutral protocol.** One identical prompt for every model, with plain-text `FINAL ANSWER:` / `CONFIDENCE:`
   markers rather than any vendor's preferred markup (XML tags, JSON mode, tool calls). There is no per-model prompt
   engineering and no system prompt. Parsing is equally lenient for everyone: it tolerates markdown bold and code
   fences, and the parser tests use reply styles from several vendors.
5. **Identical compute settings.** Same 16k output-token cap (reasoning included), the same `reasoning.effort =
   medium` request sent to every model (ignored by non-reasoning models), and provider default temperature.
6. **Roster by rule.** Models are chosen by the rule in `roster.json`, not by expected performance.
7. **Equal item counts.** Every model gets the same number of scored items per axis. The budget sets that number,
   not who gets tested.
8. **Abilities from outside sources.** The axes map to established taxonomies (CHC broad abilities, ARC, METR's
   task-horizon idea), not to what the author happens to be good at.
9. **Infra failures ≠ wrong answers.** API errors are retried and, if persistent, recorded as missing, not as 0.
   Exceeding a model's context window *is* scored 0, because that is a real capability limit.
10. **Audit trail.** Every item row (level, score, confidence, tokens, cost) is committed in `results/runs/`. Anyone
    with their own salt can rerun the whole thing and check whether the ranking replicates.

If a model from the author's own family ranks first, treat that with extra skepticism until an independent rerun
with a different salt confirms it.

## 5. Run protocol and budget

* Backend: OpenRouter (one key, every provider). Settings as in section 4.5.
* Long-context documents are capped at 640 people (~16k tokens) for the v1 sweep (`HORIZON_MAX_PEOPLE`), for cost.
  Above that cap the axis still gets harder through more question conditions, but only until level 7. This is a
  known v1 ceiling.
* **Budget allocation (cap $50):** first a pilot of 1 item per axis for every roster model, which measures the real
  cost per item. The items-per-axis number N is then the largest that fits the remaining budget **for every model
  equally**. Pilot items count toward N.

## 6. Human anchors: what "AGI" and "superintelligence" mean on these scales

`python -m horizon human --axis <axis> --name <you>` runs the exact same staircase on a person. With enough
people per axis:

* **AGI line** for an axis = median θ of skilled adults given the same time-unbounded, tool-free setting.
* **Superintelligence line** = above the best human θ observed.
* A model is "AGI-level" when it reaches the AGI line on **every** axis. It is "superhuman" when it is above the
  best human on every axis.

Until human data exists, reports show raw θ and trends only, and make **no** AGI claim.

## 7. Known limitations (v1)

* Equal-slope Rasch with integer levels assumes each level step is equally hard within an axis. That is
  approximate. Item fit can be checked once the data exists.
* Rule-induction tasks can occasionally admit more than one consistent rule. That lowers everyone's ceiling
  equally, but it adds noise.
* Item counts are small per model (budget), so CIs are wide. More items narrow them without changing the design.
* Release dates are OpenRouter listing dates.
* Text-only. Agentic coding is a 3-turn loop, not a full repository environment.
