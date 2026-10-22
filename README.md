# AGI Horizon

**Open-ended, auto-graded capability ladders for tracking how close each model release gets to general and superhuman capability.**

Most benchmarks saturate: once models score 95%, they stop telling you anything. Here every ability is an
**unbounded difficulty ladder**. Each model gets one number per ability, **θ**: the level where it succeeds
half the time. Any model can be placed on these scales, whether past, present or future. A stronger one simply
climbs higher.

```
          rule      math   coding  planning  language  long-ctx  world   calib
          induction                learning            model
            │        │       │        │         │         │        │       │
  level 20 ─┼────────┼───────┼────────┼─────────┼─────────┼────────┼──     │
            │        │       │        │         │         │        │       │
  level 10 ─┼───●────┼───────┼──●─────┼─────────┼────●────┼────────┼──     │  Brier
            │        │   ●   │        │    ●    │         │   ●    │       │  skill
  level  1 ─┼────────┼───────┼────────┼─────────┼─────────┼────────┼──     │
            ● = θ, the model's frontier level on that axis (no maximum)
```

## What makes it hard to game, and hard to bias

The author of the harness is itself an evaluated model (Claude). [DESIGN.md](DESIGN.md) lists the 10 controls.
The core ones:

- **No LLM judges.** Every score comes from code: exact answers, hidden unit tests, simulators.
- **Never-seen items.** Tasks are procedurally generated from a secret seed, so nothing can be memorized from the
  internet.
- **Pre-registered.** Tasks, grading and weights are hashed into `PREREGISTRATION.lock` before any scored run, and
  the runner refuses to score with modified code.
- **Same treatment for all.** Identical prompt, output cap, reasoning setting and item count for every model.
  Plain-text answer markers rather than any vendor's preferred format.

## The 8 axes

| Axis | Ability | Gets harder by |
|------|---------|----------------|
| `rule_induction` | abstraction (ARC-style grids) | more composed hidden transforms |
| `math` | quantitative reasoning | bigger counting, DP, recurrence and linear-algebra problems |
| `coding` | agentic coding | implementing ever larger *invented* languages, 3 turns with test feedback |
| `planning` | search and planning | larger mazes with chained key→door dependencies |
| `icl_language` | learning from examples | invented languages with more grammar |
| `long_context` | memory and state tracking | longer documents, later overrides, multi-hop joins |
| `world_model` | inferring and simulating dynamics | hidden cellular automata, longer rollouts |
| `calibration` | knowing what it knows | (derived from the stated confidence on every item) |

Pass/fail tasks still give a continuous score, because the score is *where on the ladder* a model stops passing.
That is fitted with a Rasch (item response theory) model and comes with a bootstrap interval.

## Usage

Standard library only, Python ≥ 3.10.

```bash
python -m tests.test_generators          # every generator: deterministic, oracle-solvable, no credit for junk
python -m horizon show planning 6 --answer
export OPENROUTER_API_KEY=...            # one key covers every provider
python -m horizon run --items 1          # pilot: measure cost per model
python -m horizon run --items 6          # full sweep (resumes; hard $ cap via --budget)
python -m horizon score && python -m horizon report   # -> results/summary.json, docs/index.html
python -m horizon human --axis math --name you         # add a human baseline
```

New model released? Add it to `roster.json` (or `--models vendor/model`) and run. Earlier results are
untouched, and the new model lands on the same scales.

## Status

v1: harness, generators and pre-registration are done; the first sweep over 36 models is pending. Human baselines
are not collected yet, so the report makes **no AGI claim** until they are (see DESIGN.md §6).

MIT licensed.
