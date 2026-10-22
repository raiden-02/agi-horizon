"""Long-context reasoning: multi-hop aggregation with state updates.

A long generated document records people, companies, cities and countries,
including later relocations and job changes that override earlier facts.
The question needs every relevant fact, the latest state of each person,
and a multi-hop join (person -> company -> HQ city -> country). Level
doubles the document length every two steps (capped by the run's context
budget) and adds conditions to the question.
"""
from __future__ import annotations

from ..core import Task, parse_int, rng_for

AXIS = "long_context"
MAX_PEOPLE = 640  # ~16k tokens: v1 cost cap (DESIGN.md section 5); override via HORIZON_MAX_PEOPLE

FIRST = ["Ada", "Bo", "Cyra", "Dev", "Eli", "Fen", "Gus", "Hana", "Ivo", "Jun", "Kai", "Lior", "Mira", "Nils",
         "Ola", "Pax", "Quin", "Rhea", "Sol", "Tove", "Uma", "Vik", "Wren", "Xia", "Yael", "Zed"]
LAST = ["Arden", "Brask", "Corvo", "Dahl", "Ekker", "Frost", "Galt", "Hale", "Ito", "Jansen", "Kovac", "Lund",
        "Moreau", "Novak", "Okoro", "Pryce", "Quist", "Roth", "Sato", "Tamm", "Ueda", "Voss", "Wolfe", "Yuen"]
SYL = "ka ri to mel van dor sel tun bri osk ph ae lin gor ves tal".split()
FILLER = [
    "The weather report for the region predicted light rain.",
    "A local festival drew a modest crowd this year.",
    "Several roads were repaved over the summer months.",
    "The library extended its weekend opening hours.",
    "A new bakery opened near the central square.",
    "Commuters reported delays on the northern rail line.",
]


def _max_people():
    import os
    return int(os.environ.get("HORIZON_MAX_PEOPLE", MAX_PEOPLE))


def _place(rng, used):
    while True:
        n = "".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))).capitalize()
        if n not in used:
            used.add(n)
            return n


def generate(level: int, seed: int) -> Task:
    rng = rng_for(AXIS, level, seed)
    n_people = min(_max_people(), 20 * 2 ** ((level - 1) // 2))
    used: set[str] = set()
    countries = [_place(rng, used) for _ in range(3 + level // 3)]
    cities = {_place(rng, used): rng.choice(countries) for _ in range(6 + level)}
    companies = {_place(rng, used) + " " + rng.choice(["Labs", "Works", "Group", "Systems", "Foods"]):
                 rng.choice(list(cities)) for _ in range(5 + level)}
    names = set()
    while len(names) < n_people:
        names.add(f"{rng.choice(FIRST)} {rng.choice(LAST)}-{rng.randint(10, 99)}")
    people = {p: {"city": rng.choice(list(cities)), "co": rng.choice(list(companies)), "age": rng.randint(20, 70)}
              for p in sorted(names)}

    lines = [f"{c} is a city in {k}." for c, k in cities.items()]
    lines += [f"{co} is headquartered in {c}." for co, c in companies.items()]
    lines += [f"{p} is {d['age']} years old, lives in {d['city']} and works at {d['co']}." for p, d in people.items()]
    rng.shuffle(lines)
    updates = []
    if level >= 3:  # later updates override earlier facts
        for p in rng.sample(sorted(people), max(1, n_people // 5)):
            if rng.random() < 0.5:
                people[p]["city"] = rng.choice(list(cities)); updates.append(f"Update: {p} has moved to {people[p]['city']}.")
            else:
                people[p]["co"] = rng.choice(list(companies)); updates.append(f"Update: {p} now works at {people[p]['co']}.")
    for i in range(len(lines) // 6):
        lines.insert(rng.randrange(len(lines)), rng.choice(FILLER))
    doc = "\n".join(lines + ["", "Later records:"] + updates if updates else lines)

    # Question: count people satisfying a chain of conditions on current state.
    country = rng.choice(countries)
    conds = [("hq_country", country)]
    if level >= 2:
        conds.append(("live_country", rng.choice(countries)))
    if level >= 5:
        lo = rng.randint(25, 45); conds.append(("age_range", (lo, lo + rng.randint(10, 20))))
    if level >= 7:
        conds.append(("not_same_city", None))

    def ok(d):
        for kind, v in conds:
            if kind == "hq_country" and cities[companies[d["co"]]] != v: return False
            if kind == "live_country" and cities[d["city"]] != v: return False
            if kind == "age_range" and not (v[0] <= d["age"] <= v[1]): return False
            if kind == "not_same_city" and d["city"] == companies[d["co"]]: return False
        return True

    ans = sum(ok(d) for d in people.values())
    desc = {"hq_country": "work (currently) at a company headquartered in a city in {v}",
            "live_country": "live (currently) in a city in {v}",
            "age_range": "are between {lo} and {hi} years old inclusive",
            "not_same_city": "do not live in the same city where their employer is headquartered"}
    q = " and ".join(desc[k].format(v=v, lo=v[0] if isinstance(v, tuple) else "", hi=v[1] if isinstance(v, tuple) else "")
                     for k, v in conds)
    prompt = (f"Read the records below, then answer the question.\n\n=== RECORDS ===\n{doc}\n=== END ===\n\n"
              f"Question: how many people {q}? Records under 'Later records' override earlier facts. "
              f"Answer with a single integer.")

    def grade(a: str):
        return (1.0 if parse_int(a) == ans else 0.0), None

    return Task(AXIS, level, seed, prompt, grade, str(ans), meta={"n_people": n_people, "chars": len(prompt)})
