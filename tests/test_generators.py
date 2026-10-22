"""Self-test: every generated task is deterministic, solvable by its oracle,
and gives no credit to degenerate replies. Run: python -m tests.test_generators"""
import sys
import time

from horizon.axes import AXES
from horizon.core import PROTOCOL, parse_reply

LEVELS = range(1, int(sys.argv[1]) + 1) if len(sys.argv) > 1 else range(1, 15)
SEEDS = range(4)


def wrap(ans):
    return f"Some reasoning.\nFINAL ANSWER:\n{ans}\nCONFIDENCE: 80"


PARSE_CASES = [  # (reply, expected answer, expected confidence) - styles seen from real models
    ("x\nFINAL ANSWER:\n42\nCONFIDENCE: 90", "42", 0.9),
    ("**FINAL ANSWER:**\n42\n**CONFIDENCE:** 90%", "42", 0.9),
    ("FINAL ANSWER:\n**\n\n0079\n0091\n\n---\n\n**Confidence:** 70", "0079\n0091", 0.7),
    ("FINAL ANSWER:\n```\n0110\n1001\n```\nCONFIDENCE: 55", "0110\n1001", 0.55),
    ("draft FINAL ANSWER: 1\nmore thought\nFINAL ANSWER: 2\nCONFIDENCE: 10", "2", 0.1),
    ("no marker here, answer is 5", None, None),
]


def test_parse():
    bad = 0
    for reply, ans, conf in PARSE_CASES:
        got = parse_reply(reply)
        if got != (ans, conf):
            bad += 1
            print(f"PARSE FAIL {reply!r}: {got}")
    return bad


def main():
    bad = test_parse()
    for axis, gen in AXES.items():
        for level in LEVELS:
            t0 = time.time()
            for seed in SEEDS:
                task = gen(level, seed)
                assert task.prompt == gen(level, seed).prompt, f"{axis} L{level} not deterministic"
                ans, conf = parse_reply(wrap(task.oracle))
                s, _ = task.grade(ans)
                junk = max(task.grade("")[0], task.grade("I don't know")[0])
                if s != 1.0 or junk > 0 or conf != 0.8:
                    bad += 1
                    print(f"FAIL {axis} L{level} s{seed}: oracle={s} junk={junk}")
            dt = (time.time() - t0) / len(SEEDS)
            print(f"{axis:15s} L{level:2d} ok  prompt~{len(task.full_prompt()) // 4:6d} tok  gen {dt:5.2f}s")
    print("ALL OK" if not bad else f"{bad} FAILURES")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
