"""
Run the three sample investigations from the problem statement and print
evidence-backed, markdown-formatted answers.

Usage:
    python3 demo.py
    python3 demo.py --case a        # run only Test Input A
    python3 demo.py --json data/my_case.json   # run a custom case
"""

import argparse
import os

from agent import load_case

CASES = {
    "a": "data/test_a.json",
    "b": "data/test_b.json",
    "c": "data/test_c.json",
    "d": "data/test_d.json",
}


def run_case(label: str, path: str) -> None:
    question, agent = load_case(path)
    result = agent.investigate(question)
    print("=" * 78)
    print(f"TEST INPUT {label.upper()}  ({path})")
    print("=" * 78)
    print(result.to_markdown())
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=list(CASES.keys()), help="run a single case (a/b/c)")
    parser.add_argument("--json", help="path to a custom documents.json case file")
    args = parser.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(here)

    if args.json:
        run_case("custom", args.json)
    elif args.case:
        run_case(args.case, CASES[args.case])
    else:
        for label, path in CASES.items():
            run_case(label, path)


if __name__ == "__main__":
    main()
