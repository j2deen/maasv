"""CLI runner: python -m evals.run_eval [--json] [--k N]"""

import argparse
import json
import sys

from evals.harness import run_eval, format_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the maasv eval harness")
    parser.add_argument("--k", type=int, default=5, help="retrieval depth (default 5)")
    parser.add_argument("--json", action="store_true", help="emit full metrics as JSON")
    args = parser.parse_args()

    metrics = run_eval(k=args.k)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_report(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
