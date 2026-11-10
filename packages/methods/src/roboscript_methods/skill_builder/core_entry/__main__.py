"""Read entry-check reports.

    python -m roboscript_methods.skill_builder.core_entry summarize <reports> --gates <gate file>
    python -m roboscript_methods.skill_builder.core_entry compare run1=<reports> run2=<reports> ...

`<reports>` is a directory of a benchmark glue's `<task>.json` reports; judging them again with
a benchmark's criteria is its glue's (e.g. `entry_check.py --rejudge`).
"""

import argparse
from pathlib import Path

from roboscript_methods.skill_builder.core_entry.report import compare, summarize


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m roboscript_methods.skill_builder.core_entry", description=__doc__
    )
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize", help="metrics, the gate's verdict, the failures")
    summary.add_argument("reports", type=Path)
    summary.add_argument("--gates", type=Path, required=True)
    side = commands.add_parser("compare", help="runs side by side, object by object")
    side.add_argument("runs", nargs="+", help="name=directory")
    args = parser.parse_args()
    if args.command == "summarize":
        summarize(args.reports, args.gates)
    else:
        runs = dict(arg.split("=", 1) for arg in args.runs)
        compare({name: Path(directory) for name, directory in runs.items()})


if __name__ == "__main__":
    main()
