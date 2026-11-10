"""Conformance of a package's part of a pipeline's status and summary
(`roboscript.orchestrate.report.ReportPart`): it reads at least one stage, its columns have
headers a Markdown table can hold, and on records where no job ran yet (`keys` under `root`) the
status and summary it composes alone are JSON, their tables render with a cell per column, and
the taxonomy fields it adds are labelled."""

import json
from collections.abc import Sequence
from pathlib import Path

from roboscript.orchestrate.jobs import RunKey
from roboscript.orchestrate.report import TAXONOMY, Report, ReportPart


def check_report_part(part: ReportPart, keys: Sequence[RunKey], root: Path) -> list[str]:
    """What is wrong with `part`, one line each (empty: it reports)."""
    lines = []
    if not part.readings:
        lines.append("reads no stage")
    for column in (*part.status_columns, *part.summary_columns):
        if not column.header or "|" in column.header or "\n" in column.header:
            lines.append(f"column {column.header!r}: not a Markdown table header")
    report = Report((part,))
    status = report.status(root, keys, 0.0)
    found = report.summary(root, keys, 0.0)
    for name, data in (("status", status), ("summary", found)):
        try:
            json.dumps(data)
        except TypeError as error:
            lines.append(f"{name}: not JSON ({error})")
    width = len(part.status_columns) + 8
    for line in report.status_table(status).splitlines():
        if line.startswith("| ") and line.count(" | ") != width - 1:
            lines.append(f"status table line of {line.count(' | ') + 1} cells: {line!r}")
    report.summary_page(found)
    for row in found["rows"]:
        for name in set(row["taxonomy"]) - set(TAXONOMY) - set(part.taxonomy_labels):
            lines.append(f"{row['run']}: taxonomy field {name!r} has no label")
    return lines
