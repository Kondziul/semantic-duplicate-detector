"""
reporter.py — Human-readable output for duplicate detection results.

Produces:
  • A pretty-printed terminal table (via tabulate)
  • A CSV report saved to disk
  • A summary printed to stdout
"""

from __future__ import annotations

import csv
import textwrap
from pathlib import Path
from typing import Optional

from tabulate import tabulate

from src.detector import DetectionResult


# Maximum characters shown per cell in the terminal table
_TRUNCATE_AT = 72


def _truncate(text: str, width: int = _TRUNCATE_AT) -> str:
    """Wrap long strings for cleaner terminal display."""
    return "\n".join(textwrap.wrap(text, width=width))


def print_report(result: DetectionResult) -> None:
    """Print a formatted duplicate report to stdout."""
    print("\n" + "=" * 80)
    print("  SEMANTIC DUPLICATE DETECTOR — Results")
    print("=" * 80)
    print(f"  Fragments analysed : {result.n_fragments}")
    print(f"  Similarity threshold: {result.threshold:.2f}")
    print(f"  Candidate pairs found: {result.n_pairs}")
    print("=" * 80 + "\n")

    if result.n_pairs == 0:
        print("  No duplicate candidates found above the threshold.\n")
        return

    rows = [
        [
            f"{p.idx_a} ↔ {p.idx_b}",
            f"{p.similarity:.4f}",
            _truncate(p.text_a),
            _truncate(p.text_b),
        ]
        for p in result.pairs
    ]

    headers = ["Pair", "Similarity", "Fragment A", "Fragment B"]
    print(tabulate(rows, headers=headers, tablefmt="rounded_outline"))
    print()


def save_csv(result: DetectionResult, output_path: Optional[Path] = None) -> Path:
    """
    Save duplicate pairs to a CSV file.

    Args:
        result: DetectionResult from detect_duplicates().
        output_path: Destination path. Defaults to outputs/duplicates.csv.

    Returns:
        The path where the file was written.
    """
    if output_path is None:
        output_path = Path("outputs") / "duplicates.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["pair", "similarity", "fragment_a", "fragment_b"]

    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for p in result.pairs:
            writer.writerow(
                {
                    "pair": f"{p.idx_a}-{p.idx_b}",
                    "similarity": f"{p.similarity:.6f}",
                    "fragment_a": p.text_a,
                    "fragment_b": p.text_b,
                }
            )

    print(f"[reporter] CSV saved → {output_path.resolve()}")
    return output_path
