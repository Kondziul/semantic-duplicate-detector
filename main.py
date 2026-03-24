"""
main.py — Command-line entry point for the Semantic Duplicate Detector.

Usage examples:
    # Run with default threshold (0.80) on built-in sample data
    python main.py

    # Custom threshold
    python main.py --threshold 0.75

    # Point at a plain-text file (one sentence / fragment per line)
    python main.py --input my_docs.txt --threshold 0.82

    # Skip CSV export
    python main.py --no-csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Allow running from the project root without installing as a package
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from data.sample_docs import SAMPLE_FRAGMENTS
from src.embedder import FragmentEmbedder
from src.detector import detect_duplicates
from src.reporter import print_report, save_csv


def load_fragments_from_file(path: Path) -> list[str]:
    """Read a plain-text file, one fragment per non-empty line."""
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    fragments = [l for l in lines if l]
    if not fragments:
        raise ValueError(f"No non-empty lines found in {path}")
    return fragments


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect semantically duplicate fragments in technical documentation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="FILE",
        help="Path to a .txt file with one fragment per line. "
             "If omitted, built-in sample data is used.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.80,
        metavar="FLOAT",
        help="Cosine similarity threshold (0.0–1.0). Pairs above this value are flagged.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        metavar="MODEL",
        help="sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the CSV report to disk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/duplicates.csv"),
        metavar="FILE",
        help="Path for the CSV output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load fragments
    # ------------------------------------------------------------------
    if args.input is not None:
        print(f"[main] Loading fragments from: {args.input}")
        fragments = load_fragments_from_file(args.input)
    else:
        print("[main] Using built-in sample documentation fragments.")
        fragments = SAMPLE_FRAGMENTS

    print(f"[main] {len(fragments)} fragments loaded.")

    # ------------------------------------------------------------------
    # 2. Compute embeddings
    # ------------------------------------------------------------------
    embedder = FragmentEmbedder(model_name=args.model)
    embeddings = embedder.encode(fragments, show_progress=True)

    # ------------------------------------------------------------------
    # 3. Detect duplicates
    # ------------------------------------------------------------------
    print(f"[main] Running duplicate detection (threshold={args.threshold:.2f}) …")
    result = detect_duplicates(fragments, embeddings, threshold=args.threshold)

    # ------------------------------------------------------------------
    # 4. Report
    # ------------------------------------------------------------------
    print_report(result)

    if not args.no_csv:
        save_csv(result, output_path=args.output)


if __name__ == "__main__":
    main()
