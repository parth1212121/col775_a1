#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


PROMPT_PATTERN = re.compile(r"^\s*translate to [^:]+:\s*", re.IGNORECASE)
LATIN_PATTERN = re.compile(r"[A-Za-z]")
DEVANAGARI_PATTERN = re.compile(r"[\u0900-\u097F]")


def quantile(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    return sorted_values[int((len(sorted_values) - 1) * p)]


def inspect_file(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    source_lengths = []
    target_lengths = []
    prefixed = 0
    latin_targets = 0
    no_devanagari_targets = 0
    examples_with_latin = []

    for row in rows:
        source = row["source"]
        target = row["translated"]
        source_clean = PROMPT_PATTERN.sub("", source).strip()

        if PROMPT_PATTERN.match(source):
            prefixed += 1
        if LATIN_PATTERN.search(target):
            latin_targets += 1
            if len(examples_with_latin) < 5:
                examples_with_latin.append((source[:100], target[:120]))
        if not DEVANAGARI_PATTERN.search(target):
            no_devanagari_targets += 1

        source_lengths.append(len(source_clean.split()))
        target_lengths.append(len(target.split()))

    source_lengths.sort()
    target_lengths.sort()

    print(f"FILE: {path}")
    print(f"rows: {len(rows)}")
    print(f"columns: {list(rows[0].keys()) if rows else []}")
    print(f"prompt_prefixed_sources: {prefixed}/{len(rows)}")
    print(f"source_words p50/p95/p99/max: {statistics.median(source_lengths):.0f}/{quantile(source_lengths, 0.95)}/{quantile(source_lengths, 0.99)}/{source_lengths[-1]}")
    print(f"target_words p50/p95/p99/max: {statistics.median(target_lengths):.0f}/{quantile(target_lengths, 0.95)}/{quantile(target_lengths, 0.99)}/{target_lengths[-1]}")
    print(f"targets_with_latin_chars: {latin_targets}")
    print(f"targets_without_devanagari_chars: {no_devanagari_targets}")
    print("sample_latin_targets:")
    for source, target in examples_with_latin:
        print(f"  SRC: {source}")
        print(f"  TGT: {target}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a translation dataset format and noise profile.")
    parser.add_argument(
        "--data-dir",
        default="data/translation",
        help="Dataset directory containing the Hindi/Marathi CSV files.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {data_dir}")

    for path in files:
        inspect_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
