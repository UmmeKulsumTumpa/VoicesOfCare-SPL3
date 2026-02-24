#!/usr/bin/env python3
"""
scripts/01_prepare_dataset.py
==============================
Load, clean, merge, and split the raw JSONL datasets.

What this script does:
  1. Loads all JSONL files listed in the config.
  2. Validates required fields (instruction / input / output).
  3. Deduplicates records based on (input, output) hash.
  4. Filters out extremely short or empty outputs.
  5. Splits into train / val / test sets.
  6. Saves processed splits to processed_data/ as JSONL files.
  7. Prints a summary report with example records.

Usage:
  cd training/
  python scripts/01_prepare_dataset.py [--config config/training_config.yaml]
                                        [--max_samples 500]
                                        [--seed 42]

Outputs:
  processed_data/
      train.jsonl
      val.jsonl
      test.jsonl
      dataset_stats.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

# ── Allow imports from project root ──────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_jsonl(filepath: str | Path) -> list[dict]:
    """Load a JSONL file and return a list of dicts."""
    records = []
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [WARN] Skipping invalid JSON on line {line_no}: {e}")
    return records


def record_hash(record: dict) -> str:
    """Compute a stable hash over (input, output) to detect duplicates."""
    key = record.get("input", "") + "|||" + record.get("output", "")
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def validate_record(record: dict) -> tuple[bool, str]:
    """
    Check that a record has all required fields and non-trivial content.

    Returns (is_valid, reason_if_invalid).
    """
    required = ("instruction", "input", "output")
    for field in required:
        if field not in record:
            return False, f"Missing field: '{field}'"
        if not isinstance(record[field], str):
            return False, f"Field '{field}' is not a string"
        if not record[field].strip():
            return False, f"Field '{field}' is empty"

    # Filter outputs that are too short to be meaningful responses
    min_output_chars = 15
    if len(record["output"].strip()) < min_output_chars:
        return False, f"Output too short ({len(record['output'].strip())} chars)"

    # Filter inputs with no Reddit Post content
    if "Reddit Post:" not in record["input"]:
        return False, "Input missing 'Reddit Post:' section"

    return True, ""


def save_jsonl(records: list[dict], filepath: str | Path) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  Saved {len(records):,} records → {filepath}")


def split_dataset(
    records: list[dict],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Shuffle and split records into train / val / test.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )
    rng = random.Random(seed)
    data = records.copy()
    rng.shuffle(data)

    n = len(data)
    n_val = max(1, int(n * val_ratio))
    n_test = max(1, int(n * test_ratio))
    n_train = n - n_val - n_test

    train = data[:n_train]
    val = data[n_train : n_train + n_val]
    test = data[n_train + n_val :]

    return train, val, test


def compute_stats(records: list[dict]) -> dict[str, Any]:
    """Compute basic statistics over the dataset split."""
    output_lens = [len(r["output"].split()) for r in records]
    input_lens = [len(r["input"].split()) for r in records]

    return {
        "count": len(records),
        "output_words": {
            "min": min(output_lens),
            "max": max(output_lens),
            "mean": round(sum(output_lens) / len(output_lens), 1),
        },
        "input_words": {
            "min": min(input_lens),
            "max": max(input_lens),
            "mean": round(sum(input_lens) / len(input_lens), 1),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    # ── Load config ──────────────────────────────────────────────────────────
    config = load_config(args.config)
    data_cfg = config["data"]

    seed = args.seed if args.seed is not None else data_cfg.get("seed", 42)
    max_samples = args.max_samples if args.max_samples else data_cfg.get("max_samples")
    processed_dir = Path(args.output_dir or data_cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits = data_cfg["splits"]
    train_ratio = splits["train"]
    val_ratio = splits["val"]
    test_ratio = splits["test"]

    print("\n" + "=" * 65)
    print("  Alzheimer/Dementia Caregiver Chatbot — Dataset Preparation")
    print("=" * 65)

    # ── Load raw data ────────────────────────────────────────────────────────
    all_records: list[dict] = []

    for rel_path in data_cfg["files"]:
        # Resolve relative paths from the project root
        fpath = Path(rel_path) if Path(rel_path).is_absolute() else PROJECT_DIR / rel_path
        print(f"\nLoading: {fpath}")
        records = load_jsonl(fpath)
        print(f"  Loaded {len(records):,} raw records")
        all_records.extend(records)

    print(f"\nTotal raw records: {len(all_records):,}")

    # ── Validate & deduplicate ───────────────────────────────────────────────
    print("\nValidating and deduplicating...")
    seen_hashes: set[str] = set()
    clean_records: list[dict] = []
    invalid_count = 0
    dup_count = 0

    for rec in tqdm(all_records, desc="  Processing"):
        valid, reason = validate_record(rec)
        if not valid:
            invalid_count += 1
            continue

        h = record_hash(rec)
        if h in seen_hashes:
            dup_count += 1
            continue
        seen_hashes.add(h)
        clean_records.append(rec)

    print(f"  Removed {invalid_count:,} invalid records")
    print(f"  Removed {dup_count:,} duplicate records")
    print(f"  Clean unique records: {len(clean_records):,}")

    # ── Optional: cap sample count ───────────────────────────────────────────
    if max_samples and len(clean_records) > max_samples:
        rng = random.Random(seed)
        clean_records = rng.sample(clean_records, max_samples)
        print(f"  Sampled down to: {max_samples:,} records")

    # ── Split ────────────────────────────────────────────────────────────────
    print(f"\nSplitting: {train_ratio:.0%} train / {val_ratio:.0%} val / {test_ratio:.0%} test")
    train, val, test = split_dataset(
        clean_records, train_ratio, val_ratio, test_ratio, seed
    )

    # ── Save splits ──────────────────────────────────────────────────────────
    print("\nSaving processed splits...")
    save_jsonl(train, processed_dir / "train.jsonl")
    save_jsonl(val,   processed_dir / "val.jsonl")
    save_jsonl(test,  processed_dir / "test.jsonl")

    # ── Compute & save stats ─────────────────────────────────────────────────
    stats = {
        "total_clean_records": len(clean_records),
        "train": compute_stats(train),
        "val":   compute_stats(val),
        "test":  compute_stats(test),
        "config": {
            "seed": seed,
            "max_samples": max_samples,
            "splits": splits,
        },
    }

    stats_path = processed_dir / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats   → {stats_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("DATASET SUMMARY")
    print("─" * 65)
    print(f"  Train : {len(train):>7,} examples")
    print(f"  Val   : {len(val):>7,} examples")
    print(f"  Test  : {len(test):>7,} examples")
    print(f"\n  Avg input  word count : {stats['train']['input_words']['mean']}")
    print(f"  Avg output word count : {stats['train']['output_words']['mean']}")

    print("\n--- Example training record (first) ---")
    ex = train[0]
    print(f"  instruction : {ex['instruction'][:80]}...")
    print(f"  input       : {ex['input'][:120].strip()}...")
    print(f"  output      : {ex['output'][:120].strip()}...")
    print("─" * 65)
    print("\nDataset preparation complete. Ready for training.\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare and split the caregiver chatbot dataset."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to the YAML config file (default: config/training_config.yaml)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Cap the dataset size for quick testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override the processed_data output directory",
    )

    args = parser.parse_args()
    main(args)
