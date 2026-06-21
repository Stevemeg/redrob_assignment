#!/usr/bin/env python3
"""Runs every tests/test_*.py file and reports a combined pass/fail summary."""
import subprocess
import sys
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent / "tests"


def main():
    test_files = sorted(TEST_DIR.glob("test_*.py"))
    total_fail = 0
    for tf in test_files:
        print(f"\n=== {tf.name} ===")
        result = subprocess.run([sys.executable, str(tf)])
        if result.returncode != 0:
            total_fail += 1
    print(f"\n{'='*40}")
    if total_fail:
        print(f"{total_fail}/{len(test_files)} test files had failures.")
        sys.exit(1)
    print(f"All {len(test_files)} test files passed.")


if __name__ == "__main__":
    main()
