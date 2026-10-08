#!/usr/bin/env python3
"""Convert raw NinaPro DB2 Exercise B ``.mat`` files into the processed per-subject format.

Reads ``data/raw/ninapro_db2/exercise_b/S{n}_E1_A1.mat`` and writes
``data/processed/ninapro_db2/subject_{n:02d}.npz``. Labels come from ``restimulus`` and
``rerepetition`` (D12), and the processed file records which columns it was built from, so a stale
file prepared from the uncorrected columns fails loudly at load rather than training happily on
mislabelled onsets.

Windowing is deliberately **not** done here. F0 uses stride 200 and F1 stride 100 over the same
recordings, so baking windows into the processed file would force a re-prepare whenever a stride
changed and would let two runs drift out of comparability if someone forgot.

The dataset is never redistributed with this repository and is never fabricated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from temporallens.data.ninapro import (
    NUM_SUBJECTS,
    NinaProFormatError,
    processed_path,
    read_raw_mat,
    save_processed,
)

DEFAULT_RAW = Path("data/raw/ninapro_db2/exercise_b")
DEFAULT_PROCESSED = Path("data/processed/ninapro_db2")


def raw_path(raw_dir: Path, subject: int) -> Path:
    """Exercise B is DB2's first exercise, so the shipped filename is ``E1``."""
    return raw_dir / f"S{subject}_E1_A1.mat"


def prepare(raw_dir: Path, out_dir: Path, subjects: list[int], *, force: bool) -> int:
    failures = 0
    for subject in subjects:
        source, destination = raw_path(raw_dir, subject), processed_path(out_dir, subject)
        if destination.exists() and not force:
            print(f"  S{subject:<3} skip     already prepared ({destination.name})")
            continue
        if not source.is_file():
            print(f"  S{subject:<3} MISSING  {source}")
            failures += 1
            continue
        try:
            recording = read_raw_mat(source, subject=subject)
        except NinaProFormatError as error:
            print(f"  S{subject:<3} REJECT   {error}")
            failures += 1
            continue
        save_processed(recording, destination)
        print(
            f"  S{subject:<3} ok       {recording.num_samples:>9,} samples"
            f"  -> {destination.name}"
        )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED)
    parser.add_argument(
        "--subjects",
        type=int,
        nargs="*",
        help=f"subject ids to prepare; default is 1..{NUM_SUBJECTS}",
    )
    parser.add_argument(
        "--force", action="store_true", help="re-prepare subjects that already exist"
    )
    args = parser.parse_args(argv)

    subjects = args.subjects or list(range(1, NUM_SUBJECTS + 1))
    print(f"preparing {len(subjects)} subject(s): {args.raw_dir} -> {args.out_dir}")
    failures = prepare(args.raw_dir, args.out_dir, subjects, force=args.force)
    if failures:
        print(f"\n{failures} subject(s) could not be prepared.")
        return 1
    print("\nall requested subjects prepared.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
