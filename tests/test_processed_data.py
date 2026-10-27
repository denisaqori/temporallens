"""Processed-data safety: exact integers, atomic writes, and a prepare step that checks before it skips.

Three failures this closes. Both readers cast labels, repetitions and identifiers to integers
*before* validating them, so 1.9 was read as 1 and a value too large for int16 could wrap.
`save_processed` wrote straight to the final filename, so an interrupted write left a corrupt
archive. And `prepare` skipped any existing file without opening it, so that corrupt archive was
skipped forever.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from _synthetic import recording
from scipy.io import savemat

from temporallens.data import ninapro
from temporallens.data.ninapro import (
    NinaProFormatError,
    load_processed,
    processed_path,
    read_raw_mat,
    save_processed,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _prepare_script():
    spec = importlib.util.spec_from_file_location(
        "temporallens_prepare_script", REPO_ROOT / "scripts" / "prepare_dataset.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw_payload(subject: int = 1) -> dict:
    rec = recording(subject, gestures=17, repetitions=6, samples_per_segment=10)
    return {
        "subject": np.asarray([[subject]], dtype=np.float64),
        "exercise": np.asarray([[1]], dtype=np.float64),
        "emg": rec.emg,
        "restimulus": rec.label[:, None].astype(np.float64),
        "rerepetition": rec.repetition[:, None].astype(np.float64),
    }


def _processed_payload(**changes) -> dict:
    rec = recording(1, gestures=3, repetitions=2)
    payload = {
        "format_version": np.asarray(ninapro.PROCESSED_FORMAT_VERSION, dtype=np.int16),
        "subject": np.asarray(1, dtype=np.int16),
        "emg": rec.emg,
        "label": rec.label,
        "repetition": rec.repetition,
        "label_column": np.asarray(ninapro.LABEL_COLUMN),
        "repetition_column": np.asarray(ninapro.REPETITION_COLUMN),
    }
    payload.update(changes)
    return payload


# --- exact integers ---------------------------------------------------------------------------


def test_float_encoded_whole_numbers_are_accepted_from_raw_files(tmp_path) -> None:
    """MATLAB stores numbers as doubles; whole-number doubles are legitimate."""
    path = tmp_path / "S1_E1_A1.mat"
    savemat(path, _raw_payload())
    assert read_raw_mat(path, subject=1).subject == 1


@pytest.mark.parametrize("field", ["restimulus", "rerepetition"])
def test_a_fractional_raw_label_or_repetition_is_refused(tmp_path, field) -> None:
    payload = _raw_payload()
    values = payload[field].copy()
    values[values > 0] += 0.9  # 1 -> 1.9, which truncation would have read as 1
    payload[field] = values
    path = tmp_path / "S1_E1_A1.mat"
    savemat(path, payload)
    with pytest.raises(NinaProFormatError, match="whole numbers"):
        read_raw_mat(path, subject=1)


def test_a_fractional_raw_subject_is_refused(tmp_path) -> None:
    payload = _raw_payload()
    payload["subject"] = np.asarray([[1.9]])
    path = tmp_path / "S1_E1_A1.mat"
    savemat(path, payload)
    with pytest.raises(NinaProFormatError, match="whole number"):
        read_raw_mat(path, subject=1)


def test_a_raw_value_too_large_for_int16_is_refused_rather_than_wrapped(tmp_path) -> None:
    payload = _raw_payload()
    payload["restimulus"] = payload["restimulus"].copy()
    payload["restimulus"][0, 0] = 65_536.0 + 3  # would wrap to 3 in int16
    path = tmp_path / "S1_E1_A1.mat"
    savemat(path, payload)
    with pytest.raises(NinaProFormatError, match="int16"):
        read_raw_mat(path, subject=1)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        (
            {"label": recording(1, gestures=3, repetitions=2).label.astype(np.float64) + 0.5},
            "whole numbers",
        ),
        ({"subject": np.asarray(1.9)}, "whole number"),
        ({"format_version": np.asarray(1.5)}, "whole number"),
    ],
)
def test_a_fractional_processed_value_is_refused(tmp_path, changes, reason) -> None:
    path = processed_path(tmp_path, 1)
    np.savez_compressed(path, **_processed_payload(**changes))
    with pytest.raises(NinaProFormatError, match=reason):
        load_processed(path)


# --- atomic writes ----------------------------------------------------------------------------


def test_an_interrupted_write_leaves_no_file_behind(tmp_path, monkeypatch) -> None:
    rec = recording(1, gestures=3, repetitions=2)
    destination = processed_path(tmp_path, 1)
    real = np.savez_compressed

    def interrupted(path, **arrays):
        real(path, **arrays)
        Path(path).write_bytes(Path(path).read_bytes()[:100])  # truncated, like a killed process
        raise KeyboardInterrupt

    monkeypatch.setattr(ninapro.np, "savez_compressed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_processed(rec, destination)
    assert not destination.exists(), "a partial archive must never sit at the final name"
    assert list(tmp_path.iterdir()) == [], "the temporary file must be cleaned up"


def test_a_write_replaces_an_existing_file_whole(tmp_path) -> None:
    destination = processed_path(tmp_path, 1)
    destination.write_bytes(b"stale")
    save_processed(recording(1, gestures=3, repetitions=2), destination)
    assert load_processed(destination).subject == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == [destination.name]


# --- prepare checks before it skips -----------------------------------------------------------


def _raw_dir(tmp_path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    savemat(raw / "S1_E1_A1.mat", _raw_payload())
    return raw


def test_a_corrupt_existing_file_is_rebuilt_not_skipped(tmp_path, capsys) -> None:
    raw, out = _raw_dir(tmp_path), tmp_path / "processed"
    out.mkdir()
    processed_path(out, 1).write_bytes(b"PK\x03\x04 truncated")
    assert _prepare_script().prepare(raw, out, [1], force=False) == 0
    assert "REBUILD" in capsys.readouterr().out
    assert load_processed(processed_path(out, 1)).subject == 1


def test_a_valid_existing_file_is_skipped_untouched(tmp_path, capsys) -> None:
    raw, out = _raw_dir(tmp_path), tmp_path / "processed"
    _prepare_script().prepare(raw, out, [1], force=False)
    destination = processed_path(out, 1)
    before = os.stat(destination).st_mtime_ns
    assert _prepare_script().prepare(raw, out, [1], force=False) == 0
    assert "skip" in capsys.readouterr().out
    assert os.stat(destination).st_mtime_ns == before


def test_a_bare_subjects_flag_is_an_error_not_all_forty(tmp_path) -> None:
    """`--subjects` with no values used to mean every subject: a ~3 GB surprise."""
    with pytest.raises(SystemExit) as exit_info:
        _prepare_script().main(
            ["--subjects", "--raw-dir", str(tmp_path), "--out-dir", str(tmp_path)]
        )
    assert exit_info.value.code == 2


def test_an_atomic_write_keeps_the_usual_file_permissions(tmp_path) -> None:
    """`tempfile.mkstemp` creates owner-only files, and moving one into place keeps that mode.
    A processed file must get the same permissions as any other file the user writes."""
    reference = tmp_path / "reference"
    reference.write_bytes(b"")
    destination = processed_path(tmp_path, 1)
    save_processed(recording(1, gestures=3, repetitions=2), destination)
    assert os.stat(destination).st_mode & 0o777 == os.stat(reference).st_mode & 0o777
