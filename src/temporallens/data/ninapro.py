"""Read NinaPro DB2 Exercise B, and define the processed on-disk format.

Two stages, deliberately separated:

``read_raw_mat``
    Parse one subject's ``.mat`` exactly as the dataset authors shipped it, taking labels from
    ``restimulus`` and ``rerepetition`` (D12). This is the only code that touches ``data/raw/``.

``load_processed`` / ``save_processed``
    A ``.npz`` per subject holding the continuous signal plus its label and repetition streams.
    Windowing happens at load time rather than here, because F0 and F1 use different strides
    (200 and 100) over the same recording — baking windows into the processed file would force a
    re-prepare whenever a stride changes, and would make the two runs silently incomparable if
    someone forgot.

The dataset is never redistributed with this repository and is never fabricated. Tests build
synthetic recordings with the same *shape* to exercise this code; those are test doubles, they
live in ``tmp_path``, and no number computed from them is reportable.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

#: DB2 Exercise B: 17 active hand/wrist gestures plus rest.
NUM_CLASSES = 18
#: Delsys electrodes: 8 equally spaced, 2 on the flexor/extensor spots, 2 on biceps/triceps.
NUM_CHANNELS = 12
#: Intact subjects in DB2.
NUM_SUBJECTS = 40
#: Repetitions per gesture per subject.
NUM_REPETITIONS = 6
SAMPLING_RATE_HZ = 2000
EXERCISE_NUMBER = 1  # NinaPro stores Exercise B as exercise index 1.
PROCESSED_FORMAT_VERSION = 1

#: The corrected columns. `stimulus`/`repetition` describe what the acquisition software
#: prompted, which lags what the subject did by their reaction time (D12).
LABEL_COLUMN = "restimulus"
REPETITION_COLUMN = "rerepetition"
_UNCORRECTED_COLUMNS = ("stimulus", "repetition")


class NinaProFormatError(RuntimeError):
    """A recording does not have the structure DB2 Exercise B is documented to have."""


@dataclass(frozen=True, eq=False)
class SubjectRecording:
    """One subject's continuous Exercise B recording, already label-corrected.

    ``label`` and ``repetition`` are per-sample streams aligned to ``emg``. Corrected rest samples
    can carry repetition indices 1--6 as well as 0, so ``repetition == 0`` must not be used as a
    synonym for rest or as the basis for calibration ownership.
    """

    subject: int
    emg: npt.NDArray[np.float32]
    label: npt.NDArray[np.int16]
    repetition: npt.NDArray[np.int16]

    def __post_init__(self) -> None:
        if not 1 <= self.subject <= NUM_SUBJECTS:
            raise NinaProFormatError(f"subject must lie in [1, {NUM_SUBJECTS}], got {self.subject}")
        if self.emg.ndim != 2 or self.emg.shape[1] != NUM_CHANNELS:
            raise NinaProFormatError(
                f"subject {self.subject}: emg must be (n_samples, {NUM_CHANNELS}), "
                f"got {self.emg.shape}"
            )
        n_samples = self.emg.shape[0]
        for name, stream in (("label", self.label), ("repetition", self.repetition)):
            if stream.shape != (n_samples,):
                raise NinaProFormatError(
                    f"subject {self.subject}: {name} must be ({n_samples},), got {stream.shape}"
                )
        if n_samples == 0:
            raise NinaProFormatError(f"subject {self.subject}: recording is empty")
        if not np.isfinite(self.emg).all():
            raise NinaProFormatError(f"subject {self.subject}: emg contains NaN or infinite values")

        labels = np.unique(self.label)
        if labels.min() < 0 or labels.max() >= NUM_CLASSES:
            raise NinaProFormatError(
                f"subject {self.subject}: labels must lie in [0, {NUM_CLASSES - 1}], "
                f"got [{labels.min()}, {labels.max()}]. Exercise B is 17 gestures plus rest; "
                "a wider range usually means a different exercise file was read."
            )
        repetitions = np.unique(self.repetition)
        if repetitions.min() < 0 or repetitions.max() > NUM_REPETITIONS:
            raise NinaProFormatError(
                f"subject {self.subject}: repetition indices must lie in "
                f"[0, {NUM_REPETITIONS}], got [{repetitions.min()}, {repetitions.max()}]"
            )

    @property
    def num_samples(self) -> int:
        return int(self.emg.shape[0])


def _exact_integers(values: Any, *, field: str, source: Path | str) -> npt.NDArray[np.int64]:
    """Convert to integers only if nothing is lost: no fractions, no non-finite, no int16 wrap.

    Casting first and validating afterwards read 1.9 as 1, and a value outside int16 wrapped to a
    plausible label before any range check saw it. MATLAB stores whole numbers as doubles, so
    whole-number floats are accepted.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise NinaProFormatError(f"{source}: {field} must be numeric, got {array.dtype}")
    if array.dtype.kind == "f" and not (
        np.all(np.isfinite(array)) and np.all(array == np.round(array))
    ):
        raise NinaProFormatError(f"{source}: {field} must hold whole numbers")
    limits = np.iinfo(np.int16)
    if array.size and (array.min() < limits.min or array.max() > limits.max):
        raise NinaProFormatError(
            f"{source}: {field} holds values outside int16 [{limits.min}, {limits.max}]"
        )
    return array.astype(np.int64)


def _exact_integer(values: Any, *, field: str, source: Path | str) -> int:
    array = np.asarray(values).reshape(-1)
    if array.size != 1:
        raise NinaProFormatError(f"{source}: {field} must be a scalar, got {array.size} values")
    if array.dtype.kind == "f" and not (np.isfinite(array[0]) and array[0] == np.round(array[0])):
        raise NinaProFormatError(f"{source}: {field} must be a whole number, got {array[0]!r}")
    return int(_exact_integers(array, field=field, source=source)[0])


def read_raw_mat(path: str | Path, subject: int) -> SubjectRecording:
    """Read one subject's Exercise B ``.mat`` from ``data/raw/``.

    Raises if the corrected label columns are absent. A file carrying only ``stimulus`` is not
    usable here: those labels are offset by the subject's reaction time, and the resulting
    mislabelling of movement onsets is invisible once training starts (D12).
    """
    from scipy.io import loadmat  # imported lazily: only the prepare step needs SciPy

    # Loading every variable also materialises hundreds of MiB of glove/accelerometer data that the
    # v1 Exercise-B path never consumes.
    mat: dict[str, Any] = loadmat(
        str(path),
        variable_names=[
            "subject",
            "exercise",
            "emg",
            LABEL_COLUMN,
            REPETITION_COLUMN,
            *_UNCORRECTED_COLUMNS,
        ],
    )
    corrected_missing = [
        column for column in (LABEL_COLUMN, REPETITION_COLUMN) if column not in mat
    ]
    if corrected_missing:
        present = [column for column in _UNCORRECTED_COLUMNS if column in mat]
        raise NinaProFormatError(
            f"{path}: missing corrected column(s) {corrected_missing}"
            + (
                f"; the file carries {present} instead. Those are the prompted labels, not what "
                "the subject did — see docs/experiments/README.md §3.1 (D12)."
                if present
                else ""
            )
        )

    missing = [field for field in ("emg", "subject", "exercise") if field not in mat]
    if missing:
        raise NinaProFormatError(f"{path}: missing required field(s) {missing}")

    def scalar_int(field: str) -> int:
        return _exact_integer(mat[field], field=field, source=path)

    stored_subject = scalar_int("subject")
    if stored_subject != subject:
        raise NinaProFormatError(
            f"{path}: MAT subject is {stored_subject}, but caller requested subject {subject}"
        )
    exercise = scalar_int("exercise")
    if exercise != EXERCISE_NUMBER:
        raise NinaProFormatError(
            f"{path}: expected Exercise B (stored exercise {EXERCISE_NUMBER}), got {exercise}"
        )

    emg = np.asarray(mat["emg"], dtype=np.float32)
    label: npt.NDArray[np.int16] = (
        _exact_integers(mat[LABEL_COLUMN], field=LABEL_COLUMN, source=path)
        .reshape(-1)
        .astype(np.int16)
    )
    repetition: npt.NDArray[np.int16] = (
        _exact_integers(mat[REPETITION_COLUMN], field=REPETITION_COLUMN, source=path)
        .reshape(-1)
        .astype(np.int16)
    )
    recording = SubjectRecording(subject=subject, emg=emg, label=label, repetition=repetition)

    if not np.any(recording.label == 0):
        raise NinaProFormatError(f"{path}: Exercise B recording has no rest-class samples")
    active = recording.label != 0
    if np.any(recording.repetition[active] == 0):
        raise NinaProFormatError(
            f"{path}: active Exercise-B samples must have repetition indices 1--{NUM_REPETITIONS}"
        )
    observed_pairs = set(
        np.unique(
            recording.label[active].astype(np.int64) * (NUM_REPETITIONS + 1)
            + recording.repetition[active].astype(np.int64)
        ).tolist()
    )
    expected_pairs = {
        label_value * (NUM_REPETITIONS + 1) + repetition_value
        for label_value in range(1, NUM_CLASSES)
        for repetition_value in range(1, NUM_REPETITIONS + 1)
    }
    if observed_pairs != expected_pairs:

        def decode_pair(encoded: int) -> tuple[int, int]:
            return divmod(encoded, NUM_REPETITIONS + 1)

        missing_pairs = [decode_pair(value) for value in sorted(expected_pairs - observed_pairs)]
        extra_pairs = [decode_pair(value) for value in sorted(observed_pairs - expected_pairs)]
        raise NinaProFormatError(
            f"{path}: incomplete Exercise-B gesture/repetition coverage; "
            f"missing pairs {missing_pairs}, unexpected pairs {extra_pairs}"
        )
    return recording


def save_processed(recording: SubjectRecording, path: str | Path) -> None:
    """Write one subject's processed ``.npz`` atomically.

    The archive is written to a temporary file in the same directory and moved into place in one
    step, so an interrupted write never leaves a partial archive at the final name. The temporary
    name ends in ``.npz`` because ``np.savez_compressed`` appends that suffix to any name lacking
    it, which would otherwise move the wrong file; it starts with a dot so ``available_subjects``
    never mistakes it for a subject.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(
            temporary,
            format_version=np.asarray(PROCESSED_FORMAT_VERSION, dtype=np.int16),
            subject=np.asarray(recording.subject, dtype=np.int16),
            emg=recording.emg,
            label=recording.label,
            repetition=recording.repetition,
            label_column=np.asarray(LABEL_COLUMN),
            repetition_column=np.asarray(REPETITION_COLUMN),
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_processed(path: str | Path) -> SubjectRecording:
    """Read one processed ``.npz``, validating its schema, identity, and corrected columns."""
    source = Path(path)
    try:
        return _load_processed(source)
    except NinaProFormatError:
        raise
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile, zlib.error) as error:
        raise NinaProFormatError(f"{source}: cannot read processed file: {error}") from error


def _load_processed(source: Path) -> SubjectRecording:
    with np.load(source, allow_pickle=False) as archive:
        required = {
            "format_version",
            "subject",
            "emg",
            "label",
            "repetition",
            "label_column",
            "repetition_column",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise NinaProFormatError(f"{source}: missing processed field(s) {missing}")

        format_version = _exact_integer(
            archive["format_version"], field="format_version", source=source
        )
        if format_version != PROCESSED_FORMAT_VERSION:
            raise NinaProFormatError(
                f"{source}: processed format version {format_version}, expected "
                f"{PROCESSED_FORMAT_VERSION}; re-run scripts/prepare_dataset.py"
            )

        stored_label_column = str(archive["label_column"])
        stored_repetition_column = str(archive["repetition_column"])
        if stored_label_column != LABEL_COLUMN or stored_repetition_column != REPETITION_COLUMN:
            raise NinaProFormatError(
                f"{source}: built from label/repetition columns "
                f"{stored_label_column!r}/{stored_repetition_column!r}, not "
                f"{LABEL_COLUMN!r}/{REPETITION_COLUMN!r}. "
                "Re-run scripts/prepare_dataset.py (D12)."
            )
        stored_subject = _exact_integer(archive["subject"], field="subject", source=source)
        filename_subject = source.stem.removeprefix("subject_")
        if filename_subject.isdigit() and int(filename_subject) != stored_subject:
            raise NinaProFormatError(
                f"{source}: filename identifies subject {int(filename_subject)}, but archive "
                f"stores subject {stored_subject}"
            )

        return SubjectRecording(
            subject=stored_subject,
            emg=np.asarray(archive["emg"], dtype=np.float32),
            label=_exact_integers(archive["label"], field="label", source=source).astype(np.int16),
            repetition=_exact_integers(
                archive["repetition"], field="repetition", source=source
            ).astype(np.int16),
        )


def processed_path(data_dir: str | Path, subject: int) -> Path:
    """Where subject ``n``'s processed recording lives under ``data_dir``."""
    return Path(data_dir) / f"subject_{subject:02d}.npz"


def available_subjects(data_dir: str | Path) -> list[int]:
    """Subject IDs with a processed recording present, ascending."""
    directory = Path(data_dir)
    if not directory.is_dir():
        return []
    subjects = set()
    for candidate in directory.glob("subject_*.npz"):
        stem = candidate.stem.removeprefix("subject_")
        if stem.isdigit() and 1 <= int(stem) <= NUM_SUBJECTS:
            subjects.add(int(stem))
    return sorted(subjects)
