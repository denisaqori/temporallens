"""Index a continuous recording into labelled windows, and materialise them on demand.

Two invariants, for two different reasons.

**Windows never cross a segment boundary.** A segment is a maximal run of samples sharing the
same ``(label, repetition)``. Cutting across one produces a window containing part of a rest
period and part of a gesture, carrying whichever single label the code happened to assign — the
same silent mislabelling D12 exists to prevent, one layer down, concentrated at movement onset.
The cost is dropping the tail of each segment where fewer than ``window_size`` samples remain.

**Indexing is separate from materialising.** At F1's stride of 100 a subject yields roughly
15,700 windows; materialising them is ~300 MB *per subject*, so a 28-subject fold would need
about 8 GB resident and twice that at peak. The signal itself is only ~78 MB per subject, because
stride-100 windows overlap by 75% and store every sample four times over. So the index carries
only where each window starts and what it is, and callers materialise a batch at a time.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from temporallens.data.ninapro import NUM_REPETITIONS, SubjectRecording


@dataclass(frozen=True, eq=False)
class WindowIndex:
    """Where each window starts and what it is, without the signal.

    ``eq=False`` because the fields are arrays: the generated ``__eq__`` would raise
    "truth value of an array is ambiguous", and ``__hash__`` would raise on an unhashable field.
    """

    subject: npt.NDArray[np.int16]
    start: npt.NDArray[np.int64]
    label: npt.NDArray[np.int64]
    repetition: npt.NDArray[np.int16]
    window_size: int

    def __len__(self) -> int:
        return int(self.start.size)

    def take(self, selector: npt.NDArray[np.int64] | npt.NDArray[np.bool_]) -> WindowIndex:
        """A sub-index — one batch, one subject, one fold."""
        return replace(
            self,
            subject=self.subject[selector],
            start=self.start[selector],
            label=self.label[selector],
            repetition=self.repetition[selector],
        )

    def for_subjects(self, subjects: Sequence[int]) -> WindowIndex:
        """The windows belonging to ``subjects``, in their existing order."""
        return self.take(np.isin(self.subject, np.asarray(list(subjects), dtype=np.int16)))


def segment_spans(
    label: npt.NDArray[np.int16], repetition: npt.NDArray[np.int16]
) -> list[tuple[int, int]]:
    """Half-open ``(start, stop)`` spans over which ``(label, repetition)`` is constant."""
    if label.size == 0:
        return []
    key = label.astype(np.int64) * (NUM_REPETITIONS + 1) + repetition.astype(np.int64)
    boundaries = np.flatnonzero(np.diff(key)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [key.size]))
    return [(int(a), int(b)) for a, b in zip(starts, stops, strict=True)]


def index_recording(
    recording: SubjectRecording,
    *,
    window_size: int,
    stride: int,
    max_windows: int | None = None,
    seed: int = 42,
) -> WindowIndex:
    """Index ``recording`` into windows, skipping any that would span two segments.

    ``max_windows`` takes a seeded, class-covering subsample rather than truncating. Truncation
    returns the head of the recording, which is rest plus whichever gesture came first; a purely
    uniform 50-window draw can also omit several DB2 classes. Debug caps are large enough to retain
    at least one window from every class present, then fill their remaining budget randomly.
    """
    if window_size <= 0 or stride <= 0:
        raise ValueError(f"window_size and stride must be positive, got {window_size}/{stride}")
    if max_windows is not None and max_windows <= 0:
        raise ValueError(f"max_windows must be positive when provided, got {max_windows}")

    starts: list[int] = []
    for span_start, span_stop in segment_spans(recording.label, recording.repetition):
        last_start = span_stop - window_size
        starts.extend(range(span_start, last_start + 1, stride))

    if not starts:
        raise ValueError(
            f"subject {recording.subject}: no segment is at least window_size={window_size} "
            f"samples long, so every window would cross a boundary"
        )

    start = np.asarray(starts, dtype=np.int64)
    if max_windows is not None and start.size > max_windows:
        rng = np.random.default_rng(seed + recording.subject)
        window_labels = recording.label[start]
        classes = np.unique(window_labels)
        if max_windows < classes.size:
            raise ValueError(
                f"max_windows={max_windows} cannot retain all {classes.size} classes present"
            )

        required_positions = np.asarray(
            [rng.choice(np.flatnonzero(window_labels == label)) for label in classes],
            dtype=np.int64,
        )
        remaining_positions = np.setdiff1d(
            np.arange(start.size, dtype=np.int64), required_positions, assume_unique=True
        )
        extra_count = max_windows - required_positions.size
        extra_positions = rng.choice(remaining_positions, size=extra_count, replace=False)
        selected_positions = np.sort(np.concatenate((required_positions, extra_positions)))
        start = start[selected_positions]

    return WindowIndex(
        subject=np.full(start.size, recording.subject, dtype=np.int16),
        start=start,
        label=recording.label[start].astype(np.int64),
        repetition=recording.repetition[start].astype(np.int16),
        window_size=window_size,
    )


def materialise(
    index: WindowIndex, recordings: Mapping[int, SubjectRecording]
) -> npt.NDArray[np.float32]:
    """Build the ``(n_windows, channels, window_size)`` array this index refers to.

    Channels-first, because that is what ``torch.nn.Conv1d`` consumes. Pass a *batch-sized*
    index: materialising a whole fold at F1's stride would need gigabytes.
    """
    if len(index) == 0:
        raise ValueError("cannot materialise an empty index")
    missing = sorted(set(index.subject.tolist()) - set(recordings))
    if missing:
        raise KeyError(f"no recording supplied for subject(s) {missing}")

    offsets = np.arange(index.window_size, dtype=np.int64)
    out = np.empty(
        (len(index), recordings[int(index.subject[0])].emg.shape[1], index.window_size),
        dtype=np.float32,
    )
    for subject in np.unique(index.subject):
        rows = np.flatnonzero(index.subject == subject)
        emg = recordings[int(subject)].emg
        sample_index = index.start[rows][:, None] + offsets[None, :]
        # (batch, window, channels) -> channels-first, written straight into the output.
        out[rows] = emg[sample_index].transpose(0, 2, 1)
    return out


def concatenate(indices: Sequence[WindowIndex]) -> WindowIndex:
    """Join per-subject indices into one, preserving provenance."""
    if not indices:
        raise ValueError("cannot concatenate an empty list of window indices")
    window_sizes = {index.window_size for index in indices}
    if len(window_sizes) != 1:
        raise ValueError(f"cannot concatenate indices with differing window sizes: {window_sizes}")
    return WindowIndex(
        subject=np.concatenate([index.subject for index in indices]),
        start=np.concatenate([index.start for index in indices]),
        label=np.concatenate([index.label for index in indices]),
        repetition=np.concatenate([index.repetition for index in indices]),
        window_size=window_sizes.pop(),
    )
