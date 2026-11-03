"""Builders for synthetic recordings shaped like NinaPro DB2 Exercise B.

These are test doubles, not data. They exist so the pipeline can be exercised without the real
dataset, they are never written outside ``tmp_path``, and no number computed from them is
reportable. The real data is downloaded under its own terms and is never fabricated.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from temporallens.data.ninapro import NUM_CHANNELS, SubjectRecording, processed_path, save_processed
from temporallens.data.splits import SplitManifest


def recording(
    subject: int = 1,
    *,
    gestures: int = 3,
    repetitions: int = 2,
    samples_per_segment: int = 1000,
    channel_offset: float = 0.0,
    channel_scale: float = 1.0,
    seed: int | None = None,
) -> SubjectRecording:
    """A rest/gesture/rest/gesture recording with DB2's segment structure.

    With ``seed=None`` each channel carries its segment's label as a constant, so a window's
    content is checkable against the label it was given. With a seed, channels carry noise scaled
    per channel, which is what normalization statistics need to have something to measure.
    """
    labels: list[int] = []
    reps: list[int] = []
    for repetition in range(1, repetitions + 1):
        for gesture in range(1, gestures + 1):
            labels += [0] * samples_per_segment + [gesture] * samples_per_segment
            reps += [0] * samples_per_segment + [repetition] * samples_per_segment

    label = np.asarray(labels, dtype=np.int16)
    repetition = np.asarray(reps, dtype=np.int16)

    if seed is None:
        emg = np.repeat(label.astype(np.float32)[:, None], NUM_CHANNELS, axis=1)
    else:
        rng = np.random.default_rng(seed)
        # Per-channel scale and offset, so per-channel statistics are distinguishable.
        scale = channel_scale * (1.0 + np.arange(NUM_CHANNELS, dtype=np.float32))
        emg = rng.normal(0.0, 1.0, (label.size, NUM_CHANNELS)).astype(np.float32)
        emg = emg * scale + channel_offset
    return SubjectRecording(subject=subject, emg=emg, label=label, repetition=repetition)


def recordings(subjects: tuple[int, ...], **kwargs: object) -> dict[int, SubjectRecording]:
    """One recording per subject, each with its own noise draw."""
    out = {}
    for s in subjects:
        out[s] = recording(s, seed=1000 + s, **kwargs)  # type: ignore[arg-type]
    return out


def write_development_recordings(directory: Path) -> Path:
    """Processed-format doubles for the manifest's 32 development subjects, under ``directory``.

    One rest and one gesture segment of 600 samples each, so the uncapped 400/100 index holds six
    windows per subject: enough to exercise F2's builder end to end, and quick to write.
    """
    for subject in SplitManifest.load().training_subjects:
        double = recording(
            subject, gestures=1, repetitions=1, samples_per_segment=600, seed=1000 + subject
        )
        save_processed(double, processed_path(directory, subject))
    return directory


def db2_like_recording(
    subject: int,
    *,
    gestures: int = 3,
    repetitions: int = 2,
    samples_per_segment: int = 600,
    terminal_rest: int = 600,
    separation: float = 1.5,
    seed: int | None = None,
) -> SubjectRecording:
    """A recording with DB2's repetition structure and a learnable, imperfect class signal.

    As in DB2's ``rerepetition``, each rest belongs to the repetition it follows, and the recording
    ends with one repetition-0 rest segment (D81). Each class shifts its own channel by
    ``separation``, except that the last two gestures share a channel, so no model can separate
    them: that keeps every fitted temperature interior instead of running to D39's bound.
    """
    labels: list[int] = []
    reps: list[int] = []
    for repetition in range(1, repetitions + 1):
        for gesture in range(1, gestures + 1):
            labels += [gesture] * samples_per_segment + [0] * samples_per_segment
            reps += [repetition] * (2 * samples_per_segment)
    labels += [0] * terminal_rest
    reps += [0] * terminal_rest
    label = np.asarray(labels, dtype=np.int16)
    repetition = np.asarray(reps, dtype=np.int16)

    rng = np.random.default_rng(1000 + subject if seed is None else seed)
    emg = rng.normal(0.0, 1.0, (label.size, NUM_CHANNELS)).astype(np.float32)
    channel = np.minimum(label.astype(np.int64), gestures - 1)
    emg[np.arange(label.size), channel] += np.float32(separation)
    return SubjectRecording(subject=subject, emg=emg, label=label, repetition=repetition)


def write_db2_like_development_set(directory: Path, **kwargs: object) -> Path:
    """DB2-shaped doubles for the manifest's 32 development subjects only, never the test eight."""
    for subject in SplitManifest.load().training_subjects:
        save_processed(
            db2_like_recording(subject, **kwargs),  # type: ignore[arg-type]
            processed_path(directory, subject),
        )
    return directory
