"""Builders for synthetic recordings shaped like NinaPro DB2 Exercise B.

These are test doubles, not data. They exist so the pipeline can be exercised without the real
dataset, they are never written outside ``tmp_path``, and no number computed from them is
reportable. The real data is downloaded under its own terms and is never fabricated.
"""

from __future__ import annotations

import numpy as np

from temporallens.data.ninapro import NUM_CHANNELS, SubjectRecording


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
