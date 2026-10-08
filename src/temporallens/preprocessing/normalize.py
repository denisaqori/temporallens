"""Per-channel normalization statistics, fitted on training data only.

§3.3 calls normalization the most common silent leak in the project: statistics computed over the
full dataset before splitting pull held-out distribution into training and inflate every
downstream number, leaving no trace in the logs. So the fitting set is always an explicit
argument — there is no code path here that can reach a validation or test subject.

**What the statistics are fitted over.** The samples that training windows actually cover, each
counted once. Measured on subject 2, the three candidates agree on the per-channel standard
deviation to within 1.4%, so this is not an accuracy choice — it is a choice that has to be
*frozen*, because the statistics are baked into the shared encoder checkpoint and switching
schemes later would move it:

===========================  ======================  ==========================
candidate                    std vs covered-once     depends on stride?
===========================  ======================  ==========================
whole recording              0.216%                  no
window-weighted              1.432%                  yes (0.230% per stride)
covered once  (this one)     --                      no
===========================  ======================  ==========================

Covered-once is used for two reasons. Window-weighted statistics are a function of the stride,
and these are reused by consumers that may window differently. And the samples the whole
recording adds are ones nothing else in the pipeline touches: under D24 a window must fit inside
its segment on a stride grid anchored at the segment start, so the last ``(L - window) mod
stride`` samples of every segment — 0.54% of the signal, bounded by ``stride - 1`` — host no
window and are never trained or scored on. They are measurably quieter (per-channel std 0.445 to
0.928 of the covered samples, being segment ends where a contraction is decaying), so including
them would make the statistics describe a distribution the model never sees.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from temporallens.data.ninapro import SubjectRecording
from temporallens.data.windows import WindowIndex

#: Absolute floor on a per-channel standard deviation. An EMG channel can fail and read flat;
#: dividing by its zero variance would yield non-finite inputs with no error raised.
STD_FLOOR = 1e-8


@dataclass(frozen=True, eq=False)
class ChannelStats:
    """Per-channel mean and standard deviation, plus the provenance of the fitting set.

    ``eq=False`` because the fields are arrays — the generated ``__eq__`` would raise on an
    ambiguous array truth value.
    """

    mean: npt.NDArray[np.float32]
    std: npt.NDArray[np.float32]
    fitted_on: str
    num_samples: int
    #: Channels whose variance was floored. A dead channel should be visible, not silent.
    floored_channels: tuple[int, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable payload, for embedding in a checkpoint's ``model_config``."""
        return {
            "mean": [float(v) for v in self.mean],
            "std": [float(v) for v in self.std],
            "fitted_on": self.fitted_on,
            "num_samples": int(self.num_samples),
            "floored_channels": [int(c) for c in self.floored_channels],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ChannelStats:
        return cls(
            mean=np.asarray(payload["mean"], dtype=np.float32),
            std=np.asarray(payload["std"], dtype=np.float32),
            fitted_on=payload["fitted_on"],
            num_samples=int(payload["num_samples"]),
            floored_channels=tuple(payload.get("floored_channels", ())),
        )


def _covered_mask(
    num_samples: int, starts: npt.NDArray[np.int64], window: int
) -> npt.NDArray[np.bool_]:
    """Samples touched by at least one window, via a difference array rather than a loop."""
    delta = np.zeros(num_samples + 1, dtype=np.int32)
    np.add.at(delta, starts, 1)
    np.add.at(delta, starts + window, -1)
    return np.cumsum(delta[:-1]) > 0


def fit_channel_stats(
    recordings: Mapping[int, SubjectRecording],
    index: WindowIndex,
    *,
    fitted_on: str,
) -> ChannelStats:
    """Fit per-channel statistics over the samples ``index`` covers, each counted once.

    ``recordings`` and ``index`` together *are* the fitting set. Pass only the training
    partition: there is no mode flag that widens it.
    """
    missing = sorted(set(index.subject.tolist()) - set(recordings))
    if missing:
        raise KeyError(f"no recording supplied for subject(s) {missing}")
    if len(index) == 0:
        raise ValueError("cannot fit statistics on an empty window index")

    subjects = np.unique(index.subject)
    masks: dict[int, npt.NDArray[np.bool_]] = {}
    for subject in subjects:
        recording = recordings[int(subject)]
        starts = index.start[index.subject == subject]
        masks[int(subject)] = _covered_mask(recording.num_samples, starts, index.window_size)

    channels = recordings[int(subjects[0])].emg.shape[1]

    # Two passes. The identity E[x^2] - E[x]^2 is fine in float64 at these magnitudes, but a
    # mean-centred second pass is exact and the data is already resident.
    total = np.zeros(channels, dtype=np.float64)
    count = 0
    for subject, mask in masks.items():
        total += recordings[subject].emg[mask].sum(axis=0, dtype=np.float64)
        count += int(mask.sum())
    mean = total / count

    squared = np.zeros(channels, dtype=np.float64)
    for subject, mask in masks.items():
        deviation = recordings[subject].emg[mask].astype(np.float64) - mean
        squared += np.square(deviation).sum(axis=0)
    std = np.sqrt(squared / count)

    floored = tuple(int(c) for c in np.flatnonzero(std < STD_FLOOR))
    return ChannelStats(
        mean=mean.astype(np.float32),
        std=np.maximum(std, STD_FLOOR).astype(np.float32),
        fitted_on=fitted_on,
        num_samples=count,
        floored_channels=floored,
    )


def apply_channel_stats(
    stats: ChannelStats, data: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """Standardise channels-first windows ``(n, channels, window)`` with fitted statistics."""
    if data.ndim != 3 or data.shape[1] != stats.mean.size:
        raise ValueError(f"expected (n, {stats.mean.size}, window), got {data.shape}")
    centred = data - stats.mean[None, :, None]
    return np.asarray(centred / stats.std[None, :, None], dtype=np.float32)
