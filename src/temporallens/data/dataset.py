"""A torch dataset over a :class:`~temporallens.data.windows.WindowIndex`.

Windows are materialised **per item**, never per fold. At F1's stride of 100 a subject yields
about 17,300 windows, so a materialised 28-subject fold is about 9.3 GB resident and twice that at
peak, while the signals themselves are only ~87 MB per subject — stride-100 windows overlap by 75%
and store every sample four times over. Holding the recordings and cutting windows on demand
keeps the resident cost at the signals plus one batch.

Normalization is applied here rather than in the training loop, so there is exactly one place
where a window can reach a model un-normalized.

``__getitems__`` lets the DataLoader fetch a whole batch in one call, so the windows are cut with
one vectorized gather per subject instead of one Python call each. It returns exactly what
per-item fetching would (the normalization is elementwise, so batching does not change a value).
Measured on subjects 1–3 at batch 64, loading fell from 20 to 11 µs per window, roughly 9.7 s to
5.3 s for a 484,000-window F1 fold epoch; training compute, not loading, is the larger cost.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from temporallens.data.ninapro import SubjectRecording
from temporallens.data.windows import WindowIndex, materialise
from temporallens.preprocessing.normalize import ChannelStats, apply_channel_stats


class WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Yields ``(window, label, subject)`` for each entry of a window index.

    ``subject`` travels with every item because every reported metric is also computed
    per-subject (D26), and because a prediction has to be traceable back to the person it came
    from for the paired comparison to be possible at all.
    """

    def __init__(
        self,
        index: WindowIndex,
        recordings: Mapping[int, SubjectRecording],
        *,
        normalization: ChannelStats,
    ) -> None:
        missing = sorted(set(index.subject.tolist()) - set(recordings))
        if missing:
            raise KeyError(f"no recording supplied for subject(s) {missing}")
        self.index = index
        self.recordings = dict(recordings)
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.index)

    def __getitems__(self, positions: list[int]) -> list[tuple[Tensor, Tensor, Tensor]]:
        """The items at ``positions``, materialised and normalized as one batch."""
        batch = self.index.take(np.asarray(positions, dtype=np.int64))
        windows = torch.from_numpy(
            apply_channel_stats(self.normalization, materialise(batch, self.recordings))
        )
        labels = torch.from_numpy(batch.label.astype(np.int64))
        subjects = torch.from_numpy(batch.subject.astype(np.int64))
        return list(zip(windows, labels, subjects, strict=True))

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor, Tensor]:
        one = self.index.take(np.asarray([position], dtype=np.int64))
        window = apply_channel_stats(self.normalization, materialise(one, self.recordings))
        return (
            torch.from_numpy(window[0]),
            torch.tensor(int(one.label[0]), dtype=torch.long),
            torch.tensor(int(one.subject[0]), dtype=torch.long),
        )
