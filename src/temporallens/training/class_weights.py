"""Class weights for the imbalanced-loss rule (D11).

Rest is about half of all windows — measured at 50.2% on subject 2 at stride 100, roughly 17.5x
the median gesture class — so a model predicting only rest scores 50% accuracy. D11 handles that
**in the loss and never by resampling**: duplicating windows that already overlap by 75% would
put near-identical copies in a batch, which is the window-correlation hazard §3.2 and §3.3 warn
about reintroduced through the training sampler.

Weights are inverse class frequency over the **training split only**, normalised to mean 1 so the
loss scale does not change with how skewed a particular split happens to be. A class absent from
the split gets weight 0 rather than infinity: it contributes no gradient, which is the honest
behaviour, and D21 already fixes how an absent class is scored.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def inverse_frequency_weights(
    labels: npt.NDArray[np.int64], *, num_classes: int
) -> npt.NDArray[np.float32]:
    """Inverse-frequency class weights over ``labels``, mean-normalised, 0 for absent classes."""
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(
        np.float64
    )
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    if present.any():
        weights[present] /= weights[present].mean()
    return weights.astype(np.float32)
