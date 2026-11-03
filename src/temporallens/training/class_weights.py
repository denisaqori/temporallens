"""Class weights for the imbalanced-loss rule (D11).

Rest is about half of all windows — measured at 50.2% on subject 2 at stride 100, roughly 17.5x
the median gesture class — so a model predicting only rest scores 50% accuracy. D11 handles that
**in the loss and never by resampling**: duplicating windows that already overlap by 75% would
put near-identical copies in a batch, which is the window-correlation hazard §3.2 and §3.3 warn
about reintroduced through the training sampler.

Weights are inverse class frequency over the **training split only**, normalised to mean 1 so the
loss scale does not change with how skewed a particular split happens to be. A class absent from
the training split gets weight 0 rather than infinity (D11): it has no frequency to invert, and 0
keeps the vector indexable by label without inventing a target the split cannot teach.

That is a training-side rule only. Evaluation is separate: the metrics always score all 18 classes,
and a class missing from training is scored like any other if it appears in the evaluation windows.

**The weight digest (D76/D85).** Language fits recompute their weights from their own training
windows and verify them against the F1 parent's ``class_weights_sha256``, so producer and consumer
must hash the same bytes. :func:`class_weights_sha256` fixes those bytes: the actual float32 vector
in class order, little-endian and contiguous, behind a header naming the encoding, dtype and shape,
following ``data.assignments``' recording digest. Equal weights do not prove equal fitting pools,
so the source windows' identity is recorded separately.
"""

from __future__ import annotations

import hashlib

import numpy as np
import numpy.typing as npt

#: D85's versioned encoding name, written beside every digest so a future change is visible.
CLASS_WEIGHTS_DIGEST_ENCODING = "d11_class_weights_v1_shape_dtype_little_endian_float32"


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


def class_weights_sha256(weights: npt.ArrayLike) -> str:
    """SHA-256 of a class-weight vector under :data:`CLASS_WEIGHTS_DIGEST_ENCODING` (D85).

    The vector must already be float32, as :func:`inverse_frequency_weights` returns it: casting a
    float64 vector here would hash a value no trainer used.
    """
    array = np.asarray(weights)
    if array.dtype != np.float32 or array.ndim != 1 or array.size == 0:
        raise ValueError(
            f"class weights must be a non-empty 1-D float32 vector, got {array.dtype} "
            f"with shape {list(array.shape)}"
        )
    if not np.all(np.isfinite(array)) or array.min() < 0:
        raise ValueError("class weights must be finite and non-negative")
    encoded = np.ascontiguousarray(array, dtype="<f4")
    digest = hashlib.sha256()
    digest.update(
        f"{CLASS_WEIGHTS_DIGEST_ENCODING}\nclass_weights:<f4:{list(encoded.shape)}\n".encode(
            "ascii"
        )
    )
    digest.update(encoded.tobytes())
    return digest.hexdigest()
