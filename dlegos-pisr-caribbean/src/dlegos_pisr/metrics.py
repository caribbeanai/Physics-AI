"""Verification metrics for continuous fields and thresholded precipitation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if prediction.numel() == 0:
        raise ValueError("Inputs cannot be empty.")


def _score_dtype(prediction: torch.Tensor, target: torch.Tensor) -> torch.dtype:
    # Integer and reduced-precision input must still produce fractional scores.
    return torch.promote_types(torch.float32, torch.promote_types(prediction.dtype, target.dtype))


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_pair(prediction, target)
    dtype = _score_dtype(prediction, target)
    return (prediction.to(dtype) - target.to(dtype)).abs().mean()


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_pair(prediction, target)
    dtype = _score_dtype(prediction, target)
    return torch.sqrt(torch.mean((prediction.to(dtype) - target.to(dtype)) ** 2))


def critical_success_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """CSI for an event threshold; returns 1 when neither field has an event."""
    _check_pair(prediction, target)
    forecast_event = prediction >= threshold
    observed_event = target >= threshold
    hits = (forecast_event & observed_event).sum()
    union = (forecast_event | observed_event).sum()
    score = hits.to(_score_dtype(prediction, target)) / union.clamp_min(1)
    return torch.where(union > 0, score, score.new_tensor(1.0))


def fractions_skill_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    window: int = 5,
) -> torch.Tensor:
    """Neighborhood FSS for 2-D fields or tensors ending in (lat, lon).

    Use a positive odd window to centre a square neighbourhood on each grid
    cell. Outside-grid cells count as non-events via zero padding. Leading
    dimensions are independent fields; sum squared differences and reference
    terms over all fields before taking their ratio. Return 1 for a perfect
    match or when neither field has an event.
    """
    _check_pair(prediction, target)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer.")
    if prediction.ndim < 2:
        raise ValueError("Inputs must end in latitude and longitude dimensions.")
    shape = prediction.shape
    dtype = _score_dtype(prediction, target)
    pred_event = (prediction >= threshold).to(dtype).reshape(-1, 1, *shape[-2:])
    obs_event = (target >= threshold).to(dtype).reshape(-1, 1, *shape[-2:])
    pred_frac = F.avg_pool2d(pred_event, window, stride=1, padding=window // 2)
    obs_frac = F.avg_pool2d(obs_event, window, stride=1, padding=window // 2)
    numerator = torch.mean((pred_frac - obs_frac) ** 2)
    denominator = torch.mean(pred_frac**2 + obs_frac**2)
    safe_denominator = torch.where(denominator > 0, denominator, denominator.new_tensor(1.0))
    return 1 - numerator / safe_denominator


def precipitation_scores(
    prediction: torch.Tensor,
    target: torch.Tensor,
    thresholds: tuple[float, ...] = (1.0, 5.0, 16.0),
    fss_window: int = 5,
) -> dict[str, torch.Tensor]:
    """Return CSI and FSS per threshold (precipitation units are caller-defined)."""
    scores: dict[str, torch.Tensor] = {}
    for threshold in thresholds:
        key = f"{threshold:g}"
        scores[f"csi_{key}"] = critical_success_index(prediction, target, threshold)
        scores[f"fss_{key}"] = fractions_skill_score(prediction, target, threshold, fss_window)
    return scores
