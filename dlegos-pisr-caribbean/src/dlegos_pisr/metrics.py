"""Verification metrics for continuous fields and thresholded precipitation."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    if prediction.numel() == 0:
        raise ValueError("Inputs cannot be empty.")


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_pair(prediction, target)
    return (prediction - target).abs().mean()


def rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    _check_pair(prediction, target)
    return torch.sqrt(torch.mean((prediction - target) ** 2))


def critical_success_index(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """CSI for an event threshold; returns 1 when neither field has an event."""
    _check_pair(prediction, target)
    forecast_event = prediction >= threshold
    observed_event = target >= threshold
    hits = (forecast_event & observed_event).sum(dtype=torch.float64)
    misses = (~forecast_event & observed_event).sum(dtype=torch.float64)
    false_alarms = (forecast_event & ~observed_event).sum(dtype=torch.float64)
    denominator = hits + misses + false_alarms
    return torch.where(denominator > 0, hits / denominator, denominator.new_tensor(1.0)).to(prediction.dtype)


def fractions_skill_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
    window: int = 5,
) -> torch.Tensor:
    """Neighborhood FSS for 2-D fields or tensors ending in (lat, lon).

    The score is 1 for a perfect match. Leading dimensions are treated as
    independent fields and averaged.
    """
    _check_pair(prediction, target)
    if window < 1:
        raise ValueError("window must be at least 1.")
    if prediction.ndim < 2:
        raise ValueError("Inputs must end in latitude and longitude dimensions.")
    shape = prediction.shape
    pred_event = (prediction >= threshold).to(prediction.dtype).reshape(-1, 1, *shape[-2:])
    obs_event = (target >= threshold).to(target.dtype).reshape(-1, 1, *shape[-2:])
    pred_frac = F.avg_pool2d(pred_event, window, stride=1, padding=window // 2)
    obs_frac = F.avg_pool2d(obs_event, window, stride=1, padding=window // 2)
    numerator = torch.mean((pred_frac - obs_frac) ** 2)
    denominator = torch.mean(pred_frac**2 + obs_frac**2)
    return torch.where(denominator > 0, 1 - numerator / denominator, denominator.new_tensor(1.0))


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
