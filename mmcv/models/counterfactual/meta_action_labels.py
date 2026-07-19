"""Meta-action labels and simple pseudo-label extraction.

The labels intentionally mirror MindDrive's language-action tokens. During
training these helpers derive factual labels from logged futures; no
counterfactual ground-truth labels are assumed.
"""

import torch

SPEED_META_ACTIONS = (
    "maintain moderate speed",
    "stop",
    "maintain slow speed",
    "speed up",
    "slow down",
    "maintain fast speed",
    "slow down rapidly",
)

PATH_META_ACTIONS = (
    "lanefollow",
    "straight",
    "turn left",
    "change lane left",
    "turn right",
    "change lane right",
)


def default_meta_actions(device=None):
    """Return all aligned speed/path ego meta-action candidates."""
    actions = []
    for speed_idx, speed in enumerate(SPEED_META_ACTIONS):
        for path_idx, path in enumerate(PATH_META_ACTIONS):
            actions.append(
                dict(
                    speed=speed,
                    path=path,
                    speed_idx=speed_idx,
                    path_idx=path_idx,
                )
            )
    return actions


def speed_pseudo_labels(futures, dt=0.5):
    """Classify future displacement sequences into 7 speed labels.

    Args:
        futures (Tensor): Agent future positions or deltas, shape (..., T, 2).
        dt (float): Seconds between future waypoints.

    Returns:
        Tensor: Integer labels with shape futures.shape[:-2].
    """
    if futures.numel() == 0:
        return torch.zeros(futures.shape[:-2], dtype=torch.long, device=futures.device)
    steps = futures.diff(dim=-2, prepend=torch.zeros_like(futures[..., :1, :]))
    speeds = torch.linalg.norm(steps, dim=-1) / max(dt, 1e-3)
    first = speeds[..., 0]
    mean = speeds.mean(dim=-1)
    last = speeds[..., -1]
    accel = last - first

    labels = torch.zeros_like(mean, dtype=torch.long)
    labels = torch.where(mean < 0.25, torch.ones_like(labels) * 1, labels)
    labels = torch.where((mean >= 0.25) & (mean < 2.0), torch.ones_like(labels) * 2, labels)
    labels = torch.where(mean >= 8.0, torch.ones_like(labels) * 5, labels)
    labels = torch.where(accel > 1.2, torch.ones_like(labels) * 3, labels)
    labels = torch.where(accel < -1.2, torch.ones_like(labels) * 4, labels)
    labels = torch.where(accel < -3.0, torch.ones_like(labels) * 6, labels)
    return labels


def path_pseudo_labels(futures):
    """Classify future displacement sequences into 6 path labels.

    Args:
        futures (Tensor): Agent future positions or deltas, shape (..., T, 2).

    Returns:
        Tensor: Integer labels with shape futures.shape[:-2].
    """
    if futures.numel() == 0:
        return torch.zeros(futures.shape[:-2], dtype=torch.long, device=futures.device)
    disp = futures[..., -1, :] - futures[..., 0, :]
    # MindDrive ego/agent futures are represented as (lateral, longitudinal)
    # offsets in this planning stack.
    lat = disp[..., 0]
    lon = disp[..., 1].abs()
    labels = torch.ones_like(lat, dtype=torch.long)  # straight
    labels = torch.where(lat > 3.5, torch.ones_like(labels) * 3, labels)
    labels = torch.where(lat < -3.5, torch.ones_like(labels) * 5, labels)
    labels = torch.where((lat > 1.0) & (lat <= 3.5) & (lon < 8.0), torch.ones_like(labels) * 2, labels)
    labels = torch.where((lat < -1.0) & (lat >= -3.5) & (lon < 8.0), torch.ones_like(labels) * 4, labels)
    labels = torch.where((lat.abs() < 0.75) & (lon < 1.0), torch.zeros_like(labels), labels)
    return labels
