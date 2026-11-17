"""Coordinate-safe target pooling shared by Task2 experimental models."""

import torch


def build_inserted_target_mask(hidden_token, target):
    """Return target-character mask after inserting the two marker tokens.

    Dataset target coordinates refer to BERT positions before insertion.
    After inserting the left marker, target characters occupy
    [target_start + 1, target_end + 1] inclusively.
    """
    if target is None:
        raise ValueError('target coordinates are required by this Task2 model')
    coordinates = torch.as_tensor(target, dtype=torch.long, device=hidden_token.device)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(f'expected target shape [batch, 2], got {tuple(coordinates.shape)}')
    if coordinates.shape[0] != hidden_token.shape[0]:
        raise ValueError(
            f'target batch={coordinates.shape[0]} does not match hidden batch={hidden_token.shape[0]}'
        )

    positions = torch.arange(hidden_token.shape[1], device=hidden_token.device).unsqueeze(0)
    target_start = coordinates[:, 0].unsqueeze(1) + 1
    target_end = coordinates[:, 1].unsqueeze(1) + 1
    mask = (positions >= target_start) & (positions <= target_end)
    if not bool(mask.any(dim=1).all()):
        raise ValueError('at least one target span is empty or outside the encoded sequence')
    return mask


def mean_pool_inserted_target(hidden_token, target):
    mask = build_inserted_target_mask(hidden_token, target)
    weights = mask.to(hidden_token.dtype).unsqueeze(-1)
    return (hidden_token * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

