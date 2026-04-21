from pathlib import Path
from typing import Any

import torch


def _reshape_noise_level_for_sample(
    sigma: float | torch.Tensor,
    sample: torch.Tensor,
    calc_dtype: torch.dtype,
) -> float | torch.Tensor:
    """Broadcast sigma/timesteps to match a latent tensor shape.

    Common cases in this codebase are:
    - scalar sigma
    - per-batch sigma with shape ``(B,)``
    - per-token timesteps with shape ``(B, T)``

    The diffusion math expects the noise level to multiply tensors shaped like
    ``(B, T, D)``, so we expand trailing singleton dimensions as needed.
    """
    if not isinstance(sigma, torch.Tensor):
        return sigma

    sigma = sigma.to(calc_dtype)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square (RMS) normalize `x` over its last dimension.
    Thin wrapper around `torch.nn.functional.rms_norm` that infers the normalized
    shape and forwards `weight` and `eps`.
    """
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps)


def check_config_value(config: dict, key: str, expected: Any) -> None:  # noqa: ANN401
    actual = config.get(key)
    if actual != expected:
        raise ValueError(f"Config value {key} is {actual}, expected {expected}")


def to_velocity(
    sample: torch.Tensor,
    sigma: float | torch.Tensor,
    denoised_sample: torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoised version to velocity.
    Returns:
        Velocity
    """
    sigma = _reshape_noise_level_for_sample(sigma, sample, calc_dtype)
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() == 1:
            sigma = sigma.item()
        elif torch.any(sigma == 0):
            raise ValueError("Sigma can't be 0.0")
    elif sigma == 0:
        raise ValueError("Sigma can't be 0.0")
    return ((sample.to(calc_dtype) - denoised_sample.to(calc_dtype)) / sigma).to(sample.dtype)


def to_denoised(
    sample: torch.Tensor,
    velocity: torch.Tensor,
    sigma: float | torch.Tensor,
    calc_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert the sample and its denoising velocity to denoised sample.
    Returns:
        Denoised sample
    """
    sigma = _reshape_noise_level_for_sample(sigma, sample, calc_dtype)
    return (sample.to(calc_dtype) - velocity.to(calc_dtype) * sigma).to(sample.dtype)


def find_matching_file(root_path: str, pattern: str) -> Path:
    """
    Recursively search for files matching a glob pattern and return the first match.
    """
    matches = list(Path(root_path).rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching pattern '{pattern}' found under {root_path}")
    return matches[0]
