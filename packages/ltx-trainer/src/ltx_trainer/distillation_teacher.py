"""Teacher utilities for DMD2-style video distillation."""

from __future__ import annotations

import torch

from ltx_core.model.transformer.modality import Modality
from ltx_trainer.training_strategies.base_strategy import TrainingStrategy
from ltx_trainer.distillation_constants import fine_sigmas_descending


def rectified_teacher_mean_velocity_batch(
    teacher: torch.nn.Module,
    *,
    device: torch.device,
    clean_video: torch.Tensor,
    noise_video: torch.Tensor,
    conditioning_mask: torch.Tensor,
    positions: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor | None,
    sigma_hi: torch.Tensor,
    sigma_lo: torch.Tensor,
    teacher_forward_chunk_size: int | None = None,
) -> torch.Tensor:
    """Average teacher velocity on the rectified flow path from sigma_hi to sigma_lo."""
    sigma_hi = sigma_hi.reshape(-1)
    sigma_lo = sigma_lo.reshape(-1)

    if sigma_hi.shape != sigma_lo.shape:
        raise RuntimeError(
            f"Expected sigma_hi and sigma_lo to have the same shape, got {tuple(sigma_hi.shape)} vs {tuple(sigma_lo.shape)}"
        )

    batch_size = clean_video.shape[0]
    batch_indices: list[int] = []
    sigmas_list: list[float] = []
    for batch_idx in range(batch_size):
        for sigma in fine_sigmas_descending(float(sigma_hi[batch_idx].item()), float(sigma_lo[batch_idx].item())):
            batch_indices.append(batch_idx)
            sigmas_list.append(sigma)

    idx_t = torch.tensor(batch_indices, device=device, dtype=torch.long)
    sig_t = torch.tensor(sigmas_list, device=device, dtype=clean_video.dtype)
    chunk_size = teacher_forward_chunk_size or len(sig_t)
    if chunk_size <= 0:
        raise ValueError(f"teacher_forward_chunk_size must be >= 1, got {chunk_size}")

    sum_v = torch.zeros((batch_size, *clean_video.shape[1:]), device=device, dtype=torch.float32)
    cnt_v = torch.zeros((batch_size, *([1] * (clean_video.ndim - 1))), device=device, dtype=torch.float32)
    video_dtype = clean_video.dtype

    for start in range(0, len(sig_t), chunk_size):
        end = min(start + chunk_size, len(sig_t))
        idx_chunk = idx_t[start:end]
        sig_chunk = sig_t[start:end]
        sig_exp = sig_chunk.view(-1, 1, 1)

        latent_chunk = (1 - sig_exp) * clean_video[idx_chunk] + sig_exp * noise_video[idx_chunk]
        conditioning_chunk = conditioning_mask[idx_chunk]
        latent_chunk = torch.where(conditioning_chunk.unsqueeze(-1), clean_video[idx_chunk], latent_chunk)
        timesteps_chunk = TrainingStrategy._create_per_token_timesteps(conditioning_chunk, sig_chunk)

        video_modality = Modality(
            enabled=True,
            latent=latent_chunk,
            sigma=sig_chunk,
            timesteps=timesteps_chunk,
            positions=positions[idx_chunk],
            context=context[idx_chunk],
            context_mask=context_mask[idx_chunk] if context_mask is not None else None,
        )

        teacher_video, _teacher_audio = teacher(video=video_modality, audio=None, perturbations=None)
        video_dtype = teacher_video.dtype

        sum_v.index_add_(0, idx_chunk, teacher_video.float())
        cnt_v.index_add_(
            0,
            idx_chunk,
            torch.ones((len(idx_chunk), *([1] * (teacher_video.ndim - 1))), device=device, dtype=torch.float32),
        )

    return (sum_v / cnt_v.clamp_min(1.0)).to(dtype=video_dtype)
