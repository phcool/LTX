"""Sigma schedules for DMD2-style distillation training."""

from __future__ import annotations

try:
    from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as PIPELINE_DISTILLED_SIGMA_VALUES
except Exception:
    PIPELINE_DISTILLED_SIGMA_VALUES = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

# Matches the distilled inference pipeline exactly so student rollouts use the
# same fixed trajectory as `ltx_pipelines.distilled`.
DISTILLED_PIPELINE_STAGE1_SIGMAS: tuple[float, ...] = tuple(float(v) for v in PIPELINE_DISTILLED_SIGMA_VALUES)

SIGMA_RESOLUTION: int = 160
SIGMA_FINE_STEP: float = 1.0 / SIGMA_RESOLUTION


def fine_sigmas_descending(sigma_hi: float, sigma_lo: float, step: float = SIGMA_FINE_STEP) -> tuple[float, ...]:
    """Create a descending sigma grid spanning a closed interval."""
    if sigma_hi < sigma_lo:
        raise ValueError(f"Expected sigma_hi >= sigma_lo, got {sigma_hi} < {sigma_lo}")

    out = [float(sigma_hi)]
    eps = step * 0.25
    s = float(sigma_hi)
    while s - step > sigma_lo + eps:
        s -= step
        out.append(float(s))

    if abs(out[-1] - sigma_lo) > eps:
        out.append(float(sigma_lo))

    return tuple(out)
