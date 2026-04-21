from __future__ import annotations

import torch


class VideoLatentDiscriminator(torch.nn.Module):
    """Simple latent-space video discriminator for DMD2-style training.

    Operates on patchified video latents of shape ``[B, T, D]`` and pools
    non-conditioned tokens into a global representation. Text context can be
    fused through a pooled prompt embedding for semantic alignment.
    """

    def __init__(
        self,
        latent_dim: int = 128,
        context_dim: int = 2048,
        hidden_dim: int = 1024,
        depth: int = 4,
        use_text_conditioning: bool = True,
    ) -> None:
        super().__init__()
        self.use_text_conditioning = use_text_conditioning

        self.latent_in = torch.nn.Linear(latent_dim, hidden_dim)
        blocks = []
        for _ in range(depth):
            blocks.append(
                torch.nn.Sequential(
                    torch.nn.LayerNorm(hidden_dim),
                    torch.nn.Linear(hidden_dim, hidden_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim * 4, hidden_dim),
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

        if use_text_conditioning:
            self.context_in = torch.nn.Linear(context_dim, hidden_dim)
        else:
            self.context_in = None

        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        latents: torch.Tensor,
        loss_mask: torch.Tensor,
        prompt_embeds: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        target_device = self.latent_in.weight.device
        target_dtype = self.latent_in.weight.dtype

        latents = latents.to(device=target_device, dtype=target_dtype)
        x = self.latent_in(latents)
        for block in self.blocks:
            x = x + block(x)

        pooled = self._masked_mean(x, loss_mask)
        if self.use_text_conditioning and prompt_embeds is not None and self.context_in is not None:
            pooled_prompt = self._pool_prompt(prompt_embeds, prompt_attention_mask).to(
                device=target_device,
                dtype=self.context_in.weight.dtype,
            )
            pooled = pooled + self.context_in(pooled_prompt)

        return self.head(pooled).squeeze(-1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (x * mask).sum(dim=1) / denom

    @staticmethod
    def _pool_prompt(prompt_embeds: torch.Tensor, prompt_attention_mask: torch.Tensor | None) -> torch.Tensor:
        if prompt_attention_mask is None:
            return prompt_embeds.mean(dim=1)

        mask = prompt_attention_mask.float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (prompt_embeds * mask).sum(dim=1) / denom
