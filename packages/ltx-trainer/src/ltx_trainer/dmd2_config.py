from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from ltx_trainer.config import (
    AccelerationConfig,
    CheckpointsConfig,
    ConfigBaseModel,
    DataConfig,
    HubConfig,
    LoraConfig,
    ModelConfig,
    ValidationConfig,
    WandbConfig,
)


class Dmd2OptimizationConfig(ConfigBaseModel):
    steps: int = Field(default=3000, gt=0)
    batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=1, gt=0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)

    generator_learning_rate: float = Field(default=1e-5, gt=0.0)
    fake_learning_rate: float = Field(default=2e-5, gt=0.0)
    discriminator_learning_rate: float = Field(default=2e-4, gt=0.0)

    optimizer_type: Literal["adamw"] = Field(default="adamw")
    enable_gradient_checkpointing: bool = Field(default=True)
    teacher_forward_chunk_size: int | None = Field(
        default=8,
        description="Chunk size for teacher mean-velocity evaluation. Smaller values reduce peak memory.",
        ge=1,
    )
    teacher_cpu_offload: bool = Field(
        default=False,
        description=(
            "When true, keep the frozen teacher on CPU and move it to the local GPU only for teacher forward passes. "
            "This only affects DMD2 teacher execution and does not change student/fake FSDP behavior."
        ),
    )
    teacher_fsdp_cpu_offload: bool = Field(
        default=False,
        description=(
            "When true, wrap only the frozen DMD2 teacher with FSDP parameter CPU offload. "
            "This leaves student/fake FSDP behavior unchanged."
        ),
    )


class Dmd2ObjectiveConfig(ConfigBaseModel):
    distribution_matching_weight: float = Field(default=1.0, ge=0.0)
    distribution_matching_step_size: float = Field(
        default=1.0,
        ge=0.0,
        description="Step size for the DMD2 pseudo-target update x <- x - eta * (v_real - v_fake).",
    )
    gan_loss_weight: float = Field(default=1.0, ge=0.0)
    fake_critic_steps_per_generator: int = Field(default=5, ge=1)
    fake_critic_warmup_steps: int = Field(default=0, ge=0)
    dm_intervals_per_step: int = Field(
        default=1,
        ge=1,
        description="How many adjacent sigma intervals to supervise per generator step.",
    )
    discriminator_loss_type: Literal["hinge", "bce"] = Field(default="hinge")
    fake_diffusion_sigma_max: float = Field(default=1.0, gt=0.0)
    discriminator_hidden_dim: int = Field(default=1024, gt=0)
    discriminator_depth: int = Field(default=4, gt=0)
    discriminator_text_conditioning: bool = Field(default=True)


class Dmd2TrainerConfig(ConfigBaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    lora: LoraConfig | None = Field(default=None)
    optimization: Dmd2OptimizationConfig = Field(default_factory=Dmd2OptimizationConfig)
    acceleration: AccelerationConfig = Field(default_factory=AccelerationConfig)
    data: DataConfig
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    checkpoints: CheckpointsConfig = Field(default_factory=CheckpointsConfig)
    hub: HubConfig = Field(default_factory=HubConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    dmd2: Dmd2ObjectiveConfig = Field(default_factory=Dmd2ObjectiveConfig)

    seed: int = Field(default=42)
    output_dir: str = Field(default="outputs_dmd2")

    @field_validator("output_dir")
    @classmethod
    def expand_output_path(cls, value: str) -> str:
        return str(Path(value).expanduser().resolve())

    @model_validator(mode="after")
    def validate_compatibility(self) -> "Dmd2TrainerConfig":
        if self.model.training_mode not in {"full", "lora"}:
            raise ValueError("DMD2 video training requires model.training_mode to be either 'full' or 'lora'")
        if self.model.training_mode == "lora" and self.lora is None:
            raise ValueError("LoRA configuration must be provided when model.training_mode is 'lora'")
        if self.model.training_mode == "full" and self.acceleration.quantization is not None:
            raise ValueError("Quantization is only supported for DMD2 LoRA training.")
        if self.optimization.teacher_cpu_offload and self.optimization.teacher_fsdp_cpu_offload:
            raise ValueError("Use only one teacher offload mode: teacher_cpu_offload or teacher_fsdp_cpu_offload")
        return self
