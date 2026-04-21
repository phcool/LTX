from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import yaml
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from safetensors.torch import save_file
from torch.distributed.fsdp import CPUOffload
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.distributed.ulysses import configure_ulysses_sequence_parallel, disable_ulysses_sequence_parallel
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.modality import Modality
from ltx_core.text_encoders.gemma import convert_to_additive_mask
from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape
from ltx_core.utils import to_denoised
from ltx_trainer import logger
from ltx_trainer.datasets import PrecomputedDataset
from ltx_trainer.discriminators import VideoLatentDiscriminator
from ltx_trainer.distillation_constants import DISTILLED_PIPELINE_STAGE1_SIGMAS
from ltx_trainer.dmd2_config import Dmd2TrainerConfig
from ltx_trainer.gpu_utils import free_gpu_memory_context, get_gpu_memory_gb
from ltx_trainer.model_loader import (
    load_embeddings_processor,
    load_model as load_ltx_model,
    load_text_encoder,
)
from ltx_trainer.progress import TrainingProgress
from ltx_trainer.quantization import quantize_model
from ltx_trainer.training_strategies.base_strategy import DEFAULT_FPS, TrainingStrategy
from ltx_trainer.validation_sampler import CachedPromptEmbeddings, GenerationConfig, ValidationSampler

os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings(
    "ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization"
)

IS_MAIN_PROCESS = os.environ.get("LOCAL_RANK", "0") == "0"
if not IS_MAIN_PROCESS:
    from transformers.utils.logging import disable_progress_bar

    disable_progress_bar()


VIDEO_SCALE_FACTORS = SpatioTemporalScaleFactors.default()


def _get_rank_local_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}"


def _is_fsdp_launch() -> bool:
    env_candidates = (
        os.environ.get("ACCELERATE_USE_FSDP"),
        os.environ.get("ACCELERATE_DISTRIBUTED_TYPE"),
        os.environ.get("DISTRIBUTED_TYPE"),
    )
    return any(value is not None and "FSDP" in value.upper() for value in env_candidates)


@dataclass
class Dmd2Batch:
    clean_video: torch.Tensor
    initial_noise_video: torch.Tensor
    video_prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    positions: torch.Tensor
    conditioning_mask: torch.Tensor
    loss_mask: torch.Tensor
    sigmas: torch.Tensor


@dataclass
class RolloutStep:
    latent: torch.Tensor
    x0_pred: torch.Tensor
    sigma_hi: torch.Tensor
    sigma_lo: torch.Tensor


@dataclass
class NoisedStudentSample:
    sigma: torch.Tensor
    noise: torch.Tensor
    noisy_video: torch.Tensor


class DMD2Trainer:
    def __init__(self, config: Dmd2TrainerConfig) -> None:
        self._config = config
        self._global_step = 0
        self._checkpoint_paths: list[Path] = []
        self._euler_step = EulerDiffusionStep()
        set_seed(self._config.seed)

        if IS_MAIN_PROCESS:
            logger.info(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False, allow_unicode=False))

        self._cached_validation_embeddings = self._load_text_encoder_and_cache_embeddings()
        logger.info("Text encoder and embedding cache ready")
        self._load_models()
        logger.info("Model loading complete")
        self._setup_accelerator()
        logger.info("Accelerator setup complete")
        self._prepare_models_for_training()
        logger.info("Model preparation complete")
        self._init_optimizers()
        logger.info("Optimizer initialization complete")
        self._init_dataset()
        logger.info("Dataset and dataloader initialization complete")

    @free_gpu_memory_context(after=True)
    def _load_text_encoder_and_cache_embeddings(self) -> list[CachedPromptEmbeddings] | None:
        device = _get_rank_local_device()
        text_encoder = load_text_encoder(
            gemma_model_path=self._config.model.text_encoder_path,
            device=device,
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )
        self._embeddings_processor = load_embeddings_processor(
            checkpoint_path=self._config.model.model_path,
            device=device,
            dtype=torch.bfloat16,
        )

        cached_embeddings = None
        if self._config.validation.prompts:
            cached_embeddings = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    pos_hs, pos_mask = text_encoder.encode(prompt)
                    pos_out = self._embeddings_processor.process_hidden_states(pos_hs, pos_mask)

                    neg_hs, neg_mask = text_encoder.encode(self._config.validation.negative_prompt)
                    neg_out = self._embeddings_processor.process_hidden_states(neg_hs, neg_mask)

                    cached_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=pos_out.video_encoding.cpu(),
                            audio_context_positive=pos_out.audio_encoding.cpu(),
                            video_context_negative=neg_out.video_encoding.cpu(),
                            audio_context_negative=neg_out.audio_encoding.cpu() if neg_out.audio_encoding is not None else None,
                        )
                    )

        del text_encoder
        self._embeddings_processor.feature_extractor = None
        return cached_embeddings

    def _load_models(self) -> None:
        load_audio = self._config.validation.generate_audio
        components = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=self._config.validation.images is not None,
            with_video_vae_decoder=True,
            with_audio_vae_decoder=load_audio,
            with_vocoder=load_audio,
            with_text_encoder=False,
        )

        use_fp32_trainers = self._config.model.training_mode == "full" or (
            self._config.model.training_mode == "lora" and _is_fsdp_launch()
        )
        train_model_dtype = torch.float32 if use_fp32_trainers else torch.bfloat16
        self._student = components.transformer.to(dtype=train_model_dtype)
        self._teacher = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=False,
            with_video_vae_decoder=False,
            with_audio_vae_decoder=False,
            with_vocoder=False,
            with_text_encoder=False,
        ).transformer.to(dtype=torch.bfloat16)
        self._fake = load_ltx_model(
            checkpoint_path=self._config.model.model_path,
            device="cpu",
            dtype=torch.bfloat16,
            with_video_vae_encoder=False,
            with_video_vae_decoder=False,
            with_audio_vae_decoder=False,
            with_vocoder=False,
            with_text_encoder=False,
        ).transformer.to(dtype=train_model_dtype)

        if self._config.acceleration.quantization is not None:
            logger.info(
                f'Quantizing student and fake models with "{self._config.acceleration.quantization}". This may take a while...'
            )
            self._student = quantize_model(
                self._student,
                precision=self._config.acceleration.quantization,
            )
            self._fake = quantize_model(
                self._fake,
                precision=self._config.acceleration.quantization,
            )

        self._vae_decoder = components.video_vae_decoder.to(dtype=torch.bfloat16)
        self._vae_encoder = components.video_vae_encoder
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to(dtype=torch.bfloat16)
        self._audio_vae = components.audio_vae_decoder
        self._vocoder = components.vocoder

        self._teacher.requires_grad_(False)
        self._student.requires_grad_(False)
        self._fake.requires_grad_(False)
        if self._config.model.training_mode == "lora":
            self._student = self._setup_lora_for_model(self._student)
            self._fake = self._setup_lora_for_model(self._fake)
        elif self._config.model.training_mode == "full":
            self._student.requires_grad_(True)
            self._fake.requires_grad_(True)
        else:
            raise ValueError(f"Unknown training mode: {self._config.model.training_mode}")

        self._student_trainable_params = [p for p in self._student.parameters() if p.requires_grad]
        self._fake_trainable_params = [p for p in self._fake.parameters() if p.requires_grad]
        logger.info(
            "Trainable params count: "
            f"student={sum(p.numel() for p in self._student_trainable_params):,}, "
            f"fake={sum(p.numel() for p in self._fake_trainable_params):,}"
        )

        student_model = self._student.get_base_model() if hasattr(self._student, "get_base_model") else self._student
        latent_dim = getattr(student_model, "patchify_proj").in_features
        self._discriminator = VideoLatentDiscriminator(
            latent_dim=latent_dim,
            context_dim=student_model.inner_dim,
            hidden_dim=self._config.dmd2.discriminator_hidden_dim,
            depth=self._config.dmd2.discriminator_depth,
            use_text_conditioning=self._config.dmd2.discriminator_text_conditioning,
        )

    def _setup_lora_for_model(self, model: torch.nn.Module) -> torch.nn.Module:
        lora_cfg = self._config.lora
        if lora_cfg is None:
            raise ValueError("LoRA configuration must be provided when model.training_mode is 'lora'")
        logger.debug(f"Adding LoRA adapter with rank {lora_cfg.rank}")
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            init_lora_weights=True,
        )
        # noinspection PyTypeChecker
        model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)

        target_dtype = next(
            (param.dtype for param in model.parameters() if torch.is_floating_point(param) and not param.requires_grad),
            None,
        )
        if target_dtype is not None:
            for param in model.parameters():
                if param.requires_grad and torch.is_floating_point(param) and param.dtype != target_dtype:
                    param.data = param.data.to(target_dtype)

        return model

    def _setup_accelerator(self) -> None:
        self._accelerator = Accelerator(
            mixed_precision=self._config.acceleration.mixed_precision_mode,
            gradient_accumulation_steps=self._config.optimization.gradient_accumulation_steps,
        )
        if self._accelerator.distributed_type == DistributedType.FSDP:
            fsdp_plugin = getattr(self._accelerator.state, "fsdp_plugin", None)
            if fsdp_plugin is not None and getattr(fsdp_plugin, "sync_module_states", False):
                logger.warning(
                    "Disabling FSDP sync_module_states for DMD2: models are initialized deterministically on every rank "
                    "and wrapped from CPU, so GPU-side sync at FSDP init would otherwise fail."
                )
                fsdp_plugin.sync_module_states = False
        if self._accelerator.num_processes > 1:
            logger.info(
                f"{self._accelerator.distributed_type.value} distributed training enabled "
                f"with {self._accelerator.num_processes} processes"
            )
        self._setup_ulysses_sequence_parallel()

    def _setup_ulysses_sequence_parallel(self) -> None:
        disable_ulysses_sequence_parallel()
        ulysses_cfg = self._config.acceleration.ulysses
        if not ulysses_cfg.enabled or ulysses_cfg.sequence_parallel_size <= 1:
            return
        if self._accelerator.num_processes <= 1 or not dist.is_available() or not dist.is_initialized():
            return

        sp_size = ulysses_cfg.sequence_parallel_size
        world_size = self._accelerator.num_processes
        if world_size % sp_size != 0:
            raise ValueError(
                f"acceleration.ulysses.sequence_parallel_size ({sp_size}) must divide process count ({world_size})"
            )

        heads = {module.heads for module in self._student.modules() if isinstance(module, Attention)}
        invalid_heads = sorted(head for head in heads if head % sp_size != 0)
        if invalid_heads:
            raise ValueError(f"Ulysses sequence_parallel_size must divide attention heads, got {invalid_heads}")

        local_group = None
        rank_in_group = 0
        process_index = self._accelerator.process_index
        for start in range(0, world_size, sp_size):
            ranks = list(range(start, start + sp_size))
            group = dist.new_group(ranks=ranks)
            if process_index in ranks:
                local_group = group
                rank_in_group = process_index - start

        if local_group is None:
            raise RuntimeError("Failed to create local Ulysses process group")

        configure_ulysses_sequence_parallel(
            local_group,
            sequence_parallel_size=sp_size,
            rank_in_group=rank_in_group,
        )
        logger.info(
            "Ulysses sequence parallel enabled: "
            f"group_size={sp_size}, rank_in_group={rank_in_group}, num_groups={world_size // sp_size}"
        )

    def _prepare_models_for_training(self) -> None:
        if (
            self._accelerator.distributed_type == DistributedType.FSDP
            and self._config.model.training_mode == "lora"
            and self._config.acceleration.quantization is None
        ):
            logger.debug("FSDP: casting student/fake to FP32 for uniform dtype")
            self._student = self._student.to(dtype=torch.float32)
            self._fake = self._fake.to(dtype=torch.float32)

        student_base = self._student.get_base_model() if hasattr(self._student, "get_base_model") else self._student
        fake_base = self._fake.get_base_model() if hasattr(self._fake, "get_base_model") else self._fake
        student_base.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)
        fake_base.set_gradient_checkpointing(self._config.optimization.enable_gradient_checkpointing)

        logger.info("Moving student to CPU before FSDP prepare")
        self._student = self._student.to("cpu")
        logger.info("Student moved to CPU")
        logger.info("Moving fake to CPU before FSDP prepare")
        self._fake = self._fake.to("cpu")
        logger.info("Fake moved to CPU")
        logger.info("Moving teacher to CPU before FSDP prepare")
        self._teacher = self._teacher.to("cpu")
        logger.info("Teacher moved to CPU")
        logger.info("Moving discriminator to CPU before accelerator device placement")
        self._discriminator = self._discriminator.to("cpu")
        logger.info("Discriminator moved to CPU")
        logger.info("Collecting model device summaries")
        self._log_model_device_summary("student", self._student)
        self._log_model_device_summary("fake", self._fake)
        self._log_model_device_summary("teacher", self._teacher)
        self._log_model_device_summary("discriminator", self._discriminator)
        logger.info("Model device summaries collected")

        logger.info("Moving VAE decoder to CPU")
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            logger.info("Moving VAE encoder to CPU")
            self._vae_encoder = self._vae_encoder.to("cpu")

        logger.info("Preparing student with accelerator/FSDP")
        self._student = self._accelerator.prepare(self._student)
        logger.info("Student prepared")

        logger.info("Preparing fake with accelerator/FSDP")
        self._fake = self._accelerator.prepare(self._fake)
        logger.info("Fake prepared")

        if self._config.optimization.teacher_cpu_offload:
            logger.info("Keeping teacher on CPU for DMD2-only teacher offload")
            self._teacher = self._teacher.to("cpu")
            self._teacher.eval()
        elif self._config.optimization.teacher_fsdp_cpu_offload:
            logger.info("Preparing teacher with FSDP CPU offload")
            self._teacher = self._prepare_teacher_with_fsdp_cpu_offload(self._teacher)
            logger.info("Teacher prepared with FSDP CPU offload")
            self._teacher.eval()
        else:
            logger.info("Preparing teacher with accelerator/FSDP")
            self._teacher = self._accelerator.prepare(self._teacher)
            logger.info("Teacher prepared")
            self._teacher.eval()

        logger.info("Moving discriminator to accelerator device without FSDP wrapping")
        self._discriminator = self._discriminator.to(self._accelerator.device)

    def _prepare_teacher_with_fsdp_cpu_offload(self, teacher: torch.nn.Module) -> torch.nn.Module:
        if self._accelerator.distributed_type != DistributedType.FSDP:
            raise ValueError("optimization.teacher_fsdp_cpu_offload requires launching DMD2 with FSDP")

        fsdp_plugin = getattr(self._accelerator.state, "fsdp_plugin", None)
        if fsdp_plugin is None:
            raise RuntimeError("Accelerator FSDP plugin is not available for teacher FSDP CPU offload")

        original_cpu_offload = fsdp_plugin.cpu_offload
        fsdp_plugin.cpu_offload = CPUOffload(offload_params=True)
        try:
            return self._accelerator.prepare(teacher)
        finally:
            fsdp_plugin.cpu_offload = original_cpu_offload

    def _log_model_device_summary(self, name: str, model: torch.nn.Module) -> None:
        if not IS_MAIN_PROCESS:
            return
        device_counts: dict[str, int] = {}
        for param in model.parameters():
            key = str(param.device)
            device_counts[key] = device_counts.get(key, 0) + 1
        logger.info(f"{name} parameter devices: {device_counts}")

    def _run_teacher_inference(self, modality: Modality) -> torch.Tensor:
        if self._config.optimization.teacher_cpu_offload:
            teacher_device = self._accelerator.device
            teacher_dtype = next(self._teacher.parameters()).dtype

            teacher_modality = Modality(
                enabled=modality.enabled,
                latent=modality.latent.to(device=teacher_device, dtype=teacher_dtype),
                sigma=modality.sigma.to(device=teacher_device, dtype=teacher_dtype),
                timesteps=modality.timesteps.to(device=teacher_device, dtype=teacher_dtype),
                positions=modality.positions.to(device=teacher_device, dtype=teacher_dtype),
                context=modality.context.to(device=teacher_device, dtype=teacher_dtype),
                context_mask=modality.context_mask.to(device=teacher_device)
                if modality.context_mask is not None
                else None,
                attention_mask=modality.attention_mask.to(device=teacher_device)
                if modality.attention_mask is not None
                else None,
            )

            self._teacher = self._teacher.to(teacher_device)
            try:
                with torch.no_grad():
                    teacher_velocity, _ = self._teacher(video=teacher_modality, audio=None, perturbations=None)
            finally:
                self._teacher = self._teacher.to("cpu")

            return teacher_velocity

        teacher_param = next(self._teacher.parameters())
        teacher_device = self._accelerator.device if self._config.optimization.teacher_fsdp_cpu_offload else teacher_param.device
        teacher_dtype = teacher_param.dtype

        teacher_modality = Modality(
            enabled=modality.enabled,
            latent=modality.latent.to(device=teacher_device, dtype=teacher_dtype),
            sigma=modality.sigma.to(device=teacher_device, dtype=teacher_dtype),
            timesteps=modality.timesteps.to(device=teacher_device, dtype=teacher_dtype),
            positions=modality.positions.to(device=teacher_device, dtype=teacher_dtype),
            context=modality.context.to(device=teacher_device, dtype=teacher_dtype),
            context_mask=modality.context_mask.to(device=teacher_device) if modality.context_mask is not None else None,
            attention_mask=modality.attention_mask.to(device=teacher_device)
            if modality.attention_mask is not None
            else None,
        )

        with torch.no_grad():
            teacher_velocity, _ = self._teacher(video=teacher_modality, audio=None, perturbations=None)
        return teacher_velocity

    def _init_optimizers(self) -> None:
        logger.info("Initializing optimizers")
        self._student_optimizer = AdamW(
            self._student_trainable_params,
            lr=self._config.optimization.generator_learning_rate,
        )
        self._fake_optimizer = AdamW(
            self._fake_trainable_params,
            lr=self._config.optimization.fake_learning_rate,
        )
        self._disc_optimizer = AdamW(
            self._discriminator.parameters(),
            lr=self._config.optimization.discriminator_learning_rate,
        )

        self._student_optimizer, self._fake_optimizer, self._disc_optimizer = self._accelerator.prepare(
            self._student_optimizer,
            self._fake_optimizer,
            self._disc_optimizer,
        )
        logger.info("Optimizers prepared by accelerator")

    def _init_dataset(self) -> None:
        logger.info(f"Building precomputed dataset from {self._config.data.preprocessed_data_root}")
        dataset = PrecomputedDataset(self._config.data.preprocessed_data_root, {"latents": "latents", "conditions": "conditions"})
        dataloader = DataLoader(
            dataset,
            batch_size=self._config.optimization.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self._config.data.num_dataloader_workers,
            pin_memory=self._config.data.num_dataloader_workers > 0,
            persistent_workers=self._config.data.num_dataloader_workers > 0,
        )
        logger.info(f"Dataset size: {len(dataset)} items")
        self._dataloader = self._accelerator.prepare(dataloader)
        logger.info("Dataloader prepared by accelerator")

    def _prepare_dmd2_batch(self, batch: dict[str, dict[str, torch.Tensor]]) -> Dmd2Batch:
        latents = batch["latents"]
        video_latents = self._video_patchifier.patchify(latents["latents"])

        num_frames = latents["num_frames"][0].item()
        height = latents["height"][0].item()
        width = latents["width"][0].item()

        fps = latents.get("fps", None)
        if fps is not None and not torch.all(fps == fps[0]):
            logger.warning(f"Different FPS values found in batch: {fps.tolist()}, using {fps[0].item()}")
        fps = fps[0].item() if fps is not None else DEFAULT_FPS

        conditions = batch["conditions"]
        video_prompt_embeds = conditions["video_prompt_embeds"]
        prompt_attention_mask = conditions["prompt_attention_mask"]

        batch_size, seq_len, _latent_dim = video_latents.shape
        device = video_latents.device
        dtype = video_latents.dtype

        conditioning_mask = TrainingStrategy._create_first_frame_conditioning_mask(
            batch_size=batch_size,
            sequence_length=seq_len,
            height=height,
            width=width,
            device=device,
            first_frame_conditioning_p=0.0,
        )

        initial_noise = torch.randn_like(video_latents)
        initial_noise = torch.where(conditioning_mask.unsqueeze(-1), video_latents, initial_noise)

        positions = self._video_positions(
            num_frames=num_frames,
            height=height,
            width=width,
            batch_size=batch_size,
            fps=fps,
            device=device,
            dtype=dtype,
        )

        return Dmd2Batch(
            clean_video=video_latents,
            initial_noise_video=initial_noise,
            video_prompt_embeds=video_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            positions=positions,
            conditioning_mask=conditioning_mask,
            loss_mask=~conditioning_mask,
            sigmas=torch.tensor(DISTILLED_PIPELINE_STAGE1_SIGMAS, device=device, dtype=dtype),
        )

    def _video_positions(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        batch_size: int,
        fps: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        latent_coords = self._video_patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape(
                frames=num_frames,
                height=height,
                width=width,
                batch=batch_size,
                channels=128,
            ),
            device=device,
        )
        pixel_coords = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=VIDEO_SCALE_FACTORS,
            causal_fix=True,
        ).to(dtype)
        pixel_coords[:, 0, ...] = pixel_coords[:, 0, ...] / fps
        return pixel_coords

    @property
    def _video_patchifier(self):
        if not hasattr(self, "__video_patchifier"):
            from ltx_core.components.patchifiers import VideoLatentPatchifier

            self.__video_patchifier = VideoLatentPatchifier(patch_size=1)
        return self.__video_patchifier

    def _mix_latents(
        self,
        clean_video: torch.Tensor,
        noise_video: torch.Tensor,
        conditioning_mask: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        sigma_expanded = sigma.view(-1, 1, 1)
        mixed = (1 - sigma_expanded) * clean_video + sigma_expanded * noise_video
        return torch.where(conditioning_mask.unsqueeze(-1), clean_video, mixed)

    def _build_video_modality(
        self,
        *,
        latent: torch.Tensor,
        sigma: torch.Tensor,
        conditioning_mask: torch.Tensor,
        positions: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> Modality:
        timesteps = TrainingStrategy._create_per_token_timesteps(conditioning_mask, sigma)
        return Modality(
            enabled=True,
            latent=latent,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=context,
            context_mask=context_mask,
        )

    def _masked_mse(self, pred: torch.Tensor, target: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        loss = (pred - target).pow(2)
        mask = loss_mask.unsqueeze(-1).float()
        return loss.mul(mask).div(mask.mean().clamp_min(1e-6)).mean()

    def _sample_training_sigma(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma_candidates = torch.tensor(
            [sigma for sigma in DISTILLED_PIPELINE_STAGE1_SIGMAS if sigma > 0.0],
            device=device,
            dtype=dtype,
        )
        sigma_indices = torch.randint(0, len(sigma_candidates), (batch_size,), device=device)
        return sigma_candidates[sigma_indices]

    def _noise_generated_video(
        self,
        generated_video: torch.Tensor,
        conditioning_mask: torch.Tensor,
    ) -> NoisedStudentSample:
        batch_size = generated_video.shape[0]
        sigma = self._sample_training_sigma(batch_size, generated_video.device, generated_video.dtype)
        noise = torch.randn_like(generated_video)
        noisy_video = self._mix_latents(generated_video, noise, conditioning_mask, sigma)
        return NoisedStudentSample(
            sigma=sigma,
            noise=noise,
            noisy_video=noisy_video,
        )

    def _rollout_student(self, batch: Dmd2Batch) -> tuple[torch.Tensor, list[RolloutStep]]:
        current = self._mix_latents(batch.clean_video, batch.initial_noise_video, batch.conditioning_mask, batch.sigmas[0].repeat(batch.clean_video.shape[0]))
        steps: list[RolloutStep] = []
        for step_index in range(len(batch.sigmas) - 1):
            sigma_hi = batch.sigmas[step_index].repeat(batch.clean_video.shape[0])
            sigma_lo = batch.sigmas[step_index + 1].repeat(batch.clean_video.shape[0])
            modality = self._build_video_modality(
                latent=current,
                sigma=sigma_hi,
                conditioning_mask=batch.conditioning_mask,
                positions=batch.positions,
                context=batch.video_prompt_embeds,
                context_mask=batch.prompt_attention_mask,
            )
            velocity, _ = self._student(video=modality, audio=None, perturbations=None)
            x0_pred = to_denoised(current, velocity, modality.timesteps)
            x0_pred = torch.where(batch.conditioning_mask.unsqueeze(-1), batch.clean_video, x0_pred)
            steps.append(RolloutStep(latent=current, x0_pred=x0_pred, sigma_hi=sigma_hi, sigma_lo=sigma_lo))
            current = self._euler_step.step(sample=current, denoised_sample=x0_pred, sigmas=batch.sigmas, step_index=step_index)
            current = torch.where(batch.conditioning_mask.unsqueeze(-1), batch.clean_video, current)
        return current, steps

    def _sample_dm_indices(self, num_intervals: int, device: torch.device) -> torch.Tensor:
        count = min(self._config.dmd2.dm_intervals_per_step, num_intervals)
        perm = torch.randperm(num_intervals, device=device)
        return perm[:count]

    def _compute_generator_loss(self, batch: Dmd2Batch, rollout_final: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        noised_sample = self._noise_generated_video(rollout_final, batch.conditioning_mask)

        teacher_modality = self._build_video_modality(
            latent=noised_sample.noisy_video.detach(),
            sigma=noised_sample.sigma.detach(),
            conditioning_mask=batch.conditioning_mask,
            positions=batch.positions,
            context=batch.video_prompt_embeds,
            context_mask=batch.prompt_attention_mask,
        )
        teacher_velocity = self._run_teacher_inference(teacher_modality)

        with torch.no_grad():
            fake_modality = self._build_video_modality(
                latent=noised_sample.noisy_video.detach(),
                sigma=noised_sample.sigma.detach(),
                conditioning_mask=batch.conditioning_mask,
                positions=batch.positions,
                context=batch.video_prompt_embeds,
                context_mask=batch.prompt_attention_mask,
            )
            fake_velocity, _ = self._fake(video=fake_modality, audio=None, perturbations=None)

        dm_target = rollout_final.detach() - self._config.dmd2.distribution_matching_step_size * (
            teacher_velocity.detach() - fake_velocity.detach()
        )
        dm_loss = self._masked_mse(rollout_final, dm_target, batch.loss_mask)

        gen_adv_logits = self._discriminator(
            rollout_final,
            batch.loss_mask,
            batch.video_prompt_embeds if self._config.dmd2.discriminator_text_conditioning else None,
            batch.prompt_attention_mask if self._config.dmd2.discriminator_text_conditioning else None,
        )
        if self._config.dmd2.discriminator_loss_type == "hinge":
            gan_loss = -gen_adv_logits.mean()
        else:
            gan_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                gen_adv_logits, torch.ones_like(gen_adv_logits)
            )

        total = (
            self._config.dmd2.distribution_matching_weight * dm_loss
            + self._config.dmd2.gan_loss_weight * gan_loss
        )
        metrics = {
            "loss/generator_total": float(total.detach().item()),
            "loss/dm": float(dm_loss.detach().item()),
            "loss/gan_g": float(gan_loss.detach().item()),
            "dm/sampled_sigma": float(noised_sample.sigma.detach().float().mean().item()),
        }
        return total, metrics

    def _compute_fake_loss(self, batch: Dmd2Batch, generated_video: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        noised_sample = self._noise_generated_video(generated_video, batch.conditioning_mask)
        fake_modality = self._build_video_modality(
            latent=noised_sample.noisy_video,
            sigma=noised_sample.sigma,
            conditioning_mask=batch.conditioning_mask,
            positions=batch.positions,
            context=batch.video_prompt_embeds,
            context_mask=batch.prompt_attention_mask,
        )
        fake_velocity, _ = self._fake(video=fake_modality, audio=None, perturbations=None)
        fake_target = noised_sample.noise - generated_video
        loss = self._masked_mse(fake_velocity, fake_target, batch.loss_mask)
        return loss, {
            "loss/fake_diffusion": float(loss.detach().item()),
            "fake/sampled_sigma": float(noised_sample.sigma.detach().float().mean().item()),
        }

    def _compute_discriminator_loss(self, batch: Dmd2Batch, generated_video: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        real_logits = self._discriminator(
            batch.clean_video.detach(),
            batch.loss_mask,
            batch.video_prompt_embeds if self._config.dmd2.discriminator_text_conditioning else None,
            batch.prompt_attention_mask if self._config.dmd2.discriminator_text_conditioning else None,
        )
        fake_logits = self._discriminator(
            generated_video.detach(),
            batch.loss_mask,
            batch.video_prompt_embeds if self._config.dmd2.discriminator_text_conditioning else None,
            batch.prompt_attention_mask if self._config.dmd2.discriminator_text_conditioning else None,
        )
        if self._config.dmd2.discriminator_loss_type == "hinge":
            loss = torch.relu(1.0 - real_logits).mean() + torch.relu(1.0 + fake_logits).mean()
        else:
            real_loss = torch.nn.functional.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            fake_loss = torch.nn.functional.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
            loss = real_loss + fake_loss
        return loss, {
            "loss/discriminator": float(loss.detach().item()),
            "disc/real_logits": float(real_logits.detach().mean().item()),
            "disc/fake_logits": float(fake_logits.detach().mean().item()),
        }

    def _run_validation(self) -> list[Path]:
        if not self._config.validation.prompts:
            return []

        sampler = ValidationSampler(
            transformer=self._student,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,
            audio_decoder=self._audio_vae if self._config.validation.generate_audio else None,
            vocoder=self._vocoder if self._config.validation.generate_audio else None,
            embeddings_processor=self._embeddings_processor,
        )
        sample_dir = Path(self._config.output_dir) / "samples" / f"step_{self._global_step:05d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for idx, prompt in enumerate(self._config.validation.prompts):
            config = GenerationConfig(
                prompt=prompt,
                negative_prompt=self._config.validation.negative_prompt,
                height=self._config.validation.video_dims[1],
                width=self._config.validation.video_dims[0],
                num_frames=self._config.validation.video_dims[2],
                frame_rate=self._config.validation.frame_rate,
                num_inference_steps=self._config.validation.inference_steps,
                guidance_scale=self._config.validation.guidance_scale,
                seed=self._config.validation.seed + idx,
                cached_embeddings=self._cached_validation_embeddings[idx] if self._cached_validation_embeddings else None,
                stg_scale=self._config.validation.stg_scale,
                stg_blocks=self._config.validation.stg_blocks,
                stg_mode=self._config.validation.stg_mode,
                generate_audio=self._config.validation.generate_audio,
            )
            video, _audio = sampler.generate(config, device=self._accelerator.device)
            path = sample_dir / f"sample_{idx:02d}.pt"
            torch.save(video.cpu(), path)
            saved_paths.append(path)

        return saved_paths

    def _save_checkpoint(self) -> Path | None:
        self._accelerator.wait_for_everyone()
        student_full_state = self._accelerator.get_state_dict(self._student)
        fake_full_state = self._accelerator.get_state_dict(self._fake)
        disc_state = self._accelerator.get_state_dict(self._discriminator)

        if not IS_MAIN_PROCESS:
            return None

        checkpoint_dir = Path(self._config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        student_path = checkpoint_dir / f"student_step_{self._global_step:05d}.safetensors"
        fake_path = checkpoint_dir / f"fake_step_{self._global_step:05d}.safetensors"
        disc_path = checkpoint_dir / f"discriminator_step_{self._global_step:05d}.pt"

        if self._config.model.training_mode == "lora":
            unwrapped_student = self._accelerator.unwrap_model(self._student)
            unwrapped_fake = self._accelerator.unwrap_model(self._fake)
            student_state = get_peft_model_state_dict(unwrapped_student, state_dict=student_full_state)
            fake_state = get_peft_model_state_dict(unwrapped_fake, state_dict=fake_full_state)
        else:
            student_state = student_full_state
            fake_state = fake_full_state

        save_file({k: v.to(torch.bfloat16) for k, v in student_state.items()}, student_path)
        save_file({k: v.to(torch.bfloat16) for k, v in fake_state.items()}, fake_path)
        torch.save(disc_state, disc_path)

        self._checkpoint_paths.append(student_path)
        if 0 < self._config.checkpoints.keep_last_n < len(self._checkpoint_paths):
            to_remove = self._checkpoint_paths[: -self._config.checkpoints.keep_last_n]
            for path in to_remove:
                if path.exists():
                    path.unlink()
            self._checkpoint_paths = self._checkpoint_paths[-self._config.checkpoints.keep_last_n :]

        return student_path

    def train(self, disable_progress_bars: bool = False) -> None:
        set_seed(self._config.seed)
        Path(self._config.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(self._config.output_dir) / "config.yaml", "w") as handle:
            yaml.safe_dump(self._config.model_dump(mode="json"), handle, sort_keys=False)
        logger.info("Starting DMD2 training loop")

        progress = TrainingProgress(
            enabled=IS_MAIN_PROCESS and not disable_progress_bars,
            total_steps=self._config.optimization.steps,
        )
        data_iter = iter(self._dataloader)
        start_time = time.time()
        start_mem = get_gpu_memory_gb(self._accelerator.device)

        with progress:
            if self._config.validation.interval and not self._config.validation.skip_initial_validation:
                self._run_validation()

            for step in range(self._config.optimization.steps):
                step_start_time = time.time()
                self._global_step = step + 1
                try:
                    raw_batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    raw_batch = next(data_iter)

                conditions = raw_batch["conditions"]
                audio_prompt_embeds = conditions.get("audio_prompt_embeds", conditions["video_prompt_embeds"])
                additive_mask = convert_to_additive_mask(
                    conditions["prompt_attention_mask"], conditions["video_prompt_embeds"].dtype
                )
                with torch.no_grad():
                    video_embeds, _audio_embeds, attention_mask = self._embeddings_processor.create_embeddings(
                        conditions["video_prompt_embeds"],
                        audio_prompt_embeds,
                        additive_mask,
                    )
                raw_batch["conditions"]["video_prompt_embeds"] = video_embeds.detach()
                raw_batch["conditions"]["prompt_attention_mask"] = attention_mask.detach()

                batch = self._prepare_dmd2_batch(raw_batch)
                generator_step = (
                    self._global_step > self._config.dmd2.fake_critic_warmup_steps
                    and (self._global_step - self._config.dmd2.fake_critic_warmup_steps - 1)
                    % self._config.dmd2.fake_critic_steps_per_generator
                    == 0
                )

                if generator_step:
                    rollout_final, _rollout_steps = self._rollout_student(batch)
                else:
                    with torch.no_grad():
                        rollout_final, _rollout_steps = self._rollout_student(batch)

                self._disc_optimizer.zero_grad(set_to_none=True)
                disc_loss, disc_metrics = self._compute_discriminator_loss(batch, rollout_final)
                self._accelerator.backward(disc_loss)
                self._disc_optimizer.step()

                self._fake_optimizer.zero_grad(set_to_none=True)
                fake_loss, fake_metrics = self._compute_fake_loss(batch, rollout_final.detach())
                self._accelerator.backward(fake_loss)
                self._fake_optimizer.step()

                metrics = {**disc_metrics, **fake_metrics}

                if generator_step:
                    self._student_optimizer.zero_grad(set_to_none=True)
                    gen_loss, gen_metrics = self._compute_generator_loss(batch, rollout_final)
                    self._accelerator.backward(gen_loss)
                    if self._config.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(self._student.parameters(), self._config.optimization.max_grad_norm)
                    self._student_optimizer.step()
                    metrics.update(gen_metrics)

                if IS_MAIN_PROCESS:
                    progress.update_training(
                        loss=metrics.get("loss/generator_total", metrics["loss/fake_diffusion"]),
                        lr=self._config.optimization.generator_learning_rate,
                        step_time=time.time() - step_start_time,
                    )
                    if self._global_step % 10 == 0:
                        logger.info(", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))

                if self._config.validation.interval and self._global_step % self._config.validation.interval == 0:
                    self._run_validation()

                if self._config.checkpoints.interval and self._global_step % self._config.checkpoints.interval == 0:
                    self._save_checkpoint()

        self._save_checkpoint()
        total_time = time.time() - start_time
        peak_mem = max(start_mem, get_gpu_memory_gb(self._accelerator.device))
        if IS_MAIN_PROCESS:
            logger.info(
                f"DMD2 training finished in {total_time / 60:.2f} min, "
                f"peak_gpu_memory_gb={peak_mem:.2f}, processes={self._accelerator.num_processes}"
            )
