from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import warnings

import torch
import torch.distributed.nn.functional as dist_nn


@dataclass(frozen=True)
class UlyssesSequenceParallelState:
    enabled: bool = False
    sequence_parallel_size: int = 1
    rank_in_group: int = 0
    process_group: torch.distributed.ProcessGroup | None = None


_DISABLED_STATE = UlyssesSequenceParallelState()
_state = _DISABLED_STATE
_warned_messages: set[str] = set()


def configure_ulysses_sequence_parallel(
    process_group: torch.distributed.ProcessGroup,
    *,
    sequence_parallel_size: int,
    rank_in_group: int,
) -> None:
    global _state
    _state = UlyssesSequenceParallelState(
        enabled=sequence_parallel_size > 1,
        sequence_parallel_size=sequence_parallel_size,
        rank_in_group=rank_in_group,
        process_group=process_group,
    )


def disable_ulysses_sequence_parallel() -> None:
    global _state
    _state = _DISABLED_STATE


def get_ulysses_sequence_parallel_state() -> UlyssesSequenceParallelState:
    return _state


def ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    heads: int,
    mask: torch.Tensor | None,
    attention_callable: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor | None], torch.Tensor],
) -> torch.Tensor:
    state = get_ulysses_sequence_parallel_state()
    if not state.enabled or state.process_group is None:
        return attention_callable(q, k, v, heads, mask)

    sequence_parallel_size = state.sequence_parallel_size
    if heads % sequence_parallel_size != 0:
        _warn_once(
            f"heads_{heads}_sp_{sequence_parallel_size}",
            "Ulysses sequence parallel is enabled, but attention heads are not divisible by "
            f"sequence_parallel_size ({heads} vs {sequence_parallel_size}). Falling back to regular attention.",
        )
        return attention_callable(q, k, v, heads, mask)

    original_q_len = q.shape[1]
    original_k_len = k.shape[1]
    padded_q_len = _next_multiple(original_q_len, sequence_parallel_size)
    padded_k_len = _next_multiple(original_k_len, sequence_parallel_size)

    if padded_q_len != original_q_len or padded_k_len != original_k_len:
        _warn_once(
            f"seq_pad_{original_q_len}_{original_k_len}_sp_{sequence_parallel_size}",
            "Ulysses sequence parallel padded non-divisible sequence lengths before attention. "
            f"Got query={original_q_len}, key={original_k_len}, "
            f"sequence_parallel_size={sequence_parallel_size}.",
        )
        q = _pad_sequence_length(q, padded_q_len)
        k = _pad_sequence_length(k, padded_k_len)
        v = _pad_sequence_length(v, padded_k_len)
        mask = _pad_attention_mask(mask, original_q_len, original_k_len, padded_q_len, padded_k_len)

    bsz, _, inner_dim = q.shape
    dim_head = inner_dim // heads
    local_heads = heads // sequence_parallel_size

    q_local = _local_sequence_shard(q, state.rank_in_group, sequence_parallel_size)
    k_local = _local_sequence_shard(k, state.rank_in_group, sequence_parallel_size)
    v_local = _local_sequence_shard(v, state.rank_in_group, sequence_parallel_size)

    q_dist = _sequence_to_head_parallel(q_local, heads, dim_head, state)
    k_dist = _sequence_to_head_parallel(k_local, heads, dim_head, state)
    v_dist = _sequence_to_head_parallel(v_local, heads, dim_head, state)

    out_dist = attention_callable(
        q_dist.reshape(bsz, -1, local_heads * dim_head),
        k_dist.reshape(bsz, -1, local_heads * dim_head),
        v_dist.reshape(bsz, -1, local_heads * dim_head),
        local_heads,
        mask,
    )
    out_local = _head_to_sequence_parallel(
        out_dist.reshape(bsz, -1, local_heads, dim_head),
        heads,
        dim_head,
        state,
    )
    gathered = _gather_sequence_shards(out_local.reshape(bsz, -1, heads * dim_head), state)
    return gathered[:, :original_q_len, :]


def _next_multiple(length: int, multiple: int) -> int:
    return ((length + multiple - 1) // multiple) * multiple


def _pad_sequence_length(tensor: torch.Tensor, padded_length: int) -> torch.Tensor:
    pad = padded_length - tensor.shape[1]
    if pad <= 0:
        return tensor

    pad_tensor = torch.zeros(
        tensor.shape[0],
        pad,
        tensor.shape[2],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, pad_tensor), dim=1)


def _pad_attention_mask(
    mask: torch.Tensor | None,
    original_q_len: int,
    original_k_len: int,
    padded_q_len: int,
    padded_k_len: int,
) -> torch.Tensor | None:
    if mask is None:
        padded_mask = torch.zeros(padded_q_len, padded_k_len, dtype=torch.float32)
        if padded_k_len > 0:
            padded_mask[:, :] = torch.finfo(torch.float32).min
            padded_mask[:original_q_len, :original_k_len] = 0
            if padded_q_len > original_q_len:
                padded_mask[original_q_len:, :original_k_len] = 0
        return padded_mask

    if original_q_len == padded_q_len and original_k_len == padded_k_len:
        return mask

    pad_shape = (*mask.shape[:-2], padded_q_len, padded_k_len)
    if torch.is_floating_point(mask):
        fill_value = torch.finfo(mask.dtype).min
    else:
        fill_value = False

    padded_mask = torch.full(
        pad_shape,
        fill_value=fill_value,
        dtype=mask.dtype,
        device=mask.device,
    )
    padded_mask[..., :original_q_len, :original_k_len] = mask

    if torch.is_floating_point(mask) and padded_q_len > original_q_len:
        padded_mask[..., original_q_len:, :original_k_len] = 0

    return padded_mask


def _local_sequence_shard(tensor: torch.Tensor, rank_in_group: int, sequence_parallel_size: int) -> torch.Tensor:
    shard_size = tensor.shape[1] // sequence_parallel_size
    start = rank_in_group * shard_size
    end = start + shard_size
    return tensor[:, start:end, :]


def _sequence_to_head_parallel(
    tensor: torch.Tensor,
    heads: int,
    dim_head: int,
    state: UlyssesSequenceParallelState,
) -> torch.Tensor:
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], heads, dim_head)
    input_chunks = list(tensor.chunk(state.sequence_parallel_size, dim=2))
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(state.sequence_parallel_size)]
    return torch.cat(list(dist_nn.all_to_all(output_chunks, input_chunks, group=state.process_group)), dim=1)


def _head_to_sequence_parallel(
    tensor: torch.Tensor,
    heads: int,
    dim_head: int,
    state: UlyssesSequenceParallelState,
) -> torch.Tensor:
    local_heads = heads // state.sequence_parallel_size
    tensor = tensor.reshape(tensor.shape[0], tensor.shape[1], local_heads, dim_head)
    input_chunks = list(tensor.chunk(state.sequence_parallel_size, dim=1))
    output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(state.sequence_parallel_size)]
    return torch.cat(list(dist_nn.all_to_all(output_chunks, input_chunks, group=state.process_group)), dim=2)


def _gather_sequence_shards(
    tensor: torch.Tensor,
    state: UlyssesSequenceParallelState,
) -> torch.Tensor:
    gathered = dist_nn.all_gather(tensor, group=state.process_group)
    return torch.cat(list(gathered), dim=1)


def _warn_once(key: str, message: str) -> None:
    if key in _warned_messages:
        return
    _warned_messages.add(key)
    warnings.warn(message, stacklevel=3)
