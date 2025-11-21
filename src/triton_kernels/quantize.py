from dataclasses import dataclass
from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _row_absmax_kernel(
    x_ptr,
    row_max_ptr,
    stride_xm,
    stride_xn,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    cols = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols

    x = tl.load(
        x_ptr + row_idx * stride_xm + cols * stride_xn,
        mask=mask,
        other=0.0,
    )
    x = tl.abs(x.to(tl.float32))
    block_max = tl.max(x, axis=0)

    tl.atomic_max(row_max_ptr + row_idx, block_max)


@triton.jit
def _quantize_pack_kernel(
    x_ptr,
    q_ptr,
    scale_ptr,
    stride_xm,
    stride_xn,
    stride_qm,
    stride_qn,
    n_cols,
    BLOCK_PAIRS: tl.constexpr,
):
    row_idx = tl.program_id(0)
    pair_block_idx = tl.program_id(1)

    pair_offsets = pair_block_idx * BLOCK_PAIRS + tl.arange(0, BLOCK_PAIRS)
    col_even = pair_offsets * 2
    col_odd = col_even + 1

    mask_even = col_even < n_cols
    mask_odd = col_odd < n_cols

    x_even = tl.load(
        x_ptr + row_idx * stride_xm + col_even * stride_xn,
        mask=mask_even,
        other=0.0,
    ).to(tl.float32)
    x_odd = tl.load(
        x_ptr + row_idx * stride_xm + col_odd * stride_xn,
        mask=mask_odd,
        other=0.0,
    ).to(tl.float32)

    scale = tl.load(scale_ptr + row_idx)
    inv_scale = 1.0 / tl.maximum(scale, 1e-8)

    scaled_even = x_even * inv_scale
    scaled_odd = x_odd * inv_scale

    q_even = tl.where(
        scaled_even >= 0,
        scaled_even + 0.5,
        scaled_even - 0.5,
    )
    q_odd = tl.where(
        scaled_odd >= 0,
        scaled_odd + 0.5,
        scaled_odd - 0.5,
    )
    q_even = tl.clamp(q_even, -8, 7).to(tl.int32)
    q_odd = tl.clamp(q_odd, -8, 7).to(tl.int32)

    q_even = q_even & 0xF
    q_odd = q_odd & 0xF

    packed = q_even | (q_odd << 4)
    packed = packed.to(tl.uint8)

    tl.store(
        q_ptr + row_idx * stride_qm + pair_offsets * stride_qn,
        packed,
        mask=mask_even,
    )


@dataclass
class QuantizedInt4Weights:
    packed_uint8: torch.Tensor
    scales: torch.Tensor
    original_shape: Tuple[int, int]

    def dequantize(self, out_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        return dequantize_int4(
            self.packed_uint8,
            self.scales,
            self.original_shape[1],
            out_dtype=out_dtype,
        )


def quantize_fp16_to_int4(
    weights: torch.Tensor,
    block_size: int = 64,
    max_pairs_per_block: int = 256,
    eps: float = 1e-8,
) -> QuantizedInt4Weights:
    weights = weights.contiguous()
    M, N = weights.shape

    row_max = torch.zeros((M,), dtype=torch.float32, device=weights.device)
    grid_absmax = (M, triton.cdiv(N, block_size))
    _row_absmax_kernel[grid_absmax](
        weights,
        row_max,
        weights.stride(0),
        weights.stride(1),
        N,
        BLOCK_SIZE=block_size,
    )
    scales = row_max / 7.0
    min_scale = torch.full_like(scales, eps)
    scales = torch.maximum(scales, min_scale)
    scales = scales.contiguous()

    packed_cols = triton.cdiv(N, 2)
    q_storage = torch.empty((M, packed_cols), dtype=torch.uint8, device=weights.device)

    grid_quant = (M, triton.cdiv(packed_cols, max_pairs_per_block))
    _quantize_pack_kernel[grid_quant](
        weights,
        q_storage,
        scales,
        weights.stride(0),
        weights.stride(1),
        q_storage.stride(0),
        q_storage.stride(1),
        N,
        BLOCK_PAIRS=max_pairs_per_block,
    )

    return QuantizedInt4Weights(
        packed_uint8=q_storage,
        scales=scales,
        original_shape=(M, N),
    )


def dequantize_int4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    original_cols: int | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    M = packed.shape[0]
    inferred_cols = packed.shape[1] * 2
    N = inferred_cols if original_cols is None else original_cols
    
    low = packed & 0xF
    high = (packed >> 4) & 0xF

    stacked = torch.stack((low, high), dim=-1).view(M, -1)
    stacked = stacked[:, :N]

    signed = stacked.to(torch.int8)
    signed = torch.where(signed <= 7, signed, signed - 16).to(torch.float32)

    values = signed * scales.view(-1, 1).to(torch.float32)

    return values.to(out_dtype)

