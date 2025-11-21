import torch
import triton
import triton.language as tl


_MATMUL_AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=16,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
        num_warps=4,
        num_stages=3,
    ),
]

from .quantize import QuantizedInt4Weights


@triton.autotune(configs=_MATMUL_AUTOTUNE_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _bf16_int4_matmul_kernel(
    a_ptr,
    w_ptr,
    scale_ptr,
    c_ptr,
    stride_am,
    stride_ak,
    stride_wm,
    stride_wk,
    stride_cm,
    stride_cn,
    stride_scale,
    M,
    N,
    K,
    K_PAIRS,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    scales = tl.load(scale_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    scales = tl.where(mask_n, scales, 0.0)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    pair_block = BLOCK_K // 2
    pair_range = tl.arange(0, BLOCK_K // 2)
    for k_start in range(0, K, BLOCK_K):
        pair_offsets = (k_start // 2) + pair_range
        mask_pairs = pair_offsets < K_PAIRS

        w_ptrs = w_ptr + offs_n[:, None] * stride_wm + pair_offsets[None, :] * stride_wk
        packed = tl.load(w_ptrs, mask=mask_n[:, None] & mask_pairs[None, :], other=0).to(tl.int32)

        low = packed & 0xF
        high = (packed >> 4) & 0xF

        res = tl.interleave(low, high)
        res = tl.where(res <= 7, res, res - 16).to(tl.float32)
        res = res * scales[:, None]

        k_offsets = tl.arange(0, BLOCK_K)
        mask_k = (k_start + k_offsets) < K
        a_ptrs = (
            a_ptr
            + offs_m[:, None] * stride_am
            + (k_start + k_offsets)[None, :] * stride_ak
        )
        a_block = tl.load(
            a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0
        ).to(tl.float32)

        b_even = tl.trans(res)

        acc += tl.dot(a_block, b_even)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


def bf16_int4_matmul(
    a: torch.Tensor,
    quantized: QuantizedInt4Weights,
    *,
    out_dtype: torch.dtype = torch.float32,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
    autotune: bool = True,
) -> torch.Tensor:
    a = a.contiguous()
    M, K = a.shape

    out_features, in_features = quantized.original_shape

    packed = quantized.packed_uint8
    scales = quantized.scales

    packed = packed.contiguous()
    scales = scales.contiguous()

    N = out_features
    out = torch.empty((M, N), dtype=torch.float32, device=a.device)

    def grid(meta):
        return (
            triton.cdiv(M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )

    if autotune:
        _bf16_int4_matmul_kernel[grid](
            a,
            packed,
            scales,
            out,
            a.stride(0),
            a.stride(1),
            packed.stride(0),
            packed.stride(1),
            out.stride(0),
            out.stride(1),
            scales.stride(0),
            M,
            N,
            K,
            packed.shape[1],
        )
    else:
        _bf16_int4_matmul_kernel[grid](
            a,
            packed,
            scales,
            out,
            a.stride(0),
            a.stride(1),
            packed.stride(0),
            packed.stride(1),
            out.stride(0),
            out.stride(1),
            scales.stride(0),
            M,
            N,
            K,
            packed.shape[1],
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    if out_dtype != torch.float32:
        out = out.to(out_dtype)
    return out

