from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from triton_kernels import QuantizedInt4Weights, bf16_int4_matmul, quantize_fp16_to_int4


@dataclass
class QuantLinearConfig:
    block_m: int = 128
    block_n: int = 128
    block_k: int = 64
    num_warps: int = 4
    num_stages: int = 2


class QuantLinear(nn.Module):
    def __init__(
        self,
        quantized: QuantizedInt4Weights,
        bias: torch.Tensor | None = None,
        config: QuantLinearConfig | None = None,
    ) -> None:
        super().__init__()
        self.out_features, self.in_features = quantized.original_shape
        self.register_buffer("packed_weight", quantized.packed_uint8.clone())
        self.register_buffer("scales", quantized.scales.clone())
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16))
        else:
            self.bias = None  # type: ignore[assignment]
        self.config = config or QuantLinearConfig()

    @classmethod
    def from_linear(
        cls,
        module: nn.Linear,
        *,
        config: QuantLinearConfig | None = None,
    ) -> "QuantLinear":
        if not module.weight.is_cuda:
            raise ValueError("Linear weight must reside on CUDA device for quantization.")
        weight_fp16 = module.weight.data.to(torch.float16, copy=True)
        quantized = quantize_fp16_to_int4(weight_fp16)
        bias = module.bias.data.to(torch.bfloat16, copy=True) if module.bias is not None else None
        return cls(quantized, bias=bias, config=config)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        if orig_dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)

        orig_shape = x.shape
        x_flat = x.view(-1, self.in_features).contiguous()

        quant = QuantizedInt4Weights(
            packed_uint8=self.packed_weight,
            scales=self.scales,
            original_shape=(self.out_features, self.in_features),
        )
        out = bf16_int4_matmul(
            x_flat,
            quant,
            out_dtype=torch.bfloat16,
            block_m=self.config.block_m,
            block_n=self.config.block_n,
            block_k=self.config.block_k,
            num_warps=self.config.num_warps,
            num_stages=self.config.num_stages,
        )

        if self.bias is not None:
            out += self.bias

        out = out.view(*orig_shape[:-1], self.out_features)
        if orig_dtype != torch.bfloat16:
            out = out.to(orig_dtype)
        return out


def replace_linear_with_quantized(
    module: nn.Module,
    *,
    config: QuantLinearConfig | None = None,
) -> None:
    """
    In-place replacement of nn.Linear layers with QuantLinear.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, QuantLinear):
            continue
        if isinstance(child, nn.Linear):
            # if name == "lm_head":
            #     continue
            quant_lin = QuantLinear.from_linear(child, config=config)
            setattr(module, name, quant_lin)
        else:
            replace_linear_with_quantized(child, config=config)

