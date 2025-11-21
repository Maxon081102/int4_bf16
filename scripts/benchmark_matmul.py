import argparse
import time
import triton.testing
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


import torch
import numpy as np


import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from triton_kernels import bf16_int4_matmul, quantize_fp16_to_int4

@dataclass
class LayerSpec:
    name: str
    out_features: int
    in_features: int


def _bench_kernel(callable_fn, warmup: int, iters: int) -> float:
    times = triton.testing.do_bench(callable_fn, warmup=warmup, rep=iters, return_mode='all')
    return np.mean(times)


def _tensor_memory_mb(tensor: torch.Tensor) -> float:
    return tensor.element_size() * tensor.numel() / (1024 ** 2)


def _quantized_weights_memory_mb(quantized) -> float:
    packed_bytes = quantized.packed_uint8.element_size() * quantized.packed_uint8.numel()
    scale_bytes = quantized.scales.element_size() * quantized.scales.numel()
    return (packed_bytes + scale_bytes) / (1024 ** 2)


def run_benchmark(
    token_counts: Iterable[int],
    layer_specs: Dict[str, LayerSpec],
    warmup: int,
    iters: int,
    device: torch.device,
) -> Dict[Tuple[str, int], Dict[str, float]]:
    results: Dict[Tuple[str, int], Dict[str, float]] = {}

    for spec in layer_specs.values():
        weight = torch.randn(
            spec.out_features,
            spec.in_features,
            device=device,
            dtype=torch.float16,
        )
        quantized = quantize_fp16_to_int4(weight)
        weight_bf16 = weight.to(torch.bfloat16)
        weight_mem_mb = _tensor_memory_mb(weight)
        quantized_mem_mb = _quantized_weights_memory_mb(quantized)

        for tokens in token_counts:
            activations = torch.randn(
                tokens,
                spec.in_features,
                device=device,
                dtype=torch.bfloat16,
            )

            baseline_fn = lambda: torch.matmul(activations, weight_bf16.t())
            quant_fn = lambda: bf16_int4_matmul(activations, quantized, out_dtype=torch.bfloat16)

            baseline_time = _bench_kernel(baseline_fn, warmup, iters)
            quant_time = _bench_kernel(quant_fn, warmup, iters)

            with torch.no_grad():
                ref = baseline_fn()
                quant_out = quant_fn().to(torch.bfloat16)
                diff = (ref - quant_out).abs()
                error = diff.max().item()
                mean_error = diff.mean().item()

            results[(spec.name, tokens)] = {
                "baseline_ms": baseline_time * 1000,
                "quant_ms": quant_time * 1000,
                "speedup": baseline_time / quant_time if quant_time > 0 else float("inf"),
                "max_abs_err": error,
                "mean_abs_err": mean_error,
                "weight_mb": weight_mem_mb,
                "quantized_mb": quantized_mem_mb,
            }
    return results


def main():

    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Benchmark BF16 @ INT4 Triton matmul.")
    parser.add_argument(
        "--tokens",
        type=str,
        default="128,512,2048",
        help="Comma-separated token counts for activation matrices.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warm-up iterations per measurement.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Measured iterations per configuration.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError("CUDA device is required for benchmarking.")

    device = torch.device("cuda")
    token_counts = [int(v.strip()) for v in args.tokens.split(",") if v.strip()]

    layer_specs = {
        "attn_proj": LayerSpec("attn_proj", out_features=2048, in_features=2048),
        "ffn_up": LayerSpec("ffn_up", out_features=8192, in_features=2048),
        "ffn_down": LayerSpec("ffn_down", out_features=2048, in_features=8192),
        "lm_head": LayerSpec("lm_head", out_features=128256, in_features=2048),
    }

    results = run_benchmark(token_counts, layer_specs, args.warmup, args.iters, device)

    header = (
        f"{'Layer':<10} {'Tokens':>6} {'Baseline (ms)':>16} "
        f"{'Quant (ms)':>12} {'Speedup':>10} {'Max |err|':>12} {'Mean |err|':>12} "
        f"{'Weight (MB)':>12} {'Quant (MB)':>11}"
    )
    print(header)
    print("-" * len(header))

    layer_order = list(layer_specs.keys())
    for layer_name in layer_order:
        for tokens in token_counts:
            metrics = results[(layer_name, tokens)]
            print(
                f"{layer_name:<10} "
                f"{tokens:>6d} "
                f"{metrics['baseline_ms']:>16.4f} "
                f"{metrics['quant_ms']:>12.4f} "
                f"{metrics['speedup']:>9.3f}x "
                f"{metrics['max_abs_err']:>12.6f} "
                f"{metrics['mean_abs_err']:>12.6f} "
                f"{metrics['weight_mb']:>12.2f} "
                f"{metrics['quantized_mb']:>11.2f}"
            )


if __name__ == "__main__":
    main()

