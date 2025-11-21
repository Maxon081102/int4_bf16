import argparse
import gc
import time
from typing import Dict, List, Tuple

import torch
import numpy as np
import triton.testing
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from quantized import QuantLinearConfig, replace_linear_with_quantized


def prepare_wikitext(tokenizer, block_size: int) -> List[torch.Tensor]:
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    all_ids: List[int] = []
    for text in dataset["text"]:
        tokens = tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        all_ids.extend(tokens["input_ids"])

    total_length = (len(all_ids) // block_size) * block_size

    tensor = torch.tensor(all_ids[:total_length], dtype=torch.long)
    tensor = tensor.view(-1, block_size)
    return [row.clone() for row in tensor]


def compute_perplexity(model, inputs: List[torch.Tensor], device: torch.device) -> float:
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in inputs:
            input_ids = batch.to(device)
            outputs = model(input_ids=input_ids.unsqueeze(0), labels=input_ids.unsqueeze(0))
            neg_log_likelihood = outputs.loss.detach() * input_ids.numel()
            total_log_likelihood += neg_log_likelihood.item()
            total_tokens += input_ids.numel()

    return torch.exp(torch.tensor(total_log_likelihood / total_tokens)).item()


def measure_speed(
    model,
    tokenizer,
    sequence_lengths: List[int],
    device: torch.device,
    warmup: int = 5,
    iters: int = 20,
) -> Dict[int, float]:
    results: Dict[int, float] = {}
    model.eval()

    for seq_len in sequence_lengths:
        prompt = torch.full((1, seq_len), tokenizer.eos_token_id, device=device, dtype=torch.long)
        times = triton.testing.do_bench(lambda: model(input_ids=prompt), warmup=warmup, rep=iters, return_mode='all')
        results[seq_len] = np.mean(times)
    return results


def load_model(model_name: str, quantized: bool, config: QuantLinearConfig) -> Tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quantized:
        replace_linear_with_quantized(model, config=config)

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate quantized Llama on WikiText-2.")
    parser.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    parser.add_argument("--block-size", type=int, default=4096)
    parser.add_argument("--seq-lengths", type=str, default="4096")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemError("CUDA device required for evaluation.")

    device = torch.device("cuda")
    seq_lengths = [int(v.strip()) for v in args.seq_lengths.split(",") if v.strip()]

    tokenizer_ref = AutoTokenizer.from_pretrained(args.model)
    if tokenizer_ref.pad_token_id is None:
        tokenizer_ref.pad_token = tokenizer_ref.eos_token
    eval_inputs = prepare_wikitext(tokenizer_ref, args.block_size)

    config = QuantLinearConfig()

    results = {}

    if not args.skip_baseline:
        model, tokenizer = load_model(args.model, quantized=False, config=config)
        with torch.no_grad():
            ppl = compute_perplexity(model, eval_inputs, device)
            speed = measure_speed(model, tokenizer, seq_lengths, device, args.warmup, args.iters)
        results["baseline"] = {"perplexity": ppl, "latency_ms": speed}
        del model
        gc.collect()
        torch.cuda.empty_cache()

    model, tokenizer = load_model(args.model, quantized=True, config=config)
    with torch.no_grad():
        ppl = compute_perplexity(model, eval_inputs, device)
        speed = measure_speed(model, tokenizer, seq_lengths, device, args.warmup, args.iters)
    results["quantized"] = {"perplexity": ppl, "latency_ms": speed}

    print("Evaluation Results:")
    for key, metrics in results.items():
        print(f"{key}: perplexity={metrics['perplexity']:.4f}")
        for seq_len, latency in metrics["latency_ms"].items():
            print(f"  seq {seq_len}: {latency:.2f} ms")


if __name__ == "__main__":
    main()

