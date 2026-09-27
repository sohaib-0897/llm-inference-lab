from __future__ import annotations

import argparse
import os
import sys

import psutil
import torch

from llm_lab.benchmarks.runner import BenchmarkRunner
from llm_lab.config import BenchmarkConfig, GenerationConfig, ModelConfig
from llm_lab.engine.generation import stream_generate
from llm_lab.models.transformer import CausalLM
from llm_lab.visualization.plotter import generate_benchmark_plots


def cmd_info(args: argparse.Namespace) -> None:
    """Print system hardware, PyTorch configuration, and model presets."""
    print("=" * 60)
    print("           LLM INFERENCE LAB - SYSTEM INSPECTION")
    print("=" * 60)
    print(f"Python Version    : {sys.version.split()[0]}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name       : {torch.cuda.get_device_name(0)}")
        print(f"Device Count      : {torch.cuda.device_count()}")
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        print(f"Total VRAM        : {vram_mb:.0f} MB")
    else:
        print("Device in Use     : CPU")

    cpu_count = psutil.cpu_count(logical=False)
    logical_count = psutil.cpu_count(logical=True)
    ram = psutil.virtual_memory()
    print(f"CPU Physical Cores: {cpu_count}")
    print(f"CPU Logical Cores : {logical_count}")
    print(f"Total System RAM  : {ram.total / (1024**3):.2f} GB")
    print(f"Available RAM     : {ram.available / (1024**3):.2f} GB")
    print("-" * 60)
    print("Model Presets:")
    for preset, fn in [("nano", ModelConfig.nano), ("micro", ModelConfig.micro), ("small", ModelConfig.small)]:
        cfg = fn()
        m = CausalLM(cfg)
        params = m.count_parameters()["total"]
        print(f"  - {preset:8s}: {params / 1e6:6.2f}M parameters | d_model={cfg.d_model} | layers={cfg.n_layers} | heads={cfg.n_heads}")
    print("=" * 60)


def cmd_generate(args: argparse.Namespace) -> None:
    """Run interactive or automated token generation."""
    print(f"Initializing {args.preset} model...")
    cfg = getattr(ModelConfig, args.preset)()
    model = CausalLM(cfg).to(args.device).eval()

    prompt_len = args.prompt_len
    input_ids = torch.randint(0, cfg.vocab_size, (1, prompt_len), device=args.device)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.sample,
        use_cache=not args.no_cache,
    )

    print(f"\nPrompt ({prompt_len} tokens) -> Generating {args.max_tokens} tokens (use_cache={gen_cfg.use_cache}):\n")
    generated_count = 0
    tpot_list = []
    ttft = 0.0

    for tok, step_time, is_prefill in stream_generate(model, input_ids, gen_cfg):
        if is_prefill:
            ttft = step_time
            print(f"[Prefill TTFT: {step_time*1000:.1f}ms] Token: {tok.item()}", end="", flush=True)
        else:
            tpot_list.append(step_time)
            print(f" {tok.item()}", end="", flush=True)
        generated_count += 1

    print("\n\n" + "-" * 50)
    avg_tpot = (sum(tpot_list) / len(tpot_list)) if tpot_list else 0.0
    print(f"TTFT (Prefill Latency) : {ttft*1000:.2f} ms")
    print(f"Avg TPOT (Decode Step) : {avg_tpot*1000:.2f} ms")
    print(f"Throughput             : {generated_count / (ttft + sum(tpot_list)):.2f} tokens/sec")
    print("-" * 50)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Run comprehensive benchmark suite and export metrics."""
    bench_cfg = BenchmarkConfig(
        suite_name="llm_inference_suite",
        device=args.device,
        batch_sizes=[1, 2, 4],
        prompt_lengths=[16, 32, 64],
        generate_lengths=[16, 32, 64],
        warmup_steps=args.warmup,
        num_trials=args.trials,
        output_dir=args.output_dir,
    )

    runner = BenchmarkRunner(bench_cfg)
    runner.run_all(model_preset=args.preset)

    if args.plot:
        json_path = os.path.join(args.output_dir, "benchmark_results.json")
        plots = generate_benchmark_plots(json_path, args.output_dir)
        print("Generated Plots:")
        for p in plots:
            print(f"  - {p}")


def cmd_plot(args: argparse.Namespace) -> None:
    """Generate plots from existing benchmark results."""
    json_path = os.path.join(args.results_dir, "benchmark_results.json")
    if not os.path.exists(json_path):
        print(f"Error: {json_path} does not exist. Run benchmark first.")
        sys.exit(1)

    plots = generate_benchmark_plots(json_path, args.results_dir)
    print("Charts generated:")
    for p in plots:
        print(f"  - {p}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM Inference Lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Info
    p_info = subparsers.add_parser("info", help="Inspect system and model configurations")
    p_info.set_defaults(func=cmd_info)

    # Generate
    p_gen = subparsers.add_parser("generate", help="Run token generation with streaming stats")
    p_gen.add_argument("--preset", default="nano", choices=["nano", "micro", "small"])
    p_gen.add_argument("--prompt-len", type=int, default=16)
    p_gen.add_argument("--max-tokens", type=int, default=32)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--top-k", type=int, default=50)
    p_gen.add_argument("--top-p", type=float, default=0.95)
    p_gen.add_argument("--repetition-penalty", type=float, default=1.0)
    p_gen.add_argument("--sample", action="store_true")
    p_gen.add_argument("--no-cache", action="store_true")
    p_gen.add_argument("--device", default="cpu")
    p_gen.set_defaults(func=cmd_generate)

    # Benchmark
    p_bench = subparsers.add_parser("benchmark", help="Run full benchmark suite")
    p_bench.add_argument("--preset", default="nano", choices=["nano", "micro", "small"])
    p_bench.add_argument("--device", default="cpu")
    p_bench.add_argument("--warmup", type=int, default=1)
    p_bench.add_argument("--trials", type=int, default=3)
    p_bench.add_argument("--output-dir", default="benchmarks/results")
    p_bench.add_argument("--plot", action="store_true", default=True)
    p_bench.set_defaults(func=cmd_benchmark)

    # Plot
    p_plot = subparsers.add_parser("plot", help="Generate benchmark charts from results")
    p_plot.add_argument("--results-dir", default="benchmarks/results")
    p_plot.set_defaults(func=cmd_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
