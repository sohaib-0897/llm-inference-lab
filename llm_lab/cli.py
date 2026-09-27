from __future__ import annotations

import argparse
import os
import sys

import psutil
import torch

from llm_lab.backends.base import GenerationRequest
from llm_lab.backends.ollama_backend import OllamaBackend
from llm_lab.benchmarks.ollama_benchmarks import OllamaBenchmarkSuite
from llm_lab.benchmarks.runner import BenchmarkRunner
from llm_lab.config import BenchmarkConfig, GenerationConfig, ModelConfig
from llm_lab.engine.generation import stream_generate
from llm_lab.models.transformer import CausalLM
from llm_lab.visualization.ollama_plotter import generate_ollama_plots
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


def cmd_ollama_info(args: argparse.Namespace) -> None:
    """Inspect local Ollama daemon, installed models, and active GPU memory offload."""
    backend = OllamaBackend(base_url=args.url)
    print("=" * 65)
    print("           OLLAMA RUNTIME & LOCAL MODEL INSPECTION")
    print("=" * 65)
    online = backend.is_available()
    print(f"Ollama Daemon Reachable: {online}")
    if not online:
        print(f"Error: Unable to connect to Ollama at {args.url}. Is the service running?")
        return

    ver = backend.get_runtime_version()
    print(f"Ollama Version         : {ver or 'unknown'}")
    print(f"Endpoint URL           : {args.url}")

    models = backend.list_models()
    print(f"\nInstalled Models ({len(models)} found):")
    for m in models:
        name = m.get("name", "unknown")
        size_mb = m.get("size", 0) / (1024 * 1024)
        details = m.get("details", {})
        param_sz = details.get("parameter_size", "unknown")
        q_lvl = details.get("quantization_level", "unknown")
        family = details.get("family", "unknown")
        print(f"  - {name:16s} | Size: {size_mb:7.1f} MB | Params: {param_sz:6s} | Quant: {q_lvl:8s} | Family: {family}")

    running = backend.get_running_models()
    print(f"\nActive In-Memory / GPU Models ({len(running)} loaded):")
    if not running:
        print("  (No model currently loaded in VRAM/RAM)")
    for r in running:
        r_name = r.get("name", "unknown")
        proc = r.get("processor", "unknown")
        vram_mb = r.get("size_vram", 0) / (1024 * 1024)
        total_sz = r.get("size", 0) / (1024 * 1024)
        ctx = r.get("context_length", "default")
        print(f"  - {r_name:16s} | Processor: {proc:10s} | VRAM: {vram_mb:6.1f} MB / {total_sz:6.1f} MB | Context: {ctx}")
    print("=" * 65)


def cmd_ollama_generate(args: argparse.Namespace) -> None:
    """Interactive streaming generation via local Ollama model."""
    backend = OllamaBackend(base_url=args.url)
    if not backend.is_available():
        print(f"Error: Cannot connect to Ollama at {args.url}")
        sys.exit(1)

    req = GenerationRequest(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=42 if args.deterministic else None,
    )

    print(f"\nModel: {args.model} | Prompt: \"{args.prompt}\"\n")
    res = backend.generate(req, model_name=args.model)

    print(res.text)
    print("\n" + "-" * 55)
    print(f"Client TTFT               : {res.client_ttft_ms:.2f} ms")
    print(f"Client Avg TPOT           : {res.client_avg_tpot_ms:.2f} ms")
    print(f"Client Total Latency      : {res.client_total_latency_ms:.2f} ms")
    print(f"Client Tokens/Sec         : {res.client_tokens_per_sec:.2f} tok/s")
    if res.server_prompt_eval_duration_ms:
        print(f"Server Prompt Eval Time   : {res.server_prompt_eval_duration_ms:.2f} ms ({res.server_prompt_eval_count} tokens)")
    if res.server_eval_duration_ms and res.server_eval_tokens_per_sec:
        print(f"Server Decode Speed       : {res.server_eval_tokens_per_sec:.2f} tok/s ({res.server_eval_count} tokens in {res.server_eval_duration_ms:.1f}ms)")
    print("-" * 55)


def cmd_ollama_benchmark(args: argparse.Namespace) -> None:
    """Run full Ollama benchmark experiments (Experiments A through E)."""
    backend = OllamaBackend(base_url=args.url)
    if not backend.is_available():
        print(f"Error: Cannot connect to Ollama at {args.url}")
        sys.exit(1)

    suite = OllamaBenchmarkSuite(backend=backend, output_dir=args.output_dir)
    suite.run_all(model_name=args.model)

    if args.plot:
        plots = generate_ollama_plots(args.output_dir)
        print("\nGenerated Ollama Benchmark Charts:")
        for p in plots:
            print(f"  - {p}")


def cmd_ollama_plot(args: argparse.Namespace) -> None:
    """Generate technical charts from existing Ollama benchmark JSON artifacts."""
    plots = generate_ollama_plots(args.results_dir)
    print("Ollama Benchmark Charts generated:")
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

    # Ollama Info
    p_o_info = subparsers.add_parser("ollama-info", help="Inspect Ollama daemon, models, and GPU offload")
    p_o_info.add_argument("--url", default="http://127.0.0.1:11434")
    p_o_info.set_defaults(func=cmd_ollama_info)

    # Ollama Generate
    p_o_gen = subparsers.add_parser("ollama-gen", help="Run real-model generation via Ollama")
    p_o_gen.add_argument("--model", default="qwen2.5:0.5b")
    p_o_gen.add_argument("--prompt", default="Why is KV cache important in LLM inference?")
    p_o_gen.add_argument("--max-tokens", type=int, default=48)
    p_o_gen.add_argument("--temperature", type=float, default=0.0)
    p_o_gen.add_argument("--top-k", type=int, default=50)
    p_o_gen.add_argument("--top-p", type=float, default=0.95)
    p_o_gen.add_argument("--deterministic", action="store_true", default=True)
    p_o_gen.add_argument("--url", default="http://127.0.0.1:11434")
    p_o_gen.set_defaults(func=cmd_ollama_generate)

    # Ollama Benchmark
    p_o_bench = subparsers.add_parser("ollama-bench", help="Run comprehensive real-model Ollama benchmark suite")
    p_o_bench.add_argument("--model", default="qwen2.5:0.5b")
    p_o_bench.add_argument("--output-dir", default="benchmarks/results/ollama")
    p_o_bench.add_argument("--plot", action="store_true", default=True)
    p_o_bench.add_argument("--url", default="http://127.0.0.1:11434")
    p_o_bench.set_defaults(func=cmd_ollama_benchmark)

    # Ollama Plot
    p_o_plot = subparsers.add_parser("ollama-plot", help="Generate charts from Ollama experiment JSON artifacts")
    p_o_plot.add_argument("--results-dir", default="benchmarks/results/ollama")
    p_o_plot.set_defaults(func=cmd_ollama_plot)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
