from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

from llm_lab.models.transformer import CausalLM


class ONNXForwardWrapper(nn.Module):
    """Wrapper around CausalLM to export forward pass cleanly to ONNX."""

    def __init__(self, model: CausalLM) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # In ONNX export, we run forward pass without KV cache state
        return self.model(input_ids, start_pos=0, use_cache=False)


def export_to_onnx(
    model: CausalLM,
    output_path: str,
    dummy_seq_len: int = 16,
) -> str:
    """Exports a CausalLM instance to an ONNX model file with dynamic batch and sequence axes."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wrapper = ONNXForwardWrapper(model).eval()

    dummy_input = torch.randint(
        0,
        model.config.vocab_size,
        (1, dummy_seq_len),
        dtype=torch.long,
    )

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        dynamo=False,
    )
    return output_path


class ONNXInferenceEngine:
    """Inference engine powered by ONNX Runtime for CPU evaluation."""

    def __init__(self, onnx_model_path: str) -> None:
        if not HAS_ORT:
            raise RuntimeError("onnxruntime is not installed.")
        self.model_path = onnx_model_path

        # Setup ONNX Runtime Session Options
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def forward(self, input_ids: np.ndarray | torch.Tensor) -> np.ndarray:
        """Run forward inference pass through ONNX Runtime."""
        if isinstance(input_ids, torch.Tensor):
            input_ids_np = input_ids.detach().cpu().numpy().astype(np.int64)
        else:
            input_ids_np = input_ids.astype(np.int64)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_ids_np},
        )
        return np.asarray(outputs[0])

    def benchmark_forward(
        self,
        batch_size: int = 1,
        seq_len: int = 32,
        num_runs: int = 10,
    ) -> dict[str, float]:
        """Profile forward pass latency with ONNX Runtime."""
        dummy_inputs = np.random.randint(0, 100, (batch_size, seq_len), dtype=np.int64)

        # Warmup
        for _ in range(2):
            self.forward(dummy_inputs)

        latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            self.forward(dummy_inputs)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
        }
