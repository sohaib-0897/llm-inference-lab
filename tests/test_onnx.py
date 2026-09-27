import os

import numpy as np
import pytest
import torch

from llm_lab.config import ModelConfig
from llm_lab.engine.onnx_engine import (
    HAS_ORT,
    ONNXInferenceEngine,
    export_to_onnx,
)
from llm_lab.models.transformer import CausalLM


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_onnx_export_and_inference(tmp_path):
    cfg = ModelConfig.nano()
    model = CausalLM(cfg).eval()

    onnx_file = str(tmp_path / "test_model.onnx")
    export_to_onnx(model, onnx_file, dummy_seq_len=8)

    assert os.path.exists(onnx_file)
    assert os.path.getsize(onnx_file) > 0

    engine = ONNXInferenceEngine(onnx_file)

    input_ids = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        pt_out = model(input_ids).numpy()

    ort_out = engine.forward(input_ids)

    assert ort_out.shape == pt_out.shape
    # Check max difference between PyTorch and ONNX Runtime
    diff = np.max(np.abs(pt_out - ort_out))
    assert diff < 1e-4, f"ONNX Runtime discrepancy too large: {diff}"
