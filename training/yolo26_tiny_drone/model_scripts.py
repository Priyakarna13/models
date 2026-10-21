"""Use the model-owned deployment contract from training entry points."""

import sys
from pathlib import Path

scripts = Path(__file__).resolve().parents[2] / "models" / "yolo26_tiny_drone" / "scripts"
sys.path.append(str(scripts))

from contract import DEFAULT_CONTRACT, DEFAULT_MODEL_SIZE, INPUT_NAME, DeploymentContract  # noqa: E402,F401
from deployment import audit_mpk, audit_quantized_model, preprocess_rgb_image  # noqa: E402,F401
from onnx_audit import audit_onnx, canonicalize_attention_for_modelsdk, sha256_file  # noqa: E402,F401
