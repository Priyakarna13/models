# ResNet-50

This entry pins the ONNX Community conversion of
[`microsoft/resnet-50`](https://huggingface.co/microsoft/resnet-50) at a full
Hugging Face repository commit and verifies the 102 MB ONNX file by SHA-256.

```bash
model-registry validate models/resnet_50
model-registry fetch models/resnet_50
```

The first compile variant targets Modalix BF16 with the pinned Model Compiler
2.1.3 develop snapshot. BF16 does not use representative INT8 calibration; the
Model Compiler API is given a deterministic zero-valued sample only because its
quantization entrypoint requires an iterable. `compile.py` loads the verified
ONNX, applies BF16 quantization and direct-MLA tessellation, and emits one
directly consumable MPK archive.

An INT8 variant should be added only after an immutable,
redistribution-approved representative calibration set has been pinned.

The preprocessing values are pinned from the upstream conversion's
`preprocessor_config.json`. No model-specific download script is needed.
