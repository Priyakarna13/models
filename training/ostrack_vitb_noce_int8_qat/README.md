# OSTrack ViT-B no-CE INT8 QAT qualification

This directory contains the reproducible promotion tooling for the batch-1,
no-candidate-elimination OSTrack deployment graph.  It does not contain model
weights, datasets, compiler outputs, or signed download URLs.

`export_qat_catalog.py` reconstructs the exact QAT topology from an immutable
OSTrack source bundle, loads a selected checkpoint strictly, and emits the
complete target-facing quantization catalogue.  The catalogue is an input to
the ModelSDK Quantization Workspace promotion step; it is not itself a `.sima`
model or a compiled artifact.

The promotion flow must preserve these values through all four evidence
levels:

1. PyTorch fake-quantized execution.
2. ONNX/QDQ preview.
3. serialized ModelSDK `.sima` execution.
4. compiled MLA simulator and Modalix board execution.

Independent AFE recalibration is not an acceptable substitute for the selected
QAT catalogue.
