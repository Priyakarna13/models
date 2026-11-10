# SSD-MobileNet V3 qualification

This maintainer guide records the source contract and remaining release gates;
it is not a declaration that the current native route is production-qualified.
The candidate uses TorchVision's original COCO FP32 weights. Its INT8 profile is
a new post-training-quantized build, not recovery of the historical QAT export.

## Source contract

`model.yaml` pins the original checkpoint by complete SHA-256 and byte count.
The standard conversion workflow exports the full TorchVision forward pass,
including preprocessing, decoding and suppression, to ONNX with `boxes`,
`scores` and `labels` outputs. `compilation.source: conversion` selects that
receipt-bound output; it does not pin the historical raw-head ONNX export.

Conversion records its pinned tool versions and preserves 98 convolutions and
70 BatchNorm operators. `surgery.py` owns the normalized-input/raw-head cut and
BatchNorm folding, producing the ordered twelve-head compiler interface.
The current full ONNX and prepared raw heads have full 5,000-image source-parity
evidence against the original checkpoint; no original-paper weight identity
is claimed.

The original input contract normalizes RGB to [-1, 1] before 320×320 half-pixel
bilinear resizing with no antialiasing. Source conversion and calibration keep
that contract; the native image route is separately evaluated, not asserted
bit-equivalent. The reference decoder uses 3,234 TorchVision anchors, 91-class
softmax including background, a score floor of 0.001, per-class pre-NMS top-300,
IoU 0.55 and final top-300. COCO retains maxDets [1, 10, 100].

## Data and compilation

Both profiles use the repository-pinned Model Compiler 3.0. INT8 uses the pinned
Dataset Registry calibration archive, which contains 1,000 cal2017 images. Its
recipe selects 128 deterministically and uses 2,048 histogram bins. BF16 derives
no ranges from representative images.
The canonical calibration does not include scored COCO val2017 IDs. Dataset
identity, the sampling recipe, compiler identity and profile parameters belong
to the compile fingerprint; changing them creates a different candidate.

## Native entry points and remaining gates

`accuracy.py` now consumes original RGB images with native preprocessing and
strict typed-box parsing and explicit `Ssd` selection with all 91
encoded classes. Core selects the source-multiclass policy for the supported 91-class SSD
signature. It preserves foreground anchor/class pairs and float classwise
suppression. Other SSD signatures retain the legacy policy; the supported
91-class route intentionally changes the former argmax/integer-NMS behavior.
It retains independent NumPy reference functions;
Torch and Pillow are needed by the source conversion/calibration environment,
not by the native board accuracy entry point. Board accuracy requirements are
listed in `requirements-accuracy.txt`.

The measurements below are historical local qualification results from before
the current conversion/surgery split, not acceptance of the current artifacts.
That matched runtime passed the canonical Python accuracy stages on all
5,000 validation images for both V3 profiles: BF16 mAP was 0.2118402269 and INT8
mAP was 0.2035257312. All twelve COCO metrics passed for each profile, including
the unchanged source mAP reference of 0.2126872437 with the repository's 5%
relative-drop policy. All six artifacts built with Model Compiler 3.0 passed accuracy and
performance across V1, V2 and V3. These are local qualification results, not
published develop baselines.

The eleven additional COCO metric expectations in `tests/tests.yaml` preserve
the earlier full-data native results; the original source mAP expectation is
unchanged. Native preprocessing and integer box publication can differ from
the floating-point source, so do not attribute the combined difference solely
to quantization.

The performance candidate is declared together with its public entry point and
finite per-profile regression references. Those initial references are 681.4 FPS
(BF16) and 1778.4 FPS (INT8), rounded from earlier native whole-detector C++
characterization, **not** MLA-only timing or canonical Python-stage acceptance.
The existing shared relative-drop policy and historical throughput references
remain unchanged. In addition, each complete INT8 detector must reach at least
1,700 FPS; the performance stage tests enforce this absolute floor without the
shared regression tolerance. Source pins and the original mAP gate are unchanged.

The customer workload is a seeded real COCO image in a 1280×720 RGB frame,
native preprocessing and source-multiclass SSD decode, threshold 0.2, IoU 0.6
and top-100. It runs through the public `performance.py` entry point and the
shared `scripts/run_model_performance.sh`, with real Sentinel telemetry,
execution plans, stage statistics and MLA profiling. The historical optimized
runtime completed the canonical performance workflow at 683.91 FPS (BF16) and
1,835.97 FPS (INT8), including genuine sampled power. Both the shared regression
gates and the additional 1,700 FPS INT8 floor passed without relaxed tolerances.
The matched V1 and V2 INT8 results were 1,808.50 and 1,711.10 FPS. V2 has only
11.10 FPS of margin above the floor; this qualification is not a repeatability
claim or a guarantee for every image.

Those historical full-data accuracy and performance stages used the same MPK
per profile and the same Core/Runner/Scheduler/Kernels cohort. The canonical
performance reports include whole-pipeline measurements, execution plans and separate MLA diagnostic
profiling; the MLA diagnostics are not loaded PRE/MLA/POST attribution. Local
validation does not replace publication of matching dependencies and models.
Do not replace canonical evidence with direct calls to `benchmark()` or external
C++ timing, or silently lower expectations following a regression.

## Immutable source hosting

The original 14,069,355-byte TorchVision checkpoint is now published through the
approved artifact-store conversion route. `model.yaml` pins the actual immutable
`source.pth` object from conversion run41255127615, with SHA-256
`a79551df90c79834bcd3bb3845ef9d966b5449a3a9b2833ae8404778ca5d65d2`.
The official TorchVision URL is preserved in `upstream.derived_from`; the BSD-3-Clause
notice remains in `LICENSE.torchvision`. The original publication verified object versions,
checksums, sizes and fresh HTTP readback for both the source and its provenance.
An independent download verified the same bytes after publication.

The conversion also reproduced the existing ONNX digest
`bdd8f4ef575bdc0cb02d33e571992d3c45c63f0bc00aeca5f51583c50e811bd8`.
This digest remains the historical raw-head comparator, not the current conversion output.
The current standard conversion exports the full TorchVision forward pass from the
same pinned checkpoint. `surgery.py` then cuts normalized input/raw heads and folds
BatchNorm with compiler-declared ONNX tooling. The input/output numerical contract
and calibration settings are unchanged, but the conversion/compile fingerprints
change. Fresh source parity, six-profile compilation and native qualification are
required before calling the new route production-ready; historical results above
are not fresh results for this source.
The converter accepts the mirrored transport only with the same full checkpoint
identity and original provenance; it does not silently substitute other weights.

The original mirror is already published; repeating conversion consumes that
verified source without publishing another checkpoint. Conversion outputs use the
shared registry publisher and its resolved environment, not a model-specific
publication route. No Hugging Face repository or revision is fabricated or required.
