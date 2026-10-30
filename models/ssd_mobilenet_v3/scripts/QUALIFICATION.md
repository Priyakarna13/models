# SSD-MobileNet V3

TorchVision SSDLite320 MobileNet V3 Large, using its original COCO FP32 weights.
The INT8 profile is a **new PTQ build**, not the historical QAT export.

## Source and input contract

`model.yaml` pins the official checkpoint by full SHA-256 and byte count. The
standard **Vulcan Model Convert** workflow published the converted ONNX in run
[40979330247](https://github.com/sima-neat/models/actions/runs/40979330247).
The published ARM export has the same complete graph and initializer bytes as
the paired FP32 source validation; only export-environment metadata differs.

Decode RGB, divide by 255, then normalize with mean and standard deviation
`[0.5, 0.5, 0.5]` **before** bilinear resizing to 320×320. Resizing uses half-pixel
coordinates, `align_corners=False`, and no antialiasing. Do not reuse the V1/V2
TensorFlow asymmetric 300×300 resize or the historical ImageNet-statistics label.

The FP32 source exposes twelve NCHW heads: six location heads, then six class
heads, at grids 20, 10, 5, 3, 2 and 1. Six anchors per cell yield 3,234 anchors;
class heads contain 91 logits including background. The accuracy oracle uses
TorchVision anchors and box coding, softmax, score threshold 0.001, per-class
pre-NMS top-300, IoU threshold 0.55 and at most 300 final detections. COCO scoring
retains its standard maxDets of 100.

## Compilation and accuracy

Both profiles use the repository-pinned Model Compiler 3.0. The `w16a16` profile
uses BF16. The `w8a8` profile uses HistogramMSE with 2,048 bins and 128 images
selected reproducibly from the pinned Dataset Registry calibration archive.
Calibration never draws from the validation split. Source conversion and
calibration retain the original input normalization; no hidden affine rewrite
or QAT checkpoint is used.

The accuracy expectations use the measured **original FP32 source** COCO AP of
0.2126872437 with the repository's unchanged 5% relative-drop policy. They are
acceptance gates for both candidate profiles, **not measured native or quantized
accuracy baselines**. The pinned validation archive's 5,000 images and annotation
bytes were checked against the source evaluation inputs.

Accuracy preprocessing currently requires the pinned CPU Torch and Pillow
packages in `scripts/requirements-accuracy.txt`. Substituting a lighter resize
implementation requires a separate source-equivalence check.

## Qualification status

The source export and paired full-COCO source evaluation are qualified. Native
compiled accuracy and whole-model performance remain pending. There is no
performance declaration or invented FPS expectation; `scripts/performance.py`
fails explicitly rather than publishing MLA-only throughput as model speed.
The native postprocessing route will be enabled only after its board smoke test.

The upstream TorchVision BSD-3-Clause license is retained in `scripts/LICENSE.torchvision`.
