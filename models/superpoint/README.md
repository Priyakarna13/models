# SuperPoint 480x640

This entry builds the grayscale SuperPoint dense heads used by Neat BoxDecode.
The source and calibration files are private release assets and are verified by
byte size and SHA-256 before use. Source redistribution is restricted.

## Graph contract

`models/superpoint/surgery.py` accepts the engineering export, renames its
single input to `image`, fixes the shape to `[1, 1, 480, 640]`, and cuts the
graph at:

- `/convPb/Conv_output_0`: `[1, 65, 60, 80]` score logits
- `/convDb/Conv_output_0`: `[1, 256, 60, 80]` raw descriptors

The cut removes descriptor normalization and leaves softmax, dustbin removal,
NMS, Top-K, descriptor normalization, and descriptor sampling in the A65
`A65V1` SuperPoint BoxDecode profile. The qualified source has the same 24
initializer tensors as the engineering dense-core reference.

## Calibration

INT8 quantization uses a deterministic 128-image snapshot selected for natural
scene coverage and local-feature content:

- 80 landscape or square COCO val2017 images with an Attribution license
- 32 images from distinct HPatches illumination and viewpoint sequences
- 16 frames from four TUM RGB-D indoor motion and texture sequences

Every image is center-cropped to 4:3, resized to 640x480, converted with the
deployment BT.601 fixed-point grayscale formula, and normalized to FP32 `[0, 1]`.
The immutable NPZ is fetched by the registry. `calibration.py` fails closed on
the archive, tensor, dtype, shape, and range. `calibration_manifest.json` binds
the ordered samples, source hashes, licenses, preprocessing, and diversity
statistics.

- calibration NPZ SHA-256: `7361a0089fa8391436f2cfc4f1febb69c248ca332e8d8a63957e25b4bb6e65f2`
- calibration array SHA-256: `260d9b7b7af937d7ae2d2edb6774f4e4f60af9438dc01781444fb901f0670f58`
- manifest SHA-256: `9c865cca0fcbeb7a82efd979aabfb4b8f8eeb852a1cc2822b57cd272d27372d8`

The snapshot broadens the observed brightness, contrast, edges, and scene
content relative to the previous single-sequence TUM calibration. It is still a
calibration set, not accuracy evidence; promotion requires evaluation on a
separate validation set.

## Variants

All variants use strict W8A8 MLA compute and are compiled with the repository
Model Compiler pin.

| Variant | Input | MLA/output boundary |
| --- | --- | --- |
| `modalix_int8_tessellation_ev74` | FP32 | INT8; EV74 owns tessellation and detessellation |
| `modalix_int8_tessellation_mla` | FP32 | INT8; MLA owns tessellation and detessellation |
| `modalix_bf16_tessellation_ev74` | BF16 | BF16; EV74 owns tessellation and detessellation |
| `modalix_bf16_tessellation_mla` | BF16 | BF16; MLA owns tessellation and detessellation |
| `modalix_fp32_dense` | BF16 | Dense FP32 heads |

The MPK audit requires exactly one MLA plugin, exactly one ELF, and no `.so`
files. EV74 boundary transforms are allowed. The INT8 package intentionally
retains the qualified compiler topology: the MLA heads are INT8, while the
terminal multi-output publication is FP32 after EV74 dequantization.

## Validate and compile

Private source fetches require a GitHub token with read access to this
repository.

```bash
export GITHUB_TOKEN="$(gh auth token)"
model-registry validate models/superpoint
model-registry fetch models/superpoint

python models/superpoint/compile.py \
  --model ~/.cache/model-registry/sha256/81/81e38c7886f13b7448c6b844acf7a60bda8178e4a66df13d582fd729bbcd6b8d/500672818 \
  --output-dir build/superpoint/modalix_int8_tessellation_mla \
  --variant modalix_int8_tessellation_mla
```

Vulcan activates the pinned Model Compiler environment and builds each variant
in an isolated job. After a successful staging run, install only this model:

```bash
sima-cli neat install --stg models/superpoint@develop:latest --install-dir ./compiled-models
```

The model request and provenance record are tracked in
[sima-neat/models#24](https://github.com/sima-neat/models/issues/24).
