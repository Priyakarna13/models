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

INT8 quantization uses 128 frames from the TUM RGB-D `freiburg1_desk` RGB
movie. The immutable NPZ is fetched by the registry. `calibration.py` documents
the complete regeneration path and fails closed on these values:

- source AVI SHA-256: `3faa0fa667fa95d28ecf0d231c3383ddef8695f391f07d48f484af39de43a854`
- decoded BGR stream SHA-256: `8398d4611395efdbea5c9e7086aa86e5012abe58e17627a56b8d664c2f1e0437`
- calibration array SHA-256: `3f9968092295f95e98d2429611f5b87216e7fbbf0fc29678272462228ca4ec40`
- held-out array SHA-256: `64c4ed7a99c1f47a8e09fe4a94674999a170b59b7aa5ce60e2413a01127eab41`

The TUM sequence is attributed to the TUM RGB-D benchmark and is licensed
under CC BY 4.0. Only the derived calibration array is used by CI.

## Variants

All variants use strict W8A8 MLA compute and are compiled with the repository
Model Compiler pin.

| Variant | Input | MLA/output boundary |
| --- | --- | --- |
| `modalix_int8_packed` | FP32 | INT8 packed MLA heads followed by audited EV74 detessellation/dequantization |
| `modalix_bf16_packed` | BF16 | Packed MLA output detessellated to BF16 heads |
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
  --output-dir build/superpoint/modalix_int8_packed \
  --variant modalix_int8_packed
```

Vulcan activates the pinned Model Compiler environment and builds each variant
in an isolated job. After a successful staging run, install only this model:

```bash
sima-cli neat install --stg models/superpoint@develop:latest --install-dir ./compiled-models
```

The model request and provenance record are tracked in
[sima-neat/models#24](https://github.com/sima-neat/models/issues/24).
