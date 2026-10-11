# EfficientSAM3 RepViT-M

![EfficientSAM3 RepViT-M model hero](hero.png)

## Overview

EfficientSAM3 RepViT-M finds and segments the objects that a short text prompt
names, such as "dog" or "traffic light". It is the Stage 1 EfficientSAM3
student: a RepViT-M1.1 image encoder distilled from SAM 3, with SAM 3's text
encoder, fusion encoder, and DETR decoder. The export resamples the image
encoder's output to a 48×48 token grid, where the upstream student uses 72×72.
The Modalix package splits it into a text encoder, run once per prompt, and an
image model, run once per image and prompt. Each image-model run returns 200
scored boxes with 192×192 mask logits, where the upstream student returns
288×288. Box and mask AP are measured on a Modalix DevKit with every COCO
category name as a prompt.

## Usage

Download the published build from the `develop` catalog:

```bash
sima-cli models download \
  --id efficientsam3_repvit_m \
  --variant modalix_bf16 \
  --branch develop
```

The package contains the `text_encoder` and `image_model` MPKs and the CLIP
tokenizer files in `resources/tokenizer`. An application tokenizes the prompt,
runs the text encoder, packs its features on the A65, and runs the image model
with that packed text on every image. `scripts/accuracy.py` runs the same
sequence from token IDs that conversion checks against this tokenizer.

## Model interface

```text
prompt -> CLIP BPE -> one-hot -> text_encoder.mpk -> text_features
                                       A65 packing: text_features + token mask -> text
image -> 1008x1008 RGB in [-1, 1] ----------------------------> image_model.mpk -> detections + mask_logits
                                                         text --^
```

**Prompt.** Tokenize the prompt with the lowercase CLIP BPE in
`resources/tokenizer` (`vocab.json`, `merges.txt`), as SAM 3 does: the start
token 49406, the prompt's tokens, and the end token 49407. The exported
context holds 16 positions, so a prompt may have at most 14 tokens. The
`token_one_hot` input is `1×16×49408` FP32: row `i` is one-hot for token `i`,
and the rows after the end token are all zero. The token lookup runs as a
one-hot MatMul in 16 vocabulary chunks because the MLA cannot gather from a
49,408-row table.

**Packing on the A65.** The text encoder returns `text_features`
`1×16×256`. The image model's `text` input is `1×17×256`: rows 0–15 are the
text features, and row 16 has a 1 in its first `n` columns for the `n` prompt
tokens (start and end included) and 0 elsewhere. The image model turns that row
into a −10,000 attention bias that removes the padding positions, so their
features have no effect. `calibration.packed` implements it.

| Tensor | Producer | Consumer | Shape | Type | Layout | Where | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `text_features` | text encoder | A65 packing | `1×16×256` | FP32 | NLC | A65 | One encoder run serves every image for that prompt. |
| `text` | A65 packing | image model | `1×17×256` | FP32 | NLC | MLA input | The token mask comes from the prompt length, which only the host knows. |

**Image.** Decode to RGB, stretch to 1008×1008 without keeping the aspect
ratio, and scale the pixels to `[-1, 1]` with `x / 127.5 − 1`. SAM 3 resizes the
uint8 image with antialiased bilinear filtering; `calibration.py` uses OpenCV
area averaging when shrinking and bilinear interpolation when enlarging. On
COCO images the two agree to within one uint8 level. The ONNX input
`pixel_values` is NCHW `1×3×1008×1008`; with Neat, pass the image as an HWC
`1008×1008×3` FP32 tensor.

**Outputs.** `detections` is `1×200×6`: for each query `x1, y1, x2, y2` in
1008×1008 input pixels, the score, and 0. The score is the query probability
times the image-level presence probability, as in SAM 3. `mask_logits` is
`1×200×192×192` and covers the whole 1008×1008 input; the MPK returns it as
HWC `192×192×200`. Scale the boxes by the original width and height divided by
1008, resize each mask bilinearly to the original size, and keep pixels whose
logit is positive.

With PyNeat, pass every input with `layout=pyneat.TensorLayout.HWC`.
`Tensor.from_numpy` otherwise reads a tensor whose first dimension is 1, 3, or
4 as CHW.

## Available builds

The registry declares one `modalix_bf16` build profile. Both components run
weights and activations in BF16, calibrated with histogram MSE over 2,048 bins.
The grouped `efficientsam3` invocation compiles the text encoder first and
calibrates the image model on the quantized encoder's features, as the board
runs them. Layer norms keep FP32 intermediates. The compile recipe rejects an
MPK that places a layer on the A65 or APU.

## Quality and performance

Compilation writes a fidelity report that compares the BF16 text encoder,
host packing, and BF16 image model with the converted ONNX on four COCO images.
It is diagnostic: query reassignment can lower numerical similarity for
equivalent detections.

On a Modalix DevKit, `scripts/accuracy.py` prompts each of the 100 lowest-ID
COCO 2017 val2017 images with all 80 category names, 8,000 image-model runs.
For each image it keeps the 100 highest-scoring of the 16,000 query and
category pairs and reports the 12 COCO box AP and AR metrics plus the mask
mAP 50–95 (`coco_mask_map`). Boxes are not clipped, as in SAM 3's processor.
The script encodes the 80 prompts with the text encoder first and keeps every
result in memory.

Published SAM 3 and EfficientSAM3 scores use other protocols, such as
box-prompted mask mIoU, so no reference score is declared. Absolute AP with
every category as a prompt is low, and the FP32 ONNX model at the same 48×48
grid scores as low on this protocol. The upstream 72×72 grid was not measured,
so this protocol does not separate the checkpoint from the grid change.

The benchmark script runs Neat Core's default synthetic benchmark on each MPK,
and CI publishes one result for each component.

## Source and license

The source is the Stage 1 `efficient_sam3_repvit_m.pt` checkpoint on Hugging
Face. `convert.py` rebuilds the network from the `sam3` 0.1.4 package and
timm's RepViT-M1.1, loads the checkpoint, and exports one ONNX graph.

- [EfficientSAM3 model card](https://huggingface.co/Simon7108528/EfficientSAM3)
- [Pinned checkpoint revision](https://huggingface.co/Simon7108528/EfficientSAM3/tree/85b05896928f974e308f889d7ccb2eefc069de98)
- [EfficientSAM3 source](https://github.com/SimonZeng7108/efficientsam3/tree/bd0936c788fed8d51fa799437f05abd97b401b06), for background; conversion does not use it
- [CLIP tokenizer revision](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)

The model is licensed under `Apache-2.0 AND LicenseRef-SAM-License`, and
redistribution is restricted. The EfficientSAM3 model card declares Apache-2.0
for the distilled RepViT encoder. The text encoder, fusion encoder, decoder,
and heads are Meta's SAM 3 weights, covered by the
[SAM License](https://github.com/facebookresearch/sam3/blob/a13e358df479a80938c8bfed0e0754bea8fdb277/LICENSE),
the same text as `LICENSE` in the pinned `sam3` 0.1.4 wheel. It allows
redistribution of SAM 3 and works derived from it only under its terms and with
a copy of the agreement, and it restricts uses subject to trade controls.
Anyone who redistributes this package must include that copy. The CLIP BPE vocabulary is OpenAI's, released
with the [openai/CLIP](https://github.com/openai/CLIP) repository under the MIT
License.

## Reproducing the artifact

### Conversion

The `conversion.default` entry in `model.yaml` pins four files: the checkpoint,
the `sam3` 0.1.4 wheel, and CLIP's `vocab.json` and `merges.txt`. `convert.py`
unpacks only the `sam3` package from the wheel, without its declared
dependencies; `scripts/requirements-convert.txt` pins the packages the export
imports. `sam3` 0.1.4 declares `iopath>=0.1.10`, but 0.1.10 publishes no wheel,
so the requirements pin 0.1.9; the export uses it only to open the local BPE
merges file, and the tokenizer check below verifies the result. On Linux, the
shared runner script converts in a fresh Python 3.12 environment:

```bash
scripts/run_model_conversion.sh models/efficientsam3_repvit_m default /tmp/efficientsam3-export
```

`convert.py` first checks that SAM 3's tokenizer, rebuilt from the pinned
merges, reproduces `vocab.json` and the token table in `calibration.py` for all
80 COCO category names. It then builds the stock SAM 3 detector with the RepViT
encoder resampled to the 48×48 grid, loads the checkpoint, and rewrites it into
an export graph. The rewrite fuses batch norms, folds attention scales, splits
the token lookup into vocabulary chunks, and drops the video tracker, whose
checkpoint keys the load skips. It exports at opset 17, folds constants with
onnxsim, and compares text features, scores, boxes, and masks with the stock
PyTorch model, built at the same 48×48 grid, on four synthetic prompt and image
pairs.

To publish a conversion, manually run **Vulcan Model Convert** with model
`efficientsam3_repvit_m`, then copy the printed source pin into `model.yaml`
only after surgery and compiler validation. The smoke comparison does not
measure model accuracy.

### Calibration and compilation

`surgery.py` cuts the converted graph into the text encoder and the image
model. In the image model it replaces the decoder's 5 Sin, 5 Cos, and 20 Log
nodes, which the compiler would lower to BF16 lookup tables, with polynomials
on their bounded domains, each within 2e-6 of float32. Compilation checks both
parts against the converted graph on real images before quantizing.

Both components calibrate on the same 16 images, drawn with seed 0 from the
`cal2017` split of the COCO 2017 mini dataset, each with one category prompt
spread over the 80 names; the next four images measure fidelity. No scored
val2017 image calibrates the model. In the pinned Model Compiler environment,
run the grouped invocation:

```bash
scripts/run_model_compilation.sh models/efficientsam3_repvit_m modalix_bf16 efficientsam3 /tmp/efficientsam3
```

The recipe writes:

```text
<output>/efficientsam3_repvit_m/modalix_bf16/
  text_encoder/*_mpk.tar.gz
  image_model/*_mpk.tar.gz
  image_model/fidelity.json
  compile-outputs.json
```

Cut graphs and generated models stay in a temporary directory outside Git.

### DevKit accuracy

The `coco_accuracy` entry declares the COCO 2017 `mini` dataset archive and
the published metrics. On a Modalix DevKit, run the accuracy stage on the
output of `scripts/run_model_compilation.sh`, setting `REGISTRY_PYTHON` to a
Python that has the full Model Registry tooling and Dataset Registry:

```bash
scripts/run_model_accuracy.sh models/efficientsam3_repvit_m modalix_bf16 \
  /tmp/efficientsam3 /media/nvme/neat-tests/efficientsam3-accuracy
```

The script installs
`scripts/requirements-accuracy.txt` beside PyNeat's environment, extracts the
archive under `/media/nvme/neat-tests/`, and removes both when it exits. See
[Accuracy on the DevKit](../../docs/local-accuracy-development.md#accuracy-on-the-devkit)
for the result format and publication.
