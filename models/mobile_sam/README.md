# MobileSAM

![MobileSAM model hero](hero.png)

## Overview

MobileSAM is Segment Anything with a small TinyViT image encoder distilled from
SAM's ViT-H. Given a click or a box on an image, it returns masks of the object
there. It runs as three components: the image encoder once per image, as
`image_encoder_0` then `image_encoder_1`, then the `prompt_decoder` once per
prompt on the encoder's embedding, so one image can answer many prompts. Builds: `w16a16` (BF16) and `w8a8` (INT8, calibrated on 50
COCO 2017 images).

COCO has no prompt annotations, so the compile stage compares each build with
the float32 model: the quantized encoder feeds the quantized decoder on COCO
2017 validation images with a fixed click and a fixed box.

## Inputs and outputs

**`image_encoder_0` input.** `image`, 1024×1024 RGB. Resize the image with
bilinear resampling so its longer side is 1024. Normalize the 0–255 values with
mean (123.675, 116.28, 103.53) and standard deviation (58.395, 57.12, 57.375).
Then pad with zeros on the right and bottom.

**Between the encoder parts.** `image_encoder_0` returns `features`, 160
channels on a 64×64 grid; pass them to `image_encoder_1` unchanged.

**`image_encoder_1` output.** `image_embeddings`, 256 channels on a 64×64 grid.

**`prompt_decoder` inputs.** The encoder's `image_embeddings`, and up to five
prompt points:

- `point_coords`: each point's (x, y) after the resize, plus 0.5, divided by
  1024 and mapped to −1…1, as 2 channels by 5 points.
- `point_types`: a one-hot type per point, as 5 channels by 5 points, in the
  order padding, background click, foreground click, box top-left, box
  bottom-right. Unused points are padding with coordinates 0.

**`prompt_decoder` outputs.** `masks`, four 256×256 mask logits, and
`iou_predictions`, the predicted quality of each. Mask 0 suits a box or several
points; masks 1–3 are three guesses for a single click. Upsample the chosen
mask to 1024×1024, remove the padding, resize to the image and threshold at 0.

`scripts/pipeline.py` implements these steps with NumPy and Pillow.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
