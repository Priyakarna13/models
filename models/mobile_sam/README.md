# MobileSAM

![MobileSAM model hero](hero.png)

## Overview

MobileSAM turns clicks or boxes into object masks using a TinyViT image encoder distilled from Segment Anything. The package runs as two components: the whole `image_encoder` prepares an image embedding, then `prompt_decoder` predicts masks for each prompt. Run the encoder once per image and reuse the embedding for subsequent prompts. You supply foreground or background clicks, box corners, or a combination; mask prompts are not supported.

## Inputs and outputs

Let Neat prepare `image` on `image_encoder` from an RGB image: letterbox it into the top-left corner of 1024×1024 with antialiased bilinear resampling and pad value 116, then normalize 0–255 pixels using mean (123.675, 116.28, 103.53) and standard deviation (58.395, 57.12, 57.375), rounding to nearest even.

1. Pass the RGB image to `image_encoder` once per image.
2. Keep its `image_embeddings`, 256 channels on a 64×64 grid, copied to EV74 memory with `Tensor.cvu()`, which keeps the image's size for the decoder. Pass them to `prompt_decoder` for each prompt; do not rerun the encoder while the image is unchanged.
3. Prepare up to five points with `prompt_inputs` and `PromptEncoding` in `scripts/pipeline.py`. Load the package's `prompt-encoding.json` once. The CPU produces `point_encoding` with 256 float32 channels and six slots. `point_types` has five one-hot channels: padding, background, foreground, box top-left, box bottom-right. Point-only prompts include one padding token; unused slots are zero and excluded from attention.
4. The decoder predicts `masks`, four 256×256 logit maps, and `iou_predictions`, their predicted quality. Set its `decode_type` to Neat's `MobileSam` decoder, with `multimask_output` false for boxes or several clicks (mask 0), or true for one click (the highest-quality mask among 1–3).
5. `decode_segmentation` returns that mask at the image's size, 255 where its logit, upsampled as Segment Anything does, is above zero, with its box and predicted quality.

Masks carry no class labels. The helpers use NumPy and Linux libm; `pipeline.preprocess` and `pipeline.image_mask` do the same preparation and resizing on the CPU with Pillow and OpenCV.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
