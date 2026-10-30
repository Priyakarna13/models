# SSD-MobileNet V3

![SSD-MobileNet V3 model hero](hero.png)

## Overview

SSD-MobileNet V3 is a single-shot object detector with a MobileNet V3 Large backbone, trained on COCO. It takes a 320×320 RGB image and returns raw box and class predictions for 3,234 anchors over six grid levels. An anchor is a fixed reference box that each prediction adjusts. A CPU reference decoder turns these predictions into detections. Device-level qualification remains pending.

## Inputs and outputs

**Input.** Decode RGB, divide by 255, subtract 0.5 and divide by 0.5. Then stretch to 320×320 using bilinear resampling with half-pixel coordinates and no antialiasing. Normalize before resizing; do not use the V1/V2 asymmetric resize.

**Outputs.** Twelve tensors over grids of 20, 10, 5, 3, 2 and 1 cells, with six anchors per cell:

- `ssd_loc_0` to `ssd_loc_5`: four box offsets (x, y, width, height) for each anchor.
- `ssd_confidence_0` to `ssd_confidence_5`: 91 class logits for each anchor.

Use `scripts/accuracy.py` for the reference decoding:

1. Generate TorchVision's 3,234 anchors, including its interpolated square boxes and aspect ratios 2 and 3 with their reciprocals.
2. Divide the offsets by 10, 10, 5 and 5. Shift each anchor's center and scale its width and height by the exponent of the last two offsets, capped at log(1000/16), then clip to the image.
3. Apply softmax across all 91 logits, including background, and discard background detections.
4. Keep scores above 0.001 and the best 300 candidates per class before suppressing overlaps above 0.55 intersection over union. Keep the best 300 detections overall.

Class index i is COCO category ID i; index 0 is background. The native `Ssd` decoder route is not enabled until its device-level checks pass.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
