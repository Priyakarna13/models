# SSD-MobileNet V2

![SSD-MobileNet V2 model hero](hero.png)

## Overview

SSD-MobileNet V2 is a single-shot object detector with a MobileNet V2 backbone,
trained on COCO. It predicts objects from 1,917 reference boxes across six grid
levels. These reference boxes are also called anchors. Native accuracy qualification is incomplete; do not treat this route as
a qualified replacement for the reference decoder yet.

## Inputs and outputs

**Input.** Pass the original RGB image to the Model API with image preprocessing,
stretch resizing and explicit mean and standard deviation of 0.5. Neat resizes
to 300×300 and scales RGB to -1 to 1. Do not resize or normalize again in your
application. The source reference uses TensorFlow asymmetric bilinear resizing;
the native path is not claimed to be bit-equivalent.

**Compiled outputs.** Twelve tensors over grids of 19, 10, 5, 3, 2 and 1 cells:

- `ssd_loc_0` to `ssd_loc_5`: box offsets (y, x, height, width).
- `ssd_confidence_0` to `ssd_confidence_5`: 91 class logits per reference box.

Level 0 has three boxes per cell; the other levels have six.

Set `ModelOptions.decode_type` to `BoxDecodeType.Ssd` for one typed bounding-box
output. This requires a Neat runtime with prepared SSD carrier support.
`scripts/accuracy.py` measures original images with threshold 1e-8, IoU 0.6 and
top 100. `scripts/performance.py` measures the complete detector with threshold
0.2, IoU 0.6 and top 100 on a reproducible 720p frame.

The independent reference functions retained in `scripts/accuracy.py` use
TensorFlow anchors, offset scales (10, 10, 5, 5), clipping and positive-area
filtering, sigmoid scores, per-class overlap suppression and a final top-100
selection. This reference algorithm is not a claim of native decoder equivalence.
Class index i is COCO category ID i; index 0 is background and 10 indexes are unused.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
