# FLD 68 Landmarks

![FLD 68 Landmarks model hero](hero.png)

## Overview

FLD 68 Landmarks finds the 68 standard facial landmarks on a face: the jaw line, eyebrows, nose, eyes and mouth,
numbered as in the iBUG 300-W scheme. It is ibug's Face Alignment Network with two stacked hourglasses, trained on
300W-LP. It takes one 256×256 RGB crop around a face that a face detector has found, and returns one heatmap per
landmark, which your application turns into point coordinates in the original image. The model runs as one
component.

## Inputs and outputs

**Input.** Start from a face box [left, top, right, bottom] from a face detector.

1. Take the box centre and its mean side length, and divide the side by 0.55.
2. Cut a square of that side around the centre, padding with zeros where it leaves the image.
3. Resize the square to 256×256, keep RGB order, and scale the pixel values to 0–1 by dividing by 255.

Pass the crop as `image`, a 256×256×3 (height, width, channel) float tensor.

**Outputs.** `heatmaps` holds 68 heatmaps of 64×64, one per landmark.

1. In each heatmap, find the peak, and keep only the values within 6.4 heatmap pixels of it.
2. Clamp them at zero and take their centroid, counting each pixel at its centre (index + 0.5).
3. Scale the centroid by the crop's side over 64 and add the crop's top-left corner to get the point in the image.
4. The peak value is the point's confidence.

Heatmap index i is landmark i of the iBUG 68-point scheme: 0–16 jaw, 17–26 eyebrows, 27–35 nose, 36–47 eyes, 48–67 mouth.

`scripts/performance.py` measures CPU cropping, native normalization, inference and
landmark decoding; face detection is excluded. A real AFW face is letterboxed to
720p. Neat receives the prepared uint8 crop and reproduces the reference BF16
inputs. This route requires BF16; calibration and accuracy keep the float-tensor
interface. `--heatmap-only` runs the narrower diagnostic separately.
`scripts/fp32_baseline_accuracy.py` scores the exported ONNX with the same evaluator.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
