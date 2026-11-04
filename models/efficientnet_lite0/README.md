# EfficientNet-Lite0

![EfficientNet-Lite0 model hero](hero.png)

## Overview

EfficientNet-Lite0 is a convolutional network for image classification built for mobile and edge accelerators from inverted residual blocks with depthwise convolutions and ReLU6 activations, trained on the 1,000 classes of ImageNet-1k. The package uses
Google's ImageNet weights. It takes a 224×224 RGB image and
returns one score for each class. Your application turns the scores into
probabilities with a softmax and reports the most likely classes. The model runs
as one component.

## Inputs and outputs

**Input.** Resize the image with bicubic resampling so its shorter side is 256,
then crop the center 224×224. Scale the RGB values to 0–1, subtract the mean
(0.5, 0.5, 0.5), and divide by the standard deviation (0.5, 0.5, 0.5). The
reference scripts pass this image as a 224×224×3 (height, width,
channel) float tensor, with Neat's resize, color conversion, and normalization
turned off.

**Outputs.** `logits` holds 1,000 class scores.

1. Apply a softmax to `logits` to get class probabilities.
2. Take the five highest probabilities as the top-5 classes.

Class index i is the i-th ImageNet-1k WordNet ID in sorted order, which is the
standard ImageNet-1k class order.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
