# MNIST CNN

![MNIST CNN model hero](hero.png)

## Overview

MNIST CNN recognizes a handwritten digit. It is the ONNX Model Zoo's small
convolutional network, trained in CNTK on the MNIST dataset of handwritten
digits. It takes one 28×28 grayscale image of a single digit, white on a black
background, and returns one score for each of the ten digits, 0 to 9. Your
application takes the highest score as the digit, or applies a softmax for
probabilities. The model runs as one component.

## Inputs and outputs

**Input.** Crop the digit, convert it to grayscale and resize it to 28×28, with
the digit white on a black background as in MNIST. Scale the pixel values to
0–1 by dividing by 255; the model applies no other normalization. Pass the image
as `Input3`, a 28×28×1 (height, width, channel) float tensor.

**Outputs.** `Plus214_Output_0` holds 10 digit scores before a softmax.

1. Take the index of the highest score as the digit.
2. Apply a softmax to the scores when the application needs a confidence.

Score index i is the digit i, from 0 to 9.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
