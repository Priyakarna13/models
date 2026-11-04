# MNIST FC

![MNIST FC model hero](hero.png)

## Overview

MNIST FC recognizes a handwritten digit with a fully connected network. It is
David Corvoysier's three-layer network, trained on the MNIST dataset of
handwritten digits. It takes one 28×28 grayscale image of a single digit, white
on a black background, and returns one score for each of the ten digits, from
0 to 9. Your application takes the highest score as the digit, or applies a
softmax for probabilities. The model runs as one component.

## Inputs and outputs

**Input.** Crop the digit, convert it to grayscale and resize it to 28×28, with
the digit white on a black background as in MNIST. Scale the pixel values to
0–1 by dividing by 255; the model applies MNIST's mean and standard deviation
itself. Pass the image as `image`, a 28×28×1 (height, width, channel) float
tensor.

**Outputs.** `logits` holds 10 digit scores before a softmax.

1. Take the index of the highest score as the digit.
2. Apply a softmax to the scores when the application needs a confidence.

Score index i is the digit i, from 0 to 9.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
