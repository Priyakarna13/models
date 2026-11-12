# BoT ReID R50

![BoT ReID R50 model hero](hero.png)

## Overview

BoT ReID R50 is fast-reid's bag-of-tricks person re-identification network: a
ResNet-50 with a batch-norm neck, trained on the Market-1501 pedestrian dataset.
It takes a cropped image of one person, 256 pixels high and 128 wide, and returns
a 2,048-value embedding. Crops of the same person from different cameras or
moments give embeddings that point the same way, so your application matches
people by comparing embeddings. The model runs as one component.

## Inputs and outputs

**Input.** Crop one person, for example with a person detector, and resize the
crop to 128×256 (width × height) with bicubic resampling. Keep the RGB values in
0–255: the model subtracts the ImageNet mean and divides by the standard
deviation itself. The accuracy script passes this crop as a 256×128×3 (height,
width, channel) float tensor, with Neat's resize, color conversion, and
normalization turned off. With Neat's preprocessing on, stretch the frame to the
input and normalize with mean 0 and standard deviation 1/255 for each channel,
which leaves the pixel values unchanged.

**Outputs.** `embedding` holds 2,048 values for the person in the crop.

1. Divide `embedding` by its L2 norm.
2. Compare two people by the dot product of their normalized embeddings: values near 1 mean the same person.
3. To find a person in a gallery of earlier crops, rank the gallery by that similarity, highest first.

There are no classes: identities come from the gallery your application keeps.

## Building applications

Run the package with the [Neat Model
API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example
apps](https://developer.sima.ai/examples).
