# DiT-XL/2 256

![DiT-XL/2 256 model hero](hero.png)

## Overview

DiT-XL/2 256 generates a 256 x 256 image of one of the 1000 ImageNet classes.
It starts from random noise and removes a little of it per step: four
transformer components, run in turn, predict the noise in a 4 x 32 x 32 latent
for the requested class and for no class, and your application combines the
two and steps the latent. After the last step, the VAE decoder component turns
the latent into the image.

## Inputs and outputs

**Start.** Draw a [1, 4, 32, 32] latent from a standard normal distribution.
Choose the DDIM timesteps, for example 50 evenly spaced from 999 to 0, and a
guidance scale, such as 4.

**Each step.** For timestep t, pass through `transformer_0` to `transformer_3`
twice, once for the class and once for the null class 1000. Every part takes:

1. `timestep_features` [1, 256, 1, 1]: for i < 128, cos(t · 10000^(−i/127)) at i and sin of the same at 128 + i.
2. `class_one_hot` [1, 1001, 1, 1]: 1 at the class, or at 1000.
3. `transformer_0_latent` [1, 4, 32, 32], the latent; parts 1-3 take the previous part's `hidden` [1, 1152, 16, 16] as `previous_hidden`.
4. `transformer_3_prediction` [1, 8, 32, 32]: take channels 0-3 of each pass; noise = null + scale · (class − null).
5. Take the DDIM step from t to the next timestep with that noise.

Input names carry their part's prefix, such as `transformer_2_class_one_hot`.

**Decoding.** Pass the final latent to `vae_decoder_latent` [1, 4, 32, 32] as
it is; the component divides by 0.18215. `vae_decoder_image` [1, 3, 256, 256]
is RGB in [−1, 1]: map it to 8-bit with (x + 1) · 127.5, clipped to [0, 255].

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
