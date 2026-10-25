# Parakeet TDT 0.6B v3

![Parakeet TDT 0.6B v3 model hero](hero.png)

## Overview

Parakeet converts multilingual speech into text using overlapping audio windows. The package combines one compiled encoder with separate CPU predictor and joiner models. Each call reprocesses the available history and future context, then decodes only the central region. The decoder predicts both text tokens and how far to advance through the encoded audio. Use it for buffered captions or recorded speech where a short lookahead is acceptable.

## Inputs and outputs

1. Supply 16 kHz mono audio. Compute 128-bin log-mel features with a 512-point transform, 400-sample symmetric Hann window, 160-sample hop, and preemphasis 0.97. The package includes the exact filters and window in `frontend.npz`.
2. Use a 14-second window: 10 seconds of history, 2 seconds of new audio, and 2 seconds of lookahead. Normalize each feature bin over valid frames, pad to 1,400 frames, and mask padding. Decode only the central 25 encoded frames; the packaged helper handles shorter starting and ending windows.
3. Run the encoder to obtain `encoded`, with 640 channels per frame.
4. Run `predictor_cpu_int8.onnx` with `token`, `hidden`, and `cell`; retain `prediction`, `next_hidden`, and `next_cell`. Read `token_logits` and `duration_logits` from the joiner. Update the predictor after text tokens and advance by the predicted duration, carrying any remaining advance to the next window.
5. Map token indices through `tokenizer.json`; index 8192 is blank. `transcribe.py` implements feature preparation, state handling, and decoding using the public Python API and the two CPU models.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
