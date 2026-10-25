# Nemotron 3.5 ASR Streaming 0.6B

![Nemotron 3.5 ASR Streaming 0.6B model hero](hero.png)

## Overview

Nemotron converts multilingual speech into text as audio arrives. The package combines one compiled encoder with separate CPU predictor and joiner models. It accepts mono audio, carries encoder history between calls, and returns text tokens that accumulate into a transcript. Use it for microphone captions, voice interfaces, or recorded speech. The language prompt can request a supported language or automatic recognition.

## Inputs and outputs

1. Supply 16 kHz mono audio. Compute 128-bin log-mel features with a 512-point transform, 400-sample symmetric Hann window, 160-sample hop, and preemphasis 0.97. The package includes the exact filters and window in `frontend.npz`.
2. Use 560 ms chunks with 480 ms lookahead. The first call consumes 49 feature frames; later calls consume 56. Preserve the subsampling, key, value, and convolution states between calls. Select a prompt from `processor_config.json`; `auto` selects automatic recognition.
3. Run the encoder to obtain `encoded`, with 640 channels per frame. The encoder also returns `sub/next_sub1`, `sub/next_sub2`, and each layer’s `next_keys`, `next_values`, and `next_conv` for feedback.
4. Run `predictor_cpu_int8.onnx` with `token`, `hidden`, and `cell`; retain `prediction`, `next_hidden`, and `next_cell`. Repeat the joiner on each encoded frame until it emits blank, updating the predictor after each text token.
5. Map token indices through `tokenizer.json`; index 13087 is blank. `transcribe.py` implements feature preparation, state handling, and decoding using the public Python API and the two CPU models.

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model). For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
