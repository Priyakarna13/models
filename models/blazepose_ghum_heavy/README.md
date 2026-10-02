# BlazePose GHUM Heavy 3D

This entry publishes the MediaPipe BlazePose GHUM Heavy landmark network used by
the multi-stream BlazePose 3D example in `sima-neat/apps`. It is the landmark
stage only: callers detect a person, crop a square pose ROI, and pass that ROI to
this model.

The source set pins two immutable files:

- Google's official `pose_landmark_heavy.tflite`, SHA-256
  `59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe`;
- its reviewed tf2onnx conversion, SHA-256
  `efcc9e62707d596c86bfc962a26a673401d8438702216f82fc15ead54c6f3a83`.

The ONNX conversion used TensorFlow 2.17.0, tf2onnx 1.17.0, ONNX 1.17.0,
opset 16, and the command below. tf2onnx gives internal constants generated
names that are not byte-deterministic, so the converted ONNX is itself pinned.
The compile recipe runs a deterministic numerical comparison between the pinned
ONNX and TFLite before any graph surgery or compilation.

```bash
python -m tf2onnx.convert \
  --tflite pose_landmark_heavy.tflite \
  --output pose_landmark_heavy.onnx \
  --opset 16
```

## Output contract

The upstream model exposes segmentation and heatmap tensors that the Apps
example does not consume. `surgery.py` retains the three application outputs in
their original order and replaces three compiler-incompatible channel-padding
operators with numerically equivalent 1x1 convolutions:

1. `Identity [1,195]`: 39 image-space records of x, y, z, visibility logit,
   and presence logit;
2. `Identity_1 [1,1]`: global pose-presence logit;
3. `Identity_4 [1,117]`: 39 world-space x, y, z records.

The first 33 records are the public BlazePose body landmarks. The remaining six
are auxiliary ROI-tracking landmarks. Consumers apply sigmoid to visibility and
presence logits; the model intentionally leaves them raw.

Model SDK 3.0 records the fixed NHWC float32-to-BF16 boundary directly in the
MPK manifest rather than emitting the legacy `_preproc.json` sidecar. The Apps
example configures its image resize, RGB conversion, and `[0, 1]` normalization
through Neat's runtime preprocessing options; the compile audit checks the
manifest input range, tensor shape, and boundary conversion independently.

## Local qualification

```bash
export GITHUB_TOKEN="$(gh auth token)"
model-registry validate models/blazepose_ghum_heavy
model-registry fetch models/blazepose_ghum_heavy --component model
model-registry matrix --repository . --models blazepose_ghum_heavy
```

In the repository-pinned Model Compiler environment, pass the verified ONNX
entrypoint returned by `model-registry fetch` to `compile.py`. The sibling
TFLite file is part of the verified source set and is required by the numerical
conversion gate.

```bash
python models/blazepose_ghum_heavy/compile.py \
  --model /path/to/source-set/pose_landmark_heavy.onnx \
  --output-dir /tmp/blazepose-ghum-heavy-bf16 \
  --build-profile modalix_bf16 \
  --component model
model-registry verify-mpk /tmp/blazepose-ghum-heavy-bf16
```

## Known limitations

- The numerical gates cover the TFLite-to-ONNX conversion and the ONNX
  surgery. They do not compare the compiled BF16 MPK against a reference, and
  no dataset-backed landmark accuracy test is declared.
- The ONNX file was converted locally with the command above and uploaded as a
  pinned release asset, not produced by the Vulcan Model Convert workflow.

The Apache-2.0 license and model behavior are documented in Google's
[pose model card](https://developers.google.com/ml-kit/images/vision/pose-detection/pose_model_card.pdf).
