# ResNet50 conversion-registry canary

Disposable branch: `conversion-registry-test`, based on PR #370 commit
`f71c14ea7e32f411cb997671be827d053de02ae0`.

This branch adds a real Microsoft ResNet50 checkpoint-to-ONNX exporter and opts
only ResNet50 into `compilation.source: conversion`. It is test evidence, not a
request to migrate or merge the model changes into develop.

Source pins live in `models/resnet_50/model.yaml`; exporter dependencies and
reproduction instructions live beside `scripts/convert.py`. Conversion checks
compare three seeded inputs against PyTorch, then validate the exported interface.
These are conversion fidelity checks, not ImageNet accuracy qualification.

Scratch commits use `[skip ci]` to avoid an automatic whole-catalog build. Explicit
workflow dispatches still run. The orchestrator's current `VULCAN_ENV` is
production; conversion must use the same environment. Feature-branch publication
does not approve or promote a model into a release.

From the repository root:

```bash
gh workflow run vulcan-model-convert.yml --repo sima-neat/models \
  --ref conversion-registry-test -f environment=production -f model=resnet_50 -f conversion=default
# Wait for successful conversion publication before dispatching planning.
gh workflow run vulcan-model-orchestrator.yml --repo sima-neat/models \
  --ref conversion-registry-test -f models=resnet_50 -f profiles=w16a16 \
  -f base_branch=develop
```

Evidence to collect: conversion run/fingerprint, exact versioned ONNX bundle,
compile-plan selection, compile fetch and provenance input, compiled package
publication, and any downstream benchmark results. A failed stage must be recorded
with its diagnostics; starting a workflow alone is not end-to-end success.
