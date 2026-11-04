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

## Runner availability adjustment

The initial production conversion dispatch, run 41873074919, remained queued
without a worker for ten minutes while four matching production compile workers
were occupied. The scratch branch switches only conversion to GitHub-hosted
`ubuntu-24.04-arm` and temporary directories under `/tmp`. Export, stage
tests, credentials and registry publication remain the same. Compilation still
uses the normal production Model Compiler workflow. This runner override is
canary-only and is not a proposed change to PR #370.

Local real export passed on macOS ARM64 with Python 3.12.8. Three seeded inputs
agreed with PyTorch at rtol=0.001, atol=0.0001; the maximum absolute logit error
was 0.00001049041748046875. The real conversion stage test also passed. After
isolating pinned-source unit-test fixtures from the ResNet50 catalog opt-in,
251 model, matrix, CLI and planner tests passed.
