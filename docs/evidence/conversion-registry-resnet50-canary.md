# ResNet50 conversion-registry end-to-end evidence

## Result

The live conversion-to-publication canary passed for all three ResNet50 profiles:
`w16a16`, `w8a8`, and `w8a16`. A real pinned Microsoft checkpoint was exported to
ONNX, validated, published as a versioned conversion, selected by the planner,
fetched by the Model Compiler, compiled, verified, and published as model packages.
The feature catalog contains all three new builds and packages with no pending
ResNet50 builds. Every compile's registry provenance references the exact selected
conversion run, checksum and S3 object version.

This is a disposable branch, `conversion-registry-test`, based on PR #370 commit
`f71c14ea7e32f411cb997671be827d053de02ae0`. The tested code commit is
`ab4bab5e406424ecbf9d1b4eedad345a783e5985`. Subsequent evidence-only commits do not
change the tested recipes. This is not a request to merge the canary model changes.

## Live runs and records

- [Conversion workflow](https://github.com/sima-neat/models/actions/runs/41874949759).
- [Model-only orchestrator](https://github.com/sima-neat/models/actions/runs/41876165659).
- Registry environment: production, selected by the existing `VULCAN_ENV`.
- Model Compiler identity: `3.0.0+develop.8681769b526b`, as pinned by this checkout.
- Completed conversion: `f284a26c-4579-485c-afe9-2e7690aa3056`.
- Completed compile plan: `ed0e8659-6c0b-48ea-b9ce-fc3fe6aea0f6`.
- Conversion fingerprint: `be082e1d5056a7dec6aab50bd6adde6d13fef68b972c0841fcf1318fa63a270a`.
- ONNX artifact: `d2efdf8e-ca84-48be-9bbd-8c89e2662564`.
- Bundle SHA-256: `cd1756d20e9335ae613c272c6dc3238b04e50d09136ed609f7ae5285a839d0f1`.
- Bundle size: 95,020,747 bytes.
- S3 object version: `n.m_51uuIs_FauAqDVkBdsWbaT40_o_H`.

| Profile | Completed compile run | Completed package run |
| --- | --- | --- |
| w16a16 | `e59562b9-d6ea-4e19-89e2-c08a530f9103` | `dc7f36e0-24ee-479b-a031-7101216288cc` |
| w8a16 | `52ce8923-e7e2-4584-9a22-833cee38e58f` | `bf68abc5-fefc-4d97-b510-a2a08d4b525a` |
| w8a8 | `b4cd0450-b22f-4b4e-9d1a-217c6e0b3399` | `13b45950-0f22-4c01-965a-18f9662597af` |

The machine-readable [compiled results](resnet50-compiled-results.json) include
expected and verified compile fingerprints, package fingerprints and source
provenance. The saved [conversion selection](resnet50-conversion-sources.json)
contains no signed download URL and can reproduce the selected source while it
remains retained. Feature-branch publication does not approve or promote a model
into a release.

## Checks performed

- Source checkpoint and config were downloaded through the existing checksum- and
  size-verifying source loader; exact pins live in `models/resnet_50/model.yaml`.
- The exporter ran on Linux ARM64 and compared three seeded inputs with PyTorch
  at rtol=0.001 and atol=0.0001. Largest absolute logit error:
  0.000013828277587890625. ONNX validation and the real conversion stage test passed.
- A local macOS ARM64/Python 3.12.8 export also passed, with largest absolute
  error 0.00001049041748046875. Both platforms produced identical ONNX bundle bytes.
- A live CLI fetch into an empty local cache verified the exact published object
  (`cache_hit: false`); repeating returned `cache_hit: true`. Corrupting one byte
  in the local cached ONNX caused a checksum/size rejection, after which the cache
  was restored. The registry object was never modified.
- All three Model Compiler jobs consumed the saved conversion selection, generated
  valid artifacts and passed the model's compile tests. Their publication jobs
  published the compilations, packages and updated branch catalog.
- Live registry reads confirmed all three compile fingerprints matched the
  locally calculated values. Compile and package runs are completed, and each
  package points to the corresponding new compile run.
- 251 model/matrix/CLI/planner tests and 61 conversion regression tests passed;
  lint, model metadata validation and workflow syntax checks passed.

Conversion agreement and compile fidelity are not ImageNet accuracy qualification.
DevKit benchmark workflows were dispatched separately. Their results are not
claimed by this conversion-to-package evidence.

## Reproduction

Exporter dependencies and local export instructions live beside
`models/resnet_50/scripts/convert.py`. From the repository root:

```bash
gh workflow run vulcan-model-convert.yml --repo sima-neat/models \
  --ref conversion-registry-test -f environment=production -f model=resnet_50 -f conversion=default
# Wait for successful conversion publication before dispatching planning.
gh workflow run vulcan-model-orchestrator.yml --repo sima-neat/models \
  --ref conversion-registry-test -f models=resnet_50 -f base_branch=develop
```

Select the model without `profiles`: the normal workflow intentionally disables
ordinary publication for a profile-filtered dispatch. All three declared profiles
must run to exercise the normal model publication path without changing those gates.

For exact source reproduction, export `MODEL_REGISTRY_CONVERSIONS` from the saved
conversion-selection JSON and run `model-registry fetch models/resnet_50 --json`.
Local Python used certifi through `SSL_CERT_FILE` because this macOS installation
lacks a default CA bundle.

## Scratch-only adjustments

Scratch commits use `[skip ci]` to avoid automatically building the whole catalog;
explicit dispatches still run. Only ResNet50 opts into `compilation.source: conversion`.
Existing pinned-source unit-test scenarios use an explicit fixture source rather
than depending on the catalog model staying pinned.

The initial production conversion dispatch (41873074919) remained queued without
an assigned worker for ten minutes and was cancelled. This scratch branch switches
conversion only to GitHub-hosted `ubuntu-24.04-arm` and directories under `/tmp`.
Export, validation, credentials and registry publication code remain the same;
compilation uses the normal production Model Compiler runners. The runner override
is not proposed for PR #370.

An initial profile-filtered orchestrator dispatch (41875381083) was cancelled
because it would skip publication. Its compile had started despite the parent
status API reporting it queued; it was replaced by the model-only dispatch above.
The completed plan still provides independent evidence of exact-version selection.
GitHub parent job statuses lagged the completed logs; final conclusions here use
completed job logs and authoritative registry provenance and catalog records.
