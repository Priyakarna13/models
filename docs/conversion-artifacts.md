# Converted ONNX artifacts

Models with an upstream checkpoint can use this declaration:

```yaml
conversion:
  source:
    uri: hf://models/owner/repository@0123456789012345678901234567890123456789/model.pt
    sha256: <upstream checksum>
    size_bytes: <upstream size>
compilation:
  format: onnx
  source: conversion
```

Keep upstream pins and exporter dependencies in Git. Do not copy the output URL,
checksum, size, or run ID into `model.yaml`. The folder identifies the model and
the single conversion produces `model.onnx`. Direct file and file-set sources
continue to work.

## Publication and selection

Run **Vulcan Model Convert** in the selected environment. It installs the model's
exporter requirements, runs conversion and model tests, checks the ONNX, and
publishes a deterministic bundle under `models/conversions/`. The bundle contains
`model.onnx`, any referenced external tensor files, and a checked file manifest.
Skipped or failed model tests do not publish successful conversions.

Vulcan stores ordinary `model_conversion` runs with upstream inputs, parameters,
validation and the `onnx` artifact. Fingerprint version 1 covers model identity,
upstream content, exporter code, local helpers, dependency specification and
validation code. A changed recipe needs a new conversion. Shared conversion code
is also fingerprinted. This identifies the declared recipe, not a fully locked
transitive Python environment; pin exporter requirements for reproducibility.

Catalog planning resolves conversion declarations, including on-demand models,
so their build requirements have concrete content identities. This does not
schedule compilation of an on-demand model. Before deciding compile reuse, planning resolves the latest completed matching
run and checks its artifact. The plan uses ordinary `model_compile_plan` run
records and versioned inputs. Its identity includes the GitHub workflow run,
commit, model and conversion fingerprint. Rerunning that workflow resumes its
saved selection, including after interrupted publication. A new workflow can
select a newer compatible conversion. Missing or unavailable output fails with
an instruction to run conversion; it never falls back to an older artifact.

The selected metadata travels as `MODEL_REGISTRY_CONVERSIONS` to compile,
publication and catalog jobs. It contains exact object versions, not signed
URLs. Downloads request fresh URLs for that exact artifact and verify both the
bundle checksum and member checksums. Compile fingerprints use output content;
another run or environment with identical bundle bytes does not change them.
Compile publication records the conversion as a versioned registry input.

For explicit local reproduction, save the original plan's `conversion_sources`
JSON map and use it with the same checkout and compiler:

```bash
export MODEL_REGISTRY_CONVERSIONS="$(cat conversion-sources.json)"
model-registry fetch models/<model-id> --json
scripts/run_model_compilation.sh models/<model-id> <profile> <profile> /tmp/compiled
```

The plan is also recoverable through its run's `/provenance` endpoint: its
`conversion` input records producer/artifact IDs and content metadata. Retained
compile inputs record the same selection. Use the original environment's reader;
do not substitute a current resolve result when reproducing an older build.

## Retention and rollout

Deploy Vulcan's conversion validation and migration
`019_model_conversion_retention` before enabling registry sources. Plans protect
selected conversions for 24 hours, covering the normal six-hour compile limit
and queue time. Abandoned plans do not create permanent holds. After publication,
retained compile inputs protect conversion bytes and provenance, including when
the conversion branch is deleted. After the plan window, an unpublished build
has no retention guarantee; unavailable outputs require a new workflow.

Completed conversions without consumers follow normal branch retention. Newer
conversion publication alone does not delete older outputs. The legacy S3
sweeper skips the shared `models/conversions/` prefix because it has no branch
`latest.tag`; registry cleanup owns these versioned objects.

Existing pinned model declarations stay usable while the backend rolls out.
Publish matching validated outputs in each required environment, then change each
model's `compilation.source` to `conversion` and update its reproduction section.
This PR does not claim historical uploads were validated registry conversions
and does not switch those declarations before outputs exist. No cross-environment
promotion or automatic conversion scheduler is introduced.
