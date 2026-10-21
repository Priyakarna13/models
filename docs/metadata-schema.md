# `model.yaml` reference

[`schemas/model.schema.json`](../schemas/model.schema.json) is the authoritative
contract. Unknown fields and duplicate keys are rejected. The loader checks
source pins, profile references, taxonomy and conventional script paths.

The directory supplies the model ID. All model-specific implementation lives in
`models/<model_id>/scripts/`; `model.yaml` describes its immutable inputs and
supported stages. Keep one blank line between top-level sections.

```text
model
├── identity and license
├── conversion             optional checkpoint-to-ONNX source
├── compilation            source, format and ordered component IDs
├── datasets               immutable pins reused by the stages
├── build_profiles         one compile invocation per profile
├── pipeline               optional display-only component/CPU sequence
├── numerical_fidelity     optional compile-time comparison
├── accuracy               optional task-quality measurement
├── performance            whole-model measurement, required for vision
└── reference_metrics      scores published by the original authors
```

## Product identity and provenance

`display_name`, `description`, `model_type`, `tasks` and `model_family` describe
the catalog entry. Use IDs from [Model taxonomy](model-taxonomy.md). Ownership
belongs in `CODEOWNERS`. Do not declare IDs already supplied by directory names,
modalities, maturity, component roles or tensor inventories.

```yaml
license:
  spdx: Apache-2.0
  customer_redistribution: prohibited
  attribution: RF-DETR Large is distributed by Roboflow under Apache-2.0.
```

The SPDX identifier describes the upstream license; attribution credits its
owner. `customer_redistribution` records the distribution policy and does not
prevent internal compilation or packaging.

`upstream` is a human-facing source link. It defaults to the conversion or
compilation checkpoint repository and revision when available. Keep an explicit
value when the original model differs from the actual build input. Exact bytes
belong in the `source` blocks below.

`build_on_demand: true` requires explicit model and profile selection before
compilation or measurement. Pushes still validate and publish metadata. This
flag changes scheduling only, never compile keys. See [on-demand builds](adding-a-model.md#on-demand-builds).

## Conversion

One optional `conversion.source` pins the checkpoint consumed by
`scripts/convert.py`; there is no named `default` wrapper. Conversion writes
`model.onnx`, whose reviewed pin becomes `compilation.source`. Changes to the
conversion script do not recompile an unchanged pinned ONNX.

A conversion also requires `scripts/requirements-convert.txt`. The conversion
runner creates a fresh Python environment. See [pin every input](adding-a-model.md#3-pin-every-input).

## Components and source sets

One compilation consumes one verified source and produces the components in
its declared order:

```yaml
compilation:
  format: onnx
  source:
    type: file
    uri: https://example.org/pinned/model.onnx
    sha256: <64 hexadecimal characters>
    size_bytes: 123456
  components: [backbone, decoder]
```

The example URI is illustrative; validation accepts only approved immutable
hosts and forms. Recipes are conventional: `scripts/compile.py` for ONNX and
`scripts/compile.sh` for Hugging Face checkpoints. A singleton normally uses
`components: [model]`. The recipe implements graph surgery, preprocessing,
component boundaries and compilation checks; YAML does not repeat tensors.

Use a file set when the recipe needs related files. Every member keeps its own
checksum and size. Hugging Face members can share one full repository revision:

```yaml
source:
  repository: owner/repo
  revision: 0123456789abcdef0123456789abcdef01234567
  main_file: config.json
  files:
    config.json:
      sha256: <64 hexadecimal characters>
      size_bytes: 1000
    model.safetensors:
      sha256: <64 hexadecimal characters>
      size_bytes: 2000
```

An explicit member `uri` can override the shared repository. An explicit
`type: file_set` also accepts a list of members with `path`, `uri`, `sha256`
and `size_bytes`. Relative layout is preserved and verified before `main_file`
is passed to the recipe. `runtime_entrypoint: generate` or `transcribe` names a
LLiMa API operation; it is separate from the source filename.

## Resources and calibrations

Datasets are declared once under `datasets`. Prefer Dataset Registry pins with
`id`, `variant`, `sha256` and `size_bytes`. Existing immutable archives can use
`source` under the alias instead. The consuming stage references the alias.

```yaml
datasets:
  calibration:
    id: imagenet-2012
    variant: mini
    sha256: d769e759abba1fe6e03db704020dcff3f1e8309a8cac8885ab60f5a0e512c239
    size_bytes: 356052889
```

The profile's `calibration_dataset` selects the data passed as `--dataset`.
Calibration prepares quantization ranges; numerical fidelity and accuracy are
separate measurements. Sample selection belongs to the consuming script or
its stage configuration. See [Datasets](datasets.md).

There is no separate authored resource/package graph. Native checkpoint and
deployment directories already contain their tokenizer and configuration files.

## Build profiles

Profiles are a map keyed by the variant users select. Each profile contains
exactly the components listed by `compilation.components`:

```yaml
build_profiles:
  w8a8_w16a16:
    calibration_dataset: calibration
    components:
      backbone:
        parameters: {precision: w8a8}
      decoder:
        parameters: {precision: w16a16}
```

Use precision tokens from [`naming-conventions.yaml`](../naming-conventions.yaml)
in component order, collapsing repeated tokens. The target is Modalix. Compiler
parameters remain explicit, including each component's precision. The compiler
identity is pinned centrally in [`deps/manifest.json`](../deps/manifest.json).

One invocation builds the whole profile. Its internal ID is the sole component
ID for a singleton and `default` for several components; authors do not declare
invocations. Components must share an effective runner class and support status.
An unsupported profile needs a reason. Elevated `runner_class` requests require
[runner approval](runner-class-requests.md).

A singleton MPK is `<model_id>_<profile_id>_mpk.tar.gz`; multi-component MPKs add
`_<component_id>` before `_mpk.tar.gz`. Recipes write one directory per component.
`compile-outputs.json` records paths and hashes; grouped recipes also record every
derived graph hash. When numerical fidelity is declared, exactly one `correctness`
pointer identifies the report written by the recipe. It does not assign benchmark
ownership to that component.

## Compositions and packages

Download layout is derived from the compilation components. A single ONNX component
is downloaded as its MPK; multiple MPKs form one deterministic default ZIP with
the archives at its root. The package manifest remains verified against
[`package-manifest.schema.json`](../schemas/package-manifest.schema.json). LLiMa
uses its existing deployment directory layout, described below.

An optional `pipeline` adds a simple visual sequence to the model page:

```yaml
pipeline:
  - component: backbone
  - cpu: Top 300 proposals
  - component: decoder
```

List each component exactly once in compilation order. This is display metadata,
not executable orchestration. RF-DETR still runs backbone → CPU proposal selection
→ decoder inside `performance.py` and `accuracy.py`.

## Fidelity, accuracy, and performance

Each section is singular. Omit an unimplemented optional stage. Omitted
`build_profiles` selects all supported profiles; set the list only to restrict it.
There are no scope selectors, special orchestration IDs or result components.

`numerical_fidelity` declares `status`, `dataset`, seeded `selection` and output
names. `compile.py` computes the comparison and writes `fidelity.json`; these
settings are compile inputs. See [Accuracy and fidelity](local-accuracy-development.md).

`accuracy` declares its pinned `dataset`, optional status and nonempty
`required_metrics`. Its conventional `accuracy.py` owns evaluation and returns
one model/profile report. Previously recorded external results belong in
`reference_metrics` or documented model qualification evidence, not an executable
accuracy declaration.

`performance` is required for vision models. Its `required_metrics` includes
`model.latency.avg` and `model.inference_rate`; models can require additional
telemetry. Measurement settings, warmup, decoding and orchestration stay in
`performance.py`.

```yaml
performance:
  required_metrics:
    - model.latency.avg
    - model.inference_rate
    - board.power.avg
    - model.energy_per_inference
```

Both scripts return the same named `sima.model-report` contract. One result belongs
to the model and build profile and references every compiled input. Optional
component diagnostics do not replace the whole-model score. Metric names and units
come from [`taxonomies/metrics.json`](../taxonomies/metrics.json). See
[DevKit performance](benchmarking.md) for script arguments, report fields and
telemetry requirements. LLiMa can use the same report when a stage is implemented,
without assuming that all workloads use images or frames per second.

## Published metrics

`reference_metrics` records scores the source authors published for these weights.
Each entry keeps a pinned `source` page, dataset ID/split, measurement `conditions`
and metric values; optional `note` explains a necessary caveat. Do not invent
numbers or derive unpublished scores. Source scores use the existing reference
vocabulary in [`accuracy-metrics.yaml`](../taxonomies/accuracy-metrics.yaml).
They are shown beside compatible measured datasets and enter no stage key.

## Validation

```bash
model-registry validate models/<model_id>
model-registry validate-models --repository .
```

Validation checks the authored contract. Compilation, artifact checks and board
measurements establish that its implementation works.

## LLiMa checkpoint and deployment formats

A Hugging Face compilation uses `format: huggingface` and the conventional
`scripts/compile.sh`. Its source is a verified file set with `config.json` as the
source `main_file`. Declare the `runtime_entrypoint: generate` or `transcribe`
separately; do not copy prompt/image/text tensor inventories into YAML. A complete
LLiMa deployment is one component, including its vision encoder when present.

The compiled component contains `deployment/` and a sibling
`deployment-manifest.json`. `compile-outputs.json` carries `artifact_path`
(`model/deployment-manifest.json` for the `model` component) and
`artifact_sha256`, the SHA-256 of that JSON index. `compile-result.json`
marks `format: llima` and `layout: directory`; its `size_bytes` is the sum
of deployment file sizes.

A native full package is a deployment folder with `manifest.json` at its
root. Its component uses `format: llima`, `path: .`, and the compile index's
checksum. Both manifests list deployment files with exactly these fields:

```text
files: [{path, artifact, sha256, size_bytes}]
```

`path` is relative to the deployment root. `artifact` is the stable logical
name `file-<sha256(relative POSIX path)>`; it does not encode a storage URL.
In a published package, every reference names a `model-file` artifact on
that same package run. The root artifact is `package`, type `model-package`,
media type `application/json`. Native compile runs use root artifact `llima`,
type `model-pack`, also a JSON index. Their artifact metadata includes
`layout: directory`, `storage_backend: s3`, `deployment_size_bytes`, and
`file_count`.

Files use content-addressed S3 storage, reused between compile and package
publication. Each file must be at most 5 GiB until multipart uploads
([#292](https://github.com/sima-neat/models/issues/292)); aggregate deployment
size has no limit. Retrieval uses public resolved artifact URLs to preserve
the directory and verify each file's size and SHA-256. Existing ONNX/MPK
metadata and packages retain their format.
