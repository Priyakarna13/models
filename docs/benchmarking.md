# Model performance

Every vision model keeps a standalone `models/<id>/scripts/performance.py`.
The script owns the workload, warmup, sample count, decoding, orchestration and
telemetry windows. Customers can copy the model folder and run that script in
PyNeat without importing `model_registry`.

## The performance script

For one compiled artifact:

```sh
~/pyneat/bin/python models/resnet_50/scripts/performance.py \
  --mpk /path/to/model_mpk.tar.gz --output /tmp/resnet-performance
```

For RF-DETR, pass both artifacts:

```sh
~/pyneat/bin/python models/rf_detr_small/scripts/performance.py \
  --mpk backbone=/path/to/backbone_mpk.tar.gz \
  --mpk decoder=/path/to/decoder_mpk.tar.gz --output /tmp/rf-detr-performance
```

RF-DETR's script runs backbone, CPU proposal selection and decoder. Its
`TelemetryMonitor` window covers that complete measured loop. The script
returns one whole-model score and retains the raw telemetry. Optional
component metrics and execution plans are diagnostics, separate from the
whole-model score.

Use the corresponding model script as the runnable example. Save the complete
Core report as evidence, including board telemetry. Missing measurements must
be marked unavailable with a reason. Never turn missing telemetry into zero.

## Reproduce CI locally

With a verified compile output tree, run the public command on a Modalix board:

```sh
model-registry performance models/resnet_50 --build-profile w8a8 \
  --compiled-dir /path/to/builds --output /tmp/resnet-performance
```

The command verifies every artifact in `compile-outputs.json`, sets and checks
measurement clocks, runs the model script, validates its report and attaches
local provenance. No registry receipt or publication credential is needed.
`--python` selects the PyNeat interpreter. The command resolves the configured
Core selector to an immutable revision and requires that revision to be installed.
CI supplies its earlier selection through `--runtime`; pass the same runtime JSON
to reproduce a particular CI run. Both paths record the pin and observed runtime.

## Workload and measurements

Both performance and accuracy return the same envelope:

```json
{
  "schema": "sima.model-report",
  "schema_version": 1,
  "stage": "performance",
  "context": {"workload": "synthetic_image"},
  "metrics": [
    {"name": "model.inference_rate", "status": "measured", "value": 100,
     "unit": "inferences/s", "aggregation": "mean", "sample_count": 1000}
  ],
  "evidence": [{"path": "benchmark-report.json", "media_type": "application/json"}]
}
```

The schema is [`schemas/model-report.schema.json`](../schemas/model-report.schema.json).
Canonical metric names, units and directions live in
[`taxonomies/metrics.json`](../taxonomies/metrics.json). The browser and Vulcan
consume that same catalog. New unknown metrics remain visible under Other
measurements, with their producer-supplied unit and direction.

A metric's identity includes its name, merged context, aggregation and optional
component ID. Two text/image workloads can use the same metric name with
different context. Conflicting context and duplicate identities are rejected.
A required metric must have exactly one measured whole-model value. Component
scores cannot satisfy that requirement. GenAI metrics can use tokens/s or other
units without pretending to be image FPS.

The script supplies `context`, `metrics`, `evidence`, and optional `components`
and `details`. The runner alone attaches `model` and `provenance`. Evidence paths
must stay inside the output directory. The runner checks and records their
checksums. `components.<id>` may contain metrics, execution plans and layer
statistics. Keep only diagnostics that the model actually produces.

## Package benchmark

Multi-component models use the same command and report as single-component
models. There is no special benchmark ID and no last-component attachment.
The model script receives one `--mpk component=path` per component. Its required
headline metrics measure the complete pipeline.

`pipeline` in `model.yaml` supplies the existing static UI strip. It describes
component and CPU steps; it does not claim measured CPU timings or historical
execution graphs.

## Publication and stage keys

Model authors do not publish benchmark results directly. Publication commands
are reserved for hosted CI jobs.

CI selects the immutable runtime before calculating stage keys. A hosted job
binds the model, profile, source/script identity, expected stage keys and every
registered compile artifact before dispatching the board. A separate hosted
publisher compares the returned report against that retained manifest.

Each report belongs to the model and build profile. It creates a parentless
`model_benchmark` or `model_accuracy` run with links to every compiled input,
including its artifact ID and exact S3 object version. It never belongs to
RF-DETR's decoder compilation. Failed regression tests remain published with
`test_status: failed`, then fail the workflow.

Catalog entries carry `measurements[profile][stage]`, the expected stage keys.
Vulcan selects a completed report only when its key and complete input set match
the artifacts visible for that profile. An approved older build cannot inherit
a newer candidate's scores. Catalog pages read bounded summaries; model detail
loads the selected report lazily for layer statistics.

## Contract cutover

This contract replaces the active performance v1–v6 and accuracy v2 readers.
Deploy Vulcan's additive support before enabling the new Models publisher.
Archive old evidence and refresh active measurements during the coordinated
cutover; preserve compiled artifacts, candidate identities and approvals.
The PR itself does not archive or replace live registry data.

Use one report schema major for incompatible changes. Additive fields and new
metric labels do not require another major. Only the semantics of required
metrics enter stage keys; changing an unrelated label does not remeasure every
model. Numerical fidelity remains its separate compile-time contract.
