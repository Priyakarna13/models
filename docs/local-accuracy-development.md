# Accuracy and fidelity

`model.yaml` declares two kinds of quality evidence for a build profile's
whole model:

| Section | Question it answers | Measured by | Stage |
| --- | --- | --- | --- |
| `numerical_fidelity` | Does the compiled model produce the same numbers as the source model? | `scripts/compile.py` | compile |
| `accuracy` | How well does the compiled model do its task? | `scripts/accuracy.py` on a DevKit | accuracy |

Fidelity is a numerical diagnostic, not task accuracy. Neither is calibration:
calibration data feeds quantization and proves nothing about quality.

Complete [Datasets for local model development](datasets.md) first. Fidelity
and accuracy datasets resolve through Dataset Registry and need working
Artifactory access.

## Compile-time fidelity

`numerical_fidelity` asks the recipe to compare every build profile's compiled
model with the source model on a fixed sample of a registered dataset:

```yaml
numerical_fidelity:
  status: diagnostic
  dataset: default
  selection: {method: seeded_random_without_replacement, seed: 42, offset: 0, samples: 50}
  outputs: [logits]
```

`dataset` names a `datasets` entry, `outputs` names outputs of the model, and
`status` is `diagnostic`, `provisional`, or `ready`; see
[`numerical_fidelity`](metadata-schema.md#numerical_fidelity). The section is
a compile input: changing it recompiles the model's builds.

`scripts/run_model_compilation.sh` resolves the dataset with
`model-registry fetch-fidelity-dataset` and passes it to the recipe as
`--fidelity-dataset`. The recipe selects the declared samples, runs the
reference and the candidate, and writes `fidelity.json` in the profile's
output directory, beside the component directories.

The report follows
[`numerical-fidelity.schema.json`](https://github.com/sima-neat/models/blob/develop/schemas/numerical-fidelity.schema.json).
For each declared output it records cosine similarity, normalized RMSE, and
mean and maximum absolute error. Package assembly carries the report into the
profile's `default` package, and CI rejects a package that lacks it when
`numerical_fidelity` is declared, or carries one when it is not. CI publishes
the report on the profile's package run as a `numerical-fidelity` artifact, a
`correctness/numerical-fidelity` test result, and measurements of each metric.

## Accuracy on the DevKit

Use DevKit accuracy when a task metric needs the model's own decoding and
scoring, such as top-1 accuracy for a classifier or COCO mAP for a detector.
The model's `scripts/accuracy.py` runs every compiled MPK of a build profile
on a DevKit with PyNeat:

```yaml
accuracy:
  status: ready
  dataset: validation
  required_metrics: [coco_map]
```

`status` is `ready` or `provisional`, and `dataset` names a `datasets` entry.
Without `build_profiles`, every build profile is measured; see
[`accuracy`](metadata-schema.md#accuracy). Record the source model's published
scores in the top-level [`reference_metrics`](metadata-schema.md#reference-metrics),
not in `accuracy`, so the model page shows them before the first DevKit run.

Optional `scripts/requirements-accuracy.txt` pins extra wheels. They install
into a separate folder beside PyNeat's environment, never into it.

### The accuracy script

`scripts/accuracy.py` receives:

- `--mpk <component>=<path>` once for each component, `model=<path>` for a
  single-component model;
- `--dataset <directory>`, the extracted dataset archive; and
- `--output <directory>`.

It scores the whole model and hands the scores to `tools/model_report`, which
writes `model-report.json` into the output directory:

```python
report = model_report.accuracy(
    {"top1_accuracy": top1 / len(names), "top5_accuracy": top5 / len(names)},
    dataset={"split": VALIDATION_SPLIT, "evaluated_samples": len(names), "available_samples": len(available)},
    details={"inference_seconds": round(time.monotonic() - started, 1)},
)
report.save(args.output)
```

Every score is a ratio from 0 to 1, named as an accuracy metric in
[`tools/model_report/metrics.json`](../tools/model_report/metrics.json), and
every `required_metrics` entry must be present. Existing scripts also accept
`--limit <images>` to evaluate a subset for a quick local check; CI never
passes it.

### Run it on a DevKit

`scripts/run_model_accuracy.sh` runs the stage the way CI does, against the
output of `scripts/run_model_compilation.sh`:

```bash
REGISTRY_PYTHON=<python with model-registry and dataset-registry> \
PYNEAT_PYTHON=~/pyneat/bin/python \
  scripts/run_model_accuracy.sh models/<model-id> <build-profile> <compile-output-root> <result-directory>
```

`<compile-output-root>` holds `<model-id>/<build-profile>/compile-outputs.json`
and the MPKs it lists. The script:

1. resolves the declared dataset into `DATASET_CACHE` (default
   `/media/nvme/datasets/artifacts`), holding a lock so concurrent runs share
   one download;
2. lists each component's MPK with `model-registry model-component-mpks`;
3. extracts the dataset under `WORK_DIR` (default `/media/nvme/neat-tests/`)
   and removes that directory when it exits;
4. installs `requirements-accuracy.txt` beside PyNeat;
5. runs `accuracy.py` with `PYTHONPATH` including `tools`, which writes
   `model-report.json`, and saves its output to `accuracy.log`; and
6. runs `model-registry check-report`, which validates the report against
   [`model-report.schema.json`](../schemas/model-report.schema.json) and the
   `accuracy` declaration and writes the board's installation to
   `environment.json`.

`REGISTRY_PYTHON` needs the full Model Registry tooling from this checkout
(`python -m pip install .`) and Dataset Registry, because the dataset
resolution and report checks are not in the author-only `model-registry`
package. It exits without running anything when the build profile is not
measured.

For a quicker check, resolve the dataset with Dataset Registry, extract it, and
run the script directly:

```bash
PYTHONPATH=tools ~/pyneat/bin/python models/resnet_50/scripts/accuracy.py \
  --mpk model=<path to MPK> --dataset <extracted dataset> \
  --output /tmp/resnet50-accuracy --limit 500
```

A model whose `accuracy.py` imports another file under `scripts/`, such as
ResNet-50's `calibration.py`, needs that file beside it.

### Report and publication

`model-report.json` follows
[`model-report.schema.json`](../schemas/model-report.schema.json): the stage,
the PyNeat version and build, the evaluated `dataset` split and sample counts,
one number per metric, and free-form `details`. The model, build profile, MPK
digests, and board are not report fields; publication records them on the run.

On the DevKit, `scripts/run_model_tests.sh` runs the model's
`tests/measure/test_accuracy.py` on the report, which compares each metric
with the build profile's expected value in `tests/tests.yaml` ([set up
`tests/tests.yaml`](adding-a-model.md#set-up-teststestsyaml)). CI publishes the
report with `model-registry publish-report` as one `model_accuracy` registry
run under the build profile's `model_package` run. Its `run_fingerprint` is the
accuracy benchmark fingerprint, and the run carries one `compiled-model-<component>`
input per component, the `model-report` and log artifacts, an `accuracy/model`
test result whose summary is the whole report and whose status is the test
outcome (`passed`, `failed`, or `skipped`), and one measurement per metric. A
failed test still publishes the report, then fails the workflow run. See [CI
and registry publication](benchmarking.md#ci-and-registry-publication) for the
shared flow. The model browser shows the first four metrics as tiles and lists
the rest in a table, each with its reference score when `reference_metrics`
records one for the same dataset, and the evaluated sample count.

## Qualification recorded elsewhere

Qualification measured outside the registry stages, such as a held-out
evaluation from training or a pending hardware qualification, goes under
`qualification_evidence`. Each entry needs a `status`; other fields are
free-form. No stage runs it and no fingerprint reads it. SSD-MobileNet uses such
an entry to record its quantization evaluation.

Compiler-enforced settings belong in the build profile's component `parameters`,
which enter the compile key. SSD-MobileNet keeps its unchanged raw-head gates in
`parameters.quantization_quality_thresholds`; its `qualification_evidence`
entry retains the evaluation evidence.
