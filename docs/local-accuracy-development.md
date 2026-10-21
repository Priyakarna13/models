# Local accuracy development

A model's `scripts/accuracy.py` owns preprocessing, inference, orchestration,
decoding and task evaluation. It receives verified local paths and needs no
registry credentials. Keep it and `requirements-accuracy.txt` in the model folder.

## Declare accuracy

```yaml
accuracy:
  status: ready
  dataset: accuracy
  required_metrics:
    - model.accuracy.coco_map
```

The dataset name refers to an immutable entry under `datasets`. Omit accuracy
when it is not implemented. A recorded upstream score belongs under
`reference_metrics`, not in an executable accuracy stage.

## Run the same path as CI

```sh
model-registry accuracy models/rf_detr_small --build-profile w8a8_w16a16 \
  --compiled-dir /path/to/builds --output /tmp/rf-detr-accuracy
```

The compiled tree contains `<model>/<profile>/compile-outputs.json` and all
component outputs. The command verifies that complete set, resolves and checks
the declared dataset, installs accuracy requirements in a temporary environment
using the selected PyNeat interpreter, prepares clocks, and runs the script.
`--python` chooses that interpreter; `--dataset-cache` chooses the immutable
archive cache. The command pins and verifies the configured Core revision;
`--runtime` accepts a saved CI runtime selection for exact reproduction.
CI prefetches the dataset with reader credentials, then runs the
model script without those credentials.

For script iteration, run `accuracy.py --help` and pass its local dataset and
component paths directly. A short smoke run does not replace the full declared
accuracy measurement or its regression baseline.

## Report and regression tests

The script writes the requested `--output` file using
[`sima.model-report`](benchmarking.md#workload-and-measurements), stage `accuracy`.
Metrics are records with canonical names, values, units and context. Dataset
context records evaluated and available sample counts. Evaluated samples cannot
exceed the available set. Preserve the raw evaluation output in `evidence`.

The runner writes `accuracy-result.json` with model/profile identity, artifact
hashes, script hash, source revision and actual runtime. The publisher links the result to every compiled
input, independently of any one component's compile run.

`tests/tests.yaml` records expected metrics per profile. A measurement test calls
`measured_shortfalls(model_dir, "accuracy", accuracy)` and asserts an empty list.
The shared accuracy allowance is 5%. Required metrics must be finite measured
whole-model values; optional component measurements do not satisfy the gate.
Run the model's test on the output:

```sh
scripts/run_model_tests.sh accuracy models/rf_detr_small /tmp/rf-detr-accuracy/accuracy-result.json
```

A failed test publishes its evidence and failed test status, then fails CI.
Missing or malformed reports fail before publication.
