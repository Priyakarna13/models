# Define the model contract

A Model Registry contribution describes a product that developers can find,
build, test, and download reproducibly. For a simple model, that product is one
ONNX input and one compiled MPK. A larger product can contain several compiled
components with one derived download package.
`model.yaml` joins those pieces, and the model's `scripts/` directory holds the
code that compiles and measures them.

This guide ends when the contract validates. Then follow [Qualify
locally](local-development.md) to compile and measure every build before you
open a pull request.

## 1. Decide what the product contains

Before creating files, identify:

- the product and its responsible owner;
- each independently compiled ONNX component;
- whether a component source is one file or a set of related files;
- the public entrypoint, concrete compiler inputs, preprocessing and decoding;
- checkpoint and deployment files such as tokenizers and configuration;
- the target-wide build profiles and component-specific compiler parameters;
- how the components compose and what users download;
- the calibration, fidelity, and accuracy data each build needs.

Do not add a product whose source, ownership, or license is ambiguous. Record
ownership in `CODEOWNERS`; do not add owner or maturity boilerplate to YAML.

## 2. Create the directory

The directory name supplies the lowercase snake-case model ID. A model directory may contain only these entries:

```text
models/<model-id>/
  README.md
  hero.png                      # 1200x480 catalog card image
  model.yaml                    # the model's only configuration file
  scripts/
    compile.py                  # compile stage: writes one MPK per component
    performance.py              # standalone customer benchmark, when the model declares one
    accuracy.py                 # accuracy stage, when model.yaml declares DevKit accuracy
    requirements-accuracy.txt   # optional wheels for accuracy.py
    convert.py                  # only when model.yaml declares a conversion
    requirements-convert.txt    # required with convert.py
    <helper>.py                 # model-specific surgery, calibration, or shared code
  tests/                        # the model's tests; see step 10
    tests.yaml                  # which stage tests run, and the expected DevKit values
    build/test_conversion.py    # only with convert.py
    build/test_compile.py
    measure/test_accuracy.py    # only when model.yaml declares DevKit accuracy
    measure/test_performance.py # required with performance
    fixtures/                   # optional test data
```

Put declarative model settings in `model.yaml` and executable behavior in its
scripts. Keep one blank line between top-level YAML sections. [Step 11](#11-write-the-readme) describes the README.

Keep board-only code inside `performance.py` or `accuracy.py`. For ONNX, any other
`.py` or `.sh` file under `scripts/` is a compile input, so a helper that
`accuracy.py` imports recompiles the model when it changes. See [step
9](#9-know-what-a-change-reruns).

## 3. Pin every input

The compilation declares one source shared by its ordered components. Use `type: file` for one artifact and
a file set when the recipe needs related files in a fixed relative layout.
Every file retains its SHA-256 and byte count. Hugging Face file sets can share
one immutable repository/revision declaration; see [source shorthand](metadata-schema.md#components-and-source-sets).
Exceptional members can keep explicit URIs.

Hugging Face sources use a full 40-character repository commit. Approved HTTPS
sources include pinned Hugging Face resources and GitHub release assets.
Mutable branches, credentials, query parameters, fragments, and unapproved
hosts are rejected.

Record hashes from the exact bytes the registry will fetch:

```bash
shasum -a 256 /path/to/model.onnx
wc -c < /path/to/model.onnx
```

Do not commit downloaded inputs. Once they are declared, check them:

```bash
model-registry validate models/<model-id>
model-registry fetch models/<model-id> --component <component-id>
```

`--component` may be omitted when the model has one component.

When upstream publishes a checkpoint instead of ONNX, declare it under
`conversion` and provide `scripts/convert.py` with
`scripts/requirements-convert.txt`. `convert.py` receives `--output-dir <directory>` and writes `model.onnx` there.
`scripts/run_model_conversion.sh` runs it in a fresh Python 3.12 environment:

```bash
scripts/run_model_conversion.sh models/<model-id> /tmp/<model-id>-export
```

The **Vulcan Model Convert** workflow runs the same script, publishes
`model.onnx`, and prints the `source` block to pin in `model.yaml`. Only that
pin reaches the compile stage; editing `convert.py` recompiles nothing.

## 4. Describe compilation

Declare the input format, source and ordered output components under
`compilation`. A single model uses `components: [model]`; RF-DETR uses
`components: [backbone, decoder]`. The conventional recipe is `scripts/compile.py`
for ONNX or `scripts/compile.sh` for Hugging Face checkpoints.

Put preprocessing, graph surgery, tensor boundaries and decoding in executable
model scripts. Test the actual graph and compiled artifacts. The README explains
what an application passes between components. Do not duplicate tensor inventories,
roles or recipe paths in YAML. Native models can declare `runtime_entrypoint:
generate` or `transcribe`; the source file selector is named `main_file`.

## 5. Add datasets

Declare immutable pins once under `datasets` and use their aliases in
`build_profiles.<profile>.calibration_dataset`, `numerical_fidelity.dataset`
and `accuracy.dataset`. Dataset Registry pins keep their ID, variant, SHA-256
and byte count. An existing non-registry archive can use `source` under its alias.

Keep sample selection in the consuming stage. Calibration supplies quantization
ranges; it is not accuracy evidence. See [Datasets for local model development](datasets.md).

## 6. Define build profiles and compile invocations

Each profile is one compile invocation for all declared components:

```yaml
build_profiles:
  w8a8:
    calibration_dataset: calibration
    components:
      model:
        parameters: {precision: w8a8}
```

Use precision tokens from [`naming-conventions.yaml`](../naming-conventions.yaml)
in component order, collapsing repeats. Each component keeps its own explicit
compiler parameters. The target is Modalix; the compiler is pinned in
`deps/manifest.json`. A proven unsupported profile needs a reason.

The loader derives an invocation ID: the component ID for a singleton, `default`
for several components. No `compile_invocations` section is authored. All
components share an effective runner class. Elevated runner requests require
[approval](runner-class-requests.md).

ONNX recipes implement this interface:

```text
scripts/compile.py --model <verified-source> --output-dir <new-directory> \
  --build-profile <profile-id> --compile-invocation <invocation-id> \
  [--dataset <verified-calibration>] [--fidelity-dataset <verified-fidelity-dataset>]
```

Read the effective definition with `model_registry.model.load_model` so aliases
and conventional paths resolve consistently. Write one component directory per
output. A singleton MPK is `<model_id>_<profile_id>_mpk.tar.gz`; multiple components
add `_<component_id>` before `_mpk.tar.gz`. Pass the matching basename to the
compiler. Downloading undeclared data inside a recipe is not allowed.

`compile-outputs.json` records each output path and SHA-256. A grouped recipe
also records the derived graph hash for each component. When numerical fidelity
is declared, write one `fidelity.json` and one `correctness` pointer to it. The
pointer locates compile-time evidence, not a parent for benchmark results.
The shared runner records singleton outputs if the recipe has not done so.

## 7. Describe the display pipeline

Package layout is derived: one component yields its MPK; several components
form the default ZIP; native LLiMa deployments preserve their directory layout.
Do not declare packages, compositions or resources in YAML.

For multiple components, an optional `pipeline` lists each component once in
compilation order with readable CPU steps between them. It is display-only:

```yaml
pipeline:
  - component: backbone
  - cpu: Top 300 proposals
  - component: decoder
```

Actual orchestration remains inside `performance.py` and `accuracy.py`.

## 8. Declare numerical fidelity, accuracy and performance

Each stage has one section, without an ID, scope or result component. Omit an
unimplemented optional stage. Omitted `build_profiles` selects all supported
profiles; use an explicit list for restrictions such as INT8-only accuracy.

| Section | Implementation | Result |
| --- | --- | --- |
| `numerical_fidelity` | `compile.py` compares compiled and source outputs | compile-time diagnostic |
| `accuracy` | `accuracy.py` evaluates the complete model | one model/profile report |
| `performance` | `performance.py` measures the complete model | one model/profile report |

Vision models must declare performance with `model.latency.avg` and
`model.inference_rate`. Require power/energy metrics where the workload captures
them. Accuracy declares its pinned dataset and nonempty `required_metrics`.
Metric names and units come from `taxonomies/metrics.json`.

Scripts control their own workload, orchestration, telemetry monitor, sample count,
duration and layer statistics. They return the common `sima.model-report` format;
optional component diagnostics do not replace the whole-model result. The registry
validates the report and publishes it against the model/profile and every compiled
input. No special ID changes that behavior. See [performance](benchmarking.md)
and [accuracy and fidelity](local-accuracy-development.md).

Record source authors' scores under [`reference_metrics`](metadata-schema.md#published-metrics),
with their exact dataset, conditions and pinned page. Keep model-specific
qualification evidence alongside the implementation when it is not a published
source score. Do not represent recorded scores as executable accuracy stages.

## 9. Know what a change reruns

Each stage has a stage key, a SHA-256 of everything its result depends on. CI
compiles a build only when neither the branch's catalog nor, on a feature
branch, a base branch's catalog records a build with its current compile key.
It runs a DevKit stage only when no completed registry run carries that stage's
current key. Models with `build_on_demand: true` require an explicit manual
selection before either stage can run; see [on-demand builds](#on-demand-builds).

| Stage | Key inputs |
| --- | --- |
| compile (ONNX) | source identity and layout, dependencies, component parameters and calibration, target and compatibility, consumed fidelity settings, compile/surgery code, the Model Compiler version, `scripts/run_model_compilation.sh`, and the shared model loader |
| compile (LLiMa) | checkpoint content hashes, the target, component graph `parameters`, and the pinned Model Compiler identity including its build revision |
| performance | the complete set of compile runs and artifact digests, `performance`, `performance.py`, shared execution/report validation code, the named report contract, semantics of used metrics, and the resolved immutable runtime |
| accuracy | the complete set of compile runs and artifact digests, `accuracy`, `accuracy.py`, its requirements, shared execution/report validation code, the named report contract, semantics of used metrics, and the resolved immutable runtime |

ONNX compile code includes every `.py` and `.sh` under `scripts/` except
`performance.py`, `accuracy.py`, `requirements-accuracy.txt`, `convert.py` and
`requirements-convert.txt`. The shared model loader is an input to every ONNX
compile key; descriptive preprocessing and surgery declarations are not.

A new compile run changes the performance and accuracy keys too, so a
recompiled build is always remeasured.

| Change | Reruns |
| --- | --- |
| README, display metadata, `reference_metrics`, `convert.py`, registry tooling outside a stage's tooling | nothing |
| `build_on_demand` | nothing; changes scheduling and catalog metadata only |
| a model's tests under `tests/`, or `scripts/run_model_tests.sh` | nothing; a changed test applies the next time its stage runs |
| a model's `performance.py` or performance declaration | affected performance stages |
| `accuracy.py`, its requirements, or the `accuracy` entry | that model's accuracy |
| semantic definition of a used metric | stages using that metric; display labels rerun nothing |
| the resolved DevKit runtime or shared execution/report validation code | affected performance and accuracy stages |
| a source model or graph parameter | that model's affected builds, then their DevKit stages |
| an ONNX compile script or other ONNX compile input | that model's affected builds, then their DevKit stages |
| the Model Compiler pin | every build, then every DevKit stage |
| `scripts/run_model_compilation.sh` | ONNX builds, then their DevKit stages |
| the shared model loader | ONNX builds, then their DevKit stages |
| LLiMa scripts, registry helpers, checkpoint repository name, job count, logging or runner class | no LLiMa compilation |

The keys are computed by `compile_key`, `performance_key` and `accuracy_key`
in `tools/model_registry/stage_keys.py`.

To apply a changed test at once, dispatch **Vulcan Model Orchestrator** with
`models` set to the model and `force_rebuild_all`: it recompiles the model's
builds, which runs the compile tests and remeasures every build. A changed
conversion test applies on the next conversion dispatch.

## 10. Write the model's tests

Each model carries its own tests in `tests/`, one file per stage, named by
what it checks. A test either exercises the model's scripts, or checks the real
output of its stage by taking that output's fixture from the root
`conftest.py`:

| File | Exercises | Stage-output fixture | That test runs in CI |
| --- | --- | --- | --- |
| `build/test_conversion.py` | `convert.py` | `converted`: the ONNX path | after conversion, before the ONNX upload |
| `build/test_compile.py` | `compile.py`, calibration and surgery | `compiled`: the build's output directory | after compilation, before the build is published |
| `measure/test_accuracy.py` | `accuracy.py` | `accuracy`: the accuracy result | after the DevKit accuracy stage, before publication |
| `measure/test_performance.py` | `performance.py` | `performance`: the whole-model performance report | after the DevKit performance stage, before publication |

A test without a stage-output fixture runs in `python -m pytest` on every pull
request. Import a model script with the `load_script` fixture, as in
`load_script("compile")`; it stubs Model Compiler and DevKit modules the host
lacks. `build_profile` names the build profile of the given output.

A failing conversion or compile test stops the ONNX upload or the build's
publication. A failing measurement test publishes its result with test status
`failed`, then fails the run.

### Set up `tests/tests.yaml`

`tests/tests.yaml` has one entry per stage the model has: `conversion` when
the model has `scripts/convert.py`, `compile` always, `accuracy` when
`model.yaml` declares an accuracy section, and `performance` when
it declares performance. `model-registry validate` rejects a missing or extra
entry.

Each entry sets `run_tests`. `true` runs the stage's test file. `false` skips
it and needs a `skip_reason`; a skipped measurement still publishes its result,
with test status `skipped`. Skip a stage only while its tests cannot run, for
example before the model has a published result to expect values from.

A running `accuracy` or `performance` entry also states what the model must
deliver on a DevKit in `expected`: the latest published value of every
measured build. For accuracy, supply a nonempty map of registered finite numeric
expectations per build profile, covering at least its `required_metrics`. Keep
optional expectations such as top-5: a missing expected measurement fails.
For performance, use `model.inference_rate` per build profile. Take the values from the model browser's latest
`develop` results, rounded to 4 decimal places for accuracy and 1 for
throughput.

A measured value passes when it falls at most 5% below its expected accuracy
or 10% below its expected throughput. For a metric whose result says lower is
better, it may rise by that much instead. Every model uses these limits:
`ACCURACY_ALLOWED_DROP` and `PERFORMANCE_ALLOWED_DROP` in
`tools/model_registry/model_tests.py`.

```yaml
compile:
  run_tests: true
accuracy:
  run_tests: true
  expected:
    w8a8:
      model.accuracy.top1_accuracy: 0.7764
      model.accuracy.top5_accuracy: 0.9356
performance:
  run_tests: true
  expected:
    w16a16:
      model.inference_rate: 578.6
    w8a8:
      model.inference_rate: 1491.4
```

`model-registry validate` requires an `expected` value for every build the
stage measures. The measurement test compares the result with these values
through one shared function. A performance test also checks that the result
records the execution plan the model browser draws:

```python
from pathlib import Path
from typing import Any

from model_registry.model_tests import measured_shortfalls


def test_performance_meets_its_expected_values(performance: dict[str, Any], model_dir: Path) -> None:
    assert measured_shortfalls(model_dir, "performance", performance) == []


def test_performance_records_the_execution_plan_the_browser_draws(performance: dict[str, Any]) -> None:
    for component in performance["components"].values():
        plan = component["execution_plan"]
        assert (plan["schema"], plan["schema_version"]) == ("sima.neat.execution_plan", 1)
        assert plan["nodes"] and plan["edges"]
```

Update `expected` when a reviewed change moves a published value, such as a
new compiler or a faster script, in the same pull request or right after its
results publish. No stage key reads `tests/`, so editing `tests.yaml` reruns
nothing; the new values apply the next time the stage runs. To apply them
now, dispatch `Vulcan Model Orchestrator` with `models` set to the model and
`force_rebuild_all`, which recompiles and remeasures it.

### Run the tests locally

From a virtual environment with this checkout's development extra installed
(`python -m pip install '.[dev]'`), run the model's tests, and run one stage's
file on an output you produced:

```bash
python -m pytest models/<model-id>/tests
REGISTRY_PYTHON="$(command -v python)" \
  scripts/run_model_tests.sh performance models/<model-id> <performance-result.json>
```

`run_model_tests.sh` prints `passed`, `failed`, or `skipped`, as CI publishes
it.

## 11. Write the README

The model page shows the README after what it reads from `model.yaml` and the
registry runs: metrics, benchmarks, variants, components, license, and source
links. The README adds what an application developer needs to run the model:
what it finds, how to prepare its input, and how to read its outputs. Write it
from this template:

````markdown
# <display_name>

![<display_name> model hero](hero.png)

## Overview

<50 to 120 words of prose: what the model finds, the input it takes, what it
returns, and the components it runs as.>

## Inputs and outputs

**Input.** <Each preparation step with its exact values: resize, padding,
color order, scaling, and normalization.>

**Between the components.** <Only for a model with several components: what
the application computes on the CPU and which tensors it passes on.>

**Outputs.** <Each actual output tensor by name, and what it holds.>

1. <Each decode step, in order, one line per step.>

<The class-index mapping, and the Neat decoder for these outputs, if one exists.>

## Building applications

Run the package with the [Neat Model API](https://developer.sima.ai/software/develop-apps/development-workflow/model).
For complete applications, start from the [Neat example apps](https://developer.sima.ai/examples).
````

Start from a README of the same kind:
[`yolo26n`](https://github.com/sima-neat/models/blob/develop/models/yolo26n/README.md)
for a one-component detector,
[`rf_detr_nano`](https://github.com/sima-neat/models/blob/develop/models/rf_detr_nano/README.md)
for a CPU step between two components, and
[`resnet_50`](https://github.com/sima-neat/models/blob/develop/models/resnet_50/README.md)
for a classifier. `model-registry validate-models` checks the structure, the
word limits, the Building applications text, and the internal terms a README
leaves out. The writing rules are in
[`AGENTS.md`](https://github.com/sima-neat/models/blob/develop/AGENTS.md#model-readme-requirements).

## 12. Validate the contract

```bash
model-registry validate models/<model-id>
model-registry validate-models --repository .
```

These checks establish that the contract is coherent, not that it compiles or
runs. Continue with [Qualify locally](local-development.md) and complete the
full loop for every build before submitting.


## On-demand builds

Set top-level `build_on_demand: true` for any model whose expensive builds
should run only on request. The default is `false`; existing automatic ONNX
builds keep their scheduling. Pushes validate on-demand models and publish
their metadata, but never compile or measure them, even when inputs change.
This flag is not part of the compile fingerprint.

To build Qwen3.5-0.8B, dispatch **Vulcan Model Orchestrator** with
`models=qwen3_5_0_8b` and `profiles=w4a16`. Both selections are
required for an on-demand build. Use staging; production candidate publication
is not enabled. Direct `plan-stages` calls must also pass `--include-on-demand`.
The workflow publishes all invocations and packages as one retained candidate.
Its summary shows the candidate ID, immutable digest, source and protection;
on a feature branch it also selects the finalized deployment for that branch's
website and CLI downloads. This does not approve it for Develop or Main.
Retry failed jobs within the seven-day assembly deadline; after expiry start a
new workflow run. See the [architecture contract](https://github.com/sima-neat/models/blob/develop/design-doc/model-registry-architecture.md#on-demand-builds)
for retention and the separate promotion workflow.

## LLiMa language and vision-language models

Declare a complete language model as one component, with `format: huggingface`,
`runtime_entrypoint: generate`, and a pinned Hugging Face file set whose source `main_file`
is `config.json`. Share the repository/revision while retaining every weight,
tokenizer, processor and chat-template file's path, size and SHA-256. The conventional
recipe is `scripts/compile.sh`. The verified downloader assembles the checkpoint
directory; do not list LLiMa's generated layers as components.

A build profile names one compiled configuration. Start with one profile per
model and set `max_num_tokens: 4096` explicitly in its component `parameters`,
alongside `language_group_size`. Add `input_height` and `input_width` only for
VLMs needing a resolution override. Omit both for text-only LLMs and VLMs whose
native resolution is fixed. Published context and image dimensions come from
compiled parameters; no copied image tensor inventory is needed.

The shared runner forwards these parameters as compiler options after the
checkpoint and output arguments; `compile.sh` must pass them to `llima-compile`.
Keep these values in `model.yaml` rather than duplicating them in the recipe.

The executable recipe takes two positional paths followed by compiler options.
After compilation, it writes a standalone `compile.sh` into its output folder,
recording the complete arguments it executed. The Qwen recipe shows this pattern.

Run it through the LLiMa runner, with Model Compiler and LLiMa activated. The
runner passes the selected profile to the recipe as `BUILD_PROFILE_ID`:

```bash
scripts/run_llima_compilation.sh models/qwen3_5_0_8b w4a16 model /path/to/output
```

The runner downloads the pinned files and passes the recipe a temporary symlink
named after `upstream.repository`, preserving the verified cache and giving
generated files the checkpoint's name. It then runs
`llima-deploy` into `<output-root>/<model>/<profile>/<component>/deployment/`.
This folder contains `devkit/`, `elf_files/`, any `npy_files/`, and the standalone
`compile.sh` with the selected context, image dimensions and other flags. Run
`model-registry verify-artifact <component-output> --write-manifest` to check
the deployment and write `deployment-manifest.json` beside it. The verifier
checksums every file and rejects folders without ELFs, unsafe paths, and links.

`compile-outputs.json` uses `artifact_path: model/deployment-manifest.json`
for Qwen's `model` component and `artifact_sha256` for that index file's hash.
ONNX outputs retain their MPK fields. Publication registers the deployment's
files individually, with a JSON index named `llima`. A native full package is
the deployment folder with `manifest.json` at its root; its component uses
`format: llima`, `path: .`. Retrieval preserves the folder and verifies each
file. See the [manifest and metadata contract](metadata-schema.md#llima-checkpoint-and-deployment-formats).
No native stage creates a deployment ZIP.

Whisper sets `runtime_entrypoint: transcribe` and uses a model-specific recipe calling
`WhisperModel` directly instead of `llima-compile`. Its profile selects A16W8
and the Whisper mask and log-probe options, without LLM context or image settings.
The runner forwards boolean parameters as `--name` or `--no-name`.

The native compile key covers checkpoint content, the target, graph parameters
and the pinned compiler identity. Declare graph-changing options in component
`parameters`, including precision, group size, context length and image dimensions.
Script edits, logging, job counts and runner selection do not invalidate it.
README and hero changes remain outside compile identities. The YAML migration
plans one ONNX rebuild rather than recreating removed descriptive inputs solely
to preserve their old hashes; LLiMa graph identities remain unchanged.

Omit `datasets`, `numerical_fidelity`, `accuracy`, and `performance` when no corresponding
stage is implemented. A manual Qwen build compiles and publishes only; performance
and accuracy are separate follow-ups [#294](https://github.com/sima-neat/models/issues/294)
and [#295](https://github.com/sima-neat/models/issues/295). The default Qwen profile
requests the expensive runner with twenty compile jobs, 4096 tokens, 448×448 input,
and language group size 128.
