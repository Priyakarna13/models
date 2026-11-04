# LeWAM PushT — staged registration

This is a source implementation under review, **not a published or qualified model**.
The user requested staging for normal CI publication. No fake source URLs, active
catalog entry, approved deployment, or transferred performance score is created.

## What is implemented

- A current-schema four-component model template, taxonomy, action-success metric,
  dataset metric requirements, application README and deterministic hero.
- `stage.py`: interface-checks and hashes four self-contained ONNX files into a
  new directory outside Git. It does not claim source numerical equivalence.
- `stage_data.py`: verifies and reproducibly archives the original 150-row
  PushT source fixture pack. It does not relabel it as held-out data, calibration,
  or a complete runtime evaluation bundle.
- `ci_sources.py`: prepares four ordinary `model.onnx` publication inputs,
  then verifies the four returned CI pins before writing a combined receipt.
- `promote.py`: checks published source identities with the registry Fetcher and
  atomically creates `models/lewam_pusht` from actual publication receipts.
- The ordinary grouped compile CLI, standard MPK candidate production, BF16 activations and
  weights, both FP32 normalization options disabled, explicit real calibration,
  source hash verification, fresh child processes, and zero-exit compiler failure
  handling. It writes **candidate-outputs.json**, not publishable compile outputs.
  Archive checks follow RF-DETR's `*_mpk.json` convention and verify compiler
  version, logical tensor byte sizes, one MLA executable/plugin, terminal
  pass-through output, and absence of host fallback/private runtime libraries.
- A public-PyNeat four-runner baseline with explicit full19 packing, eight own-
  feedback updates, all ten outputs, exact separate FP32 host arithmetic, bounded
  request count, poisoned-runner rejection and explicit cleanup.
- Standalone whole-replan measurement and closed-loop PushT adapter code. Neither
  is declared as a qualified benchmark in the initial promoted model.

## Stage source artifacts

From the repository, with its Python dependencies available:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/stage.py \
  --graph front=/absolute/path/front.onnx \
  --graph body=/absolute/path/body.onnx \
  --graph head=/absolute/path/head.onnx \
  --graph policy=/absolute/path/policy.onnx \
  --output /absolute/path/new-publication-input
```

The four files and `source-stage.json` are source publication inputs. Keep large
files out of Git. The checkpoint/source fields identify the intended source;
the staging check does **not** prove weight lineage or numerical equivalence.

From the repository on the host, prepare inputs for the existing publisher:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/ci_sources.py prepare \
  --source-stage /absolute/path/new-publication-input \
  --output /absolute/path/new-ci-input

PYTHONPATH=tools python contrib/lewam_pusht/stage_data.py \
  --evaluation-pack /absolute/path/original-evaluation-pack \
  --output /absolute/path/new-evaluation-input
```

`new-ci-input/<component>/model.onnx` contains each original graph unchanged.
`ci-input.json` binds all four files and the source receipt. The evaluation
directory contains the original three files, a deterministic
`pusht-eval-source.tar.gz` and `evaluation-stage.json`.

The current single-ONNX conversion workflow cannot directly consume a staged
four-file set. A trusted CI handoff must run its existing **Publish ONNX and print
source pin** step once per prepared component, preserving the conditional S3
write, KMS encryption, checksum and download verification. Use model ID
`lewam_pusht`; the key is
`models-onnx/lewam_pusht/<sha256>/<run-id>-<attempt>/model.onnx`.
The content hash distinguishes components. Retain each returned `source.yaml`.
Do not run contributor code on a privileged PR runner or bypass normal review.
No workflow has been dispatched or upload performed by these staging commands.

After that CI publication, bind the real returned pins on the host:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/ci_sources.py bind \
  --source-stage /absolute/path/new-publication-input \
  --pin front=/absolute/path/front-source.yaml \
  --pin body=/absolute/path/body-source.yaml \
  --pin head=/absolute/path/head-source.yaml \
  --pin policy=/absolute/path/policy-source.yaml \
  --cache /absolute/path/cache \
  --output /absolute/path/published-source-stage.json
```

This downloads and verifies all four hashes and sizes; one failure prevents
receipt creation. Do not substitute checkpoint URLs for ONNX URLs or rename a
raw ELF as an MPK. The evaluation source archive needs its own Dataset Registry
publication and runtime-data adaptation; it is not `pusht/cal`.

Publish the real calibration archive through Dataset Registry as `pusht/cal`.
Its six top-level files are `calibration.json`, the exact original
`selection.json`, and one `<component>.npz` each.
The JSON schema is `lewam.calibration.v1`, `source` is `source_trajectories`,
and `checkpoint_sha256` is the pinned original checkpoint. It records nonempty,
disjoint `episode_ids` and `evaluation_episode_ids`, and `components.<id>.sha256`.
The loader checks the exclusion list against the hash-pinned original selection,
not merely against a caller-supplied list.
Each NPZ contains finite FP32 NCHW `inputs` and integer `episode_ids`, one per input.
No synthetic replacement, repetition, guessed episode IDs or evaluation leakage.

After publication returns real receipts:

```bash
PYTHONPATH=tools python contrib/lewam_pusht/promote.py \
  --repository "$PWD" --cache /absolute/path/cache \
  --source-receipt /absolute/path/published-source-stage.json \
  --calibration-receipt /absolute/path/pusht-cal-pin.json
```

The calibration pin has exactly `id`, `variant`, `sha256`, `size_bytes`, as returned
by Dataset Registry. Promotion does not grant approval, publish a package, or
check Dataset Registry server availability; the normal compile resolver does.

## Remaining release gates — not implemented acceptance

1. Reproduce the front's proven normalization lowering as a standard compiler
   input/package. A shape-valid generic front ONNX is not equivalent evidence for
   the existing direct-compiler Front ELF. Porting that route is unfinished.
2. Compile all four components using `deps/manifest.json`, not local Model Compiler 2.1.3.
   Preserve actual imported-source/compiler receipts and perform operator audit
   before launching a heavy build.
3. Implement the current-compiler final-ELF precision qualification producer and
   source-equivalence checks. Archive metadata checks are implemented but do not
   prove native layout or numerical parity. Only after qualification replace
   the deliberate candidate-only stop with production of `compile-outputs.json`.
   The future `qualification.json` must bind every exact MPK and actual proof;
   manually setting booleans is not qualification. Existing compile tests reject
   candidate-only output and missing qualification.
4. Admit a DevKit freshly and qualify public-Neat output/layout, preprocessing,
   actual completion and cleanup. The old private worker is not used here.
5. Adapt and publish registered evaluation/request bundles. Staging the original
   source pack does not provide the missing runtime fields. The accuracy loader's
   schema is `lewam.episodes.v1`; it requires source-recorded reset seeds, package
   versions, arrays and the byte-pinned original 150-row non-held-out selection.
   Add a matching `requirements-accuracy.txt` from that recorded environment.
   Validate its reset/render/RNG scheduling against the upstream evaluator before accepting
   scores. The source mode is direct policy, never gradient planning.
6. The staged performance script currently measures RGB224 to ten actions, with
   Neat normalization included. It is **not** the registry's canonical 720p image
   protocol. Agree and implement the multi-image action-model protocol in the
   benchmark documentation/runner/workflow/tests before enabling `performance`.
7. Add finite expectations from actual first published develop reports. Preserve
   shared 10% throughput/5% accuracy regression limits. Until a baseline exists,
   use an explicit `run_tests: false` and `skip_reason`, never a guessed number.
8. Finish and qualify the separate Apps example. No private runner or runtime DSO
   is embedded as a package resource.

The old 15.393 image-ready replans/s and 12/14 numerical result are historical,
not results of this public runtime. No new accuracy, FPS or release PASS is claimed.

## Host checks

```bash
PYTHONPATH=tools python -m pytest tests/test_lewam_onboarding.py
PYTHONPATH=tools python -m model_registry validate-models --repository .
```

Host tests exercise real graph serialization, binding and mathematical code;
native calls and publication are replaced only in explicitly labelled test doubles.

The shared metric vocabulary is a benchmark fingerprint input. Adding the PushT
metric can schedule existing models' performance/accuracy stages in ordinary CI;
it does not change their ONNX compile fingerprints. Do not bypass that policy or
transfer old benchmark results to the new runtime.
