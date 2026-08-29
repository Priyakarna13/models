# DepthART Base strict INT8

This staging entry packages the qualified 448x576 **relative-depth** DepthART
Base candidate. The public Modalix ABI is INT8 NHWC in and INT8 NHWC C16 out;
the package contains one MLA plugin, one MLA ELF, and no BF16/FP16 compute.

## Customer contract

| Boundary | Contract |
| --- | --- |
| Input | `image`, INT8 `[1,448,576,3]` NHWC |
| Output | INT8 `[1,448,576,16]` NHWC |
| Compute | strict W8A8, per-channel symmetric INT8 Conv weights |
| Target | Modalix, one MLA plugin / one ELF |

Preprocess RGB by resizing to 576x448 without preserving aspect ratio, dividing
by 255, and applying ImageNet mean/std normalization. Convert the normalized
float tensor to public INT8 with:

```text
code = clip(rint(float64(float32(x)) * 53.59502566491281 - 14), -128, 127)
```

`rint` is nearest with ties to even. Dequantize output code with:

```text
relative_inverse_depth = (code + 128) / 7.004845439346729
```

All sixteen channels must be bit-identical; consume channel 0 after checking
that invariant. The output is **relative inverse depth**, not metric meters. A
metric result requires an external scale/shift or a metric DepthART checkpoint.

## Accuracy and performance evidence

On the private, checksum-pinned NYUv2 test654 snapshot, the Modalix result was:

- delta1 `0.7703587`, AbsRel `0.1521478`;
- delta1 retention `95.0514%` versus the matched local FP32 baseline;
- 654/654 finite outputs, zero C16 channel delta, and bit-exact repeated runs;
- mean measured-run latency `7.5889 ms` (p50 `7.5467`, p95 `7.7238`).

The registry recipe was also rebuilt independently on AEC1. Although archive
metadata changed, it reproduced the exact qualified MLA ELF
`5d887d7a...7f72d0`. On Modalix, its ten-sample outputs were byte-identical to
the qualification package, remained finite with zero C16 channel delta, and a
100-frame runtime benchmark measured `5.9959 ms` (`275.36 FPS`). The recipe
rejects a compile whose ELF differs until that program is requalified.

The candidate passes the declared 95% model-relative retention gate. It does
not pass the older absolute delta1 >= 0.90 / AbsRel <= 0.09 targets; those
failures remain explicit in `tests/correctness.yaml`. Runtime power/energy was
unavailable, so this entry makes no power claim.

## QAT lineage

QAT was applied to the original PyTorch architecture, not as ONNX-only surgery.
The trainable source:

- replaces selective scan with `QuantConvMixer`;
- makes each split branch own its SiLU and quantization grid;
- spells GELU in the target-realizable primitive form;
- fuses reparameterizable Conv/BatchNorm and folds local layer scale; and
- widens the learned terminal depth Conv to 16 identical channels so output
  publication stays within MLA.

`training/depthart_base/qat.yaml` records the model-specific architecture,
private-dataset policy, loss, staged schedule, resume requirements, and gates.
It intentionally does not vendor the shared SiMa QAT implementation. Ji Fan's
model-registry requirement is that the separately packaged QAT runner execute
this YAML on trusted, resumable GPU CI and publish its artifact to Vulcan before
compile.

## Compiler boundary

Qualification ran with custom AFE/MLC source working trees, not the stock Model
Compiler 2.1.3 package. It made no file edits inside the installed compiler
environment, but the source overlays were active. The model recipe retains two
small process-local guards:

1. use the pure-Python pickler at the deep ModelGraph serialization boundary;
2. disable an impossible INT16-exact Concat helper mode only when all involved
   tensors are INT8.

The strict INT8 SiMaIR also requires the QMLA boundary-configuration serializer
and compiler support used by qualification. Model Compiler 2.1.3 does not
contain that complete surface. This entry therefore remains `staging`; do not
promote it to customer `approved` until those compiler changes are released,
the temporary guards are either upstreamed or proven unnecessary, and CI
rebuild/board qualification passes on the pinned release compiler.

## Rebuild

With model-registry and the qualified Model Compiler installed:

```bash
model-registry validate models/depthart_base
model-registry fetch models/depthart_base
model-registry matrix --repository . --models depthart_base
```

The trusted workflow fetches the checksummed ONNX and QAT SiMaIR source, runs:

```bash
python models/depthart_base/compile.py \
  --model <verified-onnx> \
  --output-dir <new-output-directory> \
  --variant modalix_strict_int8
```

and then independently validates, publishes, and benchmarks the single MPK.
Large models, datasets, checkpoints, SiMaIR, MPKs, ELFs, and logs are never
committed to Git.

## Licensing

The pinned Hugging Face model card declares Apache-2.0 for the checkpoint. The
upstream GitHub source repository currently has no license file. Redistribution
is conservatively marked restricted; legal/source-code approval is required
before customer promotion. NYUv2 validation data remains in the private
validation-data repository and is not redistributed by this entry.
