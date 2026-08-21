# OSTrack ViT-B no-CE QAT

This directory stages the accepted Modalix `r57` qualification for registry
onboarding. The deployment is **mixed precision**, not full INT8: representable
backbone QAT paths remain quantized, while semantic attention and the 15
QAT-dequantized score/size/offset head affines execute in BF16.

The content-bound artifact and terminal silicon receipt are listed in
`artifacts/SHA256SUMS`; qualification metrics and the single-MLA-ELF contract are
recorded in `qualification.json`.

A `model.yaml` is intentionally not present yet. The accepted immutable source
has not been uploaded to a registry-accessible location, so there is no genuine
remote asset URI to declare. Do not substitute a placeholder URL. Complete the
publication blockers in `qualification.json`, then add and validate the normal
registry entry and build recipe.
