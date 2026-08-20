# LightGlue SuperPoint-600 NEAT-68 regression model

This entry preserves the exact customer-supplied MPK that exposed NEAT-68. It is
imported byte-for-byte because recompiling would change the tiled-output contract
under test. The archive's own metadata records Model SDK 2.1.0 provenance.

The regression is the DetessDequant output: one CBlock INT8 tensor with logical
shape `601x601`, physical input size `365408` bytes, and tile `67x601`, decoded to
`[1,601,601]` FP32. Both AUTO/A65 and explicit EV74 must produce the same score
matrix for identical deterministic inputs.

This staging-only entry is a runtime regression fixture, not a supported catalog
replacement for a reproducibly compiled LightGlue source model.
