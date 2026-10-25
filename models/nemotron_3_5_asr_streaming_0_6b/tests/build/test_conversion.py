import json


def test_converted_graph_has_static_regions(converted):
    import onnx

    model = onnx.load(converted, load_external_data=False)
    onnx.checker.check_model(str(converted))
    config = json.loads(next(item.value for item in model.metadata_props if item.key == "sima.asr"))
    assert config["encoder_outputs"][0] == "encoded"
    assert len(config["predictor_outputs"]) == 3
    assert len(config["joiner_outputs"]) in (1, 2)
    projection = [node for node in model.graph.node if node.name.startswith("projection/")]
    assert sum(node.op_type in {"MatMul", "Gemm"} for node in projection) == 2
    assert sum(node.op_type == "Relu" for node in projection) == 1
    for value in [*model.graph.input, *model.graph.output]:
        assert all(dim.dim_value > 0 for dim in value.type.tensor_type.shape.dim)


def test_export_matches_original_checkpoint(converted, load_script):
    import os
    import tempfile
    from pathlib import Path

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    from model_registry.onnx_export import partition
    from transformers import AutoModelForRNNT, AutoModelForTDT

    converted = Path(converted)
    graph = onnx.load(converted, load_external_data=False)
    config = json.loads(next(item.value for item in graph.metadata_props if item.key == "sima.asr"))
    encoder = partition(graph, config["encoder_outputs"])
    del graph
    inputs = {
        value.name: np.zeros([dim.dim_value for dim in value.type.tensor_type.shape.dim], np.float32)
        for value in encoder.graph.input
    }
    rng = np.random.default_rng(440)
    torch.set_num_threads(2)
    if config["kind"] == "nemotron":
        features = rng.normal(size=(1, 49, 128)).astype(np.float32)
        inputs["features"][:, 7:] = features
        inputs["subsampling_mask"].fill(1)
        inputs["subsampling_mask"][:, :, :3] = 0
        inputs["subsampling_mask"][:, :, 32] = 0
        inputs["attention_bias"][..., :56] = -10000
        inputs["language_one_hot"][..., 101] = 1
        model = AutoModelForRNNT.from_pretrained(
            converted.parent, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        with torch.inference_mode():
            hidden = torch.from_numpy(rng.normal(size=(1, 7, 1024)).astype(np.float32))
            prompt = torch.from_numpy(inputs["language_one_hot"])
            expected_projection = model.encoder_projector(
                model.prompt_projector(torch.cat([hidden, prompt.expand(1, 7, 128)], dim=-1))
            )
            projection = load_script("convert").ProjectionStep(model, frames=7)
            torch.testing.assert_close(projection(hidden, prompt), expected_projection, atol=1e-3, rtol=1e-4)
            reference = model.get_audio_features(
                torch.from_numpy(features),
                prompt_ids=torch.tensor([101]),
                use_cache=True,
                num_lookahead_tokens=6,
                output_attention_mask=False,
            ).pooler_output.numpy()
        tolerance = 0.03
    else:
        features = rng.normal(size=(1, 1400, 128)).astype(np.float32)
        inputs["features"][:, 0] = features
        inputs["padding_masks"].fill(1)
        inputs["attention_bias"][..., -1] = -10000
        model = AutoModelForTDT.from_pretrained(
            converted.parent, dtype=torch.float32, attn_implementation="sdpa"
        ).eval()
        with torch.inference_mode():
            reference = model.get_audio_features(
                torch.from_numpy(features), attention_mask=torch.ones(1, 1400, dtype=torch.long)
            ).pooler_output.numpy()
        tolerance = 0.003
    del model
    with tempfile.TemporaryDirectory(dir=converted.parent) as work:
        work = Path(work)
        os.link(converted.parent / "model.onnx.ext", work / "model.onnx.ext")
        path = work / "encoder.onnx"
        onnx.save_model(encoder, path)
        del encoder
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        actual = session.run(["encoded"], inputs)[0]
    np.testing.assert_allclose(actual, reference, atol=tolerance, rtol=2e-4)
