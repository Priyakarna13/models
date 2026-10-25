"""Public input filtering and metadata around ONNX's graph extractor."""

import pytest
from model_registry.onnx_export import partition


def test_partition_retains_only_used_inputs_and_preserves_metadata():
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper

    def value(name):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [helper.make_node("Identity", ["audio"], ["encoded"]), helper.make_node("Identity", ["token"], ["prediction"])],
        "regions",
        [value("audio"), value("token")],
        [value("encoded"), value("prediction")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    helper.set_model_props(model, {"sima.asr": "metadata"})
    encoder = partition(model, ["encoded"])
    assert [value.name for value in encoder.graph.input] == ["audio"]
    assert [value.name for value in encoder.graph.output] == ["encoded"]
    assert encoder.metadata_props == model.metadata_props
    onnx.checker.check_model(encoder)
    with pytest.raises(ValueError, match="declared"):
        partition(model, ["missing"])


@pytest.mark.parametrize("external,corrupt", [(False, False), (True, False), (True, True)])
def test_conversion_publisher_preserves_single_file_and_external_weights(tmp_path, external, corrupt):
    import hashlib
    import os
    import subprocess
    import sys
    from pathlib import Path

    import yaml

    commands = tmp_path / "bin"
    commands.mkdir()
    transfer = commands / "transfer"
    transfer.write_text(
        f"#!{sys.executable}\n"
        + """
import os, pathlib, shutil, sys
args = sys.argv[1:]
if pathlib.Path(sys.argv[0]).name == 'aws':
    source = pathlib.Path(args[args.index('--body') + 1])
    shutil.copyfile(source, pathlib.Path(os.environ['STORE']) / source.name)
else:
    name = next(value for value in args if value.startswith('https://')).rsplit('/', 1)[1]
    target = pathlib.Path(args[args.index('-o') + 1])
    shutil.copyfile(pathlib.Path(os.environ['STORE']) / name, target)
    if os.environ['CORRUPT'] == '1' and name.endswith('.ext'):
        target.write_bytes(b'corrupt')
"""
    )
    transfer.chmod(0o755)
    for name in ("aws", "curl"):
        (commands / name).symlink_to(transfer)
    store = tmp_path / "store"
    store.mkdir()
    export = tmp_path / "export"
    export.mkdir()
    payloads = {"model.onnx": b"graph"}
    if external:
        payloads["model.onnx.ext"] = b"weights"
    for name, payload in payloads.items():
        (export / name).write_bytes(payload)
    environment = dict(
        os.environ,
        PATH=str(commands) + os.pathsep + os.environ["PATH"],
        STORE=str(store),
        CORRUPT=str(int(corrupt)),
        GITHUB_RUN_ID="123",
        GITHUB_RUN_ATTEMPT="1",
        GITHUB_SHA="a" * 40,
    )
    script = Path(__file__).resolve().parents[1] / "scripts/publish_model_conversion.sh"
    result = subprocess.run(
        [str(script), "fixture_model", "default", str(export)], env=environment, capture_output=True
    )
    if corrupt:
        assert result.returncode != 0
        return
    assert result.returncode == 0, result.stderr.decode()
    source = yaml.safe_load((export / "source.yaml").read_text())["source"]
    files = source["files"] if external else {"model.onnx": source}
    assert set(files) == set(payloads)
    for name, payload in payloads.items():
        assert files[name]["sha256"] == hashlib.sha256(payload).hexdigest()
        assert files[name]["size_bytes"] == len(payload)
        assert files[name]["uri"].endswith("/123-1/" + name)
