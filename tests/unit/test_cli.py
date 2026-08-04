from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from model_registry.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_validate_json_output(capsys) -> None:
    result = main(["validate", str(ROOT / "models" / "resnet_50"), "--json"])
    assert result == 0
    output = capsys.readouterr().out
    assert '"model_id": "resnet_50"' in output
    assert '"valid": true' in output


def test_invalid_model_has_nonzero_exit(capsys) -> None:
    result = main(["validate", str(ROOT / "tests" / "fixtures" / "invalid_model")])
    assert result == 2
    assert "source.sha256" in capsys.readouterr().err


def test_matrix_emits_workflow_json(capsys) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--changed-path", "models/resnet_50/README.md"])
    assert result == 0
    output = capsys.readouterr().out
    assert '"model_id":"resnet_50"' in output
    assert '"recipe":"compile.py"' in output


def test_matrix_rejects_unknown_manual_model(capsys) -> None:
    result = main(["matrix", "--repository", str(ROOT), "--models", "does_not_exist"])
    assert result == 2
    assert "unknown model id" in capsys.readouterr().err


def test_matrix_filters_manual_variants(capsys) -> None:
    result = main(
        [
            "matrix",
            "--repository",
            str(ROOT),
            "--models",
            "superpoint",
            "--variants",
            "modalix_int8_packed,modalix_bf16_packed",
        ]
    )
    assert result == 0
    output = capsys.readouterr().out
    assert '"variant_id":"modalix_int8_packed"' in output
    assert '"variant_id":"modalix_bf16_packed"' in output
    assert '"variant_id":"modalix_fp32_dense"' not in output


def test_package_supports_python_module_entrypoint() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "model_registry", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Model Registry lifecycle tooling" in result.stdout
