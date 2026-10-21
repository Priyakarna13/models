#!/usr/bin/env python3
"""Compile a qualified YOLO26-P2 QAT export into a Modalix batch-1 MPK."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from dataset import FrameRecord, dataset_content_sha256, load_manifest, require_unchanged_dataset
from model_scripts import (
    INPUT_NAME,
    DeploymentContract,
    audit_mpk,
    audit_onnx,
    audit_quantized_model,
    preprocess_rgb_image,
    sha256_file,
)
from PIL import Image

LOG = logging.getLogger("yolo26_tiny_drone.compile")
ARCHITECTURE = "yolo26n-p2-attention-one-class"
MODEL_NAME = "yolo26n_p2_tiny_drone_int8_qat_b1"
MINIMUM_CALIBRATION_IMAGES = 128



def serialized_model_sha256(output_dir: Path, model_name: str) -> dict[str, str]:
    """Bind every ModelSDK payload saved by the qualified quantization step."""
    names = (
        f"{model_name}.sima",
        f"{model_name}.sima.json",
        f"{model_name}.fp32.sima",
        f"{model_name}.fp32.sima.json",
    )
    digests: dict[str, str] = {}
    for name in names:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"serialized ModelSDK payload does not exist: {path}")
        digests[name] = sha256_file(path)
    return digests


def snapshot_export(export_dir: Path, destination: Path) -> tuple[Path, Path]:
    """Copy one self-consistent export into a private compilation snapshot."""
    export_root = export_dir.resolve()
    report_source = (export_root / "export-report.json").resolve()
    if report_source.parent != export_root or not report_source.is_file():
        raise ValueError(f"missing or invalid export report: {report_source}")
    report_snapshot = destination / "export-report.json"
    shutil.copyfile(report_source, report_snapshot)
    try:
        report = json.loads(report_snapshot.read_text(encoding="utf-8"))
        declaration = report["onnx"]
        filename = declaration["file"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("export report has no readable ONNX declaration") from exc
    if not isinstance(filename, str):
        raise ValueError("export report ONNX filename must be a string")
    source = (export_root / filename).resolve()
    if source.parent != export_root or not source.is_file():
        raise ValueError("export report references an invalid ONNX source")
    source_snapshot = (destination / filename).resolve()
    if source_snapshot.parent != destination.resolve():
        raise ValueError("export report ONNX filename must not escape its directory")
    shutil.copyfile(source, source_snapshot)
    return report_snapshot, source_snapshot


def validate_export(
    export_dir: Path, manifest: Path, dataset_sha256: str, *, allow_smoke: bool
) -> tuple[Path, dict[str, Any], DeploymentContract]:
    report_path = export_dir / "export-report.json"
    if not report_path.is_file():
        raise ValueError(f"missing export report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("architecture") != ARCHITECTURE:
        raise ValueError("export report is not the YOLO26-P2 tiny-drone QAT contract")
    try:
        contract = DeploymentContract.from_dict(report.get("deployment"))
    except (TypeError, ValueError) as exc:
        raise ValueError("export report has an invalid deployment contract") from exc
    if report.get("manifest_sha256") != sha256_file(manifest):
        raise ValueError("compile manifest does not match the QAT/export lineage")
    if report.get("dataset_sha256") != dataset_sha256:
        raise ValueError("compile dataset contents do not match the QAT/export lineage")
    if type(report.get("qualification_eligible")) is not bool:
        raise ValueError("export report has no explicit qualification eligibility")
    if not report["qualification_eligible"] and not allow_smoke:
        raise ValueError("ineligible diagnostic export requires --allow-smoke")
    if int(report.get("observer_images", 0)) < MINIMUM_CALIBRATION_IMAGES and not allow_smoke:
        raise ValueError("export has fewer than 128 QAT observer images")

    declaration = report.get("onnx")
    if not isinstance(declaration, dict) or not isinstance(declaration.get("file"), str):
        raise ValueError("export report has no ONNX declaration")
    source = (export_dir / declaration["file"]).resolve()
    if source.parent != export_dir.resolve() or not source.is_file():
        raise ValueError("export report references an invalid ONNX source")
    audit = audit_onnx(source, contract)
    declared_audit = declaration.get("audit")
    if not isinstance(declared_audit, dict) or audit.get("sha256") != declared_audit.get("sha256"):
        raise ValueError("ONNX source digest does not match the export report")

    accuracy = report.get("accuracy")
    if not isinstance(accuracy, dict):
        raise ValueError("export report has no accuracy evidence")
    if not allow_smoke:
        required = {
            "ap50_drop": 0.02,
            "qat_ap50": 0.30,
            "qat_tiny_recall50": 0.50,
            "tiny_ground_truth_count": 100.0,
            "tiny_recall50_drop": 0.05,
        }
        try:
            values = {key: float(accuracy[key]) for key in required}
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("export report contains incomplete accuracy evidence") from exc
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("export report contains incomplete accuracy evidence")
        if values["ap50_drop"] > required["ap50_drop"]:
            raise ValueError("export exceeds the 2-point AP50-drop gate")
        if values["tiny_recall50_drop"] > required["tiny_recall50_drop"]:
            raise ValueError("export exceeds the 5-point tiny-recall-drop gate")
        if values["qat_ap50"] < required["qat_ap50"]:
            raise ValueError("export is below the 0.30 QAT AP50 gate")
        if values["qat_tiny_recall50"] < required["qat_tiny_recall50"]:
            raise ValueError("export is below the 0.50 QAT tiny-recall gate")
        if values["tiny_ground_truth_count"] < required["tiny_ground_truth_count"]:
            raise ValueError("export has fewer than 100 held-out tiny objects")
    return source, report, contract


def select_calibration_records(
    records: list[FrameRecord], count: int, *, seed: int, allow_smoke: bool
) -> list[FrameRecord]:
    minimum = 1 if allow_smoke else MINIMUM_CALIBRATION_IMAGES
    if count < minimum:
        raise ValueError(f"calibration image count must be at least {minimum}")
    train = [record for record in records if record.split == "train"]
    unique = {record.image: record for record in train}
    if len(unique) < count:
        raise ValueError(f"requested {count} calibration images, but only {len(unique)} distinct train images exist")
    # A seeded sample avoids taking a contiguous run from one video while
    # preserving the source dataset's natural sequence distribution.
    population = sorted(unique.values(), key=lambda record: str(record.image))
    selected = random.Random(seed).sample(population, count)
    return sorted(selected, key=lambda record: str(record.image))




def preprocess_image(path: Path, contract: DeploymentContract) -> np.ndarray:
    with Image.open(path) as source:
        return preprocess_rgb_image(source.convert("RGB"), contract)


def build_calibration(records: list[FrameRecord], contract: DeploymentContract) -> tuple[np.ndarray, dict[str, Any]]:
    samples = np.stack([preprocess_image(record.image, contract) for record in records])
    if (
        samples.shape
        != (
            len(records),
            contract.model_size,
            contract.model_size,
            3,
        )
        or samples.dtype != np.float32
    ):
        raise RuntimeError(f"invalid calibration tensor: shape={samples.shape}, dtype={samples.dtype}")
    digest = hashlib.sha256(memoryview(np.ascontiguousarray(samples)).cast("B")).hexdigest()
    evidence = {
        "count": len(records),
        "images": [
            {
                "path": str(record.image),
                "sequence": record.sequence,
                "sha256": sha256_file(record.image),
            }
            for record in records
        ],
        "preprocessing": (f"rgb_letterbox_{contract.model_size}_fill_114_divide_255_nhwc_float32"),
        "tensor_sha256": digest,
    }
    return samples, evidence










def compile_model(
    source: Path,
    calibration: np.ndarray,
    output_dir: Path,
    model_name: str,
    contract: DeploymentContract,
    expected_convolutions: int,
) -> tuple[Path, dict[str, Any]]:
    # Keep SDK imports out of source-validation and unit-test paths.
    from afe.apis.defines import (
        CalibrationMethod,
        InputName,
        RequantizationMode,
        default_quantization,
        gen2_target,
        quantization_scheme,
    )
    from afe.apis.loaded_net import load_model
    from afe.core.utils import convert_data_generator_to_iterable
    from afe.ir.tensor_type import ScalarType
    from afe.load.importers.general_importer import ImporterParams, ModelFormat
    from sima_utils.data.data_generator import DataGenerator

    importer = ImporterParams(
        format=ModelFormat.onnx,
        file_paths=[str(source)],
        input_names=[INPUT_NAME],
        input_shapes=[contract.input_shape],
        input_types=[ScalarType.float32],
        layout="NCHW",
        output_names=contract.output_names(),
    )
    loaded = load_model(importer, target=gen2_target)
    calibration_data = convert_data_generator_to_iterable(DataGenerator({InputName(INPUT_NAME): calibration}))
    config = (
        default_quantization.with_activation_quantization(
            quantization_scheme(asymmetric=True, per_channel=False, bits=8)
        )
        .with_weight_quantization(quantization_scheme(asymmetric=False, per_channel=True, bits=8))
        .with_requantization_mode(RequantizationMode.sima)
        .with_calibration(CalibrationMethod.from_str("min_max"))
    )
    quantized = loaded.quantize(
        calibration_data=calibration_data,
        quantization_config=config,
        model_name=model_name,
        log_level=logging.INFO,
    )
    if quantized is None:
        raise RuntimeError("ModelSDK quantization returned no model")
    quantized.save(model_name=model_name, output_directory=str(output_dir))
    quantized_report = audit_quantized_model(
        output_dir / f"{model_name}.sima.json",
        expected_convolutions,
    )
    quantized_report["serialized_model_sha256"] = serialized_model_sha256(output_dir, model_name)
    quantized.compile(
        output_path=str(output_dir),
        batch_size=1,
        log_level=logging.INFO,
    )
    archives = sorted(output_dir.glob("*_mpk.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"expected one compiled MPK, found {archives}")
    return archives[0], quantized_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-images", type=int, default=MINIMUM_CALIBRATION_IMAGES)
    parser.add_argument("--calibration-seed", type=int, default=26)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    export_dir = args.export_dir.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        parser.error(f"output directory already exists: {output_dir}")
    if not manifest.is_file():
        parser.error(f"manifest does not exist: {manifest}")
    if re.fullmatch(r"[a-z0-9_]{1,80}", args.model_name) is None:
        parser.error("model-name must contain only lowercase letters, digits, and underscores")
    records = load_manifest(manifest)
    dataset_digest = dataset_content_sha256(manifest, records)
    with tempfile.TemporaryDirectory(prefix="yolo26-export-snapshot-") as temporary:
        snapshot_dir = Path(temporary)
        report_snapshot, source_snapshot = snapshot_export(export_dir, snapshot_dir)
        source, export_report, contract = validate_export(
            snapshot_dir,
            manifest,
            dataset_digest,
            allow_smoke=args.allow_smoke,
        )
        report_digest = sha256_file(report_snapshot)
        source_digest = sha256_file(source_snapshot)
        report_snapshot.chmod(0o400)
        source_snapshot.chmod(0o400)
        selected = select_calibration_records(
            records,
            args.calibration_images,
            seed=args.calibration_seed,
            allow_smoke=args.allow_smoke,
        )
        calibration, calibration_report = build_calibration(selected, contract)
        require_unchanged_dataset(
            manifest,
            records,
            manifest_sha256=export_report["manifest_sha256"],
            dataset_sha256=dataset_digest,
            context="calibration",
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        archive, quantized_report = compile_model(
            source,
            calibration,
            output_dir,
            args.model_name,
            contract,
            int(export_report["onnx"]["audit"]["convolution_count"]),
        )
        validate_export(
            snapshot_dir,
            manifest,
            dataset_digest,
            allow_smoke=args.allow_smoke,
        )
        if sha256_file(report_snapshot) != report_digest or sha256_file(source_snapshot) != source_digest:
            raise RuntimeError("private export snapshot changed during compilation")
        report = {
            "schema_version": 1,
            "architecture": ARCHITECTURE,
            "deployment": contract.to_dict(),
            "source_qualification_eligible": export_report["qualification_eligible"],
            "qualification_eligible": export_report["qualification_eligible"] and not args.allow_smoke,
            "source": {
                "export_report_sha256": report_digest,
                "onnx_sha256": source_digest,
            },
            "calibration": calibration_report,
            "quantized_model": quantized_report,
            "mpk": audit_mpk(archive, contract),
        }
    (output_dir / "compile-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
