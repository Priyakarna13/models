from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator, FormatChecker

from .errors import SourcePolicyError, ValidationError
from .model_report import load_metric_catalog
from .model_tests import TESTS_DIR, TESTS_SCHEMA, validate_model_tests
from .published_metrics import load_accuracy_metrics, validate_published_metrics
from .source import SourceSpec
from .taxonomy import find_taxonomy, load_taxonomy, validate_model_categorization

MODEL_FILE = "model.yaml"
SCHEMA_RELATIVE_PATH = Path("schemas/model.schema.json")
SCRIPTS_DIR = "scripts"
# A model folder holds its manifest, its documentation, model-owned code in scripts/, and its tests in tests/.
MODEL_FOLDER_ENTRIES = {MODEL_FILE, "README.md", "hero.png", SCRIPTS_DIR, TESTS_DIR}
STAGES = ("numerical_fidelity", "accuracy", "performance")
# vulcan-model-convert.yml runs these files for a model that declares `conversion`.
CONVERSION_FILES = (f"{SCRIPTS_DIR}/convert.py", f"{SCRIPTS_DIR}/requirements-convert.txt")
# The accuracy stage runs this script on the DevKit.
ACCURACY_SCRIPT = f"{SCRIPTS_DIR}/accuracy.py"
ACCURACY_REQUIREMENTS = f"{SCRIPTS_DIR}/requirements-accuracy.txt"
# The performance stage runs this script on the DevKit.
PERFORMANCE_SCRIPT = f"{SCRIPTS_DIR}/performance.py"


def _by_id(items: list[dict[str, Any]], kind: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        item_id = str(item["id"])
        if item_id in indexed:
            raise ValidationError(f"duplicate {kind} ids: {item_id}")
        indexed[item_id] = item
    return indexed


def _check_file_set(directory: Path, source: dict[str, Any], label: str) -> None:
    if source["type"] != "file_set":
        return
    member_paths = [member["path"] for member in source["files"]]
    if len(member_paths) != len(set(member_paths)):
        raise ValidationError(f"{directory}: {label} source contains duplicate paths")
    if source["main_file"] not in set(member_paths):
        raise ValidationError(f"{directory}: {label} source main_file is not a declared file")


def component_for(model: dict[str, Any], component_id: str) -> dict[str, Any]:
    try:
        return next(component for component in model["components"] if component["id"] == component_id)
    except StopIteration as exc:
        raise ValidationError(f"unknown component: {component_id}") from exc


def build_profile_for(model: dict[str, Any], profile_id: str) -> dict[str, Any]:
    try:
        return next(profile for profile in model["build_profiles"] if profile["id"] == profile_id)
    except StopIteration as exc:
        raise ValidationError(f"unknown build profile: {profile_id}") from exc


def profile_component_for(
    model: dict[str, Any], profile_id: str, component_id: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile = build_profile_for(model, profile_id)
    try:
        binding = next(item for item in profile["components"] if item["id"] == component_id)
    except StopIteration as exc:
        raise ValidationError(f"build profile {profile_id!r} does not include component {component_id!r}") from exc
    return profile, binding


def mpk_filename(model: dict[str, Any], profile_id: str, component_id: str) -> str:
    """Name a compiled MPK by model and variant ID, adding the component when the variant has several."""
    profile = build_profile_for(model, profile_id)
    name = f"{model['id']}_{profile_id}"
    if len(profile["components"]) > 1:
        name = f"{name}_{component_id}"
    return f"{name}_mpk.tar.gz"


def profile_component_requires_calibration(model: dict[str, Any], profile_id: str, component_id: str) -> bool:
    _, binding = profile_component_for(model, profile_id, component_id)
    return "calibration_dataset" in profile_component_for(model, profile_id, component_id)[0]


def effective_runner_class(profile: dict[str, Any], binding: dict[str, Any]) -> str:
    return str(binding.get("runner_class", profile.get("runner_class", "default")))


def compile_invocations_for(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """One invocation builds every supported component in profile order."""
    components = [
        binding["id"] for binding in profile["components"] if binding.get("status", "supported") == "supported"
    ]
    return (
        [{"id": components[0] if len(components) == 1 else "default", "components": components}] if components else []
    )


def performance_config(model: dict[str, Any], build_profile_id: str) -> dict[str, Any] | None:
    entry = model.get("performance")
    return entry if entry is not None and build_profile_id in entry["build_profiles"] else None


def source_members(source: dict[str, Any]) -> list[tuple[str, SourceSpec]]:
    if source["type"] == "file":
        spec = SourceSpec.from_mapping(source)
        return [(spec.filename, spec)]
    return [(member["path"], SourceSpec.from_mapping(member)) for member in source["files"]]


def model_file(model_dir: str | Path) -> Path:
    directory = Path(model_dir).expanduser().resolve()
    path = directory / MODEL_FILE
    if not path.is_file():
        raise ValidationError(f"{directory}: missing {MODEL_FILE}")
    return path


class _UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate mapping keys instead of silently losing declarations."""


def _unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode) -> dict[str, Any]:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if key in result:
            raise ValidationError(f"duplicate YAML key: {key!r}")
        result[key] = loader.construct_object(value_node)
    return result


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping)


def load_model(model_dir: str | Path) -> dict[str, Any]:
    path = model_file(model_dir)
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ValidationError(f"{path}: could not load YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{path}: expected a YAML object")
    _validate_declaration(value, find_schema(path.parent), path.parent)
    return normalize_model(value, path.parent)


def _validate_declaration(value: dict[str, Any], schema_path: Path, directory: Path) -> None:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(value),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(f"{_error_path(error)}: {error.message}" for error in errors)
        raise ValidationError(f"{directory}: {details}")


def normalize_model(value: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Resolve conventional recipes, datasets and the single default package."""
    model = deepcopy(value)
    compilation = model["compilation"]
    source = _expand_source(compilation["source"])
    compilation["source"] = source
    model["id"] = directory.name
    recipe = "compile.sh" if compilation["format"] == "huggingface" else "compile.py"
    model["components"] = [
        {
            "id": name,
            "format": compilation["format"],
            "source": source,
            "recipe": f"scripts/{recipe}",
            **(
                {"runtime_entrypoint": compilation["runtime_entrypoint"]} if "runtime_entrypoint" in compilation else {}
            ),
        }
        for name in compilation["components"]
    ]
    if "conversion" in model:
        model["conversion"]["source"] = _expand_source(model["conversion"]["source"])
    if "upstream" not in model:
        origin = value.get("conversion", value["compilation"])["source"]
        repository, revision = origin.get("repository"), origin.get("revision")
        uri = origin.get("uri", "")
        if not repository and uri.startswith("hf://models/"):
            repository, rest = uri.removeprefix("hf://models/").split("@", 1)
            revision = rest.split("/", 1)[0]
        if repository and revision:
            model["upstream"] = {"repository": f"https://huggingface.co/{repository}", "revision": revision}
    if "upstream" in model:
        model["upstream"].setdefault("model_card", model["upstream"]["repository"])
    datasets = model.setdefault("datasets", {})
    for dataset in datasets.values():
        if "source" in dataset:
            dataset["source"] = _expand_source(dataset["source"])
    profiles = []
    for name, declared in model["build_profiles"].items():
        profile = {"id": name, "target": "modalix", **declared}
        profile["components"] = [
            {"id": component, **settings} for component, settings in declared["components"].items()
        ]
        profiles.append(profile)
    model["build_profiles"] = profiles
    names = compilation["components"]
    supported = _supported_profiles(profiles, names)
    model["packages"] = [
        {
            "id": "default",
            "default": True,
            "components": names,
            "resources": [],
            "build_profiles": supported,
            "composition": "default",
        }
    ]
    model["compositions"] = (
        [
            {
                "id": "default",
                "type": "single",
                "component": names[0],
                "entrypoint": compilation.get("runtime_entrypoint", "default"),
            }
        ]
        if len(names) == 1
        else [{"id": "default", "type": "pipeline"}]
    )
    for stage in STAGES:
        entry = model.get(stage)
        if entry is None:
            continue
        entry.setdefault("build_profiles", supported)
        if "dataset" in entry:
            alias = entry["dataset"]
            if alias not in datasets:
                raise ValidationError(f"{directory}: {stage} references unknown dataset {alias!r}")
            entry["dataset"] = deepcopy(datasets[alias])
    return model


def _expand_source(source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(source.get("files"), dict):
        return source
    source = deepcopy(source)
    repository = source.pop("repository", None)
    revision = source.pop("revision", None)
    if repository is not None and (not isinstance(repository, str) or not re.fullmatch(r"[^/@]+/[^/@]+", repository)):
        raise ValidationError("source repository must be owner/repository")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision)):
        raise ValidationError("source revision must be an immutable 40-character commit")
    files = []
    for path, specification in source["files"].items():
        if "path" in specification:
            raise ValidationError("source files use their mapping key as the path")
        member = {"path": path, **specification}
        if "uri" not in member:
            if not repository or not revision:
                raise ValidationError(f"source member {path!r} requires a URI or repository and revision")
            member["uri"] = f"hf://models/{repository}@{revision}/{path}"
        files.append(member)
    source["files"] = files
    source.setdefault("type", "file_set")
    return source


def _supported_profiles(profiles: list[dict[str, Any]], components: list[str]) -> list[str]:
    return [
        profile["id"]
        for profile in profiles
        if all(
            any(
                binding["id"] == component and binding.get("status") != "unsupported"
                for binding in profile["components"]
            )
            for component in components
        )
    ]


def find_schema(model_dir: str | Path, schema_path: str | Path | None = None) -> Path:
    if schema_path is not None:
        candidate = Path(schema_path).expanduser().resolve()
        if not candidate.is_file():
            raise ValidationError(f"schema not found: {candidate}")
        return candidate

    configured_root = os.environ.get("MODEL_REGISTRY_ROOT", "").strip()
    candidates = []
    if configured_root:
        candidates.append(Path(configured_root).expanduser().resolve() / SCHEMA_RELATIVE_PATH)

    directory = Path(model_dir).expanduser().resolve()
    candidates.extend(parent / SCHEMA_RELATIVE_PATH for parent in (directory, *directory.parents))
    candidates.extend(parent / SCHEMA_RELATIVE_PATH for parent in (Path.cwd(), *Path.cwd().parents))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValidationError(
        f"{directory}: could not locate {SCHEMA_RELATIVE_PATH}; run inside the repository or set MODEL_REGISTRY_ROOT"
    )


def _format_path(parts: list[Any]) -> str:
    rendered = ""
    for part in parts:
        if isinstance(part, int):
            rendered += f"[{part}]"
        else:
            rendered += ("." if rendered else "") + str(part)
    return rendered or "<root>"


def _error_path(error: Any) -> str:
    parts = list(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = next((name for name in error.validator_value if name not in error.instance), None)
        if missing is not None:
            parts.append(missing)
    return _format_path(parts)


def component_pipelines(models: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, str]]]:
    """Each declared pipeline, keyed by model ID."""
    return {model_id: model["pipeline"] for model_id, model in sorted(models.items()) if "pipeline" in model}


def validate_model(
    model_dir: str | Path,
    model: dict[str, Any],
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> None:
    directory = Path(model_dir).expanduser().resolve()
    schema_file = find_schema(directory, schema_path)
    if not isinstance(model.get("build_profiles"), list):
        _validate_declaration(model, schema_file, directory)
        model = normalize_model(model, directory)
    if model["id"] != directory.name:
        raise ValidationError(f"{directory}: id must match directory name")
    if "upstream" not in model:
        raise ValidationError(f"{directory}: upstream is required when the source has no Hugging Face repository")
    taxonomy_file = find_taxonomy(directory, taxonomy_path)
    validate_model_categorization(model, load_taxonomy(taxonomy_file))
    metrics = load_metric_catalog(taxonomy_file.parent / "metrics.json")
    vocabulary = load_accuracy_metrics(taxonomy_file.parent / "accuracy-metrics.yaml")
    if model.get("reference_metrics"):
        validate_published_metrics(model["reference_metrics"], vocabulary)
    components = _by_id(model["components"], "component")
    profiles = _by_id(model["build_profiles"], "build profile")
    sources = [model["compilation"]["source"]]
    if model.get("conversion"):
        sources.append(model["conversion"]["source"])
    sources.extend(dataset["source"] for dataset in model["datasets"].values() if "source" in dataset)
    try:
        for source in sources:
            source_members(source)
            _check_file_set(directory, source, "declared")
    except (KeyError, TypeError, ValueError, SourcePolicyError) as exc:
        raise ValidationError(f"{directory}: source policy: {exc}") from exc
    if model["compilation"]["format"] == "huggingface":
        if model["compilation"]["source"]["type"] != "file_set":
            raise ValidationError(f"{directory}: native compilation requires a complete checkpoint file set")
        if len(components) != 1 or model["compilation"].get("runtime_entrypoint") not in {"generate", "transcribe"}:
            raise ValidationError(f"{directory}: native compilation requires one component and a runtime_entrypoint")
    for profile_id, profile in profiles.items():
        bindings = _by_id(profile["components"], "profile component")
        if set(bindings) != set(components):
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} must declare every compilation component exactly once"
            )
        calibration = profile.get("calibration_dataset")
        if calibration is not None and calibration not in model["datasets"]:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} references unknown calibration dataset {calibration!r}"
            )
        classes = {effective_runner_class(profile, binding) for binding in bindings.values()}
        if len(classes) != 1:
            raise ValidationError(f"{directory}: components in one compile invocation must use one runner class")
        statuses = {binding.get("status", "supported") for binding in bindings.values()}
        if len(statuses) != 1:
            raise ValidationError(
                f"{directory}: build profile {profile_id!r} must support or exclude the complete model"
            )
    pipeline_components = [stage["component"] for stage in model.get("pipeline", []) if "component" in stage]
    if "pipeline" in model and pipeline_components != list(components):
        raise ValidationError(f"{directory}: pipeline must list each component exactly once in compilation order")
    _check_folder_layout(directory)
    referenced = [model["components"][0]["recipe"]]
    if model.get("conversion"):
        referenced.extend(CONVERSION_FILES)
    if model.get("accuracy"):
        referenced.extend([ACCURACY_SCRIPT, ACCURACY_REQUIREMENTS])
    if model.get("performance"):
        referenced.append(PERFORMANCE_SCRIPT)
    for relative in referenced:
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory) or not candidate.is_file():
            raise ValidationError(f"{directory}: referenced file does not exist inside model directory: {relative}")
    supported = set(_supported_profiles(model["build_profiles"], list(components)))
    for stage in STAGES:
        entry = model.get(stage)
        if entry is None:
            continue
        if not set(entry["build_profiles"]) <= supported:
            raise ValidationError(f"{directory}: {stage} references unsupported build profiles")
        if "dataset" in entry and "source" in entry["dataset"]:
            raise ValidationError(f"{directory}: {stage} requires a registered dataset snapshot")
        if stage in {"accuracy", "performance"}:
            invalid = [
                name for name in entry["required_metrics"] if name not in metrics or metrics[name]["stage"] != stage
            ]
            if invalid:
                raise ValidationError(f"{directory}: {stage} references unregistered or wrong-stage metrics: {invalid}")
    validate_model_tests(directory, model, schema_file.parent / TESTS_SCHEMA)


def _only_python_caches(path: Path) -> bool:
    """True for a directory git left behind holding only ignored bytecode caches."""
    files = (item for item in path.rglob("*") if item.is_file()) if path.is_dir() else None
    return files is not None and all("__pycache__" in item.relative_to(path).parts for item in files)


def _check_folder_layout(directory: Path) -> None:
    unexpected = sorted(
        entry.name
        for entry in directory.iterdir()
        if entry.name not in MODEL_FOLDER_ENTRIES and not entry.name.startswith(".") and not _only_python_caches(entry)
    )
    if unexpected:
        raise ValidationError(
            f"{directory}: unexpected model folder entries: {', '.join(unexpected)}; "
            f"keep model code in {SCRIPTS_DIR}/, tests in {TESTS_DIR}/ and stage settings in {MODEL_FILE}"
        )


def load_and_validate_model(
    model_dir: str | Path,
    schema_path: str | Path | None = None,
    taxonomy_path: str | Path | None = None,
) -> dict[str, Any]:
    model = load_model(model_dir)
    validate_model(model_dir, model, schema_path, taxonomy_path)
    return model


def load_models(repository: str | Path) -> dict[str, dict[str, Any]]:
    """Load and validate every models/<id>/model.yaml, keyed by model ID."""
    root = Path(repository).expanduser().resolve()
    model_root = root / "models"
    discovered: dict[str, dict[str, Any]] = {}
    if not model_root.is_dir():
        raise ValidationError(f"models directory not found: {model_root}")

    for definition in sorted(model_root.glob("*/model.yaml")):
        model = load_and_validate_model(definition.parent, root / "schemas" / "model.schema.json")
        model_id = str(model["id"])
        if model_id in discovered:
            raise ValidationError(f"duplicate model id {model_id!r}: {definition}")
        discovered[model_id] = model
    return discovered
