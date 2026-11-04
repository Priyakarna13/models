"""Keep pinned-source test scenarios independent of catalog-model conversion opt-ins."""

from pathlib import Path

import yaml


def pin_fixture_source(directory: Path) -> None:
    path = directory / "model.yaml"
    model = yaml.safe_load(path.read_text())
    if model["compilation"]["source"] == "conversion":
        fixture = Path(__file__).parent / "fixtures/fixture_model/model.yaml"
        model["compilation"]["source"] = yaml.safe_load(fixture.read_text())["compilation"]["source"]
        path.write_text(yaml.safe_dump(model, sort_keys=False))
