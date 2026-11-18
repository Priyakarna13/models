from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def test_local_markdown_links_resolve() -> None:
    markdown_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    failures = []
    for document in markdown_files:
        for target in MARKDOWN_LINK.findall(document.read_text(encoding="utf-8")):
            path = target.split("#", 1)[0]
            if not path or "://" in path or path.startswith("mailto:"):
                continue
            resolved = (document.parent / path).resolve()
            if not resolved.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}")
    assert not failures, "broken local Markdown links:\n" + "\n".join(failures)


def test_developer_index_covers_the_contribution_path() -> None:
    text = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")

    for guide in ("adding-a-model.md", "metadata-schema.md", "local-development.md", "contributing.md"):
        assert f"]({guide})" in text


def test_developer_sidebar_includes_every_article() -> None:
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")

    for article in sorted((ROOT / "docs").glob("*.md")):
        doc_id = article.stem
        assert f"'{doc_id}'" in sidebar, f"docs/{article.name} is missing from the developer sidebar"


def test_contributor_docs_distinguish_pr_checks_from_trusted_automation() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "vulcan-model-orchestrator.yml").read_text(encoding="utf-8")
    contributing_words = " ".join(contributing.split())

    assert "Pull requests do **not** run recipe or surgery code" in overview
    assert "## After merge: trusted staging automation" in contributing
    assert "A successful compile does not by itself mean the benchmark succeeded" in contributing_words
    assert "pull_request" not in workflow


def test_local_qualification_is_required_before_submission() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    overview_words = " ".join(overview.split())

    assert "submission prerequisites, not optional CI substitutes" in overview_words
    assert "complete the entire loop before submitting a model pull request" in local
    assert "Do not start submission until every supported component/profile pair" in contributing
    assert "checks not run locally" not in contributing


def test_contribution_guides_have_single_workflow_owners() -> None:
    overview = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    authoring = (ROOT / "docs" / "adding-a-model.md").read_text(encoding="utf-8")
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    overview_words = " ".join(overview.split())

    assert "owns the tool installation and complete test sequence" in overview_words
    assert "sima-cli neat install --stg models-utils/model-registry" not in overview
    assert "model-registry matrix --repository" not in authoring
    assert "model-registry verify-mpk" not in authoring
    assert "does not repeat the local qualification procedure" in contributing
    assert "model-registry validate models/<model-id>" not in contributing
    assert "## Local-first qualification loop" in local


def test_core_page_titles_match_the_journey_navigation() -> None:
    sidebar = (ROOT / "website" / "sidebars.ts").read_text(encoding="utf-8")
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    expected = {
        "adding-a-model.md": ("1", "Define the model contract"),
        "local-development.md": ("2", "Qualify locally"),
        "contributing.md": ("3", "Submit for review"),
    }

    for filename, (step, title) in expected.items():
        article = (ROOT / "docs" / filename).read_text(encoding="utf-8")
        assert article.startswith(f"# {title}\n")
        assert f"label: '{step}. {title}'" in sidebar
        assert f"guideLabel: '{title}'" in component


def test_dataset_guide_covers_artifactory_access_and_registration() -> None:
    text = (ROOT / "docs" / "datasets.md").read_text(encoding="utf-8")

    for expected in (
        "## Set up Artifactory access first",
        "dataset-registry resolve imagenet-2012",
        'jq -e \'.provider == "artifactory"\'',
        "## Introduce a dataset or variant that does not exist",
        "catalog/datasets.yaml",
        "artifactory_sources",
        "Dataset pull-request CI is intentionally credential-free",
        "## Troubleshooting access",
    ):
        assert expected in text

    assert "Do not add a one-off download path to the model recipe" in text


def test_contribution_path_wraps_without_horizontal_scrolling() -> None:
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    css = (ROOT / "website" / "src" / "css" / "custom.css").read_text(encoding="utf-8")

    assert "<ol" in component
    assert "<svg" not in component
    assert "'six five four'" in css
    assert "@container (max-width: 700px)" in css
    assert "@container (max-width: 480px)" in css


def test_local_guide_passes_scoped_calibration_to_the_recipe() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    assert "fetch-calibration \\" in text
    assert "models/<model-id> --build-profile <profile-id> --component <component-id> --json" in text
    assert 'recipe_args=(--dataset "$calibration_path")' in text
    assert '"${recipe_args[@]}"' in text
    assert "## Measure declared accuracy locally" in text


def test_local_guide_uses_supported_validation_commands() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    assert "sima-cli neat install --stg models-utils/model-registry@develop:latest" in text
    assert "python -m pip install -e" not in text
    for command in (
        "model-registry validate ",
        "model-registry validate-models ",
        "model-registry fetch ",
        "model-registry fetch-calibration ",
        "model-registry matrix ",
        "model-registry resolve-compiler ",
        "model-registry verify-mpk ",
    ):
        assert command in text


def test_local_guide_covers_the_complete_local_qualification_loop() -> None:
    text = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")

    for expected in (
        "Compile locally from `model.yaml`",
        "Model Compiler Python APIs available",
        "compile.py --model <verified-entrypoint>",
        "model-registry verify-mpk",
        "Benchmark locally on a Modalix DevKit",
        "python models/resnet_50/scripts/performance.py",
        "model-report.json",
        "Finish the local review",
    ):
        assert expected in text

    assert "Compilation is intentionally model-owned" in text
    assert "does not provide a public compile command" in text.replace("\n", " ")
    assert "Keep the report and log outside Git" in text
    assert "Vulcan" not in text
    assert "What CI repeats" not in text
    assert "GitHub-hosted runner" not in text


def test_docs_keep_author_tools_separate_from_ci_publication() -> None:
    local = (ROOT / "docs" / "local-development.md").read_text(encoding="utf-8")
    contributing = (ROOT / "docs" / "contributing.md").read_text(encoding="utf-8")
    benchmarking = (ROOT / "docs" / "benchmarking.md").read_text(encoding="utf-8")

    assert "models-utils/model-registry@develop:latest" in local
    assert "Compilation is intentionally model-owned" in local
    assert "Model authors do not publish benchmark results directly" in benchmarking
    assert "reserved for CI" in contributing
    assert "git push --set-upstream origin HEAD" in contributing
    assert "authors do not invoke those publication steps themselves" in contributing

    author_guides = local + benchmarking
    for command in (
        "model-registry publish ",
        "model-registry publish-benchmark ",
        "model-registry publish-catalog ",
        "model-registry package ",
    ):
        assert command not in author_guides


# The structure of the documentation: docs/ is the public site, knowledge/ explains the tooling and CI, and
# AGENTS.md indexes both. knowledge/writing-docs.md states these rules for writers.

MODELS_REPOSITORY_LINK = re.compile(r"https://github\.com/sima-neat/models/(?:blob|tree|edit)/[^/]+/([^)\s]+)")
INTERNAL_TREES = ("AGENTS.md", "knowledge/", "design-doc/")
# AGENTS.md is loaded into every agent session, so its length is a cost on every task.
AGENTS_LINE_BUDGET = 260


def _documents() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "AGENTS.md",
        ROOT / "website" / "README.md",
        *sorted((ROOT / "docs").glob("*.md")),
        *sorted((ROOT / "knowledge").glob("*.md")),
        *sorted((ROOT / "artifact-browser").glob("*.md")),
    ]


def _prose(text: str) -> str:
    """The text outside fenced code blocks, where a link or heading is part of the page."""
    kept, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            kept.append(line)
    return "\n".join(kept)


def _anchors(document: Path) -> set[str]:
    """The heading anchors GitHub and the site generate for a Markdown file."""
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    for line in _prose(document.read_text(encoding="utf-8")).splitlines():
        heading = re.match(r"#{1,6}\s+(.*)", line)
        if not heading:
            continue
        title = re.sub(r"\[([^]]+)\]\([^)]+\)", r"\1", heading.group(1)).strip()
        slug = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
        anchors.add(f"{slug}-{seen[slug]}" if slug in seen else slug)
        seen[slug] = seen.get(slug, 0) + 1
    return anchors


def _link_targets(document: Path) -> list[tuple[str, Path, str]]:
    """Each link into this repository as (written target, file, anchor)."""
    targets = []
    for written in MARKDOWN_LINK.findall(_prose(document.read_text(encoding="utf-8"))):
        target = written.split()[0]
        repository_link = MODELS_REPOSITORY_LINK.fullmatch(target)
        if repository_link:
            path, _, anchor = repository_link.group(1).partition("#")
            targets.append((target, ROOT / path, anchor))
        elif "://" not in target and not target.startswith("mailto:"):
            path, _, anchor = target.partition("#")
            targets.append((target, (document.parent / path).resolve() if path else document, anchor))
    return targets


def test_links_and_anchors_resolve_in_every_document() -> None:
    failures = []
    for document in _documents():
        for target, path, anchor in _link_targets(document):
            if not path.exists():
                failures.append(f"{document.relative_to(ROOT)} -> {target}: no such file")
            elif anchor and path.suffix == ".md" and anchor not in _anchors(path):
                failures.append(f"{document.relative_to(ROOT)} -> {target}: no such heading")
    assert not failures, "broken links:\n" + "\n".join(failures)


def test_contribution_path_routes_resolve() -> None:
    component = (ROOT / "website" / "src" / "components" / "ContributionPath.tsx").read_text(encoding="utf-8")
    routes = re.findall(r"guide: '([^']+)'", component)

    assert routes
    for route in routes:
        page, _, anchor = route.partition("#")
        document = ROOT / "docs" / f"{page}.md"
        assert document.is_file(), f"ContributionPath links to a missing page: {route}"
        assert not anchor or anchor in _anchors(document), f"ContributionPath links to a missing heading: {route}"


def test_site_pages_do_not_link_to_agent_or_maintainer_files() -> None:
    failures = []
    for page in sorted((ROOT / "docs").glob("*.md")):
        for target, path, _ in _link_targets(page):
            relative = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else target
            if relative.startswith(INTERNAL_TREES):
                failures.append(f"{page.relative_to(ROOT)} -> {target}")
    assert not failures, "docs/ is the public site and stands on its own:\n" + "\n".join(failures)


def test_agents_index_lists_every_docs_and_knowledge_file() -> None:
    agents = ROOT / "AGENTS.md"
    indexed = {path for _, path, _ in _link_targets(agents)}
    pages = [page for page in sorted((ROOT / "docs").glob("*.md")) if page.name != "README.md"]
    missing = [
        page.relative_to(ROOT).as_posix()
        for page in [*pages, *sorted((ROOT / "knowledge").glob("*.md"))]
        if page not in indexed
    ]

    assert not missing, "add an index row to AGENTS.md for: " + ", ".join(missing)
    assert len(agents.read_text(encoding="utf-8").splitlines()) <= AGENTS_LINE_BUDGET


def test_site_pages_open_with_a_title_and_an_introduction() -> None:
    for page in sorted((ROOT / "docs").glob("*.md")):
        lines = [line for line in _prose(page.read_text(encoding="utf-8")).splitlines() if line.strip()]
        lines = [line for line in lines if not line.startswith("import ")]
        assert lines[0].startswith("# "), f"{page.name} must start with its title"
        assert not lines[1].startswith("#"), f"{page.name} needs an introduction before its first section"


def test_site_code_blocks_name_their_language() -> None:
    failures = []
    for page in sorted((ROOT / "docs").glob("*.md")):
        fenced = False
        for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.lstrip().startswith("```"):
                continue
            if not fenced and line.strip() == "```":
                failures.append(f"{page.name}:{number}")
            fenced = not fenced
    assert not failures, "name the language of each code block: " + ", ".join(failures)


def test_docs_folder_holds_only_pages_and_assets() -> None:
    unexpected = [
        entry.name
        for entry in sorted((ROOT / "docs").iterdir())
        if not (entry.suffix == ".md" and entry.is_file()) and entry.name != "assets"
    ]

    assert not unexpected, "docs/ is the published site; move these elsewhere: " + ", ".join(unexpected)
