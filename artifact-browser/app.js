const config = window.MODEL_REGISTRY_CONFIG || {};
const configuredApiBase = String(config.apiBase || "").replace(/\/$/, "");
const configuredRepository = config.repository || {};
const DEFAULT_BRANCH = "main";

const state = {
  apiBase: configuredApiBase,
  repository: configuredRepository,
  runs: [],
  cursor: null,
  hasMore: false,
  availableBranches: [],
  selectedId: new URLSearchParams(location.search).get("run"),
  preferredBranch: null,
  detailCache: new Map(),
};

const elements = Object.fromEntries(
  [
    "theme-toggle", "environment-name", "registry-status", "registry-copy", "search", "branch-filter",
    "status-filter", "compiler-filter", "clear-filters", "result-title", "result-count",
    "result-list", "load-more", "detail-pane", "toast-region",
  ].map((id) => [id, document.getElementById(id)]),
);

const icons = {
  sun: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="3.5"/><path d="M12 2v2m0 16v2M4.93 4.93l1.41 1.41m11.32 11.32 1.41 1.41M2 12h2m16 0h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20.3 15.2A8.5 8.5 0 0 1 8.8 3.7 8.5 8.5 0 1 0 20.3 15.2Z"/></svg>`,
  arrow: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12h14m-5-5 5 5-5 5"/></svg>`,
  download: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3v12m-4-4 4 4 4-4M5 20h14"/></svg>`,
  copy: `<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="8" y="8" width="11" height="11" rx="1.5"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/></svg>`,
  external: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M14 5h5v5m0-5-8 8M19 14v5H5V5h5"/></svg>`,
};

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" })[char]);
}

function branchName(ref = "") {
  return ref.replace(/^refs\/heads\//, "") || "detached";
}

function shortSha(sha = "") {
  return sha.slice(0, 12);
}

function formatDate(value, includeTime = true) {
  if (!value) return "—";
  const date = new Date(value);
  return new Intl.DateTimeFormat(undefined, {
    month: "short", day: "numeric", year: date.getFullYear() !== new Date().getFullYear() ? "numeric" : undefined,
    ...(includeTime ? { hour: "numeric", minute: "2-digit" } : {}),
  }).format(date);
}

function formatDuration(start, end) {
  if (!start || !end) return "—";
  const seconds = Math.max(0, Math.round((new Date(end) - new Date(start)) / 1000));
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1000 && unit < units.length - 1) { value /= 1000; unit += 1; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

async function api(path) {
  const response = await fetch(`${state.apiBase}${path}`, {
    headers: { Accept: "application/json" },
    mode: "cors",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error?.message || payload.message || `Registry request failed (${response.status})`);
  return payload;
}

function catalogCacheKey() {
  const repo = state.repository;
  return `sima-neat-model-catalog:${state.apiBase}:${repo.organization}/${repo.name}`;
}

function branchPreferenceKey() {
  const repo = state.repository;
  return `sima-neat-model-branch:${state.apiBase}:${repo.organization}/${repo.name}`;
}

function readBranchPreference() {
  try {
    const stored = localStorage.getItem(branchPreferenceKey());
    return stored === null ? DEFAULT_BRANCH : stored;
  } catch {
    return DEFAULT_BRANCH;
  }
}

function writeBranchPreference(branch) {
  try {
    localStorage.setItem(branchPreferenceKey(), branch);
  } catch {
    // Storage may be disabled; branch selection still works for this page load.
  }
}

function readCachedCatalog() {
  try {
    const cached = JSON.parse(localStorage.getItem(catalogCacheKey()));
    if (!cached || !Array.isArray(cached.runs)) return null;
    return cached;
  } catch {
    return null;
  }
}

function writeCachedCatalog() {
  try {
    localStorage.setItem(catalogCacheKey(), JSON.stringify({
      savedAt: new Date().toISOString(),
      runs: state.runs,
      branches: state.availableBranches,
      cursor: state.cursor,
      hasMore: state.hasMore,
    }));
  } catch {
    // Storage may be disabled; the live registry remains fully functional.
  }
}

function applyCatalog(payload, { append = false, cache = true } = {}) {
  state.runs = append ? [...state.runs, ...payload.runs] : payload.runs;
  state.availableBranches = [...new Set([
    ...state.availableBranches,
    ...(payload.branches || []),
    ...payload.runs.map((run) => branchName(run.git_ref)),
  ])].sort();
  state.cursor = payload.next_cursor ?? payload.cursor ?? null;
  state.hasMore = Boolean(payload.has_more ?? payload.hasMore);
  populateFilters({ persistBranchFallback: cache });
  renderRuns();
  if (cache) writeCachedCatalog();
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("sima-neat-artifact-theme", theme);
  const dark = theme === "dark";
  elements["theme-toggle"].innerHTML = dark ? icons.sun : icons.moon;
  elements["theme-toggle"].setAttribute("aria-label", `Switch to ${dark ? "light" : "dark"} mode`);
  elements["theme-toggle"].title = `Switch to ${dark ? "light" : "dark"} mode`;
}

function filters() {
  return {
    query: elements.search.value.trim().toLowerCase(),
    branch: elements["branch-filter"].value,
    status: elements["status-filter"].value,
    compiler: elements["compiler-filter"].value,
  };
}

function filteredRuns() {
  const current = filters();
  return state.runs.filter((run) => {
    const haystack = [run.metadata?.model_id, run.metadata?.variant_id, run.git_ref, run.commit_sha, run.run_type, run.toolchain_version, run.target_platform].join(" ").toLowerCase();
    return (!current.query || haystack.includes(current.query))
      && (!current.branch || branchName(run.git_ref) === current.branch)
      && (!current.status || run.status === current.status)
      && (!current.compiler || run.toolchain_version === current.compiler);
  });
}

function populateFilters({ persistBranchFallback = false } = {}) {
  const selectedCompiler = elements["compiler-filter"].value;
  const branches = state.availableBranches;
  const compilers = [...new Set(state.runs.map((run) => run.toolchain_version).filter(Boolean))].sort().reverse();
  elements["branch-filter"].innerHTML = `<option value="">All branches</option>${branches.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  elements["compiler-filter"].innerHTML = `<option value="">Any version</option>${compilers.map((value) => `<option value="${escapeHtml(value)}">${escapeHtml(value)}</option>`).join("")}`;
  const preferredBranch = state.preferredBranch ?? DEFAULT_BRANCH;
  const selectedBranch = preferredBranch === "" || branches.includes(preferredBranch)
    ? preferredBranch
    : branches.includes(DEFAULT_BRANCH) ? DEFAULT_BRANCH : "";
  elements["branch-filter"].value = selectedBranch;
  if (persistBranchFallback && selectedBranch !== preferredBranch) {
    state.preferredBranch = selectedBranch;
    writeBranchPreference(selectedBranch);
  }
  elements["compiler-filter"].value = compilers.includes(selectedCompiler) ? selectedCompiler : "";
}

function renderRuns() {
  const runs = filteredRuns();
  elements["result-count"].textContent = `${runs.length}${state.hasMore ? "+" : ""} build${runs.length === 1 ? "" : "s"}`;
  elements["result-list"].setAttribute("aria-busy", "false");
  elements["load-more"].hidden = !state.hasMore;

  if (!runs.length) {
    elements["result-list"].innerHTML = `<div class="no-results"><span>0</span><h3>No matching artifacts</h3><p>Broaden the search or clear the current filters.</p><button type="button" data-action="clear">Reset filters</button></div>`;
    return;
  }

  elements["result-list"].innerHTML = runs.map((run) => {
    const model = run.metadata?.model_id || run.run_type.replaceAll("_", " ");
    const variant = run.metadata?.variant_id || "default artifact";
    const selected = run.id === state.selectedId;
    return `<button class="run-row${selected ? " selected" : ""}" type="button" data-run-id="${run.id}" aria-pressed="${selected}">
      <span class="run-status ${escapeHtml(run.status)}" title="${escapeHtml(run.status)}"></span>
      <span class="run-identity"><strong>${escapeHtml(model)}</strong><small>${escapeHtml(variant)}</small></span>
      <span class="run-branch"><small>Branch</small>${escapeHtml(branchName(run.git_ref))}</span>
      <span class="run-compiler"><small>Compiler</small>${escapeHtml(run.toolchain_version || "—")}</span>
      <span class="run-date"><small>${escapeHtml(shortSha(run.commit_sha))}</small>${escapeHtml(formatDate(run.finished_at || run.created_at))}</span>
      <span class="row-arrow">${icons.arrow}</span>
    </button>`;
  }).join("");
}

function detailSkeleton() {
  elements["detail-pane"].innerHTML = `<div class="detail-loading"><span></span><span></span><span></span><span></span></div>`;
}

async function selectRun(runId, { updateUrl = true } = {}) {
  state.selectedId = runId;
  renderRuns();
  detailSkeleton();
  if (updateUrl) {
    const url = new URL(location.href);
    url.searchParams.set("run", runId);
    history.replaceState({}, "", url);
  }
  try {
    let detail = state.detailCache.get(runId);
    if (!detail) {
      const [summary, provenance, tests, metrics] = await Promise.all([
        api(`/v1/runs/${runId}`), api(`/v1/runs/${runId}/provenance`),
        api(`/v1/runs/${runId}/test-results`), api(`/v1/runs/${runId}/metrics`),
      ]);
      detail = { summary, provenance, tests, metrics };
      state.detailCache.set(runId, detail);
    }
    if (state.selectedId === runId) renderDetail(detail);
  } catch (error) {
    elements["detail-pane"].innerHTML = errorPanel(error.message, "Could not load this build");
  }
}

function keyValue(label, value, { copy = false, mono = false } = {}) {
  const safeValue = escapeHtml(value ?? "—");
  return `<div class="key-value"><dt>${escapeHtml(label)}</dt><dd class="${mono ? "mono" : ""}"><span title="${safeValue}">${safeValue}</span>${copy && value ? `<button type="button" class="copy-button" data-copy="${safeValue}" aria-label="Copy ${escapeHtml(label)}">${icons.copy}</button>` : ""}</dd></div>`;
}

function renderDetail(detail) {
  const run = detail.summary.run;
  const artifacts = detail.summary.artifacts || [];
  const provenance = detail.provenance;
  const parameters = provenance.parameters?.parameters || {};
  const input = provenance.inputs?.[0];
  const tests = detail.tests.test_results || [];
  const measurements = detail.metrics.measurements || [];
  const model = run.metadata?.model_id || run.run_type;
  const variant = run.metadata?.variant_id || "default";
  const precision = parameters.compile_parameters?.precision;

  elements["detail-pane"].innerHTML = `<div class="detail-content">
    <div class="detail-topline"><span class="status-label ${escapeHtml(run.status)}"><i></i>${escapeHtml(run.status)}</span><button class="close-detail" type="button" aria-label="Close details">×</button></div>
    <div class="detail-title"><p class="kicker">${escapeHtml(run.target_platform || "Model artifact")}</p><h2>${escapeHtml(model)}</h2><p>${escapeHtml(variant)}</p></div>
    <div class="detail-tags">${precision ? `<span>${escapeHtml(precision)}</span>` : ""}<span>${escapeHtml(run.toolchain_name || "compiler")} ${escapeHtml(run.toolchain_version || "")}</span><span>${escapeHtml(branchName(run.git_ref))}</span></div>

    <section class="artifact-section"><div class="detail-section-title"><h3>Artifacts</h3><span>${artifacts.length}</span></div>
      ${artifacts.length ? artifacts.map((artifact) => `<div class="artifact-item">
        <div class="artifact-icon" aria-hidden="true">MPK</div>
        <div class="artifact-main"><strong>${escapeHtml(artifact.name)}</strong><small>${escapeHtml(formatBytes(artifact.size_bytes))} · ${escapeHtml(artifact.media_type || artifact.artifact_type)}</small></div>
        <button class="download-button" type="button" data-download="${escapeHtml(artifact.name)}" data-run-id="${run.id}">${icons.download}<span>Download</span></button>
        <div class="artifact-checksum"><span>SHA-256</span><code>${escapeHtml(artifact.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(artifact.sha256)}" aria-label="Copy artifact checksum">${icons.copy}</button></div>
      </div>`).join("") : `<p class="muted-message">No downloadable artifacts were registered for this run.</p>`}
    </section>

    <section><div class="detail-section-title"><h3>Build provenance</h3></div><dl class="key-grid">
      ${keyValue("Commit", run.commit_sha, { copy: true, mono: true })}
      ${keyValue("Branch", branchName(run.git_ref))}
      ${keyValue("Built", formatDate(run.finished_at || run.created_at))}
      ${keyValue("Duration", formatDuration(run.started_at, run.finished_at))}
      ${keyValue("Target", run.target_platform)}
      ${keyValue("Compiler", run.toolchain_version)}
      ${keyValue("Run ID", run.id, { copy: true, mono: true })}
      ${keyValue("Fingerprint", run.run_fingerprint, { copy: true, mono: true })}
    </dl>${run.ci_run_url ? `<a class="external-link" href="${escapeHtml(run.ci_run_url)}" target="_blank" rel="noreferrer">Open GitHub workflow ${icons.external}</a>` : ""}</section>

    ${input ? `<section><div class="detail-section-title"><h3>Source input</h3></div><div class="source-block"><strong>${escapeHtml(input.name)}</strong><p>${escapeHtml(input.source_reference || input.input_type)}</p><div><span>${escapeHtml(formatBytes(input.size_bytes))}</span><code>${escapeHtml(input.sha256)}</code><button type="button" class="copy-button" data-copy="${escapeHtml(input.sha256)}" aria-label="Copy source checksum">${icons.copy}</button></div></div></section>` : ""}

    <section><div class="detail-section-title"><h3>Validation</h3><span>${tests.length}</span></div>
      <div class="test-list">${tests.length ? tests.map((test) => `<div><span class="test-mark ${escapeHtml(test.status)}">${test.status === "passed" ? "✓" : "!"}</span><p><strong>${escapeHtml(test.name)}</strong><small>${escapeHtml(test.suite)}${test.duration_ms ? ` · ${escapeHtml(test.duration_ms)} ms` : ""}</small></p><span>${escapeHtml(test.status)}</span></div>`).join("") : `<p class="muted-message">No validation records.</p>`}</div>
      ${measurements.length ? `<p class="measurement-note">${measurements.length} performance measurement${measurements.length === 1 ? "" : "s"} recorded.</p>` : ""}
    </section>

    ${Object.keys(parameters.compile_parameters || {}).length ? `<details><summary>Compile parameters <span>JSON</span></summary><pre>${escapeHtml(JSON.stringify(parameters.compile_parameters, null, 2))}</pre></details>` : ""}
  </div>`;
}

function errorPanel(message, title = "Registry unavailable") {
  return `<div class="error-state"><span>!</span><p class="kicker">Connection error</p><h2>${escapeHtml(title)}</h2><p>${escapeHtml(message)}</p><button type="button" data-action="retry">Try again</button></div>`;
}

async function downloadArtifact(runId, name, button) {
  const prior = button.innerHTML;
  button.disabled = true;
  button.innerHTML = `<span class="spinner"></span><span>Preparing…</span>`;
  try {
    const payload = await api(`/v1/runs/${runId}/artifacts/${encodeURIComponent(name)}`);
    if (!payload.artifact?.download_url) throw new Error("The registry did not return a download URL.");
    showToast(`Download ready for ${name}. The link is valid for ${Math.round((payload.artifact.download_url_expires_in || 300) / 60)} minutes.`);
    location.href = payload.artifact.download_url;
  } catch (error) {
    showToast(error.message, true);
  } finally {
    button.disabled = false;
    button.innerHTML = prior;
  }
}

function showToast(message, isError = false) {
  const toast = document.createElement("div");
  toast.className = `toast${isError ? " error" : ""}`;
  toast.textContent = message;
  elements["toast-region"].append(toast);
  setTimeout(() => toast.remove(), 5000);
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
    showToast("Copied to clipboard.");
  } catch {
    showToast("Clipboard access is unavailable.", true);
  }
}

async function loadRuns({ append = false, branch = elements["branch-filter"].value, limit = 50 } = {}) {
  const repo = state.repository;
  if (!repo) return;
  elements["load-more"].disabled = true;
  const query = new URLSearchParams({ limit: String(limit) });
  if (branch) query.set("git_ref", `refs/heads/${branch}`);
  if (append && state.cursor) query.set("cursor", state.cursor);

  try {
    const payload = await api(`/v1/repositories/${encodeURIComponent(repo.organization)}/${encodeURIComponent(repo.name)}/runs?${query}`);
    if (!append && branch && !payload.runs.length && branch !== DEFAULT_BRANCH) {
      state.availableBranches = state.availableBranches.filter((value) => value !== branch);
      state.preferredBranch = DEFAULT_BRANCH;
      writeBranchPreference(DEFAULT_BRANCH);
      await loadRuns({ branch: DEFAULT_BRANCH });
      return;
    }
    applyCatalog(payload, { append });
  } finally {
    elements["load-more"].disabled = false;
  }

  const requested = state.selectedId && state.runs.some((run) => run.id === state.selectedId) ? state.selectedId : null;
  if (requested && !append) selectRun(requested, { updateUrl: false });
}

async function initialize() {
  setTheme(document.documentElement.dataset.theme || "light");
  state.preferredBranch = readBranchPreference();
  elements["environment-name"].textContent = `${config.environment || "Public"} registry`;
  elements["result-list"].innerHTML = Array.from({ length: 5 }, () => `<div class="row-skeleton"><span></span><span></span><span></span></div>`).join("");

  if (!/^https:\/\/[^/]+$/.test(state.apiBase) || !state.repository.organization || !state.repository.name) {
    elements["result-list"].innerHTML = errorPanel("Set a valid HTTPS API origin and repository in config.js.", "Portal configuration is invalid");
    elements["result-list"].setAttribute("aria-busy", "false");
    return;
  }

  const cached = readCachedCatalog();
  if (cached) {
    state.availableBranches = Array.isArray(cached.branches) ? cached.branches : [];
    applyCatalog({
      runs: cached.runs,
      next_cursor: cached.cursor,
      has_more: cached.hasMore,
    }, { cache: false });
    elements["registry-copy"].textContent = `${state.repository.organization}/${state.repository.name} · refreshing…`;
  }

  try {
    if (cached) {
      await loadRuns({ branch: state.preferredBranch });
    } else {
      // The first broad page discovers branch choices. If it is incomplete,
      // switch to branch-aware pagination so the cursor matches the visible list.
      await loadRuns({ branch: "", limit: 200 });
      const selectedBranch = elements["branch-filter"].value;
      if (selectedBranch && state.hasMore) await loadRuns({ branch: selectedBranch });
    }
    elements["registry-status"].className = "status-dot online";
    elements["registry-copy"].textContent = `${state.repository.organization}/${state.repository.name} · live public data`;
    elements["result-title"].textContent = state.repository.name === "models" ? "Builds" : state.repository.name;
  } catch (error) {
    elements["registry-status"].className = "status-dot offline";
    elements["registry-copy"].textContent = cached ? "Showing saved data · registry unavailable" : "Registry unavailable";
    if (cached) showToast(error.message, true);
    else {
      elements["result-list"].innerHTML = errorPanel(error.message);
      elements["result-list"].setAttribute("aria-busy", "false");
    }
  }
}

elements["theme-toggle"].addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
elements.search.addEventListener("input", renderRuns);
elements["branch-filter"].addEventListener("change", async () => {
  state.preferredBranch = elements["branch-filter"].value;
  writeBranchPreference(state.preferredBranch);
  try {
    await loadRuns({ branch: state.preferredBranch });
  } catch (error) {
    showToast(error.message, true);
  }
});
[elements["status-filter"], elements["compiler-filter"]].forEach((element) => element.addEventListener("change", renderRuns));
elements["clear-filters"].addEventListener("click", async () => {
  elements.search.value = "";
  elements["branch-filter"].value = "";
  state.preferredBranch = "";
  writeBranchPreference(state.preferredBranch);
  elements["status-filter"].value = "";
  elements["compiler-filter"].value = "";
  try {
    await loadRuns({ branch: "", limit: 200 });
  } catch (error) {
    showToast(error.message, true);
  }
});
elements["result-list"].addEventListener("click", (event) => {
  const run = event.target.closest("[data-run-id]");
  if (run) selectRun(run.dataset.runId);
  if (event.target.closest('[data-action="clear"]')) elements["clear-filters"].click();
  if (event.target.closest('[data-action="retry"]')) initialize();
});
elements["detail-pane"].addEventListener("click", (event) => {
  const copy = event.target.closest("[data-copy]");
  if (copy) copyText(copy.dataset.copy);
  const download = event.target.closest("[data-download]");
  if (download) downloadArtifact(download.dataset.runId, download.dataset.download, download);
  if (event.target.closest(".close-detail")) {
    state.selectedId = null;
    history.replaceState({}, "", location.pathname);
    renderRuns();
    elements["detail-pane"].innerHTML = `<div class="detail-empty"><h2>No build selected</h2><p>Select a build to inspect provenance, validation results, checksums, and downloads.</p></div>`;
  }
});
elements["load-more"].addEventListener("click", () => loadRuns({
  append: true,
  branch: elements["branch-filter"].value,
}).catch((error) => showToast(error.message, true)));
document.addEventListener("keydown", (event) => {
  if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) {
    event.preventDefault();
    elements.search.focus();
  }
  if (event.key === "Escape" && document.activeElement === elements.search) {
    elements.search.value = "";
    elements.search.blur();
    renderRuns();
  }
});

initialize();
