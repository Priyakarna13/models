// Execution Graph, Runtime Layer Analysis, and MLA Utilization of the model page.
import { selectLayers, layerTimeScale, layerFamily, LAYER_FAMILIES } from "./mla-layer-stats.js";
import {
  MLA_PEAK_INT8_TOPS, matchMlaStages, mlaPeakTops, mlaUtilizationMetrics, sustainedEffectiveTops,
} from "./mla-utilization.js";
import { shows, state, elements } from "./state.js";
import { icons, escapeHtml, disclosure } from "./ui.js";
import { componentPanels, componentSwitch, infoMark, tile } from "./page-parts.js";

const EXECUTION_STAGE_INFO = {
  Preproc: ["Prepare input", "Prepares input data for the model."],
  Cast: ["Convert input", "Changes the input data type."],
  Quant: ["Quantize input", "Converts input values to the model’s integer format."],
  Tess: ["Arrange input", "Rearranges input data for the accelerator."],
  CastTess: ["Convert & Arrange Input", "Changes the input data type and prepares its layout for the accelerator."],
  QuantTess: ["Quantize & Arrange Input", "Converts input values to integers and prepares their layout for the accelerator."],
  "Compiled Model": ["Model Inference", "Runs the compiled model."],
  Detess: ["Restore output layout", "Rearranges the model output for further processing."],
  Dequant: ["Dequantize output", "Converts integer output values back to floating point."],
  DetessCast: ["Convert & arrange output", "Restores the output layout and changes its data type."],
  DetessDequant: ["Prepare output", "Restores the output layout and converts values back to floating point."],
  SimaBoxDecode: ["Decode output", "Converts raw model outputs into usable results."],
};

const PROCESSOR_LABELS = { EV74: "Vector Processor", MLA: "Machine Learning Accelerator", A65: "Arm CPU" };

function executionStageInfo(step) {
  const [title, description] = EXECUTION_STAGE_INFO[step.name] || [step.name, "Runs this processing step."];
  const processor = [step.runsOn, PROCESSOR_LABELS[step.runsOn]].filter(Boolean).join(" · ");
  return `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p>
    ${step.runsOn ? `<span class="execution-tooltip-processor" data-runs-on="${escapeHtml(step.runsOn)}">${escapeHtml(processor)}</span>` : ""}`;
}

let executionTooltipStage = null;

let executionTooltipTimer;

export function hideExecutionTooltip() {
  clearTimeout(executionTooltipTimer);
  executionTooltipStage?.removeAttribute("aria-describedby");
  executionTooltipStage = null;
  elements["execution-tooltip"].hidden = true;
}

export function keepExecutionTooltip() {
  clearTimeout(executionTooltipTimer);
}

export function deferHideExecutionTooltip() {
  executionTooltipTimer = setTimeout(hideExecutionTooltip, 150);
}

export function showExecutionTooltip(stage) {
  hideExecutionTooltip();
  const tooltip = elements["execution-tooltip"];
  tooltip.innerHTML = stage.querySelector("template").innerHTML;
  tooltip.dataset.runsOn = stage.dataset.runsOn || "";
  tooltip.hidden = false;
  executionTooltipStage = stage;
  stage.setAttribute("aria-describedby", tooltip.id);
  const anchor = stage.getBoundingClientRect();
  const roomAbove = Math.max(0, anchor.top - 84);
  const roomBelow = Math.max(0, innerHeight - anchor.bottom - 24);
  tooltip.style.maxHeight = `${Math.max(roomAbove, roomBelow)}px`;
  const box = tooltip.getBoundingClientRect();
  tooltip.style.left = `${Math.max(12, Math.min(innerWidth - box.width - 12, anchor.left + anchor.width / 2 - box.width / 2))}px`;
  tooltip.style.top = `${box.height <= roomAbove + .5 ? anchor.top - box.height - 12 : anchor.bottom + 12}px`;
}

// Each step of the benchmarked Model and the processor it ran on. Details opens its tensor table.
function executionGraphDiagram(graph, runId) {
  if (!graph) return "";
  const steps = graph.steps.map((step) => step.runsOn
    ? `<li><button type="button" class="execution-step" data-execution-stage data-runs-on="${escapeHtml(step.runsOn)}">${escapeHtml(step.name)}<small>${escapeHtml(step.runsOn)}</small><template>${executionStageInfo(step)}</template></button></li>`
    : `<li class="execution-end"><span class="execution-step">${escapeHtml(step.name)}</span></li>`).join("");
  const rows = graph.tensors.map((tensor) => `<tr${tensor.connection ? "" : ' class="continued"'}><th scope="row">${escapeHtml(tensor.connection)}</th>${[tensor.dtype, tensor.shape, tensor.scale, tensor.zeroPoint].map((value) => `<td>${escapeHtml(value)}</td>`).join("")}</tr>`).join("");
  const table = `<div class="tensor-table"><table><thead><tr>${["Connection", "Type", "Shape", "Scale", "Zero point"].map((label) => `<th scope="col">${label}</th>`).join("")}</tr></thead><tbody>${rows}</tbody></table></div>`;
  const key = `tensors:${runId}`;
  const open = state.openDisclosures.has(key);
  const details = graph.tensors.length
    ? `<button type="button" class="execution-details" data-details="${escapeHtml(key)}" aria-controls="${escapeHtml(key)}" aria-expanded="${open}">Details${icons.chevron}</button>`
    : "";
  return `<section class="execution-graph" aria-label="Execution graph"><div class="execution-graph-row"><ol>${steps}</ol>${details}</div>
    ${graph.tensors.length ? `<div id="${escapeHtml(key)}"${open ? "" : " hidden"}>${table}</div>` : ""}</section>`;
}

const layerTime = (value) => value.toLocaleString(undefined, { maximumFractionDigits: 2 });

function layerAnalysisResults(analysis, view) {
  const { layers, totalUs, count } = selectLayers(analysis, view);
  const scaleUs = layerTimeScale(analysis);
  const label = (layer) => `<th scope="row" class="layer-name" title="${escapeHtml(layer.name)}">${escapeHtml(layer.name.replace(/^MLA_\d+\//, ""))}<small>Stage ${layer.stage} · #${layer.order}</small></th>`;
  const rows = layers.map((layer) => `<tr class="layer-family-${layerFamily(layer.name)}" data-layer-family="${layerFamily(layer.name)}">
    ${label(layer)}
    <td><div class="layer-timing"><span class="layer-timing-number">${layerTime(layer.timeUs)}</span><span class="layer-bar-track" aria-hidden="true"><span style="width:${layer.timeUs / scaleUs * 100}%"></span></span></div></td>
    <td class="layer-share">${(totalUs > 0 ? layer.timeUs / totalUs * 100 : 0).toFixed(2)}%</td>
  </tr>`).join("");
  const families = new Set(analysis.layers.map((layer) => layerFamily(layer.name)));
  const legend = Object.entries(LAYER_FAMILIES).filter(([id]) => families.has(id))
    .map(([id, title]) => `<span class="layer-family-${id}" data-layer-family="${id}"><i aria-hidden="true"></i><span class="layer-legend-label" data-label="${escapeHtml(title)}"><span>${title}</span></span></span>`).join("");
  const axis = `<span class="layer-axis" aria-label="Scale: 0 to ${layerTime(scaleUs)} microseconds"><span>0</span><span>${layerTime(scaleUs / 2)}</span><span>${layerTime(scaleUs)}</span></span>`;
  return `<p class="layer-results" role="status">${layers.length.toLocaleString()} of ${count.toLocaleString()} layers · ${view.order === "compiler" ? "Runtime order" : "Slowest layers first"}</p>
    <div class="layer-legend" aria-label="Layer types inferred from compiled names"><span>Layer type (from name)</span>${legend}</div>
    ${layers.length ? `<div class="layer-table-scroll" tabindex="0" role="region" aria-label="Compiled layer timings"><table class="layer-table">
      <colgroup><col><col><col></colgroup>
      <thead><tr><th scope="col">Compiled layer</th><th scope="col">Execution time <span class="layer-time-unit">(µs)</span>${axis}</th><th scope="col">Share (%)</th></tr></thead>
      <tbody>${rows}</tbody></table></div>` : '<p class="muted-message">No layers match these filters.</p>'}`;
}

function layerAnalysisBody(analysis, view) {
  const { count, zeroCount, top10Share } = selectLayers(analysis);
  const scaleUs = layerTimeScale(analysis);
  return `<div class="layer-summary">
    <div><strong>${layerTime(analysis.totalUs)} <small>µs</small></strong><span>Profiled MLA time</span></div>
    <div><strong>${count.toLocaleString()}</strong><span>Compiled layers · ${zeroCount} zero-time</span></div>
    <div><strong>${top10Share.toFixed(1)}<small>%</small></strong><span>Time in 10 slowest layers</span></div>
  </div>
  <p class="layer-explanation">Measured execution times from a separate MLA profiling pass. Shares use the component’s profiled time, not end-to-end latency. Compiled layer names may differ from ONNX operators.</p>
  <div class="layer-controls">
    <label class="layer-search">Layer name<input type="search" data-layer-filter="query" value="${escapeHtml(view.query)}" placeholder="Search compiled layers" autocomplete="off"></label>
    <label class="layer-threshold"><span>Minimum time: <output data-layer-threshold-value>${layerTime(view.minUs)} µs</output></span><input type="range" data-layer-filter="minUs" min="0" max="${scaleUs}" step="any" value="${view.minUs}"></label>
    <div class="layer-order-toggle" role="group" aria-label="Layer order">${[["compiler", "Runtime order"], ["time", "Slowest layers first"]].map(([order, label]) => `<button type="button" data-layer-filter="order" value="${order}" aria-pressed="${view.order === order}">${label}</button>`).join("")}</div>
  </div>
  <div data-layer-results>${layerAnalysisResults(analysis, view)}</div>`;
}

// The layer analysis of one component, or of a pipeline's components with a switch between them.
function layerAnalysisPanel({ sources, selected, view }) {
  const switcher = sources.length > 1 ? componentSwitch(sources.map(({ label }) => label), selected, 'data-layer-filter="component"') : "";
  return `${switcher}${layerAnalysisBody(sources[selected].analysis, view)}`;
}

// The execution graph of one component, or of a pipeline's components with a switch between them.
export function executionGraphSection(components, key) {
  const graphed = components.filter(({ graph }) => graph);
  if (!graphed.length) return "";
  const meta = graphed.length > 1
    ? graphed.map(({ label }) => label).join(" · ")
    : graphed[0].graph.steps.filter((step) => step.runsOn).map((step) => step.runsOn).join(" · ");
  return disclosure(`graphs:${key}`, "Execution Graph",
    componentPanels(graphed.map(({ label }) => label), graphed.map(({ id, graph }) => executionGraphDiagram(graph, id))), meta);
}

export function layerAnalysisSection(sources, key) {
  const analyzed = sources.filter(({ analysis }) => analysis);
  if (!analyzed.length) return "";
  const previous = state.layerViews.get(key);
  const entry = { sources: analyzed, selected: previous?.selected || 0, view: previous?.view || { query: "", minUs: 0, order: "time" } };
  state.layerViews.set(key, entry);
  const { analysis } = analyzed[0];
  const meta = analyzed.length > 1
    ? analyzed.map(({ label }) => label).join(" · ")
    : `${analysis.layers.length.toLocaleString()} layers · ${layerTime(analysis.totalUs)} µs MLA`;
  return disclosure(`layers:${key}`, "Runtime Layer Analysis",
    `<div data-layer-analysis="${escapeHtml(key)}">${layerAnalysisPanel(entry)}</div>`, meta);
}

export function updateLayerAnalysis(event) {
  const control = event.target.closest("[data-layer-filter]");
  if (!control) return;
  const field = control.dataset.layerFilter;
  if (event.type !== (["order", "component"].includes(field) ? "click" : "input")) return;
  const panel = control.closest("[data-layer-analysis]");
  const entry = state.layerViews.get(panel.dataset.layerAnalysis);
  if (field === "component") {
    // Components have different time scales, so the minimum time starts over.
    entry.selected = Number(control.value);
    entry.view.minUs = 0;
    panel.innerHTML = layerAnalysisPanel(entry);
    return;
  }
  const { view } = entry;
  const { analysis } = entry.sources[entry.selected];
  view[field] = field === "minUs" ? Number(control.value) : control.value;
  panel.querySelector("[data-layer-results]").innerHTML = layerAnalysisResults(analysis, view);
  panel.querySelector("[data-layer-threshold-value]").textContent = `${layerTime(view.minUs)} µs`;
  panel.querySelectorAll('[data-layer-filter="order"]').forEach((button) => {
    button.setAttribute("aria-pressed", String(button.value === view.order));
  });
}

export function highlightLayerLegend(event) {
  const row = event.target.closest(".layer-table tbody tr[data-layer-family]");
  if (!row || row.contains(event.relatedTarget)) return;
  const panel = row.closest("[data-layer-analysis]");
  const family = row.dataset.layerFamily;
  panel.querySelectorAll(".layer-legend [data-layer-family]").forEach((item) => {
    item.classList.toggle("is-highlighted", event.type === "pointerover" && item.dataset.layerFamily === family);
  });
}

// Each component's MLA stages set against the simulator's operation counts, in developer mode only: the rates rest
// on a simulator estimate.
export function renderMlaUtilization(components, simulation, precisions, throughputFps, key) {
  if (!shows.mlaUtilization || !simulation && !components.some(({ mlaStageStats }) => mlaStageStats?.status === "ok")) return "";
  const results = components.map((component) => {
    const { stages, reason } = matchMlaStages(component.mlaStageStats, simulation?.components?.[component.id]);
    const peak = mlaPeakTops(precisions[component.id]);
    return { ...component, reason, peak, metrics: stages && mlaUtilizationMetrics(stages, { peakTops: peak?.tops }) };
  });
  const number = (value, digits) => value === null ? "—" : value.toLocaleString(undefined, { maximumFractionDigits: digits });
  const percent = (value) => value === null ? "—" : `${(value * 100).toFixed(1)}%`;
  // Component numbers are never added up, so only a single-component model gets a model-level rate.
  const compared = results.length === 1 && results[0].metrics && Number.isFinite(throughputFps) && results[0].batch > 0;
  const sustainedTops = compared ? sustainedEffectiveTops(results[0].metrics.total, throughputFps, results[0].batch) : null;
  // A component's totals as tiles in the layer summary's style, each explained behind an (i) mark.
  const tile = (value, digits, unit, label, info) => `<div><strong>${number(value, digits)}${value === null ? "" : unit}</strong><span>${label}${infoMark(info)}</span></div>`;
  const summary = ({ peak, metrics: { stages, total } }) => `<div class="layer-summary mla-summary">
    ${total.utilization === null ? "" : tile(total.utilization * 100, 1, "<small>%</small>", "Utilization",
      `Effective TOPS as a share of the MLA's ${number(peak.tops, 1)} TOPS ${peak.datatype} peak.${peak.datatype === "BF16"
        ? `\nA BF16 multiply takes four times the cycles of an INT8 one, so the peak is a quarter of ${MLA_PEAK_INT8_TOPS} TOPS.` : ""}`)}
    ${tile(total.effectiveTops, 2, " <small>TOPS</small>", "Effective TOPS",
      `Useful operations per second: the Model Compiler simulator's count over the measured MLA time.${compared ? `\nSustained at the measured throughput: ${number(sustainedTops, 2)} TOPS.` : ""}`)}
    ${tile(total.packing === null ? null : total.packing * 100, 1, "<small>%</small>", "Packing",
      "Share of the operations the MLA executes that are useful. The rest is padding.")}
    ${tile(total.measuredUs, 1, " <small>µs</small>", "MLA time",
      `Measured on the board${stages.length > 1 ? ` across ${stages.length} MLA stages` : ` for MLA stage ${stages[0].index}`}.`)}
  </div>`;
  const row = (stage) => `<tr><th scope="row">MLA stage ${stage.index}</th>${[number(stage.measuredUs, 1), number(stage.simulatedUs, 1),
    number(stage.effectiveOps / 1e9, 2), percent(stage.packing), number(stage.effectiveTops, 2), percent(stage.utilization)]
    .map((value) => `<td>${value}</td>`).join("")}</tr>`;
  const headings = ["Stage", "Measured <small>(µs)</small>", "Simulated <small>(µs)</small>", "Effective GOP", "Packing",
    "Effective TOPS", "Utilization"];
  // Each stage's numbers stay behind a Details button, as the execution graph's tensors do.
  const stageDetails = ({ id, peak, metrics }) => {
    const detailsKey = `mla-stages:${key}:${id}`;
    const open = state.openDisclosures.has(detailsKey);
    return `<div class="mla-summary-row">${summary({ peak, metrics })}
      <button type="button" class="execution-details" data-details="${escapeHtml(detailsKey)}" aria-controls="${escapeHtml(detailsKey)}" aria-expanded="${open}">Details${icons.chevron}</button></div>
      <div id="${escapeHtml(detailsKey)}" class="tensor-table"${open ? "" : " hidden"}><table><thead><tr>${headings.map((heading) => `<th scope="col">${heading}</th>`).join("")}</tr></thead>
      <tbody>${metrics.stages.map(row).join("")}</tbody></table></div>`;
  };
  const panels = results.map((result) => (result.metrics
    ? stageDetails(result) : `<p class="muted-message">${escapeHtml(result.reason)}</p>`));
  const meta = compared ? `${number(sustainedTops, 2)} effective TOPS sustained`
    : results.length > 1 ? results.map(({ label }) => label).join(" · ") : "";
  return disclosure(`mla-utilization:${key}`, "MLA Utilization",
    `<div class="mla-utilization">${componentPanels(results.map(({ label }) => label), panels)}</div>`, meta);
}
