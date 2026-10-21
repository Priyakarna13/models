import { analyzeLayers } from "./layer-stats.js";

function finite(value) {
  return Number.isFinite(value) ? value : null;
}

// Keep each distinct context visible. A repeated metric never overwrites another workload.
export function reportMetrics(report) {
  const counts = new Map();
  for (const metric of report.metrics || []) counts.set(metric.name, (counts.get(metric.name) || 0) + 1);
  return Object.fromEntries((report.metrics || []).map((metric, index) => {
    const context = { ...report.context, ...metric.context };
    const key = counts.get(metric.name) === 1 ? metric.name : `${metric.name} [${JSON.stringify(metric.context || {})}] #${index + 1}`;
    return [key, {
      ...metric, key, context,
      status: metric.status === "measured" && Number.isFinite(metric.value) ? "measured" : "unavailable",
      sampleCount: metric.sample_count || 0, details: metric.details || {},
    }];
  }));
}

export function adaptBenchmarkResult(report) {
  if (!report || typeof report !== "object") return { kind: "missing" };
  if (report.schema !== "sima.model-report" || report.schema_version !== 1 || report.stage !== "performance") {
    return { kind: "unsupported", schemaVersion: report.schema_version };
  }
  const environment = report.provenance?.environment || {};
  const context = report.context || {};
  const telemetry = report.details?.telemetry;
  const components = Object.entries(report.components || {}).map(([id, component]) => ({
    id, layerAnalysis: analyzeLayers(component.layer_stats), executionGraph: executionGraph(component.execution_plan),
  }));
  return { kind: "supported", value: {
    schemaVersion: 1, workload: context.workload?.selection || context.workload?.boundary || "Whole model",
    status: "passed", tool: context.core_version ? `Neat Core ${context.core_version}` : context.tool || "",
    timestamp: report.provenance?.measured_at, measurements: reportMetrics(report),
    samples: context.num_samples, platformVersion: environment.runtime?.devkit?.platform_version,
    neatVersion: context.core_version, mlaFrequencyHz: environment.mla_frequency_hz,
    cpuFrequencyKhz: environment.cpu_frequency_khz, modelId: report.model?.id,
    buildProfileId: report.model?.build_profile_id, componentId: null, packageId: null,
    cpuCount: environment.cpu_count, batchSize: context.workload?.batch_size || context.batch_size,
    components, executionGraph: components[0]?.executionGraph, layerAnalysis: components[0]?.layerAnalysis,
    telemetry: telemetry ? { status: telemetry.status, sentinelVersion: telemetry.sentinel_version,
      samples: telemetry.samples, intervalMs: telemetry.sample_interval_ms,
      windowMs: report.details?.throughput_duration_ms } : null,
    cpuCores: report.details?.cpu_cores || [],
  } };
}

export function modelBenchmark(variant) {
  const payload = variant.performance;
  const adapted = adaptBenchmarkResult(payload?.result);
  return adapted.kind === "supported" ? { run: payload.run, payload, result: adapted.value } : null;
}

// A quantization parameter is one number, or one per channel for a per-axis tensor.
function quantText(value, format) {
  if (Array.isArray(value)) return "per channel";
  return Number.isFinite(value) ? format(value) : "—";
}

// The steps the benchmarked Model ran, in order, and the tensors on each connection, read from the
// Core execution plan the performance script records. Input and Output are the graph's ends; the MLA
// step is the compiled model. Another plan schema shows no graph.
export function executionGraph(plan) {
  if (plan?.schema !== "sima.neat.execution_plan" || plan.schema_version !== 1
    || !Array.isArray(plan.nodes) || !Array.isArray(plan.edges)) return null;
  const steps = plan.nodes.map((node) => (["Input", "Output"].includes(node.kind)
    ? { name: node.kind, runsOn: null }
    : { name: node.runs_on === "MLA" ? "Compiled Model" : String(node.kind), runsOn: String(node.runs_on) }));
  const names = new Map(plan.nodes.map((node, index) => [node.id, steps[index].name]));
  const tensors = plan.edges.flatMap((edge) => (Array.isArray(edge.tensors) ? edge.tensors : []).map((tensor, index) => ({
    connection: index === 0 ? `${names.get(edge.from) ?? edge.from} → ${names.get(edge.to) ?? edge.to}` : "",
    dtype: String(tensor.dtype),
    shape: Array.isArray(tensor.shape) ? tensor.shape.join("×") : "—",
    scale: quantText(tensor.quant?.scale, (value) => (value >= 1 ? value.toFixed(2) : String(Number(value.toPrecision(3))))),
    zeroPoint: quantText(tensor.quant?.zero_point, String),
  })));
  return { steps, tensors };
}
