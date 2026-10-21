import { METRIC_CATALOG } from "./metric-catalog.js";

export const ACCURACY_METRIC_LABELS = Object.fromEntries(Object.entries(METRIC_CATALOG)
  .filter(([, metric]) => metric.stage === "accuracy").map(([name, metric]) => [name, metric.display_name]));

export function accuracyMetricLabel(name) {
  return ACCURACY_METRIC_LABELS[name] || ACCURACY_METRIC_LABELS[`model.accuracy.${name}`] || name;
}

// Known metrics follow the label order and any others follow them in the order given.
export function orderedMetricNames(names) {
  const known = Object.keys(ACCURACY_METRIC_LABELS);
  const rank = (name) => {
    const canonical = name.startsWith("model.accuracy.") ? name : `model.accuracy.${name}`;
    return known.includes(canonical) ? known.indexOf(canonical) : known.length;
  };
  return [...names].sort((left, right) => rank(left) - rank(right));
}

// Upstream reference scores use their source vocabulary; report readings use canonical identities.
export function publishedScores(published, datasetId) {
  const scores = {};
  for (const entry of published || []) {
    if (entry.dataset?.id !== datasetId) continue;
    for (const [name, value] of Object.entries(entry.metrics || {})) {
      if (Number.isFinite(value)) scores[`model.accuracy.${name}`] = value;
    }
  }
  return scores;
}
