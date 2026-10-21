import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {readAccuracy, readPerformance} = await import(new URL('../../artifact-browser/model-report.js', import.meta.url));
const fixture = (stage) => JSON.parse(readFileSync(
  new URL(`../../artifact-browser/fixtures/model-report-${stage}.json`, import.meta.url), 'utf8'));
const run = {
  id: 'run-1', finished_at: '2026-10-01T08:07:42+00:00',
  metadata: {model_id: 'resnet_50', build_profile_id: 'w8a8', stage: 'performance'},
};
const performance = (report = fixture('performance')) => readPerformance({run, parameters: {}, report});

test('names the Sentinel source of a model report', () => {
  const {telemetry} = performance().value;
  assert.deepEqual(telemetry, {status: 'ok', sentinelVersion: telemetry.sentinelVersion, samples: 15, intervalMs: 1999});
  assert.match(telemetry.sentinelVersion, /\S/);
});

test('keeps the rail breakdown on peak SoM power', () => {
  const {measurements} = performance().value;
  const rails = measurements.peak_power_watts.details.breakdown;
  assert.ok(rails.length > 1);
  assert.ok(rails.every((rail) => rail.mean <= rail.maximum));
  assert.equal(measurements.peak_power_watts.sampleCount, 15);
  assert.deepEqual(measurements.peak_die_temperature_c.details, {});
});

test('reports no telemetry source when a report has no telemetry', () => {
  const {telemetry, ...report} = fixture('performance');
  const benchmark = performance(report).value;
  assert.equal(benchmark.telemetry, null);
  assert.deepEqual(benchmark.cpuCores, []);
  assert.equal(benchmark.measurements.avg_power_watts.sampleCount, undefined);
});

test('leaves out a metric the report did not measure as a number', () => {
  const report = fixture('performance');
  report.metrics.latency_p99_ms = null;
  const {measurements} = performance(report).value;
  assert.equal(measurements.latency_p99_ms, undefined);
  assert.equal(measurements.throughput_fps.value, report.metrics.throughput_fps);
});

test('names the model, build, runtime, and time of the run that measured it', () => {
  const benchmark = performance().value;
  assert.equal(benchmark.modelId, 'resnet_50');
  assert.equal(benchmark.buildProfileId, 'w8a8');
  assert.equal(benchmark.timestamp, run.finished_at);
  assert.match(benchmark.tool, /^PyNeat \S/);
  assert.equal(benchmark.workload, 'Model.benchmark with default options');
  assert.ok(benchmark.cpuCores.length > 1);
});

test('reads model report version 1 of the right stage and no other format', () => {
  const report = fixture('performance');
  assert.equal(readPerformance(null).kind, 'missing');
  assert.equal(readPerformance({run, parameters: {}, report: null}).kind, 'missing');
  assert.equal(performance(report).kind, 'supported');
  for (const other of [{...report, schema_version: 2}, {...report, schema: 'sima.benchmark'}, {...report, metrics: null}]) {
    assert.equal(performance(other).kind, 'unsupported');
  }
  assert.equal(readAccuracy({run, parameters: {}, report}).kind, 'unsupported');
});

test('reads accuracy as ratios over the dataset the run parameters name', () => {
  const report = fixture('accuracy');
  const parameters = {accuracy: {dataset: {id: 'imagenet-2012', variant: 'mini', sha256: 'abc'}}};
  const accuracy = readAccuracy({run, parameters, report: {...report, metrics: {...report.metrics, extra: 'n/a'}}}).value;
  assert.deepEqual(accuracy.metrics, report.metrics);
  assert.deepEqual(accuracy.dataset, {id: 'imagenet-2012', variant: 'mini', sha256: 'abc', ...report.dataset});
  assert.deepEqual(accuracy.details, report.details);
  assert.equal(readAccuracy({run, parameters: {}, report: {...report, stage: 'performance'}}).kind, 'unsupported');
});
