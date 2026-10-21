import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {adaptBenchmarkResult} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const fixture = () => JSON.parse(readFileSync(
  new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url), 'utf8'));

test('names the Sentinel source and window of a saved Core report', () => {
  const benchmark = adaptBenchmarkResult(fixture()).value;
  assert.equal(benchmark.telemetry.status, 'ok');
  assert.equal(benchmark.telemetry.samples, 15);
  assert.equal(benchmark.telemetry.intervalMs, 1999);
  assert.ok(benchmark.telemetry.windowMs >= 30000);
  assert.match(benchmark.telemetry.sentinelVersion, /\S/);
});

test('keeps the rail breakdown on peak SoM power', () => {
  const {measurements} = adaptBenchmarkResult(fixture()).value;
  const rails = measurements["board.power.peak"].details.breakdown;
  assert.ok(rails.length > 1);
  assert.ok(rails.every((rail) => rail.unit === 'W' && rail.mean <= rail.maximum));
  assert.equal(measurements["board.die_temperature.peak"].details.breakdown, undefined);
  assert.equal(measurements["board.die_temperature.peak"].unit, '°C');
});

test('reports no telemetry source when optional diagnostics are absent', () => {
  const result = fixture(); delete result.details.telemetry;
  assert.equal(adaptBenchmarkResult(result).value.telemetry, null);
});

test('unavailable measurements retain availability without a fabricated zero', () => {
  const result = fixture();
  result.metrics = [{name: 'board.power.avg', status: 'unavailable', reason: 'No samples'}];
  const {measurements} = adaptBenchmarkResult(result).value;
  assert.equal(measurements['board.power.avg'].status, 'unavailable');
  assert.equal(measurements['board.power.avg'].value, undefined);
});

test('names the model and profile without pretending a component owns the result', () => {
  const benchmark = adaptBenchmarkResult(fixture()).value;
  assert.equal(benchmark.modelId, 'resnet_50');
  assert.equal(benchmark.buildProfileId, 'w8a8');
  assert.equal(benchmark.componentId, null);
});

test('requires the named current contract and rejects old or unknown formats', () => {
  assert.equal(adaptBenchmarkResult(fixture()).kind, 'supported');
  for (const version of [2,3,4,5,6,7]) assert.equal(adaptBenchmarkResult({...fixture(), schema_version: version}).kind, 'unsupported');
  assert.equal(adaptBenchmarkResult({schema_version:1,metrics:{}}).kind, 'unsupported');
});

test('all GenAI contexts remain visible, including unknown metrics', () => {
  const result = fixture();
  result.metrics = [16,32,64].map((tokens) => ({name:'language.decode_rate',status:'measured',value:tokens+1,unit:'tokens/s',context:{input_tokens:tokens}}));
  const metrics = Object.values(adaptBenchmarkResult(result).value.measurements);
  assert.equal(metrics.length, 3);
  assert.deepEqual(metrics.map((metric) => metric.value), [17,33,65]);
  assert.deepEqual(metrics.map((metric) => metric.context.input_tokens), [16,32,64]);
});
