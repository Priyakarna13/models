import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {wholeModelMeasurements} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const componentResult = (fps) => {
  const result = JSON.parse(readFileSync(
    new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url), 'utf8'));
  result.benchmark_report.fps = fps;
  return result;
};
const packageResult = (fps) => ({
  ...componentResult(fps),
  benchmark_type: 'model.orchestration',
  components: [{component_id: 'backbone'}, {component_id: 'decoder'}],
});
const detail = (result) => ({benchmark: result ? {result} : null});

test('scores a multi-component variant by its package result', () => {
  const measurements = wholeModelMeasurements([detail(componentResult(500)), detail(packageResult(120))]);
  assert.equal(measurements.throughput_fps.value, 120);
});

test('gives a multi-component variant without a package result no value', () => {
  assert.deepEqual(wholeModelMeasurements([detail(componentResult(500)), detail(componentResult(300))]), {});
});

test('scores a single-component variant by its own result', () => {
  assert.equal(wholeModelMeasurements([detail(componentResult(886))]).throughput_fps.value, 886);
  assert.deepEqual(wholeModelMeasurements([detail(null)]), {});
});
