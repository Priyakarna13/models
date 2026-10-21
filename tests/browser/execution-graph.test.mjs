import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';

const {adaptBenchmarkResult, executionGraph} = await import(new URL('../../artifact-browser/benchmark-adapters.js', import.meta.url));
const plan = {
  schema: 'sima.neat.execution_plan', schema_version: 1,
  model: {name: 'yolo26n', batch: 1},
  nodes: [
    {id: 'p0', kind: 'Input', runs_on: 'A65'},
    {id: 'p1', kind: 'Quant', runs_on: 'EV74', kernel: 'quantize'},
    {id: 'p2', kind: 'ModelFragment', runs_on: 'MLA'},
    {id: 'p3', kind: 'SimaBoxDecode', runs_on: 'A65', kernel: 'boxdecode', settings: {decode_type: 'yolo26', num_classes: 80}},
    {id: 'p4', kind: 'Output', runs_on: 'A65'},
  ],
  edges: [
    {from: 'p0', to: 'p1', tensors: [{name: 'input', shape: [1, 640, 640, 3], dtype: 'float32'}]},
    {from: 'p1', to: 'p2', tensors: [{name: 'input', shape: [1, 640, 640, 3], dtype: 'int8', quant: {scale: 256.1254, zero_point: -128}}]},
    {from: 'p2', to: 'p3', tensors: [
      {name: 'boxes', shape: [1, 80, 80, 4], dtype: 'int8', quant: {scale: 33.0187, zero_point: -117}},
      {name: 'mask', shape: [1, 32, 160, 160], dtype: 'int8', quant: {scale: 0.0039063, zero_point: 0}},
      {name: 'scores', shape: [1, 80, 80, 80], dtype: 'int8', quant: {scale: [0.5, 0.25], zero_point: [0, 1], axis: 3}},
    ]},
    {from: 'p3', to: 'p4'},
  ],
};

test('names each step by its node kind, the MLA step as the compiled model, and keeps the graph ends', () => {
  assert.deepEqual(executionGraph(plan).steps, [
    {name: 'Input', runsOn: null},
    {name: 'Quant', runsOn: 'EV74'},
    {name: 'Compiled Model', runsOn: 'MLA'},
    {name: 'SimaBoxDecode', runsOn: 'A65'},
    {name: 'Output', runsOn: null},
  ]);
});

test('lists each connection once above its tensors and skips a connection that reports none', () => {
  assert.deepEqual(executionGraph(plan).tensors, [
    {connection: 'Input → Quant', dtype: 'float32', shape: '1×640×640×3', scale: '—', zeroPoint: '—'},
    {connection: 'Quant → Compiled Model', dtype: 'int8', shape: '1×640×640×3', scale: '256.13', zeroPoint: '-128'},
    {connection: 'Compiled Model → SimaBoxDecode', dtype: 'int8', shape: '1×80×80×4', scale: '33.02', zeroPoint: '-117'},
    {connection: '', dtype: 'int8', shape: '1×32×160×160', scale: '0.00391', zeroPoint: '0'},
    {connection: '', dtype: 'int8', shape: '1×80×80×80', scale: 'per channel', zeroPoint: 'per channel'},
  ]);
});

test('reads the component plan and rejects unknown plan schemas', () => {
  const result = JSON.parse(readFileSync(new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url)));
  result.components.model.execution_plan = plan;
  assert.deepEqual(adaptBenchmarkResult(result).value.executionGraph, executionGraph(plan));
  assert.equal(executionGraph({...plan,schema_version:2}), null);
});

test('one model report carries each component diagnostics without summing scores', () => {
  const result = JSON.parse(readFileSync(new URL('../../artifact-browser/fixtures/performance-result.json', import.meta.url)));
  const layers = [{name:'MLA_0/conv2d_0',stage_index:0,total_execution_time_us:120.5}];
  result.components = Object.fromEntries(['backbone','decoder'].map((id)=>[id,{layer_stats:layers,execution_plan:plan}]));
  const benchmark = adaptBenchmarkResult(result).value;
  assert.equal(benchmark.packageId,null);
  assert.equal(benchmark.measurements['model.inference_rate'].value,result.metrics.find((metric)=>metric.name==='model.inference_rate').value);
  assert.deepEqual(benchmark.components.map(({id})=>id),['backbone','decoder']);
  assert.deepEqual(benchmark.components[0].executionGraph,executionGraph(plan));
  assert.equal(benchmark.components[1].layerAnalysis.layers.length,1);
});
