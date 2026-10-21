import assert from 'node:assert/strict';
import {test} from 'node:test';
import {deploymentCatalog, allPages} from '../../artifact-browser/branch-builds.js';

const selection = (run, artifact = 'mpk') => ({artifacts: [{component_id: 'model', run_id: run, artifact_name: artifact}]});

for (const field of ['approved_deployments', 'branch_deployments']) {
  test(`${field} selects exact artifacts instead of arbitrary newer compilations`, () => {
    const model = deploymentCatalog({model_id: 'example', build_on_demand: true,
      builds: [{run_id: 'unselected', build_profile_id: 'int8', component_id: 'model'}],
      [field]: {int8: selection('selected', 'llima')},
    });
    assert.deepEqual(model.builds.map(build => build.run_id), ['selected']);
    assert.equal(model.builds[0].artifact_name, 'llima');
  });

  test(`${field} copies metadata only from the exact component/profile/run`, () => {
    const exact = {run_id: 'selected', build_profile_id: 'int8', component_id: 'model', toolchain_version: 'correct'};
    const mismatches = [ {...exact, run_id: 'other', toolchain_version: 'wrong'},
      {...exact, component_id: 'decoder', toolchain_version: 'wrong'},
      {...exact, build_profile_id: 'fp16', toolchain_version: 'wrong'} ];
    const model = deploymentCatalog({model_id: 'example', builds: [...mismatches, exact], [field]: {int8: selection('selected')}});
    assert.equal(model.builds.find(build => build.build_profile_id === 'int8').toolchain_version, 'correct');
    const unknown = deploymentCatalog({model_id: 'example', builds: mismatches, [field]: {int8: selection('selected')}});
    assert.equal(unknown.builds.find(build => build.build_profile_id === 'int8').toolchain_version, undefined);
  });
}

test('branch deployment overrides inherited approval without altering other profiles', () => {
  const model = deploymentCatalog({model_id: 'example', builds: [],
    approved_deployments: {int8: selection('approved'), fp16: selection('other-profile')},
    branch_deployments: {int8: selection('branch')},
  });
  assert.deepEqual(model.builds.map(build => build.run_id), ['branch', 'other-profile']);
});

test('package membership is explicit and never inferred from other component runs', () => {
  const model = deploymentCatalog({model_id: 'example',
    build_requirements: {int8: {packages: [{id: 'backbone-only', component_ids: ['backbone']},
      {id: 'complete', component_ids: ['backbone', 'head']}]}},
    approved_deployments: {int8: {artifacts: [
      {component_id: 'backbone', run_id: 'backbone-run'}, {component_id: 'head', run_id: 'head-run'},
      {package_id: 'backbone-only', run_id: 'partial-package'}, {package_id: 'complete', run_id: 'complete-package'},
    ]}},
  });
  assert.deepEqual(model.packages.map(pack => pack.component_run_ids), [
    {backbone: 'backbone-run'}, {backbone: 'backbone-run', head: 'head-run'},
  ]);
});

for (const next of [undefined, 'repeat']) {
  test(`pagination rejects ${next || 'missing'} cursor`, async () => {
    await assert.rejects(allPages(async () => ({models: [], has_more: true, next_cursor: next}), '/models', new URLSearchParams(), 'models'), /incomplete/);
  });
}
