"""Synthetic real-BIT* segment-mode replay; never experimental pilot data."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'tests')]
from corridor_v4_execution import THREAD_ENV
for key in THREAD_ENV:
    os.environ[key] = '1'
import numpy as np
from native_point_cloud import generate_cloud
from safe_length_field import build_field_file, file_hash
from terrain_io import write_map
from pri_pathsearch_validation import validate_trial
from native_paper_oracle import NativePaperOracle
from native_paper_v3_sem_evaluator import SharedSEMField


def run(worker: Path, directory: Path):
    terrain = directory / 'synthetic_segment.t25d'
    height = lambda x, y: .45 * np.exp(-((x - 3.) ** 2 + (y - 3.) ** 2) / 1.8)
    write_map(terrain, [[float(height(i * .2, j * .2)) for i in range(31)] for j in range(31)],
              spacing=(.2, .2), rectangles=[],
              metadata=dict(test_fixture=True, purpose='synthetic_pri_segment_worker_replay'))
    cloud_path = directory / 'synthetic_cloud.npy'
    np.save(cloud_path, generate_cloud(terrain), allow_pickle=False)
    field_path = directory / 'synthetic_field.txt'
    metadata = build_field_file(cloud_path, terrain, [0., 0., 6., 6.], field_path)
    info = dict(map_path=str(terrain), map_sha256=file_hash(terrain), cloud_path=str(cloud_path),
                cloud_sha256=file_hash(cloud_path), bounds=[0., 0., 6., 6.])
    base = dict(map_path=str(terrain), planner_id='bitstar_a3', variant_id='a3t_pri_segment_v2',
                planner_seed=771, start=[.8, 3.], goal=[5.2, 3.], planning_budget_s=120.,
                test_only_fixed_work=True,
                pathsearch_test_work=dict(anchor_iterations=1200, iterations_per_slice=200,
                                          optimizer_proposals_per_slice=12, cycles=2),
                samples_per_batch=100, batch_after_first_solution=200, rewire_factor=1.1, sample_attempts=100,
                state_cache=True, scalar_cost=True, sampler_query_cache=True, collision_index=True,
                exact_flat_certificate=False, objective_type='surface_length', pri_pathsearch=dict(mode='segment'),
                safe_length=dict(policy_id='a3t_safe_length_v1', field_path=str(field_path),
                                 field_sha256=metadata['field_sha256'], risk_cap=40., slope_cap_deg=35., atlas_sampling=True))
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join((str(ROOT / '.deps/install/lib'), 'C:/mingw64/bin', env.get('PATH', '')))

    def execute(name, request, expected=0):
        out = directory / name
        out.mkdir()
        (out / 'request.json').write_text(json.dumps(request), encoding='utf-8')
        command = [str(worker), str(out / 'request.json'), str(out)]
        process = subprocess.run(command, env=env, capture_output=True, text=True, timeout=150)
        (out / 'stdout.log').write_text(process.stdout, encoding='utf-8')
        (out / 'stderr.log').write_text(process.stderr, encoding='utf-8')
        (out / 'command.json').write_text(json.dumps(dict(argv=command, exit_code=process.returncode)), encoding='utf-8')
        assert process.returncode == expected, (name, process.returncode, process.stderr)
        if expected:
            return None
        return out, json.loads((out / 'worker.json').read_text())

    requests = []
    for seed in (771, 772, 773, 774):
        request = copy.deepcopy(base)
        request['planner_seed'] = seed
        requests.append(request)
    serial = [execute(f'serial_{i}', value) for i, value in enumerate(requests)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(lambda pair: execute(f'parallel_{pair[0]}', pair[1]), enumerate(requests)))
    oracle = NativePaperOracle(terrain)
    field = SharedSEMField(np.load(cloud_path, allow_pickle=False), info['bounds'])
    all_candidates = 0
    accepted_segments = 0
    accepted_segment_types = dict(segment_reduce=0, segment_contract=0)
    candidate_keys = ('snapshot_id', 'candidate_id', 'parent_ids', 'proposal_id', 'move_type', 'anchor_phase',
                      'pool_admitted', 'published_alphas', 'reason', 'feasible', 'cost_terrain_m', 'pri',
                      'risk_sum', 'risk_count', 'path_xy', 'segment_pri_cap', 'segment_length_cap',
                      'segment_first', 'segment_last')
    for request, (left, a), (right, b) in zip(requests, serial, parallel):
        assert a['planner_status'] == b['planner_status'] == 'EXACT_SOLUTION'
        assert a['backend_id'] == b['backend_id'] == 'a3t_pri_segment_v2'
        assert a['planner_instances'] == b['planner_instances'] == 1
        assert a['selected_snapshot_id'] == b['selected_snapshot_id']
        assert a['selectors'] == b['selectors'], 'fixed-work changed selectors/metrics'
        assert (left / 'path_xy.csv').read_bytes() == (right / 'path_xy.csv').read_bytes()
        assert a['counters'] == b['counters'], 'fixed-work changed BIT* counters'
        assert a['search_counters'] == b['search_counters'], 'fixed-work changed optimizer counters'
        assert a['search_counters']['segment_proposals'] > 0
        assert a['search_counters']['splice_proposals'] == 0
        assert a['search_counters']['proposals'] == 24
        assert len(a['segment_attempts']) == len(b['segment_attempts']) == a['search_counters']['segment_proposals']
        for x, y in zip(a['segment_attempts'], b['segment_attempts']):
            assert {k: v for k, v in x.items() if k not in ('began_s', 'finished_s')} == {
                k: v for k, v in y.items() if k not in ('began_s', 'finished_s')
            }, 'fixed-work changed segment operators or internal search counters'
        assert any(row['counters']['expansions'] > 0 for row in a['segment_attempts'])
        assert a['effective_policy']['motion_objective'] == 'surface_length'
        assert a['effective_policy']['nearest_neighbor_metric'] == 'euclidean_xy'
        assert a['effective_policy']['external_length_cap_in_sampler'] is False
        for key in ('motion_cost_calls', 'motion_heuristic_calls', 'cost_to_go_calls', 'sampler_allocations'):
            assert a['counters'][key] > 0, key
        aa = [json.loads(line) for line in (left / 'pathsearch_candidates.jsonl').read_text().splitlines()]
        bb = [json.loads(line) for line in (right / 'pathsearch_candidates.jsonl').read_text().splitlines()]
        assert len(aa) == len(bb)
        for x, y in zip(aa, bb):
            for key in candidate_keys:
                assert x.get(key) == y.get(key), (request['planner_seed'], x['snapshot_id'], key)
            assert x['score_completed_s'] <= request['planning_budget_s']
            if x['move_type'] == 'segment_contract':
                parent = aa[int(x['parent_ids'][0].split('_')[-1])]
                assert x['cost_terrain_m'] < parent['cost_terrain_m']
                assert x['pri'] <= parent['pri']
            elif x['move_type'] == 'segment_reduce':
                parent = aa[int(x['parent_ids'][0].split('_')[-1])]
                assert x['pri'] < parent['pri']
            accepted_segments += x['move_type'] in ('segment_reduce', 'segment_contract')
            if x['move_type'] in accepted_segment_types:
                accepted_segment_types[x['move_type']] += 1
        verdict = validate_trial(left, request, info, dict(status='COMPLETED', field_build_s=0.), oracle=oracle, field=field)
        assert verdict['success'] and not verdict['correctness_errors'], (request['planner_seed'], verdict['correctness_errors'])
        (left / 'independent_full_verdict.json').write_text(json.dumps(verdict), encoding='utf-8')
        all_candidates += len(aa)
    assert accepted_segments > 0, 'real worker never admitted a segment-operator candidate'
    assert all(accepted_segment_types.values()), 'one segment operator was never admitted'

    # A corrupted selected final must remain invalid even when earlier valid
    # snapshots exist. The independent verifier must not choose a rescue path.
    left, good = serial[0]
    damaged = directory / 'synthetic_damaged_selected'
    shutil.copytree(left, damaged)
    selected_id = good['selected_snapshot_id']
    archive = [json.loads(line) for line in (damaged / 'pathsearch_candidates.jsonl').read_text().splitlines()]
    selected = next(row for row in archive if row['snapshot_id'] == selected_id)
    assert selected['candidate_id'] > 0, 'no earlier valid path for no-rescue regression'
    selected['path_xy'][-1][0] -= .1  # Only the chosen path ceases to reach the goal.
    content = 'x,y\n' + ''.join(f'{x:.17g},{y:.17g}\n' for x, y in selected['path_xy'])
    (damaged / 'path_xy.csv').write_text(content, encoding='utf-8')
    (damaged / 'snapshots' / selected_id / 'path_xy.csv').write_text(content, encoding='utf-8')
    (damaged / 'pathsearch_candidates.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in archive), encoding='utf-8')
    failed = validate_trial(damaged, requests[0], info, dict(status='COMPLETED', field_build_s=0.), oracle=oracle, field=field)
    assert not failed['success']
    assert failed['selected_snapshot_id'] == good['selected_snapshot_id']
    assert all(not output['success'] for output in failed['selector_results'])
    assert selected_id + ':hard_safety_contract_failed' in failed['correctness_errors']
    assert any(event['safe'] and event['snapshot_id'] != selected_id for event in failed['events']), 'earlier paths did not stay valid'
    (damaged / 'independent_corrupted_final_verdict.json').write_text(json.dumps(failed), encoding='utf-8')

    invalid = copy.deepcopy(base)
    invalid['pri_pathsearch']['mode'] = 'segment_unsupported'
    execute('rejected_mode', invalid, expected=2)
    expired = copy.deepcopy(base)
    expired.pop('pathsearch_test_work')
    expired['planning_budget_s'] = 1e-12
    _, record = execute('expired_setup', expired)
    assert not record['path_available']
    assert all(row['selected_snapshot_id'] is None for row in record['selectors'])
    identity = copy.deepcopy(base)
    identity['goal'] = identity['start']
    out, record = execute('identity_query', identity)
    assert record['path_available'] and record['cost_terrain_m'] == 0.
    verdict = validate_trial(out, identity, info, dict(status='COMPLETED', field_build_s=0.), oracle=oracle, field=field)
    assert verdict['success'] and not verdict['correctness_errors'], verdict['correctness_errors']
    (out / 'independent_full_verdict.json').write_text(json.dumps(verdict), encoding='utf-8')
    print(json.dumps(dict(status='PASS', test_fixture=True, retained_fixture=str(directory),
                          real_fixed_work_trials=9, serial_parallel_workers=4,
                          serial_parallel_paths_operators_counters_identical=True,
                          independently_validated_candidate_paths=all_candidates,
                          accepted_segment_candidates=accepted_segments,
                          accepted_segment_operators=accepted_segment_types,
                          no_invalid_final_rescue=True, no_late_setup_solution=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True, type=Path)
    args = parser.parse_args()
    # Preserve logs and raw test outputs, including a failure, under fixtures.
    folder = Path(tempfile.mkdtemp(prefix='synthetic_pri_segment_worker_', dir=ROOT / 'tests/fixtures'))
    print(json.dumps(dict(test_fixture=True, artifact_directory=str(folder))), flush=True)
    run(args.worker.resolve(), folder)
