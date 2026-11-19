"""Kaggle-only same-time comparison: locked primary A4 against the C++ SEM-RRT* port, both 20 s.

Linux build -> registered CTest (plus the C++ port's exact-equivalence test) -> one lane on CPU 0, one
trial at a time -> gates on non-held-out inputs (smoke_fairtime, gate_fairtime) -> Python/C++ bit-identity
of SEM-RRT* on the gate maps -> record -> held-out collection in checkpointed chunks -> strict
validation of the primary method -> reports -> stop. Registrations: EXPERIMENTS (fairtime:
logs/a4_rpbit_fairtime_20261117, selfstop: logs/a4_rpbit_selfstop_20261117, putn: logs/a4_rpbit_putn_20261119);
a started trial is never retried.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from corridor_v4_execution import atomic_json, digest, file_hash, read, utc_now
from kaggle_pri_pathsearch.bootstrap import run_logged
from kaggle_pri_segment.transport import verify_catalog
from kaggle_a4_rpbit import bootstrap as base
import rp_bit_protocol as protocol

CONFIG = protocol.FAIRTIME_CONFIG
REGISTRATION = ROOT / 'logs/a4_rpbit_fairtime_20261117/registration.json'
HELDOUT = 'heldout_fairtime'
GATES = ('smoke_fairtime', 'gate_fairtime')
REQUIRED_CTEST = base.REQUIRED_CTEST + ('sem_native_cpp',)
IDENTITY_GATE = dict(name='sem_cpp_bit_identity', phase='gate_fairtime', seeds=(516001, 516002), work=7425, core=0)
LIBRARY = 'libsem_rrtstar_c.so'
# One entry per registered Kaggle experiment run by this pipeline; each has its own controller module.
EXPERIMENTS = dict(
    fairtime=dict(config=CONFIG, registration=REGISTRATION, heldout=HELDOUT, gates=GATES, identity_gate=IDENTITY_GATE,
                  record='a4_rpbit_fairtime_record_v1', module='kaggle_a4_rpbit.fairtime',
                  title=f'cùng thời gian 20 s với SEM-RRT* C++, Kaggle `{HELDOUT}`'),
    # Same pipeline; SEM-RRT* (C++) also stops on convergence with the A4 rule (logs/a4_rpbit_selfstop_20261117).
    selfstop=dict(config=protocol.SELFSTOP_CONFIG, registration=ROOT / 'logs/a4_rpbit_selfstop_20261117/registration.json',
                  heldout='heldout_selfstop', gates=('smoke_selfstop', 'gate_selfstop'),
                  identity_gate=dict(IDENTITY_GATE, phase='gate_selfstop'), record='a4_rpbit_selfstop_record_v1',
                  module='kaggle_a4_rpbit.selfstop',
                  title='cả hai tự dừng khi hội tụ (SEM-RRT* C++ theo luật của A4), Kaggle `heldout_selfstop`'),
    # Same pipeline with two PUTN arms beside the self-stop SEM-RRT* and A4, 10 s cap (logs/a4_rpbit_putn_20261119);
    # the CTest set adds the PUTN port tests and the C ABI test of the bridge that carries it.
    putn=dict(config=protocol.PUTN_KAGGLE_CONFIG, registration=ROOT / 'logs/a4_rpbit_putn_20261119/registration.json',
              heldout='heldout_putn', gates=('smoke_putn', 'gate_putn'), identity_gate=dict(IDENTITY_GATE, phase='gate_putn'),
              record='a4_rpbit_putn_record_v1', module='kaggle_a4_rpbit.putn',
              ctest=REQUIRED_CTEST + ('putn_paper', 'putn_static', 'terrain_capi'),
              title='A4, SEM-RRT* C++ và PUTN (hai nhánh), trần 10 s, Kaggle `heldout_putn`'))


def check_registration(config_path, config, experiment=EXPERIMENTS['fairtime']):
    """The delivered config must be exactly the registered one (file hash and every frozen phase hash)."""
    import run_rp_bit
    registered = read(experiment['registration'])
    phases = {name: digest(run_rp_bit.phase_config(config, name)) for name in config['phases']}
    if registered['config_sha256'] != file_hash(config_path) or registered['phase_config_sha256'] != phases:
        raise ValueError('configuration differs from the registration')
    if registered['heldout_phase'] != experiment['heldout'] or registered['gates'] != list(experiment['gates']):
        raise ValueError('phases differ from the registration')
    return registered


def run_record(campaign, config_path, config, identity, machine, equivalence, experiment=EXPERIMENTS['fairtime']):
    """Written once, before the first held-out trial; a later session must match it exactly."""
    import run_rp_bit
    lock = json.loads(base.LOCK.read_text(encoding='utf-8-sig'))  # written by PowerShell with a BOM
    HELDOUT, GATES, REGISTRATION = experiment['heldout'], experiment['gates'], experiment['registration']
    phase = config['phases'][HELDOUT]
    body = dict(record=experiment['record'], protocol_id=protocol.PROTOCOL_ID,
                registration=REGISTRATION.relative_to(ROOT).as_posix(), registration_sha256=file_hash(REGISTRATION),
                lock_manifest_sha256=file_hash(base.LOCK), locked_primary_method=lock['primary_method'],
                changed_after_lock=read(ROOT / '_bundle_catalog.json')['changed_after_lock'],
                config=config_path.relative_to(ROOT).as_posix(), config_sha256=file_hash(config_path),
                heldout_phase=HELDOUT, heldout_phase_config_sha256=digest(run_rp_bit.phase_config(config, HELDOUT)),
                heldout_trials=len(protocol.inventory(config, HELDOUT)), primary_method=phase['primary_method'],
                methods=config['methods'], execution=config['execution'], linux_identity=identity,
                hardware={key: machine[key] for key in ('affinity', 'registered_cores', 'registered_physical_cores', 'cpu_model')},
                gates={name: read(campaign / 'gates' / (name + '.json'))['config_sha256'] for name in GATES},
                sem_cpp_bit_identity={key: equivalence[key] for key in ('passed', 'runs', 'identical', 'work_iterations',
                                                                         'seeds', 'library_sha256')})
    path = campaign / 'heldout_deviation_record.json'
    if path.exists():
        recorded = read(path)
        session_only = ('created_utc', 'hardware')  # hardware of later sessions: setup/hardware_*.json
        if ({key: value for key, value in recorded.items() if key not in session_only}
                != {key: value for key, value in json.loads(json.dumps(body)).items() if key not in session_only}):
            raise ValueError('held-out record changed; refuse to mix held-out sessions')
        return recorded
    body['created_utc'] = utc_now()
    atomic_json(path, body)
    return body


def pipeline(args, campaign, output, python, env, started, work_epoch, hard_deadline):
    experiment = EXPERIMENTS[args.experiment]
    HELDOUT, GATES = experiment['heldout'], experiment['gates']
    config_path = experiment['config'].resolve()
    config = protocol.load_config(config_path)
    check_registration(config_path, config, experiment)
    build = ROOT / base.BUILD
    worker, bridge, library = build / 'terrain_benchmark', build / 'libterrain_compare_c.so', build / LIBRARY
    results = campaign / 'results'
    if not (campaign / 'setup/immutable_runtime.json').exists():
        if results.exists():
            raise ValueError('experiment evidence exists without runtime identity')
        base.status(campaign, 'BUILDING')
        base.setup.build(campaign, started, python, env)
    if not worker.is_file() or not bridge.is_file() or not library.is_file():
        raise ValueError('exact Linux build missing; no rebuild of an existing campaign')
    identity = dict(base.freeze_runtime(campaign, output, worker, bridge, python, env, hard_deadline),
                    sem_library_sha256=file_hash(library))
    base.status(campaign, 'CORRECTNESS')
    base.correctness(campaign, python, env, identity, hard_deadline, tests=experiment.get('ctest', REQUIRED_CTEST))
    cores = config['execution']['cores']
    machine = base.hardware(campaign, cores)
    if machine['registered_physical_cores'] != len(cores):
        # Timing is the subject of this run: every lane on its own physical core.
        raise RuntimeError(f'BLOCKED_RESOURCE: cores {cores} are not on distinct physical cores')
    validation = config['execution'].get('validation_cores', cores)
    if not set(validation) <= set(machine['affinity']):
        raise RuntimeError(f"BLOCKED_RESOURCE: validation cores {validation} are not all in the affinity {machine['affinity']}")

    def checkpoint():
        base.export_checkpoint(campaign, output, hard_deadline + 840., env)

    def run_phase(name, *extra):
        base.setup.resource_gate(campaign, started, python, env, name)
        manifest = results / name / 'manifest.json'
        before = len(read(manifest)['sessions']) if manifest.is_file() else 0
        argv = [python, '-B', 'tools/run_rp_bit.py', 'run', '--phase', name, '--output', results,
                '--worker', worker, '--bridge', bridge, '--config', config_path, *extra]
        if manifest.is_file():
            argv.append('--resume')
        run_logged(name, argv, campaign, hard_deadline, env, accepted=(0, 1, 3))
        sessions = read(manifest)['sessions'] if manifest.is_file() else []
        if len(sessions) != before + 1:
            raise RuntimeError(name + ': the runner exited without recording a session')
        return sessions[-1]

    for name in GATES:
        receipt = campaign / 'gates' / (name + '.json')
        if receipt.is_file() and read(receipt)['passed']:
            continue
        if time.time() >= work_epoch:
            base.status(campaign, 'PAUSED_TIME', phase=name)
            return
        base.status(campaign, 'RUNNING', phase=name)
        run_phase(name, '--full-validation')
        verdict = base.gate_verdict(results / name, config, name)
        atomic_json(receipt, verdict)
        if not verdict['passed']:
            base.status(campaign, 'GATE_FAILED', phase=name, problems=verdict['problems'][:20])
            checkpoint()
            return
        base.status(campaign, 'GATE_PASSED', phase=name)
        checkpoint()

    gate = experiment['identity_gate']
    receipt = campaign / 'gates' / (gate['name'] + '.json')
    if not receipt.is_file():
        if time.time() >= work_epoch:
            base.status(campaign, 'PAUSED_TIME', phase=gate['name'])
            return
        base.status(campaign, 'RUNNING', phase=gate['name'])
        run_logged(gate['name'], [python, '-B', 'tools/sem_cpp_equivalence.py', '--phase-dir', results / gate['phase'],
                   '--library', library, '--bridge', bridge, '--search-paths', ROOT / '.deps/install/lib',
                   '--seeds', *map(str, gate['seeds']), '--work', str(gate['work']), '--core', str(gate['core']),
                   '--output', receipt], campaign, hard_deadline, env, accepted=(0, 1))
    equivalence = read(receipt)
    if not equivalence['passed'] or equivalence['library_sha256'] != identity['sem_library_sha256']:
        base.status(campaign, 'GATE_FAILED', phase=gate['name'], identical=equivalence['identical'], runs=equivalence['runs'])
        checkpoint()
        return
    base.status(campaign, 'GATE_PASSED', phase=gate['name'])

    run_record(campaign, config_path, config, identity, machine, equivalence, experiment)
    if not args.allow_heldout:
        base.status(campaign, 'GATES_PASSED_HELDOUT_NOT_ALLOWED', phase=HELDOUT)
        checkpoint()
        return
    while True:
        if time.time() >= work_epoch:
            base.status(campaign, 'PAUSED_TIME', phase=HELDOUT)
            return
        base.status(campaign, 'RUNNING', phase=HELDOUT)
        stop = min(work_epoch, time.time() + args.chunk_seconds)
        session = run_phase(HELDOUT, '--allow-heldout', '--stop-dispatch-epoch', repr(stop))
        collection = session['collection_status']
        if collection == 'DEADLINE':
            base.status(campaign, 'HELDOUT_CHUNK_DONE', phase=HELDOUT, unstarted=session.get('unstarted'))
            checkpoint()
            continue
        if collection not in ('COMPLETE', 'NOTHING_PENDING'):
            base.status(campaign, 'HELDOUT_COLLECTION_FAILED', phase=HELDOUT, collection=collection)
            checkpoint()
            return
        break
    base.status(campaign, 'STRICT_VALIDATION', phase=HELDOUT)
    checkpoint()
    run_logged('strict_' + HELDOUT, [python, '-B', 'tools/run_rp_bit.py', 'validate', '--phase', HELDOUT,
               '--output', results, '--full-validation', '--methods', config['phases'][HELDOUT]['primary_method']],
               campaign, hard_deadline, env)
    base.status(campaign, 'REPORTING', phase=HELDOUT)
    reports = campaign / 'reports'
    for name, title in ((HELDOUT, experiment['title']),
                        (GATES[-1], f'cổng `{GATES[-1]}` (seed bản đồ 51001, không phải held-out)')):
        run_logged('report_' + name, [python, '-B', 'tools/analyze_rp_bit.py', '--input', results / name,
                   '--output', reports / name, '--title', title], campaign, hard_deadline, env)
    base.status(campaign, 'DONE', phase=HELDOUT, report=f'reports/{HELDOUT}/REPORT.md')
    checkpoint()


def main(argv=None, experiment='fairtime'):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(experiment=experiment)
    parser.add_argument('--started-epoch', type=float, required=True)
    parser.add_argument('--session-seconds', type=int, default=32400)
    parser.add_argument('--allow-heldout', action='store_true')
    parser.add_argument('--chunk-seconds', type=float, default=base.CHUNK_S)
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args(argv)
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or not Path('/kaggle/working').is_dir() or ROOT != (base.SCRATCH / 'repo').resolve()):
        raise RuntimeError('the build/experiment controller runs only in the Kaggle scratch repository')
    if not 3600 <= args.session_seconds <= 32400 or not 60. <= args.chunk_seconds <= 14400.:
        raise ValueError('the session budget must be 1-9 hours and a chunk 1 minute to 4 hours')
    output, campaign = base.OUTPUT.resolve(), (base.SCRATCH / 'campaign').resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign.mkdir(parents=True, exist_ok=True)
    verify_catalog(ROOT, ROOT / '_bundle_catalog.json')
    started = time.monotonic() - (time.time() - args.started_epoch)
    work_epoch = args.started_epoch + args.session_seconds - 2700
    hard_deadline = started + args.session_seconds - 900
    base.setup.BUILD_DIR, base.setup.VENV_DIR = base.BUILD, base.VENV
    base.supervision.HARD_SESSION_S, base.supervision.RESERVE_EXPORT_S = args.session_seconds, 2700
    env, failed = os.environ.copy(), False
    try:
        restored = (campaign / 'setup/immutable_runtime.json').exists()
        python, env = base.setup.isolated_runtime(campaign, started, restored=restored)
        if not args.prepared:
            forwarded = ['--started-epoch', repr(args.started_epoch), '--session-seconds', str(args.session_seconds),
                         '--chunk-seconds', repr(args.chunk_seconds)] + (['--allow-heldout'] if args.allow_heldout else [])
            run_logged('pinned_controller', [python, '-B', '-m', EXPERIMENTS[experiment]['module'], '--prepared', *forwarded],
                       output / 'controller_logs', started + args.session_seconds - 30., env, accepted=(0, 2))
        else:
            pipeline(args, campaign, output, python, env, started, work_epoch, hard_deadline)
    except BaseException as error:
        failed = True
        base.record_failure(campaign, error, traceback.format_exc(), timed_out=time.time() >= work_epoch)
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        # run_logged has joined/killed its descendants; a failed quiescence check blocks the export.
        if args.prepared or failed:
            base.export_checkpoint(campaign, output, started + args.session_seconds - 60., env)
    final = read(campaign / 'bootstrap_status.json')
    print(json.dumps(final, ensure_ascii=False))
    return 0 if final['status'] in base.FINAL_OK else 2


if __name__ == '__main__':
    raise SystemExit(main())
