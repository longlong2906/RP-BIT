"""Kaggle-only A4 / RP-BIT* held-out controller.

Linux build -> registered CTest -> gates on non-held-out inputs (smoke_final, rehearsal, kaggle_gate)
-> deviation record -> held-out collection in checkpointed chunks -> strict validation of the primary
method -> reports -> stop. A started trial is never retried; exports happen only after the supervised
children have stopped.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from corridor_v4_execution import atomic_json, digest, file_hash, read, utc_now
import kaggle_native_paper_v4 as supervision
import kaggle_sem_exposure_pilot as setup
from kaggle_pri_pathsearch.bootstrap import run_logged
from kaggle_pri_segment.transport import verify_catalog
import rp_bit_protocol as protocol

SCRATCH = Path('/kaggle/temp/a4_rpbit')
OUTPUT = Path('/kaggle/working/a4_rpbit')
BUILD = 'build-a4-rpbit'
VENV = '.venv-a4-rpbit'
RUNTIME_ROOTS = [VENV, '.managed-python', '.deps/install/lib', BUILD]
CONFIG = protocol.KAGGLE_CONFIG
LOCK = ROOT / 'logs/a4_rpbit_20261116/lock_manifest.json'
HELDOUT = 'heldout_kaggle'
GATES = ('smoke_final', 'rehearsal', 'kaggle_gate')
CHUNK_S = 3600.
FINAL_OK = ('DONE', 'PAUSED_TIME', 'GATES_PASSED_HELDOUT_NOT_ALLOWED')
REQUIRED_CTEST = tuple(
    ['safe_length_' + x for x in ('unit', 'field', 'runner', 'oracle', 'ompl')]
    + ['safe_length_v2_' + x for x in ('calibration', 'protocol', 'runner', 'validation', 'analysis', 'oracle_roundoff', 'worker')]
    + ['sem_exposure_' + x for x in ('ompl', 'protocol', 'runner', 'analysis', 'validation', 'worker')]
    + ['pri_pathsearch_' + x for x in ('core', 'worker', 'protocol', 'runner', 'validation', 'analysis')]
    + ['pri_segment_' + x for x in ('core', 'worker', 'protocol')]
    + ['rp_bit_' + x for x in ('core', 'worker', 'tooling')])
DEVIATIONS = [
    dict(item='platform', locked='Windows 11 MinGW build (binary hashes in the lock)',
         kaggle='Kaggle Linux x86_64 build of the same C++ sources (hashes under linux_identity)'),
    dict(item='execution.cores', locked=[2], kaggle=[0, 1, 2, 3], note='four lanes, one thread per trial'),
    dict(item='SEM-RRT* budget', locked='20 s wall clock',
         kaggle='7425 iterations (mean of the 75 dev runs: 556875/75) with a 180 s wall-clock cap',
         note='keeps the baseline as strong as on dev on slower shared vCPUs; every other method keeps 20 s'),
    dict(item='phase', locked='heldout', kaggle='heldout_kaggle',
         note='same maps, map/planner seeds, methods, primary method and criteria; 1500 trials'),
    dict(item='gates', kaggle='smoke_final, rehearsal and kaggle_gate (map seed 51001, not held-out) pass first'),
    dict(item='tooling', kaggle='rp_bit_protocol.py (fixed SEM work), run_rp_bit.py (soft dispatch deadline), '
         'analyze_rp_bit.py (SEM work summary) and their tests changed after the lock; see locked_sources'),
]
UNCHANGED = ('lambda 33 and M = 20 of the primary method, every other A4 parameter, the six methods, '
             'map seeds 61001-61010, planner seeds 616001-616005, analysis settings and the criteria of plan 5.3')


def status(campaign, value, **details):
    body = dict(protocol_id=protocol.PROTOCOL_ID, controller='kaggle_a4_rpbit', status=value, utc=utc_now(), **details)
    atomic_json(campaign / 'bootstrap_status.json', body)
    atomic_json(campaign / 'status_history' / (str(time.time_ns()) + '.json'), body)
    print('STATUS', value, details.get('phase', ''), flush=True)
    return body


def record_failure(campaign, error, traceback_text, *, timed_out=False):
    """Keep the inner failure when the outer supervisor reports its exit."""
    path = campaign / 'bootstrap_status.json'
    previous = read(path) if path.exists() else {}
    cause = previous.get('root_failure') or dict(error=repr(error), traceback=traceback_text,
                                                 stage=previous.get('phase', previous.get('status', 'BOOTSTRAP')))
    return status(campaign, 'PAUSED_TIME' if timed_out else 'FAILED', phase=previous.get('phase'), error=cause['error'],
                  root_failure=cause, supervisor_error=repr(error), supervisor_traceback=traceback_text)


def export_checkpoint(campaign, output, deadline, env):
    """Full campaign (audit/resume) and a light archive without trial raw data and maps (analysis)."""
    if (supervision.SUPERVISION_ERROR or (campaign / 'export_blocked.json').exists()
            or (output / 'export_logs/export_blocked.json').exists()
            or (output / 'controller_logs/export_blocked.json').exists()):
        raise RuntimeError('BLOCKED_EXPORT: descendant quiescence unproven')
    output.mkdir(parents=True, exist_ok=True)
    size = sum(p.stat().st_size for p in campaign.rglob('*') if p.is_file())
    if shutil.disk_usage(output).free < size + 1024**3:
        raise RuntimeError('BLOCKED_EXPORT_DISK: insufficient free space for an atomic checkpoint')
    light = ['--exclude=campaign/results/*/trials', '--exclude=campaign/results/*/inputs']
    for name, extra in (('campaign', []), ('results_light', light)):
        temporary = output / f'{name}.partial.{time.time_ns()}.tar.gz'
        run_logged('export_' + name, ['tar', '--use-compress-program=gzip -1', *extra, '-cf', temporary,
                                      '-C', campaign.parent, campaign.name], output / 'export_logs', deadline, env)
        checksum = file_hash(temporary)
        temporary.replace(output / f'{name}.tar.gz')
        (output / f'{name}.sha256').write_text(f'{checksum}  {name}.tar.gz\n', encoding='utf-8')
    for name in ('bootstrap_status.json', 'heldout_deviation_record.json', 'linux_runtime_sha256.json'):
        if (campaign / name).is_file():
            shutil.copy2(campaign / name, output / name)
    for folder in ('gates', 'reports'):
        if (campaign / folder).is_dir():
            shutil.copytree(campaign / folder, output / folder, dirs_exist_ok=True)


def freeze_runtime(campaign, output, worker, bridge, python, env, deadline):
    import run_pri_pathsearch as established
    runtime = established._runtime_identity(worker, bridge, python)
    identity = dict(binary_sha256=file_hash(worker), bridge_sha256=file_hash(bridge), runtime_sha256=digest(runtime),
                    sources_sha256=digest(read(ROOT / '_bundle_catalog.json')['files']))
    path = campaign / 'setup/immutable_runtime.json'
    body = dict(identity, runtime=runtime, original_runtime_root=str(ROOT))
    if path.exists() and read(path) != body:
        raise ValueError('runtime/source changed; never rebuild or mix this campaign')
    atomic_json(path, body)
    archive, receipt = output / 'linux_runtime.tar', campaign / 'linux_runtime_sha256.json'
    if receipt.exists():
        recorded = read(receipt)
        if (not archive.is_file() or file_hash(archive) != recorded['sha256']
                or any(recorded.get(k) != v for k, v in identity.items())):
            raise ValueError('exact retained Linux runtime is missing or changed')
    else:
        temporary = output / ('linux_runtime.partial.' + str(time.time_ns()))
        run_logged('retain_runtime', ['tar', '-cf', temporary, '-C', ROOT, *RUNTIME_ROOTS], campaign, deadline, env)
        checksum = file_hash(temporary)
        temporary.replace(archive)
        atomic_json(receipt, dict(identity, sha256=checksum, roots=RUNTIME_ROOTS, original_runtime_root=str(ROOT)))
    return identity


def correctness(campaign, python, env, identity, deadline, tests=REQUIRED_CTEST):
    receipt = campaign / 'ctest_receipt.json'
    if receipt.exists():
        proof = read(receipt)
        if (any(proof.get(k) != v for k, v in identity.items()) or proof.get('status') != 'PASS'
                or proof.get('tests') != list(tests)):
            raise ValueError('restored correctness receipt identity differs')
        return
    if (campaign / 'results').exists():
        raise ValueError('experiment evidence without a correctness receipt; refuse to rerun')
    common = ['ctest', '--test-dir', ROOT / BUILD, '-R', '^(' + '|'.join(tests) + ')$', '-j', '1']
    listing = json.loads(run_logged('ctest_inventory', [*common, '--show-only=json-v1'], campaign, deadline, env))
    if sorted(test['name'] for test in listing['tests']) != sorted(tests):
        raise ValueError('registered Linux CTest set differs')
    # Verbose output exposes skipped Python test cases, which must not count as a pass.
    result = run_logged('correctness', [*common, '--verbose', '--no-tests=error'], campaign, deadline, env)
    if re.search(r'OK\s*\(skipped=|\*\*\*Skipped|\*\*\*Not Run', result):
        raise ValueError('required correctness tests were skipped')
    run_logged('bundle_tests', [python, '-B', '-m', 'unittest', '-v', 'kaggle_a4_rpbit.test_bundle'], campaign, deadline, env)
    log = ROOT / BUILD / 'Testing/Temporary/LastTest.log'
    if log.is_file():
        shutil.copy2(log, campaign / 'setup/ctest_LastTest.log')
    atomic_json(receipt, dict(identity, status='PASS', exit_code=0, platform='Linux', tests=list(tests), utc=utc_now()))


def hardware(campaign, cores):
    """Record the CPU topology (hyper-thread siblings matter for timing) and require the registered cores."""
    affinity = sorted(os.sched_getaffinity(0))
    topology = {}
    for cpu in affinity:
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        topology[str(cpu)] = {name: (base / name).read_text().strip()
                              for name in ('physical_package_id', 'core_id', 'thread_siblings_list') if (base / name).is_file()}
    cpuinfo = Path('/proc/cpuinfo').read_text(errors='replace') if Path('/proc/cpuinfo').is_file() else ''
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo.splitlines() if line.startswith('model name')), None)
    lscpu = subprocess.run(['lscpu'], capture_output=True, text=True, timeout=60).stdout if shutil.which('lscpu') else None
    physical = {(topology.get(str(c), {}).get('physical_package_id'), topology.get(str(c), {}).get('core_id')) for c in cores}
    body = dict(affinity=affinity, registered_cores=cores, cpu_model=model,
                registered_physical_cores=len(physical) if (None, None) not in physical else None,
                topology=topology, lscpu=lscpu, utc=utc_now())
    # One receipt per session: a resumed session may land on other hardware and is reported separately.
    atomic_json(campaign / 'setup' / f'hardware_{time.time_ns()}.json', body)
    atomic_json(campaign / 'setup/hardware.json', body)
    if not set(cores) <= set(affinity):
        raise RuntimeError(f'BLOCKED_RESOURCE: registered cores {cores} are not all in the affinity {affinity}')
    return body


def gate_verdict(directory, config, name):
    """A gate passes with complete strict verdicts, no correctness error, every A4 and A5 trial successful and,
    under a fixed SEM budget, every SEM run having done the whole iteration budget."""
    manifest, summary = read(directory / 'manifest.json'), read(directory / 'summary_full.json')
    work = config['phases'][name].get('sem_work_iterations')
    kind = lambda row: config['methods'][row['method']]['kind']
    rows, problems = summary['rows'], []
    collection = manifest['sessions'][-1]['collection_status'] if manifest['sessions'] else None
    if collection not in ('COMPLETE', 'NOTHING_PENDING'):
        problems.append(f'collection {collection}')
    if not summary['planned'] == summary['validated'] == len(manifest['entries']) == len(rows):
        problems.append('missing verdicts')
    for row in rows:
        if row['correctness_errors']:
            problems.append(f"{row['trial_id']}: correctness {row['correctness_errors']}")
        if kind(row) == 'rp_bit' and not row['success']:
            problems.append(f"{row['trial_id']}: A4 trial not successful")
        if kind(row) == protocol.A5 and not row['success']:
            problems.append(f"{row['trial_id']}: A5 trial not successful")
        if kind(row) == 'sem_native' and work and row.get('iterations') != work:
            problems.append(f"{row['trial_id']}: SEM did {row.get('iterations')} of {work} iterations")
    times = {k: [round(r['solve_time_s'], 3) for r in rows if kind(r) == k and isinstance(r.get('solve_time_s'), (int, float))]
             for k in protocol.KINDS}
    return dict(phase=name, passed=not problems, problems=problems, planned=summary['planned'],
                validated=summary['validated'], success=summary['success'],
                correctness_failures=summary['correctness_failures'], config_sha256=manifest['config_sha256'],
                binary_sha256=manifest['binary_sha256'], sem_work_iterations=work,
                sem_iterations=[r.get('iterations') for r in rows if kind(r) == 'sem_native'],
                solve_time_s=times, utc=utc_now())


def deviation_record(campaign, config_path, config, heldout, identity, machine):
    """Written once, before the first held-out trial; a later session must match it exactly."""
    import run_rp_bit
    lock = json.loads(LOCK.read_text(encoding='utf-8-sig'))  # written by PowerShell with a BOM
    sources = []
    for name, checksum in lock['sha256'].items():
        current = file_hash(ROOT / name) if (ROOT / name).is_file() else None
        sources.append(dict(path=name, locked=checksum, kaggle=current, unchanged=current == checksum))
    phase = config['phases'][heldout]
    body = dict(record='a4_rpbit_heldout_deviation_v1', protocol_id=protocol.PROTOCOL_ID,
                lock_manifest_sha256=file_hash(LOCK),
                locked={key: lock[key] for key in ('locked_utc', 'primary_method', 'heldout_phase',
                                                   'heldout_phase_config_sha256', 'heldout_map_seeds', 'heldout_planner_seeds')},
                config=config_path.relative_to(ROOT).as_posix() if config_path.is_relative_to(ROOT) else str(config_path),
                config_sha256=file_hash(config_path), heldout_phase=heldout,
                heldout_phase_config_sha256=digest(run_rp_bit.phase_config(config, heldout)),
                heldout_trials=len(protocol.inventory(config, heldout)), primary_method=phase.get('primary_method'),
                sem_work_iterations=phase.get('sem_work_iterations'), sem_budget_cap_s=phase.get('sem_budget_cap_s'),
                execution=config['execution'], linux_identity=identity,
                hardware={key: machine[key] for key in ('affinity', 'registered_cores', 'registered_physical_cores', 'cpu_model')},
                gates={name: read(campaign / 'gates' / (name + '.json'))['config_sha256'] for name in GATES},
                locked_sources=sources, deviations=DEVIATIONS, unchanged=UNCHANGED)
    path = campaign / 'heldout_deviation_record.json'
    if path.exists():
        recorded = read(path)
        session_only = ('created_utc', 'hardware')  # hardware of later sessions: setup/hardware_*.json
        if ({key: value for key, value in recorded.items() if key not in session_only}
                != {key: value for key, value in json.loads(json.dumps(body)).items() if key not in session_only}):
            raise ValueError('held-out deviation record changed; refuse to mix held-out sessions')
        return recorded
    body['created_utc'] = utc_now()
    atomic_json(path, body)
    return body


def pipeline(args, campaign, output, python, env, started, work_epoch, hard_deadline):
    config_path = Path(args.config).resolve()
    config = protocol.load_config(config_path)
    heldout = args.heldout_phase
    if not heldout.startswith('heldout') or heldout not in config['phases']:
        raise ValueError('unregistered held-out phase: ' + heldout)
    worker, bridge = ROOT / BUILD / 'terrain_benchmark', ROOT / BUILD / 'libterrain_compare_c.so'
    results = campaign / 'results'
    if not (campaign / 'setup/immutable_runtime.json').exists():
        if results.exists():
            raise ValueError('experiment evidence exists without runtime identity')
        status(campaign, 'BUILDING')
        setup.build(campaign, started, python, env)
    if not worker.is_file() or not bridge.is_file():
        raise ValueError('exact Linux build missing; no rebuild of an existing campaign')
    identity = freeze_runtime(campaign, output, worker, bridge, python, env, hard_deadline)
    status(campaign, 'CORRECTNESS')
    correctness(campaign, python, env, identity, hard_deadline)
    machine = hardware(campaign, config['execution']['cores'])

    def checkpoint():
        export_checkpoint(campaign, output, hard_deadline + 840., env)

    def run_phase(name, *extra):
        setup.resource_gate(campaign, started, python, env, name)
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
            status(campaign, 'PAUSED_TIME', phase=name)
            return
        status(campaign, 'RUNNING', phase=name)
        run_phase(name, '--full-validation')
        verdict = gate_verdict(results / name, config, name)
        atomic_json(receipt, verdict)
        if not verdict['passed']:
            status(campaign, 'GATE_FAILED', phase=name, problems=verdict['problems'][:20])
            checkpoint()
            return
        status(campaign, 'GATE_PASSED', phase=name)
        checkpoint()

    deviation_record(campaign, config_path, config, heldout, identity, machine)
    if not args.allow_heldout:
        status(campaign, 'GATES_PASSED_HELDOUT_NOT_ALLOWED', phase=heldout)
        checkpoint()
        return
    while True:
        if time.time() >= work_epoch:
            status(campaign, 'PAUSED_TIME', phase=heldout)
            return
        status(campaign, 'RUNNING', phase=heldout)
        stop = min(work_epoch, time.time() + args.chunk_seconds)
        session = run_phase(heldout, '--allow-heldout', '--stop-dispatch-epoch', repr(stop))
        collection = session['collection_status']
        if collection == 'DEADLINE':
            status(campaign, 'HELDOUT_CHUNK_DONE', phase=heldout, unstarted=session.get('unstarted'))
            checkpoint()
            continue
        if collection not in ('COMPLETE', 'NOTHING_PENDING'):
            status(campaign, 'HELDOUT_COLLECTION_FAILED', phase=heldout, collection=collection)
            checkpoint()
            return
        break
    status(campaign, 'STRICT_VALIDATION', phase=heldout)
    checkpoint()
    run_logged('strict_' + heldout, [python, '-B', 'tools/run_rp_bit.py', 'validate', '--phase', heldout,
               '--output', results, '--full-validation', '--methods', config['phases'][heldout]['primary_method']],
               campaign, hard_deadline, env)
    status(campaign, 'REPORTING', phase=heldout)
    reports = campaign / 'reports'
    for name, title in ((heldout, f'held-out Kaggle `{heldout}`'),
                        ('kaggle_gate', 'cổng Kaggle `kaggle_gate` (seed bản đồ 51001, không phải held-out)')):
        run_logged('report_' + name, [python, '-B', 'tools/analyze_rp_bit.py', '--input', results / name,
                   '--output', reports / name, '--title', title], campaign, hard_deadline, env)
    status(campaign, 'DONE', phase=heldout, report=f'reports/{heldout}/REPORT.md')
    checkpoint()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--started-epoch', type=float, required=True)
    parser.add_argument('--session-seconds', type=int, default=32400)
    parser.add_argument('--allow-heldout', action='store_true')
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--heldout-phase', default=HELDOUT)
    parser.add_argument('--chunk-seconds', type=float, default=CHUNK_S)
    parser.add_argument('--prepared', action='store_true')
    args = parser.parse_args(argv)
    if (sys.platform != 'linux' or platform.machine() != 'x86_64'
            or not Path('/kaggle/working').is_dir() or ROOT != (SCRATCH / 'repo').resolve()):
        raise RuntimeError('the build/experiment controller runs only in the Kaggle scratch repository')
    if not 3600 <= args.session_seconds <= 32400 or not 60. <= args.chunk_seconds <= 14400.:
        raise ValueError('the session budget must be 1-9 hours and a chunk 1 minute to 4 hours')
    output, campaign = OUTPUT.resolve(), (SCRATCH / 'campaign').resolve()
    output.mkdir(parents=True, exist_ok=True)
    campaign.mkdir(parents=True, exist_ok=True)
    verify_catalog(ROOT, ROOT / '_bundle_catalog.json')
    started = time.monotonic() - (time.time() - args.started_epoch)
    work_epoch = args.started_epoch + args.session_seconds - 2700
    hard_deadline = started + args.session_seconds - 900
    setup.BUILD_DIR, setup.VENV_DIR = BUILD, VENV
    supervision.HARD_SESSION_S, supervision.RESERVE_EXPORT_S = args.session_seconds, 2700
    env, failed = os.environ.copy(), False
    try:
        restored = (campaign / 'setup/immutable_runtime.json').exists()
        python, env = setup.isolated_runtime(campaign, started, restored=restored)
        if not args.prepared:
            forwarded = ['--started-epoch', repr(args.started_epoch), '--session-seconds', str(args.session_seconds),
                         '--config', args.config, '--heldout-phase', args.heldout_phase,
                         '--chunk-seconds', repr(args.chunk_seconds)] + (['--allow-heldout'] if args.allow_heldout else [])
            run_logged('pinned_controller', [python, '-B', '-m', 'kaggle_a4_rpbit.bootstrap', '--prepared', *forwarded],
                       output / 'controller_logs', started + args.session_seconds - 30., env, accepted=(0, 2))
        else:
            pipeline(args, campaign, output, python, env, started, work_epoch, hard_deadline)
    except BaseException as error:
        failed = True
        record_failure(campaign, error, traceback.format_exc(), timed_out=time.time() >= work_epoch)
        print(traceback.format_exc(), file=sys.stderr)
    finally:
        # run_logged has joined/killed its descendants; a failed quiescence check blocks the export.
        if args.prepared or failed:
            export_checkpoint(campaign, output, started + args.session_seconds - 60., env)
    final = read(campaign / 'bootstrap_status.json')
    print(json.dumps(final, ensure_ascii=False))
    return 0 if final['status'] in FINAL_OK else 2


if __name__ == '__main__':
    raise SystemExit(main())
