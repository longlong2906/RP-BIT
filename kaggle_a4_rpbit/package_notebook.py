"""Build the self-contained Kaggle delivery of the A4 held-out run; this module never executes planners."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
from corridor_v4_execution import digest, file_hash, read
import rp_bit_protocol as protocol

DELIVERY = 'a4_rpbit_heldout_kaggle_r1'
PAYLOAD = 'a4_rpbit_heldout_source'
NOTEBOOK = 'a4_rpbit_heldout_kaggle.ipynb'
LOCK = ROOT / 'logs/a4_rpbit_20261116/lock_manifest.json'
# Tooling changed after the lock (listed in the deviation record); every other locked source must be unchanged.
CHANGED_AFTER_LOCK = ('tools/rp_bit_protocol.py', 'tools/run_rp_bit.py', 'tools/analyze_rp_bit.py',
                      'tests/test_rp_bit_tooling.py')
WINDOWS_BINARIES = ('build-a4-rpbit/terrain_benchmark.exe', 'build-a4-rpbit/libterrain_compare_c.dll')
UV_WHEEL_SHA256 = '8cf6bc2482d1293cc630f66b862b494c09acda9b7faff7307ef52667a2b3ad49'
FAIRTIME_REGISTRATION = 'logs/a4_rpbit_fairtime_20261117/registration.json'
SELFSTOP_REGISTRATION = 'logs/a4_rpbit_selfstop_20261117/registration.json'


def code_cell(source):
    return dict(cell_type='code', execution_count=None, metadata={}, outputs=[],
                source=source.splitlines(keepends=True))


SETUP = r'''import os
import shutil
import signal
import subprocess
import sys
import time

STARTED_EPOCH = time.time()
SESSION_SECONDS = 9 * 3600  # Controller ceiling, not a Kaggle quota guarantee; checkpoints are exported before it.
ALLOW_HELDOUT = True  # Held-out (map seeds HELDOUT_SEEDS) starts only after every gate has passed.
PREVIOUS_OUTPUT = None  # Resume: '/kaggle/input/<...>/a4_rpbit' holding campaign.tar.gz + linux_runtime.tar.
INPUT = Path('/kaggle/input')
WORK = Path('/kaggle/working/a4_rpbit')
SCRATCH = Path('/kaggle/temp/a4_rpbit')
REPO = SCRATCH / 'repo'
CAMPAIGN = SCRATCH / 'campaign'
RUNTIME_ROOTS = ['.venv-a4-rpbit', '.managed-python', '.deps/install/lib', 'build-a4-rpbit']
SCRATCH.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
if not sys.platform.startswith('linux') or not INPUT.is_dir():
    raise RuntimeError('Run this notebook only on Kaggle: CPU, Accelerator None, Internet on.')
if sys.version_info < (3, 11) or not hasattr(tarfile, 'data_filter'):
    raise RuntimeError('The notebook bootstrap needs Python 3.11+ with safe tar extraction.')

if not REPO.exists():
    matches = list(INPUT.rglob(PAYLOAD_NAME))
    if len(matches) != 1:
        raise ValueError('Attach exactly one input file named ' + PAYLOAD_NAME)
    decoded = SCRATCH / ('source.' + str(time.time_ns()) + '.zip')
    decode_base64(matches[0], decoded, PAYLOAD_SHA256)
    with zipfile.ZipFile(decoded) as archive:
        required = sum(item.file_size for item in archive.infolist()) + 20 * 1024**3
    if shutil.disk_usage(SCRATCH).free < required:
        raise RuntimeError('Insufficient free space to extract, build and run the campaign.')
    safe_extract_zip(decoded, REPO, PAYLOAD_SHA256)
    (REPO / '.transport_sha256').write_text(PAYLOAD_SHA256 + '\n')
    decoded.unlink()
elif (REPO / '.transport_sha256').read_text().strip() != PAYLOAD_SHA256:
    raise ValueError('The existing repository belongs to another delivery; use a fresh session.')
verify_catalog(REPO, REPO / '_bundle_catalog.json')

previous = None
if PREVIOUS_OUTPUT is not None:
    previous = Path(PREVIOUS_OUTPUT).resolve()
    if not previous.is_relative_to(INPUT.resolve()):
        raise ValueError('PREVIOUS_OUTPUT must be an attached input folder.')
    if not CAMPAIGN.exists():
        expected = (previous / 'campaign.sha256').read_text().split()[0]
        safe_extract_tar(previous / 'campaign.tar.gz', SCRATCH, expected, allowed_prefixes=['campaign'])

# Never rebuild a frozen campaign: restore its exact binaries, interpreter and libraries.
runtime_receipt = CAMPAIGN / 'linux_runtime_sha256.json'
if (CAMPAIGN / 'setup/immutable_runtime.json').exists():
    if not runtime_receipt.exists():
        raise ValueError('The frozen campaign has no retained-runtime receipt; refusing a rebuild.')
    binding = json.loads(runtime_receipt.read_text())
    if binding['roots'] != RUNTIME_ROOTS:
        raise ValueError('Unregistered retained runtime roots.')
    retained = WORK / 'linux_runtime.tar'
    if not retained.exists():
        if previous is None or sha256(previous / 'linux_runtime.tar') != binding['sha256']:
            raise ValueError('Attach the exact prior linux_runtime.tar next to campaign.tar.gz.')
        shutil.copy2(previous / 'linux_runtime.tar', retained)
    if sha256(retained) != binding['sha256']:
        raise ValueError('The retained Linux runtime changed.')
    binaries = [REPO / 'build-a4-rpbit/terrain_benchmark', REPO / 'build-a4-rpbit/libterrain_compare_c.so',
                REPO / '.venv-a4-rpbit/bin/python']
    if not all(path.is_file() for path in binaries):
        if any((REPO / root).exists() for root in RUNTIME_ROOTS):
            raise ValueError('Partial runtime restore detected; use a fresh Kaggle session.')
        safe_extract_tar(retained, REPO, binding['sha256'], allow_runtime_links=True, allowed_prefixes=RUNTIME_ROOTS,
                         original_runtime_root=binding['original_runtime_root'])

print('Verified source' + (' and checkpoint' if previous else '') +
      '. Linux build -> CTest -> gates -> held-out (only after every gate) -> strict validation -> report.')
'''

RUN = r'''env = os.environ.copy()
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS', 'OMP_THREAD_LIMIT'):
    env[key] = '1'
env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
argv = [sys.executable, '-B', '-m', 'CONTROLLER_MODULE',
        '--started-epoch', repr(STARTED_EPOCH), '--session-seconds', str(SESSION_SECONDS)]
if ALLOW_HELDOUT:
    argv.append('--allow-heldout')
process = subprocess.Popen(argv, cwd=REPO, env=env, start_new_session=True)
try:
    code = process.wait()
except BaseException:
    # Give the supervised controller time to stop its children and export a checkpoint.
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        (WORK / 'NOTEBOOK_INTERRUPTED.txt').write_text(
            'Controller forced down; keep the last exported checkpoint. Do not archive scratch by hand.\n')
    raise
finally:
    # The controller owns every export; this cell never archives a campaign that may still be written.
    print('Preserved output:', WORK)
    if (WORK / 'bootstrap_status.json').is_file():
        print((WORK / 'bootstrap_status.json').read_text())
    report = WORK / 'reports/HELDOUT_PHASE/REPORT.md'
    if report.is_file():
        print(report.read_text())
if code != 0:
    raise RuntimeError('Stopped before DONE: keep all output and read bootstrap_status.json.')
'''

INTRO = '''# A4 / RP-BIT* — held-out trên Kaggle (4 vCPU)

Notebook build lại A4 trên Linux, chạy các cổng kiểm tra trên đầu vào **không** thuộc held-out, rồi mới chạy
held-out đã đăng ký: 5 họ × seed bản đồ 61001–61010 × planner seed 616001–616005 × 6 phương pháp
= **1 500 lượt**, 4 làn × 1 thread, không dùng GPU.

Khác bản khóa ngày 16/11/2026, ghi vào `heldout_deviation_record.json` trước lượt held-out đầu tiên: build
Linux; 4 làn thay vì 1; SEM-RRT* chạy cố định **7 425 vòng lặp** (trung bình dev, trần 180 s) thay vì 20 s;
tên pha `heldout_kaggle`. λ = 33, M = 20, các phương pháp, seed và tiêu chí giữ nguyên.

Thứ tự: build OMPL 1.7.0 và dự án → 30 nhóm CTest → `smoke_final` → `rehearsal` → `kaggle_gate`
(seed 51001) → held-out theo từng đoạn khoảng 1 giờ, mỗi đoạn có checkpoint → validate nghiêm ngặt cấu hình
chính → báo cáo. Một cổng không đạt thì notebook dừng trước held-out.

1. Upload file `.b64` thành một Kaggle Dataset riêng tư rồi **Add Input** vào notebook này.
2. Settings: **Accelerator = None**, **Internet = On**. Chạy **Save Version → Save & Run All (Commit)**.
3. Kết quả trong `/kaggle/working/a4_rpbit/`: `bootstrap_status.json`, `reports/heldout_kaggle/REPORT.md`,
   `results_light.tar.gz` (nhỏ, đủ để phân tích), `campaign.tar.gz` và `linux_runtime.tar` (lớn, để kiểm
   toán và tiếp tục).

Nếu trạng thái là `PAUSED_TIME`: mở phiên mới, thêm toàn bộ Output phiên trước làm Input và đặt
`PREVIOUS_OUTPUT` ở ô cấu hình thành thư mục chứa `campaign.tar.gz` và `linux_runtime.tar`. Lượt đã bắt
đầu không bao giờ chạy lại. SEM-RRT* là bản tái dựng của dự án, không phải mã của tác giả.
'''


FAIRTIME_INTRO = '''# A4 / RP-BIT* so với SEM-RRT* bản C++ — cùng 20 s trên Kaggle

Notebook build lại A4 và bản C++ của SEM-RRT* trên Linux, chạy các cổng kiểm tra trên đầu vào **không** thuộc
held-out, rồi mới chạy thí nghiệm đã đăng ký trong `logs/a4_rpbit_fairtime_20261117/registration.json`:
5 họ × seed bản đồ 71001–71010 × planner seed 716001–716005 × 2 phương pháp = **500 lượt**.

- SEM-RRT* chạy bằng bản C++ (trùng từng bit với bản tái dựng Python), ngân sách 20 s đồng hồ thật.
- A4 là cấu hình chính đã khóa (λ = 33, M = 20), trần 20 s, tự dừng khi hội tụ.
- Chạy 1 làn trên CPU 0, mỗi lúc một lượt: một tiến trình, 1 thread. Hai phương pháp không bao giờ chạy cùng lúc.

Thứ tự:
1. Build, rồi chạy 31 nhóm CTest.
2. Chạy hai cổng `smoke_fairtime` và `gate_fairtime` (seed 51001).
3. Kiểm tra SEM Python và C++ trùng từng bit trên 5 map của cổng (7 425 vòng, 2 seed).
4. Chạy held-out theo từng đoạn khoảng 1 giờ, mỗi đoạn có checkpoint.
5. Validate nghiêm ngặt A4, rồi viết báo cáo.

Nếu một cổng không đạt, notebook dừng trước held-out.

Cách chạy:
1. Upload file `.b64` thành một Kaggle Dataset riêng tư rồi **Add Input** vào notebook này.
2. Settings: **Accelerator = None**, **Internet = On**. Chạy **Save Version → Save & Run All (Commit)**.
3. Kết quả nằm trong `/kaggle/working/a4_rpbit/`:
   - `bootstrap_status.json` và `reports/heldout_fairtime/REPORT.md`;
   - `results_light.tar.gz`: nhỏ, đủ để phân tích;
   - `campaign.tar.gz` và `linux_runtime.tar`: lớn, để kiểm toán và chạy tiếp.

Nếu trạng thái là `PAUSED_TIME`: mở phiên mới, thêm toàn bộ Output của phiên trước làm Input, rồi đặt
`PREVIOUS_OUTPUT` ở ô cấu hình thành thư mục chứa `campaign.tar.gz` và `linux_runtime.tar`. Lượt đã bắt
đầu không bao giờ chạy lại. SEM-RRT* là bản tái dựng của dự án, không phải mã của tác giả.
'''

SELFSTOP_INTRO = '''# A4 / RP-BIT* so với SEM-RRT* bản C++ — cả hai tự dừng khi hội tụ, trên Kaggle

Notebook build lại A4 và bản C++ của SEM-RRT* trên Linux, chạy các cổng kiểm tra trên đầu vào **không** thuộc
held-out, rồi mới chạy thí nghiệm đã đăng ký trong `logs/a4_rpbit_selfstop_20261117/registration.json`:
5 họ × seed bản đồ 81001–81010 × planner seed 816001–816005 × 2 phương pháp = **500 lượt**.

- SEM-RRT* chạy bằng bản C++ (trùng từng bit với bản tái dựng Python) và **tự dừng theo đúng luật của A4**: dừng khi
  chi phí native đã công bố không giảm ít nhất 0,1% (tương đối) qua 4 000 vòng lặp liên tiếp; trần 20 s.
- A4 là cấu hình chính đã khóa (λ = 33, M = 20): dừng khi J không giảm ít nhất 0,1% qua 20 batch; trần 20 s.
- Chạy 1 làn trên CPU 0, mỗi lúc một lượt: một tiến trình, 1 thread. Hai phương pháp không bao giờ chạy cùng lúc.

Thứ tự:
1. Build, rồi chạy 31 nhóm CTest (gồm kiểm tra lượt SEM có luật dừng trùng bit với lượt không dừng bị cắt cùng chỗ).
2. Chạy hai cổng `smoke_selfstop` và `gate_selfstop` (seed 51001).
3. Kiểm tra SEM Python và C++ trùng từng bit trên 5 map của cổng (7 425 vòng, 2 seed).
4. Chạy held-out theo từng đoạn khoảng 1 giờ, mỗi đoạn có checkpoint.
5. Validate nghiêm ngặt A4, rồi viết báo cáo.

Nếu một cổng không đạt, notebook dừng trước held-out.

Cách chạy:
1. Upload file `.b64` thành một Kaggle Dataset riêng tư rồi **Add Input** vào notebook này.
2. Settings: **Accelerator = None**, **Internet = On**. Chạy **Save Version → Save & Run All (Commit)**.
3. Kết quả nằm trong `/kaggle/working/a4_rpbit/`:
   - `bootstrap_status.json` và `reports/heldout_selfstop/REPORT.md`;
   - `results_light.tar.gz`: nhỏ, đủ để phân tích;
   - `campaign.tar.gz` và `linux_runtime.tar`: lớn, để kiểm toán và chạy tiếp.

Nếu trạng thái là `PAUSED_TIME`: mở phiên mới, thêm toàn bộ Output của phiên trước làm Input, rồi đặt
`PREVIOUS_OUTPUT` ở ô cấu hình thành thư mục chứa `campaign.tar.gz` và `linux_runtime.tar`. Lượt đã bắt
đầu không bao giờ chạy lại. SEM-RRT* là bản tái dựng của dự án, không phải mã của tác giả.
'''

# One delivery per registered Kaggle run; 'heldout' is the spent 2026-11-16 delivery, kept reproducible.
PROFILES = dict(
    heldout=dict(delivery=DELIVERY, payload=PAYLOAD, notebook=NOTEBOOK, config=protocol.KAGGLE_CONFIG,
                 module='kaggle_a4_rpbit.bootstrap', heldout='heldout_kaggle', seeds='61001-61010', intro=INTRO,
                 changed_after_lock=CHANGED_AFTER_LOCK, phases=('smoke_final', 'rehearsal', 'kaggle_gate', 'heldout_kaggle'),
                 extra_files=()),
    fairtime=dict(delivery='a4_rpbit_fairtime_kaggle_r1', payload='a4_rpbit_fairtime_source',
                  notebook='a4_rpbit_fairtime_kaggle.ipynb', config=protocol.FAIRTIME_CONFIG,
                  module='kaggle_a4_rpbit.fairtime', heldout='heldout_fairtime', seeds='71001-71010', intro=FAIRTIME_INTRO,
                  # CMakeLists.txt includes the C++ SEM-RRT* targets; the validator checks the SEM engine that ran.
                  changed_after_lock=CHANGED_AFTER_LOCK + ('CMakeLists.txt', 'tools/rp_bit_validation.py'),
                  phases=('smoke_fairtime', 'gate_fairtime', 'heldout_fairtime'),
                  extra_files=(FAIRTIME_REGISTRATION, 'logs/sem_cpp_20261117/dev_equivalence/summary.json')),
    selfstop=dict(delivery='a4_rpbit_selfstop_kaggle_r1', payload='a4_rpbit_selfstop_source',
                  notebook='a4_rpbit_selfstop_kaggle.ipynb', config=protocol.SELFSTOP_CONFIG,
                  module='kaggle_a4_rpbit.selfstop', heldout='heldout_selfstop', seeds='81001-81010', intro=SELFSTOP_INTRO,
                  changed_after_lock=CHANGED_AFTER_LOCK + ('CMakeLists.txt', 'tools/rp_bit_validation.py'),
                  phases=('smoke_selfstop', 'gate_selfstop', 'heldout_selfstop'),
                  # The fairtime registration rides along: the bundle tests also check that controller entry.
                  extra_files=(SELFSTOP_REGISTRATION, FAIRTIME_REGISTRATION,
                               'logs/sem_cpp_20261117/dev_equivalence/summary.json')))


def create_notebook(payload_name, payload_sha256, profile=PROFILES['heldout']):
    transport = (ROOT / 'kaggle_pri_segment/transport.py').read_text(encoding='utf-8')
    declarations = f'PAYLOAD_NAME = {payload_name!r}\nPAYLOAD_SHA256 = {payload_sha256!r}\n'
    setup = SETUP.replace('HELDOUT_SEEDS', profile['seeds'])
    run = RUN.replace('CONTROLLER_MODULE', profile['module']).replace('HELDOUT_PHASE', profile['heldout'])
    cells = [dict(cell_type='markdown', metadata={}, source=profile['intro'].splitlines(keepends=True)),
             code_cell(transport), code_cell(declarations + setup), code_cell(run),
             code_cell("from IPython.display import FileLinks, display\ndisplay(FileLinks(str(WORK)))\n")]
    for index, cell in enumerate(cells):
        cell['id'] = f'a4-rpbit-cell-{index:02d}'
    return dict(nbformat=4, nbformat_minor=5, cells=cells,
                metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                              language_info=dict(name='python', version='3.11'),
                              kaggle=dict(isInternetEnabled=True, isGpuEnabled=False)))


def read_lock():
    # The lock record was written by PowerShell with a UTF-8 BOM; it is read, never rewritten.
    return json.loads(LOCK.read_text(encoding='utf-8-sig'))


def source_files(profile=PROFILES['heldout']):
    lock = read_lock()
    for name, checksum in lock['sha256'].items():
        if name not in profile['changed_after_lock'] + WINDOWS_BINARIES and file_hash(ROOT / name) != checksum:
            raise ValueError('locked source changed after the lock: ' + name)
    files = set()
    for directory in ('apps', 'src', 'include', 'cmake', 'patches', 'configs'):
        files.update(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    files.update((ROOT / 'tools').glob('*.py'))
    files.update((ROOT / 'kaggle_a4_rpbit').glob('*.py'))
    files.update(ROOT / name for name in ('kaggle_pri_segment/__init__.py', 'kaggle_pri_segment/transport.py',
                                          'kaggle_pri_pathsearch/bootstrap.py'))
    files.update((ROOT / 'tests').glob('*.py'))
    files.update((ROOT / 'tests').glob('*.cpp'))
    files.update((ROOT / 'tests/unit').glob('*.cpp'))
    files.update((ROOT / 'tests/integration').glob('*.cpp'))
    files.update(p for p in (ROOT / 'tests/fixtures').iterdir() if p.is_file())
    # Explicit static fixture allowlist; never snapshot temporary test outputs.
    for directory in ('g5_taxonomy', 'native_paper', 'native_paper_v4_endpoint_roundoff'):
        files.update(p for p in (ROOT / 'tests/fixtures' / directory).rglob('*') if p.is_file())
    for name in ('AGENTS.md', 'CMakeLists.txt', 'README.md', 'docs/EXPERIMENT_SPEC.md', 'docs/IMPLEMENTATION_PLAN.md',
                 'docs/A4_RP_BITSTAR.md', 'logs/a4_rpbit_20261116/lock_manifest.json', *profile['extra_files']):
        files.add(ROOT / name)
    files.update(ROOT.glob('requirements*.txt'))
    files.update(ROOT.glob('LICENSE*'))
    for name, entry in read(ROOT / 'configs/dependencies.lock.json')['archives'].items():
        path = ROOT / '.deps' / (name + '.tar.gz')
        if file_hash(path) != entry['sha256']:
            raise ValueError('pinned dependency archive differs: ' + name)
        files.add(path)
    uv = ROOT / '.deps/uv-0.9.7-linux.whl'
    if file_hash(uv) != UV_WHEEL_SHA256:
        raise ValueError('pinned uv wheel differs')
    files.add(uv)
    return sorted(files)


def package(destination, profile=PROFILES['heldout']):
    destination = Path(destination).resolve()
    archive = destination / (profile['payload'] + '.zip')
    if archive.exists():
        raise FileExistsError('refuse to overwrite an existing delivery: ' + str(archive))
    destination.mkdir(parents=True, exist_ok=True)
    files = source_files(profile)
    config = protocol.load_config(profile['config'])
    catalog = dict(protocol_id=protocol.PROTOCOL_ID, schema_version=1, delivery=profile['delivery'],
                   files={p.relative_to(ROOT).as_posix(): file_hash(p) for p in files},
                   kaggle_config_sha256=file_hash(profile['config']), lock_manifest_sha256=file_hash(LOCK),
                   changed_after_lock={name: file_hash(ROOT / name) for name in profile['changed_after_lock']})
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
        bundle.writestr('_bundle_catalog.json', json.dumps(catalog, indent=2, sort_keys=True))
    checksum = file_hash(archive)
    encoded = destination / (profile['payload'] + '.b64')
    with archive.open('rb') as source, encoded.open('xb') as output:
        while chunk := source.read(3 * 1024**2):
            output.write(base64.b64encode(chunk))
    notebook = destination / profile['notebook']
    notebook.write_text(json.dumps(create_notebook(encoded.name, checksum, profile), ensure_ascii=False, indent=1),
                        encoding='utf-8')
    phases = profile['phases']
    inventory = {phase: protocol.inventory(config, phase) for phase in phases}
    (destination / 'registered_inventory.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    import run_rp_bit  # the runner's frozen phase hash, the quantity recorded in the lock
    manifest = dict(catalog, payload_sha256=checksum, files_to_upload=[encoded.name, notebook.name],
                    planned_trials={phase: len(rows) for phase, rows in inventory.items()},
                    phase_config_sha256={phase: digest(run_rp_bit.phase_config(config, phase)) for phase in phases},
                    planner_execution_count=0, linux_execution='NOT_RUN', heldout_started=False)
    (destination / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    checksums = {p.name: file_hash(p) for p in (archive, encoded, notebook, destination / 'registered_inventory.json',
                                                destination / 'delivery_manifest.json')}
    (destination / 'SHA256SUMS.txt').write_text(''.join(f'{value}  {name}\n' for name, value in checksums.items()),
                                                encoding='utf-8')
    return dict(output=str(destination), source_files=len(files), b64_bytes=encoded.stat().st_size,
                notebook=str(notebook), payload=str(encoded), archive_sha256=checksum, planner_execution_count=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--delivery', choices=sorted(PROFILES), default='heldout')
    args = parser.parse_args()
    print(json.dumps(package(args.output, PROFILES[args.delivery]), indent=2))
