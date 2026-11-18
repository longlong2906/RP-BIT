"""Build a notebook that resumes the saved pilot without changing its source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/"artifacts/a3t_sem_exposure_pilot_kaggle_v1/sem_exposure_pilot_kaggle_v1.ipynb"
RECOVERY = ROOT/"logs/a3t_sem_exposure_pilot_kaggle_recovery"
SOURCE_SHA256 = "a724eaaea4a1fb2b209d4476a23e11312519ee42c5ec822c240c6ce0739f8985"
CHECKPOINT_SHA256 = "17f0842081e879a39062157b7750dac602216aa42e4f410e9390ad93d4e05609"

# This launcher runs outside the restored repository. It never writes into the
# frozen source and does not relax the controller's process or trial checks.
LAUNCHER = r"""
import datetime
import json
from pathlib import Path
import re
import runpy
import sys
import time


def tracker_identity(pid, expected_ticks, snapshot, read_command):
    current = snapshot.get(pid)
    if current is None or current["start_ticks"] != expected_ticks:
        return False
    try:
        args = read_command(pid).rstrip(b"\x00").split(b"\x00")
    except OSError:
        return False
    return (len(args) >= 3 and args[0].rsplit(b"/", 1)[-1].startswith(b"python") and args[-2] == b"-c"
            and re.fullmatch(rb"from multiprocessing\.resource_tracker import main;main\([0-9]+\)",
                             args[-1]) is not None)


def install_drain(supervisor, workspace, grace_s=2.):
    original = supervisor.quiesce_descendants
    audit_path = workspace/"logs"/"resume_tracker_cleanup.jsonl"

    def drain(root_pid, tracked, baseline):
        if not supervisor.linux_host():
            return original(root_pid, tracked, baseline)
        until = time.monotonic()+grace_s
        while True:
            snapshot = supervisor.process_snapshot()
            live = supervisor.track_descendants(snapshot, root_pid, tracked, baseline)
            if not live or time.monotonic() >= until:
                break
            time.sleep(.05)
        identified = {pid for pid in live if pid in tracked and tracker_identity(
            pid, tracked[pid], snapshot, lambda candidate: Path(f"/proc/{candidate}/cmdline").read_bytes())}
        terminated = original(root_pid, tracked, baseline)
        if terminated and set(terminated) <= identified:
            # Original cleanup has proved these helpers are gone. All other
            # terminations and nonzero child exits still follow the old rules.
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(dict(event="cleaned_verified_python_resource_tracker",
                    pids=terminated, utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))+"\n")
            return []
        return terminated

    supervisor.quiesce_descendants = drain


if __name__ == "__main__":
    bootstrap = Path(sys.argv.pop(1))
    sys.path.insert(0, str(bootstrap.parent))
    import kaggle_native_paper_v4 as supervisor
    install_drain(supervisor, Path(sys.argv[sys.argv.index("--workspace")+1]))
    runpy.run_path(str(bootstrap), run_name="__main__")
"""


def make_notebook():
    notebook = json.loads(ORIGINAL.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    original_setup = "".join(cells[1]["source"])
    if SOURCE_SHA256 not in original_setup:
        raise ValueError("original source notebook does not match the recorded campaign")
    cells[0]["source"] = [
        "# Tiếp tục pilot A3-T SEM exposure\n",
        "Gắn Output của phiên Kaggle đã dừng vào notebook này qua Add Input. "
        "Output phải có checkpoint đã xác minh với 2.400 lượt tune đã bắt đầu. "
        "Notebook dùng nguyên mã nguồn, binary Linux, runtime và các shard của phiên đó.\n",
        "\nChạy bằng CPU có ít nhất 4 vCPU hiệu dụng và Save & Run All. "
        "100 lượt tune còn lại được chạy một lần. Confirm chỉ chạy nếu tune khóa được λ. "
        "Giữ toàn bộ Output mới nếu cần tiếp tục thêm một phiên.\n",
    ]
    cells[1]["source"] = f"""from pathlib import Path
import time
SESSION_STARTED = time.monotonic()
BASE = Path('/kaggle/working/a3t_sem_exposure_pilot_v1')
ROOT = BASE / 'repo'
WORKSPACE = BASE / 'campaign'
SOURCE_SHA256 = {SOURCE_SHA256!r}
RESUME_SHA256 = {CHECKPOINT_SHA256!r}
RESUME_INDEX = None  # Optional explicit path within attached /kaggle/input.
OUTPUT_ARCHIVE = None  # Fallback: path to the attached original ZIP saved as .payload.
STAGE = 'all'
ALLOW_PILOT = True
""".splitlines(keepends=True)
    source_cell = "".join(cells[2]["source"])
    before, remainder = source_cell.split("if PAYLOAD is None and not SOURCE_ARCHIVE.exists():", 1)
    verified_extraction = "with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:" + remainder.split(
        "with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:", 1)[1]
    audit_source = (RECOVERY/"audit_output.py").read_text(encoding="utf-8")
    stage_source = (RECOVERY/"stage_output_checkpoint.py").read_text(encoding="utf-8")
    compile(audit_source, "audit-output", "exec")
    compile(stage_source, "stage-output", "exec")
    fallback = f"""
if OUTPUT_ARCHIVE is not None:
    from types import ModuleType
    if RESUME_INDEX is not None:
        raise ValueError('Set OUTPUT_ARCHIVE or RESUME_INDEX, not both.')
    opaque = Path(OUTPUT_ARCHIVE).resolve(strict=True)
    if Path('/kaggle/input').resolve() not in opaque.parents:
        raise ValueError('OUTPUT_ARCHIVE must be in attached Kaggle Input.')
    evidence = ModuleType('audit_output')
    exec(compile({audit_source!r}, 'audit-output', 'exec'), evidence.__dict__)
    previous_audit_module = sys.modules.get('audit_output')
    sys.modules['audit_output'] = evidence
    try:
        staging = {{'__name__': 'pilot_stage'}}
        exec(compile({stage_source!r}, 'stage-output', 'exec'), staging)
        print(staging['stage'](opaque, BASE, write_files=True))
    finally:
        if previous_audit_module is None:
            sys.modules.pop('audit_output', None)
        else:
            sys.modules['audit_output'] = previous_audit_module
    RESUME_INDEX = BASE / 'campaign/exports/checkpoint_index.json'
"""
    discover = fallback+"""
elif RESUME_INDEX is None:
    matches = [path for path in Path('/kaggle/input').rglob('checkpoint_index.json')
               if path.parent.name == 'exports' and checksum(path) == RESUME_SHA256]
    if len(matches) != 1:
        raise ValueError('Attach exactly one saved Output with the pinned checkpoint index.')
    RESUME_INDEX = matches[0]
RESUME_INDEX = Path(RESUME_INDEX).resolve(strict=True)
expected_parent = BASE.resolve() if OUTPUT_ARCHIVE is not None else Path('/kaggle/input').resolve()
if expected_parent not in RESUME_INDEX.parents or checksum(RESUME_INDEX) != RESUME_SHA256:
    raise ValueError('Resume index does not match the attached frozen checkpoint.')
SOURCE_ARCHIVE = RESUME_INDEX.parents[2] / 'source_input.zip'
if not SOURCE_ARCHIVE.is_file() or checksum(SOURCE_ARCHIVE) != SOURCE_SHA256:
    raise ValueError('Original source_input.zip is missing or changed in the attached Output.')
"""
    guard = """
phase = json.loads((WORKSPACE/'pilot/tune/manifest.json').read_text(encoding='utf-8'))
attempted = sum(entry['status'] != 'NOT_RUN' for entry in phase['entries'])
if (phase['status'] != 'PAUSED_TIME' or len(phase['entries']) != 2500 or attempted != 2400
        or len(phase['shards']) != 48 or (WORKSPACE/'pilot/confirm/manifest.json').exists()):
    raise ValueError('Restored trial boundary differs from the audited 2,400-start checkpoint.')
print('Verified restore: 2,400 started attempts, 100 tune trials left.')
"""
    cells[2]["source"] = (before+discover+verified_extraction+guard).splitlines(keepends=True)
    run_cell = "".join(cells[3]["source"]).replace(
        "argv = [sys.executable, str(ROOT",
        f"resume_launcher = {LAUNCHER!r}\nargv = [sys.executable, '-c', resume_launcher, str(ROOT", 1)
    cells[3]["source"] = run_cell.splitlines(keepends=True)
    for i, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"resume-cell-{i}", "exec")
    compile(LAUNCHER, "resume-launcher", "exec")
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(make_notebook(), stream, ensure_ascii=False, indent=1)
        stream.write("\n")


if __name__ == "__main__":
    main()
