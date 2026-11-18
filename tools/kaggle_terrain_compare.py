"""Linux-only schema-10 Kaggle bootstrap. Default stage is prepare; main is opt-in.

The import path is standard-library only. Packages are installed into two local
venvs from complete hashed locks; neither host Python nor system packages change.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile

from kaggle_a3_bundle import sha256, verify
from kaggle_terrain_compare_bundle import export_checkpoint, KIND

ROOT = Path(__file__).resolve().parents[1]
PYTHONS = {"controller": "3.12.7", "sgp": "3.10.19"}
UV_VERSION = "0.9.7"
UV_WHEEL_URL = "https://files.pythonhosted.org/packages/c7/59/61d8e9f1734069049abe9e593961de602397c7194712346906c075fec65f/uv-0.9.7-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"
UV_WHEEL_SHA256 = "8cf6bc2482d1293cc630f66b862b494c09acda9b7faff7307ef52667a2b3ad49"
UV_WHEEL_SIZE = 21382006
UV_BINARY_MEMBER = "uv-0.9.7.data/scripts/uv"
UV_BINARY_SHA256 = "6583a1d002bf6165be0a38366538268e028bbcbb8802880ca09cce80d5505447"
RESERVE_EXPORT_S = 900
HARD_SESSION_S = 32400


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def show_command_failure(stem, receipt, tail_bytes=16384):
    """Expose bounded subprocess diagnostics in notebook output as well as files."""
    print(f"Command failed: {stem.name}; exit_code={receipt.get('exit_code')}; "
          f"termination={receipt.get('termination')}; receipt={stem.with_suffix('.json')}",
          file=sys.stderr, flush=True)
    for suffix in (".stderr.log", ".stdout.log"):
        path = stem.with_suffix(suffix)
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - tail_bytes))
                tail = stream.read(tail_bytes).decode("utf-8", errors="replace")
            if tail:
                print(f"--- {path.name} (last {tail_bytes} bytes at most) ---\n{tail}",
                      file=sys.stderr, flush=True)
        except OSError as error:
            print(f"Cannot read diagnostic {path}: {error}", file=sys.stderr, flush=True)


def command(name, argv, workspace, started, *, cwd=ROOT, env=None, accepted=(0,), max_seconds=None):
    logs = Path(workspace) / "logs/bootstrap"
    logs.mkdir(parents=True, exist_ok=True)
    stem = logs / (name + "_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    began = time.monotonic()
    deadline = started + HARD_SESSION_S - RESERVE_EXPORT_S
    if max_seconds is not None:
        deadline = min(deadline, began + max_seconds)
    if began >= deadline:
        raise RuntimeError("PARTIAL: session deadline; retained attempts will be exported")
    argv = list(map(str, argv))
    receipt = {"argv": argv, "cwd": str(cwd), "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    process = None
    try:
        with stem.with_suffix(".stdout.log").open("wb") as stdout, stem.with_suffix(".stderr.log").open("wb") as stderr:
            process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                       start_new_session=os.name != "nt")
            while process.poll() is None:
                if time.monotonic() >= deadline or shutil.disk_usage(workspace).free < 2 * 1024**3:
                    receipt["termination"] = "SESSION_OR_DISK_LIMIT"
                    process.send_signal(signal.SIGINT)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        if os.name != "nt":
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                    break
                time.sleep(.2)
            receipt["exit_code"] = process.wait()
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
                process.wait()
        receipt["wall_s"] = time.monotonic() - began
        write(stem.with_suffix(".json"), receipt)
    if receipt.get("termination") or receipt.get("exit_code") not in accepted:
        show_command_failure(stem, receipt)
        raise RuntimeError(f"{name} failed/partial: see {stem}.json")
    return stem.with_suffix(".stdout.log").read_text(encoding="utf-8", errors="replace").strip()


def bootstrap_uv(workspace, started):
    folder = ROOT / ".bootstrap-terrain-uv"
    folder.mkdir(exist_ok=True)
    binary, wheel = folder / "uv", folder / "uv.whl"
    bundled = ROOT / ".deps/uv-0.9.7-linux.whl"
    if bundled.exists() and not wheel.exists():
        if sha256(bundled) != UV_WHEEL_SHA256:
            raise ValueError("bundled uv wheel hash mismatch")
        shutil.copyfile(bundled, wheel)
    if binary.exists():
        if sha256(binary) != UV_BINARY_SHA256:
            raise ValueError("retained uv binary hash mismatch")
        return binary
    if wheel.exists():
        if sha256(wheel) != UV_WHEEL_SHA256:
            raise ValueError("retained uv wheel hash mismatch")
    else:
        temporary = folder / "uv.whl.partial"
        if temporary.exists():
            raise RuntimeError("partial uv download retained; use fresh output")
        total = 0
        with urllib.request.urlopen(UV_WHEEL_URL, timeout=30) as response, temporary.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > UV_WHEEL_SIZE or time.monotonic() - started > HARD_SESSION_S - RESERVE_EXPORT_S:
                    raise ValueError("uv size/session bound exceeded")
                stream.write(chunk)
        if total != UV_WHEEL_SIZE or sha256(temporary) != UV_WHEEL_SHA256:
            raise ValueError("uv download hash/size mismatch")
        temporary.replace(wheel)
    with zipfile.ZipFile(wheel) as archive, archive.open(UV_BINARY_MEMBER) as source, binary.open("xb") as dest:
        shutil.copyfileobj(source, dest)
    if sha256(binary) != UV_BINARY_SHA256:
        raise ValueError("uv executable hash mismatch")
    binary.chmod(0o755)
    actual = command("uv_version", [binary, "--version"], workspace, started)
    if actual.split()[:2] != ["uv", UV_VERSION]:
        raise ValueError("uv version mismatch")
    return binary


def isolated_runtimes(workspace, started, *, restore=False):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PIP_") and key not in ("PYTHONHOME", "PYTHONPATH")}
    env.update(UV_PYTHON_INSTALL_DIR=str(ROOT / ".managed-python"),
               UV_CACHE_DIR=str(ROOT / ".uv-terrain-cache"), PYTHONNOUSERSITE="1",
               PIP_CONFIG_FILE=os.devnull, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
               MPLBACKEND="Agg")  # Batch reports must not inherit the notebook's inline backend.
    result = {}
    uv = None
    for role, version in PYTHONS.items():
        venv = ROOT / (".venv-terrain-" + role)
        python = venv / "bin/python"
        if not python.exists():
            if restore:
                raise RuntimeError("checkpoint lacks exact Python runtime: " + role)
            uv = uv or bootstrap_uv(workspace, started)
            command("python_install_" + role, [uv, "--no-config", "python", "install", version],
                    workspace, started, env=env)
            command("venv_" + role, [uv, "--no-config", "venv", "--managed-python", "--python", version, venv],
                    workspace, started, env=env)
            command("locked_packages_" + role,
                    [uv, "--no-config", "pip", "sync", "--python", python, "--require-hashes", "--only-binary", ":all:",
                     ROOT / ("requirements-terrain-compare-" + role + ".txt")], workspace, started, env=env)
        probe = command("python_probe_" + role,
                        [python, "-c", "import json,sys,platform,importlib.metadata as m;print(json.dumps({'version':platform.python_version(),'full':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"],
                        workspace, started, env=env)
        evidence = json.loads(probe)
        if evidence["version"] != version:
            raise ValueError("Python version differs from pinned " + role)
        saved = Path(workspace) / "setup" / (role + "_runtime.json")
        if saved.exists() and json.loads(saved.read_text()) != evidence:
            raise ValueError("runtime drift: " + role)
        write(saved, evidence)
        result[role] = python
    return result, env


def choose_build_jobs(runtime):
    return max(1, min(4, runtime["cpu_capacity"], int(runtime["memory_limit_bytes"] * .8 // (2 * 1024**3))))


def build(workspace, started, runtimes, env):
    python = runtimes["controller"]
    observed = command("hardware", [python, "-c",
        "import sys,json;sys.path.insert(0,'tools');from terrain_compare_scheduler import detect_runtime;print(json.dumps(detect_runtime()))"],
        workspace, started, env=env)
    runtime = json.loads(observed)
    jobs = choose_build_jobs(runtime)
    missing = [name for name in ("g++", "gcc", "cmake", "ninja", "bash", "ldd") if not shutil.which(name)]
    if missing:
        raise RuntimeError("BLOCKED: missing system build tools " + ", ".join(missing))
    write(Path(workspace) / "setup/build_runtime.json", {"hardware": runtime, "build_jobs": jobs})
    for tool in ("g++", "cmake", "ninja"):
        command("version_" + tool.replace("+", "x"), [tool, "--version"], workspace, started)
    command("extract_pinned_sources", [python, ROOT / "tools/fetch_dependencies.py"], workspace, started, env=env)
    command("build_dependencies", [python, ROOT / "tools/build_dependencies.py", "--jobs", jobs,
                                  "--log-dir", Path(workspace) / "logs/dependencies"], workspace, started, env=env)
    build_dir, boost = ROOT / "build-terrain-compare", ROOT / ".deps/boost_1_82_0"
    command("configure", ["cmake", "-S", ROOT, "-B", build_dir, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
                          "-DCMAKE_CXX_COMPILER=" + shutil.which("g++"), f"-DPython3_EXECUTABLE={python}",
                          f"-DCMAKE_PREFIX_PATH={ROOT / '.deps/install'}", f"-DBOOST_ROOT={boost}",
                          f"-DBOOST_LIBRARYDIR={boost / 'stage/lib'}", "-DBoost_NO_SYSTEM_PATHS=ON",
                          "-DBoost_NO_BOOST_CMAKE=ON", "-DBoost_USE_STATIC_LIBS=ON",
                          "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"],
            workspace, started, env=env)
    command("build_release", ["cmake", "--build", build_dir, "--parallel", jobs], workspace, started, env=env)
    return build_dir


def campaign_command(action, workspace, runtimes, *, mode=None, resume=False, allow_main=False, started=None):
    argv = [runtimes["controller"], ROOT / "tools/terrain_compare_campaign.py", action, "--workspace", workspace]
    if action == "prepare":
        argv += ["--build-dir", ROOT / "build-terrain-compare", "--sgp-python", runtimes["sgp"]]
    if mode:
        argv += ["--mode", mode]
    if action == "run":
        if resume:
            argv.append("--resume")
        if allow_main:
            argv.append("--allow-main")
        argv += ["--session-elapsed-s", max(0, time.monotonic() - started)]
    if action == "validate":
        argv += ["--jobs", 4]
    return argv


def retain_checkpoint(workspace, input_archive, *, stage_error=None):
    """A failed export must not replace the original build/prepare failure."""
    exports = workspace / "exports"
    path = exports / ("terrain_compare_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".zip")
    try:
        exports.mkdir(exist_ok=True)
        export_checkpoint(workspace, path, root=ROOT, input_archive=input_archive)
    except Exception as error:
        print(f"Checkpoint export FAILED: {error}; retained workspace: {workspace}",
              file=sys.stderr, flush=True)
        traceback.print_exc()
        try:
            write(workspace / "logs/bootstrap" / (path.stem + "_export_failure.json"),
                  {"checkpoint": str(path), "export_error": repr(error),
                   "stage_error": repr(stage_error) if stage_error is not None else None})
        except OSError:
            # Disk exhaustion must not hide the original exception either.
            pass
        if stage_error is None:
            raise
        print("Preserving original stage failure; checkpoint was not produced.",
              file=sys.stderr, flush=True)
        return None
    print("Retained checkpoint:", path, flush=True)
    return path


def execute(args):
    if not sys.platform.startswith("linux") or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("BLOCKED: Kaggle bootstrap requires Linux x86_64")
    if args.stage == "main" and not args.allow_main:
        raise ValueError("main requires --allow-main")
    manifest = verify(args.input_archive, args.input_sha256)
    if manifest["kind"] != KIND:
        raise ValueError("wrong input bundle kind")
    for entry in manifest["files"]:
        if sha256(ROOT / entry["path"]) != entry["sha256"]:
            raise ValueError("source has changed since input bundle: " + entry["path"])
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    started = args.session_started_monotonic or time.monotonic()
    try:
        if args.stage != "export":
            runtimes, env = isolated_runtimes(workspace, started, restore=args.restored_runtime)
            env["LD_LIBRARY_PATH"] = str(ROOT / ".deps/install/lib") + os.pathsep + env.get("LD_LIBRARY_PATH", "")
            if args.stage == "prepare":
                if args.restored_runtime:
                    raise ValueError("restored runtime must not be rebuilt; resume a measured stage")
                build(workspace, started, runtimes, env)
                for action in ("prepare", "correctness"):
                    command(action, campaign_command(action, workspace, runtimes), workspace, started, env=env)
            elif args.stage in ("smoke", "main"):
                modes = ("smoke-serial", "smoke-parallel") if args.stage == "smoke" else ("main",)
                for mode in modes:
                    existing = (workspace / "campaigns" / mode / "manifest.json").exists()
                    resume = args.resume and existing
                    command("run_" + mode, campaign_command("run", workspace, runtimes, mode=mode, resume=resume,
                                                           allow_main=args.allow_main, started=started),
                            workspace, started, env=env, accepted=(0, 2))
                    command("validate_" + mode, campaign_command("validate", workspace, runtimes, mode=mode),
                            workspace, started, env=env, accepted=(0, 2))
            elif args.stage == "validate":
                command("validate_" + args.mode, campaign_command("validate", workspace, runtimes, mode=args.mode),
                        workspace, started, env=env, accepted=(0, 2))
    finally:
        retain_checkpoint(workspace, args.input_archive, stage_error=sys.exc_info()[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "main", "validate", "export"), default="prepare")
    parser.add_argument("--mode", choices=("smoke-serial", "smoke-parallel", "main"), default="main")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--restored-runtime", action="store_true")
    parser.add_argument("--session-started-monotonic", type=float)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
