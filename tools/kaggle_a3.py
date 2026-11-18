"""Kaggle CPU bootstrap and guarded A3 pilot orchestration.

The bootstrap import graph uses only the Python standard library. All measured
work runs in an isolated Python 3.12 environment; nothing is installed globally.
"""
import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile

from kaggle_a3_bundle import archive, digest, extract, sha256, verify

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_ID = "a3_improvements_pilot_v1"
SOFT_SECONDS = 8 * 3600
HARD_SECONDS = 9 * 3600
EXPORT_RESERVE_SECONDS = 15 * 60
MIN_FREE_BYTES = 2 * 1024**3
MAX_OUTPUT_BYTES = 18 * 1024**3
EXPORT_GROWTH_HEADROOM_BYTES = 256 * 1024**2
# Registered by main before launching any subprocess, including bootstrap.
EXPORT_INPUT_BYTES = 0
UV_VERSION = "0.6.17"
# Official PyPI wheel for the Kaggle Linux x86_64 CPU image. Both the download
# and the only extracted member are pinned; bootstrap never invokes host pip.
UV_WHEEL_URL = ("https://files.pythonhosted.org/packages/d4/6b/"
    "1bea7cff3104890b33c73155c811e48129eed8272e6f4ba848a116994fa4/"
    "uv-0.6.17-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl")
UV_WHEEL_SHA256 = "aa416f287c81bfffd21e82765944035e6c3f4566615bd4fc03db3a704be8e4d5"
UV_WHEEL_SIZE = 17247156
UV_BINARY_MEMBER = "uv-0.6.17.data/scripts/uv"
UV_BINARY_SHA256 = "3fd4e6bb820c05208b925822a20677a5ac3d3c53f24276d1332a349a55032b84"
RUNTIME_FILES = ("build-kaggle/terrain_benchmark", "build-kaggle/CMakeCache.txt",
    ".deps/install/include/ompl-1.7/ompl/config.h", ".deps/boost_1_82_0/boost/version.hpp",
    ".deps/install/include/eigen3/Eigen/src/Core/util/Macros.h")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def unique_id():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def image_evidence(os_release=Path("/etc/os-release")):
    """Observed image provenance, using only explicitly non-secret identifiers."""
    path = Path(os_release)
    try:
        release = path.read_text(encoding="utf-8") if path.is_file() else None
    except OSError:
        release = None
    identifiers = {key: os.environ[key] for key in ("KAGGLE_DOCKER_IMAGE", "KAGGLE_IMAGE_VERSION")
                   if os.environ.get(key)}
    return {"os_release": release, "image_identifiers": identifiers,
            "image_identifier_status": "observed" if identifiers else "unavailable"}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + unique_id() + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def output_size(output):
    return sum(file_size(p) for p in Path(output).rglob("*") if p.is_file())


def file_size(path):
    try:
        return Path(path).stat().st_size
    except FileNotFoundError:
        # A live atomic JSON commit can rename its temporary file during scans.
        return 0


def reserved_checkpoint_bytes(output, *, input_bytes=None, root=ROOT):
    """Reserve a full next checkpoint, excluding already exported ZIPs."""
    paths = [path for _, path in runtime_entries(root)]
    output = Path(output)
    paths += [path for path in output.rglob("*") if path.is_file()
              and "exports" not in path.relative_to(output).parts]
    payload = (EXPORT_INPUT_BYTES if input_bytes is None else input_bytes) + sum(file_size(p) for p in paths)
    return payload + 16 * 1024**2 + (len(paths) + 1) * 1024


def guard_disk(output, *, reserve=MIN_FREE_BYTES, input_bytes=None, root=ROOT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pending_export = reserved_checkpoint_bytes(output, input_bytes=input_bytes, root=root)
    if (shutil.disk_usage(output).free < reserve + pending_export + EXPORT_GROWTH_HEADROOM_BYTES or
            output_size(output) + pending_export + EXPORT_GROWTH_HEADROOM_BYTES >= MAX_OUTPUT_BYTES):
        raise RuntimeError("PARTIAL: disk guard reserves space for a full checkpoint; stop collection and export retained data")


def stop_process(process):
    if process.poll() is not None:
        return
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        # The leader can exit before a descendant; never leave its process
        # group running after a global/session deadline.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        process.kill()
    process.wait()


def run_command(name, argv, output, started, *, cwd=ROOT, env=None, max_seconds=None,
                check=True, reserve_seconds=EXPORT_RESERVE_SECONDS):
    """Immutable command evidence, process-group timeout, and disk guard."""
    output = Path(output)
    logs = output / "setup" / "commands"
    logs.mkdir(parents=True, exist_ok=True)
    stem = name + "_" + unique_id()
    stdout, stderr = logs / (stem + ".stdout.log"), logs / (stem + ".stderr.log")
    began = time.monotonic()
    deadline = started + HARD_SECONDS - reserve_seconds
    if max_seconds is not None:
        deadline = min(deadline, began + max_seconds)
    if began >= deadline:
        raise RuntimeError("PARTIAL: session finalization deadline reached before " + name)
    guard_disk(output)
    record = {"name": name, "argv": list(map(str, argv)), "cwd": str(cwd), "started_utc": utc(),
              "stdout": stdout.relative_to(output).as_posix(), "stderr": stderr.relative_to(output).as_posix()}
    print("Starting " + name + ": " + " ".join(record["argv"]), flush=True)
    process = None
    try:
        with stdout.open("xb") as out, stderr.open("xb") as err:
            process = subprocess.Popen(record["argv"], cwd=cwd, env=env, stdout=out, stderr=err,
                                       start_new_session=(os.name == "posix"))
            next_disk_check = began
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    raise RuntimeError("PARTIAL: command exceeded session/phase deadline: " + name)
                if time.monotonic() >= next_disk_check:
                    guard_disk(output)
                    next_disk_check = time.monotonic() + 10
                time.sleep(.1)
            record["exit_code"] = process.returncode
        if check and process.returncode:
            raise RuntimeError(f"FAILED: {name} exited {process.returncode}; see {stderr}")
    except BaseException as error:
        record["error"] = str(error)
        if process is not None:
            stop_process(process)
            record["exit_code"] = process.returncode
        else:
            record["exit_code"] = None
        raise
    finally:
        record.update(finished_utc=utc(), elapsed_s=time.monotonic() - began)
        write(logs / (stem + ".json"), record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if record.get("error") or record.get("exit_code"):
            # venv and build tools can put the useful error on either stream.
            # Keep full files while also making notebook logs self-contained.
            for label, path in (("stdout", stdout), ("stderr", stderr)):
                if path.is_file():
                    with path.open("rb") as handle:
                        handle.seek(max(0, path.stat().st_size - 8192))
                        tail = handle.read().decode("utf-8", errors="replace")
                    if tail.strip():
                        print(f"--- {name} {label} (last up to 8192 bytes) ---\n{tail}", flush=True)
    return stdout.read_text(encoding="utf-8", errors="replace").strip()


def verify_source(input_archive, expected_hash, root=ROOT):
    manifest = verify(input_archive, expected_hash)
    if (manifest.get("kind") != "A3_KAGGLE_SOURCE_INPUT" or
            manifest.get("metadata", {}).get("protocol_id") != PROTOCOL_ID):
        raise ValueError("wrong source input kind/protocol")
    root = Path(root).resolve()
    for row in manifest["files"]:
        path = root / row["path"]
        if path.is_symlink() or root not in path.resolve().parents or not path.is_file() or sha256(path) != row["sha256"]:
            raise ValueError("source input differs from extracted workspace: " + row["path"])
    return manifest


def restore_checkpoint(checkpoint, expected_input_hash, output, *, root=ROOT):
    """Verify before restoring; existing conflicting evidence is never overwritten."""
    root, output = Path(root).resolve(), Path(output).resolve()
    manifest = verify(checkpoint)
    meta = manifest.get("metadata", {})
    if (manifest.get("kind") != "A3_KAGGLE_RUNTIME_CHECKPOINT" or
            meta.get("protocol_id") != PROTOCOL_ID or meta.get("input_sha256") != expected_input_hash or
            not meta.get("runtime_ready")):
        raise ValueError("checkpoint is not resumable or has a different source input")
    stage = root / ".checkpoint_restore" / unique_id()
    extract(checkpoint, stage)
    copies = []
    for row in manifest["files"]:
        name = row["path"]
        if name.startswith("runtime/"):
            relative = name[len("runtime/"):]
            allowed_library = (Path(relative).parent.as_posix() == ".deps/install/lib" and
                               Path(relative).name.startswith("libompl"))
            if relative not in RUNTIME_FILES and not allowed_library:
                raise ValueError("checkpoint contains an unexpected runtime file: " + relative)
            target = root / relative
        elif name.startswith("artifacts/"):
            target = output / name[len("artifacts/"):]
        elif name == "input/source.zip":
            if row["sha256"] != expected_input_hash:
                raise ValueError("checkpoint embedded input SHA-256 differs")
            continue
        else:
            raise ValueError("unrecognized checkpoint member: " + name)
        boundary = root if name.startswith("runtime/") else output
        if target.is_symlink() or boundary not in target.resolve().parents:
            raise ValueError("checkpoint target escapes workspace")
        if target.exists() and (not target.is_file() or sha256(target) != row["sha256"]):
            raise ValueError("checkpoint conflicts with existing evidence: " + str(target))
        copies.append((stage / name, target))
    for source, target in copies:
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    write(output / "setup" / "restore" / (unique_id() + ".json"), {
        "checkpoint_sha256": sha256(checkpoint), "input_sha256": expected_input_hash,
        "utc": utc(), "restored_files": len(copies)})
    return meta


def runtime_entries(root=ROOT):
    """Only the measured runtime/provenance, never object files or a venv."""
    root = Path(root).resolve()
    paths = [root / name for name in RUNTIME_FILES]
    paths += sorted((root / ".deps/install/lib").glob("libompl*"))
    result = []
    for path in paths:
        if path.is_file():
            resolved = path.resolve()
            if root not in resolved.parents:
                raise ValueError("runtime symlink escapes source workspace")
            result.append(("runtime/" + path.relative_to(root).as_posix(), resolved))
    return result


def export_checkpoint(output, input_archive, input_hash, reason, *, root=ROOT):
    output, root = Path(output).resolve(), Path(root).resolve()
    entries = [("input/source.zip", Path(input_archive).resolve())]
    entries += runtime_entries(root)
    for path in output.rglob("*"):
        if not path.is_file() or "exports" in path.relative_to(output).parts:
            continue
        if path.is_symlink():
            raise ValueError("artifact symlink cannot enter checkpoint")
        entries.append(("artifacts/" + path.relative_to(output).as_posix(), path))
    runtime_ready = ((root / "build-kaggle/terrain_benchmark").is_file() and
                     (output / "setup/build_gate.json").is_file() and
                     (output / "setup/runtime.json").is_file())
    needed = sum(path.stat().st_size for _, path in entries)
    # Use the uncompressed payload plus metadata allowance as a conservative
    # bound: free scratch space is distinct from Kaggle's retained-output quota.
    estimated_archive_bytes = needed + 16 * 1024**2 + len(entries) * 1024
    if output_size(output) + estimated_archive_bytes >= MAX_OUTPUT_BYTES:
        raise RuntimeError("PARTIAL: full checkpoint would exceed retained-output quota; raw data retained")
    if shutil.disk_usage(output).free < estimated_archive_bytes + 256 * 1024**2:
        raise RuntimeError("PARTIAL: insufficient free space to export full checkpoint; raw data retained")
    target = output / "exports" / ("a3_checkpoint_" + unique_id() + ".zip")
    manifest = archive(entries, target, "A3_KAGGLE_RUNTIME_CHECKPOINT", {
        "protocol_id": PROTOCOL_ID, "input_sha256": input_hash, "utc": utc(),
        "reason": reason, "runtime_ready": runtime_ready,
        "runtime_identity_sha256": read(output / "setup/runtime.json").get("runtime_identity_sha256")
            if (output / "setup/runtime.json").is_file() else None})
    write(output / "exports/latest.json", {"path": target.name, "sha256": sha256(target),
        "files_sha256": manifest["files_sha256"], "reason": reason, "runtime_ready": runtime_ready})
    return target


def bootstrap_evidence(output, name, record):
    """Retain each session's bootstrap receipt, including before runtime exists."""
    write(Path(output) / "setup/bootstrap_sessions" / (name + "_" + unique_id() + ".json"), record)
    first = Path(output) / "setup" / (name + ".json")
    if not first.exists():
        write(first, record)


def bootstrap_uv(output, started):
    """Get the pinned standalone uv without needing host pip or ensurepip."""
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("BLOCKED: pinned uv bootstrap requires Linux x86_64")
    stage = ROOT / ".bootstrap-uv"
    stage.mkdir(exist_ok=False)
    wheel, binary = stage / "uv.whl", stage / "uv"
    receipt = {"utc": utc(), "version_requested": UV_VERSION, "url": UV_WHEEL_URL,
               "wheel_sha256_expected": UV_WHEEL_SHA256, "wheel_size_expected": UV_WHEEL_SIZE,
               "binary_member": UV_BINARY_MEMBER, "binary_sha256_expected": UV_BINARY_SHA256}
    deadline = min(started + HARD_SECONDS - EXPORT_RESERVE_SECONDS, time.monotonic() + 180)
    try:
        guard_disk(output)
        if time.monotonic() >= deadline:
            raise RuntimeError("PARTIAL: session deadline before uv bootstrap")
        print("Downloading pinned bootstrap uv " + UV_VERSION + " from official PyPI", flush=True)
        size = 0
        with urllib.request.urlopen(UV_WHEEL_URL, timeout=30) as response, wheel.open("xb") as handle:
            while True:
                if time.monotonic() >= deadline:
                    raise RuntimeError("PARTIAL: uv download deadline reached")
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > UV_WHEEL_SIZE:
                    raise ValueError("uv wheel exceeds pinned size")
                handle.write(chunk)
        receipt.update(wheel_size_observed=size, wheel_sha256_observed=sha256(wheel))
        if size != UV_WHEEL_SIZE or receipt["wheel_sha256_observed"] != UV_WHEEL_SHA256:
            raise ValueError("uv wheel SHA-256/size mismatch")
        with zipfile.ZipFile(wheel) as source, source.open(UV_BINARY_MEMBER) as member, binary.open("xb") as target:
            shutil.copyfileobj(member, target)
        receipt["binary_sha256_observed"] = sha256(binary)
        if receipt["binary_sha256_observed"] != UV_BINARY_SHA256:
            raise ValueError("uv binary SHA-256 mismatch")
        binary.chmod(0o755)
        receipt["version_observed"] = run_command("bootstrap_uv_version", [binary, "--version"], output, started)
        if receipt["version_observed"].split()[:2] != ["uv", UV_VERSION]:
            raise ValueError("uv version differs from pinned bootstrap")
        receipt["status"] = "PASS"
        return binary
    except BaseException as error:
        receipt.update(status="FAILED", error=str(error))
        raise
    finally:
        bootstrap_evidence(output, "uv_bootstrap", receipt)


def python_probe(name, executable, output, started):
    # psutil 5.9.0 has no CPython 3.12 wheel: its locked source build needs
    # headers. Detect an incomplete system Python before installing packages.
    script = ("import json,platform,sys,sysconfig; from pathlib import Path; "
              "print(json.dumps({'version':platform.python_version(),'full':sys.version,"
              "'headers_available':(Path(sysconfig.get_path('include'))/'Python.h').is_file() "
              "and Path(sysconfig.get_config_h_filename()).is_file()}))")
    return json.loads(run_command(name, [executable, "-c", script], output, started))


def isolated_python(args):
    root, output = ROOT, args.output
    prior = read(output / "setup/runtime.json") if args.resume_archive else None
    requested = prior["python_version"] if prior else "3.12"
    native = shutil.which("python3.12")
    if sys.version_info[:2] == (3, 12):
        native = sys.executable
    selection = {"utc": utc(), "requested_python": requested, "native_candidate": native}
    if native:
        probe = python_probe("python_candidate", native, output, args.notebook_started_monotonic)
        selection["native_probe"] = probe
        if not probe["version"].startswith("3.12."):
            selection["native_rejection"] = "requires Python 3.12"
            native = None
        elif not probe["headers_available"]:
            selection["native_rejection"] = "development headers unavailable for locked psutil source build"
            native = None
        elif prior and (probe["version"] != requested or probe["full"] != prior["python_full"]):
            selection["native_rejection"] = "differs from resume Python version/build"
            native = None
    venv = root / ".venv-kaggle"
    if venv.exists():
        raise RuntimeError("BLOCKED: isolated venv already exists; use a fresh notebook session")
    uv = bootstrap_uv(output, args.notebook_started_monotonic)
    inherited = {key: value for key, value in os.environ.items()
                 if not key.startswith("PIP_") and key not in ("PYTHONHOME", "PYTHONPATH")}
    env = dict(inherited, UV_PYTHON_INSTALL_DIR=str(root / ".managed-python"),
               UV_PYTHON_BIN_DIR=str(root / ".managed-python-bin"), UV_CACHE_DIR=str(root / ".uv-cache"),
               XDG_DATA_HOME=str(root / ".bootstrap-data"), XDG_CONFIG_HOME=str(root / ".bootstrap-config"),
               XDG_CACHE_HOME=str(root / ".bootstrap-cache"), PYTHONNOUSERSITE="1",
               PIP_CACHE_DIR=str(root / ".pip-cache"), PIP_CONFIG_FILE=os.devnull,
               # Managed CPython may name clang in sysconfig; build the pinned
               # psutil C extension with the image's checked GNU toolchain.
               CC=shutil.which("gcc") or "gcc", CXX=shutil.which("g++") or "g++")
    command = [uv, "--no-config", "--no-preview"]
    try:
        if native:
            selected = native
            selection["provider"] = "system"
        else:
            selection["provider"] = "uv-managed"
            run_command("provision_python", [*command, "python", "install", requested],
                        output, args.notebook_started_monotonic, env=env)
            selected = run_command("find_managed_python", [*command, "--no-python-downloads", "--managed-python",
                                   "python", "find", requested], output, args.notebook_started_monotonic, env=env)
            if not Path(selected).is_absolute() or not (root / ".managed-python").resolve() in Path(selected).resolve().parents:
                raise ValueError("uv selected Python outside managed workspace")
        selection["selected_interpreter"] = str(selected)
        run_command("create_isolated_python", [*command, "--no-python-downloads", "venv", "--seed",
                    "--python", selected, venv], output, args.notebook_started_monotonic, env=env)
        python = venv / "bin/python"
        actual = python_probe("isolated_python_probe", python, output, args.notebook_started_monotonic)
        selection["isolated_probe"] = actual
        if not actual["version"].startswith("3.12.") or not actual["headers_available"]:
            raise ValueError("isolated Python must be 3.12 with development headers")
        if prior and (actual["version"] != requested or actual["full"] != prior["python_full"]):
            raise ValueError("resume Python version/build differs from frozen runtime")
        selection["status"] = "PASS"
    except BaseException as error:
        selection.update(status="FAILED", error=str(error))
        raise
    finally:
        bootstrap_evidence(output, "python_bootstrap", selection)
    python = venv / "bin/python"
    requirements = output / "setup/python-freeze.txt" if prior else root / "requirements-cloud-benchmark.txt"
    command = [python, "-m", "pip", "install", "--disable-pip-version-check", "-r", requirements]
    if prior:
        command.append("--no-deps")
    run_command("install_locked_requirements", command, output, args.notebook_started_monotonic, env=env)
    freeze = run_command("python_freeze", [python, "-m", "pip", "freeze", "--all"],
                         output, args.notebook_started_monotonic, env=env)
    if prior:
        if sorted(freeze.splitlines()) != sorted(requirements.read_text().splitlines()):
            raise ValueError("resume Python packages differ from frozen runtime")
    else:
        (output / "setup/python-freeze.txt").write_text(freeze + "\n", encoding="utf-8")
    return python


def bootstrap(args):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("BLOCKED: Kaggle bootstrap requires Linux; no pilot runs on Windows")
    if not shutil.which("g++"):
        raise RuntimeError("BLOCKED: g++ unavailable in Kaggle image")
    if not shutil.which("gcc"):
        raise RuntimeError("BLOCKED: gcc unavailable for locked Python dependency source builds")
    verify_source(args.input_archive, args.input_sha256)
    if args.resume_archive:
        restore_checkpoint(args.resume_archive, args.input_sha256, args.output)
    python = isolated_python(args)
    command = execution_command(args, python)
    write(args.output / "setup" / ("handoff_" + unique_id() + ".json"),
          {"utc": utc(), "argv": command, "action": "exec_isolated_python", "exit_code": None})
    os.execv(str(python), command)


def execution_command(args, python):
    command = [str(python), str(ROOT / "tools/kaggle_a3.py"), "--stage", "execute",
        "--input-archive", str(args.input_archive), "--input-sha256", args.input_sha256,
        "--output", str(args.output), "--notebook-started-monotonic", str(args.notebook_started_monotonic)]
    if args.allow_a3_pilot:
        command.append("--allow-a3-pilot")
    if args.resume_archive:
        command += ["--resume-archive", str(args.resume_archive)]
    return command


def execute(args):
    from experiment_common import cloud_code_hash, cloud_environment, cloud_ompl_library, worker_env
    from setup_cloud_benchmark import (observed_dependency_versions, project_configure,
                                       require_prerequisites, verify_pinned_archives)
    from a3_pilot_campaign import runtime_identity
    require_prerequisites()
    verify_source(args.input_archive, args.input_sha256)
    output, began = args.output, args.notebook_started_monotonic
    build = ROOT / "build-kaggle"
    worker = build / "terrain_benchmark"
    gate_path = output / "setup/build_gate.json"
    config = ROOT / "configs/a3_improvements_pilot_v1.yaml"
    verify_pinned_archives()  # Missing archives fail BEFORE fetch_dependencies can download.
    if not args.resume_archive:
        run_command("extract_pinned_dependencies", [sys.executable, ROOT / "tools/fetch_dependencies.py"], output, began)
        run_command("build_pinned_dependencies", [sys.executable, ROOT / "tools/build_dependencies.py", "--jobs", "2",
                    "--log-dir", output / "setup/dependency_commands"], output, began)
        run_command("configure_release", project_configure(build, shutil.which("g++")), output, began)
        run_command("build_release", ["cmake", "--build", build, "--parallel", "2"], output, began)
        run_command("ctest", ["ctest", "--test-dir", build, "--output-on-failure", "-j", "1"],
                    output, began, env=worker_env())
        verify_source(args.input_archive, args.input_sha256)
        versions = observed_dependency_versions()
        gate = {"schema_version": 1, "status": "PASS", "ctest_passed": True, "utc": utc(),
                "code_sha256": cloud_code_hash(), "worker_sha256": sha256(worker),
                "ompl_library_sha256": sha256(cloud_ompl_library(worker)), "dependency_versions": versions,
                "dependency_archives": verify_pinned_archives()}
        write(gate_path, gate)
    else:
        if not gate_path.is_file() or not worker.is_file():
            raise ValueError("resume requires the exact previously built worker and CTest gate; rebuilding forbidden")
        observed_dependency_versions()
    environment = cloud_environment(worker)
    runtime = {"python_version": platform.python_version(), "python_full": sys.version,
               "environment": environment, "runtime_identity": runtime_identity(environment),
               "runtime_identity_sha256": digest(runtime_identity(environment)),
               "input_sha256": args.input_sha256, "image_evidence": image_evidence()}
    if args.resume_archive:
        prior = read(output / "setup/runtime.json")
        if prior["runtime_identity_sha256"] != runtime["runtime_identity_sha256"]:
            raise ValueError("resume runtime changed; exact worker/libraries/Python/packages required")
        write(output / "setup/runtime_sessions" / (unique_id() + ".json"), runtime)
    else:
        write(output / "setup/runtime.json", runtime)
    for profile in ("smoke", "a3_pilot"):
        if profile == "a3_pilot" and not args.allow_a3_pilot:
            return {"status": "SMOKE_COMPLETE_PILOT_NOT_RUN", "pilot_status": "NOT_RUN"}
        data = output / "data" / profile
        campaign = output / ("smoke" if profile == "smoke" else "pilot")
        if not (data / "manifest.json").is_file():
            run_command("generate_" + profile, [sys.executable, ROOT / "tools/generate_scenarios.py",
                        "--config", config, "--profile", profile, "--output", data], output, began)
        base = [sys.executable, ROOT / "tools/a3_pilot_campaign.py", "--config", config,
                "--map-manifest", data / "manifest.json", "--dataset-root", data,
                "--output", campaign, "--worker", worker, "--profile", profile,
                "--build-gate", gate_path]
        if profile == "a3_pilot":
            base += ["--allow-a3-pilot", "--smoke-gate", output / "smoke/smoke_gate.json"]
            if not (campaign / "manifest.json").exists():
                run_command("lock_780_trial_plan", [*base, "--dry-run"], output, began, env=worker_env())
            export_checkpoint(output, args.input_archive, args.input_sha256, "before_pilot")
        already_complete = ((campaign / "campaign_state.json").is_file() and
                            read(campaign / "campaign_state.json")["status"] == "COMPLETE_VALIDATED")
        if not already_complete:
            command = [*base, "--session-elapsed-s", str(time.monotonic() - began)]
            if (campaign / "manifest.json").exists():
                command.append("--resume")
            run_command("collect_" + profile, command, output, began, env=worker_env())
        state = read(campaign / "campaign_state.json")
        if profile == "smoke" and (state["status"] != "COMPLETE_VALIDATED" or
                                    not (campaign / "smoke_gate.json").is_file()):
            raise RuntimeError("FAILED: 52-trial smoke correctness gate did not complete")
        if profile == "a3_pilot":
            run_command("analyze_pilot", [sys.executable, ROOT / "tools/analyze_a3_pilot.py",
                        "--input", campaign, "--output", output / "report"], output, began)
            analysis_status = read(output / "report/summary.json")["status"]
            complete = state["status"] == analysis_status == "COMPLETE_VALIDATED"
            failed = any(value.startswith("FAILED") or value == "STOPPED_FAILURE"
                         for value in (state["status"], analysis_status))
            return {"status": "COMPLETE" if complete else "FAILED" if failed else "PARTIAL",
                    "pilot_status": state["status"], "analysis_status": analysis_status,
                    "counts": state["counts"]}
    raise RuntimeError("unexpected orchestration end")


def main(argv=None):
    global EXPORT_INPUT_BYTES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("bootstrap", "execute"), default="bootstrap")
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--notebook-started-monotonic", type=float, required=True)
    parser.add_argument("--allow-a3-pilot", action="store_true")
    parser.add_argument("--resume-archive", type=Path)
    args = parser.parse_args(argv)
    EXPORT_INPUT_BYTES = file_size(args.input_archive)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    status = {"status": "RUNNING", "pilot_status": "NOT_RUN", "started_utc": utc(), "stage": args.stage}
    if not math.isfinite(args.notebook_started_monotonic) or not 0 <= args.notebook_started_monotonic <= time.monotonic():
        parser.error("invalid notebook monotonic start")
    # Bootstrap restores the prior RUN_STATUS before writing this session's state.
    if args.stage == "execute" or not args.resume_archive:
        write(args.output / "RUN_STATUS.json", status)
    try:
        result = bootstrap(args) if args.stage == "bootstrap" else execute(args)
        status.update(result or {})
    except BaseException as error:
        message = str(error)
        status.update(status="BLOCKED" if "BLOCKED:" in message else "PARTIAL" if "PARTIAL:" in message else "FAILED",
                      error_type=type(error).__name__, error=message)
        print(message, file=sys.stderr, flush=True)
    finally:
        pilot_state = args.output / "pilot/campaign_state.json"
        if pilot_state.is_file():
            observed = read(pilot_state)
            status.update(pilot_status=observed["status"], counts=observed["counts"])
        if (args.stage == "execute" and (args.output / "pilot/manifest.json").is_file() and
                status["status"] in {"FAILED", "PARTIAL", "BLOCKED"} and not status.get("analysis_status")):
            try:
                run_command("analyze_partial_pilot", [sys.executable, ROOT / "tools/analyze_a3_pilot.py",
                    "--input", args.output / "pilot", "--output", args.output / "report"],
                    args.output, args.notebook_started_monotonic, max_seconds=600, reserve_seconds=300)
            except BaseException as error:
                status["analysis_error"] = str(error)
        status["finished_utc"] = utc()
        status["notebook_elapsed_s"] = time.monotonic() - args.notebook_started_monotonic
        write(args.output / "RUN_STATUS.json", status)
        try:
            checkpoint = export_checkpoint(args.output, args.input_archive, args.input_sha256, status["status"])
            status["checkpoint"] = str(checkpoint)
        except BaseException as error:
            status["export_error"] = str(error)
            if status["status"] in {"COMPLETE", "SMOKE_COMPLETE_PILOT_NOT_RUN"}:
                status["status"] = "PARTIAL_EXPORT_FAILED"
            print("Checkpoint export failed; raw output remains: " + str(error), file=sys.stderr)
        write(args.output / "RUN_STATUS.json", status)
    return 0 if status["status"] in {"COMPLETE", "PARTIAL", "SMOKE_COMPLETE_PILOT_NOT_RUN"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
