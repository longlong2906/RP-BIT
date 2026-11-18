"""Linux Kaggle gate for the registered SEM-exposure tune/confirm pilot.

Source, runtime and trial identities stay frozen across sessions. This module
never uploads data and never launches a full/held-out campaign.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
import time

from kaggle_a3_bundle import digest, sha256, verify
import kaggle_native_paper_v4 as supervised
from kaggle_sem_exposure_pilot_bundle import (BUILD_DIR, VENV_DIR, KIND, PROTOCOL_ID,
                                             export_checkpoint)

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.12.7"
LOCK = "requirements-native-paper-v4.txt"
SOFT_SESSION_S = 28800
REQUIRED_CTESTS = (
    "terrain_unit", "independent_geometry", "traversability_unit", "traversability_independent_oracle",
    "ompl_policies", "traversability_ompl_policies", "traversability_worker", "a3_optimization_regression",
    "safe_length_unit", "safe_length_field", "safe_length_runner", "safe_length_oracle", "safe_length_ompl",
    "a3_sampler_cache", "sem_native_planner", "shared_sem_evaluator",
    "safe_length_v2_calibration", "safe_length_v2_protocol", "safe_length_v2_runner", "safe_length_v2_validation",
    "safe_length_v2_analysis", "safe_length_v2_oracle_roundoff", "safe_length_v2_worker",
    "corridor_v3_core", "corridor_v3_protocol", "corridor_v3_runner", "corridor_v3_validation",
    "corridor_v3_analysis", "corridor_v3_worker", "corridor_v4_protocol", "corridor_v4_validation",
    "corridor_v4_runner", "corridor_v4_fixed_work",
    "sem_exposure_ompl", "sem_exposure_protocol", "sem_exposure_runner", "sem_exposure_analysis",
    "sem_exposure_validation", "sem_exposure_worker",
)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    supervised.write(path, value)


def record_status(workspace, status, **extra):
    value = dict(protocol_id=PROTOCOL_ID, status=status, updated_utc=dt.datetime.now(dt.timezone.utc).isoformat(), **extra)
    write(Path(workspace)/"bootstrap_status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def command(name, argv, workspace, started, *, env=None):
    print(dt.datetime.now(dt.timezone.utc).isoformat(), name, "STARTED", flush=True)
    value = supervised.command(name, argv, workspace, started, cwd=ROOT, env=env)
    print(dt.datetime.now(dt.timezone.utc).isoformat(), name, "COMPLETED", flush=True)
    return value


def checked_test(name, argv, workspace, started, env):
    logs = Path(workspace)/"logs/bootstrap"
    previous = set(logs.glob(name+"_*.json"))
    output = command(name, argv, workspace, started, env=env)
    texts = [output]
    for receipt in set(logs.glob(name+"_*.json"))-previous:
        stderr = receipt.with_suffix(".stderr.log")
        if stderr.is_file():
            texts.append(stderr.read_text(encoding="utf-8", errors="replace"))
    if any(re.search(r"OK\s*\(skipped=|\*\*\*Skipped|\*\*\*Not Run", text) for text in texts):
        raise RuntimeError("Linux correctness gate cannot accept skipped required tests: "+name)
    return output


def verify_source(path, checksum):
    manifest = verify(path, checksum)
    if manifest["kind"] != KIND or manifest["metadata"].get("protocol_id") != PROTOCOL_ID:
        raise ValueError("wrong SEM-exposure pilot source bundle")
    for entry in manifest["files"]:
        if sha256(ROOT/entry["path"]) != entry["sha256"]:
            raise ValueError("source differs from the verified bundle: "+entry["path"])
    return manifest


def isolated_runtime(workspace, started, *, restored=False):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("PIP_") and key not in ("PYTHONHOME", "PYTHONPATH")}
    env.update(UV_PYTHON_INSTALL_DIR=str(ROOT/".managed-python"), UV_CACHE_DIR=str(ROOT/".uv-terrain-cache"),
        PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull, CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg",
        OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "OMP_THREAD_LIMIT"):
        env[key] = "1"
    # abspath, not resolve(): preserve the venv executable symlink at launch.
    python = Path(os.path.abspath(ROOT/VENV_DIR/"bin/python"))
    if not python.is_file():
        if restored:
            raise RuntimeError("BLOCKED: checkpoint lacks the exact Python runtime")
        uv = supervised.bootstrap_uv(workspace, started)
        command("python_install", [uv, "--no-config", "python", "install", PYTHON_VERSION], workspace, started, env=env)
        command("venv", [uv, "--no-config", "venv", "--managed-python", "--python", PYTHON_VERSION, ROOT/VENV_DIR], workspace, started, env=env)
        command("locked_packages", [uv, "--no-config", "pip", "sync", "--python", python,
            "--require-hashes", "--only-binary", ":all:", ROOT/LOCK], workspace, started, env=env)
    probe = command("python_probe", [python, "-c",
        "import json,sys,platform,importlib.metadata as m;print(json.dumps({'version':platform.python_version(),'full':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"], workspace, started, env=env)
    evidence = json.loads(probe)
    if evidence["version"] != PYTHON_VERSION:
        raise ValueError("Python version differs from the pinned pilot runtime")
    supervised.assert_locked_packages(evidence, ROOT/LOCK)
    saved = Path(workspace)/"setup/controller_runtime.json"
    if saved.exists() and read(saved) != evidence:
        raise ValueError("Python/package runtime drift")
    write(saved, evidence)
    env["LD_LIBRARY_PATH"] = str(ROOT/".deps/install/lib")+os.pathsep+env.get("LD_LIBRARY_PATH", "")
    return python, env


def resource_gate(workspace, started, python, env, label):
    observed = command("hardware_"+label, [python, "-c",
        "import sys,json;sys.path.insert(0,'tools');from terrain_compare_scheduler import detect_runtime;print(json.dumps(detect_runtime()))"], workspace, started, env=env)
    runtime = json.loads(observed)
    receipt = dict(hardware=runtime, free_bytes=shutil.disk_usage(workspace).free,
                   required_cpu_capacity=4, minimum_memory_bytes=4*1024**3, minimum_free_bytes=6*1024**3)
    write(Path(workspace)/"setup"/("resource_"+label+".json"), receipt)
    if runtime["cpu_capacity"] < 4:
        raise RuntimeError("BLOCKED_RESOURCE: four effective CPUs required, including cgroup quota")
    if runtime["memory_limit_bytes"] < receipt["minimum_memory_bytes"]:
        raise RuntimeError("BLOCKED_RESOURCE: insufficient cgroup memory")
    if receipt["free_bytes"] < receipt["minimum_free_bytes"]:
        raise RuntimeError("BLOCKED_DISK: six GiB free required before stage admission")
    return runtime


def build(workspace, started, python, env):
    runtime = resource_gate(workspace, started, python, env, "build")
    missing = [name for name in ("g++", "gcc", "cmake", "ninja", "bash", "ldd") if not shutil.which(name)]
    if missing:
        raise RuntimeError("BLOCKED: missing build tools "+", ".join(missing))
    jobs = supervised.choose_build_jobs(runtime)
    write(Path(workspace)/"setup/build_runtime.json", dict(hardware=runtime, build_jobs=jobs))
    for name in ("g++", "cmake", "ninja"):
        command("version_"+name.replace("+", "x"), [name, "--version"], workspace, started)
    command("extract_pinned_sources", [python, ROOT/"tools/fetch_dependencies.py"], workspace, started, env=env)
    command("build_dependencies", [python, ROOT/"tools/build_dependencies.py", "--jobs", jobs,
        "--log-dir", Path(workspace)/"logs/dependencies"], workspace, started, env=env)
    boost = ROOT/".deps/boost_1_82_0"
    command("configure", ["cmake", "-S", ROOT, "-B", ROOT/BUILD_DIR, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_CXX_COMPILER="+shutil.which("g++"), f"-DPython3_EXECUTABLE={python}",
        f"-DCMAKE_PREFIX_PATH={ROOT/'.deps/install'}", f"-DBOOST_ROOT={boost}",
        f"-DBOOST_LIBRARYDIR={boost/'stage/lib'}", "-DBoost_NO_SYSTEM_PATHS=ON", "-DBoost_NO_BOOST_CMAKE=ON",
        "-DBoost_USE_STATIC_LIBS=ON", "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON", "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON"], workspace, started, env=env)
    command("build_release", ["cmake", "--build", ROOT/BUILD_DIR, "--parallel", jobs], workspace, started, env=env)


def runtime_identity(workspace, started, python, env, source_sha256, *, restored=False):
    roots = [str(ROOT/BUILD_DIR/"terrain_benchmark"), str(ROOT/BUILD_DIR/"libterrain_compare_c.so"), str(python)]
    code = ("import sys,json;sys.path.insert(0,'tools');from terrain_compare_runtime import capture_native_dependencies;"
            "from terrain_compare_campaign import environment_identity;"
            f"roots={roots!r};n=capture_native_dependencies(roots);"
            "print(json.dumps({'binary_hashes':n['binary_hashes'],'native_roots':roots,'native_linkage':n['linkage'],"
            "'environment':environment_identity(roots[2])}))")
    native = json.loads(command("runtime_identity", [python, "-c", code], workspace, started, env=env))
    value = dict(schema_version=1, protocol_id=PROTOCOL_ID, source_archive_sha256=source_sha256,
        worker_path=roots[0], bridge_path=roots[1], controller_python=roots[2], **native,
        requirements_sha256=sha256(ROOT/LOCK), build_directory=str(ROOT/BUILD_DIR),
        density_version="exact_canonical_raster_v1")
    value["identity_sha256"] = digest(value)
    path = Path(workspace)/"setup/runtime_identity.json"
    if restored and not path.exists():
        raise ValueError("restored campaign lacks its immutable runtime identity")
    if path.exists() and read(path) != value:
        raise ValueError("runtime/binary/linkage identity drift; do not rebuild this campaign")
    write(path, value)
    return path, value


def correctness(workspace, started, python, env, identity):
    regex = "^("+"|".join(REQUIRED_CTESTS)+")$"
    listing = json.loads(command("list_correctness", ["ctest", "--test-dir", ROOT/BUILD_DIR,
        "--show-only=json-v1", "-R", regex], workspace, started, env=env))
    if {item["name"] for item in listing["tests"]} != set(REQUIRED_CTESTS):
        raise ValueError("missing required scientific correctness tests")
    checked_test("correctness", ["ctest", "--test-dir", ROOT/BUILD_DIR, "--verbose", "--output-on-failure", "-j", "1", "-R", regex], workspace, started, env)
    for name in ("sem_exposure_pilot_analysis", "sem_exposure_runtime", "sem_exposure_pilot_runner",
                 "kaggle_sem_exposure_pilot"):
        checked_test(name, [python, ROOT/"tests"/("test_"+name+".py")], workspace, started, env)
    checked_test("pilot_fixed_work", [python, ROOT/"tests/test_sem_exposure_pilot_fixed_work.py",
        "--worker", ROOT/BUILD_DIR/"terrain_benchmark"], workspace, started, env)
    write(Path(workspace)/"correctness.json", dict(status="PASS", required_tests=list(REQUIRED_CTESTS),
        identity_sha256=identity["identity_sha256"], kaggle_gate_tests="PASS", linux=True))


def require_correctness(workspace, identity):
    gate = read(Path(workspace)/"correctness.json")
    if gate.get("status") != "PASS" or gate.get("identity_sha256") != identity["identity_sha256"] or gate.get("linux") is not True:
        raise ValueError("current Linux correctness gate required")


def require_no_experiment_evidence(workspace):
    # Bootstrap ordering writes no smoke/pilot files before correctness passes.
    # Conservatively refuse recovery once either experiment tree has content.
    for name in ("smoke", "pilot"):
        if any(path.is_file() or path.is_symlink() for path in (Path(workspace)/name).rglob("*")):
            raise ValueError("cannot rerun missing correctness gate after smoke/pilot evidence exists")


def smoke_gate(workspace, started, python, env, identity):
    output = Path(workspace)/"smoke"
    summary = read(output/"summary.json") if (output/"summary.json").is_file() else None
    if summary is None or summary.get("status") != "COMPLETE":
        resource_gate(workspace, started, python, env, "smoke")
        existing = (output/"manifest.json").is_file()
        argv = [python, ROOT/"tools/run_sem_exposure.py", "run" if existing else "smoke", "--output", output,
            "--worker", identity["worker_path"], "--bridge", identity["bridge_path"], "--sem-python", python,
            "--workers", "4", "--validation-workers", "4"]
        if existing:
            argv.append("--resume")
        command("smoke42", argv, workspace, started, env=env)
        summary = read(output/"summary.json")
    manifest = read(output/"manifest.json")
    if (summary.get("status") != "COMPLETE" or summary.get("planned") != 42 or summary.get("validated") != 42 or
        summary.get("correctness_failures") != 0 or len(manifest.get("entries", [])) != 42 or
        manifest.get("binary_sha256") != sha256(identity["worker_path"]) or manifest.get("bridge_sha256") != sha256(identity["bridge_path"])):
        raise ValueError("42-trial independent smoke gate failed")
    report = Path(workspace)/"reports/smoke"/dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    command("smoke_report", [python, ROOT/"tools/analyze_sem_exposure.py", "--input", output,
        "--output", report], workspace, started, env=env)
    evidence = dict(status="PASS", identity_sha256=identity["identity_sha256"], trials=42,
        manifest_sha256=sha256(output/"manifest.json"), summary_sha256=sha256(output/"summary.json"),
        report=str(report), files={p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()})
    write(Path(workspace)/"smoke_readiness.json", evidence)
    return evidence


def require_smoke(workspace, identity):
    evidence = read(Path(workspace)/"smoke_readiness.json")
    if (evidence.get("status") != "PASS" or evidence.get("identity_sha256") != identity["identity_sha256"] or
        evidence.get("manifest_sha256") != sha256(Path(workspace)/"smoke/manifest.json") or
        evidence.get("summary_sha256") != sha256(Path(workspace)/"smoke/summary.json")):
        raise ValueError("current smoke readiness evidence required")
    smoke = Path(workspace)/"smoke"
    if evidence.get("files") != {p.relative_to(smoke).as_posix(): sha256(p) for p in sorted(smoke.rglob("*")) if p.is_file()}:
        raise ValueError("smoke raw evidence changed after independent validation")


def pilot_phase(workspace, started, python, env, identity_path, identity, phase):
    output = Path(workspace)/"pilot"
    script = ROOT/"tools/run_sem_exposure_pilot.py"
    if phase == "confirm":
        selection = read(output/"selection.json")
        if selection.get("status") != "CANDIDATE_LOCKED":
            return record_status(workspace, selection.get("status", "NO_CANDIDATE"), phase="tune")
    manifest_path = output/phase/"manifest.json"
    if not manifest_path.exists():
        command("prepare_"+phase, [python, script, "prepare", "--output", output, "--phase", phase], workspace, started, env=env)
    if read(manifest_path).get("status") != "COMPLETE":
        resource_gate(workspace, started, python, env, phase)
        command("run_"+phase, [python, script, "run", "--output", output, "--phase", phase,
            "--allow-pilot", "--worker", identity["worker_path"], "--bridge", identity["bridge_path"],
            "--sem-python", python, "--runtime-identity", identity_path, "--resume",
            "--session-elapsed-s", max(0., time.monotonic()-started)], workspace, started, env=env)
    manifest = read(manifest_path)
    if manifest.get("status") != "COMPLETE":
        return record_status(workspace, manifest.get("status", "INCOMPLETE"), phase=phase)
    command("analyze_"+phase, [python, script, "analyze", "--output", output, "--phase", phase], workspace, started, env=env)
    if phase == "tune":
        selection = read(output/"selection.json")
        if selection.get("status") != "CANDIDATE_LOCKED":
            return record_status(workspace, selection.get("status", "NO_CANDIDATE"), phase="tune")
    return record_status(workspace, "COMPLETE", phase=phase)


def retain_checkpoint(workspace, source, *, error=None):
    name = "sem_exposure_pilot_"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+".payload"
    try:
        supervised.assert_workspace_quiescent(workspace)
        path = Path(workspace)/"exports"/name
        export_checkpoint(workspace, path, root=ROOT, input_archive=source)
        print("Checkpoint retained:", path, flush=True)
    except Exception as export_error:
        write(Path(workspace)/"logs/export_failure.json", dict(error=repr(export_error), stage_error=repr(error)))
        print("Checkpoint export failed; keep the current workspace:", repr(export_error), file=sys.stderr, flush=True)
        if error is None:
            raise


def execute(args):
    if not sys.platform.startswith("linux") or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("BLOCKED: this bootstrap requires Linux x86_64")
    if args.stage in ("all", "tune", "confirm") and not args.allow_pilot:
        raise ValueError("pilot stages require --allow-pilot")
    verify_source(args.input_archive, args.input_sha256)
    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    started = args.session_started_monotonic or time.monotonic()
    error = None
    try:
        if args.stage == "export":
            return record_status(workspace, "EXPORT_ONLY")
        retained_runtime = args.restored_runtime or (workspace/"setup/runtime_identity.json").exists()
        python, env = isolated_runtime(workspace, started, restored=retained_runtime)
        prepare = args.stage in ("prepare", "all") and not retained_runtime
        if prepare:
            build(workspace, started, python, env)
        identity_path, identity = runtime_identity(workspace, started, python, env, args.input_sha256,
                                                    restored=retained_runtime)
        recover_correctness = retained_runtime and not (workspace/"correctness.json").exists()
        if recover_correctness:
            require_no_experiment_evidence(workspace)
        if prepare or recover_correctness:
            correctness(workspace, started, python, env, identity)
            verify_source(args.input_archive, args.input_sha256)
            retain_checkpoint(workspace, args.input_archive)
        else:
            require_correctness(workspace, identity)
        if args.stage == "prepare":
            return record_status(workspace, "PREPARED")
        if args.stage in ("all", "smoke"):
            smoke_gate(workspace, started, python, env, identity)
            retain_checkpoint(workspace, args.input_archive)
        else:
            require_smoke(workspace, identity)
        if args.stage == "smoke":
            return record_status(workspace, "SMOKE_COMPLETE")
        phases = ("tune", "confirm") if args.stage == "all" else (args.stage,)
        if args.stage == "analyze":
            for phase in ("tune", "confirm"):
                if (workspace/"pilot"/phase/"manifest.json").exists():
                    command("analyze_"+phase, [python, ROOT/"tools/run_sem_exposure_pilot.py", "analyze",
                        "--output", workspace/"pilot", "--phase", phase], workspace, started, env=env)
            return record_status(workspace, "ANALYZED")
        for phase in phases:
            if time.monotonic()-started >= SOFT_SESSION_S:
                return record_status(workspace, "PAUSED_TIME", next_phase=phase)
            result = pilot_phase(workspace, started, python, env, identity_path, identity, phase)
            retain_checkpoint(workspace, args.input_archive)
            if result["status"] != "COMPLETE":
                return result
        return record_status(workspace, "COMPLETE")
    except BaseException as caught:
        error = caught
        record_status(workspace, "FAILED_OR_BLOCKED", error=repr(caught))
        raise
    finally:
        retain_checkpoint(workspace, args.input_archive, error=error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--input-archive", type=Path, required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--stage", choices=("prepare", "smoke", "tune", "confirm", "analyze", "export", "all"), default="prepare")
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--restored-runtime", action="store_true")
    parser.add_argument("--session-started-monotonic", type=float)
    args = parser.parse_args()
    print(json.dumps(execute(args), ensure_ascii=False))


if __name__ == "__main__":
    main()
