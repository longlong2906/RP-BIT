"""Prepare, gate and collect schema-10 campaigns. Main collection is opt-in on Linux."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiment_common import dump, digest
from terrain_compare_protocol import ROOT, load_config, source_identity, sha256, generate_dataset, plan_trials
from terrain_compare_scheduler import run_campaign, collect_reservations, read_json, atomic_json, GIB
from terrain_compare_analysis import validate_campaign, analyze_campaign, raw_hashes
from terrain_compare_receipt import run as receipt
from terrain_compare_runtime import capture_native_dependencies

MODES = ("smoke-serial", "smoke-parallel", "main")


def environment_identity(python):
    code = "import sys,json,importlib.metadata as m;print(json.dumps({'python':sys.version,'packages':sorted((d.metadata['Name'],d.version) for d in m.distributions())}))"
    result = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def binary_paths(build):
    suffix = ".exe" if os.name == "nt" else ""
    library = build/("libterrain_compare_c.dll" if os.name == "nt" else "libterrain_compare_c.so")
    if os.name == "nt" and not library.exists(): library = build/"terrain_compare_c.dll"
    return build/("terrain_benchmark"+suffix), library


def prepare(workspace, build, sgp_python, profile=None):
    workspace, build = (Path(p).resolve() for p in (workspace, build))
    # Invoking the venv symlink is essential: resolving it invokes base Python.
    sgp_python = Path(os.path.abspath(sgp_python))
    workspace.mkdir(parents=True, exist_ok=True)
    if (workspace/"prepared.json").exists():
        raise ValueError("already prepared; use existing workspace unchanged or a new directory")
    binary, library = binary_paths(build)
    for path in (binary, library, sgp_python):
        if not path.is_file(): raise ValueError("missing built dependency: "+str(path))
    controller = Path(os.path.abspath(sys.executable))
    envs = {"controller": environment_identity(controller), "sgp": environment_identity(sgp_python)}
    if not envs["controller"]["python"].startswith("3.12.") or not envs["sgp"]["python"].startswith("3.10."):
        raise ValueError("protocol needs controller Python 3.12 and SGP Python 3.10")
    identities = source_identity()
    files = {str(p): sha256(p) for p in (binary, library, controller, sgp_python)}
    native_roots=[str(p) for p in (binary,library,controller,sgp_python)]
    native=capture_native_dependencies(native_roots)
    files.update(native["binary_hashes"])
    identities.update(binary_hashes=files, environments=envs, native_linkage=native["linkage"], native_roots=native_roots)
    config = load_config(ROOT/"configs/terrain_compare_kaggle_v1.yaml")
    registry = {b: {"argv": [str(binary)], "cwd": str(ROOT)} for b in ("a3", "sem")}
    for backend in ("putn", "sgp"):
        registry[backend] = {"argv": [str(sgp_python if backend=="sgp" else controller),
                                     str(ROOT/"tools/terrain_compare_external_worker.py")], "cwd": str(ROOT)}
    dump(workspace/"registry.json", registry)
    for name in ([profile] if profile else ["smoke", "main"]):
        directory = workspace/"data"/name
        dataset = generate_dataset(config, name, directory)
        plan = plan_trials(config, name, dataset, directory, identities)
        for entry in plan["entries"]:
            if entry["backend"] in ("putn", "sgp"):
                entry["request"]["bridge_library"] = str(library)
                if os.name=="nt": entry["request"]["library_search_paths"] = [str(build), "C:/mingw64/bin"]
        dump(workspace/"plans"/(name+".json"), plan)
    frozen={str(p.relative_to(workspace)):sha256(p) for folder in ("plans","data") for p in sorted((workspace/folder).rglob("*")) if p.is_file()}
    dump(workspace/"prepared.json", dict(schema_version=10, build_dir=str(build), sgp_python=str(sgp_python),
         controller_python=str(controller), source_root=str(ROOT), identity=identities,
         registry_sha256=sha256(workspace/"registry.json"), frozen_inputs=frozen, profiles=[profile] if profile else ["smoke", "main"]))
    return {"workspace": str(workspace), "status": "PREPARED"}


def check_frozen(workspace):
    prepared = read_json(workspace/"prepared.json")
    current = source_identity()
    if current["source_sha256"] != prepared["identity"]["source_sha256"]:
        raise ValueError("source/hash mismatch; use a new workspace and rerun correctness/smoke")
    for path, expected in prepared["identity"]["binary_hashes"].items():
        if not Path(path).is_file() or sha256(path)!=expected: raise ValueError("binary/interpreter hash mismatch: "+path)
    native=capture_native_dependencies(prepared["identity"]["native_roots"])
    if native["linkage"]!=prepared["identity"]["native_linkage"]:
        raise ValueError("native dependency resolution changed")
    for key, python in (("controller", prepared["controller_python"]), ("sgp", prepared["sgp_python"])):
        if environment_identity(python)!=prepared["identity"]["environments"][key]: raise ValueError("environment identity changed")
    if sha256(workspace/"registry.json") != prepared["registry_sha256"]: raise ValueError("registry changed")
    for relative,expected in prepared["frozen_inputs"].items():
        if not (workspace/relative).is_file() or sha256(workspace/relative)!=expected:
            raise ValueError("plan/map/metadata hash mismatch: "+relative)
    return prepared


def show_correctness_failure(log, name, result, reason, tail_bytes=6000):
    """Keep nested test failures visible within the bootstrap's bounded stderr tail."""
    stem = log/name
    print(f"Correctness failure: {name}; reason={reason}; exit_code={result.get('exit_code')}; "
          f"receipt={stem.with_suffix('.json')}; argv={json.dumps(result.get('argv', []))}",
          file=sys.stderr, flush=True)
    for suffix in (".stdout.log", ".stderr.log"):
        path = stem.with_suffix(suffix)
        try:
            with path.open("rb") as stream:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size-tail_bytes))
                tail = stream.read(tail_bytes).decode("utf-8", errors="replace")
            print(f"--- {path} (last {tail_bytes} bytes at most) ---\n{tail}",
                  file=sys.stderr, flush=True)
        except OSError as error:
            # A diagnostic read failure must not replace the test failure.
            print(f"Cannot read correctness diagnostic {path}: {error}", file=sys.stderr, flush=True)


def correctness(workspace):
    prepared = check_frozen(workspace)
    log = workspace/"logs"; log.mkdir(exist_ok=True)
    env = os.environ.copy()
    temporary = workspace/"tmp"; temporary.mkdir(exist_ok=True)
    env.update(TEMP=str(temporary), TMP=str(temporary), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    if os.name=="nt":env["PATH"]="C:/mingw64/bin;"+str(ROOT/".deps/install/lib")+";"+env.get("PATH","")
    worker,library=binary_paths(Path(prepared["build_dir"]))
    commands = [(["ctest", "--test-dir", prepared["build_dir"], "--output-on-failure", "--parallel", "1"], "correctness_ctest"),
                ([prepared["sgp_python"], str(ROOT/"tests/test_sgp_static.py")], "correctness_sgp"),
                ([prepared["controller_python"], str(ROOT/"tests/test_terrain_compare_fixed_work.py"),
                  "--worker", str(worker), "--library",str(library),"--sgp-python",prepared["sgp_python"]],"correctness_fixed_work")]
    results = []
    for command, name in commands:
        result = receipt(command, log, name, cwd=ROOT, env=env)
        results.append(result)
        if result["exit_code"]:
            show_correctness_failure(log, name, result, "nonzero test exit")
            break
    passed = len(results)==len(commands) and all(r["exit_code"]==0 for r in results)
    # A missing optional GP dependency is not an acceptable Kaggle readiness test.
    sgp_log = log/"correctness_sgp.stderr.log"
    if sgp_log.exists() and "skipped=" in sgp_log.read_text(errors="replace"):
        passed=False
        sgp_result = next((result for (_, name), result in zip(commands, results)
                           if name=="correctness_sgp"), {})
        show_correctness_failure(log, "correctness_sgp", sgp_result,
                                 "SGP tests skipped; required GP dependency coverage is missing")
    dump(workspace/"correctness.json", {"status": "PASS" if passed else "FAILED",
         "identity_sha256": digest(prepared["identity"]), "commands": results})
    if not passed: raise RuntimeError(f"correctness gate failed; inspect {workspace/'correctness.json'} and {log}")
    return {"status": "PASS"}


def require_validated(workspace, mode):
    directory=workspace/"campaigns"/mode
    summary=read_json(directory/"validation_summary.json")
    manifest=read_json(directory/"manifest.json")
    if summary["manifest_sha256"]!=sha256(directory/"manifest.json"): raise ValueError("stale smoke validation")
    if summary.get("raw_hashes")!=raw_hashes(directory):raise ValueError("stale raw smoke artifacts")
    if summary["correctness_failures"] or manifest["status"]!="COLLECTED_VALIDATION_PENDING":
        raise ValueError("smoke incomplete/invalid")
    if sum(e["execution_status"]=="COMPLETED" for e in manifest["entries"])!=42:
        raise ValueError("all 42 smoke workers must complete; no crash/resource/phase timeout may pass readiness")
    if any(r.get("native_status") in ("ERROR", "MODEL_ERROR") for r in summary["rows"]):
        raise ValueError("backend model error in smoke")
    return manifest


def collect(workspace, mode, resume=False, force_local=False, allow_main=False, session_elapsed_s=0):
    prepared=check_frozen(workspace)
    gate=read_json(workspace/"correctness.json")
    if gate["status"]!="PASS" or gate["identity_sha256"]!=digest(prepared["identity"]): raise ValueError("correctness not current")
    if mode=="main" and (not allow_main or force_local or not sys.platform.startswith("linux")):
        raise ValueError("main requires --allow-main on Linux; local Windows trials are smoke only")
    if mode=="smoke-serial":
        # Unknown backend peaks: one conservative 8 GiB reservation, never four guesses.
        reservations={b:8*GIB for b in ("a3","sem","putn","sgp")}
    else:
        serial=require_validated(workspace,"smoke-serial")
        reservations=collect_reservations(serial)
        if set(reservations)!={"a3","sem","putn","sgp"}: raise ValueError("missing measured RSS backend")
        if mode=="main": require_validated(workspace,"smoke-parallel")
    if os.name=="nt":os.environ["PATH"]="C:/mingw64/bin;"+str(ROOT/".deps/install/lib")+";"+os.environ.get("PATH","")
    return run_campaign(workspace/"plans"/("main.json" if mode=="main" else "smoke.json"),
        read_json(workspace/"registry.json"), workspace/"campaigns"/mode,
        max_workers=1 if mode=="smoke-serial" else 4, reservations=reservations,
        resume=resume, force_local=force_local, session_elapsed_s=session_elapsed_s)


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    for cmd in ("prepare","correctness","run","validate"):
        parser=sub.add_parser(cmd); parser.add_argument("--workspace",type=Path,required=True)
        if cmd=="prepare":
            parser.add_argument("--build-dir",type=Path,required=True); parser.add_argument("--sgp-python",type=Path,required=True)
            parser.add_argument("--profile",choices=["smoke","main"])
        if cmd in ("run","validate"): parser.add_argument("--mode",choices=MODES,required=True)
        if cmd=="run":
            parser.add_argument("--resume",action="store_true"); parser.add_argument("--force-local",action="store_true")
            parser.add_argument("--allow-main",action="store_true"); parser.add_argument("--session-elapsed-s",type=float,default=0)
        if cmd=="validate": parser.add_argument("--jobs",type=int,default=4)
    a=p.parse_args(); w=a.workspace.resolve()
    if a.command=="prepare": result=prepare(w,a.build_dir,a.sgp_python,a.profile)
    elif a.command=="correctness":result=correctness(w)
    elif a.command=="run":
        result=collect(w,a.mode,a.resume,a.force_local,a.allow_main,a.session_elapsed_s)
        result={k:result[k] for k in ("status","counts")}
    else:
        check_frozen(w)
        result=validate_campaign(w/"campaigns"/a.mode,a.jobs)
        analyze_campaign(w/"campaigns"/a.mode,w/"reports"/a.mode)
        result={k:result[k] for k in ("slots","validated_success","correctness_failures")}
    print(json.dumps(result))
    if result.get("correctness_failures") or result.get("status") in ("PARTIAL","FAILED_CORRECTNESS"):raise SystemExit(2)


if __name__=="__main__":main()
