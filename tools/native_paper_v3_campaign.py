"""Prepare, freeze and gate schema 12 without an SGP/GP runtime."""
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

from experiment_common import dump,digest
from native_paper_v3_protocol import ROOT,SCHEMA_VERSION,PROTOCOL_ID,load_config,source_identity,sha256,generate_dataset,plan_trials,BACKENDS
from native_paper_v3_analysis import validate_campaign,analyze_campaign,raw_hashes,validation_code_identity
from terrain_compare_campaign import environment_identity,binary_paths,show_correctness_failure
from terrain_compare_scheduler import run_campaign,collect_reservations,read_json,GIB
from terrain_compare_runtime import capture_native_dependencies
from terrain_compare_receipt import run as receipt

MODES=("smoke-serial","smoke-parallel","main")


def prepare(workspace,build,sem_python,profile=None):
    workspace,build=(Path(p).resolve() for p in (workspace,build))
    sem_python=Path(os.path.abspath(sem_python)); controller=Path(os.path.abspath(sys.executable))
    workspace.mkdir(parents=True,exist_ok=True)
    if (workspace/"prepared.json").exists():raise ValueError("already prepared; resume unchanged or choose a new workspace")
    binary,library=binary_paths(build)
    for path in (binary,library,sem_python):
        if not path.is_file():raise ValueError("missing built dependency: "+str(path))
    envs={"controller":environment_identity(controller),"sem":environment_identity(sem_python)}
    if not envs["controller"]["python"].startswith("3.12.") or not envs["sem"]["python"].startswith("3.12."):
        raise ValueError("native paper protocol requires controller Python 3.12 and SEM Python 3.12")
    identities=source_identity(); roots=[str(p) for p in (binary,library,controller,sem_python)]
    native=capture_native_dependencies(roots); hashes={str(p):sha256(p) for p in (binary,library,controller,sem_python)}
    hashes.update(native["binary_hashes"])
    identities.update(binary_hashes=hashes,environments=envs,native_linkage=native["linkage"],native_roots=roots)
    config=load_config(ROOT/"configs/native_paper_kaggle_v3.yaml")
    registry={"a3":{"argv":[str(binary)],"cwd":str(ROOT)}}
    for backend in ("sem","putn"):
        registry[backend]={"argv":[str(sem_python if backend=="sem" else controller),str(ROOT/"tools/native_paper_worker.py")],"cwd":str(ROOT)}
    dump(workspace/"registry.json",registry)
    profiles=[profile] if profile else ["smoke","main"]
    for name in profiles:
        directory=workspace/"data"/name; dataset=generate_dataset(config,name,directory)
        plan=plan_trials(config,name,dataset,directory,identities)
        for entry in plan["entries"]:
            if entry["backend"]!="a3":
                entry["request"]["bridge_library"]=str(library)
                if os.name=="nt":entry["request"]["library_search_paths"]=[str(build),"C:/mingw64/bin"]
        dump(workspace/"plans"/(name+".json"),plan)
    frozen={str(p.relative_to(workspace)):sha256(p) for folder in ("plans","data") for p in sorted((workspace/folder).rglob("*")) if p.is_file()}
    dump(workspace/"prepared.json",dict(schema_version=SCHEMA_VERSION,protocol_id=PROTOCOL_ID,build_dir=str(build),sem_python=str(sem_python),
        controller_python=str(controller),source_root=str(ROOT),identity=identities,registry_sha256=sha256(workspace/"registry.json"),frozen_inputs=frozen,profiles=profiles))
    return {"workspace":str(workspace),"status":"PREPARED"}



def check_frozen(workspace):
    workspace=Path(workspace)
    prepared=read_json(workspace/"prepared.json")
    if prepared.get("schema_version")!=SCHEMA_VERSION or prepared.get("protocol_id")!=PROTOCOL_ID:
        raise ValueError("not a schema-12 prepared workspace")
    current=source_identity()
    if current["source_sha256"]!=prepared["identity"]["source_sha256"]:
        raise ValueError("source/hash mismatch; choose a new workspace")
    for path,expected in prepared["identity"]["binary_hashes"].items():
        if not Path(path).is_file() or sha256(path)!=expected:
            raise ValueError("binary/interpreter hash mismatch: "+path)
    native=capture_native_dependencies(prepared["identity"]["native_roots"])
    if native["linkage"]!=prepared["identity"]["native_linkage"]:
        raise ValueError("native dependency resolution changed")
    for key,python in (("controller",prepared["controller_python"]),("sem",prepared["sem_python"])):
        if environment_identity(python)!=prepared["identity"]["environments"][key]:
            raise ValueError("environment identity changed: "+key)
    if sha256(workspace/"registry.json")!=prepared["registry_sha256"]:
        raise ValueError("registry changed")
    for relative,expected in prepared["frozen_inputs"].items():
        if not (workspace/relative).is_file() or sha256(workspace/relative)!=expected:
            raise ValueError("plan/map/metadata hash mismatch: "+relative)
    return prepared


REQUIRED_CTESTS=("terrain_unit","traversability_unit","flat_direct_certificate","a3_preprocessing",
    "independent_geometry","traversability_independent_oracle","ompl_policies",
    "traversability_ompl_policies","traversability_worker","a3_perf_worker","a3_sampler_cache",
    "a3_optimization_regression","putn_static","putn_paper","terrain_capi","sem_native_planner",
    "terrain_compare_runtime","terrain_compare_fixed_work_runtime","terrain_compare_scheduler",
    "native_session_policy","native_paper_oracle")


def correctness_commands(prepared,worker,library):
    # This explicit gate does not discover-and-skip legacy GP tests.
    expression="^("+"|".join(REQUIRED_CTESTS)+")$"
    listed=subprocess.run(["ctest","--test-dir",prepared["build_dir"],"--show-only=json-v1","-R",expression],
        check=True,capture_output=True,text=True)
    names={test["name"] for test in json.loads(listed.stdout)["tests"]}
    if names!=set(REQUIRED_CTESTS):
        raise ValueError("missing required CTest coverage: "+str(sorted(set(REQUIRED_CTESTS)-names)))
    commands=[(["ctest","--test-dir",prepared["build_dir"],"--output-on-failure","--parallel","1","-R",expression],"correctness_ctest")]
    for name in ("native_paper_v3_protocol","native_paper_v3_campaign","native_paper_v3_analysis","shared_sem_evaluator","native_paper_v3_memory","native_paper_v3_runtime","kaggle_native_paper_v3","native_paper_worker"):
        commands.append(([prepared["controller_python"],str(ROOT/"tests"/("test_"+name+".py"))],"correctness_"+name))
    commands.append(([prepared["controller_python"],str(ROOT/"tests/test_native_paper_v3_fixed_work.py"),
        "--worker",str(worker),"--library",str(library),"--sem-python",prepared["sem_python"]],"correctness_native_fixed_work"))
    return commands

def correctness(workspace):
    prepared=check_frozen(workspace); log=workspace/"logs";log.mkdir(exist_ok=True)
    temporary=workspace/"tmp";temporary.mkdir(exist_ok=True)
    env=os.environ.copy();env.update(TEMP=str(temporary),TMP=str(temporary),OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    if os.name=="nt":env["PATH"]="C:/mingw64/bin;"+str(ROOT/".deps/install/lib")+";"+env.get("PATH","")
    worker,library=binary_paths(Path(prepared["build_dir"]))
    commands=correctness_commands(prepared,worker,library)
    results=[]
    for command,name in commands:
        result=receipt(command,log,name,cwd=ROOT,env=env); results.append(result)
        stderr=(log/(name+".stderr.log")).read_text(errors="replace")
        if result["exit_code"] or "skipped=" in stderr:
            show_correctness_failure(log,name,result,"failure or required test skipped");break
    passed=len(results)==len(commands) and all(r["exit_code"]==0 for r in results) and not any("skipped=" in (log/(name+".stderr.log")).read_text(errors="replace") for _,name in commands if (log/(name+".stderr.log")).exists())
    dump(workspace/"correctness.json",dict(status="PASS" if passed else "FAILED",identity_sha256=digest(prepared["identity"]),commands=results))
    if not passed:raise RuntimeError("correctness gate failed; inspect "+str(workspace/"correctness.json"))
    return {"status":"PASS"}


def require_validated(workspace,mode):
    directory=workspace/"campaigns"/mode;summary=read_json(directory/"validation_summary.json");manifest=read_json(directory/"manifest.json")
    if summary["manifest_sha256"]!=sha256(directory/"manifest.json") or summary.get("raw_hashes")!=raw_hashes(directory) or summary.get("validation_code_identity")!=validation_code_identity():
        raise ValueError("stale smoke validation")
    if summary["correctness_failures"] or manifest["status"]!="COLLECTED_VALIDATION_PENDING":raise ValueError("smoke incomplete or invalid")
    planned=read_json(workspace/"plans/smoke.json")["entries"]
    if len(manifest["entries"])!=len(planned) or any(e["execution_status"]!="COMPLETED" for e in manifest["entries"]):
        raise ValueError("every smoke worker must complete; crash or phase timeout cannot pass readiness")
    if any(r.get("native_status") in ("ERROR","MODEL_ERROR") for r in summary["rows"]):raise ValueError("native model error in smoke")
    if not any(e["family"]=="flat_40m" for e in manifest["entries"]):raise ValueError("40 m smoke absent")
    return manifest


def collect(workspace,mode,resume=False,force_local=False,allow_main=False,session_elapsed_s=0):
    prepared=check_frozen(workspace);gate=read_json(workspace/"correctness.json")
    if gate["status"]!="PASS" or gate["identity_sha256"]!=digest(prepared["identity"]):raise ValueError("correctness gate is not current")
    if mode=="main" and (not allow_main or force_local or not sys.platform.startswith("linux")):
        raise ValueError("main requires --allow-main on Linux; local execution is smoke only")
    if mode=="smoke-serial":reservations={b:8*GIB for b in BACKENDS}
    else:
        serial=require_validated(workspace,"smoke-serial");reservations=collect_reservations(serial)
        if set(reservations)!=set(BACKENDS):raise ValueError("missing measured backend peak RSS")
        if mode=="main":require_validated(workspace,"smoke-parallel")
    if os.name=="nt":os.environ["PATH"]="C:/mingw64/bin;"+str(ROOT/".deps/install/lib")+";"+os.environ.get("PATH","")
    return run_campaign(workspace/"plans"/("main.json" if mode=="main" else "smoke.json"),read_json(workspace/"registry.json"),workspace/"campaigns"/mode,
        max_workers=1 if mode=="smoke-serial" else 4,reservations=reservations,resume=resume,force_local=force_local,session_elapsed_s=session_elapsed_s)


def plan_summary(workspace=None, profile="main"):
    """Read a frozen plan, or show cardinality without executing any planner."""
    if workspace is not None:
        workspace=Path(workspace)
        prepared=check_frozen(workspace)
        plan=read_json(workspace/"plans"/(profile+".json"))
        entries=plan["entries"]
        return dict(protocol_id=PROTOCOL_ID,schema_version=SCHEMA_VERSION,dry_run=True,
            dataset_verified=True,maps=len({e["map_id"] for e in entries}),
            paired_blocks=len({(e["map_id"],e["planner_seed"]) for e in entries}),
            slots=len(entries),backends=list(BACKENDS),planning_budget_s=entries[0]["request"]["planning_budget_s"])
    config=load_config(ROOT/"configs/native_paper_kaggle_v3.yaml")
    settings=config["profiles"][profile]
    maps=len(settings["families"])*len(settings["map_seeds"])
    blocks=maps*len(settings["planner_seeds"])
    return dict(protocol_id=PROTOCOL_ID,schema_version=SCHEMA_VERSION,dry_run=True,
        dataset_verified=False,maps=maps,paired_blocks=blocks,slots=blocks*len(BACKENDS),
        backends=list(BACKENDS),planning_budget_s=settings["planning_budget_s"])


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    for name in ("prepare","correctness","collect","validate","report","plan","export"):
        parser=sub.add_parser(name);parser.add_argument("--workspace",type=Path,required=name!="plan")
        if name=="prepare":
            parser.add_argument("--build-dir",type=Path,required=True);parser.add_argument("--sem-python",type=Path,required=True);parser.add_argument("--profile",choices=("smoke","main"))
        if name in ("collect","validate","report"):parser.add_argument("--mode",choices=MODES,required=True)
        if name=="collect":
            parser.add_argument("--resume",action="store_true");parser.add_argument("--force-local",action="store_true");parser.add_argument("--allow-main",action="store_true");parser.add_argument("--session-elapsed-s",type=float,default=0)
        if name=="validate":parser.add_argument("--jobs",type=int,default=4)
        if name=="plan":
            parser.add_argument("--profile",choices=("smoke","main"),default="main")
            parser.add_argument("--mode",choices=MODES)
            parser.add_argument("--dry-run",action="store_true",help="This command is always read-only")
        if name=="export":
            parser.add_argument("--output",type=Path,required=True)
            parser.add_argument("--input-archive",type=Path,required=True,
                                help="The exact source bundle needed to restore this checkpoint")
    a=p.parse_args();w=a.workspace.resolve() if a.workspace else None
    if a.command=="prepare":result=prepare(w,a.build_dir,a.sem_python,a.profile)
    elif a.command=="correctness":result=correctness(w)
    elif a.command=="plan":result=plan_summary(w,("main" if a.mode=="main" else "smoke") if a.mode else a.profile)
    elif a.command=="export":
        from kaggle_native_paper_v3_bundle import export_checkpoint,KIND
        from kaggle_a3_bundle import verify
        check_frozen(w)
        source=verify(a.input_archive)
        if source["kind"]!=KIND:raise ValueError("not a schema-12 source bundle")
        for item in source["files"]:
            if sha256(ROOT/item["path"])!=item["sha256"]:
                raise ValueError("source bundle differs from current source: "+item["path"])
        result=export_checkpoint(w,a.output,input_archive=a.input_archive)
    elif a.command=="collect":result=collect(w,a.mode,a.resume,a.force_local,a.allow_main,a.session_elapsed_s);result={k:result[k] for k in ("status","counts")}
    elif a.command=="validate":
        check_frozen(w);result=validate_campaign(w/"campaigns"/a.mode,a.jobs);result={k:result[k] for k in ("slots","validated_success","correctness_failures")}
    else:
        check_frozen(w);result=analyze_campaign(w/"campaigns"/a.mode,w/"reports"/a.mode);result={k:result[k] for k in ("protocol_id","completion")}
    print(json.dumps(result))
    if result.get("correctness_failures") or result.get("status",result.get("completion")) in ("PARTIAL","FAILED_CORRECTNESS"):raise SystemExit(2)


if __name__=="__main__":main()
