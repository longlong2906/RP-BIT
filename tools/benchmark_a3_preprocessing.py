"""Run exactly 54 application-cold field builds, with no planner execution."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import subprocess
import time

from a3_traversability_protocol import DEFAULT_TRAVERSABILITY, fixture_geometry
from experiment_common import code_hash, dump, file_hash, worker_env
from terrain_io import write_map

FAMILIES = ("flat", "two_corridors_slope", "nodata_thin_obstacle")
SPACINGS = (.4, .2, .1)
POLICIES = ("legacy", "skip_dominated")


def build_plan(output):
    """Only map generation; exactly three repetitions of each declared pair."""
    output = Path(output)
    entries = []
    for family in FAMILIES:
        height, rectangles, *_ = fixture_geometry(family)
        for spacing in SPACINGS:
            map_id = f"{family}_spacing_{spacing:g}"
            path = output/"maps"/(map_id+".t25d")
            size = round(20./spacing)+1
            meta = write_map(path, [[height(i*spacing,j*spacing) for i in range(size)] for j in range(size)],
                spacing=(spacing,spacing),rectangles=rectangles,
                metadata={"family":family,"kind":"field_build_microbenchmark","map_id":map_id,
                          "feature_version":"cell_envelope_v1"})
            for repetition in range(3):
                for policy in POLICIES:
                    entries.append(dict(run_id=f"{map_id}_{policy}_r{repetition}",family=family,
                        spacing_m=spacing,map_path=str(path.resolve()),map_sha256=meta["data_sha256"],
                        build_policy=policy,repetition=repetition))
    random.Random(73129).shuffle(entries)
    return entries


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["family"],row["spacing_m"])].append(row)
    comparisons = []
    for (family,spacing), group in groups.items():
        valid = [r for r in group if r["status"] == "COMPLETED"]
        same = (len(valid)==6 and len({r["measurement"]["field_fingerprint"] for r in valid})==1)
        comparisons.append(dict(family=family,spacing_m=spacing,completed=len(valid),
            field_fingerprints_match=same,
            fingerprint_scope="regression checksum, not a proof of equality; correctness suite compares every cell bit",
            rows=group))
    return {"planned":54,"executed":len(rows),"completed":sum(r["status"]=="COMPLETED" for r in rows),
            "failures":sum(r["status"]!="COMPLETED" for r in rows),"comparisons":comparisons,
            "all_field_fingerprints_match":len(comparisons)==9 and all(c["field_fingerprints_match"] for c in comparisons)}


def run(worker, output, watchdog_s=60.):
    worker,output=Path(worker).resolve(),Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new microbenchmark output directory; no selective retry/resume")
    output.mkdir(parents=True,exist_ok=True)
    identity={"source_sha256":code_hash(),"binary_sha256":file_hash(worker),
              "traversability":DEFAULT_TRAVERSABILITY,"protocol":"a3_field_build_54_v1",
              "clock_scope":"worker field construction only; each build starts a fresh process",
              "cold_scope":"application_cold; OS filesystem page cache not controlled"}
    entries=build_plan(output)
    dump(output/"manifest.json",dict(identity=identity,planned=54,runs=entries))
    rows=[]
    for entry in entries:
        folder=output/"runs"/entry["run_id"];folder.mkdir(parents=True)
        command=[str(worker),"--map",entry["map_path"],"--policy",entry["build_policy"],
                 "--lambda","1","--slope-weight",".5","--clearance-weight",".5","--clearance-comfort",".4"]
        row=dict(entry,command=command,status="NOT_RUN",exit_code=None,measurement=None)
        began=time.perf_counter()
        try:
            result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",
                                  env=worker_env(),timeout=watchdog_s)
            row.update(exit_code=result.returncode,status="COMPLETED" if result.returncode==0 else "CRASH")
            (folder/"stdout.log").write_text(result.stdout,encoding="utf-8")
            (folder/"stderr.log").write_text(result.stderr,encoding="utf-8")
            if result.returncode==0:
                measurement=json.loads(result.stdout,parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
                if measurement.get("build_policy")!=entry["build_policy"] or not measurement.get("field_fingerprint"):
                    raise ValueError("worker measurement policy/fingerprint missing")
                row["measurement"]=measurement
        except subprocess.TimeoutExpired as error:
            row.update(status="WATCHDOG",error=str(error))
            for name,value in (("stdout.log",error.stdout),("stderr.log",error.stderr)):
                (folder/name).write_bytes(value.encode() if isinstance(value,str) else value or b"")
        except (OSError,ValueError) as error:
            row.update(status="FAILED",error=repr(error))
        row["process_wall_s"]=time.perf_counter()-began
        rows.append(row);dump(folder/"measurement.json",row)
        print(f"{len(rows)}/54 {entry['run_id']}: {row['status']}",flush=True)
    summary=summarize(rows)
    summary["identity"]=identity
    summary["identity_unchanged"]=(identity["source_sha256"]==code_hash() and identity["binary_sha256"]==file_hash(worker))
    dump(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--watchdog-s",type=float,default=60.)
    args=parser.parse_args()
    result=run(args.worker,args.output,args.watchdog_s)
    raise SystemExit(0 if result["completed"]==54 and result["all_field_fingerprints_match"] and result["identity_unchanged"] else 1)
