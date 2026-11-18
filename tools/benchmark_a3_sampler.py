"""Record fixed-work sampler throughput separately from certificate stopping."""
import argparse
import json
from pathlib import Path
import subprocess

from experiment_common import code_hash, dump, file_hash, worker_env


def verify_replay(document):
    if (document.get("kind") != "sampler_fixed_work_replay" or document.get("seed") != 11 or
            document.get("calls_per_block") != 10000 or document.get("clock_policy") != "steady_clock_per_block"):
        raise ValueError("unexpected replay protocol")
    cases={name:[] for name in ("lower_bound_guard","near_empty_flat","near_empty_hill","ellipse","full_domain")}
    for row in document["rows"]:
        if row["case_id"] not in cases or row["calls"] != 10000:
            raise ValueError("unexpected replay case/call count")
        cases[row["case_id"]].append(row)
    comparisons=[]
    for name,rows in cases.items():
        if len(rows)!=2 or {r["sampler_query_cache"] for r in rows}!={False,True}:
            raise ValueError("incomplete replay pair: "+name)
        first=next(r for r in rows if not r["sampler_query_cache"])
        second=next(r for r in rows if r["sampler_query_cache"])
        for key in ("trace_hash","calls","sample_proposals","samples_returned", "rejected_bounds",
                    "rejected_nodata","rejected_heuristic","exhausted_calls",
                    "sampler_invalid_interval_calls","sampler_lower_bound_rejected_calls","sampler_proposal_limit_calls"):
            if first[key]!=second[key]:raise ValueError("replay changed "+name+"/"+key)
        comparisons.append(dict(case_id=name,trace_equal=True,
            uncached_ns_per_call=first["elapsed_s"]*1e9/first["calls"],
            cached_ns_per_call=second["elapsed_s"]*1e9/second["calls"],
            cached_over_uncached=second["elapsed_s"]/first["elapsed_s"] if first["elapsed_s"]>0 else None,
            proposals=first["sample_proposals"],accepted=first["samples_returned"]))
    return comparisons


def run(worker,output):
    worker,output=Path(worker).resolve(),Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("use a new replay output directory")
    output.mkdir(parents=True,exist_ok=True)
    command=[str(worker),"--replay","10000"]
    identity={"source_sha256":code_hash(),"binary_sha256":file_hash(worker),"certificate_enabled":False}
    try:
        result=subprocess.run(command,capture_output=True,text=True,encoding="utf-8",errors="replace",env=worker_env(),timeout=60.)
    except (OSError,subprocess.TimeoutExpired) as error:
        summary=dict(identity=identity,command=command,exit_code=None,status="WATCHDOG" if isinstance(error,subprocess.TimeoutExpired) else "FAILED",error=str(error))
        for name,value in (("stdout.log",getattr(error,"stdout",None)),("stderr.log",getattr(error,"stderr",None))):
            (output/name).write_bytes(value.encode() if isinstance(value,str) else value or b"")
        dump(output/"summary.json",summary)
        return summary
    (output/"stdout.log").write_text(result.stdout,encoding="utf-8")
    (output/"stderr.log").write_text(result.stderr,encoding="utf-8")
    summary=dict(identity=identity,command=command,exit_code=result.returncode,status="FAILED")
    try:
        if result.returncode!=0:raise ValueError("nonzero replay exit")
        document=json.loads(result.stdout)
        dump(output/"measurements.json",document)
        summary["comparisons"]=verify_replay(document)
        if identity["source_sha256"]!=code_hash() or identity["binary_sha256"]!=file_hash(worker):
            raise ValueError("replay source/binary changed during measurement")
        summary["status"]="PASS"
    except (KeyError,ValueError,TypeError) as error:summary["error"]=str(error)
    dump(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args();result=run(args.worker,args.output)
    print(json.dumps(result,allow_nan=False))
    raise SystemExit(0 if result["status"]=="PASS" else 1)
