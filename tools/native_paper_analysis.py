"""Independent geometric validation and tables-only native-objective reporting.

There is no shared SEM risk evaluator. Only the SEM planner's own frozen raster
is read to audit its native path-cost accumulation.
"""
from __future__ import annotations
import argparse
import collections
import concurrent.futures
import csv
import datetime as dt
import json
import math
from pathlib import Path
import random
import statistics
import time

from experiment_common import digest,dump
from native_paper_oracle import NativePaperOracle as Oracle
from flat_direct_certificate import verify_claim
from a3_traversability_protocol import DEFAULT_TRAVERSABILITY
from native_paper_protocol import ROOT,PROTOCOL_ID,BACKENDS,LABELS,sha256

GEOMETRY_METRICS=("cost_terrain_m","length_xy_m","sharp_turn_count",
                  "minimum_footprint_clearance_m","minimum_obstacle_footprint_clearance_m",
                  "maximum_swept_footprint_slope_deg")
UNSTARTED={"PLANNED","NOT_RUN","BLOCKED","BLOCKED_SOURCE"}
DERIVED={"validation.json","validation_receipt.json","independent_validation.json"}
SGP_NUMERIC_POLICY=dict(version="sgp_cpu_ftz_daz_v1",torch_api="torch.set_flush_denormal(True)",
    enabled=True,scope="dedicated_sgp_worker_cpu_float32_float64",float32_tiny=1.1754943508222875e-38,
    float64_tiny=2.2250738585072014e-308,unsupported="fail_closed")


def read(path):return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite_tree(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:finite_tree(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [finite_tree(v) for v in value]
    return value


def csv_path(path):
    with Path(path).open(newline="",encoding="utf-8") as stream:
        return [[float(x) for x in row] for row in list(csv.reader(stream))[1:]]


def trial_inventory(directory):
    directory=Path(directory)
    found={str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob("*"))
           if p.is_file() and p.name not in DERIVED and not p.name.endswith(".tmp")}
    # Absence is part of identity: a later worker/path/snapshot invalidates cache.
    for name in ("request.json","worker.json","execution.json","launch.json","progress.jsonl","path_xy.csv"):
        found.setdefault(name,None)
    return found


def raw_hashes(root):
    root=Path(root); manifest=read(root/"manifest.json")
    return {e["trial_id"]:trial_inventory(root/e["trial_id"]) for e in manifest["entries"]}


def validation_code_identity():
    files=[Path(__file__),ROOT/"tools/independent_verifier.py",ROOT/"tools/native_paper_oracle.py",ROOT/"tools/flat_direct_certificate.py"]
    sem=ROOT/"tools/sem_native_planner.py"
    if sem.exists():files.append(sem)
    return digest({p.name:sha256(p) for p in files})


def sharp_turn_count(path,threshold_deg=30.):
    clean=[]
    for p in path:
        q=tuple(map(float,p))
        if not clean or q!=clean[-1]:clean.append(q)
    count=0
    for a,b,c in zip(clean,clean[1:],clean[2:]):
        u=(b[0]-a[0],b[1]-a[1]);v=(c[0]-b[0],c[1]-b[1])
        # atan2 is stable near straight, right-angle and reversal cases.
        angle=math.degrees(math.atan2(abs(u[0]*v[1]-u[1]*v[0]),u[0]*v[0]+u[1]*v[1]))
        if angle>threshold_deg+1e-12:count+=1
    return count


def close(a,b):return isinstance(a,(float,int)) and math.isfinite(a) and abs(a-b)<=1e-8*(1+abs(b))


def verify_native_artifacts(directory,expected):
    directory=Path(directory)
    actual={p.relative_to(directory).as_posix():sha256(p) for p in sorted(directory.rglob("*")) if p.is_file()}
    if not isinstance(expected,dict) or not expected or actual!=expected:
        raise ValueError("native_artifact_inventory_or_hash_mismatch")
    return actual


def sgp_numeric_policy_valid(metadata):
    policy=metadata.get("numeric_policy",{}) if isinstance(metadata,dict) else {}
    return (isinstance(metadata,dict) and metadata.get("cpu_flush_denormal") is True and
            isinstance(policy,dict) and policy.get("enabled") is True and
            all(policy.get(key)==value for key,value in SGP_NUMERIC_POLICY.items()))


def putn_path_cost(features):
    if not features:raise ValueError("missing PUTN per-vertex features")
    total=0.
    for a,b in zip(features,features[1:]):
        ta,tb=float(a["traversability"]),float(b["traversability"])
        if not 0<=ta<1 or not 0<=tb<1:raise ValueError("invalid PUTN traversability")
        ca,cb=a["center_of_mass"],b["center_of_mass"]
        if len(ca)!=3 or len(cb)!=3 or not all(math.isfinite(x) for x in (*ca,*cb)):
            raise ValueError("invalid PUTN COM")
        total+=math.dist(ca,cb)*(1+.1*(1/(1-ta)+1/(1-tb)-2))
    return total


def at_checkpoint(events,t):
    candidates=[e for e in events if e["independent_valid"] and e["elapsed_s"]<=t]
    return candidates[-1] if candidates else None


def _validate_group(task):
    root,entries,code_identity=task;root=Path(root);oracle=None;rows=[]
    for entry in entries:
        directory=root/entry["trial_id"]; inventory=trial_inventory(directory)
        request=entry["request"]; map_file=Path(request["map_path"])
        if sha256(map_file)!=entry["map_sha256"]:raise ValueError("raw map hash changed")
        if request.get("cloud_path") and sha256(request["cloud_path"])!=request["cloud_sha256"]:
            raise ValueError("raw cloud hash changed")
        cache_key=digest(dict(entry=entry,inventory=inventory,map_hash=entry["map_sha256"],code=code_identity))
        receipt_path=directory/"validation_receipt.json"
        if receipt_path.is_file() and (directory/"validation.json").is_file():
            try:
                cached=read(receipt_path)
                if cached["cache_key"]==cache_key and cached["verdict_sha256"]==sha256(directory/"validation.json"):
                    row=read(directory/"validation.json");row["validation_cache_hit"]=True;rows.append(row);continue
            except (ValueError,KeyError,OSError):pass
        began=time.perf_counter()
        row={k:entry.get(k) for k in ("trial_id","backend","track","family","map_id","query_id","planner_seed","execution_status","timing_stratum")}
        row.update(success=False,events=[],errors=[],first_solution_s=None,first_valid_latency_s=None,
                   budget_s=request["planning_budget_s"],native_status=None,verification_status="NO_WORKER",
                   validation_cache_hit=False,execution=read(directory/"execution.json") if (directory/"execution.json").exists() else {})
        if (directory/"worker.json").exists():
            actual_request=read(directory/"request.json")
            if digest(actual_request)!=digest(request):raise ValueError("raw request differs from frozen plan")
            worker=read(directory/"worker.json")
            if entry["backend"]=="sgp":
                metadata=worker.get("native_model",{})
                status=worker.get("planner_status",worker.get("status"))
                # Construction failures cannot export a successfully built model.
                # Completed models must prove the declared process FP policy,
                # including legitimate timeout/invalid-native-endpoint attempts.
                if (metadata or status not in ("ERROR","MODEL_ERROR","BLOCKED_DEPENDENCY")) and not sgp_numeric_policy_valid(metadata):
                    row["errors"].append("sgp_cpu_numeric_policy_mismatch")
            if oracle is None:oracle=Oracle(map_file)
            source_events=worker.get("snapshots")
            if source_events is None:
                source_events=[json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()
                               if line.strip() and json.loads(line).get("event_type")=="improvement"]
            seen=set();previous_time=-1.;previous_cost=math.inf;certificate_paths={};sem_model=None
            for source in source_events:
                sid=source["snapshot_id"]
                if not isinstance(sid,str) or Path(sid).name!=sid or sid in seen:raise ValueError("unsafe/repeated snapshot id")
                seen.add(sid)
                path=source.get("path_xy")
                if path is None:path=csv_path(directory/"snapshots"/sid/"path_xy.csv")
                elapsed=source.get("elapsed_s",source.get("timestamp_s"))
                native=source.get("native_objective_value",source.get("planner_objective_value",source.get("objective_value")))
                if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed<0 or elapsed<previous_time:
                    raise ValueError("nonmonotone/nonfinite snapshot time")
                if not isinstance(native,(int,float)) or not math.isfinite(native) or native<0 or native>previous_cost+1e-8:
                    raise ValueError("nonmonotone/nonfinite native cost")
                checked=oracle.validate_path(path,request["start"],request["goal"],
                    traversability=DEFAULT_TRAVERSABILITY if entry["backend"]=="a3" else None,
                    geometric_safety_metrics=True)
                for key in ("cost_terrain_m","length_xy_m"):
                    if source.get(key) is not None and not close(source[key],checked[key]):checked["errors"].append(key+"_mismatch")
                try:
                    if entry["backend"]=="a3":native_recomputed=checked["objective_value"]
                    elif entry["backend"]=="sgp":native_recomputed=math.fsum(math.dist(a,b) for a,b in zip(path,path[1:]))
                    elif entry["backend"]=="putn":
                        features=source.get("putn_features")
                        if not features or len(features)!=len(path):raise ValueError("PUTN path/features mismatch")
                        native_recomputed=putn_path_cost(features)
                    elif entry["backend"]=="sem":
                        from sem_native_planner import SEMNativeModel,accumulate_native_path_cost
                        if sem_model is None:
                            verify_native_artifacts(directory/"native_model",worker.get("native_model",{}).get("raster_artifacts"))
                            sem_model=SEMNativeModel.from_artifacts(directory/"native_model")
                        native_recomputed=accumulate_native_path_cost(path,sem_model)
                        if isinstance(native_recomputed,dict):native_recomputed=native_recomputed["total"]
                    else:raise ValueError("unsupported native backend")
                    if not close(native,native_recomputed):checked["errors"].append("native_objective_mismatch")
                except (ValueError,KeyError,TypeError,OSError) as error:
                    native_recomputed=None;checked["errors"].append("native_cost_audit: "+str(error))
                checked["valid"]=checked["valid"] and not checked["errors"]
                metrics={key:checked.get(key) for key in GEOMETRY_METRICS}
                metrics["sharp_turn_count"]=sharp_turn_count(path)
                event=dict(snapshot_id=sid,elapsed_s=elapsed,native_objective_value=native,
                           native_recomputed=native_recomputed,independent_valid=checked["valid"],metrics=metrics)
                row["events"].append(event);row["errors"].extend(checked["errors"])
                certificate_paths[sid]=dict(path_xy=path,valid=checked["valid"])
                previous_time,previous_cost=elapsed,native
            if entry["backend"]=="a3":
                certificate=verify_claim(oracle,request,worker,certificate_paths)
                row["certificate"]=certificate;row["errors"].extend(certificate["errors"])
            if (directory/"path_xy.csv").exists():
                final_path=csv_path(directory/"path_xy.csv")
                final=oracle.validate_path(final_path,request["start"],request["goal"])
                row["errors"].extend(final["errors"])
                if source_events and final_path!=certificate_paths[source_events[-1]["snapshot_id"]]["path_xy"]:
                    row["errors"].append("final_path_not_native_last_incumbent")
            valid=[e for e in row["events"] if e["independent_valid"] and e["elapsed_s"]<=row["budget_s"]]
            status=worker.get("planner_status",worker.get("status"))
            exact=status=="EXACT_SOLUTION" or worker.get("exact_goal") is True
            timings=worker.get("phase_timings",{k:worker.get(k) for k in ("load_s","preprocessing_s","setup_s","actual_solve_wall_s")})
            row.update(success=bool(entry["execution_status"]=="COMPLETED" and exact and valid and not row["errors"]),
                verification_status="FAILED" if row["errors"] else "PASS",native_status=status,timings=timings,
                solve_time_s=timings.get("solve_s",timings.get("actual_solve_wall_s")),
                first_solution_s=valid[0]["elapsed_s"] if valid else None,
                selected_snapshot_id=valid[-1]["snapshot_id"] if valid else None,
                selection_policy="last_native_incumbent_with_elapsed_at_most_budget")
            setup=sum(float(timings.get(k) or 0) for k in ("load_s","conversion_s","preprocessing_s","setup_s"))
            row["first_valid_latency_s"]=setup+row["first_solution_s"] if valid else None
            row["latency_scope"]="worker_setup_plus_first_valid_solve; offline validation excluded"
            if valid and worker.get("solve_started_utc") and (directory/"launch.json").exists():
                launched=read(directory/"launch.json").get("process_created")
                if launched is not None:
                    solve_epoch=dt.datetime.fromisoformat(worker["solve_started_utc"].replace("Z","+00:00")).timestamp()
                    row["first_valid_latency_s"]=max(0.,solve_epoch-launched)+row["first_solution_s"]
                    row["latency_scope"]="launcher_creation_to_first_valid_snapshot; UTC resolution applies"
        row["validation_wall_s"]=time.perf_counter()-began
        directory.mkdir(parents=True,exist_ok=True)
        dump(directory/"validation.json",finite_tree(row))
        if inventory!=trial_inventory(directory):raise ValueError("raw trial changed during validation")
        dump(receipt_path,dict(cache_key=cache_key,verdict_sha256=sha256(directory/"validation.json")))
        rows.append(finite_tree(row))
    return rows


def validate_campaign(root,jobs=1):
    # The scheduler uses this same exclusive lease, including the gaps between
    # paired blocks where no entry happens to carry a running status.
    from terrain_compare_scheduler import _lock
    root=Path(root);lock=_lock(root)
    try:return _validate_campaign_locked(root,jobs)
    finally:lock.unlink(missing_ok=True)


def _validate_campaign_locked(root,jobs=1):
    root=Path(root);manifest=read(root/"manifest.json")
    if any(e["execution_status"] in ("RUNNING","STARTED","START_REQUESTED") for e in manifest["entries"]):
        raise ValueError("validation cannot overlap collection")
    groups={}
    for entry in manifest["entries"]:groups.setdefault(entry["map_id"],[]).append(entry)
    before=raw_hashes(root); code=validation_code_identity()
    tasks=[(str(root),entries,code) for entries in groups.values()]
    from terrain_compare_scheduler import detect_runtime,memory_pressure
    runtime=detect_runtime();available=runtime["memory_limit_bytes"]*(1-memory_pressure(runtime))
    if available*.8<2*1024**3:raise RuntimeError("BLOCKED_VALIDATION_RAM: 2 GiB reservation unavailable")
    jobs=max(1,min(4,jobs,runtime["cpu_capacity"],int(available*.8//(2*1024**3))))
    if jobs==1:rows=[r for task in tasks for r in _validate_group(task)]
    else:
        with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
            rows=[r for group in pool.map(_validate_group,tasks) for r in group]
    if raw_hashes(root)!=before:raise ValueError("raw artifacts changed during validation")
    result=dict(protocol_id=PROTOCOL_ID,manifest_sha256=sha256(root/"manifest.json"),raw_hashes=before,rows=rows,
        slots=len(rows),validated_success=sum(r["success"] for r in rows),
        correctness_failures=sum(bool(r["errors"]) for r in rows),validation_jobs=jobs,
        validation_cache_hits=sum(r["validation_cache_hit"] for r in rows),validation_code_identity=code)
    dump(root/"validation_summary.json",finite_tree(result));return result


def final_metrics(row):
    event=at_checkpoint(row["events"],row["budget_s"])
    return event["metrics"] if event else {}


def mean_sd(values):
    xs=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)]
    return dict(n=len(xs),mean=statistics.mean(xs) if xs else None,sd=statistics.stdev(xs) if len(xs)>1 else None)


def group_summary(rows):
    attempted=[r for r in rows if r["execution_status"] not in UNSTARTED]
    valid=[r for r in attempted if r["success"]];metrics=[final_metrics(r) for r in valid]
    total_length=sum(m["cost_terrain_m"] for m in metrics)
    result=dict(planned=len(rows),attempted=len(attempted),success=len(valid),
                psr=len(valid)/len(attempted) if attempted else None,
                status_counts=dict(collections.Counter(r["execution_status"] for r in rows)),
                sti=sum(m["sharp_turn_count"] for m in metrics)/total_length if total_length>0 else None)
    for key in GEOMETRY_METRICS:result[key]=mean_sd(m.get(key) for m in metrics)
    for key in ("solve_time_s","first_solution_s","first_valid_latency_s"):
        result[key]=mean_sd(r.get(key) for r in valid)
    return result


def paired_bootstrap(rows,competitor,metric,replicates=2000):
    keyed={}
    for row in rows:keyed.setdefault((row["family"],row["map_id"],row["planner_seed"]),{})[row["backend"]]=row
    pairs=[];planned_pairs=0;attempted_pairs=0;excluded_stratum_mismatch=0
    for (family,map_id,seed),methods in keyed.items():
        if "a3" not in methods or competitor not in methods:continue
        planned_pairs+=1
        a,b=methods["a3"],methods[competitor]
        if any(r["execution_status"] in UNSTARTED for r in (a,b)):continue
        attempted_pairs+=1
        if metric.endswith("_s") and a.get("timing_stratum")!=b.get("timing_stratum"):
            excluded_stratum_mismatch+=1;continue
        if metric=="psr":av,bv=float(a["success"]),float(b["success"])
        else:
            if not a["success"] or not b["success"]:continue
            if metric=="sti":
                aa,bb=final_metrics(a),final_metrics(b)
                if aa["cost_terrain_m"]<=0 or bb["cost_terrain_m"]<=0:continue
                av,bv=(aa["sharp_turn_count"],aa["cost_terrain_m"]),(bb["sharp_turn_count"],bb["cost_terrain_m"])
            else:
                av=a.get(metric) if metric.endswith("_s") else final_metrics(a).get(metric)
                bv=b.get(metric) if metric.endswith("_s") else final_metrics(b).get(metric)
            if av is None or bv is None:continue
        pairs.append((family,map_id,(av,bv)))
    coverage=dict(planned_pairs=planned_pairs,attempted_pairs=attempted_pairs,usable_pairs=len(pairs),
                  coverage=len(pairs)/planned_pairs if planned_pairs else None,
                  excluded_stratum_mismatch=excluded_stratum_mismatch)
    if not pairs:return dict(**coverage,maps=0,mean_difference=None,ci95=None)
    groups={}
    for family,map_id,value in pairs:groups.setdefault(family,{}).setdefault(map_id,[]).append(value)
    def estimate(values):
        if metric=="sti":return sum(a[0] for a,b in values)/sum(a[1] for a,b in values)-sum(b[0] for a,b in values)/sum(b[1] for a,b in values)
        return statistics.mean(a-b for a,b in values)
    rng=random.Random(73138);draws=[]
    for _ in range(replicates):
        values=[]
        # Families are fixed strata; resample maps then paired seeds within them.
        for maps in groups.values():
            keys=list(maps)
            for _ in keys:
                ds=maps[rng.choice(keys)];values.extend(rng.choice(ds) for _ in ds)
        draws.append(estimate(values))
    draws.sort()
    return dict(**coverage,maps=len({x[1] for x in pairs}),mean_difference=estimate([x[2] for x in pairs]),
                ci95=[draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]])


def analyze_campaign(root,output):
    from terrain_compare_scheduler import _lock
    root=Path(root);lock=_lock(root)
    try:return _analyze_campaign_locked(root,output)
    finally:lock.unlink(missing_ok=True)


def _analyze_campaign_locked(root,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    validation=read(root/"validation_summary.json")
    if validation["manifest_sha256"]!=sha256(root/"manifest.json") or validation["raw_hashes"]!=raw_hashes(root):
        raise ValueError("stale validation")
    if validation["validation_code_identity"]!=validation_code_identity():raise ValueError("validation code changed")
    rows=validation["rows"];tables=[]
    for scope,key in (("all",None),("family","family"),("map","map_id"),("timing_stratum","timing_stratum")):
        values=["all"] if key is None else sorted({str(r.get(key)) for r in rows})
        for value in values:
            for backend in BACKENDS:
                group=[r for r in rows if r["backend"]==backend and (key is None or str(r.get(key))==value)]
                if group:tables.append(dict(scope=scope,group=value,backend=backend,**group_summary(group)))
    contrast_metrics=("psr","cost_terrain_m","length_xy_m","sti","solve_time_s","first_solution_s","first_valid_latency_s","minimum_footprint_clearance_m","maximum_swept_footprint_slope_deg")
    contrasts=[dict(competitor=b,metric=m,**paired_bootstrap(rows,b,m)) for b in BACKENDS[1:] for m in contrast_metrics]
    manifest=read(root/"manifest.json")
    summary=dict(protocol_id=PROTOCOL_ID,tables=tables,paired_contrasts=contrasts,correctness_failures=validation["correctness_failures"],
        completion="FAILED_CORRECTNESS" if validation["correctness_failures"] else ("COMPLETE" if all(r["execution_status"] not in UNSTARTED for r in rows) else "PARTIAL"),
        sessions=manifest.get("sessions",[]),validation_cache_hits=validation.get("validation_cache_hits",0),
        shared_sem_evaluator=False,plots=False)
    dump(output/"summary.json",finite_tree(summary))
    aggregate=[]
    for table in tables:
        record={k:v for k,v in table.items() if not isinstance(v,dict)}
        for metric,value in table.items():
            if isinstance(value,dict) and {"n","mean","sd"}==set(value):
                record.update({metric+"_"+k:v for k,v in value.items()})
        record["status_counts"]=json.dumps(table["status_counts"],sort_keys=True);aggregate.append(record)
    for name,records in (("aggregate.csv",aggregate),("paired_comparisons.csv",contrasts)):
        with (output/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=sorted({key for row in records for key in row}));writer.writeheader();writer.writerows(records)
    fields=["trial_id","backend","family","map_id","planner_seed","execution_status","native_status","errors","success","timing_stratum",
            "first_solution_s","first_valid_latency_s","solve_time_s","selected_snapshot_id",*GEOMETRY_METRICS]
    with (output/"trials.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:((final_metrics(row).get(k) if row["success"] else None) if k in GEOMETRY_METRICS else (json.dumps(row.get(k,[]),ensure_ascii=False) if k=="errors" else row.get(k))) for k in fields})
    def n(value):return "—" if value is None else f"{value:.6g}"
    def ms(value):return n(value["mean"])+(f" ± {n(value['sd'])}" if value["sd"] is not None else "")
    lines=["# A3-T và baseline native: benchmark tĩnh", "",
        "Mỗi phương pháp giữ objective riêng. Không so tổng native cost giữa phương pháp.",
        "Benchmark sử dụng PSR/APL/STI/Time của thiết kế dựa trên paper và bổ sung an toàn hình học.",
        "Không đo PRI chung, không có bộ chấm SEM độc lập dùng chung; không tạo biểu đồ.",
        "Success yêu cầu exact goal trong budget và kiểm chứng hình học độc lập. Timeout không chứng minh vô nghiệm.",
        "APL/STI/Time/safety chỉ dùng cùng tập trial success; coverage đi cùng mọi bảng.",
        "Time là solve thực tế; T_first là thời gian nghiệm hợp lệ đầu tiên. Dấu ± là sample SD; — là thiếu số đo.",
        "STI = tổng góc rẽ >30° / tổng chiều dài bề mặt, không phải trung bình các tỷ số mỗi đường.",
        "PSR giữ mọi attempt đã bắt đầu, kể cả crash/watchdog/interruption; NOT_RUN/BLOCKED không nằm mẫu số.",
        "Timing đo dưới concurrency đã ghi; không phải latency máy riêng. Đọc theo timing_stratum khi đổi môi trường.",
        f"Correctness failures: {validation['correctness_failures']}; completion: {summary['completion']}.", ""]
    for scope in ("all","family","map","timing_stratum"):
        lines.extend(["",f"## {scope}","", "| Nhóm | Phương pháp | Success/attempt/planned | PSR | APL (m) | STI (1/m) | Time (s) | T_first (s) | Latency (s) | Max slope (deg) | Min clearance (m) |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"])
        for item in (t for t in tables if t["scope"]==scope):
            lines.append("| "+" | ".join([item["group"],LABELS[item["backend"]],f"{item['success']}/{item['attempted']}/{item['planned']}",n(item["psr"]),
                ms(item["cost_terrain_m"]),n(item["sti"]),ms(item["solve_time_s"]),ms(item["first_solution_s"]),ms(item["first_valid_latency_s"]),
                ms(item["maximum_swept_footprint_slope_deg"]),ms(item["minimum_footprint_clearance_m"])])+" |")
    lines.extend(["","## Trạng thái","", "| Phương pháp | Trạng thái |","|---|---|"])
    for item in (t for t in tables if t["scope"]=="all"):
        lines.append(f"| {LABELS[item['backend']]} | "+"; ".join(f"{k}: {v}" for k,v in sorted(item["status_counts"].items()))+" |")
    lines.extend(["","## Paired contrasts","","Hiệu số A3-T − baseline; fixed family strata, bootstrap map → paired seed, 2.000 lần, CI 95%. Quality chỉ dùng cặp cùng success.","",
                  "| Baseline | Metric | Usable/attempt/planned pairs | Coverage | Map | Mean difference | CI95 |","|---|---|---|---|---|---|---|"])
    for c in contrasts:lines.append(f"| {c['competitor']} | {c['metric']} | {c['usable_pairs']}/{c['attempted_pairs']}/{c['planned_pairs']} | {n(c['coverage'])} | {c['maps']} | {n(c['mean_difference'])} | {c['ci95']} |")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path);parser.add_argument("--jobs",type=int,default=1);parser.add_argument("--analyze-only",action="store_true")
    args=parser.parse_args()
    if not args.analyze_only:
        result=validate_campaign(args.input,args.jobs)
        print(json.dumps({k:result[k] for k in ("slots","validated_success","correctness_failures")}))
        if result["correctness_failures"]:raise SystemExit(1)
    if args.output:analyze_campaign(args.input,args.output)
