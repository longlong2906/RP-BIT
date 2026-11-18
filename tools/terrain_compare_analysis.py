"""Independent validation and paired analysis; native cost is not common J."""
import argparse
import concurrent.futures
import csv
import datetime as dt
import json
import math
from pathlib import Path
import random
import statistics
import time

from experiment_common import digest, dump
from independent_verifier import Oracle
from flat_direct_certificate import verify_claim
from terrain_compare_protocol import DEFAULT_TRAVERSABILITY, sha256

METRICS = ("cost_terrain_m", "length_xy_m", "risk_slope_integral_m", "risk_clearance_integral_m",
           "risk_integral_m", "objective_value", "minimum_obstacle_footprint_clearance_m",
           "minimum_footprint_clearance_m", "maximum_swept_footprint_slope_deg")
UNOBSERVED = ("PLANNED", "BLOCKED_SOURCE", "BLOCKED", "NOT_RUN", "INTERRUPTED", "RESOURCE_PRESSURE")


def raw_hashes(root):
    """Inputs to validation; never trust an earlier verdict after a raw artifact edit."""
    root=Path(root)
    files=[]
    for name in ("request.json","worker.json","progress.jsonl","path_xy.csv","path_raw_xyz.csv","execution.json","launch.json"):
        files.extend(root.glob("*/"+name))
    files.extend(root.glob("*/snapshots/*/path*.csv"))
    return {str(p.relative_to(root)):sha256(p) for p in sorted(set(files))}


def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def csv_path(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return [[float(x) for x in row] for row in list(csv.reader(f))[1:]]


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value): return None
    if isinstance(value, dict): return {k: finite_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [finite_tree(v) for v in value]
    return value


def at_checkpoint(events, checkpoint):
    available = [e for e in events if e["independent_valid"] and e["elapsed_s"] <= checkpoint]
    return available[-1] if available else None


def _validate_group(task):
    root, entries = task
    root = Path(root); oracle = None; summaries = []
    for entry in entries:
        directory = root/entry["trial_id"]
        status = entry.get("execution_status", "PLANNED")
        record = dict(trial_id=entry["trial_id"], track=entry["track"], backend=entry["backend"],
                      family=entry["family"], map_id=entry["map_id"], planner_seed=entry["planner_seed"],
                      execution_status=status, success=False, events=[], errors=[], verification_status="NO_WORKER",
                      budget_s=entry["request"]["planning_budget_s"], first_solution_s=None)
        record["execution"]=read(directory/"execution.json") if (directory/"execution.json").is_file() else {}
        if not (directory/"worker.json").is_file():
            summaries.append(record); continue
        request = read(directory/"request.json"); worker = read(directory/"worker.json")
        expected_request=dict(entry["request"])
        if "cloud_phase_handshake" in request:expected_request["cloud_phase_handshake"]=True
        if digest(request)!=digest(expected_request):raise ValueError("raw request differs from frozen trial plan")
        if sha256(request["map_path"]) != entry["map_sha256"]: raise ValueError("raw map hash changed")
        if oracle is None: oracle = Oracle(request["map_path"])
        began = time.perf_counter(); certificate_paths = {}; events = []
        if worker.get("snapshots") is not None:
            source_events = worker["snapshots"]
        else:
            source_events = [json.loads(line) for line in (directory/"progress.jsonl").read_text().splitlines()
                             if line.strip() and json.loads(line).get("event_type") == "improvement"]
        prior_time = -1.; prior_cost = math.inf; seen = set()
        for source in source_events:
            sid = source["snapshot_id"]
            if not isinstance(sid, str) or Path(sid).name != sid or sid in seen: raise ValueError("unsafe/reused snapshot ID")
            seen.add(sid)
            xy = source.get("path_xy")
            if xy is None: xy = csv_path(directory/"snapshots"/sid/"path_xy.csv")
            elapsed = source.get("elapsed_s", source.get("timestamp_s"))
            native = source.get("planner_objective_value", source.get("objective_value"))
            if not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < prior_time:
                raise ValueError("invalid/nonmonotone snapshot time")
            if not isinstance(native, (int, float)) or not math.isfinite(native) or native < 0 or native > prior_cost+1e-8:
                raise ValueError("invalid/increasing planner objective")
            checked = oracle.validate_path(xy, request["start"], request["goal"], traversability=DEFAULT_TRAVERSABILITY)
            for key in ("cost_terrain_m", "length_xy_m", "risk_slope_integral_m", "risk_clearance_integral_m", "risk_integral_m"):
                claim = source.get(key)
                # Own-cost workers need not construct common risk while planning.
                if claim is not None and (not isinstance(claim, (int,float)) or not math.isfinite(claim) or abs(claim-checked.get(key, math.inf)) > 1e-8*(1+abs(claim))):
                    checked["errors"].append(key+"_mismatch")
            if entry["track"] == "common" or entry["backend"] == "a3":
                if abs(native-checked.get("objective_value", math.inf)) > 1e-8*(1+native):
                    checked["errors"].append("planner_common_objective_mismatch")
            elif entry["backend"] == "sgp":
                if abs(native-checked["length_xy_m"]) > 1e-8*(1+native): checked["errors"].append("native_xy_objective_mismatch")
            checked["valid"] = checked["valid"] and not checked["errors"]
            event = dict(snapshot_id=sid, elapsed_s=elapsed, planner_objective_value=native,
                         independent_valid=checked["valid"], metrics={k: checked.get(k) for k in METRICS})
            event["metrics"]["mean_risk_density"] = (checked.get("risk_integral_m", 0)/checked["cost_terrain_m"]
                                                   if checked.get("cost_terrain_m", 0)>0 else 0.)
            certificate_paths[sid] = {"path_xy": xy, "valid": checked["valid"]}
            dump(directory/"snapshots"/sid/"independent_validation.json", finite_tree(checked))
            record["errors"].extend(checked["errors"]); events.append(event)
            prior_time, prior_cost = elapsed, native
        if entry["backend"] == "a3":
            cert = verify_claim(oracle, request, worker, certificate_paths)
            record["certificate"] = cert; record["errors"].extend(cert["errors"])
        if (directory/"path_xy.csv").is_file():
            final_xy = csv_path(directory/"path_xy.csv")
            final_check = oracle.validate_path(final_xy, request["start"], request["goal"], traversability=DEFAULT_TRAVERSABILITY)
            record["errors"].extend(final_check["errors"])
            record["final_path_valid"] = final_check["valid"]
        else:
            record["final_path_valid"] = bool(events)  # External final path also lives in immutable snapshot.
        budget = request["planning_budget_s"]
        in_budget = [e for e in events if e["elapsed_s"] <= budget and e["independent_valid"]]
        exact = worker.get("planner_status", worker.get("status")) == "EXACT_SOLUTION" or worker.get("exact_goal") is True
        record.update(events=events, budget_s=budget, native_status=worker.get("planner_status", worker.get("status")),
                      success=bool(status=="COMPLETED" and exact and in_budget and record["final_path_valid"] and not record["errors"]),
                      verification_status="PASS" if not record["errors"] else "FAILED",
                      first_solution_s=in_budget[0]["elapsed_s"] if in_budget else None,
                      validation_wall_s=time.perf_counter()-began,
                      timings=worker.get("phase_timings", {k: worker.get(k) for k in ("load_s", "preprocessing_s", "setup_s", "actual_solve_wall_s")}),
                      worker_sha256=sha256(directory/"worker.json"))
        execution = read(directory/"execution.json") if (directory/"execution.json").exists() else {}
        record["execution"] = execution
        record["counters"] = worker.get("counters", {})
        record["native_model"] = worker.get("native_model", {})
        record["native_geometric_length_m"] = worker.get("native_geometric_length_m")
        overhead = sum(float(record["timings"].get(k) or 0) for k in ("load_s", "conversion_s", "preprocessing_s", "setup_s"))
        record["first_valid_latency_s"] = overhead+record["first_solution_s"] if record["first_solution_s"] is not None else None
        record["first_valid_latency_scope"] = "worker_load_conversion_preprocessing_setup_and_solve; process_import_unavailable_unless_worker_reports"
        if worker.get("solve_started_utc") and (directory/"launch.json").exists():
            created=read(directory/"launch.json").get("process_created")
            solve_epoch=dt.datetime.fromisoformat(worker["solve_started_utc"].replace("Z","+00:00")).timestamp()
            if created is not None:
                record["first_valid_latency_s"] = max(0.,solve_epoch-created)+record["first_solution_s"] if record["first_solution_s"] is not None else None
                record["first_valid_latency_scope"] = "launcher_process_creation_to_verified_snapshot; UTC timestamp resolution applies; offline validation excluded"
        dump(directory/"validation.json", finite_tree(record)); summaries.append(record)
    return summaries


def validate_campaign(root, jobs=1):
    root=Path(root); manifest=read(root/"manifest.json"); groups={}
    if any(e.get("execution_status") in ("RUNNING", "STARTED", "START_REQUESTED") for e in manifest["entries"]):
        raise ValueError("validation must not overlap measured collection")
    for e in manifest["entries"]: groups.setdefault(e["map_id"], []).append(e)
    tasks=[(str(root), entries) for entries in groups.values()]
    hashes=raw_hashes(root)
    # Conservative independent-oracle reservations; never launch four by CPU count alone.
    from terrain_compare_scheduler import detect_runtime, memory_pressure
    runtime=detect_runtime()
    available=runtime["memory_limit_bytes"]*(1-memory_pressure(runtime))
    if available*.8 < 2*1024**3: raise RuntimeError("BLOCKED_VALIDATION_RAM: 2 GiB reservation exceeds 80% available")
    jobs=max(1,min(4,jobs,runtime["cpu_capacity"],int(available*.8//(2*1024**3)) or 1))
    if jobs==1: chunks=map(_validate_group,tasks); rows=[r for chunk in chunks for r in chunk]
    else:
        with concurrent.futures.ProcessPoolExecutor(jobs) as pool:
            rows=[r for chunk in pool.map(_validate_group,tasks) for r in chunk]
    if hashes!=raw_hashes(root): raise ValueError("raw artifacts changed during validation")
    result={"manifest_sha256":sha256(root/"manifest.json"),"raw_hashes":hashes,"rows":rows,
            "validation_jobs":jobs,"validation_ram_reservation_per_process_bytes":2*1024**3,
            "slots":len(rows),"validated_success":sum(r["success"] for r in rows),
            "correctness_failures":sum(bool(r["errors"]) for r in rows)}
    dump(root/"validation_summary.json",finite_tree(result))
    return result


def paired_bootstrap(rows, track, competitor, metric, replicates=2000):
    keyed={}
    for r in rows:
        if r["track"]==track: keyed.setdefault((r["family"],r["map_id"],r["planner_seed"]),{})[r["backend"]]=r
    pairs=[]
    for (family,map_id,seed), methods in keyed.items():
        if "a3" not in methods or competitor not in methods: continue
        a,b=methods["a3"],methods[competitor]
        if any(r["execution_status"] in UNOBSERVED for r in (a,b)):continue
        if metric=="success": av,bv=float(a["success"]),float(b["success"])
        elif metric=="restricted_time":
            av=a["first_solution_s"] if a["success"] else a["budget_s"]
            bv=b["first_solution_s"] if b["success"] else b["budget_s"]
        else:
            if not a["success"] or not b["success"]:continue
            aa,bb=at_checkpoint(a["events"],a["budget_s"]),at_checkpoint(b["events"],b["budget_s"])
            av,bv=aa["metrics"].get(metric),bb["metrics"].get(metric)
            if av is None or bv is None:continue
        pairs.append((family,map_id,seed,av-bv))
    if not pairs:return {"usable_pairs":0,"mean_difference":None,"ci95":None}
    groups={}
    for f,m,s,d in pairs:groups.setdefault(f,{}).setdefault(m,[]).append(d)
    rng=random.Random(73136); draws=[]
    for _ in range(replicates):
        values=[]
        families=list(groups)
        for _ in families:
            maps=groups[rng.choice(families)]
            keys=list(maps)
            for _ in keys:
                ds=maps[rng.choice(keys)]; values.extend(rng.choice(ds) for _ in ds)
        draws.append(statistics.mean(values))
    draws.sort()
    return {"usable_pairs":len(pairs),"maps":len({p[1] for p in pairs}),
            "mean_difference":statistics.mean(p[3] for p in pairs),
            "ci95":[draws[int(.025*(replicates-1))],draws[int(.975*(replicates-1))]],
            "contrast":"A3-minus-comparator; resample family then map then paired seeds"}


def report_number(value):
    if value is None: return "—"
    if isinstance(value, (int, float)):
        return f"{value:.6g}" if math.isfinite(value) else "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def markdown_table(lines, headers, records):
    lines.extend(["", "| "+" | ".join(headers)+" |",
                  "|"+"|".join("---" for _ in headers)+"|"])
    lines.extend("| "+" | ".join(report_number(v) for v in row)+" |" for row in records)


def markdown_report(tables, families, contrasts, throughput, validation):
    """Render existing computed summaries; no plotting or statistical recomputation."""
    lines=["# A3-T / SEM-adapted / PUTN-adapted / SGP-adapted", "",
           "Thời gian đo dưới chế độ concurrency ghi trong runtime/manifest; không phải latency cô lập.",
           "Risk là exposure hình học, không phải xác suất tai nạn. Các model có thể loại thêm vùng.",
           "Dấu — là thiếu số đo/cặp hợp lệ, không thay bằng 0. Success yêu cầu exact goal và independent validation.",
           "Các median chất lượng lấy từ snapshot cuối trong budget của trial success. Median từng cột không tạo thành một đường mới.", "",
           "## Coverage và trạng thái"]
    markdown_table(lines, ["Nhánh", "Backend", "Slot", "Đã quan sát", "Success", "Execution status"],
        ([r["track"],r["backend"],r["slots"],r["executed"],r["success"],
          "; ".join(f"{k}: {v}" for k,v in r["status_counts"].items()) or "—"] for r in tables))
    lines.extend(["",f"Correctness failures: {validation['correctness_failures']}.",
                  "Timeout/crash được giữ trong số trial quan sát. Resource pressure/ngắt phiên và slot chưa chạy là thiếu quan sát, không gán thành thất bại thuật toán.",
                  "SEM own-cost giữ BLOCKED_SOURCE. Native objectives khác định nghĩa không được so trực tiếp.", "",
                  "## Thời gian đến nghiệm đã xác minh"])
    markdown_table(lines, ["Nhánh", "Backend", "Số success", "Median T_first (s)", "Median cold-start latency (s)"],
        ([r["track"],r["backend"],r["success"],r["median_first_solution_s"],r["median_first_valid_latency_s"]] for r in tables))
    lines.extend(["", "Hai latency trên chỉ tóm tắt trial success; thời gian xác minh offline không nằm trong latency.",
                  "Cold-start dùng scope ghi tại từng validation.json; process import có thể chưa đo nếu worker thiếu timestamp launcher.", "",
                  "## Thời gian các pha"])
    phase_keys=("load_s","conversion_s","preprocessing_s","setup_s","solve_s","actual_solve_wall_s")
    markdown_table(lines, ["Nhánh", "Backend", *["Median "+k for k in phase_keys]],
        ([r["track"],r["backend"],*[r["median_"+k] for k in phase_keys]] for r in tables))
    lines.extend(["", "Đơn vị giây. Median mỗi pha dùng các trial quan sát có số đo; solve_s và actual_solve_wall_s giữ tên worker gốc, không cộng hai cột.", "",
                  "## Chiều dài và exposure"])
    quality_keys=("cost_terrain_m","length_xy_m","objective_value","risk_slope_integral_m",
                  "risk_clearance_integral_m","risk_integral_m","mean_risk_density")
    markdown_table(lines, ["Nhánh", "Backend", "Median L_H (m)", "Median XY (m)", "Median common J",
                           "Median R_s (m)", "Median R_c (m)", "Median R (m)", "Median R/L_H"],
        ([r["track"],r["backend"],*[r["median_"+k] for k in quality_keys]] for r in tables))
    lines.extend(["", "Common J = L_H + R với lambda 1; chỉ so J khi cùng lambda. J không phải chiều dài vật lý. Đường ngắn hơn không mặc định tốt hơn.",
                  "Own-cost vẫn đánh giá bằng common J ở bảng này; không yêu cầu common J giảm đơn điệu trong own-cost.", "",
                  "## An toàn của đường đã xác minh"])
    safety_keys=("minimum_obstacle_footprint_clearance_m","minimum_footprint_clearance_m","maximum_swept_footprint_slope_deg")
    markdown_table(lines, ["Nhánh", "Backend", "Median min obstacle-only footprint clearance (m)",
                           "Median min forbidden-region footprint clearance (m)", "Median max swept-footprint slope (deg)"],
        ([r["track"],r["backend"],*[r["median_"+k] for k in safety_keys]] for r in tables))
    lines.extend(["", "Đây là median giữa các trial của minimum/maximum trên từng đường, không phải cực trị toàn campaign. Clearance không xác định/vô hạn không được đổi thành 0.", "",
                  "## Tài nguyên và throughput"])
    markdown_table(lines, ["Nhánh", "Backend", "Median peak RSS (MiB)", "Median process-tree CPU (s)",
                           "Median worker wall (s)", "Median max concurrency"],
        ([r["track"],r["backend"],r["median_peak_rss_bytes"]/(1024**2) if r["median_peak_rss_bytes"] is not None else None,
          r["median_process_tree_cpu_s"],r["median_worker_wall_s"],r["median_max_concurrency"]] for r in tables))
    lines.extend(["",f"Campaign status: {report_number(throughput.get('campaign_status'))}.",
                  f"Collection wall time: {report_number(throughput['collection_wall_s'])} s; started attempts: {throughput['started_attempts']}; throughput: {report_number(throughput['trials_per_hour'])} attempts/hour.",
                  "Throughput gồm các attempt đã bắt đầu; không phải số nghiệm thành công/giờ. Không chia latency trial cho bốn.",
                  "Build/validation/export nằm ngoài collection wall time. Thời gian chờ scheduler được ghi riêng và không cộng vào latency trial.", "",
                  "## Coverage theo family"])
    markdown_table(lines, ["Nhánh", "Backend", "Family", "Đã quan sát / slot", "Success", "Median L_H (m)", "Median R (m)"],
        ([r["track"],r["backend"],r["family"],f"{r['observed']} / {r['slots']}",r["success"],
          r["median_cost_terrain_m"],r["median_risk_integral_m"]] for r in families if r["slots"]))
    lines.extend(["", "## So sánh ghép cặp và CI 95%", "",
                  "Hiệu số = A3 trừ comparator; bootstrap 2.000 lần theo family → map → paired seed. Cặp phải cùng map/query/seed và cùng nhánh.",
                  "Success dùng mọi cặp quan sát. Các metric chất lượng dùng cặp cả hai thành công; số cặp được ghi riêng.",
                  "restricted_time gán budget cho trial chưa thành công để tính restricted score; không gọi đó là thời gian nghiệm quan sát được.",
                  "Hiệu số dương có nghĩa khác nhau theo metric: success/clearance cao hơn; thời gian/chiều dài/exposure/slope lớn hơn. CI smoke chỉ mang tính chẩn đoán."])
    markdown_table(lines, ["Nhánh", "Comparator", "Metric", "Cặp dùng được", "Map", "Mean A3 − comparator", "CI 95%"],
        ([r["track"],r["competitor"],r["metric"],r["usable_pairs"],r.get("maps"),r["mean_difference"],
          "["+", ".join(report_number(x) for x in r["ci95"])+"]" if r.get("ci95") is not None else None] for r in contrasts))
    lines.extend(["", "Các điểm không bị trội quan sát được nằm trong observed_nondominated.json; không chứng minh toàn bộ Pareto frontier.",
                  "summary.json giữ đầy đủ số liệu/CI; trials.csv giữ từng trial; anytime_tuples.json giữ nguyên tuple của cùng một incumbent tại mỗi mốc thời gian."])
    return lines


def analyze_campaign(root, output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    validation=read(root/"validation_summary.json")
    if validation["manifest_sha256"]!=sha256(root/"manifest.json"):raise ValueError("validation manifest stale")
    if validation.get("raw_hashes")!=raw_hashes(root):raise ValueError("validation raw artifacts stale")
    rows=validation["rows"]; tables=[]; curves=[]
    manifest=read(root/"manifest.json")
    for r in rows:
        for t in (.1,.25,.5,1.,2.,5.,10.):
            if t>r.get("budget_s",0):continue
            event=at_checkpoint(r["events"],t)
            curves.append(dict(trial_id=r["trial_id"],checkpoint_s=t,attained=bool(event),snapshot=event))
    for track in ("common","own"):
        for backend in ("a3","sem","putn","sgp"):
            group=[r for r in rows if r["track"]==track and r["backend"]==backend]
            executed=[r for r in group if r["execution_status"] not in UNOBSERVED]
            success=[r for r in executed if r["success"]]
            values=[r["first_solution_s"] for r in success]
            table={"track":track,"backend":backend,"slots":len(group),"executed":len(executed),
                           "success":len(success),"median_first_solution_s":statistics.median(values) if values else None,
                           "coverage_note":"failures/timeouts retained; interruptions/resource pressure/missing slots excluded from algorithm contrasts",
                           "status_counts":{status:sum(r["execution_status"]==status for r in group) for status in sorted({r["execution_status"] for r in group})}}
            final=[at_checkpoint(r["events"],r["budget_s"])["metrics"] for r in success]
            for key in (*METRICS,"mean_risk_density"):
                xs=[e[key] for e in final if e.get(key) is not None and math.isfinite(e[key])]
                table["median_"+key]=statistics.median(xs) if xs else None
            for key in ("load_s","conversion_s","preprocessing_s","setup_s","solve_s","actual_solve_wall_s"):
                xs=[r.get("timings",{}).get(key) for r in executed]
                xs=[x for x in xs if x is not None]
                table["median_"+key]=statistics.median(xs) if xs else None
            for key in ("peak_rss_bytes","process_tree_cpu_s","worker_wall_s","max_concurrency"):
                xs=[r.get("execution",{}).get(key) for r in executed]
                xs=[x for x in xs if x is not None]
                table["median_"+key]=statistics.median(xs) if xs else None
            xs=[r["first_valid_latency_s"] for r in success if r.get("first_valid_latency_s") is not None]
            table["median_first_valid_latency_s"]=statistics.median(xs) if xs else None
            tables.append(table)
    contrasts=[dict(track=t,competitor=b,metric=m,**paired_bootstrap(rows,t,b,m))
               for t in ("common","own") for b in ("sem","putn","sgp")
               for m in ("success","restricted_time",*METRICS,"mean_risk_density")]
    families=[]
    for table in tables:
        for family in sorted({r["family"] for r in rows}):
            group=[r for r in rows if r["family"]==family and r["track"]==table["track"] and r["backend"]==table["backend"]]
            valid=[r for r in group if r["success"]]
            item={"family":family,"track":table["track"],"backend":table["backend"],"slots":len(group),
                  "observed":sum(r["execution_status"] not in UNOBSERVED for r in group),"success":len(valid)}
            for key in METRICS:
                values=[at_checkpoint(r["events"],r["budget_s"])["metrics"].get(key) for r in valid]
                values=[v for v in values if v is not None and math.isfinite(v)]
                item["median_"+key]=statistics.median(values) if values else None
            families.append(item)
    pareto=[]; same_queries={}
    for r in rows:
        if r["success"]:same_queries.setdefault((r["track"],r["map_id"],r["planner_seed"]),[]).append(r)
    dimensions=("cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m")
    for key,group in same_queries.items():
        points=[(r,at_checkpoint(r["events"],r["budget_s"])["metrics"]) for r in group]
        frontier=[]
        for r,m in points:
            dominated=any(all(n[k]<=m[k] for k in dimensions) and any(n[k]<m[k] for k in dimensions) for _,n in points)
            if not dominated:frontier.append({"trial_id":r["trial_id"],"backend":r["backend"],"metrics":m})
        pareto.append({"track":key[0],"map_id":key[1],"planner_seed":key[2],"observed_nondominated":frontier,
                       "candidate_count":len(points),"proven_complete_frontier":False})
    seconds=sum(s.get("wall_s",0) for s in manifest.get("sessions",[]))
    attempts=sum(r["execution_status"] not in ("PLANNED","BLOCKED_SOURCE","BLOCKED","NOT_RUN") for r in rows)
    throughput={"collection_wall_s":seconds,"started_attempts":attempts,
                "trials_per_hour":3600*attempts/seconds if seconds else None,
                "excludes_build_validation_and_export":True,
                "sessions":manifest.get("sessions",[]),"scheduler_policy":manifest.get("scheduler_policy"),
                "runtime":manifest.get("runtime"),"campaign_status":manifest.get("status")}
    dump(output/"summary.json",finite_tree({"tables":tables,"by_family":families,"paired_contrasts":contrasts,"throughput":throughput,
         "correctness_failures":validation["correctness_failures"],"validation_jobs":validation.get("validation_jobs")}))
    dump(output/"observed_nondominated.json",finite_tree(pareto))
    dump(output/"anytime_tuples.json",finite_tree(curves))
    with (output/"trials.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["trial_id","track","backend","family","map_id","planner_seed","execution_status","success","first_solution_s","first_valid_latency_s","native_geometric_length_m",*METRICS]
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for r in rows:
            event=at_checkpoint(r["events"],r.get("budget_s",0))
            writer.writerow({k:(event["metrics"].get(k) if event and k in METRICS else r.get(k)) for k in fields})
    lines=markdown_report(tables,families,contrasts,throughput,validation)
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return tables


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--input",type=Path,required=True);p.add_argument("--output",type=Path)
    p.add_argument("--jobs",type=int,default=1);p.add_argument("--analyze-only",action="store_true")
    a=p.parse_args()
    if not a.analyze_only:
        v=validate_campaign(a.input,a.jobs)
        print(json.dumps({k:v[k] for k in ("slots","validated_success","correctness_failures")}))
        if v["correctness_failures"]:raise SystemExit(1)
    if a.output:analyze_campaign(a.input,a.output)
