"""Schema-8 descriptive analysis; retain each validated incumbent as one tuple."""
import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path
import tempfile
import os

from experiment_common import digest, dump, file_hash, read_json

METRICS=("objective_value","cost_terrain_m","risk_slope_integral_m","risk_clearance_integral_m",
         "risk_integral_m","length_xy_m","minimum_obstacle_footprint_clearance_m","maximum_swept_footprint_slope_deg")


def checked_events(directory, absolute=1e-8, relative=1e-8):
    from analyze_results import close,finite
    directory=Path(directory);events=[];errors=[];previous_time=-1.;previous_objective=float("inf");seen=set()
    if not (directory/"validated_events.json").exists():return events,["missing_validated_events"]
    for event in read_json(directory/"validated_events.json"):
        sid=event.get("snapshot_id");reason=None
        if not isinstance(sid,str) or Path(sid).name!=sid or sid in seen:reason="invalid_or_reused_snapshot_id"
        elif not finite(event.get("elapsed_s")) or event["elapsed_s"]<max(0.,previous_time):reason="invalid_event_time"
        elif not event.get("independent_valid"):reason="invalid_incumbent_snapshot"
        else:
            snapshot=directory/"snapshots"/sid
            try:
                validation=read_json(snapshot/"validation.json")
                if not validation.get("valid"):reason="invalid_snapshot_validation"
                elif any(file_hash(snapshot/name)!=validation.get(key) for name,key in
                         (("path_xy.csv","xy_sha256"),("path_surface.csv","surface_sha256"))):reason="snapshot_hash_mismatch"
                elif any(not close(event.get(key),validation.get(key),absolute,relative) for key in METRICS[:6]):reason="snapshot_metric_mismatch"
                elif not close(event.get("best_cost_terrain_m"),event.get("cost_terrain_m"),absolute,relative):reason="legacy_length_field_mismatch"
                elif event["objective_value"]>previous_objective and not close(event["objective_value"],previous_objective,absolute,relative):reason="increasing_objective_incumbent"
                else:
                    event=dict(event)
                    for key in METRICS[6:]:event[key]=validation.get(key)
            except (OSError,ValueError,KeyError,TypeError) as exc:reason="invalid_snapshot_artifacts:"+type(exc).__name__
        if reason:errors.append({"snapshot_id":sid,"error":reason});continue
        seen.add(sid);previous_time=event["elapsed_s"];previous_objective=event["objective_value"];events.append(event)
    return events,errors


def incumbent_at(events,time_s):
    """Last admitted tuple; taking separate minima fabricates a nonexistent path."""
    eligible=[event for event in events if event["elapsed_s"]<=time_s]
    return eligible[-1] if eligible else None


def read_trial(source,entry,budget,default_config):
    from analyze_results import finite,metric,COUNTERS,TIMINGS
    directory=Path(source)/entry["trial_id"];row=dict(entry,executed=False,success=False,status="NOT_RUN",errors=[],events=[],
        time_budget_s=budget,first_solution_time_s=None,first_solution_end_to_end_s=None,
        validation_wall_s=None,actual_solve_wall_s=None,worker_wall_s=None,end_to_end_run_wall_s=None,
        metric_tuple_at_budget=None,metrics={},metric_unavailable_reasons={},**{key:None for key in METRICS})
    config=entry.get("resolved_traversability",entry.get("traversability",default_config));row["traversability"]=config;row["lambda"]=config.get("lambda")
    objective_hash=digest({"objective_type":"surface_length_traversability","traversability":config})
    if "objective_config_sha256" in entry and entry["objective_config_sha256"]!=objective_hash:
        raise ValueError("planned objective configuration hash mismatch")
    path=directory/"trial.json"
    if not path.exists():return row
    trial=read_json(path)
    if "objective_config_sha256" in trial and trial["objective_config_sha256"]!=objective_hash:
        raise ValueError("trial objective configuration hash mismatch")
    for key in ("trial_id","map_id","map_hash","query_id","planner_id","planner_seed"):
        if key in entry and trial.get(key)!=entry[key]:raise ValueError("trial identity mismatch: "+key)
    row["status"]=trial["execution_status"];row["executed"]=row["status"] not in ("BLOCKED","NOT_RUN","PLANNED")
    row["time_budget_s"]=trial.get("time_budget_s",budget)
    row["termination_reason"]=trial.get("termination_reason")
    if "sampler_query_cache" in entry.get("resolved_optimizations", {}):
        options=entry["resolved_optimizations"]
        if entry.get("resolved_optimizations_sha256") != digest(options):
            raise ValueError("planned performance options hash mismatch")
        if row["status"] == "COMPLETED":
            for key in ("sampler_query_cache","traversability_build_policy","exact_flat_certificate"):
                if trial.get("effective_policy",{}).get(key) != options[key]:
                    raise ValueError("effective performance policy mismatch: "+key)
        row["exact_certificate"]=trial.get("exact_certificate")
        row["exact_certificate_independent_valid"]=trial.get("exact_certificate_independent_valid")
        row["traversability_build_stats"]=trial.get("traversability_build_stats")
    effective=trial.get("effective_traversability",{})
    if row["status"]=="COMPLETED" and (not effective or trial.get("requested_traversability")!=config or
        trial.get("objective_type")!="surface_length_traversability" or
        ("objective_config_sha256" in entry and trial.get("objective_config_sha256")!=objective_hash)):
        raise ValueError("completed worker lacks matching objective policy/hash")
    if effective:
        if effective!=config:raise ValueError("effective traversability differs from planned configuration")
        row["traversability"]=effective;row["lambda"]=effective.get("lambda")
    validation=read_json(directory/"validation.json") if (directory/"validation.json").exists() else {}
    row["offline_validation_available"]=bool(validation)
    row["validation_pending"]=row["executed"] and not validation
    events,errors=checked_events(directory) if (directory/"validated_events.json").exists() else ([],["missing_validated_events"] if trial.get("path_available") else [])
    row["events"]=events;row["errors"]=errors
    if trial.get("path_available") and not validation.get("valid"):errors.append("final_path_not_validated")
    if "sampler_query_cache" in entry.get("resolved_optimizations", {}) and row["status"] == "COMPLETED":
        if not validation.get("exact_certificate_validation",{}).get("valid"):
            errors.append("certificate_provenance_not_independently_validated")
    eligible=[e for e in events if e["elapsed_s"]<=row["time_budget_s"]]
    row["events_after_budget"]=len(events)-len(eligible)
    row["success"]=bool(row["status"]=="COMPLETED" and trial.get("planner_status")=="EXACT_SOLUTION" and
        trial.get("independent_valid") is True and validation.get("valid") and eligible and not errors)
    for key in ("actual_solve_wall_s","worker_wall_s","validation_wall_s"):
        row[key]=metric(trial,key)[0]
    for key in (*COUNTERS,*TIMINGS,"traversability_preprocessing_s"):
        value,reason=metric(trial,key);row["metrics"][key]=value
        if reason:row["metric_unavailable_reasons"][key]=reason
    for key in ("load_s","preprocessing_s","traversability_preprocessing_s","collision_index_build_s","setup_s","serialize_s","peak_rss_mb"):
        row[key]=metric(trial,key)[0]
    if finite(row["worker_wall_s"]) and finite(row["validation_wall_s"]):
        row["end_to_end_run_wall_s"]=row["worker_wall_s"]+row["validation_wall_s"]
    if row["success"]:
        row["first_solution_time_s"]=eligible[0]["elapsed_s"]
        last=incumbent_at(eligible,row["time_budget_s"])
        row["metric_tuple_at_budget"]={key:last.get(key) for key in ("snapshot_id","elapsed_s",*METRICS)}
        row.update({key:last.get(key) for key in METRICS})
        prefix=[metric(trial,key)[0] for key in ("load_wall_s","preprocessing_wall_s","setup_wall_s")]
        if all(finite(value) for value in prefix):row["first_solution_end_to_end_s"]=sum(prefix)+row["first_solution_time_s"]
    row["time_semantics"]="T_first is native callback time of offline-validated snapshot; end-to-end first adds load/preprocess/setup and excludes offline verification; run total is worker wall plus offline verification"
    return row


def plot_results(output,rows):
    """Plot observed tuples; dependency failure is recorded, never synthetic data."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        groups=defaultdict(list)
        for row in rows:
            if row["success"]:groups[(row["planner_id"],row["lambda"])].append(row)
        for (planner,lam),group in groups.items():
            axes[0].scatter([r["cost_terrain_m"] for r in group],[r["risk_integral_m"] for r in group],label=f"{planner}, lambda={lam}")
            axes[1].scatter([r["first_solution_time_s"] for r in group],[r["risk_integral_m"] for r in group],label=f"{planner}, lambda={lam}")
        axes[0].set(xlabel="Surface length L at budget (m)",ylabel="Risk integral R at budget (m)")
        axes[1].set(xlabel="First validated snapshot callback (s)",ylabel="Risk integral R at budget (m)")
        if groups:axes[0].legend(fontsize=7)
        fig.savefig(Path(output)/"length_risk_time.png",dpi=160);plt.close(fig)
        figures=["length_risk_time.png"]
        objective_groups=defaultdict(list)
        for row in rows:
            policy=row["traversability"]
            key=(row["lambda"],policy.get("slope_weight"),policy.get("clearance_weight"),policy.get("clearance_comfort_m"))
            objective_groups[key].append(row)
        for index,(key,group) in enumerate(objective_groups.items()):
            families=sorted({r.get("family","") for r in group});n=len(families)
            if not n:continue
            fig,axes=plt.subplots(n,1,figsize=(8,max(3,2.7*n)),squeeze=False,layout="constrained")
            for axis,family in zip(axes[:,0],families):
                plotted=False
                for row in group:
                    if row.get("family","")!=family or row["errors"]:continue
                    events=[e for e in row["events"] if e["elapsed_s"]<=row["time_budget_s"]]
                    if not events:continue
                    x=[e["elapsed_s"] for e in events]+[row["time_budget_s"]]
                    y=[e["objective_value"] for e in events]+[events[-1]["objective_value"]]
                    axis.step(x,y,where="post",label=row["planner_id"]);plotted=True
                axis.set(title=f"{family}; lambda={key[0]}, weights={key[1:3]}",xlabel="Solve time (s)",ylabel="Validated incumbent J (m)")
                if plotted:axis.legend(fontsize=6)
            name=f"convergence_objective_group_{index}.png";fig.savefig(Path(output)/name,dpi=140);plt.close(fig);figures.append(name)
        return figures,None
    except (ImportError,OSError) as error:return [],type(error).__name__+": "+str(error)


def analyze(source,output,make_plots=True):
    from analyze_results import coverage,distribution,verify_raw_artifact_inventory,COUNTERS,TIMINGS
    source,output=Path(source),Path(output)
    if source.resolve()==output.resolve():raise ValueError("report must be separate from raw results")
    manifest=read_json(source/"manifest.json");config=read_json(source/"resolved_config.json")
    if config.get("schema_version") not in (8,9):raise ValueError("traversability analyzer requires schema 8 or 9")
    inventory=verify_raw_artifact_inventory(source,required=False)
    profile=manifest["profile"];budget=config["profiles"][profile]["planning_budget_s"]
    defaults=config.get("traversability",{})
    if len({e["trial_id"] for e in manifest["trials"]})!=len(manifest["trials"]):raise ValueError("duplicate trial id")
    if manifest.get("planned_trials",len(manifest["trials"]))!=len(manifest["trials"]):raise ValueError("planned trial count mismatch")
    rows=[read_trial(source,entry,budget,defaults) for entry in manifest["trials"]]
    summary=dict(coverage(rows),schema_version=config["schema_version"],experiment_id=manifest["experiment_id"],profile=profile,
                 raw_artifact_inventory_sha256=inventory,objective_comparison_policy="within_same_lambda_and_feature_weights_only")
    summary["status"]="COMPLETE" if summary["executed"]==summary["planned"] and not summary["validation_pending"] else "PARTIAL"
    grouped=defaultdict(list)
    for row in rows:
        policy=row["traversability"]
        key=(row.get("family"),row["planner_id"],row["lambda"],policy.get("slope_weight"),policy.get("clearance_weight"),policy.get("clearance_comfort_m"))
        grouped[key].append(row)
    summary["groups"]=[]
    for key,group in grouped.items():
        family,planner,lam,sw,cw,comfort=key
        summary["groups"].append(dict(family=family,planner_id=planner,**{"lambda":lam},slope_weight=sw,clearance_weight=cw,
            clearance_comfort_m=comfort,**coverage(group),metrics={name:distribution([r.get(name) for r in group],len(group)) for name in
            (*METRICS,"first_solution_time_s","first_solution_end_to_end_s","actual_solve_wall_s","end_to_end_run_wall_s")},
            computational_metrics={name:distribution([r["metrics"].get(name) for r in group],len(group)) for name in (*COUNTERS,*TIMINGS,"traversability_preprocessing_s")}))
    curves=[dict(trial_id=r["trial_id"],planner_id=r["planner_id"],**{"lambda":r["lambda"]},budget_s=r["time_budget_s"],
                 points=[{key:e.get(key) for key in ("elapsed_s","snapshot_id",*METRICS)} for e in r["events"] if e["elapsed_s"]<=r["time_budget_s"]],
                 usable=not r["errors"],success=r["success"]) for r in rows]
    output.mkdir(parents=True,exist_ok=True)
    # The completion marker is published last, and removed before a new attempt.
    if (output/"summary.json").exists():os.replace(output/"summary.json",output/".summary.previous.json")
    with tempfile.TemporaryDirectory(prefix=output.name+".staging.",dir=output.parent) as temp:
        stage=Path(temp)
        dump(stage/"trial_metrics.json",rows);dump(stage/"anytime_curves.json",curves)
        fields=("trial_id","family","map_id","query_id","planner_id","lambda","status","executed","success",
                *METRICS,"first_solution_time_s","first_solution_end_to_end_s","actual_solve_wall_s","end_to_end_run_wall_s")
        with (stage/"trials.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for row in rows:writer.writerow({key:row.get(key) for key in fields})
        summary["figures"],summary["figure_error"]=plot_results(stage,rows) if make_plots else ([],None)
        summary["figure_status"]="CREATED" if summary["figures"] else "BLOCKED" if summary["figure_error"] else "NOT_RUN"
        lines=["# A3 Traversability — báo cáo mô tả", "",
               f"{summary['status']}: {summary['executed']}/{summary['planned']} trial đã chạy; {summary['validated_success']} thành công đã kiểm chứng.",
               f"Timeout {summary['timeout']}, crash {summary['crash']}, watchdog {summary['watchdog']}, invalid {summary['invalid']}; chưa chạy {summary['not_run']}.","",
               "Mỗi bộ L/R/J/clearance/slope tại budget thuộc cùng snapshot hợp lệ cuối cùng. L có thể tăng khi J giảm. Không lấy min từng thành phần và không xếp hạng J giữa các lambda hoặc các trọng số khác nhau.","",
               "T_first là timestamp callback của snapshot được kiểm chứng độc lập sau solve, không phải thời điểm hoàn thành kiểm chứng. End-to-end first cộng load/preprocess/setup; end-to-end run cộng worker wall và offline validation. Metric thiếu giữ null.","",
               "Clearance mới chỉ tính tới vật cản rectangle tường minh, trừ bán kính robot và margin; không có vật cản thì null kèm trạng thái. Slope mới là cực đại trên footprint quét. Hard constraints vẫn kiểm toàn capsule, NoData và biên.","",
               "Thống kê quality có điều kiện theo thành công; mọi trial thất bại vẫn ở mẫu số success và coverage. Không coi timeout là vô nghiệm hoặc EXACT_SOLUTION là chứng minh tối ưu. Smoke chỉ kiểm pipeline, không chứng minh ưu thế hay Pareto frontier.","",
               "| Family | Planner | Lambda | Success / executed / planned | L median | R median | J median |", "|---|---|---:|---:|---:|---:|---:|"]
        for g in summary["groups"]:
            show=lambda key: "N/A" if g["metrics"][key]["median"] is None else f"{g['metrics'][key]['median']:.6g}"
            lines.append(f"| {g['family']} | {g['planner_id']} | {g['lambda']} | {g['validated_success']}/{g['executed']}/{g['planned']} | {show('cost_terrain_m')} | {show('risk_integral_m')} | {show('objective_value')} |")
        if summary["figure_error"]:lines += ["", "Biểu đồ BLOCKED: "+summary["figure_error"]]
        (stage/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
        dump(stage/"summary.json",summary)
        for path in stage.iterdir():
            if path.name!="summary.json":os.replace(path,output/path.name)
        os.replace(stage/"summary.json",output/"summary.json")
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--input",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--no-plots",action="store_true");args=parser.parse_args()
    result=analyze(args.input,args.output,not args.no_plots)
    raise SystemExit(1 if result["invalid"] else 0)
