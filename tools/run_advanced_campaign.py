"""Run the locked A3-vs-ABIT*/AIT*/LazyPRM* finite-budget campaign.

Dataset generation/qualification and build/test/smoke gates happen before this
command. The campaign clock starts immediately before the first four-planner
problem block and covers collection, offline validation, analysis, and report.
"""
import argparse
import datetime as dt
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from experiment_common import (ROOT,default_worker,digest,dump,environment,
                               file_hash,load_config,read_json)
from independent_verifier import Oracle
from run_experiments import (ADVANCED_PLANNERS,TRIAL_ORDER_COMBINED,TRIAL_ORDER_LATIN,
                             TRIAL_ORDER_SEM_COMPARISON,archive_source,
                             audit_combined_plan,audit_latin_plan,audit_sem_comparison_plan,plan,
                             provenance_identity,refresh_manifest,run_trial,utc,
                             verify_resume_identity)
from validate_results import validate_trial


# Every externally-running stage is stopped with enough budget left to persist
# its terminal record. These reserves are part of harness engineering, not an
# extra planner budget.
FINAL_RECORD_RESERVE_S=2.0
MIN_ANALYSIS_RUNTIME_S=1.0
RAW_INVENTORY_NAME="raw_artifact_inventory.json"
RAW_INVENTORY_EXCLUDED={"campaign_state.json","analysis_attempt.json",RAW_INVENTORY_NAME}
REQUIRED_REPORT_ARTIFACTS={"summary.json","REPORT.md","paired_bootstrap.json",
                           "anytime_curves.json","trial_metrics.json","trials.csv",
                           "reference_targets.json"}
SEM_REQUIRED_REPORT_ARTIFACTS={"summary.json","REPORT.md","paired_bootstrap.json",
                               "trial_metrics.json","trials.csv","checkpoints.json",
                               "metric_coverage.json","descriptive_metrics.json"}


def require_explicit_allow(allowed):
    if not allowed:
        raise ValueError("advanced benchmark requires explicit --allow-advanced-benchmark")


def campaign_descriptor(config):
    """Return the locked profile/protocol shape without weakening either schema."""
    if "combined_benchmark" in config:
        return {"profile":"combined","protocol_key":"combined_benchmark",
                "protocol":config["combined_benchmark"],"dataset_schema_version":5,
                "generator_version":"5.0.0","order_policy":TRIAL_ORDER_COMBINED,
                "classification":"LOCKED_A3_VS_ALL_V1",
                "protocol_hash_key":"combined_protocol_sha256"}
    if "sem_comparison" in config:
        return {"profile":"sem_comparison","protocol_key":"sem_comparison",
                "protocol":config["sem_comparison"],"dataset_schema_version":4,
                "generator_version":"4.0.0","order_policy":TRIAL_ORDER_SEM_COMPARISON,
                "classification":"LOCKED_A3_VS_SEM_RRTSTAR_COMMON_V1",
                "protocol_hash_key":"sem_comparison_protocol_sha256"}
    return {"profile":"advanced","protocol_key":"advanced_benchmark",
            "protocol":config["advanced_benchmark"],"dataset_schema_version":3,
            "generator_version":"3.0.0","order_policy":TRIAL_ORDER_LATIN,
            "classification":"LOCKED_A3_VS_ADVANCED_V1",
            "protocol_hash_key":"advanced_protocol_sha256"}


class CampaignClock:
    """Cumulative active campaign time, persisted between resume sessions."""
    def __init__(self,prior_elapsed_s=0.):
        if not isinstance(prior_elapsed_s,(int,float)) or prior_elapsed_s<0:
            raise ValueError("invalid persisted campaign elapsed time")
        self.prior=float(prior_elapsed_s);self.session_start=None

    def start(self):
        if self.session_start is None:self.session_start=time.perf_counter()

    def elapsed(self):
        return self.prior+(time.perf_counter()-self.session_start if self.session_start is not None else 0.)

    def remaining(self,deadline_s):
        return deadline_s-self.elapsed()


def stage_elapsed(current_elapsed, previous_state, field, completed_before_resume):
    """Keep an already-completed stage's original checkpoint across resume.

    The cumulative campaign clock must continue increasing, but a report-only
    resume must not relabel its start as the end of collection or validation.
    If the stored checkpoint is absent or inconsistent, retain the freshly
    measured value rather than manufacturing a stage timestamp.
    """
    if completed_before_resume:
        previous=previous_state.get(field)
        if (isinstance(previous,(int,float)) and not isinstance(previous,bool) and
                math.isfinite(previous) and 0.<=previous<=current_elapsed):
            return float(previous)
    return current_elapsed


def _tree_overlap(first,second):
    first,second=Path(first).resolve(),Path(second).resolve()
    return first==second or first in second.parents or second in first.parents


def require_disjoint_output_trees(output,report_output):
    if _tree_overlap(output,report_output):
        raise ValueError("raw campaign output and report output must be separate, non-nested directories")


def _inventory_excluded(path,root):
    relative=path.relative_to(root)
    return (len(relative.parts)==1 and relative.name in RAW_INVENTORY_EXCLUDED) or \
        path.name.endswith(".tmp") or any(part.startswith(".analysis-staging-") for part in relative.parts)


def raw_artifact_entries(output):
    """Hash the immutable raw tree consumed by analysis.

    Mutable orchestration records are deliberately excluded. Map bytes live in
    the frozen dataset and are independently rehashed by verify_dataset; their
    exact hashes and query definitions are captured by map_manifest.json.
    """
    output=Path(output).resolve();entries=[]
    for path in sorted(output.rglob("*"),key=lambda value:value.relative_to(output).as_posix()):
        if not path.is_file() or _inventory_excluded(path,output):
            continue
        if path.is_symlink():
            raise ValueError("raw artifact inventory forbids symlinks: "+str(path))
        entries.append({"path":path.relative_to(output).as_posix(),"size_bytes":path.stat().st_size,
                        "sha256":file_hash(path)})
    return entries


def freeze_raw_artifacts(output):
    output=Path(output);entries=raw_artifact_entries(output)
    inventory={"schema_version":1,"entries":entries,"inventory_sha256":digest(entries)}
    dump(output/RAW_INVENTORY_NAME,inventory)
    return inventory


def verify_raw_artifacts(output):
    output=Path(output);path=output/RAW_INVENTORY_NAME
    if not path.exists():
        raise ValueError("raw artifact inventory is absent")
    inventory=read_json(path)
    if set(inventory)!={"schema_version","entries","inventory_sha256"} or inventory["schema_version"]!=1:
        raise ValueError("raw artifact inventory schema mismatch")
    if inventory["entries"]!=sorted(inventory["entries"],key=lambda item:item.get("path","")):
        raise ValueError("raw artifact inventory entries are not sorted")
    if digest(inventory["entries"])!=inventory["inventory_sha256"]:
        raise ValueError("raw artifact inventory digest is malformed")
    current=raw_artifact_entries(output)
    if current!=inventory["entries"]:
        raise ValueError("immutable raw artifacts changed after analysis input freeze")
    return inventory


def _report_artifacts(report_output,raw_manifest_sha256,raw_inventory_sha256):
    report_output=Path(report_output)
    if not (report_output/"summary.json").is_file():
        raise ValueError("analysis report missing required artifacts: summary.json")
    summary=read_json(report_output/"summary.json")
    required=(SEM_REQUIRED_REPORT_ARTIFACTS
              if summary.get("protocol_id")=="a3_vs_sem_rrtstar_common_v1"
              else REQUIRED_REPORT_ARTIFACTS)
    missing=sorted(name for name in required if not (report_output/name).is_file())
    if missing:
        raise ValueError("analysis report missing required artifacts: "+", ".join(missing))
    if summary.get("raw_manifest_sha256")!=raw_manifest_sha256:
        raise ValueError("analysis summary does not match frozen raw manifest")
    if summary.get("raw_artifact_inventory_sha256")!=raw_inventory_sha256:
        raise ValueError("analysis summary does not match frozen raw artifact inventory")
    if summary.get("status") not in ("COMPLETE","PARTIAL","FAILED_CORRECTNESS"):
        raise ValueError("analysis summary has an invalid status")
    figures=summary.get("figures",[])
    if not isinstance(figures,list) or any(not isinstance(name,str) or Path(name).name!=name for name in figures):
        raise ValueError("analysis summary contains an unsafe figure name")
    missing_figures=[name for name in figures if not (report_output/name).is_file()]
    if missing_figures:
        raise ValueError("analysis report missing declared figures: "+", ".join(missing_figures))
    entries=[]
    for path in sorted(report_output.rglob("*"),key=lambda value:value.relative_to(report_output).as_posix()):
        if not path.is_file():
            continue
        if path.is_symlink():
            raise ValueError("analysis artifacts forbid symlinks: "+str(path))
        entries.append({"path":path.relative_to(report_output).as_posix(),"size_bytes":path.stat().st_size,
                        "sha256":file_hash(path)})
    return summary,{"entries":entries,"inventory_sha256":digest(entries)}


def _analysis_not_started(output,status,remaining,**extra):
    record={"status":status,"exit_code":None,"remaining_s_at_start":remaining,**extra}
    dump(Path(output)/"analysis_attempt.json",record)
    return record


def _parse_utc(value):
    return dt.datetime.fromisoformat(value).astimezone(dt.timezone.utc)


def recovered_elapsed(state,config,now=None):
    """Recover uncheckpointed active-phase time after hard process death.

    State is checkpointed immediately before every planner subprocess. If the
    harness dies, an active trial is bounded by its watchdog; other running
    phases are bounded by their protocol checkpoint and never by more than the
    remaining 7200-second campaign budget. Cleanly interrupted/finalized states
    need no reconstruction.
    """
    elapsed=state.get("elapsed_s")
    elapsed=0. if elapsed is None else float(elapsed)
    if state.get("status")!="RUNNING":
        return elapsed,0.
    now=now or dt.datetime.now(dt.timezone.utc)
    checkpoint=(state.get("active_trial_started_utc") if state.get("phase")=="ACTIVE_TRIAL" else
                state.get("session_checkpoint_utc"))
    if not checkpoint:
        return elapsed,0.
    began=_parse_utc(checkpoint)
    observed=max(0.,(now-began).total_seconds())
    descriptor=campaign_descriptor(config);protocol=descriptor["protocol"]
    phase=state.get("phase")
    if phase=="ACTIVE_TRIAL" and state.get("active_trial_id"):
        base=float(state.get("elapsed_before_active_trial_s",elapsed))
        phase_limit=(config["profiles"][descriptor["profile"]]["planning_budget_s"]+
                     config["execution"]["watchdog_grace_s"])
    else:
        base=elapsed
        deadline={"COLLECTION":protocol["collection_soft_stop_s"],
                  "OFFLINE_VALIDATION":protocol["validation_checkpoint_s"],
                  "ANALYSIS_REPORT":protocol["hard_finalize_s"]}.get(phase,protocol["hard_finalize_s"])
        phase_limit=max(0.,deadline-base)
    hard_remaining=max(0.,protocol["hard_finalize_s"]-base)
    recovered=min(observed,phase_limit,hard_remaining)
    return min(protocol["hard_finalize_s"],max(elapsed,base+recovered)),recovered


def verify_dataset(config,maps):
    descriptor=campaign_descriptor(config);profile=descriptor["profile"]
    protocol=descriptor["protocol"];settings=config["profiles"][profile]
    if (maps.get("schema_version")!=descriptor["dataset_schema_version"] or
            maps.get("generator_version")!=descriptor["generator_version"]):
        raise ValueError(f"{profile} dataset schema/generator version mismatch")
    if maps.get("profile")!=profile or maps.get("held_out") is not True:
        raise ValueError(f"{profile} campaign requires its held-out dataset")
    if maps.get("config_hash")!=digest(config):
        raise ValueError(f"generate maps for this exact {profile} config first")
    if maps.get("seed_replacement_policy")!="forbidden" or maps.get("seed_replacements")!=[]:
        raise ValueError("held-out dataset seed replacement policy mismatch")
    if maps.get("qualification_policy")!=("all declared family/seed maps and q0 witnesses must pass before "
                                          "planner execution; abort, never replace"):
        raise ValueError("held-out dataset qualification policy mismatch")
    if maps.get("declared_map_seeds")!=settings["map_seeds"]:
        raise ValueError("held-out dataset seed declaration mismatch")
    expected={(family,seed) for family in settings["families"] for seed in settings["map_seeds"]}
    items=maps.get("maps",[])
    if not isinstance(items,list):
        raise ValueError(f"{profile} dataset maps must be a list")
    actual={(item["family"],item["map_seed"]) for item in items}
    if actual!=expected or len(maps.get("maps",[]))!=len(expected):
        raise ValueError("held-out dataset must contain every declared family/seed exactly once")
    map_ids=[item.get("map_id") for item in items]
    map_paths=[str(Path(item.get("path","")).resolve()) for item in items]
    if len(set(map_ids))!=len(items) or len(set(map_paths))!=len(items):
        raise ValueError(f"{profile} dataset has duplicate map ids or paths")
    for item in maps["maps"]:
        expected_map_id=f'{item["family"]}_{item["map_seed"]}'
        if item.get("map_id")!=expected_map_id:
            raise ValueError(f"{profile} dataset map id mismatch: "+str(item.get("map_id")))
        map_path=Path(item["path"])
        if not map_path.is_file() or map_path.is_symlink() or file_hash(map_path)!=item.get("sha256"):
            raise ValueError("map data changed: "+item["map_id"])
        sidecar=map_path.with_suffix(".json")
        if not sidecar.is_file() or sidecar.is_symlink():
            raise ValueError(f"{profile} map metadata sidecar absent: "+item["map_id"])
        metadata=read_json(sidecar)
        for key,value in (("map_id",item["map_id"]),("family",item["family"]),
                          ("generator_seed",item["map_seed"]),("data_sha256",item["sha256"])):
            if metadata.get(key)!=value:
                raise ValueError(f"{profile} map metadata mismatch: {item['map_id']} / {key}")
        for key,value in (("generator_version",descriptor["generator_version"]),("dataset_role","held_out_evaluation"),
                          ("seed_selection","predeclared_no_replacement_no_planner_results_used")):
            if metadata.get(key)!=value:
                raise ValueError(f"{profile} map qualification metadata mismatch: {item['map_id']} / {key}")
        if metadata.get("queries")!=item.get("queries"):
            raise ValueError(f"{profile} map/query sidecar mismatch: "+item["map_id"])
        if not isinstance(item.get("queries"),list):
            raise ValueError(f"{profile} map queries must be a list: "+item["map_id"])
        query_ids=[query.get("query_id") for query in item["queries"]]
        if query_ids!=protocol["required_query_ids"]:
            raise ValueError(f"{profile} dataset query set mismatch: "+item["map_id"])
        oracle=Oracle(map_path)
        absolute=config["model"]["numerical_abs_tolerance_m"]
        relative=config["model"]["numerical_rel_tolerance"]
        def independently_matches(path_xy,record,start,goal):
            checked=oracle.validate_path(path_xy,start,goal)
            expected_cost=record.get("cost_terrain_m")
            actual_cost=checked.get("cost_terrain_m")
            close=(isinstance(expected_cost,(int,float)) and isinstance(actual_cost,(int,float)) and
                   math.isfinite(expected_cost) and math.isfinite(actual_cost) and
                   abs(expected_cost-actual_cost)<=absolute+relative*max(abs(expected_cost),abs(actual_cost)))
            return checked.get("valid") is True and record.get("valid") is True and close
        for query in item["queries"]:
            for key in ("start","goal"):
                point=query.get(key)
                if (not isinstance(point,list) or len(point)!=2 or
                        any(not isinstance(value,(int,float)) or isinstance(value,bool) or
                            not math.isfinite(value) for value in point)):
                    raise ValueError(f"{profile} dataset invalid {key}: {item['map_id']}/{query.get('query_id')}")
            expected_query_hash=digest({"start":query["start"],"goal":query["goal"],
                                        "map_hash":item["sha256"],"model":config["model"]})
            if query.get("query_hash")!=expected_query_hash:
                raise ValueError(f"{profile} dataset query hash mismatch: "+item["map_id"])
            if (query.get("feasibility")!="feasible_witness_validated" or
                    not query.get("reference",{}).get("validation",{}).get("valid")):
                raise ValueError(f"{profile} dataset lacks a validated frozen witness: "+item["map_id"])
            reference=query["reference"]
            if reference.get("map_sha256")!=item["sha256"] or not reference.get("provenance"):
                raise ValueError(f"{profile} dataset reference provenance mismatch: "+item["map_id"])
            reference_path=reference.get("path_xy")
            reference_hash=(digest(reference_path) if isinstance(reference_path,list) else
                            file_hash(reference_path) if isinstance(reference_path,str) and Path(reference_path).is_file() else None)
            if reference_hash!=reference.get("path_sha256"):
                raise ValueError(f"{profile} dataset reference path hash mismatch: "+item["map_id"])
            if (not isinstance(reference_path,list) or
                    not independently_matches(reference_path,reference.get("validation",{}),query["start"],query["goal"]) or
                    not independently_matches(reference_path,{"valid":True,"cost_terrain_m":reference.get("cost_terrain_m")},
                                              query["start"],query["goal"])):
                raise ValueError(f"{profile} dataset reference fails independent verification: "+item["map_id"])
            for witness in query.get("witnesses",[]):
                if (not isinstance(witness.get("path_xy"),list) or
                        digest(witness["path_xy"])!=witness.get("sha256") or
                        not witness.get("validation",{}).get("valid") or
                        not independently_matches(witness["path_xy"],witness["validation"],
                                                  query["start"],query["goal"])):
                    raise ValueError(f"{profile} dataset witness mismatch: "+item["map_id"])
            direct=query.get("direct_edge_validation")
            if not isinstance(direct,dict):
                raise ValueError(f"{profile} dataset direct-edge qualification absent: "+item["map_id"])
            recomputed_direct=oracle.validate_path([query["start"],query["goal"]],query["start"],query["goal"])
            if bool(recomputed_direct.get("valid"))!=bool(direct.get("valid")):
                raise ValueError(f"{profile} dataset direct-edge qualification changed: "+item["map_id"])


def planner_blocks(trials,expected_size=None):
    blocks=[];offset=0
    while offset<len(trials):
        slot=trials[offset].get("problem_slot_id");end=offset+1
        while end<len(trials) and trials[end].get("problem_slot_id")==slot:end+=1
        block=trials[offset:end]
        if slot is None or len({entry.get("problem_slot_id") for entry in block})!=1:
            raise ValueError("planner block mixes problem slots")
        if expected_size is not None and len(block)!=expected_size:
            raise ValueError("planner block size differs from the locked protocol")
        blocks.append(block);offset=end
    return blocks


def collection_block_allowed(elapsed_s,protocol,block_started):
    """Hard deadline dominates; soft deadline only forbids a new block."""
    return (elapsed_s<protocol["hard_finalize_s"] and
            (block_started or elapsed_s<protocol["collection_soft_stop_s"]))


def campaign_counts(output,manifest):
    output=Path(output)
    executed=validated=success=invalid=0
    for entry in manifest["trials"]:
        path=output/entry["trial_id"]/"trial.json"
        if not path.exists():continue
        trial=read_json(path)
        executed_trial=trial.get("execution_status") not in ("BLOCKED","NOT_RUN","PLANNED")
        executed+=executed_trial
        directory=output/entry["trial_id"]
        offline_done=validation_is_terminal(directory,trial)
        if offline_done and executed_trial:
            validated+=1
            validation_path=directory/"validation.json"
            validation=read_json(validation_path) if validation_path.exists() else {}
            invalid+=bool((trial.get("path_available") and not validation.get("valid")) or trial.get("snapshots_valid") is False)
        success+=bool(executed_trial and offline_done and trial.get("success"))
    return {"planned":len(manifest["trials"]),"executed":executed,"not_run":len(manifest["trials"])-executed,
            "offline_validated":validated,"validation_pending":executed-validated,
            "validated_success":success,"geometric_invalid":invalid}


def validation_is_terminal(directory,trial):
    """A validation is committed only by the final atomic trial.json write."""
    if trial.get("validation_status")=="NO_REQUEST_INTERRUPTED_OR_BLOCKED":
        return True
    wall=trial.get("validation_wall_s")
    return ((Path(directory)/"validation.json").is_file() and
            isinstance(wall,(int,float)) and not isinstance(wall,bool) and
            math.isfinite(wall) and wall>=0.)


def final_campaign_status(counts,analysis_record,deadline_met):
    if counts["geometric_invalid"] or analysis_record.get("summary_status")=="FAILED_CORRECTNESS":
        return "FAILED_CORRECTNESS"
    if (not deadline_met or counts["not_run"] or counts["validation_pending"] or
            analysis_record.get("status")!="COMPLETE" or
            analysis_record.get("summary_status")!="COMPLETE"):
        return "PARTIAL"
    return "COMPLETE"


def enforce_final_deadline(output,state,hard_finalize_s,elapsed_s=None):
    if elapsed_s is not None:
        state["elapsed_s"]=elapsed_s
    elapsed=state.get("elapsed_s")
    state["hard_deadline_met"]=(isinstance(elapsed,(int,float)) and math.isfinite(elapsed) and
                                elapsed<hard_finalize_s)
    if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed) or elapsed>=hard_finalize_s:
        state["status"]="PARTIAL" if state.get("status")=="COMPLETE" else state.get("status","PARTIAL")
        state["hard_deadline_met"]=False
        state["deadline_violation_s"]=(None if not isinstance(elapsed,(int,float)) or not math.isfinite(elapsed)
                                       else max(0.,elapsed-hard_finalize_s))
    dump(Path(output)/"campaign_state.json",state)
    return state


def record_state(output,manifest,status,phase,clock,protocol,**extra):
    checkpoint_utc=utc()
    counts=campaign_counts(output,manifest)
    state={"schema_version":1,"protocol_id":protocol["protocol_id"],"status":status,"phase":phase,
           "utc":checkpoint_utc,"session_checkpoint_utc":checkpoint_utc,
           "elapsed_s":None if clock is None else clock.elapsed(),
           "deadlines_s":{"collection_soft_stop":protocol["collection_soft_stop_s"],
                          "validation_checkpoint":protocol["validation_checkpoint_s"],
                          "hard_finalize":protocol["hard_finalize_s"]},
           "timer_scope":protocol["timer_scope"],"counts":counts,**extra}
    dump(Path(output)/"campaign_state.json",state)
    return state


def resolve_campaign_inputs(config_path,map_manifest,worker,campaign_kind="advanced"):
    profiles={"advanced":"advanced","sem_comparison":"sem_comparison","combined":"combined"}
    if campaign_kind not in profiles:
        raise ValueError("unknown campaign kind: "+str(campaign_kind))
    profile=profiles[campaign_kind]
    config=load_config(config_path,profile);descriptor=campaign_descriptor(config)
    maps=read_json(map_manifest)
    verify_dataset(config,maps)
    trials=plan(config,profile,maps,order_policy=descriptor["order_policy"])
    schedule=(audit_sem_comparison_plan(trials,config["profiles"][profile])
              if profile=="sem_comparison" else
              audit_combined_plan(trials,config["profiles"][profile])
              if profile=="combined" else audit_latin_plan(trials))
    schedule.update(shuffle_seed=config["execution"]["shuffle_trial_order_seed"],
                    classification=descriptor["classification"],seed_replacement_policy="forbidden",
                    planner_block_size=descriptor["protocol"]["planner_block_size"])
    if len(trials)!=descriptor["protocol"]["expected_trials"]:
        raise ValueError(f"{profile} campaign trial cardinality mismatch")
    identity=provenance_identity(config,maps,worker)
    identity.update(trial_order_policy=descriptor["order_policy"],schedule_sha256=digest(schedule),
                    plan_sha256=digest(trials),
                    **{descriptor["protocol_hash_key"]:digest(descriptor["protocol"])})
    return config,maps,trials,schedule,identity


def prepare(config_path,map_manifest,output,worker,campaign_kind="advanced"):
    config,maps,trials,schedule,identity=resolve_campaign_inputs(
        config_path,map_manifest,worker,campaign_kind)
    descriptor=campaign_descriptor(config);profile=descriptor["profile"]
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("campaign output is nonempty; attempts are never overwritten or retried")
    output.mkdir(parents=True,exist_ok=True)
    env=environment()
    manifest={"schema_version":3,"experiment_id":output.name,"profile":profile,"identity":identity,
        "environment_sha256":digest(env),
        "planned_trials":len(trials),"trials":trials,"status":"PLANNED","history":[],"created_utc":utc(),
        "schedule":schedule,"attempt_policy":"one primary attempt; no retry, replacement, or outcome-based seed selection",
        "resume_policy":"exact config/code/worker/dependency/patch/scenario/schedule/plan hashes; cumulative persisted elapsed; incomplete attempt becomes CRASH without retry",
        "campaign_clock":"starts immediately before first planner block; includes collection, offline validation, analysis, report",
        "config_field_scope":{"roughness_window_m":"inactive_because_roughness_limit_m_is_null",
                              "logging.save_ompl_benchmark_log":"raw_OMPL_OutputHandler_log_not_Planner_Arena"}}
    archive_source(output,identity,config_path)
    dump(output/"environment.json",env);dump(output/"resolved_config.json",config);dump(output/"map_manifest.json",maps)
    dump(output/"manifest.json",manifest)
    record_state(output,manifest,"PLANNED","PRE_TIMER",None,descriptor["protocol"])
    return config,maps,manifest,env


def load_resume(config_path,map_manifest,output,worker,campaign_kind="advanced"):
    output=Path(output)
    if not (output/"manifest.json").exists() or not (output/"campaign_state.json").exists():
        raise ValueError("--resume requires an existing campaign manifest and state")
    config,maps,trials,schedule,identity=resolve_campaign_inputs(
        config_path,map_manifest,worker,campaign_kind)
    descriptor=campaign_descriptor(config)
    manifest=read_json(output/"manifest.json");state=read_json(output/"campaign_state.json")
    verify_resume_identity(manifest.get("identity"),identity)
    if manifest.get("trials")!=trials or manifest.get("schedule")!=schedule:
        raise ValueError("resume plan/schedule mismatch")
    if read_json(output/"resolved_config.json")!=config or read_json(output/"map_manifest.json")!=maps:
        raise ValueError("resume stored config/scenario mismatch")
    source_manifest=read_json(output/"source_manifest.json")
    if file_hash(output/"source_at_run.zip")!=source_manifest["archive_sha256"]:
        raise ValueError("resume source archive changed")
    if state.get("protocol_id")!=descriptor["protocol"]["protocol_id"]:
        raise ValueError("resume campaign protocol mismatch")
    elapsed,recovered=recovered_elapsed(state,config)
    env=read_json(output/"environment.json")
    if manifest.get("environment_sha256")!=digest(env):
        raise ValueError("resume stored environment record changed")
    current_env=environment()
    if current_env!=env:
        raise ValueError("resume environment drift: interpreter/platform/packages/build environment changed")
    return config,maps,manifest,env,state,elapsed,recovered


def correctness_gate(worker,identity):
    gate_path=Path(worker).resolve().parent/"correctness_gate.json"
    if not gate_path.exists():
        raise RuntimeError("correctness gate absent: run the build/test/smoke gate before the campaign")
    gate=read_json(gate_path)
    if gate.get("worker_sha256")!=identity["worker_sha256"] or gate.get("code_sha256")!=identity["code_sha256"]:
        raise RuntimeError("correctness gate is stale for the selected worker/source")


def retain_interrupted_attempt(output,entry,manifest,config):
    """Convert an existing incomplete attempt into a terminal CRASH, never retry."""
    directory=Path(output)/entry["trial_id"]
    if not directory.exists() or (directory/"trial.json").exists():
        return False
    attempt=read_json(directory/"attempt.json") if (directory/"attempt.json").exists() else None
    profile=campaign_descriptor(config)["profile"]
    trial=dict(entry,schema_version=2,experiment_id=manifest["experiment_id"],
        config_hash=manifest["identity"]["config_hash"],code_sha256=manifest["identity"]["code_sha256"],
        execution_status="CRASH",planner_status="UNKNOWN",independent_valid=None,success=False,
        path_available=False,time_budget_s=config["profiles"][profile]["planning_budget_s"],
        termination_reason="INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY",administrative_interruption=True,
        attempt_history=[attempt,{"utc":utc(),"event":"INTERRUPTED_PREVIOUS_RUN_RETAINED_NO_RETRY"}])
    dump(directory/"trial.json",trial)
    return True


def _validate_batch_directories(directories):
    """Internal subprocess entry point; one Oracle is reused for one map."""
    directories=[Path(directory) for directory in directories]
    oracle=None;map_path=None
    for directory in directories:
        trial=read_json(directory/"trial.json")
        request_path=directory/"request.json"
        if request_path.exists():
            current=read_json(request_path)["map_path"]
            if map_path is None:
                map_path=current;oracle=Oracle(current)
            elif Path(current).resolve()!=Path(map_path).resolve():
                raise ValueError("internal validation batch mixes maps")
        validate_trial(directory,oracle)


def _validation_batch_command(directories):
    return [sys.executable,str(Path(__file__).resolve()),"--internal-validate-batch",
            *[str(Path(directory).resolve()) for directory in directories]]


def validate_until_checkpoint(output,manifest,clock,checkpoint_s,hard_finalize_s=None):
    by_map={}
    for entry in manifest["trials"]:
        trial_path=Path(output)/entry["trial_id"]/"trial.json"
        if trial_path.exists() and read_json(trial_path).get("execution_status") not in ("BLOCKED","NOT_RUN","PLANNED"):
            by_map.setdefault(entry["map_path"],[]).append(entry)
    validated=0;deadline=min(checkpoint_s,hard_finalize_s if hard_finalize_s is not None else checkpoint_s)
    for map_path,entries in by_map.items():
        pending=[]
        for entry in entries:
            directory=Path(output)/entry["trial_id"]
            trial=read_json(directory/"trial.json")
            if validation_is_terminal(directory,trial):
                continue
            pending.append(directory)
        if not pending:
            continue
        remaining=deadline-clock.elapsed()
        if remaining<=FINAL_RECORD_RESERVE_S:
            return validated,False
        command=_validation_batch_command(pending)
        try:
            result=subprocess.run(command,capture_output=True,text=True,
                                  timeout=max(.1,remaining-FINAL_RECORD_RESERVE_S))
        except subprocess.TimeoutExpired:
            return validated,False
        if result.returncode:
            raise RuntimeError("offline validation subprocess failed: "+
                               (result.stderr or result.stdout or "no diagnostic")[-4000:])
        for directory in pending:
            trial=read_json(directory/"trial.json")
            if not validation_is_terminal(directory,trial):
                raise RuntimeError("offline validation subprocess omitted a trial: "+directory.name)
            validated+=1
    return validated,True


def run_analysis_with_deadline(output,report_output,protocol,clock,allow_existing=False):
    output,report_output=Path(output),Path(report_output)
    require_disjoint_output_trees(output,report_output)
    manifest_sha=file_hash(output/"manifest.json")

    # Reuse is permitted only when every raw and report artifact still matches
    # the completed attempt, never merely because summary.json exists.
    if allow_existing and report_output.exists() and any(report_output.iterdir()):
        try:
            inventory=verify_raw_artifacts(output)
            prior=read_json(output/"analysis_attempt.json")
            summary,artifacts=_report_artifacts(report_output,manifest_sha,inventory["inventory_sha256"])
            if (prior.get("status")!="COMPLETE" or
                    prior.get("raw_artifact_inventory_sha256")!=inventory["inventory_sha256"] or
                    prior.get("report_artifact_inventory_sha256")!=artifacts["inventory_sha256"]):
                raise ValueError("existing analysis attempt provenance mismatch")
            record=dict(prior,reused_existing_report=True,wall_s=0.,summary_status=summary["status"])
            dump(output/"analysis_attempt.json",record)
            return record
        except (OSError,ValueError,KeyError):
            # A partial campaign may legitimately add trials on resume. Its old
            # report is preserved during promotion, but never mixed with a new run.
            pass

    remaining=clock.remaining(protocol["hard_finalize_s"])
    if remaining<=FINAL_RECORD_RESERVE_S+MIN_ANALYSIS_RUNTIME_S:
        return _analysis_not_started(output,"NOT_STARTED_INSUFFICIENT_HARD_BUDGET",remaining,
                                     required_reserve_s=FINAL_RECORD_RESERVE_S,
                                     minimum_runtime_s=MIN_ANALYSIS_RUNTIME_S)
    if report_output.exists() and any(report_output.iterdir()) and not allow_existing:
        raise ValueError("report output is nonempty; choose a new report directory")

    inventory=freeze_raw_artifacts(output)
    remaining=clock.remaining(protocol["hard_finalize_s"])
    if remaining<=FINAL_RECORD_RESERVE_S+MIN_ANALYSIS_RUNTIME_S:
        return _analysis_not_started(output,"NOT_STARTED_INSUFFICIENT_HARD_BUDGET",remaining,
                                     required_reserve_s=FINAL_RECORD_RESERVE_S,
                                     minimum_runtime_s=MIN_ANALYSIS_RUNTIME_S,
                                     raw_artifact_inventory_sha256=inventory["inventory_sha256"])

    report_output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=".analysis-staging-"+report_output.name+"-",dir=report_output.parent))
    if protocol.get("protocol_id")=="a3_vs_sem_rrtstar_common_v1":
        # The SEM common-objective protocol has different preregistered
        # estimands (family->map->paired-seed bootstrap and censored RMST).
        # Its analyzer locks those values internally and accepts no CLI
        # overrides that could silently change the analysis after collection.
        analyzer=ROOT/"tools/analyze_a3_sem_rrtstar_common.py"
        analysis_options=[]
    else:
        analyzer=ROOT/"tools/analyze_results.py"
        analysis_options=["--bootstrap-replicates",str(protocol["bootstrap_replicates"]),
                          "--bootstrap-seed",str(protocol["bootstrap_seed"])]
        if protocol.get("protocol_id")=="a3_vs_all_v1":
            # This environment's Pillow DLL cannot currently be imported by
            # matplotlib.  Keep the preregistered JSON/CSV analysis products
            # and make plot omission explicit instead of failing the campaign.
            analysis_options.append("--no-plots")
    command=[sys.executable,str(analyzer),"--input",str(output.resolve()),
             "--output",str(stage.resolve()),*analysis_options]
    began=time.perf_counter();record=None
    try:
        result=subprocess.run(command,capture_output=True,text=True,
                              timeout=max(.1,remaining-FINAL_RECORD_RESERVE_S))
        wall=time.perf_counter()-began
        if result.returncode!=0:
            record={"status":"FAILED_NONZERO_EXIT","exit_code":result.returncode,"wall_s":wall,
                    "command":command,"stdout":result.stdout,"stderr":result.stderr,
                    "staging_output":str(stage.resolve())}
        elif clock.remaining(protocol["hard_finalize_s"])<=FINAL_RECORD_RESERVE_S:
            record={"status":"FAILED_INSUFFICIENT_FINALIZE_BUDGET","exit_code":result.returncode,"wall_s":wall,
                    "command":command,"stdout":result.stdout,"stderr":result.stderr,
                    "staging_output":str(stage.resolve())}
        else:
            current=verify_raw_artifacts(output)
            if current["inventory_sha256"]!=inventory["inventory_sha256"]:
                raise ValueError("raw artifact inventory changed while analyzer was running")
            summary,artifacts=_report_artifacts(stage,manifest_sha,inventory["inventory_sha256"])
            superseded=None
            if report_output.exists():
                if any(report_output.iterdir()):
                    suffix=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
                    superseded=report_output.with_name(report_output.name+".superseded."+suffix)
                    report_output.rename(superseded)
                else:
                    report_output.rmdir()
            try:
                stage.rename(report_output)
            except BaseException:
                if superseded is not None and not report_output.exists():
                    superseded.rename(report_output)
                raise
            record={"status":"COMPLETE","exit_code":result.returncode,"wall_s":wall,
                    "command":command,"stdout":result.stdout,"stderr":result.stderr,
                    "summary_status":summary["status"],"manifest_sha256":manifest_sha,
                    "raw_artifact_inventory_sha256":inventory["inventory_sha256"],
                    "report_artifact_inventory_sha256":artifacts["inventory_sha256"],
                    "report_artifacts":artifacts["entries"],"reused_existing_report":False}
            if superseded is not None:
                record["superseded_report_preserved_at"]=str(superseded.resolve())
    except subprocess.TimeoutExpired as error:
        record={"status":"TERMINATED_BEFORE_HARD_DEADLINE_RESERVE","exit_code":None,
                "wall_s":time.perf_counter()-began,"command":command,
                "stdout":error.stdout.decode(errors="replace") if isinstance(error.stdout,bytes) else error.stdout,
                "stderr":error.stderr.decode(errors="replace") if isinstance(error.stderr,bytes) else error.stderr,
                "staging_output":str(stage.resolve())}
    except (OSError,ValueError,KeyError) as error:
        record={"status":"FAILED_ARTIFACT_VERIFICATION","exit_code":0,"wall_s":time.perf_counter()-began,
                "command":command,"error":type(error).__name__+": "+str(error),
                "staging_output":str(stage.resolve())}
    if record.get("raw_artifact_inventory_sha256") is None:
        record["raw_artifact_inventory_sha256"]=inventory["inventory_sha256"]
    dump(output/"analysis_attempt.json",record)
    return record


def verify_complete_campaign(output,report_output,manifest,state):
    if state.get("status")!="COMPLETE" or state.get("phase")!="FINALIZED":
        raise ValueError("campaign is not a finalized COMPLETE state")
    hard=state.get("deadlines_s",{}).get("hard_finalize")
    if (state.get("hard_deadline_met") is not True or not isinstance(hard,(int,float)) or
            not isinstance(state.get("elapsed_s"),(int,float)) or state["elapsed_s"]>=hard):
        raise ValueError("COMPLETE campaign violated its hard deadline")
    counts=campaign_counts(output,manifest)
    if counts["not_run"] or counts["validation_pending"] or counts["geometric_invalid"]:
        raise ValueError("COMPLETE campaign raw counts no longer satisfy completion")
    inventory=verify_raw_artifacts(output)
    attempt=read_json(Path(output)/"analysis_attempt.json")
    summary,artifacts=_report_artifacts(report_output,file_hash(Path(output)/"manifest.json"),
                                        inventory["inventory_sha256"])
    if (attempt.get("status")!="COMPLETE" or summary.get("status")!="COMPLETE" or
            attempt.get("raw_artifact_inventory_sha256")!=inventory["inventory_sha256"] or
            attempt.get("report_artifact_inventory_sha256")!=artifacts["inventory_sha256"]):
        raise ValueError("COMPLETE campaign analysis provenance/artifacts failed revalidation")
    state_inventory=state.get("raw_artifact_inventory_sha256")
    if state_inventory!=inventory["inventory_sha256"]:
        raise ValueError("COMPLETE campaign state does not match raw artifact inventory")


def run_campaign(config_path,map_manifest,output,report_output,worker,*,resume=False,
                 campaign_kind="advanced"):
    output=Path(output);report_output=Path(report_output)
    require_disjoint_output_trees(output,report_output)
    if not resume and report_output.exists() and any(report_output.iterdir()):
        raise ValueError("report output is nonempty; choose a new report directory")
    if resume:
        config,maps,manifest,env,previous_state,prior_elapsed,recovered=load_resume(
            config_path,map_manifest,output,worker,campaign_kind)
        if previous_state.get("report_output") and Path(previous_state["report_output"]).resolve()!=report_output.resolve():
            raise ValueError("resume report output changed")
    else:
        config,maps,manifest,env=prepare(config_path,map_manifest,output,worker,campaign_kind)
        previous_state=read_json(output/"campaign_state.json");prior_elapsed=recovered=0.
        record_state(output,manifest,"PLANNED","PRE_TIMER",None,campaign_descriptor(config)["protocol"],
                     report_output=str(report_output.resolve()))
    del maps
    descriptor=campaign_descriptor(config);protocol=descriptor["protocol"];profile=descriptor["profile"]
    correctness_gate(worker,manifest["identity"])
    if resume and previous_state.get("status")=="COMPLETE":
        verify_complete_campaign(output,report_output,manifest,previous_state)
        return 0
    blocks=planner_blocks(manifest["trials"],protocol.get("planner_block_size",len(ADVANCED_PLANNERS)))
    clock=CampaignClock(prior_elapsed)
    if clock.elapsed()>=protocol["hard_finalize_s"]:
        manifest["status"]="PARTIAL"
        manifest["history"].append({"utc":utc(),"event":"RESUME_REFUSED_HARD_DEADLINE",
                                    "elapsed_s":clock.elapsed()})
        dump(output/"manifest.json",manifest)
        _analysis_not_started(output,"NOT_STARTED_HARD_DEADLINE",clock.remaining(protocol["hard_finalize_s"]),
                              reason="resume_elapsed_at_or_beyond_hard_deadline")
        record_state(output,manifest,"PARTIAL","FINALIZED",clock,protocol,
                     report_output=str(report_output.resolve()),hard_deadline_met=False,
                     resume_recovered_active_trial_s=recovered)
        return 2

    session_started=False;raw_frozen=False
    def start_session():
        nonlocal session_started
        if session_started:return
        began=dt.datetime.now(dt.timezone.utc).isoformat();clock.start();session_started=True
        manifest["history"].append({"utc":began,
            "event":"CAMPAIGN_CLOCK_RESUMED" if resume or prior_elapsed else "CAMPAIGN_CLOCK_STARTED",
            "prior_elapsed_s":prior_elapsed,"resume_recovered_active_trial_s":recovered,
            "timer_scope":protocol["timer_scope"],"monotonic_origin_not_serialized":True})

    def completed_block_count():
        return sum(all((output/entry["trial_id"]/"trial.json").exists() for entry in block) for block in blocks)

    completed_blocks=completed_block_count()
    collection_complete_before_resume=resume and completed_blocks==len(blocks)
    try:
        hard_stop=False
        for block in blocks:
            if all((output/entry["trial_id"]/"trial.json").exists() for entry in block):
                continue
            block_started=any((output/entry["trial_id"]).exists() for entry in block)
            if not collection_block_allowed(clock.elapsed(),protocol,block_started):
                if clock.elapsed()>=protocol["hard_finalize_s"]:hard_stop=True
                break
            if clock.elapsed()>=protocol["hard_finalize_s"]:
                hard_stop=True;break
            start_session()
            # A directory proves an attempt began. If its terminal wrapper is
            # absent after process death, preserve it as CRASH and never retry.
            for entry in block:
                retain_interrupted_attempt(output,entry,manifest,config)
            for entry in block:
                directory=output/entry["trial_id"]
                if (directory/"trial.json").exists():
                    continue
                maximum_trial_wall=(config["profiles"][profile]["planning_budget_s"]+
                                    config["execution"]["watchdog_grace_s"])
                if clock.remaining(protocol["hard_finalize_s"])<=maximum_trial_wall+FINAL_RECORD_RESERVE_S:
                    hard_stop=True;break
                active_started=utc();elapsed_before=clock.elapsed()
                record_state(output,manifest,"RUNNING","ACTIVE_TRIAL",clock,protocol,
                    completed_blocks=completed_blocks,report_output=str(report_output.resolve()),
                    active_trial_id=entry["trial_id"],active_trial_started_utc=active_started,
                    elapsed_before_active_trial_s=elapsed_before)
                result=run_trial(entry,config,directory,worker,env,manifest["experiment_id"],
                                 digest(config),profile=profile)
                record_state(output,manifest,"RUNNING","COLLECTION",clock,protocol,
                    completed_blocks=completed_blocks,report_output=str(report_output.resolve()))
                print(f'{entry["shuffle_order"]+1}/{len(manifest["trials"])} {entry["trial_id"]}: '
                      f'{result["execution_status"]} / {result["planner_status"]}',flush=True)
            completed_blocks=completed_block_count()
            refresh_manifest(output,manifest)
            record_state(output,manifest,"RUNNING","COLLECTION",clock,protocol,
                         completed_blocks=completed_blocks,report_output=str(report_output.resolve()))
            if hard_stop:break
        collection_elapsed=stage_elapsed(clock.elapsed(),previous_state,"collection_elapsed_s",
                                         collection_complete_before_resume)
        collection_complete=completed_blocks==len(blocks)
        manifest["history"].append({"utc":utc(),"event":"COLLECTION_COMPLETE" if collection_complete else
                                    "COLLECTION_HARD_STOP" if hard_stop else "COLLECTION_SOFT_STOP",
                                    "elapsed_s":collection_elapsed,"completed_blocks":completed_blocks,
                                    "planned_blocks":len(blocks)})
        refresh_manifest(output,manifest)
        if not session_started and clock.elapsed()<protocol["hard_finalize_s"]:
            start_session()
        record_state(output,manifest,"RUNNING","OFFLINE_VALIDATION",clock,protocol,
                     completed_blocks=completed_blocks,collection_elapsed_s=collection_elapsed,
                     report_output=str(report_output.resolve()))
        validation_complete_before_resume=(resume and
            campaign_counts(output,manifest)["validation_pending"]==0)
        if clock.elapsed()<protocol["validation_checkpoint_s"]:
            validated_now,validation_complete=validate_until_checkpoint(
                output,manifest,clock,protocol["validation_checkpoint_s"],protocol["hard_finalize_s"])
        else:
            validated_now=0;validation_complete=campaign_counts(output,manifest)["validation_pending"]==0
        validation_elapsed=stage_elapsed(clock.elapsed(),previous_state,
                                         "validation_checkpoint_elapsed_s",
                                         validation_complete_before_resume)
        manifest["history"].append({"utc":utc(),"event":"VALIDATION_COMPLETE" if validation_complete else "VALIDATION_CHECKPOINT",
                                    "elapsed_s":validation_elapsed,"validated_in_stage":validated_now})
        refresh_manifest(output,manifest,validated=validation_complete)
        preanalysis_counts=campaign_counts(output,manifest)
        manifest["status"]=("FAILED_VALIDATION" if preanalysis_counts["geometric_invalid"] else
                            "PARTIAL" if preanalysis_counts["not_run"] or preanalysis_counts["validation_pending"] else
                            "COMPLETE_VALIDATED")
        manifest["history"].append({"utc":utc(),"event":"ANALYSIS_INPUTS_FINALIZED",
                                    "elapsed_s":clock.elapsed(),"raw_status":manifest["status"]})
        dump(output/"manifest.json",manifest)
        record_state(output,manifest,"RUNNING","ANALYSIS_REPORT",clock,protocol,
                     completed_blocks=completed_blocks,collection_elapsed_s=collection_elapsed,
                     validation_checkpoint_elapsed_s=validation_elapsed,validation_complete=validation_complete,
                     report_output=str(report_output.resolve()))
        analysis_record=run_analysis_with_deadline(output,report_output,protocol,clock,allow_existing=resume)
        raw_frozen=(output/RAW_INVENTORY_NAME).exists()
        counts=campaign_counts(output,manifest)
        final_status=final_campaign_status(
            counts,analysis_record,clock.elapsed()<protocol["hard_finalize_s"])
        state=record_state(output,manifest,final_status,"FINALIZED",clock,protocol,
            completed_blocks=completed_blocks,collection_elapsed_s=collection_elapsed,
            validation_checkpoint_elapsed_s=validation_elapsed,validation_complete=validation_complete,
            report_output=str(report_output.resolve()),analysis=analysis_record,
            hard_deadline_met=clock.elapsed()<protocol["hard_finalize_s"],
            raw_artifact_inventory_sha256=(read_json(output/RAW_INVENTORY_NAME)["inventory_sha256"]
                                           if (output/RAW_INVENTORY_NAME).exists() else None),
            resume_recovered_active_trial_s=recovered)
        # campaign_counts()/serialization are intentionally inside the campaign
        # clock.  If that final bookkeeping itself crosses 7,200 s, never leave
        # a contradictory COMPLETE state or exit code behind.
        state=enforce_final_deadline(output,state,protocol["hard_finalize_s"],clock.elapsed())
        # Detect the only remaining race: the final atomic state write itself
        # may cross the boundary after the fresh elapsed sample above.
        after_final_dump=clock.elapsed()
        if after_final_dump>=protocol["hard_finalize_s"] and state["status"]=="COMPLETE":
            state=enforce_final_deadline(output,state,protocol["hard_finalize_s"],after_final_dump)
        return 0 if state["status"]=="COMPLETE" else 2
    except KeyboardInterrupt:
        raw_frozen=raw_frozen or (output/RAW_INVENTORY_NAME).exists()
        if not raw_frozen:
            manifest["history"].append({"utc":utc(),"event":"ADMINISTRATIVE_INTERRUPT","elapsed_s":clock.elapsed()})
            refresh_manifest(output,manifest)
        record_state(output,manifest,"PARTIAL","INTERRUPTED",clock,protocol,
                     completed_blocks=completed_block_count(),report_output=str(report_output.resolve()),
                     raw_artifact_inventory_sha256=(read_json(output/RAW_INVENTORY_NAME)["inventory_sha256"]
                                                    if raw_frozen else None))
        return 130
    except BaseException as error:
        raw_frozen=raw_frozen or (output/RAW_INVENTORY_NAME).exists()
        if not raw_frozen:
            manifest["history"].append({"utc":utc(),"event":"CAMPAIGN_ERROR","elapsed_s":clock.elapsed(),
                                        "error":type(error).__name__+": "+str(error)})
            refresh_manifest(output,manifest)
        record_state(output,manifest,"FAILED","FINALIZED_AFTER_ERROR",clock,protocol,
                     completed_blocks=completed_block_count(),report_output=str(report_output.resolve()),
                     raw_artifact_inventory_sha256=(read_json(output/RAW_INVENTORY_NAME)["inventory_sha256"]
                                                    if raw_frozen else None),
                     error=type(error).__name__+": "+str(error))
        raise


def main():
    if len(sys.argv)>=3 and sys.argv[1]=="--internal-validate-batch":
        _validate_batch_directories(sys.argv[2:])
        return 0
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/a3_vs_advanced_v1.yaml")
    parser.add_argument("--map-manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report-output",type=Path,required=True)
    parser.add_argument("--worker",type=Path,default=default_worker())
    parser.add_argument("--allow-advanced-benchmark",action="store_true")
    parser.add_argument("--resume",action="store_true",help="continue only with exact identity and cumulative persisted campaign time")
    parser.add_argument("--dry-run",action="store_true",help="validate and print the locked plan without creating campaign output")
    args=parser.parse_args();require_explicit_allow(args.allow_advanced_benchmark)
    if args.dry_run and args.resume:parser.error("--dry-run and --resume are mutually exclusive")
    if args.dry_run:
        config=load_config(args.config,"advanced");maps=read_json(args.map_manifest);verify_dataset(config,maps)
        trials=plan(config,"advanced",maps,order_policy=TRIAL_ORDER_LATIN)
        print({"planned_trials":len(trials),"schedule":audit_latin_plan(trials),
               "campaign_clock":"NOT_STARTED_DRY_RUN"})
        return 0
    return run_campaign(args.config,args.map_manifest,args.output,args.report_output,args.worker,resume=args.resume)


if __name__=="__main__":raise SystemExit(main())
