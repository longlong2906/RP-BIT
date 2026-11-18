"""Frozen corridor experiment registration; inventory never launches planners."""
from __future__ import annotations

from pathlib import Path
from run_safe_length import digest, read
from safe_length_field import file_hash
from safe_length_v2_protocol import (SMOKE_FAMILIES, FAMILIES, source_hashes,
                                    request_for as previous_request, default_config as previous_config)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_corridor_v3.json"
PROTOCOL_ID = "a3t_corridor_v3"
ALPHAS = (1.05, 1.10, 1.20)


def default_config():
    config = previous_config()
    config.update(schema_version=3, protocol_id=PROTOCOL_ID)
    config.pop("calibration")
    config["corridor_policy"] = dict(alpha=1.1, label_bins=16, dilation_cells=1,
        max_labels=2000000, max_memory_bytes=536870912)
    config["guidance"] = dict(grid_spacing_m=.1, endpoint_radius_m=.2,
        risk_ceiling=40., slope_cap_deg=35., exposure="integral_risk_ds_surface",
        guide_selection="whole_path_unique_cell_PRI_then_length_then_node_sequence",
        guide_used_as_incumbent=False, immutable_single_corridor=True,
        approximation="16_slack_bins_per_node; no_continuous_or_PRI_optimality_claim")
    config["profiles"]["smoke"].pop("risk_slack")
    config["profiles"]["smoke"]["alpha"] = 1.1
    pilot = config["profiles"]["pilot"]
    pilot.pop("risk_slacks")
    pilot["alphas"] = list(ALPHAS)
    # These new held-out seeds are disjoint from v2's development and reserve.
    config["profiles"]["full"].update(map_seeds=list(range(33001,33011)),
        planner_seeds=list(range(336001,336011)))
    config["analysis"].update(bootstrap_seed=83158,
        selection_order=["paired_candidate_mean_length","paired_candidate_mean_pri","alpha"])
    return config


def load_config(path=CONFIG):
    value = read(path)
    if digest(value) != digest(default_config()):
        raise ValueError("unregistered corridor configuration; create a new protocol")
    return value


def alpha_suffix(value):
    return format(value,"g").replace(".","p")


def method_specs(profile, selected_alpha=None):
    if profile == "smoke":
        return [("a3t_corridor_v3",1.1),("a3t_cap40",None),
                ("sem_native",None),("sem_common_corridor_v3",1.1)]
    if profile == "pilot":
        return [("a3t_corridor_a"+alpha_suffix(a),a) for a in ALPHAS]+[
            ("sem_common_corridor_a"+alpha_suffix(a),a) for a in ALPHAS]+[
            ("a3t_v2_d0",None),("a3t_cap40",None),("sem_native",None)]
    if profile == "full":
        if selected_alpha is not None and (type(selected_alpha) not in (int,float) or selected_alpha not in ALPHAS):
            raise ValueError("full alpha must be selected from registered pilot candidates")
        return [("a3t_corridor_v3",selected_alpha),("sem_common_corridor_v3",selected_alpha),
                ("a3t_cap40",None),("sem_native",None)]
    raise ValueError("unknown corridor profile")


def inventory(config, profile="smoke", selected_alpha=None):
    if digest(config) != digest(default_config()):
        raise ValueError("unregistered config")
    settings, specs = config["profiles"][profile], method_specs(profile,selected_alpha)
    entries = []
    for family in settings["families"]:
        for seed in settings["map_seeds"]:
            map_id = f"{family}_{seed}"
            for planner_seed in settings["planner_seeds"]:
                offset = len(entries)//len(specs) % len(specs)
                for method,alpha in specs[offset:]+specs[:offset]:
                    entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}",map_id=map_id,
                        family=family,map_seed=seed,planner_seed=planner_seed,method=method,
                        alpha=alpha,risk_slack=0. if method=="a3t_v2_d0" else None,status="NOT_RUN"))
    return dict(protocol_id=PROTOCOL_ID,schema_version=3,profile=profile,status="PREPARED_INVENTORY",
        config_sha256=digest(config),entries=entries,planned_trials=len(entries),
        map_count=len(settings["families"])*len(settings["map_seeds"]),
        planning_budget_s=settings["planning_budget_s"],requires_selected_pilot=profile=="full",
        planner_execution_count=0,seed_replacements=[])


def check_selection(path, config, *, binary_sha256=None):
    from analyze_corridor_v3 import build_report, select_pilot
    selection = read(path)
    if (selection.get("status")!="SELECTED" or selection.get("protocol_id")!=PROTOCOL_ID or
            selection.get("config_sha256")!=digest(config)):
        raise ValueError("full requires a SELECTED artifact for this corridor protocol")
    for kind in ("manifest","summary"):
        evidence=Path(selection["pilot_"+kind+"_path"])
        if not evidence.is_file() or file_hash(evidence)!=selection["pilot_"+kind+"_sha256"]:
            raise ValueError("pilot evidence changed: "+kind)
    manifest,summary=read(selection["pilot_manifest_path"]),read(selection["pilot_summary_path"])
    if (manifest.get("profile")!="pilot" or manifest.get("status")!="COMPLETE" or
            manifest.get("synthetic_fixture") or summary.get("synthetic_fixture") or
            any(row.get("synthetic_fixture") for row in summary.get("rows",[]))):
        raise ValueError("selection does not reference a complete real pilot")
    if manifest.get("sources")!=source_hashes() or selection.get("sources_sha256")!=digest(source_hashes()):
        raise ValueError("code changed after pilot")
    if (selection.get("binary_sha256")!=manifest.get("binary_sha256") or
            binary_sha256 is not None and binary_sha256!=selection.get("binary_sha256")):
        raise ValueError("full worker differs from selected pilot worker")
    actual=select_pilot(build_report(summary,manifest,config))
    for key in ("status","selected_alpha","selected_method"):
        if actual.get(key)!=selection.get(key):
            raise ValueError("pilot selection gate mismatch: "+key)
    return selection


def request_for(entry,info,config,profile,field_path=None,field_metadata=None,bridge=None):
    method=entry["method"]
    specs=dict(method_specs(profile,entry.get("alpha")))
    if method not in specs or entry.get("alpha")!=specs[method]:
        raise ValueError("unregistered corridor method/alpha")
    if method in ("sem_native","a3t_cap40","a3t_v2_d0"):
        old_entry=dict(entry,method="a3t_v2_d0" if method=="a3t_v2_d0" else method,
            risk_slack=0. if method=="a3t_v2_d0" else None)
        # v2 pilot methods include all three controls; override only the budget.
        old_config=previous_config()
        old_config["profiles"]["pilot"]["planning_budget_s"]=config["profiles"][profile]["planning_budget_s"]
        result=previous_request(old_entry,info,old_config,"pilot",field_path,field_metadata,bridge)
        result["variant_id"]=method
        return result
    alpha=entry.get("alpha")
    if type(alpha) not in (int,float) or alpha not in ALPHAS:
        raise ValueError("corridor request requires locked pilot alpha")
    common=method.startswith("sem_common_corridor")
    policy=dict(config["corridor_policy"],alpha=alpha)
    safety=dict(field_path=str(Path(field_path).resolve()),slope_cap_deg=35.,risk_cap=40.,
        atlas_sampling=not common,policy_id=PROTOCOL_ID,corridor_policy=policy,
        **{key:field_metadata[key] for key in ("field_sha256","cloud_sha256","builder_source_sha256")})
    result=dict(planner_id="sem_rrtstar_common_v1" if common else "bitstar_a3",variant_id=method,
        map_path=info["map_path"],planner_seed=entry["planner_seed"],start=info["start"],goal=info["goal"],
        planning_budget_s=config["profiles"][profile]["planning_budget_s"],
        samples_per_batch=200 if common else 100,batch_after_first_solution=0 if common else 200,
        rewire_factor=1.1,sample_attempts=100,progress_interval_s=.05,collision_index=True,
        state_cache=True,scalar_cost=True,sampler_query_cache=not common,exact_flat_certificate=False,
        objective_type="surface_length",cloud_phase_handshake=True,safe_length=safety)
    if common:
        result.update(sem_range_m=1.,sem_neighbor_radius_m=3.)
    return result
