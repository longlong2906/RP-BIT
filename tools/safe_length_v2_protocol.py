"""Registered A3-T v2 protocols; preparation never launches a planner."""
from __future__ import annotations

from pathlib import Path

from run_safe_length import digest, dump, read, source_hashes as previous_source_hashes
from safe_length_field import file_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_safe_length_v2.json"
PROTOCOL_ID = "a3t_safe_length_v2"
SMOKE_FAMILIES = ("flat", "tilted", "unsafe_short_safe_detour", "two_corridors_slope",
                  "two_corridors_clearance", "nodata_thin_obstacle", "flat_40m")
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
SLACKS = (0., .5, 1., 2., 4.)


def default_config():
    return dict(schema_version=2, protocol_id=PROTOCOL_ID,
        model=dict(state="xy", robot_radius_m=.3, safety_margin_m=.1, max_slope_deg=35.,
                   objective="surface_length", nearest_neighbor_metric="euclidean_xy"),
        calibration=dict(grid_spacing_m=.1, endpoint_radius_m=.2, risk_ceiling=40.,
                         slope_cap_deg=35., policy="immutable_query_minimax_plus_slack",
                         endpoint_radius_tie_ulps=16,
                         endpoint_radius_tie_scale="max(1,abs(endpoint_xy),abs(grid_node_xy),radius)",
                         witness_used_as_incumbent=False),
        algorithm=dict(initial_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
                       sample_attempts=100, just_in_time_sampling=False),
        validation=dict(independent_risk_comparison_ulps=64,
            risk_roundoff_scale="max(1,abs(cap),abs(risk_max))",common_risk_ceiling_strict=True,
            planner_risk_comparison="exact_leq_cap",raw_metrics_unchanged=True),
        profiles=dict(
            smoke=dict(families=list(SMOKE_FAMILIES), map_seeds=[101], planner_seeds=[11],
                       planning_budget_s=2., extent_m=20., spacing_m=.2, risk_slack=.5),
            pilot=dict(families=list(FAMILIES), map_seeds=[31001,31002],
                       planner_seeds=[316001,316002,316003], planning_budget_s=20.,
                       extent_m=40., spacing_m=.2, risk_slacks=list(SLACKS)),
            full=dict(families=list(FAMILIES), map_seeds=list(range(32001,32011)),
                      planner_seeds=list(range(326001,326011)), planning_budget_s=20.,
                      extent_m=40., spacing_m=.2, requires_selected_pilot=True)),
        execution=dict(max_workers=1, threads_per_trial=1, process_watchdog_s=600.,
                       retry_started_trials=False, seed_replacements=[],
                       pilot_requires_explicit_flag=True, full_requires_explicit_flag=True),
        analysis=dict(bootstrap_replicates=2000, bootstrap_seed=83148, confidence=.95,
                      bootstrap_order=["family_fixed","map","paired_seed"],
                      selection_order=["paired_candidate_mean_length","paired_candidate_mean_pri","risk_slack"],
                      final_selection="last_published_incumbent_in_budget_before_validation"))


def load_config(path=CONFIG):
    value = read(path)
    if digest(value) != digest(default_config()):
        raise ValueError("unregistered v2 configuration; create a new protocol rather than modify frozen parameters")
    return value


def method_specs(profile, selected_slack=None):
    if profile == "smoke":
        return [("a3t_v2", .5), ("a3t_cap40", None), ("sem_native", None), ("sem_common_v2", .5)]
    if profile == "pilot":
        return [("a3t_v2_d"+format(v,"g").replace(".","p"),v) for v in SLACKS] + [
            ("a3t_cap40",None),("sem_native",None)]
    if profile == "full":
        if selected_slack is not None and (type(selected_slack) not in (float,int) or selected_slack not in SLACKS):
            raise ValueError("full slack must be selected from the registered pilot candidates")
        return [("a3t_v2",selected_slack),("a3t_cap40",None),("sem_native",None)]
    raise ValueError("unknown v2 profile")


def inventory(config, profile="smoke", selected_slack=None):
    if digest(config) != digest(default_config()):
        raise ValueError("unregistered config")
    settings = config["profiles"][profile]
    specs = method_specs(profile, selected_slack)
    entries = []
    for family in settings["families"]:
        for seed in settings["map_seeds"]:
            map_id = f"{family}_{seed}"
            for planner_seed in settings["planner_seeds"]:
                # Deterministic rotated order, shared seed for every comparator.
                offset = len(entries)//len(specs) % len(specs)
                for method,slack in specs[offset:]+specs[:offset]:
                    entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}",
                        map_id=map_id, family=family, map_seed=seed, planner_seed=planner_seed,
                        method=method, risk_slack=slack, status="NOT_RUN"))
    return dict(protocol_id=PROTOCOL_ID, schema_version=2, profile=profile,
                status="PREPARED_INVENTORY", config_sha256=digest(config), entries=entries,
                planned_trials=len(entries), map_count=len(settings["families"])*len(settings["map_seeds"]),
                planning_budget_s=settings["planning_budget_s"],
                requires_selected_pilot=profile=="full", planner_execution_count=0,
                seed_replacements=[])


def source_hashes():
    result = previous_source_hashes()
    for path in sorted((ROOT/"tools").glob("*.py")):
        result[path.relative_to(ROOT).as_posix()] = file_hash(path)
    return result


def check_selection(path, config, *, binary_sha256=None):
    """Bind selected slack to complete pilot evidence, not a user-edited scalar."""
    from analyze_safe_length_v2 import build_report, select_pilot
    selection = read(path)
    if (selection.get("status") != "SELECTED" or selection.get("protocol_id") != PROTOCOL_ID or
            selection.get("config_sha256") != digest(config)):
        raise ValueError("full requires a SELECTED artifact for this registered protocol")
    for kind in ("manifest", "summary"):
        source = Path(selection["pilot_"+kind+"_path"])
        if not source.is_file() or file_hash(source) != selection["pilot_"+kind+"_sha256"]:
            raise ValueError("pilot evidence changed: "+kind)
    manifest = read(selection["pilot_manifest_path"])
    summary = read(selection["pilot_summary_path"])
    if (manifest.get("profile") != "pilot" or manifest.get("status") != "COMPLETE" or
            summary.get("synthetic_fixture") or manifest.get("synthetic_fixture") or
            any(row.get("synthetic_fixture") for row in summary.get("rows",[]))):
        raise ValueError("selection does not reference a complete pilot")
    if selection.get("sources_sha256") != digest(source_hashes()) or manifest.get("sources") != source_hashes():
        raise ValueError("code changed after pilot; selected slack cannot authorize this full run")
    if selection.get("binary_sha256") != manifest.get("binary_sha256"):
        raise ValueError("selected binary identity differs from pilot")
    if binary_sha256 is not None and selection.get("binary_sha256") != binary_sha256:
        raise ValueError("full worker differs from selected pilot worker")
    actual = select_pilot(build_report(summary, manifest, config))
    for key in ("status", "selected_risk_slack", "selected_method"):
        if actual.get(key) != selection.get(key):
            raise ValueError("pilot selection gate mismatch: "+key)
    return selection


def request_for(entry, info, config, profile, field_path=None, field_metadata=None, bridge=None):
    method = entry["method"]
    specs = dict(method_specs(profile, entry.get("risk_slack")))
    if method not in specs or entry.get("risk_slack") != specs[method]:
        raise ValueError("unregistered method")
    if entry.get("risk_slack") is not None and type(entry["risk_slack"]) not in (int,float):
        raise ValueError("risk slack must be a registered real number")
    if method == "a3t_v2" and profile == "full" and entry.get("risk_slack") is None:
        raise ValueError("full request requires selected risk slack")
    settings = config["profiles"][profile]
    if method == "sem_native":
        from native_paper_v3_protocol import request_for as native_request, default_config as native_config
        request = native_request("sem", dict(start=info["start"],goal=info["goal"]),
            info["map_path"],info["cloud_path"],info["cloud_sha256"],entry["planner_seed"],
            settings["planning_budget_s"],native_config())
        request.update(map_sha256=info["map_sha256"],bridge_library=str(Path(bridge).resolve()) if bridge else None,
            library_search_paths=[str(Path(bridge).resolve().parent),"C:/mingw64/bin"] if bridge else [],
            variant_id="sem_native_unchanged_in_safe_length_v2")
        return request
    common = method == "sem_common_v2"
    calibrated = method != "a3t_cap40"
    safety = dict(field_path=str(Path(field_path).resolve()),slope_cap_deg=35.,atlas_sampling=not common,
        policy_id="a3t_safe_length_v2" if calibrated else "a3t_safe_length_v1",
        **{key:field_metadata[key] for key in ("field_sha256","cloud_sha256","builder_source_sha256")})
    safety.update(dict(risk_slack=entry["risk_slack"]) if calibrated else dict(risk_cap=40.))
    request = dict(planner_id="sem_rrtstar_common_v1" if common else "bitstar_a3",variant_id=method,
        map_path=info["map_path"],planner_seed=entry["planner_seed"],start=info["start"],goal=info["goal"],
        planning_budget_s=settings["planning_budget_s"],samples_per_batch=200 if common else 100,
        batch_after_first_solution=0 if common else 200,rewire_factor=1.1,sample_attempts=100,
        progress_interval_s=.05,collision_index=True,state_cache=True,scalar_cost=True,
        sampler_query_cache=not common,exact_flat_certificate=False,objective_type="surface_length",
        cloud_phase_handshake=True,safe_length=safety)
    if common:
        request.update(sem_range_m=1.,sem_neighbor_radius_m=3.)
    return request
