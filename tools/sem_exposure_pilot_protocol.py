"""Frozen 2,500-trial tuning and 600-trial confirmation registration."""
from __future__ import annotations

import math
from pathlib import Path

from corridor_v4_execution import digest, read

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_sem_exposure_pilot_v1.json"
PROTOCOL_ID = "a3t_sem_exposure_pilot_v1"
FAMILIES = ("traversable_hill", "ridge_pass", "rolling_terrain", "narrow_passage", "mixed_obstacles")
LAMBDAS = (0., .5, 1., 2., 4., 8., 16., 32., 64.)
POSITIVE_LAMBDAS = LAMBDAS[1:]
NATIVE = "sem_native"


def method_for(weight):
    if type(weight) not in (int, float) or not math.isfinite(weight) or weight not in LAMBDAS:
        raise ValueError("unregistered SEM exposure lambda")
    return "a3t_sem_exposure_l" + format(weight, "g").replace(".", "p")


CONTROL = method_for(0.)
METHODS = tuple(map(method_for, LAMBDAS)) + (NATIVE,)


def default_config():
    return dict(protocol_id=PROTOCOL_ID, schema_version=1,
        model=dict(state="xy", robot_radius_m=.3, safety_margin_m=.1, max_slope_deg=35.,
                   nearest_neighbor_metric="euclidean_xy", objective="surface_length_sem_exposure"),
        sem_exposure=dict(lambdas=list(LAMBDAS), positive_candidates=list(POSITIVE_LAMBDAS),
            risk_scale=40., formula="J = L_H + lambda * integral(F_R ds_H) / 40", exposure_is_pri=False),
        safety=dict(policy_id="a3t_safe_length_v1", risk_cap=40., slope_cap_deg=35., atlas_sampling=True,
                    corridor=False),
        algorithm=dict(initial_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
                       sample_attempts=100, just_in_time_sampling=False),
        profiles=dict(
            tune=dict(families=list(FAMILIES), map_seeds=list(range(37001, 37011)),
                planner_seeds=list(range(376001, 376006)), planning_budget_s=20.,
                extent_m=40., spacing_m=.2, query_id="q0", planned_trials=2500),
            confirm=dict(families=list(FAMILIES), map_seeds=list(range(38001, 38005)),
                planner_seeds=list(range(386001, 386011)), planning_budget_s=20.,
                extent_m=40., spacing_m=.2, query_id="q0", planned_trials=600)),
        execution=dict(max_workers=4, validation_workers=4, threads_per_trial=1, start_method="spawn",
            phase_order=["collection", "independent_validation"], process_watchdog_s=600.,
            reserve_bytes_per_collection_worker=1024**3, reserve_bytes_per_validation_worker=512*1024**2,
            retry_started_trials=False, seed_replacements=[], map_order="map_seed_then_family",
            trial_order="planner_seed_then_rotated_methods_by_block_index", wave_map_count=4),
        analysis=dict(final_selection="last_published_incumbent_in_budget_before_validation",
            tuning_gate="PSR>=SEM; paired_mean_delta_L_H<0; paired_mean_delta_PRI<0",
            ranking_cohort="SEM_and_all_eligible_positive_lambdas_success_at_same_map_query_seed",
            ranking_requires_every_family=True,
            ranking_order=["common_cohort_mean_surface_length", "common_cohort_mean_PRI", "lambda"],
            confirmation_gate="mean_gates_and_upper_CI_delta_L_H<0_and_upper_CI_delta_PRI<0_and_lower_CI_delta_PSR>=0",
            confidence=.95, bootstrap_replicates=2000, bootstrap_seed=83158,
            bootstrap_order=["family_fixed", "map", "paired_seed"],
            no_confirmation_runner_up=True, full_status="NOT_RUN"))


def validate_config(config):
    # Canonical JSON distinguishes booleans from numeric weights/seed values.
    if digest(config) != digest(default_config()):
        raise ValueError("unregistered pilot configuration; create a new protocol for changes")
    return config


def load_config(path=CONFIG):
    return validate_config(read(path))


def method_specs(phase="tune", selected_lambda=None):
    if phase == "tune":
        if selected_lambda is not None:
            raise ValueError("tuning cannot be preselected")
        return list(zip(METHODS, (*LAMBDAS, None)))
    if phase != "confirm":
        raise ValueError("registered phases are tune and confirm; full NOT_RUN")
    if type(selected_lambda) not in (int, float) or selected_lambda not in POSITIVE_LAMBDAS:
        raise ValueError("confirmation requires a locked positive candidate lambda")
    return [(method_for(selected_lambda), float(selected_lambda)), (CONTROL, 0.), (NATIVE, None)]


def inventory(config, phase="tune", selected_lambda=None):
    validate_config(config)
    specs = method_specs(phase, selected_lambda)
    settings, entries, block = config["profiles"][phase], [], 0
    for map_seed in settings["map_seeds"]:
        for family in settings["families"]:
            map_id = f"{family}_{map_seed}"
            for planner_seed in settings["planner_seeds"]:
                offset = block % len(specs)
                for method, weight in specs[offset:] + specs[:offset]:
                    entries.append(dict(trial_id=f"{map_id}__s{planner_seed}__{method}", phase=phase,
                        selected_lambda=selected_lambda,
                        map_id=map_id, family=family, map_seed=map_seed, planner_seed=planner_seed, query_id="q0",
                        method=method, sem_lambda=weight, alpha=None,
                        policy="native" if method == NATIVE else "full_safe_exposure", block_index=block, status="NOT_RUN"))
                block += 1
    if len(entries) != settings["planned_trials"]:
        raise AssertionError("registered count mismatch")
    return dict(protocol_id=PROTOCOL_ID, schema_version=1, phase=phase, profile=phase,
        status="PREPARED_INVENTORY", selected_lambda=selected_lambda,
        config_sha256=digest(config), entries=entries, planned_trials=len(entries),
        map_count=len(settings["map_seeds"])*len(settings["families"]),
        planning_budget_s=settings["planning_budget_s"], planner_execution_count=0, seed_replacements=[])


def source_hashes():
    from safe_length_v2_protocol import source_hashes as scientific_sources
    return scientific_sources()


def request_for(entry, info, config, field_path=None, field_metadata=None, bridge=None):
    phase = entry.get("phase")
    # Confirmation's selection binding is verified by the controller; the
    # request still independently checks the complete registered slot identity.
    selected = entry.get("selected_lambda")
    if phase == "confirm" and selected is None:
        if entry.get("method") not in (CONTROL, NATIVE):
            selected = entry.get("sem_lambda")
        else:
            selected = info.get("selected_lambda")
    candidates = inventory(config, phase, selected)["entries"]
    expected = next((item for item in candidates if item["trial_id"] == entry.get("trial_id")), None)
    if expected is None or any(entry.get(key) != value for key, value in expected.items() if key != "status"):
        raise ValueError("unregistered pilot method/map/query/seed slot")
    if info.get("map_id", entry["map_id"]) != entry["map_id"] or info.get("query_id", "q0") != "q0":
        raise ValueError("request input does not match registered map/query")
    budget = config["profiles"][phase]["planning_budget_s"]
    if entry["method"] == NATIVE:
        from native_paper_v3_protocol import request_for as native_request, default_config as native_config
        request = native_request("sem", dict(start=info["start"], goal=info["goal"]),
            info["map_path"], info["cloud_path"], info["cloud_sha256"], entry["planner_seed"], budget, native_config())
        request.update(map_sha256=info["map_sha256"], bridge_library=str(Path(bridge).resolve()) if bridge else None,
            library_search_paths=[str(Path(bridge).resolve().parent), "C:/mingw64/bin"] if bridge else [],
            variant_id=NATIVE)
        return request
    from run_safe_length import request_for as safe_request
    request = safe_request("atlas_adaptive100_200", info,
        dict(planning_budget_s=budget, risk_cap=40., slope_cap_deg=35.), field_path, field_metadata, bridge)
    request.update(variant_id=entry["method"], planner_seed=entry["planner_seed"],
        objective_type="surface_length_sem_exposure", sem_exposure={"lambda": entry["sem_lambda"], "risk_scale": 40.})
    return request
