"""Frozen smoke registration for additive SEM exposure, not unique-cell PRI."""
from __future__ import annotations

from pathlib import Path

from corridor_v4_execution import digest, read
from safe_length_v2_protocol import SMOKE_FAMILIES

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/a3t_sem_exposure_v1.json"
PROTOCOL_ID = "a3t_sem_exposure_v1"
LAMBDAS = (0., 1., 4., 16.)
NOMINAL = "a3t_sem_exposure_l1"
CORRIDOR = "a3t_corridor_v4"
NATIVE = "sem_native"
METHODS = tuple("a3t_sem_exposure_l" + format(v, "g") for v in LAMBDAS) + (CORRIDOR, NATIVE)


def default_config(workers=4, validation_workers=4):
    if any(type(value) is not int or value not in (1, 4) for value in (workers, validation_workers)):
        raise ValueError("registered worker counts are one or four")
    return dict(protocol_id=PROTOCOL_ID, schema_version=1,
        model=dict(state="xy", robot_radius_m=.3, safety_margin_m=.1, max_slope_deg=35.,
                   nearest_neighbor_metric="euclidean_xy", objective="surface_length_sem_exposure"),
        sem_exposure=dict(lambdas=list(LAMBDAS), risk_scale=40., nominal_lambda=1.,
            formula="J = L_H + lambda * integral(F_R ds_H) / 40", exposure_is_pri=False),
        safety=dict(policy_id="a3t_safe_length_v1", risk_cap=40., slope_cap_deg=35.,
                    atlas_sampling=True, corridor=False),
        algorithm=dict(initial_batch=100, batch_after_first_solution=200, rewire_factor=1.1,
                       sample_attempts=100, just_in_time_sampling=False),
        methods=list(METHODS),
        profiles=dict(smoke=dict(families=list(SMOKE_FAMILIES), map_seeds=[101], planner_seeds=[11],
            planning_budget_s=2., extent_m=20., spacing_m=.2)),
        execution=dict(max_workers=workers, validation_workers=validation_workers,
            threads_per_trial=1, start_method="spawn", phase_order=["collection", "independent_validation"],
            process_watchdog_s=600., reserve_bytes_per_collection_worker=1024**3,
            reserve_bytes_per_validation_worker=512*1024**2, retry_started_trials=False, seed_replacements=[]),
        analysis=dict(final_selection="last_published_incumbent_in_budget_before_validation",
            nominal_method=NOMINAL, smoke_selects_weight=False, pilot_status="NOT_RUN", full_status="NOT_RUN",
            comparisons=["sem_native", "a3t_corridor_v4", "a3t_sem_exposure_l0"]))


def validate_config(config):
    execution = config.get("execution", {})
    if config != default_config(execution.get("max_workers"), execution.get("validation_workers")):
        raise ValueError("unregistered SEM-exposure config; create a new protocol for changes")
    return config


def load_config(path=CONFIG):
    return validate_config(read(path))


def inventory(config, profile="smoke"):
    validate_config(config)
    if profile != "smoke":
        raise ValueError("only smoke is registered; pilot/full NOT_RUN")
    entries = []
    for block, family in enumerate(config["profiles"]["smoke"]["families"]):
        map_id = family + "_101"
        offset = block % len(METHODS)
        for method in METHODS[offset:] + METHODS[:offset]:
            weight = LAMBDAS[METHODS.index(method)] if method in METHODS[:4] else None
            entries.append(dict(trial_id=f"{map_id}__s11__{method}", map_id=map_id, family=family,
                map_seed=101, planner_seed=11, query_id="q0", method=method, sem_lambda=weight,
                alpha=1.2 if method == CORRIDOR else None,
                policy="preview_d0" if method == CORRIDOR else "native" if method == NATIVE else "full_safe_exposure",
                block_index=block, status="NOT_RUN"))
    return dict(protocol_id=PROTOCOL_ID, schema_version=1, profile=profile, status="PREPARED_INVENTORY",
        config_sha256=digest(config), entries=entries, planned_trials=len(entries), map_count=7,
        planning_budget_s=2., planner_execution_count=0, seed_replacements=[])


def source_hashes():
    # The existing collector binds all scientific C++ and Python sources. A
    # separate binary/DLL hash is captured before any actual trial launch.
    from safe_length_v2_protocol import source_hashes as registered_sources
    return registered_sources()


def request_for(entry, info, config, profile="smoke", field_path=None, field_metadata=None, bridge=None):
    registered = {item["trial_id"]: item for item in inventory(config, profile)["entries"]}
    expected = registered.get(entry.get("trial_id"))
    if expected is None or any(entry.get(key) != value for key, value in expected.items() if key != "status"):
        raise ValueError("unregistered method/map/seed or objective weight")
    method = entry["method"]
    if method in (CORRIDOR, NATIVE):
        from corridor_v4_protocol import default_config as corridor_config, request_for as corridor_request
        baseline = corridor_config(beam_scheduler="length_first_stable")
        baseline_entry = dict(entry, alpha=None if method == NATIVE else 1.2,
                              policy="native" if method == NATIVE else "preview_d0")
        return corridor_request(baseline_entry, info, baseline, "smoke", field_path, field_metadata, bridge)
    from run_safe_length import request_for as safe_request
    request = safe_request("atlas_adaptive100_200", info,
        dict(planning_budget_s=2., risk_cap=40., slope_cap_deg=35.), field_path, field_metadata, bridge)
    request.update(variant_id=method, objective_type="surface_length_sem_exposure",
                   sem_exposure={"lambda": entry["sem_lambda"], "risk_scale": 40.})
    return request
